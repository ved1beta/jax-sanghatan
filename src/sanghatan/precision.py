"""
Cast the params to bfloat16; every matmul/cache then follows that dtype
(RMSNorm still reduces in f32 internally — see transformer.rmsnorm). We
measure the same KV-cache decode as step 6 in fp32 vs bf16: scratch memory,
throughput, and the numerical drift vs the fp32 reference.

Honest note: on CPU bf16 is *emulated*, so the speed win does not show here
(it can even be slower). The memory halving is real everywhere; on TPU bf16
is also the fast path. We report all three and don't pretend otherwise.
"""

import time

import jax
import jax.numpy as jnp

from sanghatan.kvcache import _mem, decode, gen_cached, prefill
from sanghatan.transformer import Config, Transformer


def cast(tree, dt):
    """Cast only the float leaves; int tables (token ids) stay int."""
    def f(a):
        return a.astype(dt) if jnp.issubdtype(a.dtype, jnp.floating) else a
    return jax.tree.map(f, tree)


def measure(P=32, N=96):
    Lmax = P + N
    cfg = Config(vocab=50, d_model=128, n_heads=8,
                 n_layers=4, d_ff=512, max_seq=Lmax)
    net = Transformer(jax.random.PRNGKey(0), cfg)
    prompt = jax.random.randint(jax.random.PRNGKey(1), (P,), 0, cfg.vocab)
    step = jax.jit(lambda pa, c, po, t: decode(net, pa, c, po, t))

    out = {}
    for tag, params in (("fp32", net.params),
                        ("bf16", cast(net.params, jnp.bfloat16))):
        _, c0 = prefill(net, params, prompt, Lmax)
        # the KV cache is the thing bf16 halves (it's an arg, not scratch).
        cache_kb = sum(a.size * a.dtype.itemsize
                       for a in jax.tree.leaves(c0)) / 1024
        comp = step.lower(params, c0, jnp.int32(P), jnp.int32(0)).compile()
        temp = _mem(comp)[0] / 1024
        gen_cached(net, params, prompt, 2, Lmax, step)        # warm
        t = time.perf_counter()
        _, logits = gen_cached(net, params, prompt, N, Lmax, step)
        out[tag] = {"cache_kb": cache_kb, "temp_kb": temp,
                    "tok_s": N / (time.perf_counter() - t),
                    "logits": logits.astype(jnp.float32)}

    d = jnp.abs(out["fp32"]["logits"] - out["bf16"]["logits"])
    return {"fp32": out["fp32"], "bf16": out["bf16"],
            "max_logit_drift": float(jnp.max(d))}


if __name__ == "__main__":
    m = measure()
    f, b = m["fp32"], m["bf16"]
    print(f"{'':14}{'fp32':>12}{'bf16':>12}")
    print(f"{'cache (KB)':14}{f['cache_kb']:>12.1f}{b['cache_kb']:>12.1f}")
    print(f"{'temp (KB)':14}{f['temp_kb']:>12.1f}{b['temp_kb']:>12.1f}")
    print(f"{'tokens/sec':14}{f['tok_s']:>12.1f}{b['tok_s']:>12.1f}")
    print(f"max logit drift fp32->bf16: {m['max_logit_drift']:.3f}")
    # bf16 cache is half the bytes; drift stays small (logits are O(1)).
    assert b["cache_kb"] < f["cache_kb"] * 0.6, "bf16 cache should ~halve"
    assert m["max_logit_drift"] < 2.0, "bf16 drift unexpectedly large"
   