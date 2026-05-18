"""
Backward needs each block's forward activations. By default XLA *stores* them
(O(n_layers) activation memory). `jax.checkpoint` tells autodiff to drop them
and recompute the block in the backward pass instead: less peak activation
memory, more compute.

We grad a deep (24-layer) block stack with and without remat. Honest finding
on CPU: XLA's buffer reuse makes `peak_memory_in_bytes` come out identical,
so the memory *benefit* is masked here — it is the dominant, intended effect
on TPU/GPU where activations actually pin HBM. What IS deterministically
measurable on any backend is the *cost*: remat's extra recompute shows up as
~+23% FLOPs (cost_analysis) and a slower backward. We measure and report all
three, and don't claim a CPU memory win that isn't there.
"""

import time

import jax
import jax.numpy as jnp

from sanghatan.transformer import Config, Transformer, block, rmsnorm


def make_grad(net, remat):
    H = net.cfg.n_heads
    # static_argnums=2: n_heads must stay a concrete int (reshape uses it).
    b = jax.checkpoint(block, static_argnums=(2,)) if remat else block

    def loss(params, x):
        for bp in params["blocks"]:
            x = b(bp, x, H)
        return jnp.mean(rmsnorm(params["lnf"], x) ** 2)   # scalar

    return jax.jit(jax.grad(loss))


def measure(T=256):
    # deep + long enough that the O(n_layers) activation stack is real.
    cfg = Config(vocab=50, d_model=256, n_heads=8,
                 n_layers=24, d_ff=1024, max_seq=T)
    net = Transformer(jax.random.PRNGKey(0), cfg)
    x = jax.random.normal(jax.random.PRNGKey(1), (T, cfg.d_model))

    out = {}
    for tag, remat in (("store", False), ("remat", True)):
        g = make_grad(net, remat)
        comp = g.lower(net.params, x).compile()
        gflops = comp.cost_analysis()["flops"] / 1e9
        peak = comp.memory_analysis().peak_memory_in_bytes / 1024 / 1024
        jax.block_until_ready(g(net.params, x))           # warm
        t = time.perf_counter()
        for _ in range(20):
            r = g(net.params, x)
        jax.block_until_ready(r)
        ms = (time.perf_counter() - t) / 20 * 1e3
        out[tag] = {"gflops": gflops, "peak_mb": peak, "ms": ms}
    return out


if __name__ == "__main__":
    m = measure()
    s, r = m["store"], m["remat"]
    print(f"{'':14}{'store':>12}{'remat':>12}")
    print(f"{'GFLOPs':14}{s['gflops']:>12.1f}{r['gflops']:>12.1f}")
    print(f"{'peak (MB)':14}{s['peak_mb']:>12.1f}{r['peak_mb']:>12.1f}")
    print(f"{'grad (ms)':14}{s['ms']:>12.2f}{r['ms']:>12.2f}")
    # The cost side is deterministic everywhere: remat recomputes -> more
    # FLOPs and a slower backward. (Memory benefit: accelerator-visible.)
    assert r["gflops"] > s["gflops"] * 1.1, "remat should add recompute FLOPs"
    print(f"remat adds {(r['gflops']/s['gflops']-1)*100:.0f}% FLOPs "
          f"(the price of recompute); memory win is TPU/GPU-side")
