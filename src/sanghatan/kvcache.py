"""

The Generalist-role centerpiece: prefill the prompt in one parallel pass,
then generate one token at a time reusing cached K/V instead of recomputing
the whole prefix every step.

  naive baseline : each new token => full forward over the WHOLE prefix
                   (O(L) layers * O(L^2) attention, L growing). transformer.py
                   already is this — we just call it on the growing prefix.
  kv-cache       : prefill writes K,V for all prompt positions; each decode
                   step computes Q,K,V for ONE token, appends K,V into a
                   fixed (Lmax,...) buffer, attends over the live slice.

Cache buffers are fixed-size with a traced write index (dynamic_update_slice)
— the shape stays static so the decode step jits/compiles once, the realistic
TPU-serving pattern. Output is asserted bit-identical to the baseline.
"""

import time

import jax
import jax.numpy as jnp
from jax import lax

from sanghatan.transformer import Config, Transformer, mlp, rmsnorm


def _qkv(h, w, n_heads):
    """(L,d) @ (d,3d) -> three (L, n_heads, dh) tensors. Same split/reshape
    as transformer.attention, factored out so prefill and decode share it.
    """
    L = h.shape[0]
    q, k, v = jnp.split(h @ w, 3, axis=-1)
    return (a.reshape(L, n_heads, -1) for a in (q, k, v))


def prefill(net, params, tokens, Lmax):
    """Parallel pass over the prompt. Returns logits (P, vocab) and a cache:
    per layer {"k","v"} each (Lmax, n_heads, dh) with the first P slots live.
    """
    p, P, H = params, tokens.shape[0], net.cfg.n_heads
    x = p["tok_emb"][tokens] + p["pos_emb"][:P]          # (P, d)
    cache = []
    for bp in p["blocks"]:
        q, k, v = _qkv(rmsnorm(bp["ln1"], x), bp["attn"]["qkv"], H)
        dh = q.shape[-1]
        scores = jnp.einsum("thd,shd->hts", q, k) / jnp.sqrt(dh)
        scores = jnp.where(jnp.tril(jnp.ones((P, P), bool)), scores, -jnp.inf)
        ctx = jnp.einsum("hts,shd->thd", jax.nn.softmax(scores, -1), v)
        x = x + ctx.reshape(P, -1) @ bp["attn"]["o"]
        x = x + mlp(bp["mlp"], rmsnorm(bp["ln2"], x))
        # seed fixed-size buffers; only [:P] is valid for now.
        # dtype follows k so a bf16 model gets a bf16 (half-size) cache.
        z = jnp.zeros((Lmax, H, dh), k.dtype)
        cache.append({"k": z.at[:P].set(k), "v": z.at[:P].set(v)})
    return rmsnorm(p["lnf"], x) @ p["head"], cache


def decode(net, params, cache, pos, token):
    """One token at position `pos`. Appends its K,V into the cache and
    attends over keys 0..pos. Returns logits (vocab,) and the new cache.
    Pure: cache in, cache out — no mutation, so it jits cleanly.
    """
    p, H = params, net.cfg.n_heads
    x = (p["tok_emb"][token] + p["pos_emb"][pos])[None, :]   # (1, d)
    new = []
    for bp, c in zip(p["blocks"], cache):
        q, k, v = _qkv(rmsnorm(bp["ln1"], x), bp["attn"]["qkv"], H)  # (1,H,dh)
        dh = q.shape[-1]
        # write this token's K,V at row `pos` (traced index, static shape).
        K = lax.dynamic_update_slice(c["k"], k, (pos, 0, 0))
        V = lax.dynamic_update_slice(c["v"], v, (pos, 0, 0))
        scores = jnp.einsum("thd,shd->hts", q, K) / jnp.sqrt(dh)  # (H,1,Lmax)
        live = jnp.arange(K.shape[0]) <= pos                      # mask future
        scores = jnp.where(live, scores, -jnp.inf)
        ctx = jnp.einsum("hts,shd->thd", jax.nn.softmax(scores, -1), V)
        x = x + ctx.reshape(1, -1) @ bp["attn"]["o"]
        x = x + mlp(bp["mlp"], rmsnorm(bp["ln2"], x))
        new.append({"k": K, "v": V})
    return (rmsnorm(p["lnf"], x) @ p["head"])[0], new


def gen_cached(net, params, prompt, n_new, Lmax, step):
    logits, cache = prefill(net, params, prompt, Lmax)
    ids, nxt = list(map(int, prompt)), int(jnp.argmax(logits[-1]))
    outs = [logits[-1]]
    for i in range(n_new):
        ids.append(nxt)
        pos = len(prompt) + i
        logits, cache = step(params, cache, pos, jnp.int32(nxt))
        outs.append(logits)
        nxt = int(jnp.argmax(logits))
    return ids, jnp.stack(outs)


def gen_naive(net, params, prompt, n_new):
    """Recompute the entire prefix every step (the baseline)."""
    ids = list(map(int, prompt))
    outs = [net(params, jnp.array(ids))[-1]]
    for _ in range(n_new):
        ids.append(int(jnp.argmax(outs[-1])))
        outs.append(net(params, jnp.array(ids))[-1])
    return ids, jnp.stack(outs)


def _mem(compiled):
    a = compiled.memory_analysis()
    g = lambda n: getattr(a, n, 0) or 0
    return g("temp_size_in_bytes"), g("output_size_in_bytes")


if __name__ == "__main__":
    P, N = 32, 96
    Lmax = P + N
    cfg = Config(vocab=50, d_model=128, n_heads=8,
                 n_layers=4, d_ff=512, max_seq=Lmax)
    net = Transformer(jax.random.PRNGKey(0), cfg)
    prompt = jax.random.randint(jax.random.PRNGKey(1), (P,), 0, cfg.vocab)

    step = jax.jit(lambda pa, c, po, t: decode(net, pa, c, po, t))


    ids_n, log_n = gen_naive(net, net.params, prompt, N)
    ids_c, log_c = gen_cached(net, net.params, prompt, N, Lmax, step)
    assert ids_n == ids_c, "generated token streams diverge"
    assert jnp.allclose(log_n, log_c, atol=1e-4), "per-step logits diverge"
    print(f"Step 6 OK  {N} tokens identical (naive == kv-cache)")

    _, cache0 = prefill(net, net.params, prompt, Lmax)
    t = time.perf_counter()
    sc = step.lower(net.params, cache0, jnp.int32(P), jnp.int32(0)).compile()
    kv_compile = time.perf_counter() - t

    naive = jax.jit(lambda pa, ids: net(pa, ids)[-1])
    full = jnp.arange(Lmax, dtype=jnp.int32)
    t = time.perf_counter()
    nc = naive.lower(net.params, full).compile()
    nv_compile = time.perf_counter() - t

    kv_temp, _ = _mem(sc)            # scratch at max length, static
    nv_temp, _ = _mem(nc)            # scratch for a full Lmax recompute

    # throughput: generate N tokens, wall clock (after warmup compile).
    gen_naive(net, net.params, prompt, 2)                 # warm
    gen_cached(net, net.params, prompt, 2, Lmax, step)
    t = time.perf_counter()
    gen_naive(net, net.params, prompt, N)
    nv_s = time.perf_counter() - t
    t = time.perf_counter()
    gen_cached(net, net.params, prompt, N, Lmax, step)
    kv_s = time.perf_counter() - t

    print(f"{'':14}{'naive':>14}{'kv-cache':>14}")
    print(f"{'tokens/sec':14}{N/nv_s:>14.1f}{N/kv_s:>14.1f}")
    print(f"{'total (s)':14}{nv_s:>14.3f}{kv_s:>14.3f}")
    print(f"{'compile (s)':14}{nv_compile:>14.3f}{kv_compile:>14.3f}")
    print(f"{'temp (KB)':14}{nv_temp/1024:>14.1f}{kv_temp/1024:>14.1f}")
