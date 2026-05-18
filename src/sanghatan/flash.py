"""
Naive attention (transformer.attention) materialises the full (H, T, T)
score matrix — O(T^2) memory. Flash never does: it streams over key blocks
with an online softmax (running max m, running denom l, running output acc),
so the largest score tensor is only (H, T, Bk). Same math, same numbers — we
assert bit-closeness and show the (T,T) tensor is absent from the flash HLO.

Single sequence, causal, multi-head. Masking uses a large finite negative
(not -inf): every block stays finite so the online rescale never hits
inf-inf = nan, and exp(-1e9) underflows to exactly 0, so it matches the
-inf naive result to float precision.
"""

import jax
import jax.numpy as jnp
from jax import lax

from sanghatan.transformer import Config, Transformer, attention

NEG = -1e9


def flash_attention(p, x, n_heads, Bk):
    """Drop-in for transformer.attention, computed block-wise. T % Bk == 0."""
    T, d = x.shape
    dh = d // n_heads
    q, k, v = jnp.split(x @ p["qkv"], 3, axis=-1)
    q, k, v = (a.reshape(T, n_heads, dh) for a in (q, k, v))
    kb = k.reshape(-1, Bk, n_heads, dh)                  # (n_blk, Bk, H, dh)
    vb = v.reshape(-1, Bk, n_heads, dh)
    tpos = jnp.arange(T)

    def step(carry, blk):
        m, l, acc, b = carry                             # m,l:(H,T) acc:(T,H,dh)
        s = jnp.einsum("thd,jhd->htj", q, blk[0]) / jnp.sqrt(dh)  # (H,T,Bk)
        jpos = b * Bk + jnp.arange(Bk)                    # key global indices
        s = jnp.where(jpos[None, None, :] <= tpos[None, :, None], s, NEG)
        m_new = jnp.maximum(m, s.max(-1))                 # (H,T)
        alpha = jnp.exp(m - m_new)                        # rescale old stats
        pe = jnp.exp(s - m_new[..., None])                # (H,T,Bk)
        l = l * alpha + pe.sum(-1)
        a2 = alpha.T[..., None]                           # (T,H,1)
        acc = acc * a2 + jnp.einsum("htj,jhd->thd", pe, blk[1])
        return (m_new, l, acc, b + 1), None

    H = n_heads
    init = (jnp.full((H, T), NEG), jnp.zeros((H, T)),
            jnp.zeros((T, H, dh)), 0)
    (_, l, acc, _), _ = lax.scan(step, init, (kb, vb))
    out = acc / l.T[..., None]                            # (T,H,dh)
    return out.reshape(T, d) @ p["o"]


def measure(T=64, Bk=16):
    cfg = Config(vocab=50, d_model=64, n_heads=4,
                 n_layers=1, d_ff=64, max_seq=T)
    net = Transformer(jax.random.PRNGKey(0), cfg)
    ap = net.params["blocks"][0]["attn"]
    x = jax.random.normal(jax.random.PRNGKey(1), (T, cfg.d_model))
    H = cfg.n_heads

    ref = attention(ap, x, H)
    fla = flash_attention(ap, x, H, Bk)
    drift = float(jnp.max(jnp.abs(ref - fla)))

    big = f"x{T}x{T}x"                                    # the (H,T,T) tensor
    nv_hlo = jax.jit(lambda p, x: attention(p, x, H)).lower(ap, x).as_text()
    fl_hlo = jax.jit(
        lambda p, x: flash_attention(p, x, H, Bk)).lower(ap, x).as_text()
    return {"drift": drift,
            "naive_has_TxT": big in nv_hlo,
            "flash_has_TxT": big in fl_hlo}


if __name__ == "__main__":
    m = measure()
    print(f"max |naive - flash| = {m['drift']:.2e}")
    print(f"(T,T) score tensor in HLO  naive={m['naive_has_TxT']}  "
          f"flash={m['flash_has_TxT']}")
    assert m["drift"] < 1e-4, "flash attention diverges from naive"
    assert m["naive_has_TxT"] and not m["flash_has_TxT"], \
        "flash must avoid the O(T^2) score tensor"
    print("Stretch OK  flash == naive numerically, no (T,T) tensor")
