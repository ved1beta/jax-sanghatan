

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    vocab: int
    d_model: int          # residual-stream width
    n_heads: int
    n_layers: int
    d_ff: int             # MLP hidden width
    max_seq: int          # size of the learned positional table


def rmsnorm(p, x):
    """Normalise each token by its own RMS, then rescale per-channel.

    Cheaper than LayerNorm (no mean-subtraction / bias). float32 for the
    reduction even if x is bf16 later — the sum-of-squares is the part that
    loses precision.
    """
    ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(ms + 1e-6)).astype(x.dtype) * p["scale"]


def attention(p, x, n_heads):
    """Causal multi-head self-attention on a single sequence x: (T, d) -> (T, d).

    einsum subscripts ARE the shapes — read them, not a pile of transposes:
    t = query position, s = key position, h = head, d = per-head width.
    """
    T, d = x.shape
    dh = d // n_heads
    qkv = x @ p["qkv"]                                  # (T, 3d) one matmul
    q, k, v = jnp.split(qkv, 3, axis=-1)                 # each (T, d)
    q, k, v = (a.reshape(T, n_heads, dh) for a in (q, k, v))

    scores = jnp.einsum("thd,shd->hts", q, k) / jnp.sqrt(dh)   # (h, T, T)

    # Causal mask: query t may attend to key s only if s <= t. jnp.where, NOT
    # a Python `if` — the values are traced; you mask with arrays, you don't
    # branch on them. tril(ones) is the lower triangle (s <= t).
    causal = jnp.tril(jnp.ones((T, T), bool))
    scores = jnp.where(causal, scores, -jnp.inf)

    attn = jax.nn.softmax(scores, axis=-1)               # over keys s; -inf -> 0
    ctx = jnp.einsum("hts,shd->thd", attn, v).reshape(T, d)
    return ctx @ p["o"]


def mlp(p, x):
    """Position-wise GELU MLP: widen -> nonlinearity -> project back."""
    return jax.nn.gelu(x @ p["w1"]) @ p["w2"]


def block(p, x, n_heads):
    """One pre-norm decoder block. Residual + (norm -> sublayer): gradients
    get a clean identity path, and each sublayer sees a normalised input.
    """
    x = x + attention(p["attn"], rmsnorm(p["ln1"], x), n_heads)
    x = x + mlp(p["mlp"], rmsnorm(p["ln2"], x))
    return x


class Transformer:
    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        ks = iter(jax.random.split(key, 4 + 4 * cfg.n_layers))
        n = lambda *s: jax.random.normal(next(ks), s) * 0.02
        ln = lambda: {"scale": jnp.ones((cfg.d_model,))}
        d, ff = cfg.d_model, cfg.d_ff

        self.params = {
            "tok_emb": n(cfg.vocab, d),
            "pos_emb": n(cfg.max_seq, d),
            "blocks": [
                {
                    "ln1": ln(),
                    "attn": {"qkv": n(d, 3 * d), "o": n(d, d)},
                    "ln2": ln(),
                    "mlp": {"w1": n(d, ff), "w2": n(ff, d)},
                }
                for _ in range(cfg.n_layers)
            ],
            "lnf": ln(),
            "head": n(d, cfg.vocab),
        }

    def __call__(self, params, tokens):
        """tokens: int (T,) -> logits (T, vocab). `params` passed in (not
        self.params) so step 4's jax.value_and_grad can differentiate it.
        """
        p, T = params, tokens.shape[0]
        x = p["tok_emb"][tokens] + p["pos_emb"][:T]      # embed + position
        for bp in p["blocks"]:
            x = block(bp, x, self.cfg.n_heads)
        return rmsnorm(p["lnf"], x) @ p["head"]          # (T, vocab)


if __name__ == "__main__":
    cfg = Config(vocab=50, d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq=16)
    net = Transformer(jax.random.PRNGKey(0), cfg)
    T = 8
    tokens = jax.random.randint(jax.random.PRNGKey(1), (T,), 0, cfg.vocab)


    logits = net(net.params, tokens)
    assert logits.shape == (T, cfg.vocab), logits.shape
    print("Step 3 OK  logits shape:", logits.shape)

    poked = tokens.at[T - 1].set((tokens[T - 1] + 1) % cfg.vocab)
    a, b = net(net.params, tokens), net(net.params, poked)
    assert jnp.allclose(a[: T - 1], b[: T - 1], atol=1e-6), "future leaked into past!"
    assert not jnp.allclose(a[T - 1], b[T - 1]), "last position should change"
    print("Step 3 OK  attention is causal")
