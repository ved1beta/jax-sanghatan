"""Step 4 — train the step-3 transformer on a tiny char dataset.

Cross-entropy + vmap (batch) + value_and_grad (pytree) + hand-rolled Adam +
the whole loop as one jit-ed scan. transformer.py stays forward-only.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sanghatan.transformer import Config, Transformer

TEXT = "the quick brown fox jumps over the lazy dog. " * 8
VOCAB = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}


def loss_fn(params, net, xb, yb):
    """Mean next-token cross-entropy over a batch. `seq` is written for ONE
    sequence (scalar out, as value_and_grad needs); vmap adds the batch dim
    — no batch dim ever written into the model.
    """
    def seq(x, y):
        lp = jax.nn.log_softmax(net(params, x), -1)          # (T, vocab)
        return -jnp.mean(lp[jnp.arange(x.shape[0]), y])
    return jnp.mean(jax.vmap(seq)(xb, yb))


def adam_step(params, grads, state, t, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Adam, from scratch. State mirrors the params pytree, so each update is
    one jax.tree.map zipping params/grads/moments. t = traced step (bias fix).
    """
    m, v = state
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)
    mc, vc = 1 - b1 ** t, 1 - b2 ** t
    params = jax.tree.map(
        lambda p, m, v: p - lr * (m / mc) / (jnp.sqrt(v / vc) + eps),
        params, m, v)
    return params, (m, v)


def make_train(net, data_x, data_y, batch):
    """Returns a jit-ed `train(params, key, n_steps)`. net/data are closed
    over so jit only sees array args (net is not a jittable type).
    """
    n_win = data_x.shape[0]

    def step(carry, t):
        params, state, key = carry
        key, sk = jax.random.split(key)
        i = jax.random.randint(sk, (batch,), 0, n_win)       # random minibatch
        # Trace-time print: scan traces its body once, so this fires ONCE for
        # all n_steps. That single line in stdout is the proof.
        print("tracing train step")
        xb, yb = data_x[i], data_y[i]
        loss, grads = jax.value_and_grad(loss_fn)(params, net, xb, yb)
        params, state = adam_step(params, grads, state, t)
        return (params, state, key), loss

    def train(params, key, n_steps):
        z = jax.tree.map(jnp.zeros_like, params)             # Adam (m, v)
        # Whole loop = one scan: carry threads (params, opt state, rng);
        # xs = 1-based step index; ys = stacked loss history.
        init = (params, (z, z), key)
        ts = jnp.arange(1, n_steps + 1)
        (params, _, _), losses = jax.lax.scan(step, init, ts)
        return params, losses

    return jax.jit(train, static_argnums=2)                  # n_steps static


def generate(net, params, prompt, n_new, max_seq):
    """Greedy decode. Plain Python loop — variable length, not the hot path."""
    ids = [STOI[c] for c in prompt]
    for _ in range(n_new):
        ctx = jnp.array(ids[-max_seq:])
        ids.append(int(jnp.argmax(net(params, ctx)[-1])))
    return "".join(ITOS[i] for i in ids)


if __name__ == "__main__":
    T = 32
    cfg = Config(len(VOCAB), d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, max_seq=64)
    net = Transformer(jax.random.PRNGKey(0), cfg)

    ids = np.array([STOI[c] for c in TEXT])                  # sliding windows
    nw = len(ids) - T
    data_x = jnp.stack([ids[w:w + T] for w in range(nw)])
    data_y = jnp.stack([ids[w + 1:w + 1 + T] for w in range(nw)])

    train = make_train(net, data_x, data_y, batch=16)
    l0 = loss_fn(net.params, net, data_x[:16], data_y[:16])
    print(f"start loss: {float(l0):.3f}  (ln(vocab)={np.log(len(VOCAB)):.3f})")

    params, losses = train(net.params, jax.random.PRNGKey(1), 400)
    print(f"loss {float(losses[0]):.3f} -> {float(losses[-1]):.3f}")
    assert float(losses[-1]) < float(losses[0]) * 0.3, "loss didn't drop enough"
    out = generate(net, params, "the quick", 36, cfg.max_seq)
    print("sample:", repr(out))
    assert "quick brown fox" in out, "greedy output is garbage"

