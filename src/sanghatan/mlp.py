
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class MLP:
    def __init__(self, key, sizes):
        """sizes is the full layer spec, e.g. [1, 64, 64, 1] = 3 layers.

        params is a list of {"W", "b"} dicts (a pytree). zip(sizes[:-1],
        sizes[1:]) walks consecutive pairs to get each layer's (in, out).
        """
        self.sizes = sizes
        pairs = list(zip(sizes[:-1], sizes[1:]))
        keys = jax.random.split(key, len(pairs))  # one subkey per layer

        self.params = []
        for k, (in_dim, out_dim) in zip(keys, pairs):
            W = jax.random.normal(k, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
            b = jnp.zeros((out_dim,))
            self.params.append({"W": W, "b": b})

    def forward(self, params, x):
        """Single example x (sizes[0],) -> output (sizes[-1],).

        Takes `params` explicitly (NOT self.params) so jax.grad can
        differentiate through it. Hidden layers: affine then ReLU (nonlinearity
        is essential — stacked affine maps collapse into one). Final layer:
        affine only (raw regression head; ReLU would clamp output to >= 0).
        """
        for layer in params[:-1]:
            x = jnp.maximum(x @ layer["W"] + layer["b"], 0.0)  # dense + relu
        last = params[-1]
        return x @ last["W"] + last["b"]                       # raw head

    def loss(self, params, x, y):
        """Mean squared error for a single (x, y) example. Returns a SCALAR —
        jax.grad only differentiates scalar-valued functions. The grad w.r.t.
        params flows by the chain rule back through forward; the finite-
        difference check below verifies it numerically.
        """
        pred = self.forward(params, x)
        return jnp.mean((pred - y) ** 2)


if __name__ == "__main__":
    net = MLP(jax.random.PRNGKey(0), [1, 16, 1])


    out = net.forward(net.params, jnp.ones((1,)))
    assert out.shape == (1,)
    print("Step 1 OK  output:", out)

    x = jnp.array([0.5])
    y = jnp.sin(x)

    loss_val, grads = jax.value_and_grad(net.loss)(net.params, x, y)
    print("loss:", float(loss_val))
    # grads is a pytree with the SAME structure as params:
    assert jax.tree.structure(grads) == jax.tree.structure(net.params)

    # Flatten params to a flat vector so we can poke single coordinates.
    flat, unflatten = ravel_pytree(net.params)

    def loss_flat(v):
        return net.loss(unflatten(v), x, y)

    g = jax.grad(loss_flat)(flat)

    eps = 1e-2
    for i in [0, 5, len(flat) - 1]:                 # spot-check a few coords
        bump = jnp.zeros_like(flat).at[i].set(eps)
        numeric = (loss_flat(flat + bump) - loss_flat(flat - bump)) / (2 * eps)
        assert jnp.allclose(g[i], numeric, rtol=1e-2, atol=1e-4), \
            (i, float(g[i]), float(numeric))
    print("Step 2 OK  analytic grad matches finite differences")
