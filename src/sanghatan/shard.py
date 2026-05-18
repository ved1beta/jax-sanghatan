"""
Two strategies, same forward pass:

  Tensor parallel (primary): split each weight matrix across devices.
    qkv,w1 = column-parallel  P(None,'tp')   -> activations sharded on heads
    o ,w2  = row-parallel     P('tp',None)   -> contraction is sharded, so
                                                GSPMD must all-reduce after.
    Residual stream stays replicated, so every block forces collectives.
    THIS is what shows up in the HLO and what the TPU role asks about.

  Data parallel (contrast): replicate weights, shard the batch. Forward is
    embarrassingly parallel -> the HLO has *no* weight collectives (the
    all-reduce only appears in the backward pass, which is out of scope here).

Run this; it writes the optimized HLO to notes/ and prints a collective
census. The hand-annotated walkthrough lives in notes/05_hlo_walkthrough.md.
"""

import os


os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import pathlib  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from sanghatan.transformer import Config, Transformer  # noqa: E402

NOTES = pathlib.Path(__file__).parents[2] / "notes"
COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter",
               "all-to-all", "collective-permute")


def tp_specs(cfg: Config):
    """PartitionSpec for every param leaf, mirroring the params pytree.

    P() = replicated. Megatron pattern: column-parallel matrix shards its
    OUTPUT dim, row-parallel shards its INPUT (contraction) dim.
    """
    R = P()

    def block():
        return {
            "ln1": {"scale": R},
            "attn": {"qkv": P(None, "tp"), "o": P("tp", None)},
            "ln2": {"scale": R},
            "mlp": {"w1": P(None, "tp"), "w2": P("tp", None)},
        }

    return {
        "tok_emb": R,
        "pos_emb": R,
        "blocks": [block() for _ in range(cfg.n_layers)],
        "lnf": {"scale": R},
        "head": R,
    }


def census(hlo: str) -> dict:
    """Count collective ops in an HLO text dump (the thing to annotate)."""
    return {c: hlo.count(c) for c in COLLECTIVES if hlo.count(c)}


def dump(name: str, hlo: str):
    NOTES.mkdir(exist_ok=True)
    (NOTES / name).write_text(hlo)


if __name__ == "__main__":
    cfg = Config(vocab=50, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, max_seq=16)
    net = Transformer(jax.random.PRNGKey(0), cfg)
    n_dev = len(jax.devices())
    print(f"devices: {n_dev}")

    T = 8
    tokens = jax.random.randint(jax.random.PRNGKey(1), (T,), 0, cfg.vocab)
    ref = net(net.params, tokens)                       # single-device truth

    # ---------- tensor parallel ----------
    tp_mesh = Mesh(mesh_utils.create_device_mesh((n_dev,)), ("tp",))
    specs = tp_specs(cfg)
    tp_params = jax.device_put(
        net.params,
        jax.tree.map(lambda s: NamedSharding(tp_mesh, s), specs),
    )
    # out_shardings replicated: GSPMD must bring sharded results back ->
    # the collectives become explicit in the HLO.
    fwd = jax.jit(net.__call__, out_shardings=NamedSharding(tp_mesh, P()))
    tp_out = fwd(tp_params, tokens)
    assert jnp.allclose(ref, tp_out, atol=1e-4), "TP forward != single-device"

    tp_low = fwd.lower(tp_params, tokens)
    tp_unopt = tp_low.as_text()           # logical collectives
    tp_hlo = tp_low.compile().as_text()   # XLA-lowered (CPU ring)
    dump("05_tp_unoptimized_hlo.txt", tp_unopt)
    dump("05_tp_optimized_hlo.txt", tp_hlo)
    print("TP  numerically matches; unopt collectives:", census(tp_unopt))
    print("TP  optimized collectives:", census(tp_hlo))

    # ---------- data parallel (contrast) ----------
    dp_mesh = Mesh(mesh_utils.create_device_mesh((n_dev,)), ("data",))
    batch = jnp.broadcast_to(tokens, (n_dev, T))         # B = n_dev seqs
    batched = jax.vmap(net.__call__, in_axes=(None, 0))  # add batch dim
    dp = jax.jit(
        batched,
        # params replicated, batch sharded over 'data'
        in_shardings=(NamedSharding(dp_mesh, P()),
                      NamedSharding(dp_mesh, P("data"))),
        out_shardings=NamedSharding(dp_mesh, P("data")),
    )
    dp_out = dp(net.params, batch)
    assert jnp.allclose(ref, dp_out[0], atol=1e-4), "DP != single-device"

    dp_hlo = dp.lower(net.params, batch).compile().as_text()
    dump("05_dp_optimized_hlo.txt", dp_hlo)
    print("DP  numerically matches; collectives:", census(dp_hlo) or "none")

    print("Step 5 OK  sharded forwards match single-device; HLO dumped")
