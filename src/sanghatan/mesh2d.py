"""
8 fake devices arranged as a (2, 4) mesh: axis 'data' (DP) x axis 'tp' (TP).
Weights are TP-sharded on 'tp' (and implicitly replicated over 'data'); the
batch is sharded on 'data'. One `jax.jit` with NamedShardings *is* pjit
(pjit is now an alias of jit) — and the SAME code runs multi-host unchanged:
each process calls `jax.distributed.initialize()`, `jax.devices()` then spans
all hosts and the Mesh covers them. No API change, which is the whole point.

Reuses shard.tp_specs (the Megatron spec from step 5) — on a 2D mesh, axes a
PartitionSpec doesn't mention are replicated, so 'data' is automatically
free for the weights. Asserts the 2D-parallel forward equals single-device.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from sanghatan.shard import census  # noqa: E402
from sanghatan.shard import tp_specs  # noqa: E402
from sanghatan.transformer import Config, Transformer  # noqa: E402


def measure(data=2, tp=4):
    cfg = Config(vocab=50, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, max_seq=16)
    net = Transformer(jax.random.PRNGKey(0), cfg)
    T, B = 8, data * 2                                    # B seqs over 'data'
    tok = jax.random.randint(jax.random.PRNGKey(1), (T,), 0, cfg.vocab)
    ref = net(net.params, tok)                            # single-device truth

    mesh = Mesh(mesh_utils.create_device_mesh((data, tp)), ("data", "tp"))
    pspec = jax.tree.map(lambda s: NamedSharding(mesh, s), tp_specs(cfg))
    params = jax.device_put(net.params, pspec)            # TP weights
    batch = jnp.broadcast_to(tok, (B, T))

    fwd = jax.jit(
        jax.vmap(net.__call__, in_axes=(None, 0)),        # add batch dim
        in_shardings=(pspec, NamedSharding(mesh, P("data", None))),
        out_shardings=NamedSharding(mesh, P("data", None, None)),
    )
    out = fwd(params, batch)
    match = bool(jnp.allclose(ref, out[0], atol=1e-4))
    hlo = fwd.lower(params, batch).compile().as_text()
    return {"mesh": f"{data}x{tp} (data x tp)", "match": match,
            "collectives": census(hlo)}


if __name__ == "__main__":
    m = measure()
    print(f"mesh {m['mesh']}; matches single-device: {m['match']}")
    print(f"collectives: {m['collectives']}")
    # DP axis is free in the forward; only the TP axis costs all-reduces
    # (row-parallel o/w2, same as step 5). That asymmetry IS 2D parallelism.
    assert m["match"], "2D-parallel forward != single-device"
    assert "all-reduce" in m["collectives"], "expected TP all-reduce"
    print("Stretch OK  data x tensor parallel forward correct")
