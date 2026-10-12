"""python -m accel.logits --ckpt DIR --prompt-ids a,b,c [--ctx N] [--precision f32|bf16]

`forge logits` on a JAX device (cli/src/main.rs logits_cmd, train/src/lib.rs Sampler): the
logits of the position after the last of the (last `ctx`) prompt ids, every window starting at
position 0. forge runs the full ctx window padded with id 0; padding is causally irrelevant,
so accel runs just the window. Prints {"logits":[V],"prompt_ids":[...],"type":"logits"}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .config import ConfigError
from .runtime import emit


def next_logits(mc, params: dict, ids: list[int], ctx: int | None = None, precision: str = "f32") -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from .model import forward
    ctx = mc.max_seq_len if ctx is None else min(ctx, mc.max_seq_len)
    window = list(ids[max(len(ids) - ctx, 0):]) or [0]  # no ids: forge reads position 0 of an all-zero window
    if any(not 0 <= int(i) < mc.vocab_size for i in window):
        raise ConfigError(f"token ids must be in [0, {mc.vocab_size})")
    f = jax.jit(forward, static_argnums=(1, 3))
    lg = f({k: jnp.asarray(v) for k, v in params.items()}, mc, jnp.asarray([window], jnp.int32), precision)
    return np.asarray(lg[0, len(window) - 1], np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="accel.logits", description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt-ids", required=True)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--precision", choices=("f32", "bf16"), default=os.environ.get("ACCEL_PRECISION", "f32"))
    ap.add_argument("--threads", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        from .eval import load_model
        ids = [int(s.strip()) for s in a.prompt_ids.split(",")]
        mc, w, _ = load_model(a.ckpt)
        lg = next_logits(mc, w, ids, a.ctx, a.precision)
        emit({"type": "logits", "prompt_ids": ids, "logits": [float(x) for x in lg]})
    except (ConfigError, ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"accel: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
