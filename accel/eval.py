"""python -m accel.eval --ckpt DIR --data META|synthetic:P [--split val|train] [--batch 8] [--seq N]
                       [--batches 20] [--seed 1234] [--precision f32|bf16]

`forge eval` on JAX devices (cli/src/main.rs eval_cmd, train/src/lib.rs evaluate_weights): the
same windows (forge's Rng(--seed) and sampler), the same f64-per-batch / f32-register loss
accumulation, and the same output line:
{"ckpt","data","loss","next_token_acc","perplexity","quant","sha256","split","tokens","type":"eval","weight_bytes"}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import forge_io as fio
from .config import ConfigError, ModelConfig
from .forge_io import Rng, Store
from .runtime import Mesh, emit, init_distributed


def load_model(ckpt: str) -> tuple[ModelConfig, dict, str]:
    """What forge's CLI needs from a checkpoint (main.rs:146-159): state.json config.model and
    an F32 model.safetensors holding every parameter."""
    state, w, sha = fio.load_checkpoint(ckpt)
    try:
        mc = ModelConfig.from_dict(state["config"]["model"])
    except (KeyError, TypeError) as e:
        raise ConfigError(f"{ckpt}: state.json has no config.model ({e})")
    return mc, fio.check_weights(w, mc.param_specs(), f"{ckpt}/model.safetensors"), sha


def evaluate(ckpt: str, data: str, split: str = "val", batch: int = 8, seq: int | None = None, batches: int = 20,
             seed: int = 1234, precision: str = "f32", devices: int = 0, xent: str = "forge") -> dict:
    from .train import build_eval, eval_windows
    mc, w, sha = load_model(ckpt)
    seq = min(mc.max_seq_len, 256) if seq is None else seq
    if seq < 2 or seq > mc.max_seq_len or batch == 0 or batches == 0:
        raise ConfigError(f"--seq must be in 2..={mc.max_seq_len} for this model, --batch and --batches positive")
    store = Store(data, mc.vocab_size)
    mesh = Mesh(devices, batch)
    params = mesh.put_tree(w, mesh.repl)
    loss, acc = eval_windows(build_eval(mc, mesh, precision, xent), params, mesh, store, Rng(seed), split, batch, seq, batches)
    return {"type": "eval", "ckpt": ckpt, "sha256": sha, "data": data, "split": "Train" if split == "train" else "Val",
            "tokens": batch * seq * batches, "loss": float(loss), "perplexity": float(fio.expf(loss)),
            "next_token_acc": float(acc), "quant": "f32", "weight_bytes": 4.0 * mc.num_params(),
            "precision": precision}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="accel.eval", description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", help="train or val (anything else is val, like forge)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--precision", choices=("f32", "bf16"), default=os.environ.get("ACCEL_PRECISION", "f32"))
    ap.add_argument("--devices", type=int, default=int(os.environ.get("ACCEL_DEVICES", "0") or 0))
    ap.add_argument("--xent", choices=("forge", "fast"), default="forge",
                    help="softmax denominator: forge's sequential f32 sum (default) or a tree sum")
    ap.add_argument("--quant", help="not supported by accel (use forge eval --quant)")
    ap.add_argument("--threads", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        if a.quant:
            raise ConfigError("--quant is not supported by accel; use forge eval --quant")
        init_distributed()
        emit(evaluate(a.ckpt, a.data, "train" if a.split == "train" else "val", a.batch, a.seq, a.batches, a.seed,
                      a.precision, a.devices, a.xent))
    except (ConfigError, ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"accel: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
