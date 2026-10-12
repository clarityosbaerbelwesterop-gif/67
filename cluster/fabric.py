"""forge cluster fabric: DiLoCo across machines that only exchange files.

Every round, W workers start from the same global model, each trains H steps of
`forge train` on its own random windows of the corpus (seed = round x worker), and
writes its delta. The aggregator averages the deltas into the outer gradient and
applies SGD with Nesterov momentum (lr 0.7, momentum 0.9; Douillard et al. 2023).
Measured on this project's fabric branch (gpu/fabric-v0, T1b): at H = 100 DiLoCo
needs about 2.5x the tokens of synchronous training for the same quality, so W
workers give roughly W / 2.5 the progress of one machine, with H times less
communication. Workers need no network between them: GitHub runners, home machines
and VGPU nodes all qualify.

  state dir   global/ (a forge checkpoint: model.safetensors + state.json),
              momentum.safetensors, round.json
  init        python3 cluster/fabric.py init --from runs/base-g3/FINAL --state S
              python3 cluster/fabric.py init --scratch training/configs/base-s.json --state S   (from scratch)
  work        python3 cluster/fabric.py work --state S --worker 3 --steps 300 --data META --out D
  aggregate   python3 cluster/fabric.py aggregate --state S --deltas D1 D2 ... [--quorum 0.5]
  eval        python3 cluster/fabric.py eval --state S --data META

Deltas are stored as float16 (lossless enough: int8 cost +0.15 % in T2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rebase as R  # noqa: E402  (safetensors reader/writer as forge writes them)

FORGE = os.environ.get("FORGE", str(ROOT / "target" / "release" / "forge"))
DT = {"F32": np.float32, "F16": np.float16}


def load(path: Path) -> tuple[dict, dict | None]:
    ts, meta, _ = R.read_safetensors(path)
    out = {}
    for name, t in ts.items():
        out[name] = np.frombuffer(bytes(t["data"]), DT[t["dtype"]]).reshape(t["shape"]).astype(np.float32)
    return out, meta


def save(path: Path, arrays: dict, dtype: str = "F32", meta: dict | None = None) -> str:
    tensors = {n: {"dtype": dtype, "shape": list(a.shape), "data": a.astype(DT[dtype]).tobytes()} for n, a in arrays.items()}
    return R.write_safetensors(path, tensors, meta if meta is not None else {"format": "pt"})


def write_ckpt(d: Path, arrays: dict, model_cfg: dict, parent: str | None, extra: dict) -> str:
    d.mkdir(parents=True, exist_ok=True)
    sha = save(d / "model.safetensors", arrays)
    (d / "state.json").write_text(json.dumps({"step": 0, "config": {"model": model_cfg}, "model_sha256": sha,
                                              "parent_sha256": parent, **extra}, indent=2))
    return sha


def read_state(s: Path) -> tuple[dict, dict, dict, dict]:
    g, _ = load(s / "global" / "model.safetensors")
    m, _ = load(s / "momentum.safetensors")
    st = json.loads((s / "global" / "state.json").read_text())
    return g, m, st, json.loads((s / "round.json").read_text())


def scratch_ckpt(model_cfg: dict, d: Path, seed: int) -> Path:
    """forge's own seeded initialisation: one step at lr 1e-12 on a synthetic store (the update is ~1e-12 per
    weight, i.e. none); the data never matters, so no corpus is needed to start a cluster from scratch."""
    shutil.rmtree(d, ignore_errors=True)
    store = d / "store"
    store.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for name in ("train.bin", "train.val.bin"):
        rng.integers(0, model_cfg["vocab_size"] - 2, 1 << 16, dtype=np.uint16).tofile(store / name)
    (store / "tokenizer.json").write_bytes((ROOT / "data" / "tokenizer.json").read_bytes())
    (store / "train.meta.json").write_text(json.dumps({"tokens": 1 << 16, "val_tokens": 1 << 16, "dtype": "uint16",
                                                       "vocab_size": model_cfg["vocab_size"], "bin": "train.bin", "val_bin": "train.val.bin",
                                                       "mix": {"synthetic": 1}, "mix_fractions": {"synthetic": 1.0}}))
    cfg = {"run": "scratch", "model": model_cfg, "data": str(store / "train.meta.json"), "batch": 1, "seq_len": 16, "grad_accum": 1,
           "max_steps": 1, "lr": 1e-12, "min_lr": 1e-12, "warmup_steps": 1, "lr_schedule": "constant", "weight_decay": 0.0,
           "grad_clip": 1.0, "seed": seed, "out_dir": str(d), "log_every": 1, "eval_every": 0, "eval_batches": 1, "ckpt_every": 0, "threads": 0}
    (d / "config.json").write_text(json.dumps(cfg))
    r = subprocess.run([FORGE, "train", "--config", str(d / "config.json")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout[-1000:] + r.stderr[-1000:])
    return Path(json.loads((d / "scratch" / "latest.json").read_text())["dir"])


def cmd_init(a) -> int:
    s = Path(a.state)
    if a.scratch:
        model_cfg = json.loads(Path(a.scratch).read_text())
        src = scratch_ckpt(model_cfg.get("model", model_cfg), s.parent / f".{s.name}-scratch", a.seed)
    else:
        src = Path(a.src)
    w, _ = load(src / "model.safetensors")
    st = json.loads((src / "state.json").read_text())
    s.mkdir(parents=True, exist_ok=True)
    origin = f"scratch seed {a.seed}" if a.scratch else str(src)
    parent = None if a.scratch else st.get("model_sha256")
    sha = write_ckpt(s / "global", w, st["config"]["model"], parent, {"cluster": {"round": 0, "from": origin}})
    save(s / "momentum.safetensors", {n: np.zeros_like(x) for n, x in w.items()})
    (s / "round.json").write_text(json.dumps({"round": 0, "global_sha256": sha, "from": origin, "history": []}, indent=2))
    if a.scratch:
        shutil.rmtree(s.parent / f".{s.name}-scratch", ignore_errors=True)
    print(json.dumps({"init": str(s), "global_sha256": sha}))
    return 0


def cmd_work(a) -> int:
    s, out = Path(a.state), Path(a.out)
    rnd = json.loads((s / "round.json").read_text())["round"]
    st = json.loads((s / "global" / "state.json").read_text())
    work = out / "run"
    shutil.rmtree(work, ignore_errors=True)
    run = f"cluster-r{rnd}-w{a.worker}"
    cfg = {"run": run, "model": st["config"]["model"], "data": a.data, "batch": a.batch, "seq_len": a.seq, "grad_accum": a.accum,
           "max_steps": a.steps, "lr": a.lr, "min_lr": a.lr, "warmup_steps": 1, "lr_schedule": "constant",
           "weight_decay": 0.1, "grad_clip": 1.0, "seed": 1_000_003 * (rnd + 1) + a.worker, "init_from": str(s / "global"),
           "out_dir": str(work), "log_every": 50, "eval_every": 0, "eval_batches": 4, "ckpt_every": 0, "threads": a.threads}
    work.mkdir(parents=True)
    (work / "config.json").write_text(json.dumps(cfg))
    t0 = time.time()
    log = work / "train.jsonl"
    with open(log, "w") as f:
        rc = subprocess.run([FORGE, "train", "--config", str(work / "config.json")], stdout=f, stderr=subprocess.STDOUT).returncode
    secs = time.time() - t0
    if rc != 0:
        print(log.read_text()[-2000:], file=sys.stderr)
        return rc
    latest = json.loads((work / run / "latest.json").read_text())
    w, _ = load(Path(latest["dir"]) / "model.safetensors")
    g, _ = load(s / "global" / "model.safetensors")
    delta = {n: g[n] - w[n] for n in g}  # global - worker: the outer gradient's contribution
    out.mkdir(parents=True, exist_ok=True)
    save(out / "delta.safetensors", delta, "F16")
    losses = [json.loads(x)["loss"] for x in log.read_text().splitlines() if x.startswith("{") and '"telemetry"' in x]
    tps = [json.loads(x)["tokens_per_s"] for x in log.read_text().splitlines() if x.startswith("{") and '"telemetry"' in x]
    rec = {"round": rnd, "worker": a.worker, "steps": a.steps, "tokens": a.steps * a.batch * a.accum * a.seq, "seconds": secs,
           "tokens_per_s": float(np.median(tps)) if tps else None, "loss_first": losses[0] if losses else None,
           "loss_last": losses[-1] if losses else None, "global_sha256": json.loads((s / "round.json").read_text())["global_sha256"],
           "delta_sha256": hashlib.sha256((out / "delta.safetensors").read_bytes()).hexdigest()}
    (out / "worker.json").write_text(json.dumps(rec, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(rec))
    return 0


def cmd_aggregate(a) -> int:
    s = Path(a.state)
    g, m, st, rnd = read_state(s)
    deltas, recs = [], []
    for d in a.deltas:
        d = Path(d)
        if not (d / "worker.json").exists():
            continue
        rec = json.loads((d / "worker.json").read_text())
        if rec.get("global_sha256") != rnd["global_sha256"] or rec.get("round") != rnd["round"]:
            continue  # a delta of another round or another global model
        dl, _ = load(d / "delta.safetensors")
        if any(not np.all(np.isfinite(x)) for x in dl.values()):
            continue
        deltas.append(dl)
        recs.append(rec)
    need = max(1, math.ceil(a.quorum * a.expected)) if a.expected else 1
    if len(deltas) < need:
        print(json.dumps({"error": f"{len(deltas)} valid deltas, quorum {need}"}))
        return 2
    new = {}
    for n in g:
        outer = np.mean([d[n] for d in deltas], axis=0)
        m[n] = a.momentum * m[n] + outer
        new[n] = g[n] - a.outer_lr * (outer + a.momentum * m[n])
    parent = rnd["global_sha256"]
    shutil.rmtree(s / "global.new", ignore_errors=True)
    sha = write_ckpt(s / "global.new", new, st["config"]["model"], parent, {"cluster": {"round": rnd["round"] + 1}})
    save(s / "momentum.safetensors", m)
    shutil.rmtree(s / "global")
    (s / "global.new").rename(s / "global")
    tokens = sum(r["tokens"] for r in recs)
    rnd["history"].append({"round": rnd["round"], "workers": len(recs), "tokens": tokens,
                           "seconds": max(r["seconds"] for r in recs), "loss_last": float(np.mean([r["loss_last"] for r in recs if r["loss_last"] is not None] or [float("nan")])),
                           "tokens_per_s": sum(r["tokens_per_s"] or 0 for r in recs), "ts": time.time()})
    rnd.update(round=rnd["round"] + 1, global_sha256=sha)
    (s / "round.json").write_text(json.dumps(rnd, indent=2))
    print(json.dumps({"round": rnd["round"], "workers": len(recs), "tokens": tokens, "global_sha256": sha}))
    return 0


def cmd_eval(a) -> int:
    s = Path(a.state)
    r = subprocess.run([FORGE, "eval", "--ckpt", str(s / "global"), "--data", a.data, "--split", "val", "--batch", "8",
                        "--seq", "256", "--batches", str(a.batches), "--seed", "4242"], capture_output=True, text=True)
    lines = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if r.returncode or not lines:
        print(r.stderr[-1000:], file=sys.stderr)
        return r.returncode or 1
    loss = json.loads(lines[-1])["loss"]
    rnd = json.loads((s / "round.json").read_text())
    if rnd["history"]:
        rnd["history"][-1]["val_loss"] = loss
        (s / "round.json").write_text(json.dumps(rnd, indent=2))
    print(json.dumps({"round": rnd["round"], "val_loss": loss}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--from", dest="src", help="a forge checkpoint dir")
    g.add_argument("--scratch", help="a forge run config (or model config) JSON: fresh seeded weights")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--state", required=True)
    p = sub.add_parser("work")
    p.add_argument("--state", required=True)
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--threads", type=int, default=0)
    p = sub.add_parser("aggregate")
    p.add_argument("--state", required=True)
    p.add_argument("--deltas", nargs="+", required=True)
    p.add_argument("--expected", type=int, default=0)
    p.add_argument("--quorum", type=float, default=0.5)
    p.add_argument("--outer-lr", type=float, default=0.7)
    p.add_argument("--momentum", type=float, default=0.9)
    p = sub.add_parser("eval")
    p.add_argument("--state", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batches", type=int, default=32)
    a = ap.parse_args(argv)
    return {"init": cmd_init, "work": cmd_work, "aggregate": cmd_aggregate, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
