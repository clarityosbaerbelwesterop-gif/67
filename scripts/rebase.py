#!/usr/bin/env python3
"""Task-vector rebase of forge checkpoints, tensor by tensor:

    out = new_base + lam * (child - old_base)

A child fine-tuned from old_base (quasnir-1 from base-s, an RSI champion from
quasnir-g2, ...) carries its specialisation as the task vector child - old_base.
Adding that vector to a newer base (base-g2) carries the specialisation over
without retraining it, so base gains and fine-tune gains of every cycle add up.
lam scales the task vector: 0 gives new_base, 1 the full transfer.

Checkpoint layout (core_engine/train/src/ckpt.rs, trainer.rs, cli main.rs):
`forge eval --ckpt` reads model.safetensors and state.json (config.model), a
train config's "init_from" reads model.safetensors only, and the RSI loop wants
state.json's model_sha256 to match the file. OUT gets:
  model.safetensors  the rebased tensors in new_base's order, with its __metadata__
  state.json         new_base's, with model_sha256 = OUT's, parent_sha256 =
                     new_base's, loss/val_loss cleared (they measured new_base)
                     and a "rebase" record
  rebase.json        inputs (dirs + sha256 of model.safetensors), lam, OUT's sha256
  every other regular file of new_base, unchanged, except optim.safetensors: those
  Adam moments belong to new_base's weights, so OUT is an init_from start, not a
  resume_from point.
Inputs whose state.json records a model_sha256 must hash to it, and the model
configs in their state.json files must agree. Tensor names, shapes and dtypes
must match across the three checkpoints; anything else is refused.

safetensors is parsed and written by hand (the package is not installed): u64
LE header length, JSON header {name: {dtype, shape, data_offsets}, "__metadata__":
{...}} padded with spaces to 8 bytes, then the raw little-endian buffer. The
header is written the way forge writes it (sorted keys, compact), so OUT with
lam = 0 is byte-identical to new_base's model.safetensors. Dtypes F64, F32, F16
and BF16: arithmetic runs in float64 and every element is rounded once to its
dtype (BF16: uint16 <-> float32 bit shifts; on the way back float64 -> float32
rounds to odd, then the top 16 bits round to nearest even, which avoids double
rounding). With lam = 1, elements on which new_base and old_base agree take
child's value exactly; elements the child left unchanged keep new_base's. A
non-finite result is refused.

Atomic: everything is written to OUT.tmp (fsynced), then renamed to OUT. An
existing OUT is refused; an OUT.tmp left by an aborted rebase is replaced.

Usage:
  python3 scripts/rebase.py --new-base runs/base-g2/FINAL --old-base runs/base-s/FINAL \\
      --child runs/quasnir-1/FINAL [--lam 1.0] --out runs/quasnir-1-on-g2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = "model.safetensors"
STATE = "state.json"
RECORD = "rebase.json"
NOT_COPIED = ("optim.safetensors",)  # new_base's optimiser moments do not belong to the rebased weights
TMP_MARK = ".rebase-partial"
ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
            "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
FLOAT = {"F64": np.dtype("<f8"), "F32": np.dtype("<f4"), "F16": np.dtype("<f2"), "BF16": np.dtype("<u2")}


def rel(p: Path | str) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- safetensors

def read_safetensors(path: Path) -> tuple[dict, dict | None, str]:
    """(tensors in buffer order {name: {dtype, shape, data}}, __metadata__ or None, sha256 of the file)."""
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()

    def bad(msg: str) -> ValueError:
        return ValueError(f"{path}: {msg}")

    if len(raw) < 8:
        raise bad("truncated")
    hlen = int.from_bytes(raw[:8], "little")
    if 8 + hlen > len(raw):
        raise bad("header length out of range")
    try:
        header = json.loads(raw[8:8 + hlen])
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise bad(f"bad header: {e}")
    if not isinstance(header, dict):
        raise bad("header is not a JSON object")
    meta = header.pop("__metadata__", None)
    buf = memoryview(raw)[8 + hlen:]
    entries = []
    for name, v in header.items():
        if not isinstance(v, dict) or v.get("dtype") not in ITEMSIZE:
            raise bad(f"{name}: unknown dtype {v.get('dtype') if isinstance(v, dict) else v!r}")
        shape, off = v.get("shape"), v.get("data_offsets")
        if not (isinstance(shape, list) and all(isinstance(x, int) and x >= 0 for x in shape)):
            raise bad(f"{name}: bad shape {shape!r}")
        if not (isinstance(off, list) and len(off) == 2 and all(isinstance(x, int) for x in off)):
            raise bad(f"{name}: bad data_offsets {off!r}")
        b, e = off
        if not 0 <= b <= e <= len(buf) or e - b != ITEMSIZE[v["dtype"]] * int(np.prod(shape, dtype=np.int64)):
            raise bad(f"{name}: data range {off} does not hold {v['dtype']} {shape} inside a {len(buf)}-byte buffer")
        entries.append((b, e, name, v["dtype"], shape))
    entries.sort()
    for (_, e0, n0, *_), (b1, _, n1, *_) in zip(entries, entries[1:]):
        if b1 < e0:
            raise bad(f"{n0} and {n1} overlap")
    tensors = {name: {"dtype": dt, "shape": shape, "data": buf[b:e]} for b, e, name, dt, shape in entries}
    return tensors, meta, sha


def write_safetensors(path: Path, tensors: dict, meta: dict | None) -> str:
    """Write {name: {dtype, shape, data}} in the given order (header as forge writes it); returns the sha256."""
    header, off = {}, 0
    for name, t in tensors.items():
        n = len(t["data"])
        if n != ITEMSIZE[t["dtype"]] * int(np.prod(t["shape"], dtype=np.int64)):
            raise ValueError(f"{name}: {n} bytes do not hold {t['dtype']} {t['shape']}")
        header[name] = {"dtype": t["dtype"], "shape": list(t["shape"]), "data_offsets": [off, off + n]}
        off += n
    if meta is not None:
        header["__metadata__"] = meta
    hb = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    hb += b" " * (-len(hb) % 8)
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for chunk in [len(hb).to_bytes(8, "little"), hb] + [t["data"] for t in tensors.values()]:
            f.write(chunk)
            h.update(chunk)
        f.flush()
        os.fsync(f.fileno())
    return h.hexdigest()


# --------------------------------------------------------------------------- dtype conversion

def to_f64(dtype: str, data) -> np.ndarray:
    if dtype not in FLOAT:
        raise ValueError(f"dtype {dtype} is not a float type; a task vector needs F64, F32, F16 or BF16")
    x = np.frombuffer(data, dtype=FLOAT[dtype])
    if dtype == "BF16":
        x = (x.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(invalid="ignore"):  # signalling NaN patterns
        return x.astype(np.float64)


def f64_to_f32_odd(x: np.ndarray) -> np.ndarray:
    """float64 -> float32 rounded to odd: an inexact result takes the neighbour whose last bit
    is 1. Rounding that to fewer bits (bf16) equals one correct rounding of x."""
    with np.errstate(over="ignore", invalid="ignore"):
        r = x.astype(np.float32)
    fix = (r.astype(np.float64) != x) & np.isfinite(r) & ((r.view(np.uint32) & 1) == 0)
    if fix.any():
        r[fix] = np.nextafter(r[fix], np.where(x[fix] > r[fix], np.float32(np.inf), np.float32(-np.inf)))
    return r


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (uint16), round to nearest even; NaN stays a quiet NaN."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    out = ((b + (0x7FFF + ((b >> 16) & 1))) >> 16).astype(np.uint16)  # wraps only for NaN, fixed below
    nan = np.isnan(x)
    out[nan] = ((b[nan] >> 16) | 0x0040).astype(np.uint16)
    return out


def from_f64(dtype: str, x: np.ndarray) -> bytes:
    if dtype == "BF16":
        return f32_to_bf16(f64_to_f32_odd(x)).astype("<u2").tobytes()
    with np.errstate(over="ignore"):
        return x.astype(FLOAT[dtype]).tobytes()


def combine(name: str, dtype: str, n_raw, o_raw, c_raw, lam: float) -> tuple[bytes, float, float]:
    """One tensor of new + lam * (child - old); also the squared norms of child - old and new - old."""
    n, o, c = to_f64(dtype, n_raw), to_f64(dtype, o_raw), to_f64(dtype, c_raw)
    if lam == 0.0:
        out, data = n, bytes(n_raw)
    else:
        out = n + lam * (c - o)
        keep = c == o  # zero task vector: new_base's value exactly
        out[keep] = n[keep]
        if lam == 1.0:
            same = n == o  # the bases agree: the child's value exactly
            out[same] = c[same]
        data = from_f64(dtype, out)
        out = to_f64(dtype, data)
    if not np.isfinite(out).all():
        raise ValueError(f"{name}: the rebased tensor has non-finite values (check the inputs and --lam)")
    return data, float(np.square(c - o).sum()), float(np.square(n - o).sum())


# --------------------------------------------------------------------------- checkpoints

def load_ckpt(d: Path, role: str) -> dict:
    d = Path(d)
    wp = d / WEIGHTS
    if not wp.is_file():
        raise FileNotFoundError(f"{role} {d}: no {WEIGHTS}")
    tensors, meta, sha = read_safetensors(wp)
    state = None
    if (d / STATE).is_file():
        state = json.loads((d / STATE).read_text())
        recorded = state.get("model_sha256") if isinstance(state, dict) else None
        if recorded and recorded != sha:
            raise ValueError(f"{role} {d}: {WEIGHTS} has sha256 {sha}, state.json records {recorded}")
    return {"role": role, "path": str(d), "dir": d.resolve(), "tensors": tensors, "meta": meta, "sha": sha, "state": state}


def check_compatible(ck: list[dict]) -> None:
    every = set().union(*(c["tensors"] for c in ck))
    missing = [f"{c['role']} lacks {sorted(every - set(c['tensors']))[:8]}" for c in ck if every - set(c["tensors"])]
    if missing:
        raise ValueError("tensor names differ: " + "; ".join(missing))
    for name in ck[0]["tensors"]:
        sig = [(c["tensors"][name]["dtype"], list(c["tensors"][name]["shape"])) for c in ck]
        if any(s != sig[0] for s in sig[1:]):
            raise ValueError(f"{name}: dtype/shape differ: " + ", ".join(f"{c['role']} {s[0]} {s[1]}" for c, s in zip(ck, sig)))
    models = [(c["role"], (c["state"].get("config") or {}).get("model")) for c in ck if isinstance(c["state"], dict)]
    models = [(r, m) for r, m in models if m is not None]
    if any(m != models[0][1] for _, m in models[1:]):
        raise ValueError("model configs differ: " + "; ".join(f"{r} {m}" for r, m in models))


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rebase(new_base, old_base, child, out, lam: float = 1.0) -> dict:
    """Write OUT = new_base + lam * (child - old_base); returns the rebase.json record."""
    lam = float(lam)
    if not np.isfinite(lam):
        raise ValueError("lam must be finite")
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"{out} exists; rebase never overwrites a checkpoint")
    nb, ob, ch = load_ckpt(new_base, "new_base"), load_ckpt(old_base, "old_base"), load_ckpt(child, "child")
    if not isinstance(nb["state"], dict) or not (nb["state"].get("config") or {}).get("model"):
        raise ValueError(f"new_base {new_base}: state.json with config.model is required (forge eval reads it)")
    check_compatible([nb, ob, ch])

    tensors, dtypes, tv2, shift2, params = {}, {}, 0.0, 0.0, 0
    for name, t in nb["tensors"].items():  # everything is computed (and checked) before anything is written
        data, a, b = combine(name, t["dtype"], t["data"], ob["tensors"][name]["data"], ch["tensors"][name]["data"], lam)
        tensors[name] = {"dtype": t["dtype"], "shape": t["shape"], "data": data}
        dtypes[t["dtype"]] = dtypes.get(t["dtype"], 0) + 1
        tv2, shift2, params = tv2 + a, shift2 + b, params + int(np.prod(t["shape"], dtype=np.int64))

    if tmp.exists():
        if not (tmp / TMP_MARK).exists():
            raise FileExistsError(f"{tmp} exists and was not left by rebase; remove it first")
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        (tmp / TMP_MARK).write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}))
        sha = write_safetensors(tmp / WEIGHTS, tensors, nb["meta"])
        copied = []
        for f in sorted(nb["dir"].iterdir()):
            if f.name in (WEIGHTS, STATE, RECORD, *NOT_COPIED) or f.name.endswith(".tmp") or not f.is_file():
                continue
            shutil.copyfile(f, tmp / f.name)
            with open(tmp / f.name, "rb+") as fh:
                os.fsync(fh.fileno())
            copied.append(f.name)
        inputs = {c["role"]: {"path": c["path"], "dir": rel(c["dir"]), "model_sha256": c["sha"]} for c in (nb, ob, ch)}
        record = {"type": "rebase", "formula": "out = new_base + lam * (child - old_base)", "lam": lam, "inputs": inputs,
                  "out": {"dir": rel(out.resolve()), "model_sha256": sha}, "tensors": len(tensors), "params": params,
                  "dtypes": dtypes, "task_vector_norm": tv2 ** 0.5, "base_shift_norm": shift2 ** 0.5,
                  "copied": copied, "not_copied": [n for n in NOT_COPIED if (nb["dir"] / n).exists()], "ts": time.time()}
        state = dict(nb["state"])
        state.update(model_sha256=sha, parent_sha256=nb["sha"], loss=None, val_loss=None,
                     rebase={"lam": lam, **{k: v["model_sha256"] for k, v in inputs.items()},
                             "dirs": {k: v["dir"] for k, v in inputs.items()}})
        for name, obj in ((STATE, state), (RECORD, record)):
            with open(tmp / name, "w") as f:
                f.write(json.dumps(obj, indent=2) + "\n")
                f.flush()
                os.fsync(f.fileno())
        (tmp / TMP_MARK).unlink()
        _fsync_dir(tmp)
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _fsync_dir(out.resolve().parent)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 scripts/rebase.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-base", required=True, help="checkpoint dir the task vector is added to")
    ap.add_argument("--old-base", required=True, help="checkpoint dir the child was fine-tuned from")
    ap.add_argument("--child", required=True, help="fine-tuned checkpoint dir (task vector = child - old_base)")
    ap.add_argument("--lam", type=float, default=1.0, help="task-vector scale")
    ap.add_argument("--out", required=True, help="new checkpoint dir (must not exist)")
    a = ap.parse_args(argv)
    try:
        rec = rebase(a.new_base, a.old_base, a.child, a.out, a.lam)
    except (ValueError, OSError) as e:
        print(json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}), file=sys.stderr)
        return 1
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
