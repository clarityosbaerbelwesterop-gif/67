"""forge's on-disk formats and host-side algorithms, bit-exact where forge is deterministic.

- safetensors read/write through scripts/rebase.py (the repository's reader/writer), with
  forge's conventions: F32 only, tensors in byte order of their names, temp file
  `model.tmp` / `optim.tmp` then rename (ckpt.rs:98-135).
- state.json / latest.json / telemetry JSON exactly as serde_json prints them: sorted keys,
  ryu float formatting (0.000039999998989515007, 9.999999747378752e-6), pretty = 2 spaces,
  no trailing newline (trainer.rs:520-561).
- forge's Rng (xoshiro256** seeded through splitmix64, rng.rs), so data windows, fresh
  initialisation and rng_state match forge for the same seed.
- Token stores (`train.meta.json` + headerless LE uint16/uint32 bins) and `synthetic:P`,
  with forge's window sampler (data.rs).
- Lineage: model_sha256 = SHA-256 of model.safetensors; parent_sha256 = SHA-256 of the
  init_from checkpoint's model.safetensors (copied on resume, null for a fresh init).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rebase as R  # noqa: E402  (the repository's safetensors reader/writer, as cluster/fabric.py uses it)

f32 = np.float32
M64 = (1 << 64) - 1

# ---------------------------------------------------------------------------------------------- libm
# forge computes RoPE tables, bias corrections and the LR cosine with Rust's f32::powf/cos/sin, which
# call the C library's powf/cosf/sinf. Calling the same functions keeps those constants bit-identical.
try:
    _libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    for _n in ("powf", "cosf", "sinf"):
        getattr(_libm, _n).restype = ctypes.c_float
    _libm.powf.argtypes = (ctypes.c_float, ctypes.c_float)
    _libm.cosf.argtypes = (ctypes.c_float,)
    _libm.sinf.argtypes = (ctypes.c_float,)
except (OSError, AttributeError):  # pragma: no cover - non-glibc host: correctly rounded f64 fallback
    _libm = None


def powf(a, b) -> np.float32:
    return f32(_libm.powf(float(a), float(b))) if _libm else f32(math.pow(float(a), float(b)))


def cosf(a) -> np.float32:
    return f32(_libm.cosf(float(a))) if _libm else f32(math.cos(float(a)))


def sinf(a) -> np.float32:
    return f32(_libm.sinf(float(a))) if _libm else f32(math.sin(float(a)))


# ---------------------------------------------------------------------------------------------- RNG
class Rng:
    """forge's xoshiro256** (core_engine/train/src/rng.rs)."""

    def __init__(self, seed: int = 0, state=None):
        if state is not None:
            self.s = [int(x) & M64 for x in state]
            return
        x = int(seed) & M64
        s = []
        for _ in range(4):
            x = (x + 0x9E3779B97F4A7C15) & M64
            z = x
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
            s.append(z ^ (z >> 31))
        self.s = s

    def state(self) -> list[int]:
        return list(self.s)

    def clone(self) -> "Rng":
        return Rng(state=self.s)

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self.s
        x = (s1 * 5) & M64
        r = ((((x << 7) | (x >> 57)) & M64) * 9) & M64
        t = (s1 << 17) & M64
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = ((s3 << 45) | (s3 >> 19)) & M64
        self.s = [s0, s1, s2, s3]
        return r

    def next_f64(self) -> float:
        return (self.next_u64() >> 11) / float(1 << 53)

    def below(self, n: int) -> int:
        return (self.next_u64() * int(n)) >> 64

    def normal(self) -> np.float32:
        u1 = max(self.next_f64(), 2.2250738585072014e-308)
        u2 = self.next_f64()
        return f32(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))

    def normals(self, n: int) -> np.ndarray:
        """n successive normal() values (f32), the loop inlined (~1.7 s per million)."""
        s0, s1, s2, s3 = self.s
        out = np.empty(n, np.float64)
        log, cos, sqrt = math.log, math.cos, math.sqrt
        two_pi, inv53, tiny, M = 2.0 * math.pi, 1.0 / float(1 << 53), 2.2250738585072014e-308, M64
        for i in range(n):
            x = (s1 * 5) & M
            r = ((((x << 7) | (x >> 57)) & M) * 9) & M
            t = (s1 << 17) & M
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t  # noqa: E702
            s3 = ((s3 << 45) | (s3 >> 19)) & M
            u1 = (r >> 11) * inv53
            x = (s1 * 5) & M
            r = ((((x << 7) | (x >> 57)) & M) * 9) & M
            t = (s1 << 17) & M
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t  # noqa: E702
            s3 = ((s3 << 45) | (s3 >> 19)) & M
            u2 = (r >> 11) * inv53
            if u1 < tiny:
                u1 = tiny
            out[i] = sqrt(-2.0 * log(u1)) * cos(two_pi * u2)
        self.s = [s0, s1, s2, s3]
        return out.astype(f32)  # each value rounded once from f64, like `as f32`


# ---------------------------------------------------------------------------------------------- JSON
def _fmt_float(x: float) -> str:
    """serde_json's f64 formatting (the ryu crate's `format64`): shortest round-trip digits,
    decimal notation for 1e-5 <= |x| < 1e16, otherwise d.ddde<exp> without '+'."""
    if not math.isfinite(x):
        return "null"  # serde_json: Value::from(NaN) is Null
    r = repr(float(x))
    sign = ""
    if r[0] == "-":
        sign, r = "-", r[1:]
    mant, _, exp = r.partition("e")
    e10 = int(exp) if exp else 0
    ip, _, fp = mant.partition(".")
    ds = (ip + fp).lstrip("0")
    e10 -= len(fp)
    if not ds:
        return sign + "0.0"
    stripped = ds.rstrip("0")
    e10 += len(ds) - len(stripped)
    ds = stripped
    length, k = len(ds), e10
    kk = length + k
    if 0 <= k and kk <= 16:
        return sign + ds + "0" * k + ".0"
    if 0 < kk <= 16:
        return sign + ds[:kk] + "." + ds[kk:]
    if -5 < kk <= 0:
        return sign + "0." + "0" * (-kk) + ds
    if length == 1:
        return sign + ds + "e" + str(kk - 1)
    return sign + ds[0] + "." + ds[1:] + "e" + str(kk - 1)


def _enc(v, indent: int | None, level: int) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return _fmt_float(float(v))
    if isinstance(v, (str, Path)):
        return json.dumps(str(v), ensure_ascii=False)
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = sorted((str(k), x) for k, x in v.items())
        if indent is None:
            return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _enc(x, None, 0) for k, x in items) + "}"
        pad, pad1 = " " * (indent * level), " " * (indent * (level + 1))
        body = ",\n".join(pad1 + json.dumps(k, ensure_ascii=False) + ": " + _enc(x, indent, level + 1) for k, x in items)
        return "{\n" + body + "\n" + pad + "}"
    if isinstance(v, (list, tuple, np.ndarray)):
        v = list(v)
        if not v:
            return "[]"
        if indent is None:
            return "[" + ",".join(_enc(x, None, 0) for x in v) + "]"
        pad, pad1 = " " * (indent * level), " " * (indent * (level + 1))
        return "[\n" + ",\n".join(pad1 + _enc(x, indent, level + 1) for x in v) + "\n" + pad + "]"
    raise TypeError(f"cannot serialise {type(v).__name__}")


def dumps(v, pretty: bool = False) -> str:
    """JSON exactly as forge's serde_json prints a Value: sorted keys, compact (`{v}`) or
    to_string_pretty (2 spaces), ryu floats, NaN/inf as null."""
    return _enc(v, 2 if pretty else None, 0)


# ---------------------------------------------------------------------------------------------- safetensors
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors(path, allow=("F32",)) -> tuple[dict, dict | None, str]:
    """({name: float32 ndarray}, __metadata__, sha256). forge loads F32 only (ckpt.rs:169)."""
    ts, meta, sha = R.read_safetensors(Path(path))
    out = {}
    for name, t in ts.items():
        if t["dtype"] not in allow:
            raise ValueError(f"{path}: {name}: only {'/'.join(allow)} is supported, not {t['dtype']}")
        dt = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}[t["dtype"]]
        out[name] = np.frombuffer(t["data"], dt).astype(f32).reshape(t["shape"])
    return out, meta, sha


def write_safetensors(path, arrays: dict, meta: dict) -> str:
    """forge's save_safetensors: F32, tensors in name order (BTreeMap), temp file `<stem>.tmp`
    then rename. Returns the file's SHA-256."""
    path = Path(path)
    tensors = {}
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name], dtype="<f4")
        tensors[name] = {"dtype": "F32", "shape": list(a.shape), "data": a.reshape(-1).view(np.uint8)}
    tmp = path.with_suffix(".tmp")
    sha = R.write_safetensors(tmp, tensors, {k: str(v) for k, v in meta.items()})
    os.replace(tmp, path)
    return sha


# ---------------------------------------------------------------------------------------------- checkpoints
def load_checkpoint(d) -> tuple[dict, dict, str]:
    """What forge's CLI needs from a checkpoint (main.rs:146-159): (state.json, {name: f32}, sha256 of
    model.safetensors)."""
    d = Path(d)
    state = json.loads((d / "state.json").read_text())
    weights, _, sha = read_safetensors(d / "model.safetensors")
    return state, weights, sha


def check_weights(weights: dict, specs: list, what: str) -> dict:
    out = {}
    for name, shape in specs:
        if name not in weights:
            raise ValueError(f"{what}: missing {name}")
        if tuple(weights[name].shape) != tuple(shape):
            raise ValueError(f"{what}: {name} has shape {list(weights[name].shape)}, model wants {list(shape)}")
        out[name] = weights[name]
    return out


def save_checkpoint(d, step: int, params: dict, m: dict, v: dict, state: dict) -> str:
    """trainer.rs save(): model.safetensors, optim.safetensors (m under <name>, v under v.<name>) and
    state.json (pretty, sorted, no trailing newline). `state` holds every key but model_sha256.
    Returns model_sha256."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"format": "pt", "step": str(step)}
    sha = write_safetensors(d / "model.safetensors", params, meta)
    opt = dict(m)
    opt.update({f"v.{k}": x for k, x in v.items()})
    write_safetensors(d / "optim.safetensors", opt, meta)
    st = dict(state)
    st["model_sha256"] = sha
    (d / "state.json").write_text(dumps(st, pretty=True))
    return sha


def write_latest(run_dir, ckpt_dir: str, step: int, sha: str) -> None:
    """<out_dir>/<run>/latest.json, compact sorted JSON; `dir` as forge's Path::join spells it."""
    Path(run_dir, "latest.json").write_text(dumps({"dir": ckpt_dir, "step": step, "model_sha256": sha}))


def set_final(run_dir, target) -> Path:
    """FINAL -> step-N as scripts/longrun.py finalize() writes it: a relative link, swapped in atomically."""
    run_dir, target = Path(run_dir), Path(target)
    link, tmp = run_dir / "FINAL", run_dir / ".FINAL.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target.name, tmp)
    os.replace(tmp, link)
    return link


# ---------------------------------------------------------------------------------------------- data
class Store:
    """data.rs: a store meta (train.meta.json) or `synthetic:P`. batch() is forge's sample_windows."""

    def __init__(self, spec: str, model_vocab: int):
        self.spec = spec
        if spec.startswith("synthetic:"):
            p = spec.split(":", 1)[1]
            if not p.isdigit():
                raise ValueError(f"bad synthetic period {p}")
            self.period, self.vocab, self.n = int(p), model_vocab, 1 << 20
            if self.period == 0:
                raise ValueError("synthetic period must be positive")
            self.train = self.val = None
            return
        self.period = None
        path = Path(spec)
        m = json.loads(path.read_text())
        dt = m.get("dtype", "uint16")
        if dt not in ("uint16", "uint32"):
            raise ValueError(f"{spec}: unsupported dtype {dt}")
        if not isinstance(m.get("vocab_size"), int):
            raise ValueError(f"{spec}: vocab_size missing")
        self.vocab = m["vocab_size"]
        width = 2 if dt == "uint16" else 4
        bins = {}
        for key, nkey in (("bin", "tokens"), ("val_bin", "val_tokens")):
            if not isinstance(m.get(key), str):
                raise ValueError(f"{spec}: {key} missing")
            p = path.parent / Path(m[key]).name
            size = p.stat().st_size
            if size % width or (isinstance(m.get(nkey), int) and m[nkey] != size // width):
                raise ValueError(f"{spec}: {nkey} does not match the bin size")
            bins[key] = np.memmap(p, "<u2" if width == 2 else "<u4", "r") if size else np.zeros(0, np.uint16)
        self.train, self.val = bins["bin"], bins["val_bin"]

    def __len__(self):
        return self.n if self.period else len(self.train)

    def split_len(self, split: str) -> int:
        if self.period:
            return self.n
        return len(self.val if split == "val" else self.train)

    def _tokens(self, split: str, idx: np.ndarray) -> np.ndarray:
        if self.period:
            return ((idx % self.period) % self.vocab).astype(np.int32)
        tok = self.val if split == "val" else self.train
        return np.asarray(tok[idx], dtype=np.int64).astype(np.int32)

    def starts(self, rng: Rng, split: str, b: int, t: int) -> np.ndarray:
        n = self.split_len(split)
        if n <= t + 1:
            raise ValueError(f"split has {n} tokens, need more than {t + 1}")
        return np.array([rng.below(n - t - 1) for _ in range(b)], dtype=np.int64)

    def windows(self, split: str, starts: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        idx = starts[:, None] + np.arange(t + 1, dtype=np.int64)[None, :]
        w = self._tokens(split, idx)
        return w[:, :-1], w[:, 1:]

    def batch(self, rng: Rng, split: str, b: int, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) int32 [b, t]: per row start = below(N - t - 1), x = w[s:s+t], y = w[s+1:s+t+1]."""
        return self.windows(split, self.starts(rng, split, b, t), t)
