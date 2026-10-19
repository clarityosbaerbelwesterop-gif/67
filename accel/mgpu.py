"""Multi-device FSDP trainer for Quasnir (jax.sharding Mesh + NamedSharding, GSPMD).

Mesh axes ('data', 'fsdp'). Params AND AdamW m/v sharded along 'fsdp' (largest divisible dim);
batch rows sharded over ('data','fsdp') (every device sees distinct rows, as in FSDP/ZeRO-3).
XLA inserts the all-gathers of params and reduce-scatters of grads. Model math = accel.model
(precision "f32" or "bf16": bf16 GEMM inputs, f32 accumulate, f32 master params/opt state).
Checkpoints: forge format (model.safetensors, optim.safetensors with m=<name>, v=v.<name>, state.json),
gathered to host per tensor. Data order is a pure function of (seed, step) -> deterministic resume.
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from accel.config import ModelConfig
from accel import model as M
from accel import forge_io as fio

f32 = np.float32

from accel.config import TrainConfig
from accel.forge_io import Rng, Store, powf

# Champion AdamW constants (accel/train.py:49-50, identical in phaseb-trainer/accel/train.py:49)
B1, B2, EPS = f32(0.9), f32(0.95), f32(1e-8)
OMB1, OMB2 = f32(f32(1.0) - B1), f32(f32(1.0) - B2)

DEFAULT_CLEAN_TSV = "/workspace/contam-check/dedup-v4/basev3-full/kept_t50_drop20.tsv"

class DataSafetyError(RuntimeError):
    """Clean-data requirement violated; the trainer must stop, never fall back to unfiltered data."""

@dataclass
class HP:
    batch: int = 16          # global rows per micro-batch
    seq: int = 32
    accum: int = 2
    # champion values (quasnir-phase-a-100m step-005744/state.json "config"):
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    lr_schedule: str = "wsd"  # decay = round(0.2*max_steps) unless decay_steps > 0 (config.rs via accel/config.py)
    max_steps: int = 5744
    decay_steps: int = 0
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    optim_offset: int = 0    # Adam bias-correction step offset (champion resumed with 972)
    seed: int = 1234
    precision: str = "f32"   # or "bf16"
    remat: bool = False
    attn: str = "naive"      # naive | blockwise[:QB[:KB]] | xla | cudnn | auto
    data: str = ""           # Store spec (train.meta.json or synthetic:P); "" = toy synthetic batch_at
    clean_tsv: str = DEFAULT_CLEAN_TSV  # kept-docs TSV (doc_idx,start,len); REQUIRED whenever data is set
    verify_sha: bool = True  # check train.bin sha256 vs report.json next to the TSV (False only for tests)
    allow_unfiltered: bool = False  # tests only: permit sampling raw train.bin without the kept-docs filter

    # exact reuse of the existing trainer's schedule code (f32, forge semantics)
    @staticmethod
    def preset(name: str, **kw) -> "HP":
        return HP(**{**PRESETS[name], **kw})

    wsd_decay_steps = TrainConfig.wsd_decay_steps
    lr_at = TrainConfig.lr_at

PRESETS = {
    # champion quasnir-phase-a-100m (step-005744/state.json config)
    "champion": dict(lr=3e-4, min_lr=3e-5, warmup_steps=100, lr_schedule="wsd", decay_steps=0,
                     weight_decay=0.1, grad_clip=1.0),
    # s2 (959.5M grown): lower peak, longer warmup; rest as champion (wsd, decay = 20% of max_steps)
    "s2": dict(lr=2e-4, min_lr=3e-5, warmup_steps=500, lr_schedule="wsd", decay_steps=0,
               weight_decay=0.1, grad_clip=1.0),
    # first budget run of s2 (decided): 1000 steps = warmup 300 (fixed) + stable 500 @2e-4 + decay last 200 -> 3e-5
    "s2_run1": dict(lr=2e-4, min_lr=3e-5, warmup_steps=300, lr_schedule="wsd", decay_steps=0, max_steps=1000,
                    weight_decay=0.1, grad_clip=1.0),
}

def make_mesh(data: int, fsdp: int, devices=None) -> Mesh:
    devs = np.array((devices or jax.devices())[: data * fsdp]).reshape(data, fsdp)
    return Mesh(devs, ("data", "fsdp"))

def param_spec(shape, fsdp: int) -> P:
    if len(shape) < 2 or fsdp == 1:
        return P()
    for ax in sorted(range(len(shape)), key=lambda a: -shape[a]):
        if shape[ax] % fsdp == 0:
            s = [None] * len(shape); s[ax] = "fsdp"; return P(*s)
    return P()

def shardings(mc: ModelConfig, mesh: Mesh) -> dict:
    n = mesh.shape["fsdp"]
    return {k: NamedSharding(mesh, param_spec(s, n)) for k, s in mc.param_specs()}

def batch_sharding(mesh):  # xs [A, B, T]
    return NamedSharding(mesh, P(None, ("data", "fsdp"), None))

def batch_at(hp: HP, vocab: int, step: int):
    """Toy deterministic synthetic batch (used when hp.data == "")."""
    r = np.random.default_rng([hp.seed, step])
    x = r.integers(0, vocab, (hp.accum, hp.batch, hp.seq + 1), dtype=np.int64)
    x[..., 1:] = (x[..., :-1] * 7 + 3) % vocab
    noise = r.random(x.shape) < 0.1
    x = np.where(noise, r.integers(0, vocab, x.shape), x).astype(np.int32)
    return x[..., :-1], x[..., 1:]

M64 = (1 << 64) - 1
def step_seed(seed: int, step: int, micro: int) -> int:
    """splitmix-style mix of (seed, step, micro) -> Rng seed. Window starts depend only on these,
    never on how many batches were drawn before -> resume needs only `step`."""
    z = (seed * 0x9E3779B97F4A7C15 + step * 0xBF58476D1CE4E5B9 + micro * 0x94D049BB133111EB + 1) & M64
    z ^= z >> 31; z = (z * 0xD6E8FEB86CD8B5AD) & M64; z ^= z >> 29
    return z

class _TrainBin:
    """train.bin of a store meta, memmapped read-only; same checks as Store but no val_bin requirement."""
    def __init__(self, meta: str, vocab: int):
        m = json.loads(Path(meta).read_text())
        dt = m.get("dtype") if isinstance(m.get("dtype"), str) else "uint16"
        if dt not in ("uint16", "uint32"):
            raise ValueError(f"{meta}: unsupported dtype {dt}")
        w = 2 if dt == "uint16" else 4
        p = Path(meta).parent / Path(m["bin"]).name
        n = p.stat().st_size
        if n % w or (isinstance(m.get("tokens"), int) and m["tokens"] != n // w):
            raise ValueError(f"{meta}: tokens does not match the bin size")
        self.train = np.memmap(p, "<u2" if w == 2 else "<u4", "r"); self.limit = int(vocab)

    def __len__(self):
        return len(self.train)

class CleanStore:
    """Only kept docs. Virtual stream = concatenation of kept docs, each WITH its EOS:
    segment i covers train.bin[start_i .. start_i + len_i] inclusive (len excludes the EOS at start+len).
    Windows are sampled uniformly over the virtual stream and gathered from (possibly several) kept
    segments, so a window never touches a dropped doc. Index lives in RAM (3 int64 arrays), bin is memmapped."""
    def __init__(self, meta: str, tsv: str, vocab: int, verify_sha: bool = True):
        import hashlib, time
        self.base = _TrainBin(meta, vocab)   # train split only: the val bin is never needed (nor shipped to pods)
        # kept-docs index: TSV with a header, or (corpus v6, ~1e8+ documents) the same 3 columns as a .npy array
        a = (np.load(tsv).astype(np.int64).reshape(-1, 3) if str(tsv).endswith(".npy")
             else np.loadtxt(tsv, dtype=np.int64, delimiter="\t", skiprows=1, ndmin=2))
        self.doc, self.seg_start, self.seg_len = a[:, 0], a[:, 1], a[:, 2] + 1
        if np.any(np.diff(self.seg_start) <= 0):
            raise ValueError("kept TSV starts must be strictly increasing")
        if int(self.seg_start[-1] + self.seg_len[-1]) > len(self.base.train):
            raise ValueError("kept segment beyond train.bin")
        self.cum = np.concatenate([[0], np.cumsum(self.seg_len)])  # virtual offset of each segment
        self.n = int(self.cum[-1])
        self.sha_seconds = None
        if verify_sha:
            rep = json.loads((Path(tsv).parent / "report.json").read_text())
            binp = Path(meta).parent / json.loads(Path(meta).read_text())["bin"]
            t0 = time.time(); h = hashlib.sha256()
            with open(binp, "rb") as f:
                for c in iter(lambda: f.read(1 << 24), b""):
                    h.update(c)
            self.sha_seconds = time.time() - t0
            if h.hexdigest() != rep["train_bin_sha256_expected"]:
                raise DataSafetyError(f"train.bin sha256 {h.hexdigest()} != report {rep['train_bin_sha256_expected']}")
        self.limit = self.base.limit

    def split_len(self, split="train"):
        return self.n

    def starts(self, rng, split, b, t):
        return np.array([rng.below(self.n - t - 1) for _ in range(b)], dtype=np.int64)

    def physical(self, vpos: np.ndarray) -> np.ndarray:
        seg = np.searchsorted(self.cum, vpos, side="right") - 1
        return self.seg_start[seg] + (vpos - self.cum[seg])

    def windows(self, split, starts, t):
        v = starts[:, None] + np.arange(t + 1, dtype=np.int64)[None, :]
        w = np.asarray(self.base.train[self.physical(v)], dtype=np.int64)
        x, y = w[:, :-1], w[:, 1:]
        fio.check_ids(x, self.limit); fio.check_ids(y, self.limit)
        return x.astype(np.int32), y.astype(np.int32)

def open_store(hp: "HP", vocab: int):
    """Fail-closed: clean kept-docs store, or raise DataSafetyError. Raw Store only with allow_unfiltered (tests)."""
    if hp.data.startswith("synthetic:"):
        return Store(hp.data, vocab)
    if not hp.clean_tsv:
        if hp.allow_unfiltered:
            return Store(hp.data, vocab)
        raise DataSafetyError("clean_tsv not set: refusing to train on unfiltered data")
    try:
        return CleanStore(hp.data, hp.clean_tsv, vocab, hp.verify_sha)
    except DataSafetyError:
        raise
    except Exception as e:  # missing/unreadable TSV or report.json, bad index, ... -> no fallback
        raise DataSafetyError(f"clean data unavailable ({type(e).__name__}: {e}); refusing to fall back") from e

class ShardedData:
    """Store (memmap, read-only) -> global xs/ys [A,B,T] whose rows are sharded over ('data','fsdp');
    each device's callback reads only its own rows from the memmap."""
    def __init__(self, hp: HP, mesh: Mesh, vocab: int):
        self.hp, self.sh = hp, batch_sharding(mesh)
        self.store = open_store(hp, vocab)
        self._cache = {}

    def starts(self, step: int) -> np.ndarray:  # [A,B] int64
        if step not in self._cache:
            n, T = self.store.split_len("train"), self.hp.seq
            self._cache = {step: np.stack([self.store.starts(Rng(step_seed(self.hp.seed, step, a)), "train",
                                                             self.hp.batch, T) for a in range(self.hp.accum)])}
        return self._cache[step]

    def global_batch(self, step: int):
        st, T, A, B = self.starts(step), self.hp.seq, self.hp.accum, self.hp.batch
        def rows(which, idx):
            ia, ib = idx[0], idx[1]
            s = st[ia, ib]                                   # [a,b]
            x, y = self.store.windows("train", s.reshape(-1), T)
            return (x if which == 0 else y).reshape(s.shape + (T,))
        mk = lambda w: jax.make_array_from_callback((A, B, T), self.sh, lambda idx: rows(w, idx))
        return mk(0), mk(1)

def build_step(mc: ModelConfig, hp: HP, mesh: Mesh):
    """(params, m, v, xs, ys, hyper[step_size=lr/bc1, sqrt(bc2), 1-lr*wd, 0]) -> (params, m, v, loss, gnorm).
    AdamW exactly as accel/train.py build_step (same constants, operation order, no-fma rounding,
    decay only on >=2-D tensors, global-norm clip incl. tok_emb); gradients reduce-scattered to fsdp shards."""
    sh = shardings(mc, mesh); bs = batch_sharding(mesh); rep = NamedSharding(mesh, P())
    decays = {k: len(s) >= 2 for k, s in mc.param_specs()}
    gs = f32(1.0) / f32(hp.batch * hp.seq * hp.accum)
    gc = f32(hp.grad_clip)
    from accel.attn import resolve
    attn = resolve(hp.attn)

    def lossf(p, x, y):
        ce, _ = M.row_losses(p, mc, x, y, hp.precision, hp.remat, "fast", attn)
        return jnp.sum(ce) * gs, M.pairwise_sum(ce)

    vg = jax.value_and_grad(lossf, has_aux=True)

    def step(params, m, v, xs, ys, hyper):
        step_size, bc2s, decay = hyper[0], hyper[1], hyper[2]
        zero = jax.lax.bitcast_convert_type(hyper[3], jnp.int32)
        rounded = lambda x: jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(x, jnp.int32) ^ zero, jnp.float32)

        def micro(g, xy):
            (_, cs), gi = vg(params, xy[0], xy[1])
            gi = jax.lax.with_sharding_constraint(gi, sh)
            return jax.tree.map(jnp.add, g, gi), cs
        g, ce_sums = jax.lax.scan(micro, jax.tree.map(jnp.zeros_like, params), (xs, ys))
        sumsq = f32(0.0)
        for k in params:
            sumsq = sumsq + M.pairwise_sum(g[k] * g[k])
        norm = jnp.sqrt(jnp.where(jnp.isnan(sumsq), f32(0.0), jnp.maximum(sumsq, f32(0.0))))
        clip = jnp.minimum(gc / (norm + f32(1e-6)), f32(1.0))
        P_, M_, V_ = {}, {}, {}
        for k in params:
            gi = g[k] * clip
            mm = rounded(B1 * m[k]) + rounded(OMB1 * gi)
            vv = rounded(B2 * v[k]) + rounded((OMB2 * gi) * gi)
            dec = decay if decays[k] else f32(1.0)
            P_[k] = rounded(params[k] * dec) - (step_size * mm) / (jnp.sqrt(vv) / bc2s + EPS)
            M_[k], V_[k] = mm, vv
        loss = jnp.sum(ce_sums) / f32(hp.batch * hp.seq * hp.accum)
        return P_, M_, V_, loss, norm

    return jax.jit(step, in_shardings=(sh, sh, sh, bs, bs, rep),
                   out_shardings=(sh, sh, sh, rep, rep), donate_argnums=(0, 1, 2))

def hyper_at(hp: HP, step: int) -> np.ndarray:
    """As accel Trainer.train_step: update `step` (0-based) uses lr_at(step), bias correction t = offset+step+1."""
    lr = hp.lr_at(step)
    t = f32(hp.optim_offset + step + 1)
    bc1 = f32(f32(1.0) - powf(B1, t))
    bc2s = np.sqrt(f32(f32(1.0) - powf(B2, t)), dtype=f32)
    return np.array([f32(lr / bc1), bc2s, f32(f32(1.0) - lr * f32(hp.weight_decay)), 0.0], f32)

def put(tree: dict, sh: dict) -> dict:
    return {k: jax.device_put(np.asarray(a, f32), sh[k]) for k, a in tree.items()}

def init_state(mc, mesh, params_host: dict):
    sh = shardings(mc, mesh)
    p = put(params_host, sh)
    z = jax.jit(lambda q: jax.tree.map(jnp.zeros_like, q), out_shardings=sh)
    return p, z(p), z(p), jnp.zeros((), jnp.int32)

def gather(tree):  # per-tensor to host (only one full tensor live at a time on device side)
    return {k: np.asarray(jax.device_get(a)) for k, a in tree.items()}

def save(d, mc, hp, params, m, v, t):
    step = int(t)
    st = {"step": step, "seed": hp.seed, "hp": {k: getattr(hp, k) for k in hp.__dataclass_fields__}, "data_position": {"rule": "starts = Store.starts(Rng(step_seed(seed, step, micro)))", "next_step": step}, "config": {"model": mc.to_dict()}, "trainer": "mgpu"}
    return fio.save_checkpoint(d, step, gather(params), gather(m), gather(v), st)

def load(d, mc, mesh, with_optim=True):
    d = Path(d); st = json.loads((d / "state.json").read_text())
    w, _, _ = fio.read_safetensors(d / "model.safetensors")
    w = fio.check_weights(w, mc.param_specs(), str(d))
    sh = shardings(mc, mesh)
    p = put(w, sh)
    if with_optim and (d / "optim.safetensors").exists():
        o, _, _ = fio.read_safetensors(d / "optim.safetensors")
        m = put({k: o[k] for k in w}, sh); v = put({k: o["v." + k] for k in w}, sh)
        t = jnp.asarray(st.get("step", 0), jnp.int32)
    else:  # init_from (e.g. champion / s2): weights only, fresh AdamW state
        _, m, v, t = init_state(mc, mesh, w)
    return p, m, v, t, st

# ---- lazy, shard-wise loading of big checkpoints (no full host copy) ----
def st_header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little"); h = json.loads(f.read(n))
    return h, 8 + n

def lazy_sharded(path, name, sharding, header=None):
    """Build a global jax.Array for tensor `name` where each device reads only its slice via np.memmap."""
    h, off = header or st_header(path)
    e = h[name]; assert e["dtype"] == "F32"
    a, b = e["data_offsets"]
    mm = np.memmap(path, dtype="<f4", mode="r", offset=off + a, shape=tuple(e["shape"]))
    return jax.make_array_from_callback(tuple(e["shape"]), sharding, lambda idx: np.ascontiguousarray(mm[idx]))

def train(mc, hp, mesh, state, n_steps, log=None, record_ids=None):
    fn = build_step(mc, hp, mesh); bs = batch_sharding(mesh); rep = NamedSharding(mesh, P())
    data = ShardedData(hp, mesh, mc.vocab_size) if hp.data else None
    p, m, v, t = state; t = int(t); losses = []
    for _ in range(n_steps):
        if data:
            xs, ys = data.global_batch(t)
        else:
            x, y = batch_at(hp, mc.vocab_size, t); xs, ys = jax.device_put(x, bs), jax.device_put(y, bs)
        if record_ids is not None:
            record_ids.append(np.asarray(jax.device_get(xs)))
        p, m, v, l, gn = fn(p, m, v, xs, ys, jax.device_put(hyper_at(hp, t), rep))
        t += 1; losses.append(float(l))
        if log: log(t, float(l), float(gn))
    return (p, m, v, jnp.asarray(t, jnp.int32)), losses


def export_bf16(path, params: dict, meta: dict | None = None) -> dict:
    """Weights-only export for download (Gate B): safetensors, dtype BF16 (round-to-nearest-even from the
    f32 master), tensors in name order, NO optimizer state. Streams one gathered tensor at a time.
    The full f32 checkpoint (model+optim+state) is written separately by save() and stays on the pod."""
    import hashlib
    import ml_dtypes
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(params); hdr = {"__metadata__": {"format": "pt", "dtype": "bf16", "weights_only": "true",
                                                     **{k: str(v) for k, v in (meta or {}).items()}}}
    off = 0
    for k in names:
        n = int(np.prod(params[k].shape)) * 2
        hdr[k] = {"dtype": "BF16", "shape": list(params[k].shape), "data_offsets": [off, off + n]}; off += n
    hj = json.dumps(hdr, separators=(",", ":")).encode(); hj += b" " * (-len(hj) % 8)
    tmp = path.with_suffix(".tmp"); h = hashlib.sha256()
    with open(tmp, "wb") as f:
        for chunk in (len(hj).to_bytes(8, "little"), hj):
            f.write(chunk); h.update(chunk)
        for k in names:
            b = np.asarray(jax.device_get(params[k]), np.float32).astype(ml_dtypes.bfloat16).tobytes()
            f.write(b); h.update(b)
    os.replace(tmp, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest(), "tensors": len(names)}

def finish_run(run_dir, mc, hp, params, m, v, t, export: bool = True) -> dict:
    """Run end: full resumable f32 checkpoint (stays on pod) + optional bf16 weights-only export."""
    run_dir = Path(run_dir); step = int(t)
    sha = save(run_dir / f"step-{step:06d}", mc, hp, params, m, v, t)
    out = {"ckpt": str(run_dir / f"step-{step:06d}"), "model_sha256": sha}
    if export:
        out["export"] = export_bf16(run_dir / "export" / f"weights-bf16-step{step:06d}.safetensors", params,
                                    {"step": step, "f32_model_sha256": sha})
        (run_dir / "export" / "EXPORT.json").write_text(json.dumps(out, indent=1))
    return out

def main(argv=None) -> int:
    """CLI data gate: `python -m accel.mgpu --check-data --data META [--clean-tsv TSV] [--no-verify-sha]`.
    Exits 2 on any clean-data problem (DataSafetyError), 0 if the clean store opens."""
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-data", action="store_true", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--clean-tsv", default=DEFAULT_CLEAN_TSV)
    ap.add_argument("--no-verify-sha", action="store_true")
    a = ap.parse_args(argv)
    hp = HP(data=a.data, clean_tsv=a.clean_tsv, verify_sha=not a.no_verify_sha)
    try:
        st = open_store(hp, 8192)
    except DataSafetyError as e:
        print(f"DATA GATE FAILED: {e}", file=sys.stderr); return 2
    print(f"data gate ok: {type(st).__name__} virtual tokens {st.split_len('train')}"
          + (f" sha256 {st.sha_seconds:.1f}s" if getattr(st, "sha_seconds", None) else ""))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
