"""Function-preserving growth of a forge checkpoint: wider residual stream (more heads at the same head_dim), wider
FFN and more layers. The grown model computes the same logits as the source, up to float rounding, and keeps
learning from there. This is how Quasnir goes from the 1B to the 10B point without starting over.

How every part keeps the function:
- **Residual stream d -> D.** The new dimensions start at exactly zero: every weight that writes into them (token
  embedding, which is also the tied output head, attn.wo and ffn.w2 rows) is zero. RMSNorm then divides by an RMS
  that is sqrt(d/D) of the old one. The old norm weights are multiplied by sqrt(d/D) and norm_eps by d/D, which
  gives the old normalised values again.
- **Heads h -> H (head_dim fixed).** Query heads stay grouped onto KV heads as before (H/KV = h/kv). The new heads
  read with random weights (std 0.02, forge's init) but write through zero attn.wo columns.
- **FFN f -> F.** The new units read with random w1/w3 rows and write through zero w2 columns.
- **Layers L -> L'.** Layer j of the grown model is a copy of source layer j*L//L'. Every repeat after the first
  gets zero attn.wo and ffn.w2, so its residual branch adds nothing.
Weights that read the new residual dimensions are random: they multiply zeros now and give the gradient a path
into the new dimensions. The optimizer starts fresh (accel.run_s2 --init and forge init_from read only
model.safetensors).

Output: OUT/model.safetensors (F32; tensors written one at a time, so memory stays near the source size) and
OUT/state.json with the grown config and the lineage (parent_sha256, grown_from).

Usage: python3 scripts/grow.py SRC_CKPT OUT_DIR --dim 4096 --ffn-hidden 14336 --n-layers 44
       [--max-seq-len N] [--seed 1] [--dry-run]
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rebase as R  # noqa: E402

STD = 0.02


def load(src: Path) -> tuple[dict, dict, str]:
    state = json.loads((src / "state.json").read_text())
    ts, _, sha = R.read_safetensors(src / "model.safetensors")
    w = {}
    for name, t in ts.items():
        if t["dtype"] == "F32":
            a = np.frombuffer(t["data"], "<f4")
        elif t["dtype"] == "BF16":
            a = (np.frombuffer(t["data"], "<u2").astype(np.uint32) << 16).view("<f4")
        else:
            raise SystemExit(f"{name}: dtype {t['dtype']} not supported (F32, BF16)")
        w[name] = a.reshape(t["shape"]).astype(np.float32)
    return state, w, sha


def target_config(src: dict, dim: int, ffn: int, layers: int, max_seq_len: int | None) -> dict:
    hd = src["dim"] // src["n_heads"]
    rep = src["n_heads"] // src["n_kv_heads"]
    if dim % hd or (dim // hd) % rep:
        raise SystemExit(f"dim {dim} must be a multiple of head_dim {hd} x group {rep} = {hd * rep}")
    if dim < src["dim"] or ffn < src["ffn_hidden"] or layers < src["n_layers"]:
        raise SystemExit("growth only: dim, ffn_hidden and n_layers must not shrink")
    eps = float(np.float32(np.float32(src["norm_eps"]) * np.float32(src["dim"]) / np.float32(dim)))
    return {**src, "dim": dim, "n_heads": dim // hd, "n_kv_heads": dim // hd // rep, "ffn_hidden": ffn,
            "n_layers": layers, "norm_eps": eps, "max_seq_len": max_seq_len or src["max_seq_len"]}


def num_params(c: dict) -> int:
    hd = c["dim"] // c["n_heads"]
    per = (2 * c["dim"] + c["dim"] * hd * (2 * c["n_heads"] + 2 * c["n_kv_heads"]) + 3 * c["dim"] * c["ffn_hidden"])
    return c["vocab_size"] * c["dim"] + c["n_layers"] * per + c["dim"]


def grown_tensors(w: dict, s: dict, t: dict, seed: int):
    """(name, f32 array) in name order, one at a time."""
    d, D, f, F = s["dim"], t["dim"], s["ffn_hidden"], t["ffn_hidden"]
    hd = d // s["n_heads"]
    q_old, q_new = s["n_heads"] * hd, t["n_heads"] * hd
    kv_old, kv_new = s["n_kv_heads"] * hd, t["n_kv_heads"] * hd
    scale = np.float32(math.sqrt(d / D))
    src_of = [j * s["n_layers"] // t["n_layers"] for j in range(t["n_layers"])]
    rng = np.random.default_rng(seed)

    def embed(a, rows, cols, random_rows=False, random_cols=False):
        """a in the top-left corner of a [rows, cols] matrix; the rest random (reads) or zero (writes)."""
        out = np.zeros((rows, cols), np.float32)
        if random_cols:
            out[: a.shape[0], a.shape[1]:] = rng.standard_normal((a.shape[0], cols - a.shape[1]), np.float32) * STD
        if random_rows:
            out[a.shape[0]:, :] = rng.standard_normal((rows - a.shape[0], cols), np.float32) * STD
        out[: a.shape[0], : a.shape[1]] = a
        return out

    def norm(a):
        out = np.full(D, np.float32(a.mean()) * scale, np.float32)
        out[:d] = a * scale
        return out

    names = {"tok_emb.weight": None, "norm.weight": None}
    for j in range(t["n_layers"]):
        for k in ("attn_norm", "attn.wq", "attn.wk", "attn.wv", "attn.wo", "ffn_norm", "ffn.w1", "ffn.w3", "ffn.w2"):
            names[f"blocks.{j}.{k}.weight"] = (j, k)
    for name in sorted(names):
        if name == "tok_emb.weight":
            yield name, embed(w[name], w[name].shape[0], D)
            continue
        if name == "norm.weight":
            yield name, norm(w[name])
            continue
        j, k = names[name]
        i = src_of[j]
        repeat = j > 0 and src_of[j - 1] == i
        a = w[f"blocks.{i}.{k}.weight"]
        if k in ("attn_norm", "ffn_norm"):
            yield name, norm(a)
        elif k == "attn.wq":
            yield name, embed(a, q_new, D, random_rows=True, random_cols=True)
        elif k in ("attn.wk", "attn.wv"):
            yield name, embed(a, kv_new, D, random_rows=True, random_cols=True)
        elif k in ("ffn.w1", "ffn.w3"):
            yield name, embed(a, F, D, random_rows=True, random_cols=True)
        else:  # attn.wo [D, H*hd], ffn.w2 [D, F]: writes; zero outside the old block, all zero in a repeat
            out = np.zeros((D, q_new if k == "attn.wo" else F), np.float32)
            if not repeat:
                out[:d, : a.shape[1]] = a
            yield name, out


def write_streaming(path: Path, shapes: dict, tensors, meta: dict) -> str:
    """safetensors as rebase.write_safetensors writes it (F32, name order, compact sorted header padded to 8),
    with the tensors produced one at a time."""
    header, off = {}, 0
    for name in sorted(shapes):
        n = 4 * int(np.prod(shapes[name], dtype=np.int64))
        header[name] = {"dtype": "F32", "shape": list(shapes[name]), "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = meta
    hb = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    hb += b" " * (-len(hb) % 8)
    h = hashlib.sha256()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        for chunk in (len(hb).to_bytes(8, "little"), hb):
            fh.write(chunk)
            h.update(chunk)
        for name, a in tensors:
            if list(a.shape) != header[name]["shape"]:
                raise SystemExit(f"{name}: shape {list(a.shape)} != {header[name]['shape']}")
            b = np.ascontiguousarray(a, "<f4").tobytes()
            fh.write(b)
            h.update(b)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--ffn-hidden", type=int, required=True)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="print the grown config and size only")
    a = ap.parse_args(argv)
    state = json.loads((a.src / "state.json").read_text())
    s = state["config"]["model"]
    t = target_config(s, a.dim, a.ffn_hidden, a.n_layers, a.max_seq_len)
    info = {"source": s, "grown": t, "source_params": num_params(s), "grown_params": num_params(t),
            "grown_bytes_f32": 4 * num_params(t)}
    if a.dry_run:
        print(json.dumps(info, indent=1))
        return 0
    if (a.out / "model.safetensors").exists():
        raise SystemExit(f"{a.out} already holds a model; nothing is overwritten")
    state, w, sha = load(a.src)
    a.out.mkdir(parents=True, exist_ok=True)
    shapes = {}
    hd = t["dim"] // t["n_heads"]
    shapes["tok_emb.weight"] = (t["vocab_size"], t["dim"])
    shapes["norm.weight"] = (t["dim"],)
    for j in range(t["n_layers"]):
        b = f"blocks.{j}"
        shapes.update({f"{b}.attn_norm.weight": (t["dim"],), f"{b}.attn.wq.weight": (t["n_heads"] * hd, t["dim"]),
                       f"{b}.attn.wk.weight": (t["n_kv_heads"] * hd, t["dim"]),
                       f"{b}.attn.wv.weight": (t["n_kv_heads"] * hd, t["dim"]),
                       f"{b}.attn.wo.weight": (t["dim"], t["n_heads"] * hd), f"{b}.ffn_norm.weight": (t["dim"],),
                       f"{b}.ffn.w1.weight": (t["ffn_hidden"], t["dim"]), f"{b}.ffn.w3.weight": (t["ffn_hidden"], t["dim"]),
                       f"{b}.ffn.w2.weight": (t["dim"], t["ffn_hidden"])})
    missing = sorted({k for k in w} - {f"blocks.{i}.{n}" for i in range(s["n_layers"]) for n in (
        "attn_norm.weight", "attn.wq.weight", "attn.wk.weight", "attn.wv.weight", "attn.wo.weight", "ffn_norm.weight",
        "ffn.w1.weight", "ffn.w3.weight", "ffn.w2.weight")} - {"tok_emb.weight", "norm.weight"})
    if missing:
        raise SystemExit(f"unknown tensors in the source: {missing[:5]}")
    new_sha = write_streaming(a.out / "model.safetensors", shapes, grown_tensors(w, s, t, a.seed),
                              {"format": "pt", "step": "0"})
    out_state = {"config": {**state.get("config", {}), "model": t}, "model_sha256": new_sha, "parent_sha256": sha,
                 "step": 0, "tokens": 0,
                 "grown_from": {"sha256": sha, "model": s, "seed": a.seed, "tool": "scripts/grow.py",
                                "source_step": state.get("step"), "source_tokens": state.get("tokens")}}
    (a.out / "state.json").write_text(json.dumps(out_state, indent=1, sort_keys=True) + "\n")
    print(json.dumps({**info, "model_sha256": new_sha, "parent_sha256": sha}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
