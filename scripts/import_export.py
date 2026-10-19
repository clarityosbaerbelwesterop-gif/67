"""Turn a bf16 weights-only export of the accel trainer (accel/mgpu.py export_bf16) into a forge
checkpoint directory, so `forge generate/eval/logits` and scripts/humaneval.py can run it on CPU.

forge reads <dir>/model.safetensors (F32 or BF16 since ckpt.rs widens BF16 exactly) and the model
config from <dir>/state.json. The config is inferred from the tensor shapes (head_dim 64 as in all
forge models); max_seq_len and rope_theta are not stored in weights and come from the flags.

The export file is hard-linked when possible (no extra disk), else copied. Nothing is deleted.

Usage: python3 scripts/import_export.py EXPORT.safetensors OUT_DIR [--max-seq-len 2048] [--rope-theta 10000]
       [--expect-sha256 HEX]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

HEAD_DIM = 64


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def infer(h: dict, max_seq_len: int, rope_theta: float) -> dict:
    t = {k: v for k, v in h.items() if k != "__metadata__"}
    vocab, dim = t["tok_emb.weight"]["shape"]
    layers = sorted({int(m.group(1)) for k in t if (m := re.match(r"blocks\.(\d+)\.", k))})
    if layers != list(range(len(layers))):
        raise SystemExit(f"non-contiguous blocks: {layers[:10]}")
    q, kv, ffn = (t[f"blocks.0.{n}.weight"]["shape"][0] for n in ("attn.wq", "attn.wk", "ffn.w1"))
    expected = {"tok_emb.weight", "norm.weight"} | {
        f"blocks.{i}.{n}.weight" for i in layers for n in
        ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "ffn.w1", "ffn.w2", "ffn.w3", "attn_norm", "ffn_norm")}
    if set(t) != expected:
        raise SystemExit(f"tensor names differ from forge: missing {sorted(expected - set(t))[:5]}, "
                         f"extra {sorted(set(t) - expected)[:5]}")
    if q % HEAD_DIM or kv % HEAD_DIM:
        raise SystemExit(f"q {q} / kv {kv} not multiples of head_dim {HEAD_DIM}")
    return {"vocab_size": vocab, "dim": dim, "n_layers": len(layers), "n_heads": q // HEAD_DIM,
            "n_kv_heads": kv // HEAD_DIM, "ffn_hidden": ffn, "max_seq_len": max_seq_len,
            "norm_eps": 1e-5, "rope_theta": rope_theta}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("out")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    ap.add_argument("--expect-sha256", default=None)
    a = ap.parse_args()
    src, out = Path(a.export), Path(a.out)
    h = header(src)
    model = infer(h, a.max_seq_len, a.rope_theta)
    sha = sha256(src)
    if a.expect_sha256 and sha != a.expect_sha256:
        raise SystemExit(f"sha256 mismatch: {sha} != {a.expect_sha256}")
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "model.safetensors"
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    params = sum(int.__mul__(*v["shape"]) if len(v["shape"]) == 2 else v["shape"][0]
                 for k, v in h.items() if k != "__metadata__")
    meta = h.get("__metadata__", {})
    state = {"config": {"model": model}, "model_sha256": sha, "params": params, "imported_from": str(src),
             "export_metadata": meta, "step": int(meta["step"]) if str(meta.get("step", "")).isdigit() else None}
    (out / "state.json").write_text(json.dumps(state, indent=1) + "\n")
    print(json.dumps({"out": str(out), "params": params, "model": model, "sha256": sha}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
