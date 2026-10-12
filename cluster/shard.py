"""Build one worker's data shard on the worker itself, from pinned public packages.

Every cluster worker downloads its own share of the corpus v3 packages
(data/manifests/v3/sources.json: registry archive URL + sha256, licence already
checked when the manifest was built), so nothing private is ever shipped to a
runner. The share is deterministic: packages are sorted by task and worker i of
N takes every N-th one. Each archive must hash to the manifest's sha256 (a changed
upload is skipped, never trained on). Documents then pass the same gates as the
long-run corpus: the v2 quality rules, the generated-code marker, the v4
high-quality filter (data/hq_filter.py), exact dedup and 13-gram decontamination
against HumanEval, MBPP and GSM8K. About 1 % goes to val by content hash.

  python3 cluster/shard.py --shard 3 --of 8 --out shards/3 [--max-tokens 30e6]

Output: a forge store (train.bin, train.val.bin, train.meta.json, tokenizer.json,
manifest.json with every package used and the drop counts).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("corpus_registries", ROOT / "data" / "registries.py")
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)
v2, bc = reg.v2, reg.v2.bc
sys.path.insert(0, str(ROOT / "data"))
import hq_filter  # noqa: E402

from build_corpus_v3 import GENERATED, banned_ngrams  # noqa: E402

MANIFEST = ROOT / "data" / "manifests" / "v3" / "sources.json"
TOKENIZER = ROOT / "data" / "tokenizer.json"
FORGE = ROOT / "target" / "release" / "forge"
EOS, VOCAB = v2.EOS, v2.VOCAB
CHARS_PER_TOKEN = 2.8  # measured on the v3 mix; only used to stop downloading


def share(entries: list[dict], shard: int, of: int) -> list[dict]:
    ok = sorted((e for e in entries if e.get("status") == "ok" and e.get("url") and e.get("sha256")), key=lambda e: e["task"])
    return ok[shard::of]


def members(url: str, data: bytes):
    if url.endswith(".gem"):
        return reg.gem_members(data)
    if url.endswith((".zip", ".jar", ".whl")):
        return v2.zip_members(data)
    return v2.tar_members(data)


def tokenize(texts: list[str], forge: Path, tokenizer: Path, tmp: Path) -> array:
    buf = array("H")
    tmp.write_text("".join(json.dumps(t) + "\n" for t in texts))
    res = subprocess.run([str(forge), "tokenize", "--tokenizer", str(tokenizer), "--jsonl", str(tmp)],
                         check=True, capture_output=True, text=True).stdout
    for line in res.splitlines():
        buf.extend(json.loads(line))
        buf.append(EOS)
    tmp.unlink(missing_ok=True)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf


def build(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = share(json.loads(Path(a.manifest).read_text()), a.shard, a.of)
    banned, evals = banned_ngrams(bc.OUT / "evals", fetch_missing=not a.no_decontam) if not a.no_decontam else (set(), [])
    seen: set[str] = set()
    stats = {"sha_mismatch": 0, "unavailable": 0, "duplicate": 0, "quality": 0, "generated": 0, "hq": 0, "contaminated": 0}
    used, toks = [], {"train": 0, "val": 0}
    pending = {"train": [], "val": []}
    files = {s: open(out / ("train.bin" if s == "train" else "train.val.bin"), "wb") for s in ("train", "val")}
    chars, t0 = 0, time.time()

    def flush(split: str) -> None:
        if pending[split]:
            buf = tokenize(pending[split], Path(a.forge), Path(a.tokenizer), out / f".tok-{split}.jsonl")
            buf.tofile(files[split])
            toks[split] += len(buf)
            pending[split].clear()

    for e in todo:
        if chars / CHARS_PER_TOKEN >= a.max_tokens:
            break
        data = reg.fetch(e["url"])
        if data is None:
            stats["unavailable"] += 1
            continue
        if hashlib.sha256(data).hexdigest() != e["sha256"]:
            stats["sha_mismatch"] += 1
            continue
        kept = 0
        for path, kind, text in reg.extract(members(e["url"], data)):
            text = text.strip()
            h = hashlib.sha1(v2.NORM.sub(" ", text).encode()).hexdigest()
            if h in seen:
                stats["duplicate"] += 1
                continue
            seen.add(h)
            if not text or not v2.quality(text, kind):
                stats["quality"] += 1
                continue
            if kind == "code" and GENERATED.search(text[:2000]):
                stats["generated"] += 1
                continue
            if not hq_filter.keep(text, kind)[0]:
                stats["hq"] += 1
                continue
            if banned and bc.ngrams(text) & banned:
                stats["contaminated"] += 1
                continue
            split = "val" if int(h[:8], 16) % 100 == 0 else "train"
            pending[split].append(text)
            chars += len(text)
            kept += 1
            if len(pending[split]) >= 2000:
                flush(split)
        used.append({"task": e["task"], "version": e.get("version"), "sha256": e["sha256"], "license": e.get("license"),
                     "attribution": e.get("attribution"), "kept": kept})
    for s in ("train", "val"):
        flush(s)
        files[s].close()
    (out / "tokenizer.json").write_bytes(Path(a.tokenizer).read_bytes())
    meta = {"tokens": toks["train"], "val_tokens": toks["val"], "dtype": "uint16", "vocab_size": VOCAB, "bin": "train.bin",
            "val_bin": "train.val.bin", "mix": {"packages": len(used)}, "mix_fractions": {"packages": 1.0}}
    (out / "train.meta.json").write_text(json.dumps(meta, indent=2))
    man = {"shard": a.shard, "of": a.of, **meta, "seconds": round(time.time() - t0, 1), "drops": stats, "evals": evals,
           "pipeline": "cluster/shard.py: pinned corpus v3 packages (sha256), v2 quality + generated marker + v4 hq_filter, "
                       "exact dedup, 13-gram decontamination, hash val split, forge tokenize, eos per document",
           "packages": used}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=float, default=30e6, help="stop downloading after about this many tokens")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--forge", default=str(FORGE))
    ap.add_argument("--no-decontam", action="store_true", help="tests only")
    a = ap.parse_args(argv)
    if not 0 <= a.shard < a.of:
        ap.error("need 0 <= shard < of")
    m = build(a)
    print(json.dumps({k: m[k] for k in ("shard", "of", "tokens", "val_tokens", "seconds", "drops")} | {"packages": len(m["packages"])}))
    return 0 if m["tokens"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
