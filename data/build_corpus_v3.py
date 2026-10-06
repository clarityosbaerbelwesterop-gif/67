"""Corpus v3: corpus v2 plus general educational English from FineWeb-Edu
(HuggingFaceFW/fineweb-edu, licence ODC-By 1.0; attribution in the manifest).

Corpus v2 is almost all code (39M of 402M tokens are prose), which caps what
Rouge 1 and Darus can learn about language. v3 adds up to --target-tokens of
FineWeb-Edu text with the same rules as v1/v2:
  * exact deduplication (sha1 of whitespace-normalised text);
  * 13-gram decontamination against HumanEval, MBPP (incl. tests) and GSM8K;
  * ~1 % validation split chosen by document hash;
  * tokenised by forge (= scp_model.bpe) with <eos> per document, sharing
    data/stores/base/tokenizer.json.

Stores:
  data/out/general-v3  = general-v2 (package docs) + FineWeb-Edu
  data/out/base-v3     = code-v2 + general-v3
The code store stays data/out/code-v2.

Needs network access to huggingface.co and its file CDN (cas-bridge.xethub.hf.co,
cdn-lfs.huggingface.co). Downloads resume with HTTP Range requests. pyarrow is
required (pip install pyarrow).

Usage:
  python3 data/build_corpus_v3.py [--target-tokens 600e6] [--files 1]
  python3 data/build_corpus_v3.py --parquet LOCAL.parquet --root DIR   # tests / offline
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_corpus_v2", ROOT / "data" / "build_corpus_v2.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)
bc = v2.bc

DATASET = "HuggingFaceFW/fineweb-edu"
FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]
LICENCE = "ODC-By-1.0"
ATTRIBUTION = ("FineWeb-Edu by Hugging Face (Lozhkov, Ben Allal, von Werra, Wolf; 2024), "
               "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, licensed under ODC-By 1.0")


def log(msg: str) -> None:
    print(f"[corpus-v3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url: str, dest: Path, chunk: int = 1 << 22) -> Path:
    """Resumable download (HTTP Range) to `dest`; returns the path once complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        return dest
    for attempt in range(8):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, part.open("ab" if have and r.status == 206 else "wb") as f:  # noqa: S310
                total = have + int(r.headers.get("Content-Length", 0)) if r.status == 206 else int(r.headers.get("Content-Length", 0))
                got, last = (have if r.status == 206 else 0), time.time()
                while block := r.read(chunk):
                    f.write(block)
                    got += len(block)
                    if time.time() - last > 30:
                        log(f"{dest.name}: {got / 1e6:.0f} / {total / 1e6:.0f} MB")
                        last = time.time()
            if not total or part.stat().st_size >= total:
                part.rename(dest)
                return dest
        except OSError as e:
            log(f"download interrupted ({e}); retry {attempt + 1}")
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"could not download {url}")


def banned_ngrams() -> set[int]:
    banned: set[int] = set()
    bc.get_evals()
    for f in sorted((bc.OUT / "evals").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                vals = [v for v in row.values() if isinstance(v, str)]
                vals += [x for v in row.values() if isinstance(v, list) for x in v if isinstance(x, str)]
                banned |= bc.ngrams(" ".join(vals))
    return banned


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=float, default=600e6, help="FineWeb-Edu tokens to add")
    ap.add_argument("--files", type=int, default=1, help="parquet files of sample/10BT to use (each ~2.15 GB)")
    ap.add_argument("--parquet", type=Path, action="append", default=[], help="local parquet file(s) instead of downloading")
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--root", type=Path, default=None, help="write under this directory (tests)")
    a = ap.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("pyarrow is required: pip install pyarrow")
        return 2
    out_root = a.root / "out" if a.root else v2.OUT
    cache = (a.root / "cache" if a.root else ROOT / "data" / "cache") / "v3"
    manifests = a.root / "manifests" if a.root else ROOT / "data" / "manifests" / "v3"
    v2.CACHE = cache
    for p in (cache, manifests, out_root):
        p.mkdir(parents=True, exist_ok=True)
    code_v2, general_v2 = out_root / "code-v2", out_root / "general-v2"
    if not (code_v2 / "train.meta.json").exists() or not (general_v2 / "train.meta.json").exists():
        log(f"corpus v2 is required first ({code_v2}, {general_v2}); run data/build_corpus_v2.py")
        return 2

    paths = list(a.parquet)
    if not paths:
        for name in FILES[: a.files]:
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{name}"
            log(f"downloading {url}")
            paths.append(download(url, cache / name.replace("/", "_")))

    banned = banned_ngrams()
    log(f"{len(banned):,} banned 13-grams")
    docs = {"train": cache / "fineweb-train.jsonl", "val": cache / "fineweb-val.jsonl"}
    seen, chars, stats = set(), 0, {"short": 0, "duplicate": 0, "contaminated": 0, "kept": 0}
    handles = {k: p.open("w") for k, p in docs.items()}
    target_chars = a.target_tokens * v2.CHARS_PER_TOKEN
    file_meta = []
    for path in paths:
        pf = pq.ParquetFile(path)
        file_meta.append({"file": path.name, "rows": pf.metadata.num_rows, "sha256_head": hashlib.sha256(path.read_bytes()[:1 << 20]).hexdigest()})
        for batch in pf.iter_batches(columns=["text"], batch_size=2048):
            for text in batch.column(0).to_pylist():
                text = (text or "").strip()
                if len(text) < a.min_chars:
                    stats["short"] += 1
                    continue
                h = hashlib.sha1(v2.NORM.sub(" ", text).encode()).hexdigest()
                if h in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(h)
                if bc.ngrams(text) & banned:
                    stats["contaminated"] += 1
                    continue
                split = "val" if int(h[:8], 16) % 100 == 0 else "train"
                handles[split].write(json.dumps({"t": text, "h": h}) + "\n")
                chars += len(text)
                stats["kept"] += 1
                if stats["kept"] % 50_000 == 0:
                    log(f"kept {stats['kept']:,} docs, ~{chars / v2.CHARS_PER_TOKEN / 1e6:.0f} M tokens, {stats}")
            if chars >= target_chars:
                break
        if chars >= target_chars:
            break
    for h in handles.values():
        h.close()
    log(f"FineWeb-Edu: {stats}")

    gen = out_root / "general-v3"
    gen.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, fname in (("train", "train.bin"), ("val", "train.val.bin")):
        with open(gen / fname, "wb") as out:
            with open(general_v2 / fname, "rb") as src:
                while chunk := src.read(1 << 24):
                    out.write(chunk)
            counts[split] = v2.tokenize_into(docs[split], out)
    g2 = json.loads((general_v2 / "train.meta.json").read_text())
    mix = {"general-v2": sum(g2["mix"].values()), "fineweb-edu": counts["train"][1]}
    gm = v2.finish_store(gen, g2["tokens"] + counts["train"][0], g2["val_tokens"] + counts["val"][0], mix)
    log(f"store general-v3: {gm['tokens']:,} train / {gm['val_tokens']:,} val tokens")

    base = out_root / "base-v3"
    base.mkdir(parents=True, exist_ok=True)
    for fname in ("train.bin", "train.val.bin"):
        with open(base / fname, "wb") as out:
            for d in (code_v2, gen):
                with open(d / fname, "rb") as src:
                    while chunk := src.read(1 << 24):
                        out.write(chunk)
    c2 = json.loads((code_v2 / "train.meta.json").read_text())
    bm = v2.finish_store(base, c2["tokens"] + gm["tokens"], c2["val_tokens"] + gm["val_tokens"],
                         {"code": sum(c2["mix"].values()), **mix})
    log(f"store base-v3: {bm['tokens']:,} train / {bm['val_tokens']:,} val tokens")
    (manifests / "stores.json").write_text(json.dumps({
        "source": {"dataset": DATASET, "licence": LICENCE, "attribution": ATTRIBUTION, "files": file_meta},
        "filters": stats, "stores": {"general-v3": gm, "base-v3": bm},
        "evals_decontaminated_against": ["humaneval", "mbpp", "gsm8k"]}, indent=2))
    for p in docs.values():
        p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
