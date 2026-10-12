"""Corpus v4 = corpus v3, high-quality only: content filter + exact and near-duplicate removal.

  code-v4     the code-v3 documents that pass data/hq_filter.py (code rules)
  general-v4  the general-v3 documents that pass (prose rules)
  base-v4     code-v4 + general-v4

Offline: every document is decoded from the v3 token stores (byte-level BPE,
lossless) and the token span of every kept document is copied unchanged (no
re-tokenisation). Near duplicates are found with one-permutation MinHash over
word 5-shingles (64 bins, LSH 8 bands x 8 rows, a pair at Jaccard 0.8 becomes a
candidate with p ~ 0.97, one at 0.5 with p ~ 0.03) and confirmed when >= 80 % of
the bins agree; the first document of a cluster stays. The val splits are copied
byte for byte from v3, so gate and champion val losses stay comparable across the
switch. Stores are staged in data/out/.v4-staging and moved into place with
base-v4 last (its train.meta.json is what scripts/longrun.py looks for).

  python3 data/build_corpus_v4.py [--dry-run --sample 2000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
import hq_filter  # noqa: E402

OUT = ROOT / "data" / "out"
TOKENIZER = ROOT / "data" / "stores" / "base" / "tokenizer.json"
EOS = 8190
BINS, BANDS, ROWS, CONFIRM = 64, 8, 8, 0.8
MAX_WORDS = 20_000  # shingles of the first 20k words per document
WORD = re.compile(r"\w+")
P61 = (1 << 61) - 1


def log(msg: str) -> None:
    print(f"[v4 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vocab_bytes(path: Path) -> list:
    data = json.loads(path.read_text())
    vocab = {i: bytes([i]) for i in range(256)}
    for a, b, idx in sorted(data["merges"], key=lambda m: m[2]):
        vocab[idx] = vocab[a] + vocab[b]
    return [vocab.get(i, b"") for i in range(max(vocab) + 1)]


def doc_spans(tokens: np.ndarray) -> np.ndarray:
    """(start, end) of every document; end includes its eos."""
    ends = np.flatnonzero(tokens == EOS) + 1
    starts = np.concatenate(([0], ends[:-1]))
    return np.stack([starts, ends], axis=1)


def sketch(text: str) -> np.ndarray | None:
    """One-permutation MinHash: 64 bins of the minimum 61-bit hash of word 5-shingles."""
    words = WORD.findall(text.lower())[:MAX_WORDS]
    if len(words) < 5:
        return None
    wh = np.fromiter((int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little") & P61 for w in words),
                     np.uint64, len(words))
    # shingle hash = mix of 5 consecutive word hashes (wrapping uint64 arithmetic)
    sh = wh[:-4].copy()
    for k in range(1, 5):
        sh = sh * np.uint64(0x9E3779B97F4A7C15) ^ wh[k:len(wh) - 4 + k]
    sh ^= sh >> np.uint64(29)
    sh *= np.uint64(0xBF58476D1CE4E5B9)
    sh ^= sh >> np.uint64(32)
    out = np.full(BINS, np.iinfo(np.uint64).max, np.uint64)
    np.minimum.at(out, (sh % np.uint64(BINS)).astype(np.int64), sh // np.uint64(BINS))
    return out


class NearDup:
    def __init__(self):
        self.bands = [dict() for _ in range(BANDS)]
        self.sketches: list[np.ndarray] = []

    def seen(self, s: np.ndarray) -> bool:
        """True if s matches a kept sketch on >= CONFIRM of the bins; else s is added."""
        keys = [s[b * ROWS:(b + 1) * ROWS].tobytes() for b in range(BANDS)]
        cands = {i for b, k in enumerate(keys) for i in self.bands[b].get(k, ())}
        if any(np.mean(self.sketches[i] == s) >= CONFIRM for i in cands):
            return True
        i = len(self.sketches)
        self.sketches.append(s)
        for b, k in enumerate(keys):
            self.bands[b].setdefault(k, []).append(i)
        return False


def filter_store(kind: str, src: Path, vocab: list, stats: Counter, near: NearDup, exact: set,
                 dst: Path | None = None, sample: int = 0, seed: int = 7) -> dict:
    tokens = np.memmap(src / "train.bin", np.uint16, "r")
    spans = doc_spans(tokens)
    if sample:
        rng = np.random.default_rng(seed)
        spans = spans[np.sort(rng.choice(len(spans), min(sample, len(spans)), replace=False))]
    out = open(dst / "train.bin", "wb") if dst else None
    kept_tok = kept_docs = 0
    for n, (a, b) in enumerate(spans.tolist()):
        ids = tokens[a:b - 1].tolist()
        text = b"".join(map(vocab.__getitem__, ids)).decode("utf-8", "replace")
        stats[f"{kind}:docs"] += 1
        stats[f"{kind}:tokens"] += b - a
        h = hashlib.sha1(" ".join(text.split()).encode()).digest()
        ok, why = (False, "duplicate") if h in exact else hq_filter.keep(text, kind)
        if ok:
            s = sketch(text)
            if s is not None and near.seen(s):
                ok, why = False, "near-duplicate"
        if not ok:
            stats[f"{kind}:drop:{why}"] += 1
            stats[f"{kind}:drop_tokens:{why}"] += b - a
            continue
        exact.add(h)
        kept_docs += 1
        kept_tok += b - a
        if out:
            out.write(np.asarray(tokens[a:b]).tobytes())
        if n and n % 20000 == 0:
            log(f"{kind}: {n:,}/{len(spans):,} docs, kept {kept_docs:,}")
    if out:
        out.close()
    return {"docs": kept_docs, "tokens": kept_tok}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="v3")
    ap.add_argument("--dry-run", action="store_true", help="filter a sample and print the drop shares only")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--min-free-gb", type=float, default=5.5, help="free disk that must remain after the build")
    a = ap.parse_args(argv)
    os.nice(19)
    vocab = vocab_bytes(TOKENIZER)
    src = {k: OUT / f"{k}-{a.src}" for k in ("code", "general")}
    stats, near, exact = Counter(), NearDup(), set()
    if a.dry_run:
        for kind in ("code", "general"):
            filter_store(kind, src[kind], vocab, stats, near, exact, sample=a.sample)
        report = {k: v for k, v in sorted(stats.items())}
        for kind in ("code", "general"):
            d, t = stats[f"{kind}:docs"], stats[f"{kind}:tokens"]
            dropped = sum(v for k, v in stats.items() if k.startswith(f"{kind}:drop_tokens:"))
            report[f"{kind}:dropped_token_share"] = round(dropped / max(1, t), 4)
            report[f"{kind}:dropped_doc_share"] = round(sum(v for k, v in stats.items() if k.startswith(f"{kind}:drop:")) / max(1, d), 4)
        print(json.dumps(report, indent=2))
        return 0
    if (OUT / "base-v4" / "train.meta.json").exists():
        log("base-v4 exists; nothing to do")
        return 0
    need = sum((src[k] / "train.bin").stat().st_size for k in src) * 2
    free = shutil.disk_usage(OUT).free
    if free - need < a.min_free_gb * 1e9:
        log(f"refusing: {free / 1e9:.1f} GB free, the build needs up to {need / 1e9:.1f} GB and {a.min_free_gb} GB must remain")
        return 2
    sys.path.insert(0, str(ROOT / "data"))
    import build_corpus_v2 as v2
    stage = OUT / ".v4-staging"
    shutil.rmtree(stage, ignore_errors=True)
    stores = {}
    for kind in ("code", "general"):
        d = stage / f"{kind}-v4"
        d.mkdir(parents=True)
        res = filter_store(kind, src[kind], vocab, stats, near, exact, dst=d)
        shutil.copyfile(src[kind] / "train.val.bin", d / "train.val.bin")
        m3 = json.loads((src[kind] / "train.meta.json").read_text())
        stores[kind] = v2.finish_store(d, res["tokens"], m3["val_tokens"], {f"{kind}-v4": res["docs"]})
        log(f"{kind}-v4: {res['tokens']:,} tokens, {res['docs']:,} docs")
    d = stage / "base-v4"
    d.mkdir()
    for fname in ("train.bin", "train.val.bin"):
        with open(d / fname, "wb") as f:
            for kind in ("code", "general"):
                with open(stage / f"{kind}-v4" / fname, "rb") as g:
                    shutil.copyfileobj(g, f, 1 << 24)
    c, g = stores["code"], stores["general"]
    v2.finish_store(d, c["tokens"] + g["tokens"], c["val_tokens"] + g["val_tokens"],
                    {"code": sum(c["mix"].values()), "general": sum(g["mix"].values())})
    manifests = ROOT / "data" / "manifests" / "v4"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "filters.json").write_text(json.dumps({"source": a.src, "stats": dict(sorted(stats.items())),
                                                        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    for name in ("code-v4", "general-v4", "base-v4"):  # base-v4 last
        (stage / name).rename(OUT / name)
    shutil.rmtree(stage, ignore_errors=True)
    log(f"done: base-v4 {c['tokens'] + g['tokens']:,} train tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
