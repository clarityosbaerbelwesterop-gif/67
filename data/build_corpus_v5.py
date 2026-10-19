"""Corpus v5 add-on: new licensed package source code and docs, beyond everything in v2 and v3.

The add-on holds only NEW documents, so it can be mixed into base-v4 on the training machine
without rebuilding v2-v4:

  code-v5add      source files of packages no earlier corpus used (PyPI, npm, RubyGems, Hackage,
                  Maven Central, Go, crates.io), next in the registries' popularity lists
  general-v5add   their prose docs (README/docs markdown, rst, ...)

Same rules as corpus v3 (this file reuses its code): licence allow-list with copyleft deny-list,
per-package provenance in data/manifests/v5/sources.json, quality filters, generated-code filter,
exact dedup (sha1 of whitespace-normalised text) against every v2+v3 document
(data/out/base-v3/docs.sha1) and within v5, 13-gram decontamination against HumanEval, MBPP
(incl. test_list) and GSM8K, ~1 % val split by document hash, forge tokenizer, <eos> after
every document. No Hugging Face, no outputs of closed models.

Resumable: fetched documents go to data/cache/v5/docs/*.jsonl, finished packages to
data/cache/v5/done.jsonl. The cache is kept unless --drop-cache.

  python3 data/build_corpus_v5.py --target-tokens 300e6 [--registries pypi,npm] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("corpus_v3", ROOT / "data" / "build_corpus_v3.py")
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
reg, v2, bc = v3.reg, v3.v2, v3.bc

OUT = ROOT / "data" / "out"
CACHE = ROOT / "data" / "cache" / "v5"
MANIFESTS = ROOT / "data" / "manifests" / "v5"
EVALS = ROOT / "data" / "out" / "evals"
PIPELINE = "corpus v5 add-on (data/build_corpus_v5.py, reusing data/build_corpus_v3.py)"


def log(msg: str) -> None:
    print(f"[v5 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def earlier_tasks() -> set[str]:
    tasks = set()
    for v in ("v2", "v3"):
        p = ROOT / "data" / "manifests" / v / "sources.json"
        if p.exists():
            tasks |= {r["task"] for r in json.loads(p.read_text()) if "task" in r}
    return tasks


def earlier_hashes() -> set[str]:
    raw = (OUT / "base-v3" / "docs.sha1").read_bytes()
    return {raw[i:i + 20].hex() for i in range(0, len(raw), 20)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def write_stores(docs: Path, counts: dict) -> dict:
    stores = {}
    for kind in ("code", "general"):
        name = f"{kind}-v5add"
        stage, dest = OUT / f".{name}-staging", OUT / name
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        n = {}
        for split, fname in (("train", "train.bin"), ("val", "train.val.bin")):
            p = docs / f"{kind}-{split}.jsonl"
            with open(stage / fname, "wb") as f:
                n[split] = v2.tokenize_into(p, f) if p.exists() and p.stat().st_size else (0, 0)
        mix = {g: c[f"{kind}-train"] for g, c in sorted(counts.items()) if c.get(f"{kind}-train")}
        m = v2.finish_store(stage, n["train"][0], n["val"][0], mix)
        m.update(pipeline=PIPELINE, docs=n["train"][1] + n["val"][1],
                 sha256={f: sha256(stage / f) for f in ("train.bin", "train.val.bin")})
        (stage / "manifest.json").write_text(json.dumps(m, indent=2))
        if dest.exists():
            raise SystemExit(f"{dest} exists; move it away first (nothing is overwritten)")
        stage.rename(dest)
        stores[name] = m
        log(f"store {name}: {n['train'][0]:,} train / {n['val'][0]:,} val tokens, {m['docs']:,} documents")
    return stores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-tokens", type=float, default=300e6)
    ap.add_argument("--max-disk-gb", type=float, default=3.5, help="cap on cache + projected add-on stores")
    ap.add_argument("--registries", type=lambda s: [x for x in s.split(",") if x], default=list(v3.REGISTRIES))
    ap.add_argument("--limit", type=int, default=0, help="at most this many packages per registry (smoke runs)")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--drop-cache", action="store_true", help="delete data/cache/v5/docs after the stores are written")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    done_before = earlier_tasks()
    plan_args = SimpleNamespace(packages=None, registries=a.registries, limit=a.limit)
    if a.dry_run:
        tasks = v3.plan(plan_args, done_before)
        print(json.dumps({"planned": len(tasks), "by_registry": {r: sum(t[0] == r for t in tasks) for r in a.registries},
                          "first": [f"{r}:{n}" for r, n in tasks[:12]]}))
        return 0
    os.nice(max(0, 15 - os.nice(0)))
    docs = CACHE / "docs"
    for p in (docs, MANIFESTS):
        p.mkdir(parents=True, exist_ok=True)
    v2.CACHE = CACHE  # forge tokenize's chunk file
    seen = earlier_hashes()
    banned, eval_names = v3.banned_ngrams(EVALS, fetch_missing=False)
    col = v3.Collector(docs, banned, seen)
    log(f"{len(seen):,} v2+v3 documents for dedup, {len(col.seen):,} v5 documents cached, {len(banned):,} banned 13-grams")
    try:
        v2._MODS = bc.scp_modules()
    except Exception:  # noqa: BLE001 - SCP checkout absent: v2's simple heuristics
        v2._MODS = None
    if (CACHE / "done.jsonl").exists():
        v3.repair(CACHE / "done.jsonl")
    done = {r["task"] for r in v3.read_jsonl(CACHE / "done.jsonl") if "task" in r}
    disk = v3.Disk(CACHE, OUT, 0, a.max_disk_gb)
    with (CACHE / "done.jsonl").open("a") as donef:
        tasks = v3.plan(plan_args, done_before)
        todo = [t for t in tasks if f"{t[0]}:{t[1]}" not in done]
        log(f"{len(tasks)} packages planned beyond v2+v3, {len(tasks) - len(todo)} already done in v5")
        stop = v3.collect_packages(todo, col, donef, disk, a.target_tokens * v2.CHARS_PER_TOKEN)
    col.close()
    sources = v3.read_jsonl(CACHE / "done.jsonl")
    (MANIFESTS / "sources.json").write_text(json.dumps(sorted(sources, key=lambda s: s["task"]), indent=1))
    (MANIFESTS / "filters.json").write_text(json.dumps({"filters": col.stats, "registries": col.counts}, indent=2))
    if a.fetch_only:
        log(f"--fetch-only: {len(col.seen):,} documents cached ({stop})")
        return 0
    stores = write_stores(docs, col.counts)
    (MANIFESTS / "stores.json").write_text(json.dumps({
        "stores": stores, "stopped": stop, "registries": col.counts, "filters": col.stats,
        "licence_policy": {"allow": bc.ALLOW.pattern, "deny": bc.DENY.pattern, "deny_also": reg.COPYLEFT.pattern},
        "dedup": {"method": "sha1 of whitespace-normalised text", "against": "every v2+v3 document (base-v3/docs.sha1)",
                  "earlier_documents": len(seen), "v5_documents": len(col.seen)},
        "evals_decontaminated_against": eval_names}, indent=2))
    if a.drop_cache:
        shutil.rmtree(docs, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
