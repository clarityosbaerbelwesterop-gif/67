"""Corpus v3: corpus v2 plus more, and more diverse, licensed data, without
depending on Hugging Face.

  code-v3     code-v2 + the source code of new packages: Java/Kotlin/Scala (Maven
              Central sources jars), Ruby (RubyGems), Haskell (Hackage), and more
              PyPI, npm, crates.io and Go packages than v2 used
  general-v3  general-v2 + the prose docs of those packages (README/docs markdown,
              rst, rdoc, asciidoc, javadoc package.html) + FineWeb-Edu
              (HuggingFaceFW/fineweb-edu, ODC-By 1.0) only when huggingface.co is
              reachable at build time (skipped with a log line otherwise)
  base-v3     code-v3 + general-v3
How packages are chosen, where licences are read and the request etiquette:
data/registries.py and data/manifests/v3/README.md.

Same rules as v1/v2:
  * licence allow-list (v2's; the deny-list also catches spelled-out copyleft
    names); per-package provenance, licence and attribution in
    data/manifests/v3/sources.json;
  * v2's quality filters (SCP corpora.is_quality_code, data_pipeline.is_quality),
    plus: code files whose header marks them as generated (protoc, bindgen,
    "Code generated ... DO NOT EDIT", @generated) are dropped;
  * exact dedup (sha1 of whitespace-normalised text) across all sources,
    including every v2 document (data/cache/v2/docs);
  * 13-gram decontamination against HumanEval, MBPP (incl. test_list) and GSM8K;
  * ~1 % validation split chosen by document hash;
  * forge tokenize (= scp_model.bpe), <eos> 8190 after every document; the
    stores share data/stores/base/tokenizer.json.

One nice(15) process; registry requests are sequential and rate-limited.
Resumable: every package is fetched, filtered and appended to
data/cache/v3/docs/*.jsonl once and recorded in data/cache/v3/done.jsonl;
FineWeb parquet downloads resume with HTTP Range. The stores are written to
data/out/.v3-staging and moved into place at the end (base-v3 last, so
scripts/longrun.py never sees a half-built v3), then the caches are deleted
(done.jsonl and built.json stay). base-v3/docs.sha1 keeps the sha1 of every
v2 + v3 document for later dedup. --target-tokens bounds the new tokens
(FineWeb-Edu gets --fineweb-share of them when reachable), --max-disk-gb the
build's footprint (cache + the three stores).

Usage:
  python3 data/build_corpus_v3.py [--target-tokens 400e6] [--max-disk-gb 6]
  python3 data/build_corpus_v3.py --dry-run              # plan only
  python3 data/build_corpus_v3.py --limit 2 --root DIR   # smoke run: 2 packages per registry
  python3 data/build_corpus_v3.py --tokenize-only        # rebuild the stores from cached docs
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("corpus_registries", ROOT / "data" / "registries.py")
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)
v2 = reg.v2
bc = v2.bc
log = reg.log

DATASET = "HuggingFaceFW/fineweb-edu"
FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]
LICENCE = "ODC-By-1.0"
ATTRIBUTION = ("FineWeb-Edu by Hugging Face (Lozhkov, Ben Allal, von Werra, Wolf; 2024), "
               "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, licensed under ODC-By 1.0")
HF_PROBE = f"https://huggingface.co/api/datasets/{DATASET}"
PARQUET_BYTES = 2.3e9  # one sample/10BT file, reserved before it is downloaded
REGISTRIES = ("maven", "rubygems", "hackage", "pypi", "npm", "go", "crates")
STORES = ("code-v3", "general-v3", "base-v3")
MARGIN = 1e9  # free disk space the build never uses
DISK_CHARS_PER_TOKEN = 2.5  # conservative for the disk guard (the v3 smoke mix measured ~2.8; v2.CHARS_PER_TOKEN is 3.3)
GENERATED = re.compile(r"@generated|\bdo not edit\b|\bauto-?generated\b|\b(?:code|file|bindings|source) (?:is |was )?generated (?:by|from)\b",
                       re.I)
PIPELINE = ("data/build_corpus_v3.py: v2 + Maven/RubyGems/Hackage + more PyPI/npm/crates/Go packages (+ FineWeb-Edu when "
            "reachable); licence allow-list, SCP quality filters, exact dedup across v2+v3, 13-gram decontamination (HumanEval, "
            "MBPP, GSM8K), hash-based ~1% val split, forge tokenize (= scp_model.bpe), eos per document")


def download(url: str, dest: Path, chunk: int = 1 << 22) -> Path:
    """Resumable download (HTTP Range) to `dest`; returns the path once complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        return dest
    for attempt in range(8):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": reg.USER_AGENT} | ({"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r, \
                    part.open("ab" if have and r.status == 206 else "wb") as f:  # noqa: S310
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


def banned_ngrams(evals: Path, fetch_missing: bool) -> tuple[set[int], list[str]]:
    """13-grams of every string (and list of strings, e.g. MBPP test_list) of the eval sets."""
    if fetch_missing:
        bc.get_evals()
    banned: set[int] = set()
    names = []
    for f in sorted(evals.glob("*.jsonl")):
        names.append(f.stem)
        for line in f.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                vals = [v for v in row.values() if isinstance(v, str)]
                vals += [x for v in row.values() if isinstance(v, list) for x in v if isinstance(x, str)]
                banned |= bc.ngrams(" ".join(vals))
    missing = {"humaneval", "mbpp", "gsm8k"} - set(names)
    if missing:
        log(f"WARNING: eval sets {sorted(missing)} missing in {evals}; not decontaminated against them")
    return banned, names


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a partial last line of an interrupted run
    return rows


def doc_hashes(path: Path):
    """The "h" of every document of a docs jsonl (written as {"h": ..., ...}, so usually without parsing)."""
    with path.open() as fh:
        for line in fh:
            if line.startswith('{"h": "') and line[47:48] == '"':
                yield line[7:47]
            elif line.strip():
                try:
                    yield json.loads(line)["h"]
                except (ValueError, KeyError):
                    continue


def repair(path: Path) -> None:
    """Cut a partial last line left by an interrupted append, so appends stay line-aligned."""
    with path.open("rb+") as f:
        size = f.seek(0, 2)
        pos = size
        while pos > 0:
            step = min(1 << 16, pos)
            f.seek(pos - step)
            block = f.read(step)
            if pos == size and block.endswith(b"\n"):
                return
            i = block.rfind(b"\n")
            if i >= 0:
                f.truncate(pos - step + i + 1)
                return
            pos -= step
        f.truncate(0)


class Collector:
    """Dedup, quality and decontamination of new documents; hash split; append to the docs cache."""

    def __init__(self, docs: Path, banned: set[int], seen_v2: set[str]):
        self.docs, self.banned, self.seen_v2 = docs, banned, seen_v2
        self.seen: set[str] = set()
        self.chars = 0
        self.counts: dict[str, dict[str, int]] = {}
        self.stats = {"duplicate_v2": 0, "duplicate": 0, "quality": 0, "generated": 0, "short": 0, "contaminated": 0}
        for kind in ("code", "general"):
            for split in ("train", "val"):
                p = docs / f"{kind}-{split}.jsonl"
                if not p.exists():
                    continue
                repair(p)
                with p.open() as fh:
                    for line in fh:
                        d = json.loads(line)
                        self._count(d["h"], kind, split, d["s"].split(":", 1)[0], len(d["t"]))
        self.handles = {(k, s): (docs / f"{k}-{s}.jsonl").open("a") for k in ("code", "general") for s in ("train", "val")}

    def _count(self, h: str, kind: str, split: str, group: str, chars: int) -> None:
        self.seen.add(h)
        self.chars += chars
        c = self.counts.setdefault(group, {"chars": 0})
        c["chars"] += chars
        c[f"{kind}-{split}"] = c.get(f"{kind}-{split}", 0) + 1

    def group_chars(self, group: str) -> int:
        return self.counts.get(group, {}).get("chars", 0)

    def add(self, text: str, kind: str, source: str, path: str, min_chars: int = 0) -> bool:
        """Keep `text` unless it is a duplicate (of v2 or v3), low quality / too short, generated code, or contaminated."""
        text = text.strip()
        h = hashlib.sha1(v2.NORM.sub(" ", text).encode()).hexdigest()
        if h in self.seen_v2 or h in self.seen:
            self.stats["duplicate_v2" if h in self.seen_v2 else "duplicate"] += 1
            return False
        if min_chars and len(text) < min_chars:
            self.stats["short"] += 1
            return False
        if not text or not min_chars and not v2.quality(text, kind):
            self.stats["quality"] += 1
            return False
        if kind == "code" and GENERATED.search(text[:2000]):
            self.stats["generated"] += 1
            return False
        if self.banned and bc.ngrams(text) & self.banned:
            self.stats["contaminated"] += 1
            return False
        split = "val" if int(h[:8], 16) % 100 == 0 else "train"
        self.handles[(kind, split)].write(json.dumps({"h": h, "k": kind, "s": source, "p": path, "t": text}) + "\n")
        self._count(h, kind, split, source.split(":", 1)[0], len(text))
        return True

    def flush(self) -> None:
        for h in self.handles.values():
            h.flush()

    def close(self) -> None:
        for h in self.handles.values():
            h.close()


class Disk:
    """--max-disk-gb guard: cache + the projected three stores (each v2/v3 token is in a kind store and in base)."""

    def __init__(self, cache: Path, out: Path, v2_bytes: int, max_gb: float):
        self.cache, self.out, self.v2_bytes, self.max = cache, out, v2_bytes, max_gb * 1e9

    def stores(self, new_chars: float) -> float:
        return 2 * (self.v2_bytes + 2 * new_chars / DISK_CHARS_PER_TOKEN)

    def over(self, new_chars: float, extra: float = 0) -> str | None:
        """Why fetching more would break the guard, or None."""
        cache = sum(f.stat().st_size for f in self.cache.rglob("*") if f.is_file())
        need = self.stores(new_chars) + extra
        if cache + need > self.max:
            return f"projected {(cache + need) / 1e9:.2f} GB > --max-disk-gb {self.max / 1e9:g}"
        free = shutil.disk_usage(self.out).free
        if free - need < MARGIN:
            return f"only {free / 1e9:.2f} GB free for {need / 1e9:.2f} GB of stores (+{MARGIN / 1e9:g} GB margin)"
        return None


def plan(a, v2_done: set[str]) -> list[tuple[str, str]]:
    """Packages to fetch, registries interleaved so that a partial build is still diverse."""
    if a.packages:
        tasks = [tuple(x.strip().split(":", 1)) for x in a.packages.read_text().splitlines() if x.strip() and not x.startswith("#")]
        bad = [t for t in tasks if len(t) != 2 or t[0] not in reg.SOURCES]
        if bad:
            raise SystemExit(f"--packages: unknown registry in {bad[:5]} (known: {', '.join(reg.SOURCES)})")
        new = [t for t in tasks if f"{t[0]}:{t[1]}" not in v2_done]
        if len(new) < len(tasks):
            log(f"--packages: {len(tasks) - len(new)} already in corpus v2, skipped")
        return new
    lists = []
    for r in a.registries:
        skip = {t.split(":", 1)[1] for t in v2_done if t.startswith(r + ":")}
        try:
            names = reg.LISTS[r](a.limit or reg.DEFAULT_MAX[r], skip)
        except Exception as e:  # noqa: BLE001 - one unreachable index must not stop the others
            log(f"{r}: package list unavailable ({type(e).__name__}: {e})")
            names = []
        log(f"{r}: {len(names)} packages beyond corpus v2")
        lists.append([(r, x) for x in names])
    tasks, i = [], 0
    while any(i < len(x) for x in lists):
        tasks += [x[i] for x in lists if i < len(x)]
        i += 1
    return tasks


def collect_fineweb(a, col: Collector, cache: Path, done: set[str], donef, disk: Disk, target_chars: float) -> str:
    """FineWeb-Edu documents up to --fineweb-share of the target; returns what happened (for the manifest)."""
    budget = a.fineweb_share * target_chars
    if a.no_fineweb or budget <= 0:
        return "disabled"
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("pyarrow missing: FineWeb-Edu skipped (pip install pyarrow)")
        return "skipped: pyarrow missing"
    jobs = [(p, None) for p in a.parquet]
    if not jobs:
        if not reg.reachable(HF_PROBE):
            log("huggingface.co unreachable: FineWeb-Edu skipped; general-v3 = general-v2 + package docs")
            return "skipped: huggingface.co unreachable"
        jobs = [(cache / name.replace("/", "_"), f"https://huggingface.co/datasets/{DATASET}/resolve/main/{name}") for name in FILES[: a.files]]
    for path, url in jobs:
        task = f"fineweb-edu:{path.name}"
        if task in done:
            continue
        if col.group_chars("fineweb-edu") >= budget:
            break
        if url:
            why = None if path.exists() else disk.over(col.chars, PARQUET_BYTES)
            if why:
                log(f"FineWeb-Edu file skipped by the disk guard: {why}")
                break
            try:
                log(f"downloading {url}")
                download(url, path)
            except RuntimeError as e:
                log(f"FineWeb-Edu skipped: {e}")
                break
        pf = pq.ParquetFile(path)
        rows, kept, before = 0, 0, col.group_chars("fineweb-edu")
        for batch in pf.iter_batches(columns=["text"], batch_size=2048):
            for text in batch.column(0).to_pylist():
                kept += col.add(text or "", "general", f"fineweb-edu:{path.name}", f"row {rows}", min_chars=a.min_chars)
                rows += 1
            if col.group_chars("fineweb-edu") >= budget:
                break
        col.flush()
        with path.open("rb") as f:
            head = hashlib.sha256(f.read(1 << 20)).hexdigest()
        donef.write(json.dumps({"task": task, "registry": "fineweb-edu", "name": DATASET, "file": path.name, "rows": pf.metadata.num_rows,
                                "rows_read": rows, "license": LICENCE, "attribution": ATTRIBUTION, "url": url, "sha256_head": head,
                                "status": "ok", "kept": kept, "chars": col.group_chars("fineweb-edu") - before}) + "\n")
        donef.flush()
        log(f"FineWeb-Edu {path.name}: {kept:,} of {rows:,} rows kept")
        if url:
            path.unlink(missing_ok=True)
    return "ok"


def collect_packages(todo: list, col: Collector, donef, disk: Disk, target_chars: float) -> str:
    """Fetch, filter and cache packages until the token target or the disk guard stops it."""
    for n, (r, name) in enumerate(todo, 1):
        if col.chars >= target_chars:
            log(f"target of {target_chars / v2.CHARS_PER_TOKEN / 1e6:.0f} M new tokens reached")
            return "target reached"
        why = disk.over(col.chars)
        if why:
            log(f"disk guard: {why}; no more packages")
            return f"disk guard: {why}"
        try:
            entry, files = reg.SOURCES[r](name)
        except Exception as e:  # noqa: BLE001 - a broken package is recorded, never fatal
            entry, files = {"source": f"{r}:{name}", "registry": r, "name": name, "status": f"error: {type(e).__name__}: {e}"[:200]}, []
        before = col.chars
        kept = sum(col.add(text, kind, entry["source"], path) for path, kind, text in files)
        col.flush()
        entry.setdefault("status", "ok")
        entry.update(files=len(files), kept=kept, chars=col.chars - before)
        donef.write(json.dumps({"task": entry.pop("source"), **entry}) + "\n")
        donef.flush()
        if len(todo) <= 100 or n % 25 == 0:
            log(f"{n}/{len(todo)} {r}:{name} {entry.get('version') or ''} [{entry['status']}] kept {kept}/{len(files)}; "
                f"~{col.chars / v2.CHARS_PER_TOKEN / 1e6:.1f} M new tokens, filtered {col.stats}")
    return "all planned packages done"


def finish(d: Path, tokens: int, val_tokens: int, mix: dict, new: dict) -> dict:
    m = v2.finish_store(d, tokens, val_tokens, mix)
    m.update(pipeline=PIPELINE, new=new)
    (d / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


def append_file(src: Path, out) -> None:
    with src.open("rb") as f:
        shutil.copyfileobj(f, out, 1 << 24)


def write_stores(out: Path, v2_out: Path, docs: Path, counts: dict, hashes: set[str]) -> dict:
    """code-v3 / general-v3 = the v2 store + the new documents; base-v3 = code-v3 + general-v3; staged, then moved into place."""
    stage = out / ".v3-staging"
    shutil.rmtree(stage, ignore_errors=True)
    stores = {}
    for kind in ("code", "general"):
        d, src = stage / f"{kind}-v3", v2_out / f"{kind}-v2"
        d.mkdir(parents=True)
        m2 = json.loads((src / "train.meta.json").read_text())
        n = {}
        for split, fname in (("train", "train.bin"), ("val", "train.val.bin")):
            with open(d / fname, "wb") as f:
                append_file(src / fname, f)
                p = docs / f"{kind}-{split}.jsonl"
                n[split] = v2.tokenize_into(p, f) if p.exists() and p.stat().st_size else (0, 0)
        mix = {f"{kind}-v2": sum(m2["mix"].values())} | {g: c[f"{kind}-train"] for g, c in sorted(counts.items()) if c.get(f"{kind}-train")}
        stores[f"{kind}-v3"] = finish(d, m2["tokens"] + n["train"][0], m2["val_tokens"] + n["val"][0], mix,
                                      {"tokens": n["train"][0], "val_tokens": n["val"][0], "docs": n["train"][1] + n["val"][1]})
        log(f"store {kind}-v3: {stores[f'{kind}-v3']['tokens']:,} train / {stores[f'{kind}-v3']['val_tokens']:,} val tokens "
            f"({n['train'][0]:,} / {n['val'][0]:,} new)")
    d = stage / "base-v3"
    d.mkdir()
    for fname in ("train.bin", "train.val.bin"):
        with open(d / fname, "wb") as f:
            for kind in ("code", "general"):
                append_file(stage / f"{kind}-v3" / fname, f)
    c, g = stores["code-v3"], stores["general-v3"]
    stores["base-v3"] = finish(d, c["tokens"] + g["tokens"], c["val_tokens"] + g["val_tokens"],
                               {"code": sum(c["mix"].values()), "general": sum(g["mix"].values())},
                               {k: c["new"][k] + g["new"][k] for k in c["new"]})
    (d / "docs.sha1").write_bytes(b"".join(sorted(bytes.fromhex(h) for h in hashes)))
    log(f"store base-v3: {stores['base-v3']['tokens']:,} train / {stores['base-v3']['val_tokens']:,} val tokens")
    for name in STORES:  # base-v3 last: its train.meta.json is what makes v3 visible to scripts/longrun.py
        dest, old = out / name, stage / f"{name}.old"
        if dest.exists():
            dest.rename(old)
        (stage / name).rename(dest)
        shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(stage, ignore_errors=True)
    return stores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-tokens", type=float, default=400e6, help="new tokens to add (packages + FineWeb-Edu)")
    ap.add_argument("--max-disk-gb", type=float, default=6.0, help="cap on the build's disk footprint (cache + the three stores)")
    ap.add_argument("--registries", type=lambda s: [x for x in s.split(",") if x], default=list(REGISTRIES),
                    help=f"comma-separated subset of {','.join(REGISTRIES)}")
    ap.add_argument("--limit", type=int, default=0, help="at most this many packages per registry (tests / smoke runs)")
    ap.add_argument("--packages", type=Path, default=None, help="explicit registry:name list (one per line) instead of the popularity lists")
    ap.add_argument("--fineweb-share", type=float, default=0.5, help="share of --target-tokens for FineWeb-Edu when reachable")
    ap.add_argument("--no-fineweb", action="store_true")
    ap.add_argument("--files", type=int, default=1, help="FineWeb-Edu parquet files of sample/10BT to use (each ~2.15 GB)")
    ap.add_argument("--parquet", type=Path, action="append", default=[], help="local FineWeb-Edu parquet file(s) instead of downloading")
    ap.add_argument("--min-chars", type=int, default=500, help="FineWeb-Edu documents shorter than this are dropped")
    ap.add_argument("--root", type=Path, default=None, help="write cache, stores and manifests under this directory (tests)")
    ap.add_argument("--v2-out", type=Path, default=None, help="directory with code-v2/ and general-v2/ (default: the output dir)")
    ap.add_argument("--v2-cache", type=Path, default=None, help="corpus v2 cache (done.jsonl, docs/) for dedup (default: <cache>/v2)")
    ap.add_argument("--evals", type=Path, default=None, help="eval sets to decontaminate against (default: data/out/evals)")
    ap.add_argument("--fetch-only", action="store_true", help="collect documents, do not tokenise")
    ap.add_argument("--tokenize-only", action="store_true", help="rebuild the stores from the cached documents")
    ap.add_argument("--keep-cache", action="store_true", help="keep the cached documents after tokenisation")
    ap.add_argument("--fresh", action="store_true", help="forget an earlier v3 build (cache, done.jsonl) and start over")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args(argv)
    if unknown := set(a.registries) - set(REGISTRIES):
        ap.error(f"unknown registries {sorted(unknown)}")
    out = a.root / "out" if a.root else v2.OUT
    cache_root = a.root / "cache" if a.root else ROOT / "data" / "cache"
    manifests = a.root / "manifests" if a.root else ROOT / "data" / "manifests" / "v3"
    cache = cache_root / "v3"
    docs = cache / "docs"
    v2_out, v2_cache = a.v2_out or out, a.v2_cache or cache_root / "v2"
    evals = a.evals or bc.OUT / "evals"
    v2_done = {r["task"] for r in read_jsonl(v2_cache / "done.jsonl") if "task" in r}
    if not v2_done and a.root is None and (ROOT / "data" / "manifests" / "v2" / "sources.json").exists():
        v2_done = {r["task"] for r in json.loads((ROOT / "data" / "manifests" / "v2" / "sources.json").read_text())}
    if a.dry_run:
        tasks = plan(a, v2_done)
        print(json.dumps({"planned": len(tasks), "by_registry": {r: sum(1 for t in tasks if t[0] == r) for r in a.registries},
                          "first": [f"{r}:{n}" for r, n in tasks[:20]]}))
        return 0
    os.nice(max(0, 15 - os.nice(0)))
    if not v2.FORGE.exists() or not v2.TOKENIZER.exists():
        log(f"need {v2.FORGE} and {v2.TOKENIZER}")
        return 2
    if not all((v2_out / f"{k}-v2" / "train.meta.json").exists() for k in ("code", "general")):
        log(f"corpus v2 is required first ({v2_out}/code-v2, general-v2); run data/build_corpus_v2.py")
        return 2
    if a.fresh:
        shutil.rmtree(cache, ignore_errors=True)
    for p in (docs, manifests, out):
        p.mkdir(parents=True, exist_ok=True)
    v2.CACHE = cache  # forge tokenize's chunk file
    built = cache / "built.json"
    if built.exists() and not a.tokenize_only:
        log(f"corpus v3 already built ({built}); --fresh rebuilds it")
        return 0
    if built.exists() and not any(p.stat().st_size for p in docs.glob("*.jsonl")):
        log("the cached documents were deleted after the build; --fresh rebuilds corpus v3")
        return 2
    v2_bytes = sum((v2_out / f"{k}-v2" / f).stat().st_size for k in ("code", "general") for f in ("train.bin", "train.val.bin"))
    disk = Disk(cache, out, v2_bytes, a.max_disk_gb)
    if disk.stores(0) > disk.max:
        log(f"--max-disk-gb {a.max_disk_gb:g} cannot even hold the v2 part of the v3 stores ({disk.stores(0) / 1e9:.2f} GB)")
        return 3
    t0 = time.time()

    log(f"reading corpus v2 document hashes from {v2_cache}/docs")
    seen_v2: set[str] = set()
    for p in sorted((v2_cache / "docs").glob("*.jsonl")):
        seen_v2.update(doc_hashes(p))
    if not seen_v2:
        log(f"WARNING: no corpus v2 documents under {v2_cache}/docs; dedup only within v3")
    if a.tokenize_only:  # the cached documents were decontaminated when they were collected
        banned, eval_names = set(), sorted(f.stem for f in evals.glob("*.jsonl"))
    else:
        banned, eval_names = banned_ngrams(evals, fetch_missing=a.root is None and a.evals is None)
    col = Collector(docs, banned, seen_v2)
    log(f"{len(seen_v2):,} v2 documents for dedup, {len(col.seen):,} v3 documents cached (~{col.chars / v2.CHARS_PER_TOKEN / 1e6:.1f} M tokens), "
        f"{len(banned):,} banned 13-grams")
    fineweb = stop = "not run (--tokenize-only)"
    if not a.tokenize_only:
        try:
            v2._MODS = bc.scp_modules()
        except Exception:  # noqa: BLE001 - SCP checkout absent: v2's simple heuristics
            v2._MODS = None
        if (cache / "done.jsonl").exists():
            repair(cache / "done.jsonl")
        done = {r["task"] for r in read_jsonl(cache / "done.jsonl") if "task" in r}
        target_chars = a.target_tokens * v2.CHARS_PER_TOKEN
        with (cache / "done.jsonl").open("a") as donef:
            fineweb = collect_fineweb(a, col, cache, done, donef, disk, target_chars)
            tasks = plan(a, v2_done)
            todo = [t for t in tasks if f"{t[0]}:{t[1]}" not in done]
            log(f"{len(tasks)} packages planned, {len(tasks) - len(todo)} already done")
            stop = collect_packages(todo, col, donef, disk, target_chars)
    col.close()

    sources = read_jsonl(cache / "done.jsonl")
    (manifests / "sources.json").write_text(json.dumps(sorted(sources, key=lambda s: s["task"]), indent=1))
    by_status: dict[str, dict[str, int]] = {}
    for s in sources:
        st = by_status.setdefault(s.get("registry", s["task"].split(":", 1)[0]), {})
        st[s.get("status", "?").split(":")[0]] = st.get(s.get("status", "?").split(":")[0], 0) + 1
    (manifests / "filters.json").write_text(json.dumps({"filters": col.stats, "registries": col.counts}, indent=2))
    if a.fetch_only:
        log(f"--fetch-only: {len(col.seen):,} documents cached in {time.time() - t0:.0f}s; packages {by_status}")
        return 0

    need = disk.stores(col.chars)
    if shutil.disk_usage(out).free - need < MARGIN:
        log(f"not enough free disk for the stores ({need / 1e9:.2f} GB + margin); documents kept for --tokenize-only")
        return 3
    stores = write_stores(out, v2_out, docs, col.counts, seen_v2 | col.seen)
    (manifests / "stores.json").write_text(json.dumps({
        "stores": stores, "packages": by_status, "registries": col.counts, "filters": col.stats, "stopped": stop,
        "fineweb_edu": {"status": fineweb, "dataset": DATASET, "licence": LICENCE, "attribution": ATTRIBUTION},
        "licence_policy": {"allow": bc.ALLOW.pattern, "deny": bc.DENY.pattern, "deny_also": reg.COPYLEFT.pattern},
        "dedup": {"method": "sha1 of whitespace-normalised text", "v2_documents": len(seen_v2), "v3_documents": len(col.seen)},
        "evals_decontaminated_against": eval_names}, indent=2))
    if not a.keep_cache:
        for p in [*docs.glob("*.jsonl"), *cache.glob("*.parquet*"), cache / "tok-chunk.jsonl"]:
            p.unlink(missing_ok=True)
    built.write_text(json.dumps({"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tokens": {k: v["tokens"] for k, v in stores.items()},
                                 "new_tokens": stores["base-v3"]["new"]["tokens"]}, indent=2))
    log(f"done in {time.time() - t0:.0f}s; packages {by_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
