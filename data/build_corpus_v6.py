"""Corpus v6: the sharded, resumable pipeline for 250-500 B tokens, run on RunPod CPU pods with the network volume.

Text is the source of truth, tokens are derived: kept documents are stored as gzip jsonl shards, so a later model
with a bigger vocabulary re-tokenizes without re-downloading. Every stage is restartable (done markers, atomic
renames) and parallel (--worker i --workers n on one or several pods sharing the volume).

  plan      list the files of every enabled source (Hugging Face API or local glob) -> OUT/plan.json
  filter    per raw file: download (resumable), licence tier, content rules (data/premium.py on top of
            data/hq_filter.py), 13-gram decontamination (HumanEval, MBPP, GSM8K test), exact key and MinHash
            sketch; kept documents -> OUT/kept/NNNNNNN.jsonl.gz; the raw file is a cache and is removed after
  dedup     global exact and MinHash near-duplicate removal over all shards, hash-partitioned so memory stays
            bounded (OUT/dedup/drop.npy); the earliest document wins (sources in priority order)
  tokenize  forge tokenizer, <eos> after every document, ~0.05 % val split by document hash
  pack      one train.bin / train.val.bin store with the kept-docs index and report.json the accel trainer's
            fail-closed CleanStore expects, plus manifest.json (sha256, tokens per source and licence tier)
  status    counts so far

Licence tiers (data/sources_v6.json): A = public domain, CC0, CC BY, permissive code licences (the policy of corpus
v1-v5); B = share-alike (CC BY-SA), excluded by that policy (build_corpus.DENY) unless the owner changes it;
C = web crawls with only a dataset licence (ODC-By), only with --tiers including C after an owner decision.
Never: NC/ND licences, copyleft code, outputs of closed models, unknown licences.

Secrets: a Hugging Face token for gated datasets is read from the HF_TOKEN environment variable (a RunPod secret)
and is never printed or written anywhere.

PYTHONHASHSEED is forced to 0 when run as a script (the decontamination n-gram hashes must agree across
processes and pods).

  python3 data/build_corpus_v6.py plan --out /vol/data/v6 [--tiers A] [--sources a,b] [--pilot-tokens 3e9]
  python3 data/build_corpus_v6.py filter --out /vol/data/v6 --worker 0 --workers 16
  python3 data/build_corpus_v6.py dedup --out /vol/data/v6
  python3 data/build_corpus_v6.py tokenize --out /vol/data/v6 --worker 0 --workers 16
  python3 data/build_corpus_v6.py pack --out /vol/data/v6
  python3 data/build_corpus_v6.py inspect --source NAME      # schema and licence values of the first file
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
import hq_filter  # noqa: E402,F401  (premium builds on it)
import premium  # noqa: E402


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "data" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load("build_corpus", "build_corpus.py")
v4 = _load("corpus_v4", "build_corpus_v4.py")
CONFIG = ROOT / "data" / "sources_v6.json"
EVALS = ROOT / "data" / "out" / "evals"
TOKENIZER = ROOT / "data" / "tokenizer.json"
FORGE = ROOT / "target" / "release" / "forge"
EOS, VOCAB = 8190, 8192
SHIFT = 24                      # gid = file index << SHIFT | row of the kept shard
VAL_MOD = 2000                  # val = exact key % VAL_MOD == 0 (0.05 %)
CHARS_PER_TOKEN = 3.3           # forge tokenizer on code+prose (v1 stores); only for budgets
HF = "https://huggingface.co"
READ_EXT = (".parquet", ".jsonl", ".jsonl.gz", ".json.gz", ".jsonl.zst", ".json")


def log(msg: str) -> None:
    print(f"[v6 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- licences

CC_BY = re.compile(r"\bcc[-_ ]?by\b(?![-_ ]?(?:sa|nc|nd)\b)", re.I)
NC_ND = re.compile(r"(?i)\b(?:cc[-_ ]?by[-_ ]?(?:sa[-_ ]?)?(?:nc|nd)|non[-_ ]?commercial|no[-_ ]?deriv)")
SHARE_ALIKE = re.compile(r"(?i)\b(?:cc[-_ ]?by[-_ ]?sa|share[-_ ]?alike)\b")


def licence_tier(lic, kind: str) -> str | None:
    """'A', 'B' or None (drop) for one licence string or a list of them (all must pass; the worst tier counts)."""
    if isinstance(lic, (list, tuple)):
        tiers = [licence_tier(x, kind) for x in lic] or [None]
        return None if None in tiers else max(tiers)
    lic = str(lic or "").strip()
    if not lic or NC_ND.search(lic):
        return None
    if SHARE_ALIKE.search(lic):
        return None if kind == "code" else "B"
    if bc.DENY.search(lic) or premium_copyleft(lic):
        return None
    if bc.ALLOW.search(lic) or CC_BY.search(lic) or re.search(r"(?i)\b(public[-_ ]?domain|pd|cc[-_ ]?0)\b", lic):
        return "A"
    return None


_COPYLEFT = re.compile(r"(?i)\b(?:a|l)?gpl|general public license|mozilla public|eclipse public|\bepl\b|\bcddl\b|"
                       r"european union public|server side public")


def premium_copyleft(lic: str) -> bool:
    return bool(_COPYLEFT.search(lic))


# ---------------------------------------------------------------- hashing

NORM = re.compile(r"\s+")


def exact_key(text: str) -> int:
    """First 8 bytes of sha1 of the whitespace-normalised text (the v2-v5 dedup rule)."""
    return int.from_bytes(hashlib.sha1(NORM.sub(" ", text).strip().encode()).digest()[:8], "little")


class Sketcher:
    """v4.sketch with a word-hash cache (same values, several times faster on natural text)."""

    def __init__(self, cache: int = 2_000_000):
        self.cache: dict[str, int] = {}
        self.max = cache

    def word(self, w: str) -> int:
        h = self.cache.get(w)
        if h is None:
            h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little") & v4.P61
            if len(self.cache) >= self.max:
                self.cache.clear()
            self.cache[w] = h
        return h

    def sketch(self, text: str) -> np.ndarray | None:
        words = v4.WORD.findall(text.lower())[:v4.MAX_WORDS]
        if len(words) < 5:
            return None
        wh = np.fromiter((self.word(w) for w in words), np.uint64, len(words))
        sh = wh[:-4].copy()
        for k in range(1, 5):
            sh = sh * np.uint64(0x9E3779B97F4A7C15) ^ wh[k:len(wh) - 4 + k]
        sh ^= sh >> np.uint64(29)
        sh *= np.uint64(0xBF58476D1CE4E5B9)
        sh ^= sh >> np.uint64(32)
        out = np.full(v4.BINS, np.iinfo(np.uint64).max, np.uint64)
        np.minimum.at(out, (sh % np.uint64(v4.BINS)).astype(np.int64), sh // np.uint64(v4.BINS))
        return out


def band_keys(sk: np.ndarray) -> np.ndarray:
    return np.array([int.from_bytes(hashlib.blake2b(sk[b * v4.ROWS:(b + 1) * v4.ROWS].tobytes(), digest_size=8)
                                    .digest(), "little") | 1 for b in range(v4.BANDS)], np.uint64)  # 0 = no sketch


# ---------------------------------------------------------------- sources and reading


def load_config(path: Path = CONFIG) -> dict:
    cfg = json.loads(path.read_text())
    names = [s["name"] for s in cfg["sources"]]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate source names in config")
    return cfg


def _auth() -> dict:
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


class _KeepTokenOnHF(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but send the Hugging Face token only to Hugging Face hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        host = urllib.parse.urlparse(newurl).hostname or ""
        if new is not None and not (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co"))):
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_KeepTokenOnHF)


def _get(url: str, timeout: int = 120):
    return _OPENER.open(urllib.request.Request(url, headers=_auth()), timeout=timeout)


def list_files(src: dict) -> list[dict]:
    if "local" in src:
        return [{"path": p, "size": os.path.getsize(p), "url": "file://" + os.path.abspath(p)}
                for p in sorted(glob.glob(src["local"], recursive=True)) if p.endswith(READ_EXT)]
    repo, rev = src["repo"], src.get("revision", "main")
    out, url = [], (f"{HF}/api/datasets/{repo}/tree/{rev}/{urllib.parse.quote(src.get('dir', ''))}"
                    "?recursive=true&expand=false")
    while url:
        with _get(url) as r:
            items = json.loads(r.read())
            link = r.headers.get("Link", "")
        for it in items:
            p = it.get("path", "")
            if it.get("type") == "file" and fnmatch.fnmatch(p, src["files"]) and p.endswith(READ_EXT):
                out.append({"path": p, "size": int(it.get("size", 0)),
                            "url": f"{HF}/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(p)}"})
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return sorted(out, key=lambda f: f["path"])


def download(url: str, dest: Path) -> Path:
    if url.startswith("file://"):
        return Path(url[7:])
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(6):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={**_auth(), **({"Range": f"bytes={have}-"} if have else {})})
            with _OPENER.open(req, timeout=300) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.truncate(0)
                shutil.copyfileobj(r, f, 1 << 22)
            part.rename(dest)
            return dest
        except OSError as e:
            log(f"download retry {attempt + 1} ({type(e).__name__})")
            time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError(f"download failed: {url.split('?')[0]}")


def _field(row: dict, dotted: str | None):
    if not dotted:
        return None
    v = row
    for k in dotted.split("."):
        if isinstance(v, str) and v.startswith("{"):
            try:
                v = json.loads(v)
            except ValueError:
                return None
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def rows(path: Path, src: dict):
    name = str(path)
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = sorted({c.split(".")[0] for c in (src["text"], src.get("path"), src.get("licence"), src.get("id"),
                                                   *src.get("where", {}).keys()) if c} & set(pf.schema_arrow.names))
        for batch in pf.iter_batches(batch_size=2048, columns=cols):
            yield from batch.to_pylist()
        return
    if name.endswith(".zst"):
        import zstandard
        fh = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8")
    elif name.endswith(".gz"):
        fh = gzip.open(path, "rt", encoding="utf-8")
    else:
        fh = open(path, encoding="utf-8")
    with fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _where(row: dict, cond: dict) -> bool:
    """Source-level row conditions: {"field": {">=": 4}} or {"field": "value"} or {"field": ["a", "b"]}."""
    for k, want in cond.items():
        v = _field(row, k)
        if isinstance(want, dict):
            for op, x in want.items():
                if v is None or not {">=": v >= x, ">": v > x, "<=": v <= x, "<": v < x}[op]:
                    return False
        elif isinstance(want, list):
            if v not in want:
                return False
        elif v != want:
            return False
    return True


# ---------------------------------------------------------------- stages


def stage_plan(out: Path, cfg: dict, tiers: set[str], only: set[str] | None, pilot: float | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        raise SystemExit(f"{plan_path} exists; the plan fixes the document ids, so it is never rewritten")
    files, sources = [], []
    for src in sorted(cfg["sources"], key=lambda s: -s.get("priority", 0)):
        if src["tier"] not in tiers or (only and src["name"] not in only) or src.get("disabled"):
            continue
        fl = list_files(src)
        log(f"{src['name']}: {len(fl)} files, {sum(f['size'] for f in fl) / 1e9:.1f} GB")
        if pilot:  # pilot run: small budget per source until the value gate has measured it (docs/PLAN_100B_MOE.md)
            src = {**src, "budget_tokens": min(src.get("budget_tokens", pilot), pilot)}
        sources.append(src)
        for f in fl:
            files.append({"idx": len(files), "source": src["name"], **f})
    if len(files) >= 1 << (64 - SHIFT):
        raise SystemExit("too many files for the gid layout")
    plan_path.write_text(json.dumps({"tiers": sorted(tiers), "pilot_tokens": pilot, "sources": sources, "files": files,
                                     "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    log(f"plan: {len(files)} files, {sum(f['size'] for f in files) / 1e12:.2f} TB raw")


def _plan(out: Path) -> dict:
    return json.loads((out / "plan.json").read_text())


def _done(out: Path, idx: int) -> Path:
    return out / "kept" / f"{idx:07d}.done.json"


def kept_chars(out: Path) -> dict[str, int]:
    tot: dict[str, int] = {}
    for p in (out / "kept").glob("*.done.json"):
        d = json.loads(p.read_text())
        tot[d["source"]] = tot.get(d["source"], 0) + d["chars"]
    return tot


def stage_filter(out: Path, worker: int, workers: int, min_free_gb: float) -> None:
    plan = _plan(out)
    srcs = {s["name"]: s for s in plan["sources"]}
    (out / "kept").mkdir(exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    banned, names = bc_banned()
    log(f"worker {worker}/{workers}: {len(banned):,} banned 13-grams from {names}")
    sk = Sketcher()
    for f in plan["files"]:
        if f["idx"] % workers != worker or _done(out, f["idx"]).exists():
            continue
        src = srcs[f["source"]]
        budget = src.get("budget_tokens", float("inf")) * CHARS_PER_TOKEN
        if kept_chars(out).get(src["name"], 0) >= budget:
            continue
        free = shutil.disk_usage(out).free / 1e9
        if free < min_free_gb + f["size"] / 1e9:
            raise SystemExit(f"only {free:.0f} GB free on the volume; stop (nothing is deleted)")
        filter_file(out, f, src, banned, sk)


def banned_by_set(evals: Path | None = None) -> dict[str, set[int]]:
    """The 13-gram hashes of every eval set (*.jsonl) in `evals` (default EVALS), by set name."""
    if evals is None and not all((EVALS / f"{n}.jsonl").exists() for n in ("humaneval", "mbpp", "gsm8k")):
        bc.get_evals()  # fresh checkout: fetch the eval sets (raw.githubusercontent.com) into data/out/evals
    by = {}
    for p in sorted((evals or EVALS).glob("*.jsonl")):
        b = set()
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                vals = [v for v in row.values() if isinstance(v, str)]
                vals += [x for v in row.values() if isinstance(v, list) for x in v if isinstance(x, str)]
                b |= bc.ngrams(" ".join(vals))
        by[p.stem] = b
    if not by or (evals is None and not {"humaneval", "mbpp", "gsm8k"} <= set(by)):
        raise SystemExit(f"eval sets missing in {evals or EVALS}: decontamination impossible, refusing to run")
    return by


def bc_banned() -> tuple[set[int], list[str]]:
    by = banned_by_set()
    return set().union(*by.values()), list(by)


def filter_file(out: Path, f: dict, src: dict, banned: set[int], sk: Sketcher) -> dict:
    t0, idx = time.time(), f["idx"]
    raw = download(f["url"], out / "raw" / f"{idx:07d}{''.join(Path(f['path']).suffixes)}")
    kind, reasons = src["kind"], {}
    exact, bands, fps, chars = [], [], [], []
    tmp = out / "kept" / f".{idx:07d}.jsonl.gz.tmp"
    n_rows = 0
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as g:
        for row in rows(raw, src):
            n_rows += 1
            why = keep_row(row, src, kind, banned)
            if why[0] is None:
                reasons[why[1]] = reasons.get(why[1], 0) + 1
                continue
            text, path, lic, tier = why
            s = sk.sketch(text)
            exact.append(exact_key(text))
            bands.append(band_keys(s) if s is not None else np.zeros(v4.BANDS, np.uint64))
            fps.append((s & np.uint64(0xFFFF)).astype(np.uint16) if s is not None else np.zeros(v4.BINS, np.uint16))
            chars.append(len(text))
            g.write(json.dumps({"t": text, "p": path, "l": lic, "tier": tier, "s": src["name"]}) + "\n")
            reasons["ok"] = reasons.get("ok", 0) + 1
    if len(exact) >= 1 << SHIFT:
        raise SystemExit(f"file {idx}: more than {1 << SHIFT} kept documents; split the source files")
    np.savez(out / "kept" / f".{idx:07d}.meta.npz", exact=np.array(exact, np.uint64),
             bands=np.array(bands, np.uint64).reshape(-1, v4.BANDS), chars=np.array(chars, np.int64))
    np.save(out / "kept" / f".{idx:07d}.fp.npy", np.array(fps, np.uint16).reshape(-1, v4.BINS))
    for a, b in ((f".{idx:07d}.jsonl.gz.tmp", f"{idx:07d}.jsonl.gz"), (f".{idx:07d}.meta.npz", f"{idx:07d}.meta.npz"),
                 (f".{idx:07d}.fp.npy", f"{idx:07d}.fp.npy")):
        os.replace(out / "kept" / a, out / "kept" / b)
    done = {"idx": idx, "source": src["name"], "path": f["path"], "rows": n_rows, "kept": len(exact),
            "chars": int(sum(chars)), "reasons": reasons, "seconds": round(time.time() - t0, 1)}
    _done(out, idx).write_text(json.dumps(done))
    if not f["url"].startswith("file://"):
        raw.unlink()  # download cache only; the kept text is the product
    log(f"{idx:07d} {src['name']}: kept {len(exact):,}/{n_rows:,} docs, {sum(chars) / 1e6:.0f} M chars, "
        f"{done['seconds']:.0f}s")
    return done


def keep_row(row: dict, src: dict, kind: str, banned: set[int]):
    """(text, path, licence, tier) or (None, reason)."""
    text = _field(row, src["text"])
    if not isinstance(text, str) or not text.strip():
        return None, "empty"
    if src.get("where") and not _where(row, src["where"]):
        return None, "where"
    path = _field(row, src.get("path")) or ""
    if src["tier"] == "C":
        lic, tier = src.get("dataset_licence", ""), "C"
    else:
        lic = _field(row, src["licence"]) if src.get("licence") else src.get("source_licence")
        tier = licence_tier(lic, kind)
        if tier is None or tier > src["tier"]:
            return None, "licence"
    if kind == "code":
        ok, why = premium.keep(text, "code", path)
    elif kind == "reasoning":  # traces of data/build_reasoning.py are verified when generated
        ok, why = len(text) <= premium.TEXT_MAX, "too-long"
    else:
        ok, why = premium.keep_text(text)
    if not ok:
        return None, why
    if bc.ngrams(text) & banned:
        return None, "contaminated"
    return text, path, lic if isinstance(lic, str) else json.dumps(lic), tier


def _shards(out: Path) -> list[int]:
    return sorted(int(p.name[:7]) for p in (out / "kept").glob("*.done.json"))


def stage_dedup(out: Path, partitions: int) -> None:
    d = out / "dedup"
    if (d / "drop.npy").exists():
        raise SystemExit(f"{d / 'drop.npy'} exists; dedup ran already")
    d.mkdir(exist_ok=True)
    rec = np.dtype([("k", "<u8"), ("g", "<u8")])
    ex_f = [open(d / f"exact-{p:04d}.bin", "wb") for p in range(partitions)]
    bd_f = [open(d / f"band-{p:04d}.bin", "wb") for p in range(partitions)]
    total = 0
    for idx in _shards(out):
        m = np.load(out / "kept" / f"{idx:07d}.meta.npz")
        exact, bands = m["exact"], m["bands"]
        n = len(exact)
        total += n
        g = (np.uint64(idx) << np.uint64(SHIFT)) | np.arange(n, dtype=np.uint64)
        for keys, gg, files in ((exact, g, ex_f), (bands.reshape(-1), np.repeat(g, v4.BANDS), bd_f)):
            ok = keys != 0
            r = np.empty(int(ok.sum()), rec)
            r["k"], r["g"] = keys[ok], gg[ok]
            part = (r["k"] % np.uint64(partitions)).astype(np.int64)
            for p in np.unique(part):
                r[part == p].tofile(files[p])
    for fh in ex_f + bd_f:
        fh.close()
    exact_drop = []
    for p in range(partitions):
        r = np.fromfile(d / f"exact-{p:04d}.bin", rec)
        if len(r) < 2:
            continue
        r = r[np.lexsort((r["g"], r["k"]))]
        dup = np.concatenate([[False], r["k"][1:] == r["k"][:-1]])
        exact_drop.append(r["g"][dup])
    exact_drop = np.unique(np.concatenate(exact_drop)) if exact_drop else np.zeros(0, np.uint64)
    fps = _FpCache(out)
    near_drop, pairs = [], 0
    for p in range(partitions):
        r = np.fromfile(d / f"band-{p:04d}.bin", rec)
        r = r[~np.isin(r["g"], exact_drop)]
        if len(r) < 2:
            continue
        r = r[np.lexsort((r["g"], r["k"]))]
        first = np.concatenate([[True], r["k"][1:] != r["k"][:-1]])
        anchor = r["g"][np.maximum.accumulate(np.where(first, np.arange(len(r)), 0))]
        cand = ~first
        for a, m in zip(anchor[cand], r["g"][cand]):
            pairs += 1
            if np.mean(fps.get(int(a)) == fps.get(int(m))) >= v4.CONFIRM:
                near_drop.append(m)
    near_drop = np.setdiff1d(np.unique(np.array(near_drop, np.uint64)), exact_drop)
    drop = np.union1d(exact_drop, near_drop)
    np.save(d / ".drop.npy", drop)
    os.replace(d / ".drop.npy", d / "drop.npy")
    stats = {"documents": int(total), "exact_duplicates": int(len(exact_drop)), "near_duplicates": int(len(near_drop)),
             "candidate_pairs": int(pairs), "kept": int(total - len(drop)), "partitions": partitions,
             "minhash": {"bins": v4.BINS, "bands": v4.BANDS, "rows": v4.ROWS, "confirm": v4.CONFIRM,
                         "fingerprint_bits": 16}}
    (d / "dedup.json").write_text(json.dumps(stats, indent=1))
    for p in range(partitions):  # partition files are scratch
        (d / f"exact-{p:04d}.bin").unlink()
        (d / f"band-{p:04d}.bin").unlink()
    log(f"dedup: {stats}")


class _FpCache:
    def __init__(self, out: Path, size: int = 256):
        self.out, self.size, self.c = out, size, OrderedDict()

    def get(self, gid: int) -> np.ndarray:
        idx, row = gid >> SHIFT, gid & ((1 << SHIFT) - 1)
        a = self.c.get(idx)
        if a is None:
            a = np.load(self.out / "kept" / f"{idx:07d}.fp.npy", mmap_mode="r")
            self.c[idx] = a
            if len(self.c) > self.size:
                self.c.popitem(last=False)
        else:
            self.c.move_to_end(idx)
        return a[row]


def stage_tokenize(out: Path, worker: int, workers: int, forge: Path, tokenizer: Path, chunk: int = 20_000) -> None:
    drop = np.load(out / "dedup" / "drop.npy")
    tdir = out / "tok"
    tdir.mkdir(exist_ok=True)
    tmp = tdir / f".chunk-{worker}.jsonl"
    for idx in _shards(out):
        if idx % workers != worker or (tdir / f"{idx:07d}.done.json").exists():
            continue
        m = np.load(out / "kept" / f"{idx:07d}.meta.npz")
        g = (np.uint64(idx) << np.uint64(SHIFT)) | np.arange(len(m["exact"]), dtype=np.uint64)
        keep = ~np.isin(g, drop)
        val = (m["exact"] % np.uint64(VAL_MOD)) == 0
        bins = {s: open(tdir / f".{idx:07d}.{s}.bin", "wb") for s in ("train", "val")}
        lens = {"train": [], "val": []}
        counts: dict[str, dict[str, int]] = {}
        batch: list[tuple[str, str, str]] = []

        def flush() -> None:
            if not batch:
                return
            tmp.write_text("".join(json.dumps(t) + "\n" for t, _, _ in batch))
            res = subprocess.run(["nice", "-n", "10", str(forge), "tokenize", "--tokenizer", str(tokenizer), "--jsonl",
                                  str(tmp)], check=True, capture_output=True, text=True).stdout.splitlines()
            if len(res) != len(batch):
                raise RuntimeError(f"forge tokenize returned {len(res)} lines for {len(batch)} documents")
            for (_, split, key), line in zip(batch, res):
                ids = np.array(json.loads(line) + [EOS], np.uint16)
                if int(ids.max(initial=0)) >= VOCAB:
                    raise RuntimeError("token id outside the vocabulary")
                ids.astype("<u2").tofile(bins[split])
                lens[split].append(len(ids))
                c = counts.setdefault(key, {"train": 0, "val": 0, "docs": 0})
                c[split] += len(ids)
                c["docs"] += 1
            batch.clear()

        with gzip.open(out / "kept" / f"{idx:07d}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not keep[i]:
                    continue
                d = json.loads(line)
                batch.append((d["t"], "val" if val[i] else "train", f"{d['s']}|{d['tier']}"))
                if len(batch) >= chunk:
                    flush()
        flush()
        for s, fh in bins.items():
            fh.close()
            os.replace(tdir / f".{idx:07d}.{s}.bin", tdir / f"{idx:07d}.{s}.bin")
            np.save(tdir / f"{idx:07d}.{s}.len.npy", np.array(lens[s], np.int64))
        (tdir / f"{idx:07d}.done.json").write_text(json.dumps({"idx": idx, "counts": counts}))
        log(f"tok {idx:07d}: {sum(lens['train']):,} train / {sum(lens['val']):,} val tokens")
    tmp.unlink(missing_ok=True)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 24):
            h.update(c)
    return h.hexdigest()


def stage_pack(out: Path, tokenizer: Path) -> None:
    tdir, sdir = out / "tok", out / "store"
    if sdir.exists():
        raise SystemExit(f"{sdir} exists; move it away first (nothing is overwritten)")
    missing = [i for i in _shards(out) if not (tdir / f"{i:07d}.done.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} shards not tokenized yet (first {missing[:5]})")
    stage = out / ".store-staging"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    tokens, index, per = {"train": 0, "val": 0}, [], {}
    for split, fname in (("train", "train.bin"), ("val", "train.val.bin")):
        with open(stage / fname, "wb") as dst:
            for idx in _shards(out):
                with open(tdir / f"{idx:07d}.{split}.bin", "rb") as src:
                    shutil.copyfileobj(src, dst, 1 << 24)
                if split == "train":
                    ln = np.load(tdir / f"{idx:07d}.train.len.npy")
                    starts = tokens["train"] + np.concatenate([[0], np.cumsum(ln)[:-1]])
                    index.append(np.stack([starts, ln - 1], axis=1))
                    tokens["train"] += int(ln.sum())
                else:
                    tokens["val"] += int(np.load(tdir / f"{idx:07d}.val.len.npy").sum())
    for idx in _shards(out):
        for key, c in json.loads((tdir / f"{idx:07d}.done.json").read_text())["counts"].items():
            p = per.setdefault(key, {"train": 0, "val": 0, "docs": 0})
            for k in p:
                p[k] += c[k]
    idx_arr = np.concatenate(index) if index else np.zeros((0, 2), np.int64)
    kept = np.concatenate([np.arange(len(idx_arr), dtype=np.int64)[:, None], idx_arr], axis=1)
    np.save(stage / "kept.npy", kept)  # (doc, start, len without eos): the CleanStore index
    train_sha = _sha(stage / "train.bin")
    (stage / "report.json").write_text(json.dumps({"train_bin_sha256_expected": train_sha,
                                                   "kept_docs": int(len(kept))}, indent=1))
    shutil.copyfile(tokenizer, stage / "tokenizer.json")
    meta = {"tokens": tokens["train"], "val_tokens": tokens["val"], "dtype": "uint16", "vocab_size": VOCAB,
            "bin": "train.bin", "val_bin": "train.val.bin",
            "mix": {k: v["train"] for k, v in sorted(per.items())}}
    (stage / "train.meta.json").write_text(json.dumps(meta, indent=1))
    plan = _plan(out)
    manifest = {"store": "corpus-v6", **meta, "documents": int(len(kept)), "per_source_tier": per,
                "tiers": plan["tiers"], "sources": [{k: s.get(k) for k in ("name", "repo", "local", "files", "kind",
                                                                           "tier", "dataset_licence", "where")}
                                                    for s in plan["sources"]],
                "dedup": json.loads((out / "dedup" / "dedup.json").read_text()),
                "sha256": {"train.bin": train_sha, "train.val.bin": _sha(stage / "train.val.bin"),
                           "tokenizer.json": _sha(stage / "tokenizer.json"), "kept.npy": _sha(stage / "kept.npy")},
                "pipeline": "data/build_corpus_v6.py: licence tiers, data/premium.py content rules, 13-gram "
                            "decontamination (HumanEval, MBPP, GSM8K test), exact + MinHash dedup, forge tokenizer, "
                            f"val = exact key % {VAL_MOD} == 0"}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
    stage.rename(sdir)
    log(f"store: {tokens['train']:,} train / {tokens['val']:,} val tokens, {len(kept):,} documents")


def stage_status(out: Path) -> None:
    plan = _plan(out)
    done = [json.loads(p.read_text()) for p in (out / "kept").glob("*.done.json")]
    by: dict[str, dict] = {}
    for d in done:
        b = by.setdefault(d["source"], {"files": 0, "kept": 0, "chars": 0, "reasons": {}})
        b["files"] += 1
        b["kept"] += d["kept"]
        b["chars"] += d["chars"]
        for k, v in d["reasons"].items():
            b["reasons"][k] = b["reasons"].get(k, 0) + v
    for b in by.values():
        b["approx_tokens"] = int(b["chars"] / CHARS_PER_TOKEN)
    print(json.dumps({"files_done": len(done), "files_planned": len(plan["files"]), "sources": by}, indent=1))


def stage_inspect(name: str, cfg: dict) -> None:
    src = next(s for s in cfg["sources"] if s["name"] == name)
    fl = list_files(src)
    print(json.dumps({"files": len(fl), "bytes": sum(f["size"] for f in fl), "first": fl[0]["path"] if fl else None}))
    if not fl:
        return
    tmpd = Path(os.environ.get("TMPDIR", "/tmp")) / "v6-inspect"
    tmpd.mkdir(parents=True, exist_ok=True)
    p = download(fl[0]["url"], tmpd / Path(fl[0]["path"]).name)
    lic_vals: dict[str, int] = {}
    for i, row in enumerate(rows(p, {**src, "where": {}})):
        if i == 0:
            print(json.dumps({"keys": sorted(row.keys())}))
        lic = _field(row, src.get("licence")) if src.get("licence") else None
        k = json.dumps(lic)[:80]
        lic_vals[k] = lic_vals.get(k, 0) + 1
        if i >= 2000:
            break
    print(json.dumps({"licence_values_first_2000_rows": dict(sorted(lic_vals.items(), key=lambda x: -x[1])[:20])},
                     indent=1))


_SCAN: dict = {}


def _scan_init(banned_by: dict) -> None:
    _SCAN["banned_by"] = banned_by


def _scan_file(job: tuple) -> dict:
    """Content rules and 13-gram decontamination of one local file; ids of the rows to exclude."""
    path, src = job
    banned_by = _SCAN["banned_by"]
    stats = {"file": str(path), "rows": 0, "kept": 0, "kept_chars": 0, "reasons": {}, "contaminated_by": {}}
    excluded = []
    for i, row in enumerate(rows(Path(path), src)):
        stats["rows"] += 1
        text = _field(row, src["text"])
        rid = _field(row, src["id"]) if src["id"] else None
        rid = str(rid) if rid is not None else f"{Path(path).name}:{i}"
        if not isinstance(text, str) or not text.strip():
            why = "empty"
        elif hit := sorted(n for grams in [bc.ngrams(text)] for n, b in banned_by.items() if grams & b):
            why = "contaminated"
            for n in hit:
                stats["contaminated_by"][n] = stats["contaminated_by"].get(n, 0) + 1
        else:
            if src["kind"] == "code":
                ok, why = premium.keep(text, "code", _field(row, src.get("path")) or "")
            elif src["kind"] == "general":
                ok, why = premium.keep_text(text)
            else:  # reasoning traces: verified when generated
                ok, why = len(text) <= premium.TEXT_MAX, "too-long"
            why = "ok" if ok else why
        if why == "ok":
            stats["kept"] += 1
            stats["kept_chars"] += len(text)
        else:
            excluded.append({"id": rid, "reason": why})
        stats["reasons"][why] = stats["reasons"].get(why, 0) + 1
    stats["excluded"] = excluded
    return stats


def stage_scan(out: Path, files: str, text: str, kind: str, rid: str, licence: str, workers: int,
               evals: Path | None = None) -> dict:
    """Check data that already lies on a machine (for example FineMath on the box) without building a store:
    13-gram decontamination against HumanEval, MBPP and GSM8K test, the content rules of `kind`, and a list of
    the rows to exclude. `evals` swaps in other sets, for example a new held-out set as JSONL. Nothing is
    changed or deleted; the report goes to OUT/scan.json, the ids to OUT/excluded.jsonl."""
    import glob
    paths = sorted({p for g in files.split(",") for p in glob.glob(g.strip()) if Path(p).is_file()})
    if not paths:
        raise SystemExit(f"no files match {files}")
    banned_by = banned_by_set(evals)
    names = list(banned_by)
    src = {"text": text, "kind": kind, "id": rid or None}
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(max(1, min(workers, len(paths))), initializer=_scan_init, initargs=(banned_by,)) as pool:
        per = pool.map(_scan_file, [(p, src) for p in paths], chunksize=1)
    tot = {"rows": 0, "kept": 0, "kept_chars": 0, "reasons": {}, "contaminated_by": {}}
    with open(out / "excluded.jsonl", "w") as f:
        for st in per:
            for e in st.pop("excluded"):
                f.write(json.dumps({**e, "file": Path(st["file"]).name}) + "\n")
            for k in ("rows", "kept", "kept_chars"):
                tot[k] += st[k]
            for k in ("reasons", "contaminated_by"):
                for r, c in st[k].items():
                    tot[k][r] = tot[k].get(r, 0) + c
    rep = {"files": len(paths), "kind": kind, "licence": licence, "eval_sets": names, **tot,
           "kept_tokens_est": int(tot["kept_chars"] / CHARS_PER_TOKEN), "yield": tot["kept"] / max(1, tot["rows"]),
           "seconds": round(time.time() - t0, 1), "per_file": per,
           "method": "13-gram decontamination (data/build_corpus.ngrams) against the eval sets above; content "
                     f"rules of data/premium.py for kind {kind}; nothing changed or deleted"}
    (out / "scan.json").write_text(json.dumps(rep, indent=1))
    log(f"scan: {tot['rows']:,} rows, kept {tot['kept']:,}, contaminated {tot['reasons'].get('contaminated', 0):,} "
        f"{tot['contaminated_by']}, report {out / 'scan.json'}")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["plan", "filter", "dedup", "tokenize", "pack", "status", "inspect", "scan"])
    ap.add_argument("--out", type=Path, default=Path("/vol/data/v6"))
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--tiers", default="A")
    ap.add_argument("--sources", default="")
    ap.add_argument("--pilot-tokens", type=float, default=None, help="plan: cap every source at this many tokens")
    ap.add_argument("--source", default="")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--partitions", type=int, default=128)
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--forge", type=Path, default=FORGE)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    ap.add_argument("--files", default="", help="scan: comma-separated globs of local parquet/jsonl[.gz|.zst] files")
    ap.add_argument("--text", default="text", help="scan: text field (dotted for nested)")
    ap.add_argument("--kind", default="general", choices=["general", "code", "reasoning"])
    ap.add_argument("--id", default="", help="scan: id field for excluded.jsonl (default file:row)")
    ap.add_argument("--licence", default="", help="scan: dataset licence, recorded in the report")
    ap.add_argument("--evals", type=Path, default=None, help="scan: directory of *.jsonl sets to check against")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.stage == "plan":
        stage_plan(a.out, cfg, set(a.tiers.split(",")), set(a.sources.split(",")) - {""} or None, a.pilot_tokens)
    elif a.stage == "filter":
        stage_filter(a.out, a.worker, a.workers, a.min_free_gb)
    elif a.stage == "dedup":
        stage_dedup(a.out, a.partitions)
    elif a.stage == "tokenize":
        stage_tokenize(a.out, a.worker, a.workers, a.forge, a.tokenizer)
    elif a.stage == "pack":
        stage_pack(a.out, a.tokenizer)
    elif a.stage == "status":
        stage_status(a.out)
    elif a.stage == "scan":
        stage_scan(a.out, a.files, a.text, a.kind, a.id, a.licence, a.workers, a.evals)
    else:
        stage_inspect(a.source, cfg)
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":  # stable str hashes: the 13-gram set must agree across processes
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(main())
