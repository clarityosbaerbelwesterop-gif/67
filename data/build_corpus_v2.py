"""Corpus v2: licensed code and technical prose at the scale multi-day training
needs (hundreds of millions of tokens), from the package registries this
environment can reach: PyPI (top packages), npm (most popular), crates.io
(curated popular crates) and the Go module proxy (curated modules).

Same rules as data/build_corpus.py, whose helpers it reuses:
  * licence allow-list only (MIT, BSD, Apache, PSF, ISC, Unlicense, CC0, Zlib,
    0BSD, public domain); GPL/LGPL/AGPL/MPL/EUPL/SSPL/CC-BY-SA/unknown are dropped,
    and every package's licence lands in data/manifests/v2/sources.json;
  * quality filters from SCP (corpora.is_quality_code, data_pipeline.is_quality);
  * exact deduplication across all sources (sha1 of whitespace-normalised text);
  * 13-gram decontamination against HumanEval, MBPP and GSM8K;
  * ~1 % validation split chosen by document hash (stable across reruns).

Streaming and resumable: every package is fetched, filtered and appended to
data/cache/v2/docs/*.jsonl once; data/cache/v2/done.jsonl records it, so a rerun
continues. Tokenisation uses the Rust tokenizer (`forge tokenize`, identical to
scp_model.bpe), with <eos> after every document, into SCP stores
data/out/{code-v2,general-v2,base-v2}/ that share data/stores/base/tokenizer.json.

Usage:
  python3 data/build_corpus_v2.py [--target-tokens 400e6] [--workers 6]
  python3 data/build_corpus_v2.py --small            # a few packages, for tests
  python3 data/build_corpus_v2.py --tokenize-only    # rebuild stores from cached docs
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.parse
import zipfile
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "v2"
DOCS = CACHE / "docs"
OUT = ROOT / "data" / "out"
MANIFESTS = ROOT / "data" / "manifests" / "v2"
TOKENIZER = ROOT / "data" / "stores" / "base" / "tokenizer.json"
FORGE = ROOT / "target" / "release" / "forge"
EOS, VOCAB = 8190, 8192
CHARS_PER_TOKEN = 3.3  # measured on the v1 stores; only used to stop fetching

_spec = importlib.util.spec_from_file_location("build_corpus_v1", ROOT / "data" / "build_corpus.py")
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)
ALLOW, DENY = bc.ALLOW, bc.DENY

CODE_EXT = (".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".rs", ".go")
DOC_EXT = (".md", ".rst", ".txt")
SKIP = re.compile(r"(^|/)(node_modules|vendor|third_party|testdata|dist|build|\.git)/|\.min\.js$|\.d\.ts$|\.pb\.go$|_pb2\.py$")
MAX_FILE, MAX_PACKAGE = 200_000, 4_000_000

CRATES = """serde serde_json serde_derive tokio rand regex clap anyhow thiserror log env_logger tracing tracing-subscriber
futures futures-util bytes itertools once_cell lazy_static chrono time uuid hashbrown indexmap smallvec parking_lot
crossbeam crossbeam-channel rayon num num-traits num-bigint libc memchr byteorder base64 hex sha2 md-5 digest
hmac rand_core rand_chacha getrandom url percent-encoding http hyper reqwest h2 tower tower-http axum actix-web
warp rustls ring webpki tokio-util tokio-stream async-trait pin-project pin-project-lite mio socket2 bitflags
cfg-if either semver toml toml_edit serde_yaml csv walkdir glob tempfile dirs which shlex nom pest syn quote
proc-macro2 heck strum strum_macros derive_more paste darling ahash fnv siphasher arrayvec tinyvec bstr unicode-width
unicode-segmentation unicode-normalization textwrap termcolor colored indicatif console dialoguer crossterm ratatui
criterion proptest quickcheck insta pretty_assertions assert_cmd predicates mockall approx ndarray nalgebra glam
image png flate2 zstd lz4_flex bzip2 tar zip miniz_oxide crc32fast adler memmap2 bincode postcard rmp-serde prost
tonic bytemuck zerocopy static_assertions scopeguard slab dashmap arc-swap atomic-waker event-listener async-channel
async-std smol rustix nix signal-hook ctrlc humantime humansize bumpalo typed-arena petgraph rustc-hash im
fixedbitset bit-vec roaring sqlx rusqlite diesel redis mongodb postgres tokio-postgres jsonwebtoken argon2 bcrypt
aes chacha20poly1305 ed25519-dalek x25519-dalek curve25519-dalek rsa p256 sha1 sha3 blake3 subtle zeroize
config dotenvy envy structopt argh pico-args lexopt tera handlebars askama minijinja pulldown-cmark comrak
syntect similar diff unicode-ident ryu itoa lexical-core fast-float encoding_rs html5ever scraper select
quick-xml roxmltree xml-rs serde_with serde_bytes serde_repr schemars validator governor moka lru cached
wasm-bindgen js-sys web-sys getopts difference ansi_term atty is-terminal""".split()

GO = """github.com/spf13/cobra github.com/spf13/pflag github.com/spf13/viper github.com/sirupsen/logrus
go.uber.org/zap go.uber.org/multierr go.uber.org/atomic github.com/stretchr/testify github.com/google/uuid
github.com/google/go-cmp github.com/pkg/errors github.com/gorilla/mux github.com/gorilla/websocket
github.com/gin-gonic/gin github.com/labstack/echo/v4 github.com/go-chi/chi/v5 github.com/gofiber/fiber/v2
github.com/valyala/fasthttp github.com/urfave/cli/v2 github.com/fatih/color github.com/mattn/go-isatty
github.com/mattn/go-colorable github.com/mattn/go-runewidth github.com/charmbracelet/bubbletea
github.com/charmbracelet/lipgloss github.com/rivo/tview github.com/gdamore/tcell/v2 github.com/BurntSushi/toml
gopkg.in/yaml.v3 github.com/json-iterator/go github.com/tidwall/gjson github.com/tidwall/sjson
github.com/mitchellh/mapstructure github.com/hashicorp/go-multierror github.com/hashicorp/golang-lru/v2
github.com/patrickmn/go-cache github.com/dgraph-io/ristretto github.com/redis/go-redis/v9
github.com/jackc/pgx/v5 github.com/lib/pq github.com/go-sql-driver/mysql github.com/mattn/go-sqlite3
github.com/jmoiron/sqlx gorm.io/gorm github.com/golang-jwt/jwt/v5 golang.org/x/crypto golang.org/x/net
golang.org/x/sync golang.org/x/text golang.org/x/sys golang.org/x/exp golang.org/x/time golang.org/x/oauth2
google.golang.org/protobuf google.golang.org/grpc github.com/golang/protobuf github.com/prometheus/client_golang
github.com/prometheus/common github.com/rs/zerolog github.com/rs/cors github.com/cenkalti/backoff/v4
github.com/avast/retry-go/v4 github.com/robfig/cron/v3 github.com/go-playground/validator/v10
github.com/schollz/progressbar/v3 github.com/olekukonko/tablewriter github.com/dustin/go-humanize
github.com/shopspring/decimal github.com/klauspost/compress github.com/pierrec/lz4/v4 github.com/golang/snappy
github.com/cespare/xxhash/v2 github.com/vmihailenco/msgpack/v5 github.com/fsnotify/fsnotify
github.com/alecthomas/kong github.com/alecthomas/chroma/v2 github.com/yuin/goldmark github.com/gomarkdown/markdown
github.com/PuerkitoBio/goquery github.com/andybalholm/cascadia github.com/samber/lo github.com/samber/mo
github.com/kelseyhightower/envconfig github.com/joho/godotenv github.com/caarlos0/env/v10
github.com/nats-io/nats.go github.com/segmentio/kafka-go github.com/IBM/sarama github.com/minio/minio-go/v7
github.com/aws/aws-sdk-go-v2 github.com/google/go-github/v62 github.com/sourcegraph/conc github.com/oklog/run
github.com/benbjohnson/clock github.com/jonboulle/clockwork github.com/hashicorp/raft github.com/etcd-io/bbolt
go.etcd.io/bbolt github.com/dgraph-io/badger/v4 github.com/blevesearch/bleve/v2 github.com/expr-lang/expr
github.com/google/cel-go github.com/itchyny/gojq github.com/antonmedv/expr github.com/Masterminds/semver/v3
github.com/Masterminds/sprig/v3 github.com/gobwas/glob github.com/bmatcuk/doublestar/v4 github.com/otiai10/copy""".split()

NPM_QUERIES = ["keywords:javascript", "keywords:typescript", "keywords:node", "keywords:cli", "keywords:utility",
               "keywords:react", "keywords:parser", "keywords:http", "keywords:testing", "keywords:stream",
               "keywords:promise", "keywords:string", "keywords:array", "keywords:date", "keywords:validation",
               "keywords:async", "keywords:json", "keywords:css", "keywords:compiler", "keywords:framework"]


def log(msg: str) -> None:
    print(f"[corpus-v2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(url: str, timeout: int = 90) -> bytes | None:
    return bc.fetch(url, timeout)


def licence_ok(lic: str | None) -> bool:
    return bool(lic) and bool(ALLOW.search(lic)) and not DENY.search(lic)


def classify_licence_text(text: str) -> str | None:
    t = text[:6000]
    if re.search(r"GNU (Affero |Lesser )?General Public License|Mozilla Public License|Creative Commons Attribution-ShareAlike", t):
        return None
    if "Apache License" in t:
        return "Apache-2.0"
    if "Permission is hereby granted, free of charge" in t:
        return "MIT"
    if "Redistribution and use in source and binary forms" in t:
        return "BSD"
    if "ISC License" in t or "Permission to use, copy, modify, and/or distribute this software" in t:
        return "ISC"
    if "This is free and unencumbered software released into the public domain" in t:
        return "Unlicense"
    return None


def kind_of(path: str) -> str | None:
    low = path.lower()
    if SKIP.search(low):
        return None
    if low.endswith(CODE_EXT):
        return "code"
    if low.endswith(DOC_EXT) and "/test" not in low and "license" not in low.rsplit("/", 1)[-1] and "changelog" not in low:
        return "general"
    return None


def extract(members) -> list[tuple[str, str, str]]:
    """(path, kind, text) of the useful text files of an archive, size-capped."""
    out, total = [], 0
    for name, size, read in members:
        kind = kind_of(name)
        if kind is None or size > MAX_FILE:
            continue
        try:
            text = read().decode("utf-8")
        except (UnicodeDecodeError, OSError, KeyError, tarfile.TarError, zipfile.BadZipFile):
            continue
        total += len(text)
        if total > MAX_PACKAGE:
            break
        out.append((name, kind, text))
    return out


def tar_members(data: bytes):
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            for m in tf.getmembers():
                if m.isfile():
                    yield m.name, m.size, (lambda m=m: tf.extractfile(m).read())
    except (tarfile.TarError, EOFError, OSError):
        return


def zip_members(data: bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for i in zf.infolist():
                if not i.is_dir():
                    yield i.filename, i.file_size, (lambda i=i: zf.read(i))
    except (zipfile.BadZipFile, OSError):
        return


# ---------------------------------------------------------------- sources


def pypi(name: str) -> tuple[dict, list]:
    entry = {"source": f"pypi:{name}"}
    meta = fetch(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
    if not meta:
        return entry | {"status": "unavailable"}, []
    m = json.loads(meta)
    info = m.get("info") or {}
    lic = " | ".join(x for x in [info.get("license_expression") or "", (info.get("license") or "")[:300]] +
                     [c for c in info.get("classifiers") or [] if c.startswith("License ::")] if x and x.upper() != "UNKNOWN")
    sdist = next((u for u in m.get("urls") or [] if u.get("packagetype") == "sdist"), None)
    entry.update(version=info.get("version"), license=lic[:300] or None)
    if not licence_ok(lic):
        return entry | {"status": "rejected-licence"}, []
    if not sdist or sdist.get("size", 0) > 60_000_000:
        return entry | {"status": "no-sdist"}, []
    data = fetch(sdist["url"])
    if not data:
        return entry | {"status": "unavailable"}, []
    return entry | {"sha256": hashlib.sha256(data).hexdigest()}, extract(tar_members(data) if not sdist["url"].endswith(".zip") else zip_members(data))


def npm(name: str) -> tuple[dict, list]:
    entry = {"source": f"npm:{name}"}
    meta = fetch(f"https://registry.npmjs.org/{name.replace('/', '%2F')}/latest")
    if not meta:
        return entry | {"status": "unavailable"}, []
    m = json.loads(meta)
    lic = m.get("license") if isinstance(m.get("license"), str) else (m.get("license") or {}).get("type")
    entry.update(version=m.get("version"), license=lic)
    if not licence_ok(lic):
        return entry | {"status": "rejected-licence"}, []
    data = fetch(m.get("dist", {}).get("tarball", ""))
    if not data:
        return entry | {"status": "unavailable"}, []
    return entry | {"sha256": hashlib.sha256(data).hexdigest()}, extract(tar_members(data))


def crate_index_path(n: str) -> str:
    n = n.lower()
    return {1: f"1/{n}", 2: f"2/{n}", 3: f"3/{n[0]}/{n}"}.get(len(n), f"{n[:2]}/{n[2:4]}/{n}")


def crate(name: str) -> tuple[dict, list]:
    entry = {"source": f"crates:{name}"}
    idx = fetch(f"https://index.crates.io/{crate_index_path(name)}")
    if not idx:
        return entry | {"status": "unavailable"}, []
    lines = [json.loads(x) for x in idx.decode().splitlines() if x.strip()]
    rel = [x for x in lines if not x.get("yanked") and "-" not in x["vers"]] or lines
    vers = rel[-1]["vers"]
    data = fetch(f"https://static.crates.io/crates/{name}/{name}-{vers}.crate")
    if not data:
        return entry | {"version": vers, "status": "unavailable"}, []
    lic = None
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            cargo = tf.extractfile(f"{name}-{vers}/Cargo.toml")
            lic = (tomllib.loads(cargo.read().decode()).get("package") or {}).get("license") if cargo else None
    except (tarfile.TarError, KeyError, tomllib.TOMLDecodeError, OSError, UnicodeDecodeError):
        pass
    entry.update(version=vers, license=lic, sha256=hashlib.sha256(data).hexdigest())
    if not licence_ok(lic):
        return entry | {"status": "rejected-licence"}, []
    return entry, extract(tar_members(data))


def gomod(module: str) -> tuple[dict, list]:
    entry = {"source": f"go:{module}"}
    esc = re.sub(r"[A-Z]", lambda m: "!" + m.group(0).lower(), module)
    latest = fetch(f"https://proxy.golang.org/{esc}/@latest")
    if not latest:
        return entry | {"status": "unavailable"}, []
    vers = json.loads(latest)["Version"]
    data = fetch(f"https://proxy.golang.org/{esc}/@v/{vers}.zip")
    if not data:
        return entry | {"version": vers, "status": "unavailable"}, []
    lic = None
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            root = f"{module}@{vers}/"
            lf = next((n for n in zf.namelist() if n.startswith(root) and n[len(root):].upper().startswith(("LICENSE", "LICENCE", "COPYING")) and "/" not in n[len(root):]), None)
            lic = classify_licence_text(zf.read(lf).decode("utf-8", "replace")) if lf else None
    except (zipfile.BadZipFile, OSError):
        pass
    entry.update(version=vers, license=lic, sha256=hashlib.sha256(data).hexdigest())
    if not licence_ok(lic):
        return entry | {"status": "rejected-licence"}, []
    return entry, extract(zip_members(data))


def pypi_names(n: int) -> list[str]:
    raw = fetch("https://raw.githubusercontent.com/hugovk/top-pypi-packages/main/top-pypi-packages.min.json")
    if raw:
        return [r["project"] for r in json.loads(raw)["rows"][:n]]
    log("top-pypi list unavailable; using the v1 list")
    return bc.PYPI.split()[:n]


def npm_names(n: int) -> list[str]:
    names: dict[str, None] = {}
    for q in NPM_QUERIES:
        for start in range(0, 1000, 250):
            raw = fetch(f"https://registry.npmjs.org/-/v1/search?text={urllib.parse.quote(q)}&size=250&from={start}&popularity=1.0&quality=0.0&maintenance=0.0")
            if not raw:
                break
            objs = json.loads(raw).get("objects", [])
            names.update({o["package"]["name"]: None for o in objs})
            if len(objs) < 250 or len(names) >= n:
                break
        if len(names) >= n:
            break
    return list(names)[:n]


# ---------------------------------------------------------------- filtering

_MODS = None
_BANNED: set[int] = set()


def _init(banned: set[int]) -> None:
    global _MODS, _BANNED
    _BANNED = banned
    try:
        _MODS = bc.scp_modules()
    except Exception:  # noqa: BLE001 - SCP checkout absent: fall back to simple heuristics
        _MODS = None
    os.nice(15)


def quality(text: str, kind: str) -> bool:
    if _MODS is not None:
        return _MODS["corpora"].is_quality_code(text) if kind == "code" else _MODS["data_pipeline"].is_quality(text, min_chars=200)
    lines = text.splitlines()
    return len(text) >= 200 and max(map(len, lines), default=0) < 1000 and sum(map(len, lines)) / max(1, len(lines)) < 120


NORM = re.compile(r"\s+")


def work(task: tuple[str, str]) -> tuple[dict, list[dict], dict]:
    kind_src, name = task
    fn = {"pypi": pypi, "npm": npm, "crates": crate, "go": gomod}[kind_src]
    stats = {"quality": 0, "contaminated": 0}
    try:
        entry, files = fn(name)
    except Exception as e:  # noqa: BLE001 - a broken package is recorded, never fatal
        return {"source": f"{kind_src}:{name}", "status": f"error: {type(e).__name__}: {e}"[:200]}, [], stats
    docs = []
    for path, kind, text in files:
        text = text.strip()
        if not text or not quality(text, kind):
            stats["quality"] += 1
            continue
        if _BANNED and bc.ngrams(text) & _BANNED:
            stats["contaminated"] += 1
            continue
        h = hashlib.sha1(NORM.sub(" ", text).encode()).hexdigest()
        docs.append({"h": h, "k": kind, "s": entry["source"], "p": path, "t": text})
    entry.setdefault("status", "ok")
    entry["files"] = len(files)
    return entry, docs, stats


# ---------------------------------------------------------------- stores


def load_seen() -> tuple[set[str], int]:
    seen, chars = set(), 0
    for f in DOCS.glob("*.jsonl"):
        with f.open() as fh:
            for line in fh:
                d = json.loads(line)
                seen.add(d["h"])
                chars += len(d["t"])
    return seen, chars


def tokenize_into(src: Path, out, chunk_docs: int = 50_000) -> tuple[int, int]:
    """Append the tokens (+ eos) of every document of `src` to the open binary `out`."""
    tokens = docs = 0
    tmp = CACHE / "tok-chunk.jsonl"
    with src.open() as fh:
        while True:
            batch = [json.loads(line)["t"] for _, line in zip(range(chunk_docs), fh)]
            if not batch:
                break
            tmp.write_text("".join(json.dumps(t) + "\n" for t in batch))
            res = subprocess.run(["nice", "-n", "15", str(FORGE), "tokenize", "--tokenizer", str(TOKENIZER), "--jsonl", str(tmp)],
                                 check=True, capture_output=True, text=True).stdout
            buf = array("H")
            for line in res.splitlines():
                ids = json.loads(line)
                buf.extend(ids)
                buf.append(EOS)
                tokens += len(ids) + 1
                docs += 1
            if sys.byteorder != "little":
                buf.byteswap()
            buf.tofile(out)
    tmp.unlink(missing_ok=True)
    return tokens, docs


def write_stores() -> dict:
    stores = {}
    parts = {k: {s: DOCS / f"{k}-{s}.jsonl" for s in ("train", "val")} for k in ("code", "general")}
    for kind in ("code", "general"):
        d = OUT / f"{kind}-v2"
        d.mkdir(parents=True, exist_ok=True)
        counts = {}
        for split, fname in (("train", "train.bin"), ("val", "train.val.bin")):
            with open(d / fname, "wb") as out:
                counts[split] = tokenize_into(parts[kind][split], out) if parts[kind][split].exists() else (0, 0)
        stores[kind] = finish_store(d, counts["train"][0], counts["val"][0], {kind: counts["train"][1]})
        log(f"store {kind}-v2: {counts['train'][0]:,} train / {counts['val'][0]:,} val tokens")
    d = OUT / "base-v2"
    d.mkdir(parents=True, exist_ok=True)
    for fname in ("train.bin", "train.val.bin"):
        with open(d / fname, "wb") as out:
            for kind in ("code", "general"):
                with open(OUT / f"{kind}-v2" / fname, "rb") as src:
                    while chunk := src.read(1 << 24):
                        out.write(chunk)
    stores["base"] = finish_store(d, stores["code"]["tokens"] + stores["general"]["tokens"],
                                  stores["code"]["val_tokens"] + stores["general"]["val_tokens"],
                                  {"code": stores["code"]["mix"]["code"], "general": stores["general"]["mix"]["general"]})
    log(f"store base-v2: {stores['base']['tokens']:,} train / {stores['base']['val_tokens']:,} val tokens")
    return stores


def finish_store(d: Path, tokens: int, val_tokens: int, mix: dict) -> dict:
    (d / "tokenizer.json").write_bytes(TOKENIZER.read_bytes())
    meta = {"tokens": tokens, "val_tokens": val_tokens, "dtype": "uint16", "vocab_size": VOCAB, "bin": "train.bin",
            "val_bin": "train.val.bin", "mix": mix, "mix_fractions": {k: v / max(1, sum(mix.values())) for k, v in mix.items()}}
    (d / "train.meta.json").write_text(json.dumps(meta, indent=2))

    def sha_file(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        return h.hexdigest()

    manifest = {"store": d.name, **meta, "sha256": {n: sha_file(d / n) for n in ("train.bin", "train.val.bin", "tokenizer.json")},
                "pipeline": "data/build_corpus_v2.py: licence allow-list, SCP quality filters, exact dedup, 13-gram decontamination "
                            "(HumanEval, MBPP, GSM8K), hash-based ~1% val split, forge tokenize (= scp_model.bpe), eos per document"}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=float, default=400e6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-pypi", type=int, default=6000)
    ap.add_argument("--max-npm", type=int, default=3000)
    ap.add_argument("--small", action="store_true", help="a handful of packages (tests)")
    ap.add_argument("--tokenize-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", type=Path, default=None, help="write cache, stores and manifests under this directory (tests)")
    a = ap.parse_args()
    if a.root is not None:
        global CACHE, DOCS, OUT, MANIFESTS
        CACHE, OUT, MANIFESTS = a.root / "cache", a.root / "out", a.root / "manifests"
        DOCS = CACHE / "docs"
    for p in (DOCS, MANIFESTS):
        p.mkdir(parents=True, exist_ok=True)
    if not FORGE.exists() or not TOKENIZER.exists():
        log(f"need {FORGE} and {TOKENIZER}")
        return 2
    t0 = time.time()
    done_path = CACHE / "done.jsonl"
    done = {json.loads(x)["task"] for x in done_path.read_text().splitlines()} if done_path.exists() else set()

    if not a.tokenize_only:
        if a.small:
            tasks = [("pypi", "six"), ("pypi", "idna"), ("npm", "ms"), ("crates", "itoa"), ("go", "github.com/google/uuid")]
            a.target_tokens = min(a.target_tokens, 2e6)
        else:
            # Interleave registries so a partial build is still diverse.
            lists = [[("pypi", n) for n in pypi_names(a.max_pypi)], [("npm", n) for n in npm_names(a.max_npm)],
                     [("crates", n) for n in CRATES], [("go", m) for m in GO]]
            tasks, i = [], 0
            while any(i < len(x) for x in lists):
                tasks += [x[i] for x in lists if i < len(x)]
                i += 1
        todo = [t for t in tasks if f"{t[0]}:{t[1]}" not in done]
        log(f"{len(tasks)} packages planned, {len(tasks) - len(todo)} already done")
        if a.dry_run:
            print(json.dumps({"planned": len(tasks), "todo": len(todo), "by_registry": {r: sum(1 for t in tasks if t[0] == r) for r in ("pypi", "npm", "crates", "go")}}))
            return 0
        banned: set[int] = set()
        bc.get_evals()  # downloads the eval sets if missing
        for f in sorted((bc.OUT / "evals").glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    vals = [v for v in row.values() if isinstance(v, str)]
                    vals += [x for v in row.values() if isinstance(v, list) for x in v if isinstance(x, str)]
                    banned |= bc.ngrams(" ".join(vals))
        seen, chars = load_seen()
        log(f"resuming with {len(seen):,} documents, ~{chars / CHARS_PER_TOKEN / 1e6:.1f} M tokens; {len(banned):,} banned 13-grams")
        stats = {"duplicate": 0, "quality": 0, "contaminated": 0}
        handles = {(k, s): (DOCS / f"{k}-{s}.jsonl").open("a") for k in ("code", "general") for s in ("train", "val")}
        with mp.Pool(a.workers, initializer=_init, initargs=(banned,)) as pool, done_path.open("a") as donef:
            for n, (entry, docs, st) in enumerate(pool.imap_unordered(work, todo, chunksize=1), 1):
                stats["quality"] += st["quality"]
                stats["contaminated"] += st["contaminated"]
                kept = 0
                for d in docs:
                    if d["h"] in seen:
                        stats["duplicate"] += 1
                        continue
                    seen.add(d["h"])
                    split = "val" if int(d["h"][:8], 16) % 100 == 0 else "train"
                    handles[(d["k"], split)].write(json.dumps(d) + "\n")
                    chars += len(d["t"])
                    kept += 1
                entry["kept"] = kept
                donef.write(json.dumps({"task": entry["source"], **{k: v for k, v in entry.items() if k != "source"}}) + "\n")
                donef.flush()
                if n % 25 == 0:
                    log(f"{n}/{len(todo)} packages, ~{chars / CHARS_PER_TOKEN / 1e6:.1f} M tokens, {len(seen):,} docs, filtered {stats}")
                if chars / CHARS_PER_TOKEN >= a.target_tokens:
                    log(f"target of {a.target_tokens / 1e6:.0f} M tokens reached")
                    pool.terminate()
                    break
        for h in handles.values():
            h.close()
        (MANIFESTS / "filters.json").write_text(json.dumps(stats, indent=2))

    sources = [json.loads(x) for x in done_path.read_text().splitlines()] if done_path.exists() else []
    (MANIFESTS / "sources.json").write_text(json.dumps(sorted(sources, key=lambda s: s["task"]), indent=1))
    stores = write_stores()
    by_status: dict[str, int] = {}
    for s in sources:
        by_status[s.get("status", "?").split(":")[0]] = by_status.get(s.get("status", "?").split(":")[0], 0) + 1
    (MANIFESTS / "stores.json").write_text(json.dumps({"stores": stores, "packages": by_status,
                                                        "evals_decontaminated_against": ["humaneval", "mbpp", "gsm8k"]}, indent=2))
    log(f"done in {time.time() - t0:.0f}s; packages {by_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
