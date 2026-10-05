"""Licensed training corpus for Rouge 1, Quasnir and Darus on top of the SCP pipelines.

Uses SCP's own modules (by path, without importing its torch-dependent package
__init__): `bpe.BPETokenizer` for the shared tokenizer, `data_pipeline.clean_corpus`
for quality filtering and exact dedup, `corpora.is_quality_code` for code, and the
SCP CorpusStore layout for the output (`tokenizer.json`, `train.bin`,
`train.val.bin`, `train.meta.json`, `manifest.json`).

Sources (only permissive / public-domain terms; every file records its licence):
  code    CPython 3.11 stdlib (PSF-2.0, local), PyPI sdists with an allow-listed
          licence, npm packages with an allow-listed licence
  general Python PEPs (public domain / CC0-1.0), The Rust Book, Reference, Rust by
          Example and Rustonomicon (MIT OR Apache-2.0), CPython docs (PSF-2.0),
          Node.js API docs (MIT), README/docs files of the allow-listed packages
Evaluation sets (HumanEval, MBPP, GSM8K test) are downloaded for 13-gram
decontamination and kept under data/out/evals/.

Usage: python3 data/build_corpus.py [--vocab 8192] [--workers 4]
Outputs: data/out/{base,general,code}/ (gitignored) and data/manifests/*.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import time
import types
import urllib.request
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"
CACHE = ROOT / "data" / "cache"
MANIFESTS = ROOT / "data" / "manifests"
SCP = Path(os.environ.get("SCP_ROOT", "/home/user/swarm-compute-protocol-")) / "model" / "scp_model"
RAW = "https://raw.githubusercontent.com"

ALLOW = re.compile(r"\b(MIT|BSD|Apache|PSF|Python Software Foundation|ISC|Unlicense|CC0|Public Domain|0BSD|Zlib)\b", re.I)
DENY = re.compile(r"\b(GPL|LGPL|AGPL|MPL|EUPL|SSPL|CC-BY-SA|Commons Clause|proprietary)\b", re.I)

PYPI = """requests urllib3 idna charset-normalizer click flask jinja2 markupsafe itsdangerous werkzeug attrs packaging six
python-dateutil pytz pyyaml toml tomli typing_extensions pluggy pytest iniconfig more-itertools wheel rich pygments
markdown-it-py mdurl httpx httpcore h11 anyio sniffio starlette fastapi pydantic uvicorn websockets multidict yarl
frozenlist aiosignal sqlalchemy alembic mako networkx colorama decorator wrapt cachetools filelock platformdirs
virtualenv distlib tabulate toolz cloudpickle dill msgpack simplejson arrow python-slugify black isort flake8
pyflakes pycodestyle mccabe mypy-extensions coverage mock responses requests-oauthlib oauthlib pyjwt bcrypt passlib
invoke marshmallow jsonschema referencing pyrsistent defusedxml bleach html5lib beautifulsoup4 soupsieve validators
email-validator python-multipart bandit pyopenssl truststore keyring jeepney argon2-cffi sortedcontainers boltons
attrs cattrs structlog loguru tenacity backoff retrying schedule apscheduler croniter humanize inflect natsort
parse regex docopt typer fire pathspec watchdog psutil py-cpuinfo distro tzlocal babel itsdangerous blinker
python-dotenv pyparsing six asgiref django-environ whitenoise gunicorn waitress hypercorn trio outcome
""".split()

NPM = """lodash express react vue axios chalk commander debug uuid ws yargs semver minimist glob mkdirp async dayjs
date-fns underscore zod joi validator helmet cors jsonwebtoken bcryptjs dompurify sanitize-html xss escape-html cookie
qs body-parser morgan dotenv nanoid immer redux acorn esprima marked highlight.js socket.io graphql node-fetch undici
fastify koa hono preact express-rate-limit csrf-csrf ajv yup superstruct pino winston ramda rxjs
""".split()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scp_modules():
    """Load SCP's torch-free modules by path; its package __init__ imports torch."""
    pkg = types.ModuleType("scp_model")
    pkg.__path__ = [str(SCP)]
    sys.modules.setdefault("scp_model", pkg)
    mods = {}
    for name in ("bpe", "data_pipeline", "corpora"):
        spec = importlib.util.spec_from_file_location(f"scp_model.{name}", SCP / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"scp_model.{name}"] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


def fetch(url: str, timeout: int = 60) -> bytes | None:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed https hosts
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:  # noqa: BLE001 - retried, then reported as missing
            time.sleep(1 + attempt)
    return None


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


TEXT_CODE = (".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx")
TEXT_DOC = (".md", ".rst", ".txt")


def doc(text: str, source: str, license_: str, path: str, kind: str) -> dict:
    return {"text": text, "source": source, "license": license_, "path": path, "kind": kind}


def files_from_tar(data: bytes, source: str, license_: str, cap: int = 3_000_000) -> list[dict]:
    out, total = [], 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile() or m.size > 200_000 or "/node_modules/" in m.name:
                    continue
                low = m.name.lower()
                kind = "code" if low.endswith(TEXT_CODE) and not low.endswith((".min.js", ".d.ts")) else "general" if low.endswith(TEXT_DOC) else None
                if kind is None or "/test" in low and kind == "general":
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    text = f.read().decode("utf-8")
                except UnicodeDecodeError:
                    continue
                total += len(text)
                if total > cap:
                    break
                out.append(doc(text, source, license_, m.name, kind))
    except (tarfile.TarError, EOFError, OSError):
        return []
    return out


def pypi_license(data: bytes) -> str | None:
    """Licence from PKG-INFO (License-Expression, License, classifiers)."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            pkg = next((m for m in tf.getmembers() if m.name.count("/") == 1 and m.name.endswith("PKG-INFO")), None)
            if pkg is None:
                return None
            meta = tf.extractfile(pkg).read().decode("utf-8", "replace")
    except (tarfile.TarError, EOFError, OSError):
        return None
    fields = []
    for line in meta.splitlines():
        if line.startswith(("License-Expression:", "License:", "Classifier: License ::")):
            fields.append(line.split(":", 1)[1].strip())
        if not line.strip():
            break
    text = " | ".join(f for f in fields if f and f.upper() != "UNKNOWN")
    return text[:200] or None


def get_pypi(name: str) -> tuple[list[dict], dict]:
    d = CACHE / "pypi"
    d.mkdir(parents=True, exist_ok=True)
    have = list(d.glob(f"{name.replace('-', '_')}-*.tar.gz")) + list(d.glob(f"{name}-*.tar.gz"))
    if not have:
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:", "-q", "-d", str(d), name],
                       capture_output=True, timeout=300)
        have = list(d.glob(f"{name.replace('-', '_')}-*.tar.gz")) + list(d.glob(f"{name}-*.tar.gz"))
    if not have:
        return [], {"source": f"pypi:{name}", "status": "unavailable"}
    data = have[0].read_bytes()
    lic = pypi_license(data)
    entry = {"source": f"pypi:{name}", "file": have[0].name, "sha256": sha(data), "license": lic}
    if not lic or not ALLOW.search(lic) or DENY.search(lic):
        entry["status"] = "rejected-licence"
        return [], entry
    docs = files_from_tar(data, f"pypi:{name}", lic)
    entry.update(status="ok", documents=len(docs))
    return docs, entry


def get_npm(name: str) -> tuple[list[dict], dict]:
    meta = fetch(f"https://registry.npmjs.org/{name}/latest")
    if not meta:
        return [], {"source": f"npm:{name}", "status": "unavailable"}
    m = json.loads(meta)
    lic = m.get("license") if isinstance(m.get("license"), str) else (m.get("license") or {}).get("type")
    entry = {"source": f"npm:{name}", "version": m.get("version"), "license": lic}
    if not lic or not ALLOW.search(lic) or DENY.search(lic) and not ALLOW.search(lic.split(" OR ")[0]):
        entry["status"] = "rejected-licence"
        return [], entry
    tgz = fetch(m["dist"]["tarball"])
    if not tgz:
        entry["status"] = "unavailable"
        return [], entry
    docs = files_from_tar(tgz, f"npm:{name}", lic)
    entry.update(status="ok", sha256=sha(tgz), documents=len(docs))
    return docs, entry


def get_summary_book(repo: str, branch: str, license_: str) -> tuple[list[dict], dict]:
    base = f"{RAW}/{repo}/{branch}/src"
    summary = fetch(f"{base}/SUMMARY.md")
    if not summary:
        return [], {"source": f"github:{repo}", "status": "unavailable"}
    paths = sorted(set(re.findall(r"\]\(\.?/?([^)#\s]+\.md)", summary.decode())))
    with cf.ThreadPoolExecutor(16) as ex:
        texts = list(ex.map(lambda p: fetch(f"{base}/{p}"), paths))
    docs = [doc(t.decode("utf-8", "replace"), f"github:{repo}", license_, p, "general") for p, t in zip(paths, texts) if t]
    return docs, {"source": f"github:{repo}", "branch": branch, "license": license_, "status": "ok", "documents": len(docs)}


def get_peps(limit: int) -> tuple[list[dict], dict]:
    urls = [(n, f"{RAW}/python/peps/main/peps/pep-{n:04d}.rst") for n in range(1, limit + 1)]
    with cf.ThreadPoolExecutor(24) as ex:
        texts = list(ex.map(lambda u: fetch(u[1], timeout=30), urls))
    docs = []
    for (n, _), t in zip(urls, texts):
        if not t:
            continue
        s = t.decode("utf-8", "replace")
        # Only PEPs that state public domain or CC0 terms.
        if re.search(r"public domain|CC0", s, re.I):
            docs.append(doc(s, "github:python/peps", "Public Domain / CC0-1.0", f"pep-{n:04d}.rst", "general"))
    return docs, {"source": "github:python/peps", "license": "Public Domain / CC0-1.0 (per-PEP statement checked)", "status": "ok", "documents": len(docs)}


def get_cpython_docs() -> tuple[list[dict], dict]:
    base = f"{RAW}/python/cpython/3.11/Doc"
    seen, queue, docs = set(), ["tutorial/index", "howto/index", "reference/index", "faq/index", "library/index", "glossary", "extending/index"], []
    depth = {q: 0 for q in queue}
    while queue:
        batch, queue = queue[:32], queue[32:]
        with cf.ThreadPoolExecutor(16) as ex:
            texts = list(ex.map(lambda p: fetch(f"{base}/{p}.rst", timeout=30), batch))
        for p, t in zip(batch, texts):
            seen.add(p)
            if not t:
                continue
            s = t.decode("utf-8", "replace")
            docs.append(doc(s, "github:python/cpython/Doc", "PSF-2.0", f"{p}.rst", "general"))
            if depth[p] >= 2:
                continue
            for block in re.findall(r"\.\. toctree::(.*?)(?:\n\S|\Z)", s, re.S):
                for line in block.splitlines():
                    entry = line.strip()
                    if not entry or entry.startswith(":") or "://" in entry:
                        continue
                    target = os.path.normpath(os.path.join(os.path.dirname(p), entry.replace(".rst", "")))
                    if target not in seen and target not in depth:
                        depth[target] = depth[p] + 1
                        queue.append(target)
    return docs, {"source": "github:python/cpython/Doc@3.11", "license": "PSF-2.0", "status": "ok", "documents": len(docs)}


def get_node_docs() -> tuple[list[dict], dict]:
    base = f"{RAW}/nodejs/node/main/doc/api"
    idx = fetch(f"{base}/index.md")
    if not idx:
        return [], {"source": "github:nodejs/node/doc/api", "status": "unavailable"}
    names = sorted(set(re.findall(r"\]\(([a-z0-9_]+)\.md\)", idx.decode())))
    with cf.ThreadPoolExecutor(16) as ex:
        texts = list(ex.map(lambda n: fetch(f"{base}/{n}.md", timeout=30), names))
    docs = [doc(t.decode("utf-8", "replace"), "github:nodejs/node/doc/api", "MIT", f"{n}.md", "general") for n, t in zip(names, texts) if t]
    return docs, {"source": "github:nodejs/node/doc/api", "license": "MIT", "status": "ok", "documents": len(docs)}


def get_stdlib() -> tuple[list[dict], dict]:
    root = Path("/usr/lib/python3.11")
    docs = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(("site-packages", "dist-packages", "lib2to3/tests", "test/")) or p.stat().st_size > 200_000:
            continue
        try:
            docs.append(doc(p.read_text("utf-8"), "local:cpython-3.11-stdlib", "PSF-2.0", rel, "code"))
        except (UnicodeDecodeError, OSError):
            continue
    return docs, {"source": "local:cpython-3.11-stdlib", "path": str(root), "license": "PSF-2.0", "status": "ok", "documents": len(docs)}


def get_evals() -> dict[str, list[str]]:
    d = OUT / "evals"
    d.mkdir(parents=True, exist_ok=True)
    srcs = {
        "humaneval": (f"{RAW}/openai/human-eval/master/data/HumanEval.jsonl.gz", True),
        "mbpp": (f"{RAW}/google-research/google-research/master/mbpp/mbpp.jsonl", False),
        "gsm8k": (f"{RAW}/openai/grade-school-math/master/grade_school_math/data/test.jsonl", False),
    }
    texts = {}
    for name, (url, gz) in srcs.items():
        path = d / f"{name}.jsonl"
        if not path.exists():
            data = fetch(url)
            if data is None:
                log(f"eval set {name} unavailable")
                continue
            path.write_bytes(gzip.decompress(data) if gz else data)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        texts[name] = [" ".join(str(v) for v in r.values() if isinstance(v, str)) for r in rows]
        log(f"eval set {name}: {len(rows)} items")
    return texts


WORD = re.compile(r"\w+")


def ngrams(text: str, n: int = 13) -> set[int]:
    w = WORD.findall(text.lower())
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


_TOK = None


def _init_tok(path: str) -> None:
    global _TOK
    mods = scp_modules()
    _TOK = mods["bpe"].BPETokenizer.load(path)


def _encode(text: str) -> list[int]:
    return _TOK.encode(text, eos=True)


def write_store(name: str, train_docs: list[dict], val_docs: list[dict], tok_path: Path, vocab: int, workers: int, mix: dict) -> dict:
    import numpy as np

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokenizer.json").write_bytes(tok_path.read_bytes())
    counts = {}
    with Pool(workers, initializer=_init_tok, initargs=(str(tok_path),)) as pool:
        for split, docs, fname in (("train", train_docs, "train.bin"), ("val", val_docs, "train.val.bin")):
            n = 0
            with open(d / fname, "wb") as f:
                for ids in pool.imap(_encode, (x["text"] for x in docs), chunksize=16):
                    np.asarray(ids, dtype=np.uint16).tofile(f)
                    n += len(ids)
            counts[split] = n
    meta = {"tokens": counts["train"], "val_tokens": counts["val"], "dtype": "uint16", "vocab_size": vocab,
            "bin": "train.bin", "val_bin": "train.val.bin", "mix": mix,
            "mix_fractions": {k: v / max(1, sum(mix.values())) for k, v in mix.items()}}
    (d / "train.meta.json").write_text(json.dumps(meta, indent=2))
    manifest = {"store": name, "documents": {"train": len(train_docs), "val": len(val_docs)}, **meta,
                "sha256": {"train.bin": sha((d / "train.bin").read_bytes()), "train.val.bin": sha((d / "train.val.bin").read_bytes()),
                           "tokenizer.json": sha(tok_path.read_bytes())},
                "pipeline": "SCP scp_model.bpe + data_pipeline.clean_corpus + CorpusStore layout; per-source val split; 13-gram decontamination"}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--peps", type=int, default=760)
    ap.add_argument("--tok-sample-mb", type=float, default=2.5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    mods = scp_modules()
    t0 = time.time()

    jobs = {"stdlib": get_stdlib, "peps": lambda: get_peps(args.peps), "cpython-docs": get_cpython_docs, "node-docs": get_node_docs,
            "rust-book": lambda: get_summary_book("rust-lang/book", "main", "MIT OR Apache-2.0"),
            "rust-reference": lambda: get_summary_book("rust-lang/reference", "master", "MIT OR Apache-2.0"),
            "rust-by-example": lambda: get_summary_book("rust-lang/rust-by-example", "master", "MIT OR Apache-2.0"),
            "nomicon": lambda: get_summary_book("rust-lang/nomicon", "master", "MIT OR Apache-2.0")}
    jobs.update({f"pypi:{n}": (lambda n=n: get_pypi(n)) for n in dict.fromkeys(PYPI)})
    jobs.update({f"npm:{n}": (lambda n=n: get_npm(n)) for n in dict.fromkeys(NPM)})
    docs, sources = [], []
    with cf.ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(f): k for k, f in jobs.items()}
        for fut in cf.as_completed(futs):
            try:
                ds, entry = fut.result()
            except Exception as e:  # noqa: BLE001 - a failed source is recorded, not fatal
                ds, entry = [], {"source": futs[fut], "status": f"error: {e}"}
            docs.extend(ds)
            sources.append(entry)
    log(f"fetched {len(docs)} files from {sum(1 for s in sources if s.get('status') == 'ok')} sources in {time.time() - t0:.0f}s")

    evals = get_evals()
    banned: set[int] = set()
    for items in evals.values():
        for t in items:
            banned |= ngrams(t)

    kept = {"code": [], "general": []}
    stats = {"dropped_quality": 0, "dropped_duplicate": 0, "dropped_contaminated": 0}
    seen: set[str] = set()
    norm = re.compile(r"\s+")
    for x in docs:
        text = x["text"].strip()
        if not text:
            continue
        key = hashlib.sha1(norm.sub(" ", text).encode()).hexdigest()
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        ok = mods["corpora"].is_quality_code(text) if x["kind"] == "code" else mods["data_pipeline"].is_quality(text, min_chars=200)
        if not ok:
            stats["dropped_quality"] += 1
            continue
        if banned and ngrams(text) & banned:
            stats["dropped_contaminated"] += 1
            continue
        kept[x["kind"]].append({**x, "text": text, "sha1": key})
    log(f"kept code={len(kept['code'])} general={len(kept['general'])} {stats}")

    # Per-source validation split (~1 %, at least one document per source with >= 5 docs).
    splits = {}
    for kind, items in kept.items():
        by_src: dict[str, list[dict]] = {}
        for x in sorted(items, key=lambda x: x["sha1"]):
            by_src.setdefault(x["source"], []).append(x)
        tr, va = [], []
        for src, xs in by_src.items():
            k = max(1, len(xs) // 100) if len(xs) >= 5 else 0
            va.extend(xs[:k])
            tr.extend(xs[k:])
        random.Random(67).shuffle(tr)
        splits[kind] = (tr, va)

    # Shared tokenizer from a balanced sample (SCP BPE, vocab incl. 256 bytes + 3 specials).
    rnd = random.Random(1417)
    sample, budget = [], int(args.tok_sample_mb * 1e6 / 2)
    for kind in ("general", "code"):
        pool_ = splits[kind][0][:]
        rnd.shuffle(pool_)
        size = 0
        for x in pool_:
            sample.append(x["text"][:20_000])
            size += len(sample[-1])
            if size >= budget:
                break
    tok_path = OUT / "tokenizer.json"
    if not tok_path.exists():
        log(f"training SCP BPE on {sum(map(len, sample)) / 1e6:.1f} MB, vocab {args.vocab}")
        tok = mods["bpe"].BPETokenizer()
        tok.train(sample, args.vocab)
        tok.save(str(tok_path))
    log("tokenizer ready")

    manifests = {}
    for kind in ("general", "code"):
        tr, va = splits[kind]
        manifests[kind] = write_store(kind, tr, va, tok_path, args.vocab, args.workers, {kind: len(tr)})
        log(f"store {kind}: {manifests[kind]['tokens']} train / {manifests[kind]['val_tokens']} val tokens")
    base_tr = splits["general"][0] + splits["code"][0]
    random.Random(7).shuffle(base_tr)
    base_va = splits["general"][1] + splits["code"][1]
    manifests["base"] = write_store("base", base_tr, base_va, tok_path, args.vocab, args.workers,
                                    {"general": len(splits["general"][0]), "code": len(splits["code"][0])})
    log(f"store base: {manifests['base']['tokens']} train / {manifests['base']['val_tokens']} val tokens")

    (MANIFESTS / "sources.json").write_text(json.dumps(sorted(sources, key=lambda s: s["source"]), indent=2))
    (MANIFESTS / "stores.json").write_text(json.dumps({"filters": stats, "stores": manifests,
                                                       "evals_decontaminated_against": sorted(evals)}, indent=2))
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
