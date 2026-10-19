"""Corpus v6 end to end on local sources: licence tiers, content rules, decontamination, exact and near
dedup, tokenization and a store the trainer's CleanStore contract accepts."""
import gzip
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "data"))
v6 = importlib.import_module("build_corpus_v6")

pytestmark = pytest.mark.skipif(not (v6.EVALS / "humaneval.jsonl").exists() or not v6.FORGE.exists(),
                                reason="needs data/out/evals and target/release/forge")

PY = '''"""Small statistics helpers number {i} used by the report module."""
from __future__ import annotations

import math


def mean_{i}(values: list[float]) -> float:
    """Arithmetic mean; raises on an empty list."""
    if not values:
        raise ValueError("mean of an empty list")
    return sum(values) / len(values)


def stdev_{i}(values: list[float]) -> float:
    """Sample standard deviation (n - 1 in the denominator) for case {i}."""
    m = mean_{i}(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


class Running{i}:
    """Welford's online mean and variance, variant {i}."""

    def __init__(self) -> None:
        self.n, self.m, self.s = 0, 0.0, 0.0

    def push(self, x: float) -> None:
        self.n += 1
        d = x - self.m
        self.m += d / self.n
        self.s += d * (x - self.m)
'''
PROSE = ["argparse", "json", "subprocess", "csv", "heapq", "unittest", "statistics", "shelve", "sched", "socket", "ssl"]


def _write_sources(tmp: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    he = json.loads((v6.EVALS / "humaneval.jsonl").read_text().splitlines()[3])
    leak = next(v for v in he.values() if isinstance(v, str) and len(v.split()) > 30)
    code = [PY.format(i=i) for i in range(6)]
    rows = [(code[0], "pkg/a.py", ["MIT"]), (code[1], "pkg/b.py", ["Apache-2.0"]), (code[2], "pkg/c.py", ["GPL-3.0"]),
            (code[0], "copy/a.py", ["MIT"]),                                             # exact duplicate
            (code[3].replace("raises on an empty list", "raises if the list is empty"), "pkg/d.py", ["BSD-3-Clause"]),
            (code[3], "pkg/e.py", ["MIT"]),                                              # near duplicate of the line above
            (code[4] + "\n" + "\n".join("# " + ln for ln in leak.splitlines()) + "\n", "pkg/f.py", ["MIT"]),  # leak
            (code[5], "vendor/x/g.py", ["MIT"])]                                         # vendored path
    (tmp / "code").mkdir()
    pq.write_table(pa.table({"content": [r[0] for r in rows], "max_stars_repo_path": [r[1] for r in rows],
                             "max_stars_repo_licenses": [r[2] for r in rows]}), tmp / "code" / "part-0.parquet")
    (tmp / "text").mkdir()
    lic = ["CC BY 4.0", "CC0", "CC BY-SA 4.0", "CC BY-NC 4.0", "Public Domain", "CC-BY-4.0", "unknown", "CC BY 4.0",
           "CC BY 4.0", "CC BY 4.0", "CC BY 4.0"]
    with gzip.open(tmp / "text" / "part-0.jsonl.gz", "wt") as f:
        for m, li in zip(PROSE, lic):
            f.write(json.dumps({"text": inspect.getdoc(importlib.import_module(m)), "metadata": {"license": li}}) + "\n")
    cfg = {"sources": [
        {"name": "code", "local": str(tmp / "code" / "*.parquet"), "kind": "code", "tier": "A", "text": "content",
         "path": "max_stars_repo_path", "licence": "max_stars_repo_licenses", "priority": 2},
        {"name": "text", "local": str(tmp / "text" / "*.jsonl.gz"), "kind": "general", "tier": "A", "text": "text",
         "licence": "metadata.license", "priority": 1}]}
    (tmp / "cfg.json").write_text(json.dumps(cfg))
    return tmp / "cfg.json"


def test_licence_tiers():
    assert v6.licence_tier("CC BY 4.0", "general") == "A"
    assert v6.licence_tier("cc-by-4.0", "general") == "A"
    assert v6.licence_tier("CC BY-SA 4.0", "general") == "B"
    assert v6.licence_tier("CC BY-SA 4.0", "code") is None
    assert v6.licence_tier("CC BY-NC-SA 4.0", "general") is None
    assert v6.licence_tier("CC BY-ND", "general") is None
    assert v6.licence_tier(["MIT", "Apache-2.0"], "code") == "A"
    assert v6.licence_tier(["MIT", "GPL-3.0"], "code") is None
    assert v6.licence_tier("", "code") is None and v6.licence_tier("other", "general") is None


def test_sketcher_matches_corpus_v4():
    sk = v6.Sketcher()
    text = inspect.getdoc(importlib.import_module("argparse"))
    assert np.array_equal(sk.sketch(text), v6.v4.sketch(text))
    assert np.array_equal(sk.sketch(text), v6.v4.sketch(text))  # second call served from the word cache


def test_end_to_end(tmp_path):
    cfg = _write_sources(tmp_path)
    out = tmp_path / "out"
    base = ["--out", str(out), "--config", str(cfg)]
    assert v6.main(["plan", *base]) == 0
    for w in range(2):
        v6.main(["filter", *base, "--worker", str(w), "--workers", "2", "--min-free-gb", "0"])
    done = {d["source"]: d for d in (json.loads(p.read_text()) for p in (out / "kept").glob("*.done.json"))}
    code, text = done["code"]["reasons"], done["text"]["reasons"]
    assert code == {"ok": 5, "licence": 1, "contaminated": 1, "path": 1}, code
    assert text["licence"] == 3 and text["ok"] == 8, text  # SA in a tier-A source, NC, unknown
    v6.main(["dedup", *base, "--partitions", "4"])
    st = json.loads((out / "dedup" / "dedup.json").read_text())
    assert st["exact_duplicates"] == 1 and st["near_duplicates"] == 1 and st["kept"] == 11, st
    v6.main(["tokenize", *base, "--worker", "0", "--workers", "1"])
    v6.main(["pack", *base])
    s = out / "store"
    meta = json.loads((s / "train.meta.json").read_text())
    kept = np.load(s / "kept.npy")
    train = np.fromfile(s / "train.bin", "<u2")
    val = np.fromfile(s / "train.val.bin", "<u2")
    assert meta["tokens"] == len(train) and meta["val_tokens"] == len(val)
    assert int((kept[:, 2] + 1).sum()) == len(train) and np.all(np.diff(kept[:, 1]) > 0)
    assert np.all(train[kept[:, 1] + kept[:, 2]] == v6.EOS)          # every segment ends on its eos
    rep = json.loads((s / "report.json").read_text())
    assert rep["train_bin_sha256_expected"] == hashlib.sha256((s / "train.bin").read_bytes()).hexdigest()
    man = json.loads((s / "manifest.json").read_text())
    assert man["documents"] == len(kept) and sum(v["docs"] for v in man["per_source_tier"].values()) == 11
