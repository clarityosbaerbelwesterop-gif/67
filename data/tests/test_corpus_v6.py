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
    # Semantic Scholar spelling (peS2o metadata.oa_license)
    assert v6.licence_tier("CCBY", "general") == "A" and v6.licence_tier("CCBYSA", "general") == "B"
    assert v6.licence_tier("CCBYNC", "general") is None and v6.licence_tier("Unlicense", "code") == "A"


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


def test_scan_reports_contamination_and_rules_without_changing_data(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    gsm = json.loads((v6.EVALS / "gsm8k.jsonl").read_text().splitlines()[5])
    leak = next(v for v in gsm.values() if isinstance(v, str) and len(v.split()) > 25)
    good = [inspect.getdoc(importlib.import_module(m)) for m in ("argparse", "json", "heapq")]
    texts = good + ["Worked example.\n\n" + leak + "\n\nThat is the answer.", "", "x" * 50]
    (tmp_path / "in").mkdir()
    pq.write_table(pa.table({"text": texts[:3], "id": ["a", "b", "c"]}), tmp_path / "in" / "fm-0.parquet")
    with gzip.open(tmp_path / "in" / "fm-1.jsonl.gz", "wt") as f:
        for i, t in enumerate(texts[3:]):
            f.write(json.dumps({"text": t, "id": f"d{i}"}) + "\n")
    before = {p.name: p.read_bytes() for p in (tmp_path / "in").iterdir()}
    out = tmp_path / "scan"
    assert v6.main(["scan", "--out", str(out), "--files", str(tmp_path / "in" / "fm-*"), "--id", "id",
                    "--licence", "odc-by", "--workers", "2"]) == 0
    rep = json.loads((out / "scan.json").read_text())
    assert rep["rows"] == 6 and rep["files"] == 2 and rep["licence"] == "odc-by"
    assert rep["reasons"]["contaminated"] == 1 and rep["contaminated_by"] == {"gsm8k": 1}, rep["reasons"]
    assert rep["reasons"]["empty"] == 1 and rep["kept"] == 3, rep["reasons"]
    ex = [json.loads(l) for l in (out / "excluded.jsonl").read_text().splitlines()]
    assert {e["id"] for e in ex} == {"d0", "d1", "d2"} and {e["reason"] for e in ex} >= {"contaminated", "empty"}
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "in").iterdir()}  # nothing changed
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "newset.jsonl").write_text(json.dumps({"prompt": good[1]}) + "\n")
    v6.main(["scan", "--out", str(out), "--files", str(tmp_path / "in" / "fm-*"), "--evals", str(tmp_path / "own")])
    rep = json.loads((out / "scan.json").read_text())
    assert rep["eval_sets"] == ["newset"] and rep["contaminated_by"] == {"newset": 1}, rep["contaminated_by"]


def test_plan_max_bytes_takes_whole_files_up_to_the_cap(tmp_path):
    (tmp_path / "src").mkdir()
    for i in range(5):
        (tmp_path / "src" / f"part-{i}.jsonl").write_text("x" * 100)
    cfg = {"sources": [{"name": "s", "local": str(tmp_path / "src" / "*.jsonl"), "kind": "general", "tier": "A",
                        "text": "text", "source_licence": "CC0", "max_bytes": 250}]}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    assert v6.main(["plan", "--out", str(tmp_path / "out"), "--config", str(tmp_path / "cfg.json")]) == 0
    files = json.loads((tmp_path / "out" / "plan.json").read_text())["files"]
    assert len(files) == 3 and sum(f["size"] for f in files) == 300


def test_template_joins_task_and_answer_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    src = {"name": "oci", "template": "{input}\n\n{output}", "kind": "reasoning", "tier": "A",
           "source_licence": "CC-BY-4.0", "where": {"average_test_score": ["1", "1.0"]}}
    assert v6._template_fields(src) == ["input", "output"]
    row = {"input": "Add two numbers.", "output": "```python\ndef add(a, b):\n    return a + b\n```",
           "average_test_score": "1", "unused": "x"}
    assert v6._text(row, src) == "Add two numbers.\n\n```python\ndef add(a, b):\n    return a + b\n```"
    assert v6._text({"input": "only the task"}, src) is None            # a missing column gives no document
    assert v6._text({"text": "plain"}, {"text": "text"}) == "plain"    # sources without a template are unchanged
    kept = v6.keep_row(row, src, "reasoning", set())
    assert kept[0].startswith("Add two numbers.") and kept[2] == "CC-BY-4.0"
    assert v6.keep_row({**row, "average_test_score": "0.9"}, src, "reasoning", set()) == (None, "where")
    p = tmp_path / "x.parquet"
    pq.write_table(pa.Table.from_pylist([row]), p)
    got = list(v6.rows(p, src))
    assert got == [{"average_test_score": "1", "input": row["input"], "output": row["output"]}]  # only needed columns
