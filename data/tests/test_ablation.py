"""Replacement-ablation arm: whole documents from both stores, the requested share of candidate tokens, and the
CleanStore contract (index, eos, sha256)."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ab = importlib.import_module("build_ablation")


def _store(d: Path, n_docs: int, lo: int, seed: int, tsv: bool) -> tuple[Path, Path, set]:
    rng = np.random.default_rng(seed)
    d.mkdir()
    toks, rows, docs, pos = [], [], set(), 0
    for i in range(n_docs):
        body = rng.integers(lo, lo + 1000, rng.integers(5, 60)).astype(np.uint16)
        toks += [*body, ab.EOS]
        rows.append((i, pos, len(body)))
        docs.add(tuple(body.tolist()))
        pos += len(body) + 1
    np.array(toks, "<u2").tofile(d / "train.bin")
    (d / "train.meta.json").write_text(json.dumps({"tokens": pos, "dtype": "uint16", "bin": "train.bin"}))
    if tsv:
        (d / "kept.tsv").write_text("doc\tstart\tlen\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in rows))
        return d / "train.meta.json", d / "kept.tsv", docs
    np.save(d / "kept.npy", np.array(rows, np.int64))
    return d / "train.meta.json", d / "kept.npy", docs


def test_arm_has_the_requested_mix_and_the_trainer_contract(tmp_path):
    bm, bi, bdocs = _store(tmp_path / "base", 400, 0, 1, tsv=True)
    cm, ci, cdocs = _store(tmp_path / "cand", 300, 4000, 2, tsv=False)
    man = ab.build(bm, bi, cm, ci, 0.3, 5000, 7, tmp_path / "arm")
    assert 5000 <= man["tokens"] < 5200 and abs(man["frac_actual"] - 0.3) < 0.03
    out = tmp_path / "arm"
    train = np.fromfile(out / "train.bin", "<u2")
    kept = np.load(out / "kept.npy")
    assert int((kept[:, 2] + 1).sum()) == len(train) and np.all(train[kept[:, 1] + kept[:, 2]] == ab.EOS)
    seen = [tuple(train[s:s + ln].tolist()) for _, s, ln in kept]
    assert all(d in bdocs or d in cdocs for d in seen)                      # whole documents only
    cand_tokens = sum(len(d) + 1 for d in seen if d in cdocs)
    assert cand_tokens == man["tokens_per_part"]["candidate"]
    rep = json.loads((out / "report.json").read_text())
    assert rep["train_bin_sha256_expected"] == hashlib.sha256((out / "train.bin").read_bytes()).hexdigest()
    assert json.loads((out / "train.meta.json").read_text())["tokens"] == len(train)


def test_refuses_overwrite_and_too_small_stores(tmp_path):
    bm, bi, _ = _store(tmp_path / "base", 20, 0, 1, tsv=True)
    cm, ci, _ = _store(tmp_path / "cand", 20, 4000, 2, tsv=False)
    with pytest.raises(SystemExit):
        ab.build(bm, bi, cm, ci, 0.3, 10 ** 6, 1, tmp_path / "big")
    (tmp_path / "exists").mkdir()
    with pytest.raises(SystemExit):
        ab.build(bm, bi, cm, ci, 0.3, 100, 1, tmp_path / "exists")
