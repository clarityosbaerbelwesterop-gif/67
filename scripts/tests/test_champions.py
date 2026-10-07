"""Champion arbiter: the selection rule on injected metrics (every branch), evaluate()'s
forge flags and cache with a stub forge, the CLI, and one tiny real evaluate().

Outputs go under TOOLS_TEST_DIR (default: the session scratchpad .../scratchpad/wf/tools,
else a pytest tmp dir)."""

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("champions", ROOT / "scripts" / "champions.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

FORGE = ROOT / "target" / "release" / "forge"
SCRATCHPAD = Path("/tmp/claude-0/-home-user/d103ac1f-f13d-5616-bc4d-c936a331fc71/scratchpad/wf/tools")


@pytest.fixture
def scratch(request, tmp_path) -> Path:
    base = Path(os.environ["TOOLS_TEST_DIR"]) if os.environ.get("TOOLS_TEST_DIR") else (
        SCRATCHPAD if SCRATCHPAD.parent.exists() else tmp_path)
    d = base / "champions" / request.node.name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def cand(name: str, val: dict, heldout=None, incumbent: bool = False) -> dict:
    """A candidate with injected metrics; heldout = pass rate per level (None: family base)."""
    return {"name": name, "dir": f"runs/{name}", "incumbent": incumbent,
            "metrics": {"sha256": f"sha-{name}", "dir": f"runs/{name}", "val": val,
                        "heldout": {str(lv): {"pass_rate": r, "passed": None, "n": 64} for lv, r in enumerate(heldout or [])}}}


def chosen(d: dict) -> str:
    return d["chosen"]["name"]


# --------------------------------------------------------------------------- selection rule

def test_challenger_needs_the_heldout_margin():
    inc = cand("inc", {"code": 2.00, "general": 3.00}, [0.40, 0.20], incumbent=True)
    big = cand("big", {"code": 2.01, "general": 3.02}, [0.50, 0.20])  # +0.05 mean held-out, val within 1 %
    d = C.select([inc, big])
    assert chosen(d) == "big" and d["changed"] and d["incumbent"] == "inc" and d["mode"] == "heldout"
    assert any(r.startswith("big qualifies") for r in d["reasons"]) and d["reasons"][-1] == "big replaces the incumbent inc"
    small = cand("small", {"code": 2.00, "general": 3.00}, [0.42, 0.20])  # +0.01: within the margin, val not lower
    d = C.select([inc, small])
    assert chosen(d) == "inc" and not d["changed"] and d["reasons"][-1] == "the incumbent inc stays"
    assert any("small does not replace inc" in r and "not lower" in r for r in d["reasons"])


def test_tie_within_margin_needs_half_a_percent_lower_val():
    inc = cand("inc", {"code": 2.000}, [0.50], incumbent=True)
    better = cand("better", {"code": 1.988}, [0.52])  # inc is +0.60 % above the best: 0.6 % lower >= 0.5 %
    d = C.select([inc, better])
    assert chosen(d) == "better" and any("ties within the margin" in r and "better qualifies" in r for r in d["reasons"])
    slightly = cand("slightly", {"code": 1.995}, [0.52])  # 0.25 % lower: not enough
    assert chosen(C.select([inc, slightly])) == "inc"
    lower_h = cand("lower_h", {"code": 1.988}, [0.48])  # 0.02 below the incumbent is still a tie within the margin
    assert chosen(C.select([inc, lower_h])) == "lower_h"


def test_challenger_below_by_the_margin_never_wins_on_val():
    inc = cand("inc", {"code": 2.000}, [0.50], incumbent=True)
    worse_h = cand("worse_h", {"code": 1.985}, [0.45])  # 0.75 % lower val, but held-out 0.05 below
    d = C.select([inc, worse_h])
    assert chosen(d) == "inc" and any("below the incumbent by at least the margin" in r for r in d["reasons"])


def test_infeasible_challenger_is_ignored():
    inc = cand("inc", {"code": 2.00, "general": 3.00}, [0.10], incumbent=True)
    hacker = cand("hacker", {"code": 2.00, "general": 3.06}, [0.90])  # +2 % on general
    d = C.select([inc, hacker])
    assert chosen(d) == "inc"
    row = next(r for r in d["ranking"] if r["name"] == "hacker")
    assert not row["feasible"] and "general" in row["infeasible_because"][0] and "+2.00%" in row["infeasible_because"][0]
    assert any(r.startswith("hacker is infeasible") for r in d["reasons"])


def test_infeasible_incumbent_is_replaced_by_best_feasible():
    inc = cand("inc", {"code": 2.05}, [0.90], incumbent=True)  # +2.5 % vs the best
    a = cand("a", {"code": 2.00}, [0.40])
    b = cand("b", {"code": 2.01}, [0.40])  # same held-out, higher val
    c = cand("c", {"code": 2.015}, [0.30])
    d = C.select([inc, b, c, a])
    assert chosen(d) == "a" and d["changed"] and any(r.startswith("the incumbent inc is infeasible: a has") for r in d["reasons"])
    assert [r["name"] for r in d["ranking"]] == ["a", "b", "c", "inc"]
    assert [r["rank"] for r in d["ranking"]] == [1, 2, 3, 4] and [r["chosen"] for r in d["ranking"]] == [True, False, False, False]


def test_no_incumbent_highest_heldout_then_lower_val():
    a = cand("a", {"code": 2.00}, [0.30, 0.10])
    b = cand("b", {"code": 2.01}, [0.20, 0.30])  # higher mean held-out (0.25 vs 0.20)
    d = C.select([a, b])
    assert chosen(d) == "b" and d["changed"] and d["incumbent"] is None
    assert any(r.startswith("no incumbent: b has the highest mean held-out pass rate") for r in d["reasons"])
    c = cand("c", {"code": 2.005}, [0.25, 0.25])  # ties b, lower val
    assert chosen(C.select([a, b, c])) == "c"


def test_several_challengers_highest_heldout_wins():
    inc = cand("inc", {"code": 2.00}, [0.20], incumbent=True)
    c1 = cand("c1", {"code": 2.00}, [0.25])
    c2 = cand("c2", {"code": 2.00}, [0.30])
    c3 = cand("c3", {"code": 1.985}, [0.21])  # qualifies by val (0.75 % lower), lower held-out than c2
    d = C.select([inc, c1, c3, c2])
    assert chosen(d) == "c2"
    assert sum(r.split(":")[0].endswith("qualifies") for r in d["reasons"]) == 3


def test_base_family_lowest_val_with_hysteresis():
    inc = cand("inc", {"base": 2.000}, incumbent=True)
    near = cand("near", {"base": 1.995})  # 0.25 % lower
    d = C.select([inc, near])
    assert d["mode"] == "val-only" and d["levels"] == [] and chosen(d) == "inc"
    far = cand("far", {"base": 1.985})  # 0.75 % lower
    d = C.select([inc, near, far])
    assert chosen(d) == "far" and any(r.startswith("far qualifies") for r in d["reasons"])
    stale = cand("stale", {"base": 2.05}, incumbent=True)
    assert chosen(C.select([stale, near, cand("x", {"base": 1.999})])) == "near"
    assert chosen(C.select([cand("a", {"base": 2.0}), cand("b", {"base": 1.9999})])) == "b"


def test_nobody_feasible_keeps_incumbent_or_minimises_worst_case():
    a = cand("a", {"code": 2.0, "general": 3.10}, [0.5])  # +3.3 % on general
    b = cand("b", {"code": 2.1, "general": 3.00}, [0.9])  # +5 % on code
    d = C.select([a, {**b, "incumbent": True}])
    assert chosen(d) == "b" and not d["changed"] and any("the incumbent b stays" in r for r in d["reasons"])
    d = C.select([a, b])
    assert chosen(d) == "a" and any("smallest worst-case" in r for r in d["reasons"])


def test_missing_or_non_finite_metrics_are_infeasible():
    inc = cand("inc", {"code": 2.0}, [0.1], incumbent=True)
    nan = cand("nan", {"code": math.nan}, [0.9])
    none = cand("none", {"code": None}, [0.9])
    noh = cand("noh", {"code": 2.0}, [None])
    d = C.select([inc, nan, none, noh])
    assert chosen(d) == "inc"
    why = {r["name"]: r["infeasible_because"] for r in d["ranking"]}
    assert why["nan"] == why["none"] == ["val loss on code missing or not finite"] and why["noh"] == ["held-out pass rate missing"]
    assert d["best_val"] == {"code": 2.0}


def test_boundaries_tolerate_float_noise():
    inc = cand("inc", {"code": 2.00}, [0.55], incumbent=True)
    edge = cand("edge", {"code": 2.02}, [0.58])  # 2.02/2.00 - 1 and 0.58 - 0.55 are off by one ulp
    d = C.select([inc, edge])
    assert chosen(d) == "edge" and all(r["feasible"] for r in d["ranking"])
    assert next(r for r in d["ranking"] if r["name"] == "edge")["rel_val"]["code"] == pytest.approx(0.01)


def test_ranking_carries_all_metrics():
    d = C.select([cand("a", {"code": 2.0, "general": 3.03}, [0.5, 0.1], incumbent=True),
                  cand("b", {"code": 2.02, "general": 3.0}, [0.4, 0.2])], tol=0.02, margin=0.05)
    assert d["tol"] == 0.02 and d["margin"] == 0.05 and d["val_margin"] == 0.005 and d["stores"] == ["code", "general"]
    assert d["best_val"] == {"code": 2.0, "general": 3.0} and d["levels"] == ["0", "1"]
    a = next(r for r in d["ranking"] if r["name"] == "a")
    assert a["heldout_mean"] == pytest.approx(0.3) and a["heldout"] == {"0": 0.5, "1": 0.1} and a["sha256"] == "sha-a"
    assert a["rel_val"]["general"] == pytest.approx(0.01) and a["worst_rel_val"] == pytest.approx(0.01)
    assert a["mean_rel_val"] == pytest.approx(0.005) and a["metrics"]["val"] == {"code": 2.0, "general": 3.03}
    assert d["chosen"] == {"name": "a", "dir": "runs/a", "sha256": "sha-a"} and "0.5 %" in d["rule"]
    json.dumps(d)


def test_select_refuses_inconsistent_inputs():
    a, b = cand("a", {"code": 2.0}, [0.1]), cand("b", {"code": 2.0}, [0.1])
    for bad, msg in [([], "no candidates"), ([a, a], "unique"),
                     ([{**a, "incumbent": True}, {**b, "incumbent": True}], "at most one"),
                     ([a, cand("b", {"general": 2.0}, [0.1])], "same, non-empty set of gate stores"),
                     ([cand("a", {}, [0.1])], "non-empty"),
                     ([a, cand("b", {"code": 2.0}, [0.1, 0.2])], "same held-out levels"),
                     ([a, cand("b", {"code": 2.0})], "same held-out levels")]:
        with pytest.raises(ValueError, match=msg):
            C.select(bad)


# --------------------------------------------------------------------------- evaluate with a stub forge

def stub_forge(d: Path, fail: bool = False) -> Path:
    """An executable that logs its argv and answers `forge eval` with the loss in <ckpt>/loss.txt."""
    p = d / "forge-stub"
    p.write_text(f"""#!{sys.executable}
import json, sys
from pathlib import Path
with open({str(d / 'calls.jsonl')!r}, "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if {fail!r}:
    sys.exit("stub failure")
a = sys.argv[1:]
ck = Path(a[a.index("--ckpt") + 1])
print("warming up")
print(json.dumps({{"type": "eval", "loss": float((ck / "loss.txt").read_text())}}))
""")
    p.chmod(0o755)
    return p


def fake_ckpt(d: Path, loss: float, max_seq_len: int = 128) -> Path:
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(d.name.encode() * 8)
    sha = hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest()
    (d / "state.json").write_text(json.dumps({"step": 1, "model_sha256": sha, "config": {"model": {"max_seq_len": max_seq_len}}}))
    (d / "loss.txt").write_text(str(loss))
    return d


def fake_store(d: Path) -> Path:
    d.mkdir(parents=True)
    (d / "train.val.bin").write_bytes(b"\x01\x00" * 8)
    (d / "train.meta.json").write_text(json.dumps({"bin": "train.bin", "val_bin": "train.val.bin", "vocab_size": 8192}))
    return d / "train.meta.json"


def calls(d: Path) -> list:
    p = d / "calls.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_evaluate_uses_the_gate_flags_and_caches_per_sha(scratch):
    forge, ck, meta = stub_forge(scratch), fake_ckpt(scratch / "ck", 2.5), fake_store(scratch / "store")
    cache = scratch / "cache.json"
    m = C.evaluate(ck, "base", {"g": meta}, levels=[0, 1], batches=3, seed=7, cache=cache, forge=forge)
    assert m["val"] == {"g": 2.5} and m["heldout"] == {} and m["heldout_mean"] is None and m["tasks"] is None
    assert m["settings"]["levels"] == [] and m["settings"]["seq"] == 128 and m["sha256"] in json.loads(cache.read_text())
    argv = calls(scratch)[0]
    want = ["eval", "--ckpt", str(ck.resolve()), "--data", str(meta.resolve()), "--split", "val", "--batch", "8", "--seq", "128",
            "--batches", "3", "--seed", "7"]
    assert argv == want  # RSILoop.val_losses' flags; --threads only when set
    assert C.evaluate(ck, "base", {"g": meta}, batches=3, seed=7, cache=cache, forge=scratch / "missing") == m  # cache hit
    assert len(calls(scratch)) == 1
    C.evaluate(ck, "base", {"g": meta}, batches=3, seed=7, cache=cache, forge=forge, threads=2)  # --threads is not in the key
    assert len(calls(scratch)) == 1
    C.evaluate(ck, "base", {"g": meta}, batches=4, seed=7, cache=cache, forge=forge, threads=2)
    assert calls(scratch)[-1][-2:] == ["--threads", "2"] and len(calls(scratch)) == 2
    (meta.parent / "train.val.bin").write_bytes(b"\x02\x00" * 9)  # a rebuilt store is measured again
    C.evaluate(ck, "base", {"g": meta}, batches=3, seed=7, cache=cache, forge=forge)
    assert len(calls(scratch)) == 3
    entry = json.loads(cache.read_text())[m["sha256"]]
    assert len(entry["val"]) == 3 and entry["dirs"] == [C.rel(ck)]


def test_evaluate_refusals(scratch):
    forge, meta = stub_forge(scratch), fake_store(scratch / "store")
    ck = fake_ckpt(scratch / "ck", 2.5)
    with pytest.raises(ValueError, match="family"):
        C.evaluate(ck, "llama", {"g": meta}, forge=forge)
    with pytest.raises(ValueError, match="gate store"):
        C.evaluate(ck, "base", {}, forge=forge)
    with pytest.raises(ValueError, match="levels"):
        C.evaluate(ck, "quasnir", {"g": meta}, levels=[10], forge=forge)
    with pytest.raises(FileNotFoundError, match="gate store g"):
        C.evaluate(ck, "base", {"g": scratch / "nope.json"}, forge=forge)
    with pytest.raises(FileNotFoundError, match="not a forge checkpoint"):
        C.evaluate(scratch / "nope", "base", {"g": meta}, forge=forge)
    (ck / "model.safetensors").write_bytes(b"changed after training")
    with pytest.raises(RuntimeError, match="state.json records"):
        C.evaluate(ck, "base", {"g": meta}, forge=forge)
    with pytest.raises(RuntimeError, match="forge eval failed"):
        C.evaluate(fake_ckpt(scratch / "ck2", 2.0), "base", {"g": meta}, forge=stub_forge(scratch / "ck2", fail=True))
    assert calls(scratch) == []


def test_cli_decides_and_aborts_on_errors(scratch, capsys):
    forge, meta = stub_forge(scratch), fake_store(scratch / "store")
    a, b = fake_ckpt(scratch / "a", 2.000), fake_ckpt(scratch / "b", 1.985)
    out, cache = scratch / "decision.json", scratch / "cache.json"
    argv = ["--family", "base", "--candidate", f"a={a}", "--candidate", f"b={b}", "--incumbent", "a", "--gate", f"base={meta}",
            "--batches", "2", "--cache", str(cache), "--out", str(out), "--forge", str(forge)]
    assert C.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    d = json.loads(out.read_text())
    assert printed == d and chosen(d) == "b" and d["changed"] and d["family"] == "base" and d["mode"] == "val-only"
    assert d["settings"]["batches"] == 2 and d["settings"]["gates"] == {"base": str(meta)} and len(calls(scratch)) == 2
    assert not out.with_name(out.name + ".tmp").exists()
    out.unlink()
    assert C.main(argv[:5] + [f"b={scratch / 'missing'}"] + argv[6:]) == 1  # an evaluation error: no decision at all
    assert not out.exists() and json.loads(capsys.readouterr().err)["type"] == "error"
    for bad in (["--incumbent", "zz"], ["--candidate", "a"], ["--levels", "x"], ["--candidate", f"a={b}"]):
        with pytest.raises(SystemExit) as e:
            C.main(argv + bad)
        assert e.value.code == 2


# --------------------------------------------------------------------------- real checkpoint

REAL_CKPT = ROOT / "runs/quasnir-g2/FINAL"
CODE_V2 = ROOT / "data/out/code-v2/train.meta.json"


@pytest.mark.skipif(not FORGE.exists() or not CODE_V2.exists() or not (REAL_CKPT / "model.safetensors").exists()
                    or not (ROOT / "data/stores/evals").is_dir(), reason="forge, data/out/code-v2, runs/quasnir-g2 or eval sets missing")
def test_real_evaluate_tiny(scratch):
    cache = scratch / "cache.json"
    m = C.evaluate(REAL_CKPT, "quasnir", {"code": CODE_V2}, levels=[0], n_tasks=4, batches=1, threads=2, cache=cache)
    sha = json.loads((REAL_CKPT / "state.json").read_text())["model_sha256"]
    assert m["sha256"] == sha and m["tasks"] == "code" and math.isfinite(m["val"]["code"]) and 0 < m["val"]["code"] < 20
    h = m["heldout"]["0"]
    assert 1 <= h["n"] <= 4 and 0 <= h["passed"] <= h["n"] and h["pass_rate"] == pytest.approx(h["passed"] / h["n"])
    assert h["decoding"] == "greedy" and m["heldout_mean"] == h["pass_rate"]
    rows = [json.loads(x) for x in (ROOT / h["results"]).read_text().splitlines()]  # rel() paths: repo-relative or absolute
    assert len(rows) == h["n"] and all(int(r["task_id"].rsplit("-s", 1)[1]) >= C.T.SEED_SPACE for r in rows)  # held-out seeds only
    assert (scratch / "cache.work").stat().st_mode & 0o777 == 0o700
    again = C.evaluate(REAL_CKPT, "quasnir", {"code": CODE_V2}, levels=[0], n_tasks=4, batches=1, threads=2, cache=cache,
                       forge=scratch / "no-forge")
    assert again == m
