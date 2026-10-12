"""Release ladder: the 2x rule, half steps, McNemar non-regression and the registry."""
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("releases", Path(__file__).resolve().parents[1] / "releases.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def min_new(a, n, bar):
    return next(b for b in range(n + 1) if R.lor_lb(a, b, n) >= bar)


@pytest.mark.parametrize("old,major,half", [(0, 11, 8), (3, 18, 13), (11, 38, 28), (36, 88, 67), (101, 185, 152)])
def test_minimum_passes_for_a_release_out_of_384(old, major, half):
    assert abs(min_new(old, 384, 2.0) - major) <= 1
    assert abs(min_new(old, 384, math.sqrt(2)) - half) <= 1


def test_mcnemar_exact():
    assert R.mcnemar_loss_p(0, 0) == 1.0
    assert R.mcnemar_loss_p(3, 0) == pytest.approx(1 / 8)
    assert R.mcnemar_loss_p(35, 1) < 1e-8
    assert R.mcnemar_loss_p(1, 35) > 0.99


def heldout(tmp, tag, passes):
    out = {}
    for lv, (p, n) in passes.items():
        f = tmp / f"{tag}-L{lv}.jsonl"
        f.write_text("".join(json.dumps({"task_id": f"t{lv}-{i}", "status": "pass" if i < p else "wrong"}) + "\n"
                             for i in range(n)))
        out[str(lv)] = {"passed": p, "n": n, "pass_rate": p / n, "results": f.name}
    return out


def test_gate_major_half_none_and_blocks(tmp_path):
    val = {"code": 1.8, "general": 2.6}
    ref = {"val": val, "heldout": heldout(tmp_path, "ref", {0: (1, 128), 1: (1, 128), 2: (1, 128)})}
    big = {"val": val, "heldout": heldout(tmp_path, "big", {0: (41, 128), 1: (58, 128), 2: (2, 128)})}
    g = R.gate("quasnir", ref, big, tmp_path)
    assert g["decision"] == "major" and g["evidence"]["new_passed"] == 101
    small = {"val": val, "heldout": heldout(tmp_path, "small", {0: (5, 128), 1: (5, 128), 2: (4, 128)})}
    assert R.gate("quasnir", ref, small, tmp_path)["decision"] == "half"
    worse_val = {"val": {"code": 1.9, "general": 2.6}, "heldout": big["heldout"]}
    assert R.gate("quasnir", ref, worse_val, tmp_path)["decision"] == "none"
    # level 0 forgotten (30 passed before, 0 now) blocks even with a large pooled gain
    ref2 = {"val": val, "heldout": heldout(tmp_path, "ref2", {0: (30, 128), 1: (0, 128), 2: (0, 128)})}
    cand2 = {"val": val, "heldout": heldout(tmp_path, "cand2", {0: (0, 128), 1: (120, 128), 2: (0, 128)})}
    g2 = R.gate("quasnir", ref2, cand2, tmp_path)
    assert g2["decision"] == "none" and "level 0" in g2["reasons"][0]
    other = {"val": val, "heldout": heldout(tmp_path, "o", {0: (41, 64)})}
    assert R.gate("quasnir", ref, other, tmp_path)["decision"] == "none"


def test_base_perplexity_rule():
    ref = {"val": {"base": 2.0, "code": 1.8}}
    assert R.gate("base", ref, {"val": {"base": 2.0 - math.log(2) - 1e-6, "code": 1.8}})["decision"] == "major"
    assert R.gate("base", ref, {"val": {"base": 1.6, "code": 1.8}})["decision"] == "half"
    assert R.gate("base", ref, {"val": {"base": 1.9, "code": 1.8}})["decision"] == "none"
    assert R.gate("base", ref, {"val": {"base": 1.0, "code": 1.9}})["decision"] == "none"


def test_versions_never_compound():
    fam = {"releases": [{"version": 1.0}]}
    assert R.next_version(fam, "half") == 1.5
    assert R.next_version(fam, "major") == 2.0
    fam["releases"].append({"version": 1.5})
    assert R.next_version(fam, "half") is None, "1.5 exists: the next step needs 2x over version 1"
    assert R.next_version(fam, "major") == 2.0
    assert R.next_version({"releases": []}, "major") is None
    assert R.display_name("rouge", 1.5) == "Rouge 1.5" and R.display_name("darus", 2.0) == "Darus 2"


def test_choose_levels():
    ref = {"heldout": {"0": {"passed": 120, "n": 128}, "1": {"passed": 100, "n": 128}, "2": {"passed": 60, "n": 128}}}
    assert R.choose_levels(ref, [0, 1, 2]) == [3, 4, 5]
    assert R.choose_levels({"heldout": {"0": {"passed": 1, "n": 128}}}, [0, 1, 2]) == [0, 1, 2]


def test_record_release(tmp_path):
    src = tmp_path / "runs/q/FINAL"
    src.mkdir(parents=True)
    (src / "model.safetensors").write_bytes(b"weights")
    (src / "state.json").write_text("{}")
    sha = R._sha256(src / "model.safetensors")
    reg = tmp_path / "runs/releases.json"
    r1 = R.record_release(tmp_path, reg, "rouge", 1, "runs/q/FINAL", sha, {"val": {}}, None, 1, restricted=True)
    out = tmp_path / r1["dir"]
    assert (out / "model.safetensors").read_bytes() == b"weights"
    assert os.stat(out).st_mode & 0o777 == 0o700
    assert R.record_release(tmp_path, reg, "rouge", 1, "runs/q/FINAL", sha, {}, None, 1) == r1, "idempotent"
    assert json.loads(reg.read_text())["families"]["rouge"]["current"] == 1.0
    with pytest.raises(ValueError):
        R.record_release(tmp_path, reg, "rouge", 2, "runs/q/FINAL", "0" * 64, {}, None, 2)
    assert not (tmp_path / "runs/releases/rouge-v2").exists()
