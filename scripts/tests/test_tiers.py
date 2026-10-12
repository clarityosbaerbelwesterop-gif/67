"""Capability tiers: champion < frontier < dangerous."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("tiers", Path(__file__).resolve().parents[1] / "tiers.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def test_tiers():
    zero = {"humaneval": {"pass@1": 0.0}, "mbpp": {"pass@1": 0.0}, "arc-agi-2": {"pass@2": 0.0, "pass@1": 0.0}}
    assert T.classify(zero, {"0": {"pass_rate": 0.3}})["tier"] == "champion"
    arc = dict(zero, **{"arc-agi-2": {"pass@2": 0.95}})
    r = T.classify(arc, {"0": {"pass_rate": 0.3}})
    assert r["tier"] == "frontier" and "arc-agi-2" in r["reasons"][0]
    allin = {"humaneval": {"pass@1": 1.0}, "mbpp": {"pass@1": 0.995}, "arc-agi-2": {"pass@2": 0.97}}
    held = {"0": {"pass_rate": 1.0}, "1": {"pass_rate": 0.99}, "2": {"pass_rate": 1.0}}
    d = T.classify(allin, held)
    assert d["tier"] == "dangerous" and d["autonomy"].startswith("not measured")
    held["1"]["pass_rate"] = 0.9  # one coding level not solved: frontier, not dangerous
    assert T.classify(allin, held)["tier"] == "frontier"
    assert T.classify({}, None)["tier"] == "champion"
