"""disabled_families: a switched-off family (rouge, darus) trains nothing, its steps are
skipped, its champion stays, and its compute goes to quasnir and quasnir_rsi."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("longrun_fs", ROOT / "scripts" / "longrun.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)

FOCUS = ROOT / "training/configs/longrun-quasnir-focus.json"


def test_default_is_off_and_config_still_equals_defaults():
    assert lr.DEFAULTS["disabled_families"] == []
    assert lr.load_config(ROOT / "training/configs/longrun.json") == lr.DEFAULTS
    assert lr.effective_shares(lr.DEFAULTS) == lr.DEFAULTS["shares"]


@pytest.mark.parametrize("bad", [["base"], ["quasnir"], ["nope"], "rouge"])
def test_only_rouge_and_darus_can_be_disabled(tmp_path, bad):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"disabled_families": bad}))
    with pytest.raises(lr.Fatal):
        lr.load_config(p)


def test_focus_config_plan_gives_freed_time_to_quasnir():
    cfg = lr.load_config(FOCUS)
    assert cfg["disabled_families"] == ["rouge", "darus"]
    sh = lr.effective_shares(cfg)
    d = lr.DEFAULTS["shares"]
    assert abs(sum(sh.values()) - 1.0) < 1e-9
    assert sh["rouge"] == sh["rouge_rsi"] == sh["darus"] == 0.0
    assert sh["base"] == d["base"] and sh["champions"] == d["champions"] and sh["report"] == d["report"]
    freed = d["rouge"] + d["rouge_rsi"] + d["darus"]
    assert sh["quasnir"] + sh["quasnir_rsi"] == pytest.approx(d["quasnir"] + d["quasnir_rsi"] + freed)
    assert sh["quasnir"] / sh["quasnir_rsi"] == pytest.approx(d["quasnir"] / d["quasnir_rsi"])
    on = lr.plan_cycle(lr.DEFAULTS, 5, 86400.0, 1700.0, "t", 8192)["phases"]
    off = lr.plan_cycle(cfg, 5, 86400.0, 1700.0, "t", 8192)["phases"]
    assert off["rouge"]["steps"] == 0 and off["rouge"]["disabled"]  # no min_steps floor
    assert on["rouge"]["steps"] >= lr.DEFAULTS["min_steps"]
    assert off["rouge_rsi"]["hours"] == off["darus"]["hours"] == 0.0
    assert off["base"]["steps"] == on["base"]["steps"]
    assert off["quasnir"]["steps"] > on["quasnir"]["steps"] and off["quasnir_rsi"]["hours"] > on["quasnir_rsi"]["hours"]


class Stub:
    """Just what LongRun.run_cycle / step_champions touch; every step records its call."""

    def __init__(self, cfg):
        self.cfg, self.calls, self.logs = cfg, [], []
        self.state = {"current": {f: {"dir": f"runs/{f}-champ"} for f in ("quasnir", "rouge", "darus")}}
        self.code_sha = "x"

    def save(self): pass
    def stop_requested(self): return False
    def code_digest(self): return "x"
    def reload_config(self): return False
    def status_file(self, **kw): pass
    def log(self, msg, **kw): self.logs.append(msg)
    def arbiter_on(self, fam): return True
    def arbitrate(self, fam, g, cyc):
        self.calls.append(("arbitrate", fam))
        return {"decision": "measured"}
    def write_champions(self): pass
    def p(self, x): return Path(x)
    def rel(self, x): return str(x)

    def __getattr__(self, name):
        if name.startswith("step_"):
            return lambda g, cyc, st, n=name[5:]: self.calls.append(n) or {}
        raise AttributeError(name)


def run(cfg):
    s = Stub(cfg)
    cyc = {"steps": {}, "plan": {"seconds": 86400.0, "phases": {}}}
    s.state["cycles"] = {"5": cyc}
    lr.LongRun.run_cycle(s, 5)
    return s, cyc


def test_run_cycle_skips_only_the_disabled_families_steps():
    s, cyc = run(lr.load_config(FOCUS))
    off = {n for f in ("rouge", "darus") for n in lr.FAMILY_STEPS[f]}
    assert off == {"rouge_mix", "rouge", "rouge_rsi", "rouge_gate", "darus_merge", "darus_rsi", "darus_gate"}
    assert s.calls == [n for n in lr.CYCLE_STEPS if n not in off]
    for n in off:
        assert cyc["steps"][n]["status"] == "skipped" and "disabled" in cyc["steps"][n]["reason"]
    assert cyc["status"] == "done"
    s, cyc = run(lr.DEFAULTS)  # default: every step runs
    assert s.calls == list(lr.CYCLE_STEPS)


def test_champions_keep_a_disabled_familys_champion_and_arbitrate_the_rest():
    s = Stub(lr.load_config(FOCUS))
    cyc = {"steps": {}, "plan": {"seconds": 86400.0, "phases": {"champions": {"hours": 1.0}}}}
    res = lr.LongRun.step_champions(s, 5, cyc, {})
    assert ("arbitrate", "quasnir") in s.calls
    assert ("arbitrate", "rouge") not in s.calls and ("arbitrate", "darus") not in s.calls
    for f in ("rouge", "darus"):
        assert res["families"][f]["decision"] == "kept" and res["families"][f]["dir"] == f"runs/{f}-champ"
