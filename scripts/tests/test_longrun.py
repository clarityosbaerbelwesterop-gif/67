"""Long-run orchestrator: config consistency, plan arithmetic, configs, automatic
promotion, the champion arbiter, rebased inits, hot config reload, corpus v3
adoption, hygiene, and resuming a state.json written by the previous version."""

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("longrun", ROOT / "scripts" / "longrun.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)
R = lr.tool("rebase")
C = lr.tool("champions")
REAL_EVALUATE = C.evaluate

MODEL = {"vocab_size": 8192, "dim": 8, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "ffn_hidden": 16, "max_seq_len": 64}


def test_longrun_json_equals_defaults():
    assert json.loads((ROOT / "training/configs/longrun.json").read_text()) == json.loads(json.dumps(lr.DEFAULTS))


def test_promotion_is_automatic_and_approval_can_be_reenabled():
    rsi = lr.DEFAULTS["rsi"]
    assert not any(m["require_approval"] for m in rsi["models"].values())  # owner decision: automatic promotion
    assert 0 < rsi["approval_timeout_hours"] <= 6 and rsi["anchor_tol"] == 0.02  # the approval mechanism stays
    assert abs(sum(lr.DEFAULTS["shares"].values()) - 1.0) < 1e-9 and lr.DEFAULTS["shares"]["champions"] > 0
    assert set(lr.DEFAULTS["champions"]["families"]) == set(lr.ARBITER_FAMILIES) <= set(rsi["models"])
    assert lr.DEFAULTS["corpus"]["v3"]["code"] == "data/out/code-v3/train.meta.json"
    assert lr.load_config(ROOT / "training/configs/longrun.json") == lr.DEFAULTS
    assert lr.CYCLE_STEPS.index("darus_gate") < lr.CYCLE_STEPS.index("champions") < lr.CYCLE_STEPS.index("report")


def test_load_config_rejects_bad_shares_and_families(tmp_path):
    for over in ({"shares": {"base": 0.5}}, {"shares": {"champions": -0.03, "base": 0.53}},
                 {"champions": {"families": ["nope"]}}, {"rebase": {"families": ["darus"]}}):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(over))
        with pytest.raises(lr.Fatal):
            lr.load_config(p)


def dry(days: str) -> dict:
    out = subprocess.run([sys.executable, str(ROOT / "scripts/longrun.py"), "--dry-run", "--json", "--days", days, "--allow-v1"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout[out.stdout.index("{"):])


def test_dry_run_budget_scales_with_days():
    p3, p20 = dry("3"), dry("20")
    assert len(p3["cycles"]) == 3 and len(p20["cycles"]) == 20
    p10 = dry("10")
    assert len(p10["cycles"]) == 10
    tok = lambda p: sum(c["phases"]["base"]["tokens"] for c in p["cycles"])  # noqa: E731
    assert 6.0 < tok(p20) / tok(p3) < 7.4
    for c in p10["cycles"]:
        assert abs(sum(ph["share"] for ph in c["phases"].values()) - 1.0) < 1e-9
        assert abs(c["phases"]["champions"]["hours"] - 0.03 * c["hours"]) < 1e-9
    assert p10["promotion"]["automatic"] and p10["promotion"]["arbiter"] == ["quasnir", "rouge", "darus"]


def test_base_m_is_a_valid_40m_config():
    c = json.loads((ROOT / "training/configs/base-m.json").read_text())
    m = c["model"]
    assert m["dim"] % m["n_heads"] == 0 and m["n_heads"] % m["n_kv_heads"] == 0 and (m["dim"] // m["n_heads"]) % 2 == 0
    hd = m["dim"] // m["n_heads"]
    n = m["vocab_size"] * m["dim"] + m["n_layers"] * (m["dim"] * m["n_heads"] * hd * 2 + m["dim"] * m["n_kv_heads"] * hd * 2
                                                       + 3 * m["dim"] * m["ffn_hidden"] + 2 * m["dim"]) + m["dim"]
    assert 35e6 <= n <= 45e6
    assert c["seq_len"] <= m["max_seq_len"]


# --------------------------------------------------------------------------- a small world on disk

def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=1))


def ckpt(root: Path, rel: str, w, init_from=None, parent=None, **extra) -> str:
    """A forge-like checkpoint (one F32 tensor "w"); returns its model sha256."""
    d = root / rel
    d.mkdir(parents=True)
    sha = R.write_safetensors(d / "model.safetensors", {"w": {"dtype": "F32", "shape": [4],
                                                            "data": np.asarray(w, np.float32).tobytes()}}, {"format": "pt"})
    write(d / "state.json", {"step": int(d.name[5:]) if d.name.startswith("step-") else 0, "model_sha256": sha,
                             "parent_sha256": parent,
                             "config": {"run": rel.split("/")[1], "model": MODEL, "init_from": init_from}, **extra})
    return sha


def weights(root: Path, d) -> np.ndarray:
    d = Path(d) if Path(d).is_absolute() else root / d
    ts, _, _ = R.read_safetensors(d / "model.safetensors")
    return np.frombuffer(bytes(ts["w"]["data"]), np.float32).astype(np.float64)


def final(root: Path, run: str, step: str) -> None:
    (root / "runs" / run / "FINAL").symlink_to(step)
    write(root / "runs" / run / "latest.json", {"step": int(step[5:]), "dir": f"runs/{run}/{step}"})


def store(root: Path, rel: str, tok: Path) -> str:
    d = root / rel
    d.mkdir(parents=True)
    docs = np.tile(np.array([5, 6, 7, lr.EOS], np.uint16), 64)
    docs.tofile(d / "train.bin")
    docs[:64].tofile(d / "train.val.bin")
    (d / "tokenizer.json").write_bytes(tok.read_bytes())
    write(d / "train.meta.json", {"tokens": int(docs.size), "val_tokens": 64, "dtype": "uint16", "vocab_size": 8192,
                                  "bin": "train.bin", "val_bin": "train.val.bin"})
    return f"{rel}/train.meta.json"


def loss(w, s: str) -> float:
    """Synthetic val loss: w[0] helps code, w[1] general, w[3] base; w[2] is the held-out skill."""
    return 3.0 - 0.1 * w[{"code": 0, "general": 1}.get(s, 3)] - 0.01 * float(np.mean(w))


def world(tmp_path: Path, v3: bool = False) -> tuple[Path, dict]:
    """A repository with the checkpoints, stores and configs of a run that finished cycle 2
    and is in the middle of base-g3, and the state.json the previous version wrote for it
    (shaped like runs/longrun/state.json of the live run)."""
    root = tmp_path / "repo"
    write(root / "training/configs/base-s.json", {"run": "base-s", "model": MODEL, "data": "x", "batch": 2, "seq_len": 8,
                                                  "grad_accum": 1, "lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 2})
    write(root / "training/configs/longrun.json", {
        "base_config": "training/configs/base-s.json", "forge": "forge-stub", "min_free_gb": 0, "retry_backoff_seconds": 0,
        "wait_for": {"paths": [], "procs": []}, "report": {"benchmarks": {"darus": [], "quasnir": []}},
        "rsi": {"module_file": None}, "champions": {"levels": [0]}, "control": {"poll_seconds": 0.01}})
    tok = root / "data/stores/base/tokenizer.json"
    write(tok, '{"tokenizer": "ref"}')
    metas = {k: store(root, f"data/out/{k}-v2", tok) for k in ("base", "code", "general")}
    if v3:
        for k in ("base", "code", "general"):
            store(root, f"data/out/{k}-v3", tok)
    b = {}
    b["s"] = ckpt(root, "runs/base-s/step-000010", [0, 0, 0, 0])
    final(root, "base-s", "step-000010")
    b["g2"] = ckpt(root, "runs/base-g2/step-000100", [1, 1, 1, 1], "runs/base-s/FINAL", b["s"])
    final(root, "base-g2", "step-000100")
    b["q2"] = ckpt(root, "runs/quasnir-g2/step-000050", [1.5, 1, 1, 1], "runs/base-g2/FINAL", b["g2"])
    final(root, "quasnir-g2", "step-000050")
    b["r2"] = ckpt(root, "runs/rouge-g2/step-000050", [1, 1.5, 1, 1], "runs/base-g2/FINAL", b["g2"])
    final(root, "rouge-g2", "step-000050")
    rr = "runs/rsi/rouge-g2/rounds/rsi-rouge-g2-r2/step-000060"
    b["rr"] = ckpt(root, rr, [1, 1.6, 1, 1], str(root / "runs/rouge-g2/step-000050"), b["r2"])
    write(root / "runs/rsi/rouge-g2/champion.json", {"model": "rouge-g2", "dir": rr, "model_sha256": b["rr"], "round": 2,
                                                     "history": [{"dir": "runs/rouge-g2/step-000050", "round": 0}]})
    b["d2"] = ckpt(root, "runs/darus-g2/FINAL", [1.25, 1.25, 1, 1], None, b["g2"],
                   merge={"children": [{"dir": "runs/rouge-g2/FINAL"}, {"dir": "runs/quasnir-g2/FINAL"}], "method": "Linear"})
    ckpt(root, "runs/base-g3/step-000100", [1.05, 1.05, 1.05, 1.05], "runs/base-g2/FINAL", b["g2"])
    write(root / "runs/base-g3/latest.json", {"step": 100, "dir": "runs/base-g3/step-000100"})
    cfgs = {"base-g2": ("runs/base-s/FINAL", 100), "quasnir-g2": ("runs/base-g2/FINAL", 50),
            "rouge-g2": ("runs/base-g2/FINAL", 50), "base-g3": ("runs/base-g2/FINAL", 200)}
    for run, (init, steps) in cfgs.items():
        write(root / f"runs/longrun/configs/{run}.json", {"run": run, "model": MODEL, "max_steps": steps, "init_from": init,
                                                          "data": metas["base"], "out_dir": "runs"})
    t0 = time.time() - 30 * 3600
    phases = {"base": {"run": "base-g{g}", "kind": "train", "share": 0.5, "steps": 200, "tokens": 3200, "hours": 12.0},
              "quasnir": {"run": "quasnir-g{g}", "kind": "train", "share": 0.12, "steps": 40, "tokens": 640, "hours": 2.88},
              "rouge": {"run": "rouge-g{g}", "kind": "train", "share": 0.13, "steps": 40, "tokens": 640, "hours": 3.12},
              "quasnir_rsi": {"kind": "rsi", "share": 0.08, "hours": 1.92},
              "rouge_rsi": {"kind": "rsi", "share": 0.06, "hours": 1.44},
              "darus": {"kind": "merge+rsi", "share": 0.08, "hours": 1.92, "run": "darus-g{g}"},
              "report": {"kind": "eval", "share": 0.03, "hours": 0.72}}  # written before the champions share existed

    def plan(g):
        return {"generation": g, "seconds": 86400.0, "hours": 24.0, "tokens_per_s": 3458.0, "tps_source": "median telemetry",
                "tokens_per_step": 16, "phases": {k: {kk: (vv.format(g=g) if isinstance(vv, str) else vv) for kk, vv in v.items()}
                                                  for k, v in phases.items()}}

    def done(**kw):
        return {"status": "done", "started_at": t0, "finished_at": t0 + 1, "wall_seconds": 1.0, **kw}

    def gate(fam, run, d, sha, src, prev_dir):
        return done(family=fam, generation=2, decision="accept", candidate={"run": run, "dir": d, "sha256": sha, "source": src},
                    previous={"gen": 1, "run": prev_dir.split("/")[1], "dir": prev_dir, "source": "initial"}, stores={},
                    tolerance=0.01, worst_relative_change=-0.3, reason="worst change -0.3 within 0.01", ts=t0,
                    rule="candidate val loss <= previous * (1 + tolerance) on every gate store")

    steps2 = {
        "base": done(family="base", ctl="runs/base-g2.ctl", final="runs/base-g2/FINAL",
                     config="runs/longrun/configs/base-g2.json", init_from="runs/base-s/FINAL"),
        "base_gate": gate("base", "base-g2", "runs/base-g2/FINAL", b["g2"], "final", "runs/base-s/FINAL"),
        "quasnir": done(family="quasnir", ctl="runs/quasnir-g2.ctl", final="runs/quasnir-g2/FINAL",
                        config="runs/longrun/configs/quasnir-g2.json", init_from="runs/base-g2/FINAL"),
        "quasnir_rsi": done(rsi_out="runs/rsi/quasnir-g2", rsi_model="quasnir-g2", cmd=["python3"], spent_hours=0.6,
                            model_dir="runs/quasnir-g2/step-000050", promoted=True, rounds=8, decisions={"reject": 8},
                            kinds={"supervised-warmstart": 8}, self_improvement_rounds=0, heldout_pass=[[1, 0.0, 0.78]],
                            audit="runs/rsi/quasnir-g2/audit.jsonl", budget_hours=1.92, require_approval=True, log="x"),
        "quasnir_gate": gate("quasnir", "quasnir-g2", "runs/quasnir-g2/step-000050", b["q2"], "rsi-champion",
                             "runs/quasnir-1/FINAL"),
        "rouge_mix": done(meta="runs/longrun/stores/rouge-mix-935ef0cadfb9/train.meta.json", tokens=1, reused=False),
        "rouge": done(family="rouge", ctl="runs/rouge-g2.ctl", final="runs/rouge-g2/FINAL",
                      config="runs/longrun/configs/rouge-g2.json", init_from="runs/base-g2/FINAL"),
        "rouge_rsi": done(rsi_out="runs/rsi/rouge-g2", rsi_model="rouge-g2", model_dir=rr, promoted=True, rounds=4,
                          decisions={"promote": 1}, require_approval=True),
        "rouge_gate": gate("rouge", "rouge-g2", rr, b["rr"], "rsi-champion", "runs/rouge-1/FINAL"),
        "darus_merge": done(final="runs/darus-g2/FINAL", config="runs/longrun/configs/darus-g2.merge.json",
                            search="runs/darus-g2/search.json", best={"method": "linear"}),
        "darus_rsi": done(rsi_out="runs/rsi/darus-g2", rsi_model="darus-g2", model_dir="runs/darus-g2/FINAL", promoted=False,
                          decisions={"reject": 1, "pending-approval": 1}, require_approval=True),
        "darus_gate": gate("darus", "darus-g2", "runs/darus-g2/FINAL", b["d2"], "final", "runs/darus-1/FINAL"),
        "report": done(report="runs/longrun/report-g2.json", errors=[]),
        "hygiene": done(removed=12, disk_free_bytes=1),
    }
    state = {
        "version": 1, "created_at": t0 - 3600, "status": "running",
        "cycles": {"2": {"status": "done", "started_at": t0, "plan": plan(2), "steps": steps2,
                         "parent_base": "runs/base-g2/FINAL", "finished_at": t0 + 2},
                   "3": {"status": "running", "started_at": t0 + 3, "plan": plan(3),
                         "steps": {"base": {"status": "running", "started_at": t0 + 3, "family": "base", "pid": 2 ** 22 + 7,
                                            "ctl": "runs/base-g3.ctl"}}}},
        "runs": {"base-g2": {"config": "runs/longrun/configs/base-g2.json", "finished": True, "family": "base",
                             "final": "runs/base-g2/FINAL", "step": 100, "final_target": "step-000100", "sha256": b["g2"]},
                 "quasnir-g2": {"config": "runs/longrun/configs/quasnir-g2.json", "finished": True, "family": "quasnir",
                                "final": "runs/quasnir-g2/FINAL", "step": 50, "final_target": "step-000050", "sha256": b["q2"]},
                 "rouge-g2": {"config": "runs/longrun/configs/rouge-g2.json", "finished": True, "family": "rouge",
                              "final": "runs/rouge-g2/FINAL", "step": 50, "final_target": "step-000050", "sha256": b["r2"]},
                 "base-g3": {"config": "runs/longrun/configs/base-g3.json", "finished": False, "family": "base"}},
        "eval_cache": {},
        "throughput": {"bench": {"tokens_per_s": 1749.9, "step_ms": 4681.5, "params": 12587904, "steps": 3,
                                 "config": "training/configs/base-s.json", "measured_at": t0, "loadavg_before": [3.0, 3.0, 3.0]},
                       "telemetry": {"base-g2": {"median_tokens_per_s": 3458.0, "samples": 500}}},
        "budget_started_at": t0, "cycle": 3,
        "current": {"base": {"gen": 2, "run": "base-g2", "dir": "runs/base-g2/FINAL", "sha256": b["g2"], "source": "final"},
                    "quasnir": {"gen": 2, "run": "quasnir-g2", "dir": "runs/quasnir-g2/step-000050", "sha256": b["q2"],
                                "source": "rsi-champion"},
                    "rouge": {"gen": 2, "run": "rouge-g2", "dir": rr, "sha256": b["rr"], "source": "rsi-champion"},
                    "darus": {"gen": 2, "run": "darus-g2", "dir": "runs/darus-g2/FINAL", "sha256": b["d2"], "source": "final"}},
        "config_path": "training/configs/longrun.json", "config_sha256": "5a82e99b", "days": 10.0, "forever": False,
        "max_cycles": None, "cycle_hours": 24, "pid": 385, "updated_at": t0 + 4, "message": "STOP requested while waiting",
        "corpus": {"version": "v2", "stores": {k: {"meta": m, "tokens": 256, "val_tokens": 64, "meta_sha256": "x"}
                                               for k, m in metas.items()}, "checked_at": t0, "build": None},
        "mix": {"key": "935ef0cadfb9", "meta": "runs/longrun/stores/rouge-mix-935ef0cadfb9/train.meta.json", "tokens": 1,
                "val_tokens": 1, "built_at": t0},
    }
    write(root / "runs/longrun/state.json", state)
    return root, b


def args(**kw) -> argparse.Namespace:
    a = dict(config="training/configs/longrun.json", root=None, days=None, forever=False, cycle_hours=None, max_cycles=None,
             allow_v1=False, dry_run=False, json=False, tok_per_s=1000.0, status=False, prepare=False, no_wait=True,
             forge="forge-stub")
    a.update(kw)
    return argparse.Namespace(**a)


def orchestrator(root: Path, **kw) -> "lr.LongRun":
    return lr.LongRun(root, lr.load_config(root / "training/configs/longrun.json"), args(**kw))


def fake_evaluate(calls: list, fail=()):
    """champions.evaluate on the synthetic losses (held-out pass rate = w[2] - 1); the call
    is bound to the real signature, so a changed API fails here."""
    def evaluate(*a, **kw):
        ba = inspect.signature(REAL_EVALUATE).bind(*a, **kw)
        ba.apply_defaults()
        x = ba.arguments
        d = Path(x["ckpt_dir"])
        calls.append((str(d), x["family"], sorted(x["gate_stores"]), list(x["levels"])))
        if any(f in str(d) for f in fail):
            raise RuntimeError(f"forge eval failed on {d}")
        st = json.loads((d / "state.json").read_text())
        w = np.frombuffer(bytes(R.read_safetensors(d / "model.safetensors")[0]["w"]["data"]), np.float32).astype(np.float64)
        rate = float(min(1.0, max(0.0, w[2] - 1.0)))
        held = {str(lv): {"pass_rate": rate, "passed": round(rate * 64), "n": 64, "level": lv, "decoding": "greedy",
                          "results": "x"} for lv in x["levels"]}
        return {"dir": str(d), "sha256": st["model_sha256"], "family": x["family"], "tasks": None,
                "val": {name: loss(w, name) for name in x["gate_stores"]}, "heldout": held,
                "heldout_mean": rate if x["levels"] else None, "settings": {}}
    return evaluate


def fake_eval_ckpt(self, ckpt, s):
    return {"loss": loss(weights(self.root, ckpt), s), "perplexity": None, "next_token_acc": None, "ckpt": ckpt,
            "sha256": lr.model_sha(self.p(ckpt)), "store": s}


# --------------------------------------------------------------------------- resume an old state.json end to end

def test_fixture_is_shaped_like_the_live_state(tmp_path):
    live = ROOT / "runs/longrun/state.json"  # read only
    if not live.exists():
        pytest.skip("no live runs/longrun/state.json")
    s = json.loads(live.read_text())
    root, _ = world(tmp_path)
    f = json.loads((root / "runs/longrun/state.json").read_text())
    assert set(s) - {"config_reloaded_at"} <= set(f)
    assert {k for e in s["current"].values() for k in e} <= {k for e in f["current"].values() for k in e} | {
        "base", "metrics", "decided"}
    assert set(s["cycles"]["2"]["steps"]) <= set(lr.CYCLE_STEPS)
    assert set(f["cycles"]["2"]["steps"]) == set(s["cycles"]["2"]["steps"])
    for k, v in s["cycles"]["2"]["steps"].items():
        assert set(v) - {"benchmarks"} <= set(f["cycles"]["2"]["steps"][k]) | {"cmd", "spent_hours", "rounds", "kinds",
                                                                             "self_improvement_rounds", "heldout_pass", "audit",
                                                                             "budget_hours", "log", "ctl", "pid"}, k


def test_resume_old_state_runs_the_cycle_with_arbiter_and_rebase(tmp_path, monkeypatch):
    root, b = world(tmp_path, v3=True)
    calls, rsi_cmds = [], []
    monkeypatch.setattr(C, "evaluate", fake_evaluate(calls))
    monkeypatch.setattr(lr, "pid_alive", lambda pid: False)  # the old forge train exited with the old longrun
    monkeypatch.setattr(lr.LongRun, "eval_ckpt", fake_eval_ckpt)

    def forge_train(self, run, cmd, st, restricted):
        cfg = json.loads(self.p(cmd[cmd.index("--config") + 1]).read_text())
        src = cmd[cmd.index("--resume") + 1] if "--resume" in cmd else cfg.get("init_from")
        step = f"step-{int(cfg['max_steps']):06d}"
        ckpt(self.root, f"runs/{run}/{step}", weights(self.root, src) + 0.1, cfg.get("init_from"), lr.model_sha(self.p(src)))
        write(self.root / f"runs/{run}/latest.json", {"step": cfg["max_steps"], "dir": f"runs/{run}/{step}"})
        return 0, False

    def run_logged(self, cmd, logp, st=None, on_stop=None, stop_kill=False):
        if "training.rsi.loop" in cmd:
            rsi_cmds.append(cmd)
            model, champ, out = (cmd[cmd.index(f) + 1] for f in ("--model", "--champion", "--out"))
            if model.startswith("rouge"):
                return 1  # an RSI loop that fails: the arbiter still sees the FINAL
            d = f"{out}/rounds/rsi-{model}-r1/step-000060"
            sha = ckpt(self.root, d, weights(self.root, champ) + [0, 0, 0.3, 0], str(self.p(champ).resolve()),
                       lr.model_sha(self.p(champ)))
            write(self.p(out) / "champion.json", {"model": model, "dir": d, "model_sha256": sha, "round": 1, "history": []})
            write(self.p(out) / "audit.jsonl", json.dumps({"round": 1, "decision": "promote", "kind": "self-generated"}) + "\n")
            return 0
        if any("merge_search" in c for c in cmd):
            m = json.loads(self.p(cmd[cmd.index("--config") + 1]).read_text())
            bw = weights(self.root, m["base"])
            w = bw + sum(weights(self.root, c) - bw for c in m["children"])
            ckpt(self.root, m["out"], w, None, lr.model_sha(self.p(m["base"])), merge={"children": m["children"]})
            write(self.root / "runs" / m["name"] / "search.json", {"best": {"method": "linear", "lambda": 1.0}})
            return 0
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(lr.LongRun, "forge_train", forge_train)
    monkeypatch.setattr(lr.LongRun, "run_logged", run_logged)
    old = json.loads((root / "runs/longrun/state.json").read_text())

    assert orchestrator(root, max_cycles=2).run() == lr.EX_OK
    s = json.loads((root / "runs/longrun/state.json").read_text())
    assert s["status"] == "finished" and (root / "runs/longrun/FINISHED").exists()
    assert s["cycles"]["2"] == old["cycles"]["2"], "the finished cycle is untouched"
    c3 = s["cycles"]["3"]
    assert c3["status"] == "done" and c3["plan"] == old["cycles"]["3"]["plan"], "the cycle in progress resumes, not re-planned"
    assert list(c3["steps"]) == list(lr.CYCLE_STEPS)
    assert {k: v["status"] for k, v in c3["steps"].items() if v["status"] != "done"} == {"rouge_rsi": "error"}
    assert c3["steps"]["base"]["final"] == "runs/base-g3/FINAL" and (root / "runs/base-g3/step-000200").is_dir()
    assert c3["steps"]["base_gate"]["decision"] == "accept" and c3["parent_base"] == "runs/base-g3/FINAL"
    assert s["corpus"]["version"] == "v2", "corpus v3 is never adopted mid-cycle"

    # rebased inits: base-g3 + (champion - base-g2) won over the plain base, for both specialists
    q_init = json.loads((root / "runs/longrun/configs/quasnir-g3.json").read_text())["init_from"]
    r_init = json.loads((root / "runs/longrun/configs/rouge-g3.json").read_text())["init_from"]
    assert q_init == f"runs/quasnir-g3/{lr.REBASED_PREFIX}{b['q2'][:12]}-lam1"
    assert r_init == f"runs/rouge-g3/{lr.REBASED_PREFIX}{b['rr'][:12]}-lam1"
    ic = c3["steps"]["quasnir"]["init_choice"]
    assert ic["old_base"] == "runs/base-g2/FINAL" and ic["plain"] == "runs/base-g3/FINAL" and len(ic["candidates"]) == 3
    assert s["runs"]["quasnir-g3"]["base"] == "runs/base-g3/FINAL"
    assert (root / "runs/rouge-g3").stat().st_mode & 0o777 == 0o700
    q3 = weights(root, "runs/quasnir-g3/FINAL")
    assert np.allclose(q3, weights(root, "runs/base-g3/FINAL") + [0.5, 0, 0, 0] + 0.1, atol=1e-6)

    # automatic promotion through the arbiter: RSI with --anchor-tol, never --require-approval
    assert len(rsi_cmds) == 3
    assert all("--require-approval" not in c and c[c.index("--anchor-tol") + 1] == "0.02" for c in rsi_cmds)
    assert all(c3["steps"][f"{f}_gate"]["decision"] == "deferred" for f in ("quasnir", "rouge", "darus"))
    fams = c3["steps"]["champions"]["families"]
    assert {f: r["decision"] for f, r in fams.items()} == {"quasnir": "promote", "rouge": "promote", "darus": "promote"}
    assert s["current"]["quasnir"]["dir"] == "runs/rsi/quasnir-g3/rounds/rsi-quasnir-g3-r1/step-000060"
    assert s["current"]["quasnir"]["base"] == "runs/base-g3/FINAL" and s["current"]["quasnir"]["decided"]["by"] == "arbiter"
    assert s["current"]["rouge"]["dir"] == "runs/rouge-g3/FINAL" and s["current"]["rouge"]["metrics"]["heldout_mean"] > 0
    assert s["current"]["darus"]["dir"] == "runs/rsi/darus-g3/rounds/rsi-darus-g3-r1/step-000060"
    arb = [c for c in calls if len(c[3]) == 1]
    assert {(c[1], tuple(c[2])) for c in arb} == {("quasnir", ("code", "general")), ("rouge", ("base", "general")),
                                                  ("darus", ("code", "general"))}
    assert sum(1 for c in arb if c[1] == "rouge") == 2, "the failed rouge RSI left only incumbent and FINAL"

    champs = json.loads((root / "runs/champions.json").read_text())
    assert set(champs["champions"]) == {"base", "quasnir", "rouge", "darus"} and champs["cycle"] == 3
    assert champs["champions"]["rouge"]["restricted"] and champs["champions"]["quasnir"]["dir"] == s["current"]["quasnir"]["dir"]
    log = [json.loads(x) for x in (root / "runs/longrun/champions.jsonl").read_text().splitlines()]
    assert [(r["cycle"], r["family"], r["promoted"]) for r in log] == [(3, f, True) for f in ("quasnir", "rouge", "darus")]
    assert log[0]["incumbent"]["dir"] == "runs/quasnir-g2/step-000050" and log[0]["decision"]["ranking"][0]["chosen"]

    hist = [json.loads(x) for x in (root / "runs/longrun/history.jsonl").read_text().splitlines()]
    assert hist[-1]["cycle"] == 3 and hist[-1]["decisions"]["quasnir"] == "promote" and hist[-1]["decisions"]["base"] == "accept"
    report = json.loads((root / "runs/longrun/report-g3.json").read_text())
    assert report["champions"]["darus"]["decision"] == "promote" and report["champions_file"]["cycle"] == 3

    # hygiene: rebased inits of the finished runs are gone, every champion and its base stays
    assert not list((root / "runs/quasnir-g3").glob(lr.REBASED_PREFIX + "*"))
    for e in champs["champions"].values():
        assert (root / e["dir"] / "model.safetensors").exists() and (e["base"] is None or (root / e["base"]).exists())


def test_derived_base_of_entries_written_by_the_old_version(tmp_path):
    root, b = world(tmp_path)
    ckpt(root, "runs/quasnir-1/step-000010", [0.5, 0, 0, 0], "runs/base-s/FINAL", b["s"])
    o = orchestrator(root)
    o.init_state()
    cur = o.state["current"]
    assert o.derived_base(cur["quasnir"]) == "runs/base-g2/FINAL"  # the run's train config
    assert o.derived_base(cur["rouge"]) == "runs/base-g2/FINAL"  # an RSI round of rouge-g2
    assert o.derived_base(cur["darus"]) == "runs/base-g2/FINAL"  # a merge: parent_sha256
    assert o.derived_base({"run": "quasnir-1", "dir": "runs/quasnir-1/step-000010"}) == "runs/base-s/FINAL"  # lineage
    assert o.derived_base({"run": "x", "dir": "runs/nowhere"}) is None
    o.state["runs"]["quasnir-g2"]["config"] = "runs/longrun/configs/missing.json"
    assert o.derived_base(cur["quasnir"]) == "runs/base-g2/FINAL", "cycle records, then the checkpoint lineage"


# --------------------------------------------------------------------------- the arbiter alone

def arbiter_world(tmp_path, monkeypatch, fail=()):
    root, b = world(tmp_path)
    o = orchestrator(root)
    o.init_state()
    ckpt(root, "runs/quasnir-g3/step-000040", [1.6, 1.0, 1.1, 1.0], "runs/base-g2/FINAL", b["g2"])  # held-out +0.1, val ok
    (root / "runs/quasnir-g3/FINAL").symlink_to("step-000040")
    rsi = "runs/rsi/quasnir-g3/rounds/rsi-quasnir-g3-r1/step-000060"
    ckpt(root, rsi, [1.6, 1.0, 1.5, 1.0], str(root / "runs/quasnir-g3/step-000040"))  # held-out +0.5
    cyc = o.state["cycles"]["3"]
    cyc["parent_base"] = "runs/base-g2/FINAL"
    cyc["steps"].update(quasnir_rsi={"status": "done", "model_dir": rsi}, quasnir_gate={"status": "done", "decision": "deferred"})
    calls = []
    monkeypatch.setattr(C, "evaluate", fake_evaluate(calls, fail))
    return o, cyc, rsi, calls


def test_arbiter_promotes_the_best_and_logs_it(tmp_path, monkeypatch):
    o, cyc, rsi, calls = arbiter_world(tmp_path, monkeypatch)
    res = o.arbitrate("quasnir", 3, cyc)
    assert res["decision"] == "promote" and res["chosen"]["dir"] == rsi and o.state["current"]["quasnir"]["dir"] == rsi
    assert o.arbitrate("quasnir", 3, cyc)["reason"] == "decided before a restart" and len(calls) == 3


def test_arbiter_never_crowns_a_failed_or_missing_candidate(tmp_path, monkeypatch):
    o, cyc, rsi, calls = arbiter_world(tmp_path, monkeypatch, fail=("rsi-quasnir-g3",))
    res = o.arbitrate("quasnir", 3, cyc)  # the RSI champion cannot be measured: it drops out, the FINAL wins
    assert res["decision"] == "promote" and res["chosen"]["dir"] == "runs/quasnir-g3/FINAL" and len(res["errors"]) == 1

    o, cyc, rsi, calls = arbiter_world(tmp_path / "b", monkeypatch, fail=("quasnir-g2",))
    res = o.arbitrate("quasnir", 3, cyc)  # the incumbent cannot be measured: it stays, nobody else is measured
    assert res["decision"] == "keep" and o.state["current"]["quasnir"]["dir"] == "runs/quasnir-g2/step-000050"
    assert len(calls) == 1 and "could not be measured" in res["reason"]

    o, cyc, rsi, calls = arbiter_world(tmp_path / "c", monkeypatch)
    cyc["steps"]["quasnir_rsi"]["model_dir"] = "runs/rsi/quasnir-g3/rounds/gone/step-000060"
    cands, skipped = o.champion_candidates("quasnir", 3, cyc)
    assert [c["name"] for c in cands] == ["incumbent:quasnir-g2", "final:quasnir-g3"] and "missing" in skipped[0]["reason"]


def test_arbiter_keeps_a_better_incumbent(tmp_path, monkeypatch):
    o, cyc, rsi, calls = arbiter_world(tmp_path, monkeypatch)
    for d in (rsi, "runs/quasnir-g3/step-000040"):  # replace both challengers with ones that regress on code
        p = o.p(d)
        for f in p.iterdir():
            f.unlink()
        p.rmdir()
        ckpt(o.root, d, [1.2, 1.0, 1.0, 1.0])
    res = o.arbitrate("quasnir", 3, cyc)
    d = o.state["current"]["quasnir"]["decided"]
    assert res["decision"] == "keep" and d["by"] == "arbiter" and d["changed"] is False and d["cycle"] == 3
    assert o.state["current"]["quasnir"]["dir"] == "runs/quasnir-g2/step-000050"
    rec = json.loads((o.root / "runs/longrun/champions.jsonl").read_text().splitlines()[-1])
    assert not rec["promoted"] and any("final:quasnir-g3 is infeasible" in r for r in rec["decision"]["reasons"])


# --------------------------------------------------------------------------- hot reload, corpus, hygiene

def test_hot_reload_applies_valid_edits_and_ignores_invalid_ones(tmp_path):
    root, _ = world(tmp_path)
    o = orchestrator(root)
    o.init_state()
    p = root / "training/configs/longrun.json"
    assert not o.reload_config(), "unchanged file: nothing to do"
    cfg = json.loads(p.read_text())
    p.write_text(json.dumps({**cfg, "rsi": {**cfg["rsi"], "rounds": 3}}))
    assert o.reload_config() and o.cfg["rsi"]["rounds"] == 3 and o.state["config_reloaded_at"]
    good = o.cfg
    p.write_text(json.dumps({**cfg, "shares": {"base": 0.9}}))
    assert not o.reload_config() and o.cfg is good
    assert not o.reload_config(), "an invalid file is reported once"
    write(root / "training/configs/base-m.json", {"model": {**MODEL, "dim": 16}, "batch": 2, "seq_len": 8})
    p.write_text(json.dumps({**cfg, "base_config": "training/configs/base-m.json"}))
    assert not o.reload_config() and o.cfg is good, "a model change needs a restart"
    events = [json.loads(x)["msg"] for x in (root / "runs/longrun/events.jsonl").read_text().splitlines()]
    assert events.count("config reloaded") == 1 and sum("is invalid" in e for e in events) == 2


def test_corpus_v3_only_between_cycles_and_only_when_valid(tmp_path):
    root, _ = world(tmp_path, v3=True)
    o = orchestrator(root)
    o.init_state()
    assert not o.between_cycles()
    o.ensure_corpus(adopt=o.between_cycles())
    assert o.state["corpus"]["version"] == "v2"
    o.ensure_corpus(adopt=True)
    assert o.state["corpus"]["version"] == "v3" and o.store_meta("code") == "data/out/code-v3/train.meta.json"
    (root / "data/out/code-v3/tokenizer.json").write_text("other")
    o.ensure_corpus(adopt=True)
    assert o.state["corpus"]["version"] == "v2"


def test_hygiene_never_deletes_champions_their_bases_or_rebased_inits_in_use(tmp_path):
    root, b = world(tmp_path)
    o = orchestrator(root)
    o.init_state()
    # quasnir-g2 is superseded; its FINAL moves to step-000060, step-000050 is only named by champions.json
    for s in ("step-000055", "step-000060"):
        ckpt(root, f"runs/quasnir-g2/{s}", [1.5, 1, 1, 1], "runs/base-g2/FINAL", b["g2"])
    (root / "runs/quasnir-g2/FINAL").unlink()
    final(root, "quasnir-g2", "step-000060")
    (root / "runs/base-g3/latest.json").unlink()  # base-g3/step-000100 is protected only as a champion's base
    o.state["current"]["quasnir"] = {"gen": 3, "run": "quasnir-g3", "dir": "runs/quasnir-g3/step-000200",
                                     "base": "runs/base-g3/step-000100"}
    write(root / "runs/champions.json", {"champions": {"quasnir": {"dir": "runs/quasnir-g2/step-000050", "base": None}}})
    for s in ("step-000100", "step-000200"):
        ckpt(root, f"runs/quasnir-g3/{s}", [2, 1, 1, 1])
    (root / "runs/quasnir-g3/FINAL").symlink_to("step-000200")
    ckpt(root, f"runs/quasnir-g3/{lr.REBASED_PREFIX}abc-lam1", [1, 1, 1, 1])
    ckpt(root, f"runs/rouge-g4/{lr.REBASED_PREFIX}def-lam1", [1, 1, 1, 1])  # rouge-g4 still trains from it
    write(root / "runs/longrun/configs/rouge-g4.json", {"init_from": f"runs/rouge-g4/{lr.REBASED_PREFIX}def-lam1"})
    o.state["runs"].update({"quasnir-g3": {"finished": True, "family": "quasnir"},
                            "rouge-g4": {"finished": False, "family": "rouge", "config": "runs/longrun/configs/rouge-g4.json"},
                            "base-g3": {"finished": True, "family": "base"}})
    o.state["cycle"] = 4
    o.step_hygiene(4, {}, {})
    assert (root / "runs/quasnir-g2/step-000050/model.safetensors").exists(), "named by runs/champions.json"
    assert (root / "runs/quasnir-g2/step-000060").exists(), "a FINAL"
    assert (root / "runs/base-g3/step-000100").exists(), "the base of a current champion"
    assert not (root / "runs/quasnir-g2/step-000055").exists(), "superseded and unprotected"
    assert (root / "runs/quasnir-g3/step-000200").exists()
    assert not (root / f"runs/quasnir-g3/{lr.REBASED_PREFIX}abc-lam1").exists(), "rebased init of a finished run"
    assert (root / f"runs/rouge-g4/{lr.REBASED_PREFIX}def-lam1/model.safetensors").exists(), "rebased init still in use"
    assert (root / "runs/rsi/rouge-g2/rounds/rsi-rouge-g2-r2/step-000060").exists(), "the current rouge champion"
