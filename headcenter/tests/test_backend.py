"""Headcenter backend tests: ingestion, control channel, agents, HTTP/WebSocket API."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from headcenter.backend.agents import JitOptimizer, SiliconWatchdog, profile
from headcenter.backend.app import Hub, create_app
from headcenter.backend.control import CommandError, Controller, NotLive, validate
from headcenter.backend.telemetry import RunState, Tail, load_max_steps


def tele(step, loss=3.0, lr=1e-3, gn=1.0, ts=None, gflops=200.0, **kw):
    ns = 1e9
    rec = {
        "type": "telemetry", "run": "r", "step": step, "loss": loss, "lr": lr, "grad_norm": gn,
        "tokens_per_s": 1000.0, "tflops": 0.15, "step_ms": 5000.0, "paused": False, "tokens": step * 8192,
        "ops": {"MATMUL": {"calls": 10, "ns": ns, "flops": gflops * ns, "bytes": 1.0},
                "SOFTMAX_CAUSAL": {"calls": 2, "ns": ns / 10, "flops": 1e6, "bytes": 1.0}},
        "hbvm": {"capacity": 1000, "used": 500, "peak": 600, "live": 3, "fragmentation": 0.1},
        "ts": ts if ts is not None else 1000.0 + step,
    }
    rec.update(kw)
    return rec


class Reader:
    """Opens a FIFO like train_all.sh does (read-write) and collects lines."""

    def __init__(self, path: Path):
        os.mkfifo(path)
        self.fd = os.open(path, os.O_RDWR)
        self.lines: list[str] = []
        self.buf = b""

    def drain(self, timeout=1.0) -> list[str]:
        os.set_blocking(self.fd, False)
        end = time.time() + timeout
        while time.time() < end:
            try:
                chunk = os.read(self.fd, 65536)
            except BlockingIOError:
                chunk = b""
            if chunk:
                self.buf += chunk
                *done, self.buf = self.buf.split(b"\n")
                self.lines += [d.decode() for d in done]
                end = time.time() + 0.05
            else:
                time.sleep(0.01)
        return self.lines

    def close(self):
        os.close(self.fd)


# ---------------------------------------------------------------- telemetry


def test_tail_reads_only_complete_lines_and_restarts_on_truncation(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":')
    t = Tail(p)
    assert t.read() == ['{"a":1}']
    with p.open("ab") as f:
        f.write(b"2}\n")
    assert t.read() == ['{"b":2}']
    assert t.read() == []
    p.write_bytes(b'{"c":3}\n')
    assert t.read() == ['{"c":3}']


def test_runstate_tracks_history_evals_status_and_eta():
    r = RunState("r", max_steps=100)
    for s in (10, 20, 30):
        r.ingest(tele(s, ts=1000.0 + 50 * s))
    r.ingest({"type": "eval", "step": 30, "val_loss": 2.5, "val_acc": 0.4, "ts": 2600.0})
    r.ingest({"type": "event", "level": "warn", "msg": "x", "ts": 2601.0})
    r.ingest({"type": "checkpoint", "path": "p", "sha256": "ab", "step": 30, "ts": 2602.0})
    s = r.summary()
    assert s["status"] == "running" and s["step"] == 30
    assert [h[0] for h in s["history"]] == [10, 20, 30]
    assert s["evals"] == [[30, 2.5, 0.4]]
    assert s["eta_s"] == pytest.approx(70 * 5.0)
    assert s["b300_fraction"] == pytest.approx(0.15 / 2500)
    assert r.median_interval() == 500.0
    r.ingest({"type": "done", "step": 100, "ts": 3000.0})
    assert r.summary()["status"] == "done" and r.eta_s() is None


def test_nonfinite_loss_is_null_in_history():
    r = RunState("r")
    r.ingest(tele(1, loss=None))
    assert r.history[-1][1] is None


def test_load_max_steps(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"max_steps": 42}))
    assert load_max_steps(tmp_path, "x") == 42
    assert load_max_steps(tmp_path, "missing") is None
    assert load_max_steps(None, "x") is None


# ---------------------------------------------------------------- control


@pytest.mark.parametrize(
    "cmd,canon",
    [
        ({"cmd": "pause", "junk": 1}, {"cmd": "pause"}),
        ({"cmd": "set_lr", "value": 3e-4}, {"cmd": "set_lr", "value": 3e-4}),
        ({"cmd": "set_threads", "value": 2}, {"cmd": "set_threads", "value": 2}),
        ({"cmd": "swap_kernel", "variant": "avx2-fma"}, {"cmd": "swap_kernel", "op": "MATMUL", "variant": "avx2-fma"}),
    ],
)
def test_validate_accepts_protocol_commands(cmd, canon):
    assert validate(cmd) == canon


@pytest.mark.parametrize(
    "cmd",
    [
        None, [], {"cmd": "rm"}, {"cmd": "set_lr", "value": 0}, {"cmd": "set_lr", "value": 1.5},
        {"cmd": "set_lr", "value": True}, {"cmd": "set_lr", "value": float("nan")},
        {"cmd": "set_threads", "value": 0}, {"cmd": "set_threads", "value": 2.0},
        {"cmd": "swap_kernel", "variant": "cuda"}, {"cmd": "swap_kernel", "variant": "scalar", "op": "XENT"},
    ],
)
def test_validate_rejects_everything_else(cmd):
    with pytest.raises(CommandError):
        validate(cmd)


def test_controller_writes_canonical_line_to_live_fifo(tmp_path):
    reader = Reader(tmp_path / "run1.ctl")
    try:
        c = Controller(tmp_path)
        assert c.send("run1", {"cmd": "set_lr", "value": 1e-4, "x": 1}) == {"cmd": "set_lr", "value": 1e-4}
        assert reader.drain() == ['{"cmd":"set_lr","value":0.0001}']
    finally:
        reader.close()


def test_controller_refuses_dead_missing_and_unsafe_targets(tmp_path):
    c = Controller(tmp_path)
    with pytest.raises(NotLive):
        c.send("ghost", {"cmd": "pause"})
    os.mkfifo(tmp_path / "dead.ctl")
    with pytest.raises(NotLive):
        c.send("dead", {"cmd": "pause"})
    (tmp_path / "plain.ctl").write_text("")
    with pytest.raises(CommandError):
        c.send("plain", {"cmd": "pause"})
    with pytest.raises(CommandError):
        c.send("../etc", {"cmd": "pause"})


# ---------------------------------------------------------------- agents


def run_agent(agent, recs, live=True, name="r"):
    state, out = RunState(name), []
    for rec in recs:
        state.ingest(rec)
        out += agent.observe(state, rec, live)
    return state, out


def test_watchdog_pauses_once_on_nonfinite_loss_and_rearms():
    w = SiliconWatchdog()
    _, out = run_agent(w, [tele(1), tele(2, loss=None), tele(3, loss=None), tele(4), tele(5, loss=None)])
    pauses = [d for d in out if d.action == {"cmd": "pause"}]
    assert [d.step for d in pauses] == [2, 5]
    assert all(d.severity == "critical" for d in pauses)


def test_watchdog_halves_lr_peak_on_sustained_spike_only():
    w = SiliconWatchdog()
    recs = [tele(s, loss=3.0, lr=1e-3 if s > 5 else 2e-4 * s) for s in range(1, 30)]
    recs += [tele(30, loss=5.0)]  # one spike line: no action
    recs += [tele(31, loss=3.0)]
    recs += [tele(32, loss=5.0), tele(33, loss=5.1)]  # sustained
    _, out = run_agent(w, recs)
    acts = [d for d in out if d.action]
    assert len(acts) == 1 and acts[0].step == 33
    assert acts[0].action == {"cmd": "set_lr", "value": pytest.approx(5e-4)}


def test_watchdog_grad_explosion_memory_and_replay_is_silent():
    w = SiliconWatchdog()
    recs = [tele(s, gn=1.0) for s in range(1, 25)] + [tele(25, gn=100.0)]
    recs += [tele(26, hbvm={"capacity": 100, "used": 99, "peak": 99, "live": 1, "fragmentation": 0.7})]
    _, out = run_agent(w, recs)
    reasons = " | ".join(d.reason for d in out)
    assert "gradient norm" in reasons and "HBVM peak" in reasons and "fragmentation" in reasons
    _, replay = run_agent(SiliconWatchdog(), recs + [tele(27, loss=None)], live=False)
    assert replay == []


def test_watchdog_stall_detection():
    w = SiliconWatchdog()
    state, _ = run_agent(w, [tele(s, ts=1000.0 + 10 * s) for s in range(1, 6)])
    runs = {"r": state}
    assert w.tick(runs, now=state.last_ts + 100) == []
    out = w.tick(runs, now=state.last_ts + 181)
    assert len(out) == 1 and "no telemetry" in out[0].reason
    assert w.tick(runs, now=state.last_ts + 400) == []  # edge-triggered


def test_jit_reautotunes_on_sustained_degradation_with_cooldown():
    j = JitOptimizer()
    recs = [tele(s, gflops=200.0, ts=1000.0 + s) for s in range(1, 6)]
    recs += [tele(s, gflops=100.0, ts=1000.0 + s) for s in range(6, 9)]
    _, out = run_agent(j, recs)
    acts = [d for d in out if d.action]
    assert len(acts) == 1 and acts[0].action == {"cmd": "autotune"} and acts[0].step == 8
    infos = [d for d in out if d.severity == "info"]
    assert infos and "MATMUL" in infos[0].reason


def test_profile_flags_non_gemm_dominance():
    ops = {"MATMUL": {"ns": 40, "flops": 4000}, "SOFTMAX_CAUSAL": {"ns": 60, "flops": 60}}
    assert "non-GEMM ops take 60%" in profile(ops)
    assert profile({}) is None


# ---------------------------------------------------------------- hub + api


def write(path: Path, *recs):
    with path.open("a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_hub_executes_in_act_mode_and_only_advises_otherwise(tmp_path):
    reader = Reader(tmp_path / "r.ctl")
    try:
        log = tmp_path / "r.jsonl"
        write(log, tele(1))
        hub = Hub(tmp_path, modes={"silicon-watchdog": "act"})
        hub.scan()
        write(log, tele(2, loss=None))
        msgs = hub.scan()
        d = [m for m in msgs if m["type"] == "decision"][0]
        assert d["executed"] and d["mode"] == "act" and d["action"] == {"cmd": "pause"}
        assert reader.drain() == ['{"cmd":"pause"}']
        audit = [json.loads(x) for x in (tmp_path / "headcenter" / "decisions.jsonl").read_text().splitlines()]
        assert [a["executed"] for a in audit if a["agent"] == "silicon-watchdog"] == [True]

        hub.set_mode("silicon-watchdog", "advise")
        write(log, tele(3), tele(4, loss=None))
        d = [m for m in hub.scan() if m["type"] == "decision"][0]
        assert not d["executed"] and d["mode"] == "advise"
        assert reader.drain(0.2) == ['{"cmd":"pause"}']
    finally:
        reader.close()


def test_hub_never_acts_on_replayed_history(tmp_path):
    write(tmp_path / "old.jsonl", tele(1), tele(2, loss=None))
    hub = Hub(tmp_path)
    msgs = hub.scan()
    assert [m["type"] for m in msgs] == ["run"]
    assert not (tmp_path / "headcenter").exists()


def test_api_requires_token(tmp_path):
    app = create_app(tmp_path, token="s3cret", poll=0.05)
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/").status_code == 200
        assert c.get("/api/runs").status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.post("/api/runs/x/cmd", json={"cmd": "pause"}).status_code == 401
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws?token=nope") as ws:
                ws.receive_json()


def test_api_commands_and_modes(tmp_path):
    reader = Reader(tmp_path / "r.ctl")
    try:
        write(tmp_path / "r.jsonl", tele(1))
        app = create_app(tmp_path, token="t", poll=0.05)
        h = {"Authorization": "Bearer t"}
        with TestClient(app) as c:
            r = c.post("/api/runs/r/cmd", json={"cmd": "checkpoint"}, headers=h)
            assert r.status_code == 200 and r.json()["cmd"] == {"cmd": "checkpoint"}
            assert c.post("/api/runs/r/cmd", json={"cmd": "format_disk"}, headers=h).status_code == 400
            assert c.post("/api/runs/none/cmd", json={"cmd": "pause"}, headers=h).status_code == 409
            assert c.post("/api/agents/jit-optimizer", json={"mode": "advise"}, headers=h).json() == {"ok": True}
            assert c.post("/api/agents/jit-optimizer", json={"mode": "yolo"}, headers=h).status_code == 400
            assert c.get("/api/runs", headers=h).json()["agents"]["jit-optimizer"] == "advise"
        assert reader.drain() == ['{"cmd":"checkpoint"}']
    finally:
        reader.close()


def test_websocket_snapshot_live_records_and_commands(tmp_path):
    reader = Reader(tmp_path / "r.ctl")
    try:
        log = tmp_path / "r.jsonl"
        write(log, tele(1))
        app = create_app(tmp_path, token="t", poll=0.02)
        with TestClient(app) as c:
            with c.websocket_connect("/ws?token=t") as ws:
                snap = ws.receive_json()
                while snap["type"] != "snapshot":
                    snap = ws.receive_json()
                deadline = time.time() + 5
                while "r" not in snap["runs"] and time.time() < deadline:
                    m = ws.receive_json()
                    if m["type"] == "run":
                        snap["runs"][m["run"]["name"]] = m["run"]
                assert snap["runs"]["r"]["step"] == 1
                write(log, tele(2, loss=2.5))
                m = ws.receive_json()
                while m["type"] != "record":
                    m = ws.receive_json()
                assert m["record"]["step"] == 2 and m["summary"]["step"] == 2
                ws.send_json({"type": "cmd", "run": "r", "cmd": {"cmd": "autotune"}, "id": "7"})
                ws.send_json({"type": "cmd", "run": "r", "cmd": {"cmd": "nope"}, "id": "8"})
                acks = {}
                while len(acks) < 2:
                    m = ws.receive_json()
                    if m["type"] == "ack":
                        acks[m["id"]] = m
                assert acks["7"]["ok"] and acks["7"]["cmd"] == {"cmd": "autotune"}
                assert not acks["8"]["ok"] and "unknown command" in acks["8"]["error"]
        assert reader.drain() == ['{"cmd":"autotune"}']
    finally:
        reader.close()


def make_ckpt(d: Path, payload=b"weights"):
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(payload)
    (d / "state.json").write_text('{"step": 1}')


def test_models_served_from_final_or_latest_inside_runs_only(tmp_path):
    runs = tmp_path / "runs"
    make_ckpt(runs / "a" / "step-000010", b"A10")
    (runs / "a" / "latest.json").write_text(json.dumps({"dir": "runs/a/step-000010"}))
    make_ckpt(runs / "b" / "step-000020", b"B20")
    make_ckpt(runs / "b" / "step-000030", b"B30")
    (runs / "b" / "latest.json").write_text(json.dumps({"dir": "runs/b/step-000030"}))
    (runs / "b" / "FINAL").symlink_to(runs / "b" / "step-000020")
    make_ckpt(tmp_path / "outside", b"secret")
    (runs / "evil").mkdir()
    (runs / "evil" / "latest.json").write_text(json.dumps({"dir": str(tmp_path / "outside")}))
    tok = tmp_path / "tokenizer.json"
    tok.write_text('{"version": 2}')
    app = create_app(runs, token="k", poll=0.05, tokenizer=tok)
    with TestClient(app) as c:
        assert c.get("/models/k/a/model.safetensors").content == b"A10"
        assert c.get("/models/k/b/model.safetensors").content == b"B20"  # FINAL wins
        assert c.get("/models/k/b/state.json").json() == {"step": 1}
        assert c.get("/models/k/x/tokenizer.json").json() == {"version": 2}
        assert c.get("/models/wrong/a/model.safetensors").status_code == 401
        assert c.get("/models/k/evil/model.safetensors").status_code == 404
        assert c.get("/models/k/a/latest.json").status_code == 404
        assert c.get("/models/k/..%2Fa/model.safetensors").status_code == 404
        listing = c.get("/api/models", headers={"Authorization": "Bearer k"}).json()["models"]
        assert set(listing) == {"a", "b"} and listing["b"]["final"] is True
        assert c.get("/webgpu/").status_code == 200
        assert "javascript" in c.get("/webgpu/forge-webgpu.mjs").headers["content-type"]
        assert c.get("/webgpu/package.json").status_code == 404


def test_results_are_served_and_benchmarks_are_not_a_run(tmp_path):
    write(tmp_path / "r.jsonl", tele(1))
    (tmp_path / "report.json").write_text(json.dumps({"evaluations": {"darus-1": {"code": {"loss": 2.0}}}, "accepted": True}))
    (tmp_path / "darus-1").mkdir()
    (tmp_path / "darus-1" / "search.json").write_text(json.dumps(
        {"selection": {"seed": 777}, "best_parent": {"code": 2.1}, "best": {"method": "ties", "objective": -0.01},
         "candidates": [{}, {}, {}]}))
    write(tmp_path / "benchmarks.jsonl", {"ckpt": "runs/darus-1/FINAL", "suite": "humaneval", "passed": 0, "problems": 164, "pass@1": 0.0},
          {"type": "finished"})
    app = create_app(tmp_path, token="t", poll=0.05)
    with TestClient(app) as c:
        res = c.get("/api/results", headers={"Authorization": "Bearer t"}).json()
        assert res["report"]["accepted"] is True
        assert res["search"]["candidates"] == 3 and res["search"]["best"]["method"] == "ties"
        assert [b["suite"] for b in res["benchmarks"]] == ["humaneval"]
        assert c.get("/api/results").status_code == 401
        runs = c.get("/api/runs", headers={"Authorization": "Bearer t"}).json()
        assert set(runs["runs"]) == {"r"} and runs["results"]["report"]["accepted"] is True
