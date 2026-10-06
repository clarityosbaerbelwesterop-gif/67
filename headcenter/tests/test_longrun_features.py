"""Restricted models (ACL), controlled-RSI approvals, long-run view and the swarm rendezvous."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from headcenter.backend.access import token_hash
from headcenter.backend.app import create_app

ROOT = Path(__file__).resolve().parents[2]
H = {"Authorization": "Bearer op"}


def ckpt(d: Path, payload: bytes) -> None:
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(payload)
    (d / "state.json").write_text("{}")


def test_restricted_model_only_for_listed_companies(tmp_path):
    runs = tmp_path / "runs"
    ckpt(runs / "rouge-g2" / "step-000010", b"SECRET")
    (runs / "rouge-g2" / "FINAL").symlink_to(runs / "rouge-g2" / "step-000010")
    ckpt(runs / "quasnir-g2" / "step-000010", b"OPEN")
    (runs / "quasnir-g2" / "FINAL").symlink_to(runs / "quasnir-g2" / "step-000010")
    acl = tmp_path / "acl.json"
    acl.write_text(json.dumps({"restricted": {"rouge-": [token_hash("acme-key")]}, "companies": {token_hash("acme-key"): "ACME"}}))
    with TestClient(create_app(runs, token="op", poll=0.05, acl=acl)) as c:
        assert c.get("/models/acme-key/rouge-g2/model.safetensors").content == b"SECRET"
        assert c.get("/models/op/rouge-g2/model.safetensors").status_code == 403, "operator token alone must not open"
        assert c.get("/models/other/rouge-g2/model.safetensors").status_code == 403
        assert c.get("/models/op/quasnir-g2/model.safetensors").content == b"OPEN"
        assert c.get("/models/acme-key/quasnir-g2/model.safetensors").status_code == 401
        listing = c.get("/api/models", headers=H).json()["models"]
        assert listing["rouge-g2"]["restricted"] is True and listing["quasnir-g2"]["restricted"] is False
    log = [json.loads(x) for x in (runs / "headcenter" / "access.jsonl").read_text().splitlines()]
    assert [(r["allowed"], r["company"]) for r in log] == [(True, "ACME"), (False, None), (False, None)]
    assert "acme-key" not in acl.read_text() and "acme-key" not in (runs / "headcenter" / "access.jsonl").read_text()


def test_default_policy_restricts_rouge_to_nobody(tmp_path):
    runs = tmp_path / "runs"
    ckpt(runs / "rouge-1" / "step-000001", b"X")
    (runs / "rouge-1" / "latest.json").write_text(json.dumps({"dir": str(runs / "rouge-1" / "step-000001")}))
    with TestClient(create_app(runs, token="op", poll=0.05, acl=tmp_path / "missing.json")) as c:
        assert c.get("/models/op/rouge-1/model.safetensors").status_code == 403


def test_rsi_approval_only_while_pending_and_stop(tmp_path):
    runs = tmp_path / "runs"
    d = runs / "rsi" / "darus-g2"
    d.mkdir(parents=True)
    (d / "audit.jsonl").write_text(json.dumps({"round": 1, "decision": "reject", "kind": "supervised-warmstart"}) + "\n")
    with TestClient(create_app(runs, token="op", poll=0.05)) as c:
        assert c.post("/api/rsi/darus-g2/approve", json={"round": 2, "approved": True}, headers=H).status_code == 409
        (d / "pending.json").write_text(json.dumps({"round": 2, "candidate_sha256": "ab" * 32}))
        assert c.post("/api/rsi/darus-g2/approve", json={"round": 3, "approved": True}, headers=H).status_code == 409
        assert c.post("/api/rsi/darus-g2/approve", json={"round": 2, "approved": True}).status_code == 401
        r = c.post("/api/rsi/darus-g2/approve", json={"round": 2, "approved": True}, headers=H)
        assert r.status_code == 200
        assert json.loads((d / "approve-2.json").read_text())["approved"] is True
        assert c.post("/api/rsi/darus-g2/approve", json={"round": 2, "approved": False}, headers=H).status_code in (409, 400)
        assert json.loads((d / "approve-2.json").read_text())["approved"] is True, "a decision is never overwritten"
        assert c.post("/api/rsi/darus-g2/stop", headers=H).status_code == 200 and (d / "STOP").exists()
        assert c.post("/api/rsi/../etc/stop", headers=H).status_code in (400, 404)
        ov = c.get("/api/rsi", headers=H).json()
        assert "darus-g2" in json.dumps(ov)
        assert c.get("/api/longrun", headers=H).status_code == 200


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_swarm_rendezvous_assigns_consistent_ranks(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    cfg = json.loads((ROOT / "training/configs/smoke.json").read_text())
    (runs / "swarm-smoke.json").write_text(json.dumps(cfg))
    with TestClient(create_app(runs, token="op", poll=0.05)) as c:
        assert c.post("/api/swarm/join", json={"group": "g", "host": "127.0.0.1", "port": 47000, "threads": 1}, headers=H).status_code == 404
        assert c.post("/api/swarm/g/open", json={"world": 2, "config": str(runs / "swarm-smoke.json"), "inner_steps": 10}, headers=H).status_code == 200
        assert c.post("/api/swarm/g/open", json={"world": 2, "config": "/etc/passwd.json"}, headers=H).status_code >= 400
        assert c.post("/api/swarm/join", json={"group": "g", "host": "127.0.0.1", "port": 80, "threads": 1}, headers=H).status_code == 400
        w = [c.post("/api/swarm/join", json={"group": "g", "host": "127.0.0.1", "port": 47000 + i, "threads": 1}, headers=H).json()["worker_id"] for i in range(2)]
        a = [c.get(f"/api/swarm/g/assignment?worker_id={x}", headers=H) for x in w]
        assert all(r.status_code == 200 for r in a)
        a = [r.json() for r in a]
        assert sorted(x["rank"] for x in a) == [0, 1] and a[0]["peers"] == a[1]["peers"] == ["127.0.0.1:47000", "127.0.0.1:47001"]
        assert a[0]["config_sha256"] == a[1]["config_sha256"]


@pytest.mark.skipif(not (ROOT / "target/release/forge").exists(), reason="needs target/release/forge")
def test_two_real_workers_train_through_the_rendezvous(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    cfg = json.loads((ROOT / "training/configs/smoke.json").read_text())
    cfg.update(run="swarm-e2e", out_dir=str(tmp_path / "out"), threads=1, autotune=False, max_steps=40)
    (runs / "swarm-e2e.json").write_text(json.dumps(cfg))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(runs, token="op", poll=0.2), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        import httpx
        r = httpx.post(f"http://127.0.0.1:{port}/api/swarm/e2e/open", headers=H,
                       json={"world": 2, "config": str(runs / "swarm-e2e.json"), "inner_steps": 10, "compression": "int8"})
        assert r.status_code == 200, r.text
        p0, p1 = free_port(), free_port()
        env = {**os.environ, "HEADCENTER_TOKEN": "op"}
        procs = [subprocess.Popen([sys.executable, str(ROOT / "scripts/swarm_worker.py"), "--headcenter", f"http://127.0.0.1:{port}",
                                   "--group", "e2e", "--host", "127.0.0.1", "--port", str(p), "--threads", "1", "--no-bench", "--poll", "0.2", "--log-dir", str(tmp_path)],
                                  cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for p in (p0, p1)]
        outs = [p.communicate(timeout=240)[0] for p in procs]
        assert all(p.returncode == 0 for p in procs), outs
        assert all('"type":"done"' in o.replace(" ", "") for o in outs), outs
        ov = httpx.get(f"http://127.0.0.1:{port}/api/swarm", headers=H).json()
        assert "done" in json.dumps(ov)
        assert sorted(p.name for p in tmp_path.glob("e2e-w*.jsonl")) == ["e2e-w0.jsonl", "e2e-w1.jsonl"]
    finally:
        server.should_exit = True
        t.join(timeout=10)
