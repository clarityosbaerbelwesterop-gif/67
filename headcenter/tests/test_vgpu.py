"""VGPU pool: proof of compute is exact, cheating is caught, capacity is timed by the coordinator."""
import time

import pytest
from fastapi.testclient import TestClient

from headcenter.backend import vgpu as V
from headcenter.backend.app import create_app


def test_proof_rows_match_the_independent_check():
    rows = V.solve(42, 64, 5)
    assert len(rows) == 64
    check = V.check_rows(42, 64, 5, [0, 13, 63])
    assert all(rows[i] == d for i, d in check.items())
    assert V.check_rows(43, 64, 5, [0])[0] != rows[0], "another seed, another result"


def test_pool_verifies_times_and_rejects_cheats(tmp_path, monkeypatch):
    p = V.Pool(tmp_path, samples=8)
    nid = p.register({"label": "mac", "kind": "apple", "claimed_tflops": 0.001})["node_id"]
    ch = p.challenge(nid)
    assert ch["n"] >= 256 and ch["k"] >= 2
    rows = V.solve(ch["seed"], ch["n"], ch["k"])
    res = p.proof({"challenge_id": ch["challenge_id"], "rows": rows})
    assert res["verified"] and res["tflops"] > 0 and res["checked_rows"] == 8
    with pytest.raises(V.VGPUError):  # a challenge is answered once
        p.proof({"challenge_id": ch["challenge_id"], "rows": rows})
    ch2 = p.challenge(nid)
    fake = V.solve(ch2["seed"], ch2["n"], ch2["k"])
    fake[: ch2["n"] // 2] = ["0" * 64] * (ch2["n"] // 2)  # skipped half of the work
    monkeypatch.setattr(V.secrets.SystemRandom, "sample", lambda self, pop, k: list(range(k)))
    with pytest.raises(V.VGPUError) as e:
        p.proof({"challenge_id": ch2["challenge_id"], "rows": fake})
    assert e.value.status == 422 and p.nodes[nid]["failed"] == 1
    ov = p.overview()
    assert ov["live_nodes"] == 1 and ov["verified_tflops"] == pytest.approx(res["tflops"])
    assert ov["b300_equivalents"] == pytest.approx(res["tflops"] / 2500)
    # capacity expires without fresh proofs; state survives a restart
    p.nodes[nid]["proofs"][0]["ts"] -= 7 * 3600
    p._save()
    assert V.Pool(tmp_path).overview()["verified_tflops"] == 0.0


def test_slow_answers_count_as_slow(tmp_path):
    p = V.Pool(tmp_path)
    nid = p.register({"label": "box", "kind": "cpu", "claimed_tflops": 0.001})["node_id"]
    ch = p.challenge(nid)
    rows = V.solve(ch["seed"], ch["n"], ch["k"])
    p.challenges[ch["challenge_id"]]["issued"] -= 100  # the coordinator's clock decides
    res = p.proof({"challenge_id": ch["challenge_id"], "rows": rows})
    assert res["seconds"] >= 100


def test_api_flow(tmp_path):
    app = create_app(tmp_path, token="t", poll=0.05)
    h = {"Authorization": "Bearer t"}
    with TestClient(app) as c:
        assert c.get("/api/vgpu").status_code == 401
        nid = c.post("/api/vgpu/register", json={"label": "ipad", "kind": "webgpu", "claimed_tflops": 0.001}, headers=h).json()["node_id"]
        ch = c.get(f"/api/vgpu/challenge?node_id={nid}", headers=h).json()
        rows = V.solve(ch["seed"], ch["n"], ch["k"])
        assert c.post("/api/vgpu/proof", json={"challenge_id": ch["challenge_id"], "rows": rows}, headers=h).json()["verified"]
        ov = c.get("/api/vgpu", headers=h).json()
        assert ov["live_nodes"] == 1 and ov["nodes"][0]["label"] == "ipad"
        assert c.post("/api/vgpu/register", json={"label": "bad\nlabel"}, headers=h).status_code == 400
        assert c.get("/api/vgpu/challenge?node_id=nope", headers=h).status_code == 404
