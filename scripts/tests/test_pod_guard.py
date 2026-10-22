"""jobs/pod_guard.sh: the self-check, ending a pod (runpodctl, GraphQL, REST in that order), the deadline and done
triggers, and the public status copy with secrets redacted. A local HTTP server stands in for RunPod."""
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "jobs/pod_guard.sh"
POD = "abc123xyz789"


@pytest.fixture()
def api():
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["query"]
            calls.append(("POST", q, self.headers.get("Authorization")))
            if "podTerminate" in q:
                return self._reply(200, {"errors": [{"message": "denied"}]} if H.deny else {"data": {"podTerminate": None}})
            return self._reply(200, {"data": {"pod": {"id": POD, "desiredStatus": "RUNNING"}}} if H.known else {"data": {"pod": None}})

        def do_DELETE(self):
            calls.append(("DELETE", self.path, self.headers.get("Authorization")))
            self._reply(200, {})

    H.known, H.deny = True, False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, calls, H
    srv.shutdown()


def run(args, base, tmp_path, runpodctl=None, **env):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    if runpodctl is not None:
        (bindir / "runpodctl").write_text(f"#!/bin/sh\necho \"$@\" >> {tmp_path}/ctl.log\nexit {runpodctl}\n")
        (bindir / "runpodctl").chmod(0o755)
    e = {"PATH": f"{bindir}:/usr/bin:/bin", "RUNPOD_POD_ID": POD, "RUNPOD_API_KEY": "k" * 20,
         "RUNPOD_GQL_URL": f"{base}/graphql", "RUNPOD_REST_URL": base, "HOME": str(tmp_path), **env}
    return subprocess.run(["bash", str(GUARD), *args], capture_output=True, text=True, env=e, timeout=60)


def test_check(api, tmp_path):
    base, calls, H = api
    r = run(["check"], base, tmp_path)
    assert r.returncode == 0 and json.loads(r.stdout) == {"ok": True, "how": "graphql"}
    assert calls[0][2] == "Bearer " + "k" * 20 and POD in calls[0][1]
    H.known = False
    r = run(["check"], base, tmp_path)
    assert r.returncode == 1 and json.loads(r.stdout)["ok"] is False
    r = run(["check"], base, tmp_path, RUNPOD_API_KEY="")
    assert r.returncode == 1 and "missing" in r.stdout


def test_end_prefers_runpodctl_then_graphql_then_rest(api, tmp_path):
    base, calls, H = api
    assert run(["end"], base, tmp_path, runpodctl=0).returncode == 0
    assert (tmp_path / "ctl.log").read_text().strip() == f"remove pod {POD}" and not calls
    assert run(["end"], base, tmp_path, runpodctl=1).returncode == 0          # runpodctl fails -> GraphQL
    assert any("podTerminate" in c[1] for c in calls if c[0] == "POST")
    H.deny = True
    calls.clear()
    assert run(["end"], base, tmp_path).returncode == 0                       # GraphQL denied -> REST DELETE
    assert ("DELETE", f"/pods/{POD}", "Bearer " + "k" * 20) in calls


def test_watch_deadline_and_done(api, tmp_path):
    base, calls, H = api
    later = str(int(time.time()) + 3600)
    assert run(["watch", later], base, tmp_path, runpodctl=0, GUARD_ONCE="1").returncode == 2   # nothing yet
    assert not (tmp_path / "ctl.log").exists()
    assert run(["watch", str(int(time.time()) - 1)], base, tmp_path, runpodctl=0, GUARD_ONCE="1").returncode == 0
    assert "remove pod" in (tmp_path / "ctl.log").read_text()
    (tmp_path / "ctl.log").unlink()
    done = tmp_path / "exit_code"
    done.write_text("0")
    assert run(["watch", later, str(done), "3600"], base, tmp_path, runpodctl=0, GUARD_ONCE="1").returncode == 2
    os.utime(done, (time.time() - 700, time.time() - 700))
    assert run(["watch", later, str(done)], base, tmp_path, runpodctl=0, GUARD_ONCE="1").returncode == 0
    assert "remove pod" in (tmp_path / "ctl.log").read_text()


def test_publish_redacts(api, tmp_path):
    base, _, _ = api
    src, dst = tmp_path / "out", tmp_path / "pub"
    src.mkdir()
    secret = "0123456789abcdef0123"
    (src / "result.json").write_text(json.dumps({"phases": {"pull": {"error": f"http://1.2.3.4:5/{secret}/f/x"}}}))
    (src / "log.txt").write_text("not published")
    r = run(["publish", str(src), str(dst)], base, tmp_path, GUARD_ONCE="1", REDACT=f"{secret} short")
    assert r.returncode == 0
    assert secret not in (dst / "result.json").read_text() and "***" in (dst / "result.json").read_text()
    assert not (dst / "log.txt").exists() and not (dst / "exit_code").exists()
