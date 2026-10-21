"""jobs/volgw.py + jobs/volsync.py: pull and push through the gateway, chunk checks, no overwrite, token, paths."""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jobs"))
import volgw  # noqa: E402
import volsync  # noqa: E402

TOKEN = "t" * 32


@pytest.fixture()
def gw(tmp_path):
    vol = tmp_path / "vol"
    vol.mkdir()
    srv = volgw.serve(vol, 0, TOKEN, host="127.0.0.1")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/{TOKEN}", vol
    srv.shutdown()


def write(p: Path, data: bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_push_pull_roundtrip(gw, tmp_path):
    url, vol = gw
    local = tmp_path / "local"
    big = os.urandom(3 * 1024 * 1024 + 17)
    write(local / "runs/r/10b/step-000010/optim.safetensors", big)
    write(local / "runs/r/10b/DONE.json", b'{"status": "ok"}')
    write(local / "runs/r/10b/empty.txt", b"")
    write(local / "runs/r/10b/x.part", b"skip me")
    r = volsync.push(url, local, ["runs/r/10b"], float("inf"), 4, 1024 * 1024)
    assert r["pushed"] == 3 and not r["missing"] and not r["conflicts"], r
    assert (vol / "runs/r/10b/step-000010/optim.safetensors").read_bytes() == big
    assert (vol / "runs/r/10b/empty.txt").read_bytes() == b""
    assert not (vol / "runs/r/10b/x.part").exists() and not list(vol.rglob("*.part"))
    # again: everything already there
    r = volsync.push(url, local, ["runs/r/10b"], float("inf"), 4, 1024 * 1024)
    assert r["pushed"] == 0 and len(r["skipped"]) == 3
    # pull into a fresh root, with sha256 check and a name filter
    fresh = tmp_path / "fresh"
    r = volsync.pull(url, fresh, ["runs/r/10b"], ["*.safetensors"], True, True)
    assert r["pulled"] == ["runs/r/10b/step-000010/optim.safetensors"] and not r["bad"], r
    assert (fresh / "runs/r/10b/step-000010/optim.safetensors").read_bytes() == big
    r = volsync.pull(url, fresh, ["runs/r/10b"], ["*.safetensors"], True, True)
    assert r["kept"] and not r["pulled"]
    r = volsync.pull(url, fresh, ["no/such/dir"], [], False, True)
    assert r["missing"] == ["no/such/dir"]


def test_never_overwrites(gw, tmp_path):
    url, vol = gw
    write(vol / "ckpt/a.bin", b"old data")
    local = tmp_path / "local"
    write(local / "ckpt/a.bin", b"new, longer data")
    r = volsync.push(url, local, ["ckpt"], float("inf"), 2, 4)
    assert r["conflicts"] == ["ckpt/a.bin"] and r["pushed"] == 0
    assert (vol / "ckpt/a.bin").read_bytes() == b"old data"
    h = hashlib.sha256(b"xx").hexdigest()
    code, _ = volsync.call(url, "PUT", "f", "ckpt/a.bin", f"offset=0&size=2&sha256={h}", data=b"xx")
    assert code == 409


def test_bad_chunks_do_not_commit(gw):
    url, vol = gw
    good, bad = hashlib.sha256(b"abcd").hexdigest(), hashlib.sha256(b"zzzz").hexdigest()
    code, _ = volsync.call(url, "PUT", "f", "d/f.bin", f"offset=0&size=8&sha256={bad}", data=b"abcd")
    assert code == 400
    code, _ = volsync.call(url, "PUT", "f", "d/f.bin", f"offset=0&size=8&sha256={good}", data=b"abcd")
    assert code == 200
    total = hashlib.sha256(good.encode()).hexdigest()
    code, _ = volsync.call(url, "POST", "commit", "d/f.bin", f"size=8&sha256={total}", data=b"")
    assert code == 400 and not (vol / "d/f.bin").exists() and not (vol / "d/f.bin.part").exists()


def test_token_and_paths(gw):
    url, vol = gw
    base = url.rsplit("/", 1)[0]
    with pytest.raises(urllib.error.HTTPError) as e:
        volsync.OPEN(f"{base}/{'x' * 32}/health", timeout=10)
    assert e.value.code == 404
    (vol.parent / "secret.txt").write_text("no")
    for rel in ("../secret.txt", "a/../../secret.txt"):
        code, _ = volsync.call(url, "GET", "f", rel, tries=1)
        assert code in (400, 404)
    os.symlink(vol.parent / "secret.txt", vol / "link.txt")
    code, _ = volsync.call(url, "GET", "f", "link.txt", tries=1)
    assert code == 400
    code, h = volsync.jcall(url, "GET", "health")
    assert code == 200 and h["free"] > 0


def test_deadline_leaves_files_missing(gw, tmp_path):
    url, vol = gw
    local = tmp_path / "local"
    write(local / "r/a.bin", b"a" * 10)
    r = volsync.push(url, local, ["r"], time.time() - 1, 2, 4)
    assert r["missing"] == ["r/a.bin"] and not (vol / "r/a.bin").exists()


def test_cli_summary(gw, tmp_path):
    url, vol = gw
    local = tmp_path / "local"
    write(local / "r/a.bin", b"abc")
    import subprocess
    env = {**os.environ, "GW": url}
    out = subprocess.run([sys.executable, str(ROOT / "jobs/volsync.py"), "push", "r", "--root", str(local)],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1])["pushed"] == 1
    out = subprocess.run([sys.executable, str(ROOT / "jobs/volsync.py"), "health"], capture_output=True, text=True,
                         env=env)
    assert out.returncode == 0 and json.loads(out.stdout)["ok"]


def test_pull_resumes_a_broken_download(gw, tmp_path):
    url, vol = gw
    data = os.urandom(3 * 1024 * 1024 + 5)
    write(vol / "store/train.bin", data)
    fresh = tmp_path / "fresh"
    part = fresh / "store/train.bin.part"
    write(part, data[:1_000_000])            # a download that broke after 1 MB
    r = volsync.pull(url, fresh, ["store"], [], True, True)
    assert r["pulled"] == ["store/train.bin"] and not r["bad"], r
    assert (fresh / "store/train.bin").read_bytes() == data and not part.exists()
    req = urllib.request.Request(f"{url}/f/store/train.bin", headers={"Range": "bytes=100-"})
    with volsync.OPEN(req, timeout=10) as resp:
        assert resp.status == 206 and resp.read() == data[100:]
