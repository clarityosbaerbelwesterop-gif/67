"""Volume gateway: serves the network volume of one data centre to a training pod in another one.

Runs on a small pod that has the network volume mounted (Osirus runpod-job, gpu=...@X.XX/any). The training pod has
no network volume: it pulls its inputs through this gateway and pushes its results back (jobs/volsync.py). The port
is public, so every request carries the token as the first path part (/TOKEN/...); without it the answer is 404.

  GET  /T/health                          {"ok": true, "free": bytes}
  GET  /T/tree/REL                        [{"path": rel, "size": n}, ...] for every file under REL ([] if missing)
  GET  /T/sha/REL                         {"sha256": hex, "size": n}
  GET  /T/f/REL                           the file (Range: bytes=N- resumes from byte N)
  PUT  /T/f/REL?offset=O&size=S&sha256=H  one chunk of REL, written into REL.part at offset O (S = size of the whole
                                          file, H = sha256 of this chunk; the chunk counts only if H matches)
  POST /T/commit/REL?size=S&sha256=H      REL.part becomes REL once the chunks cover every byte; H = sha256 of the
                                          chunk hashes (hex, in offset order). Never overwrites: 409 if REL exists.

Paths stay inside --root (no "..", no symlinks out of it, no names ending in .part). Nothing is deleted except
this gateway's own REL.part when its commit is refused.

Usage: python3 jobs/volgw.py --root /workspace --port 8001 --token HEX
"""
import argparse
import hashlib
import hmac
import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

PIECE = 8 << 20


class Gateway:
    def __init__(self, root: Path, token: str):
        self.root, self.token = root.resolve(), token
        self.lock = threading.Lock()
        self.chunks: dict[str, dict[int, tuple[int, str]]] = {}   # rel -> offset -> (length, sha256)

    def path(self, rel: str, write: bool = False) -> Path:
        parts = [p for p in rel.split("/") if p]
        if not parts or any(p in (".", "..") or "\0" in p for p in parts) or (write and parts[-1].endswith(".part")):
            raise ValueError("bad path")
        p = self.root.joinpath(*parts)
        real = (p.parent.resolve() / p.name) if write else p.resolve()
        if real != self.root and self.root not in real.parents:
            raise ValueError("path leaves the root")
        return real

    def free(self) -> int:
        return shutil.disk_usage(self.root).free


class Handler(BaseHTTPRequestHandler):
    gw: Gateway
    protocol_version = "HTTP/1.1"
    timeout = 600

    def log_message(self, fmt, *a):
        pass

    def reply(self, code: int, obj=None):
        body = json.dumps(obj if obj is not None else {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def route(self):
        u = urlsplit(self.path)
        parts = u.path.lstrip("/").split("/", 2)
        if len(parts) < 2 or not hmac.compare_digest(parts[0].encode(), self.gw.token.encode()):
            return None
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        return parts[1], unquote(parts[2]) if len(parts) > 2 else "", q

    def drain(self):
        n = int(self.headers.get("Content-Length") or 0)
        while n > 0:
            b = self.rfile.read(min(n, PIECE))
            if not b:
                break
            n -= len(b)

    def do_GET(self):
        r = self.route()
        if r is None:
            return self.reply(404)
        op, rel, _ = r
        try:
            if op == "health":
                return self.reply(200, {"ok": True, "free": self.gw.free()})
            p = self.gw.path(rel)
            if op == "tree":
                if p.is_file():
                    return self.reply(200, [{"path": rel.strip("/"), "size": p.stat().st_size}])
                out = []
                if p.is_dir():
                    for f in sorted(p.rglob("*")):
                        if f.is_file() and not f.name.endswith(".part") and self.gw.root in f.resolve().parents:
                            out.append({"path": str(f.relative_to(self.gw.root)), "size": f.stat().st_size})
                return self.reply(200, out)
            if not p.is_file():
                return self.reply(404, {"error": "no such file"})
            if op == "sha":
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    while b := fh.read(PIECE):
                        h.update(b)
                return self.reply(200, {"sha256": h.hexdigest(), "size": p.stat().st_size})
            if op == "f":
                size, start = p.stat().st_size, 0
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes=") and rng[6:].split("-")[0].isdigit():
                    start = min(int(rng[6:].split("-")[0]), size)
                self.send_response(206 if start else 200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size - start))
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
                self.end_headers()
                with open(p, "rb") as fh:
                    fh.seek(start)
                    while b := fh.read(PIECE):
                        self.wfile.write(b)
                return
            return self.reply(404)
        except ValueError as e:
            return self.reply(400, {"error": str(e)})

    def do_PUT(self):
        r = self.route()
        if r is None or r[0] != "f":
            self.drain()
            return self.reply(404)
        _, rel, q = r
        try:
            p = self.gw.path(rel, write=True)
            off, size, want = int(q["offset"]), int(q["size"]), q["sha256"]
            n = int(self.headers["Content-Length"])
        except (KeyError, ValueError) as e:
            self.drain()
            return self.reply(400, {"error": f"bad request: {e}"})
        if p.exists():
            self.drain()
            return self.reply(409, {"error": "exists"})
        if off < 0 or n < 0 or off + n > size:
            self.drain()
            return self.reply(400, {"error": "chunk outside the file"})
        part = p.with_name(p.name + ".part")
        with self.gw.lock:
            full = not part.exists() and self.gw.free() < size + (1 << 30)
            if not full and not part.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(part, "wb") as fh:
                    fh.truncate(size)
                self.gw.chunks[rel] = {}
        if full:
            self.drain()
            return self.reply(507, {"error": "no space", "free": self.gw.free()})
        h, pos = hashlib.sha256(), 0
        fd = os.open(part, os.O_WRONLY)
        try:
            while pos < n:
                b = self.rfile.read(min(PIECE, n - pos))
                if not b:
                    return self.reply(400, {"error": "short body"})
                h.update(b)
                os.pwrite(fd, b, off + pos)
                pos += len(b)
        finally:
            os.close(fd)
        if h.hexdigest() != want:
            return self.reply(400, {"error": "chunk sha256 mismatch"})
        with self.gw.lock:
            self.gw.chunks.setdefault(rel, {})[off] = (n, want)
        return self.reply(200, {"ok": True})

    def do_POST(self):
        r = self.route()
        self.drain()
        if r is None or r[0] != "commit":
            return self.reply(404)
        _, rel, q = r
        try:
            p = self.gw.path(rel, write=True)
            size, want = int(q["size"]), q["sha256"]
        except (KeyError, ValueError) as e:
            return self.reply(400, {"error": f"bad request: {e}"})
        if p.exists():
            return self.reply(409, {"error": "exists"})
        part = p.with_name(p.name + ".part")
        with self.gw.lock:
            got = self.gw.chunks.pop(rel, {})
            if size == 0 and not part.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                part.touch()
            pos, h = 0, hashlib.sha256()
            for off in sorted(got):
                if off != pos:
                    break
                pos += got[off][0]
                h.update(got[off][1].encode())
            if pos != size or h.hexdigest() != want or not part.exists() or part.stat().st_size != size:
                part.unlink(missing_ok=True)
                return self.reply(400, {"error": "incomplete or wrong chunks", "covered": pos})
            fd = os.open(part, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
            try:
                os.link(part, p)
            except FileExistsError:
                part.unlink(missing_ok=True)
                return self.reply(409, {"error": "exists"})
            part.unlink()
        return self.reply(200, {"ok": True, "size": size})


def serve(root: Path, port: int, token: str, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    gw = Gateway(root, token)
    h = type("H", (Handler,), {"gw": gw})
    srv = ThreadingHTTPServer((host, port), h)
    srv.daemon_threads = True
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--token", default=os.environ.get("GW_TOKEN", ""))
    a = ap.parse_args(argv)
    if len(a.token) < 16:
        raise SystemExit("token of at least 16 characters needed")
    serve(a.root, a.port, a.token).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
