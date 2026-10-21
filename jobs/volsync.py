"""Client of jobs/volgw.py: the training pod (no network volume) pulls its inputs from the volume and pushes its
results back. GW is http://HOST:PORT/TOKEN (env GW); paths are relative to the volume root and the same on --root.

  pull REL... [--name GLOB ...] [--verify] [--require]   files under REL (or REL itself) into ROOT/REL; a local
       file of the same size is kept. --verify compares sha256 with the gateway's, --require fails if REL is empty.
  push REL... [--until EPOCH] [--streams N] [--chunk-mb M]   local files under ROOT/REL to the volume, smallest
       first (logs, exports, then weights, then optimizer state), each in chunks with sha256. A file that is already
       there with the same size is skipped; one with another size is a conflict and is never overwritten. No new
       chunk starts after --until; what is left is listed under "missing".
  health

The last stdout line is a JSON summary. Exit 0 = everything pulled or pushed, 2 = something missing or in conflict.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

PIECE = 8 << 20
OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({})).open   # the gateway is a direct address


class Deadline(Exception):
    pass


def call(gw: str, method: str, op: str, rel: str = "", query: str = "", data=None, timeout: int = 600, tries: int = 4):
    url = f"{gw.rstrip('/')}/{op}/{quote(rel)}" + (f"?{query}" if query else "")
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, method=method)
            with OPEN(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or e.code == 507 or i == tries - 1:
                return e.code, e.read()
        except OSError:
            if i == tries - 1:
                raise
        time.sleep(2 ** i)


def jcall(*a, **k):
    code, body = call(*a, **k)
    try:
        return code, json.loads(body or b"null")
    except ValueError:
        return code, None


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(PIECE):
            h.update(b)
    return h.hexdigest()


def remote_tree(gw: str, rel: str) -> list:
    code, t = jcall(gw, "GET", "tree", rel)
    if code != 200 or not isinstance(t, list):
        raise SystemExit(f"tree {rel}: HTTP {code}")
    return t


def fetch(gw: str, rel: str, size: int, tmp: Path, tries: int = 6) -> bool:
    """Download into tmp, resuming after a broken connection (Range); True once tmp holds `size` bytes."""
    url = f"{gw.rstrip('/')}/f/{quote(rel)}"
    if tmp.exists() and tmp.stat().st_size > size:
        tmp.unlink()
    for i in range(tries):
        have = tmp.stat().st_size if tmp.exists() else 0
        if have == size:
            return True
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with OPEN(req, timeout=120) as r, open(tmp, "ab" if have and r.status == 206 else "wb") as fh:
                while b := r.read(PIECE):
                    fh.write(b)
        except OSError:
            time.sleep(min(2 ** i, 30))
    return tmp.exists() and tmp.stat().st_size == size


def pull(gw: str, root: Path, rels, names, verify: bool, require: bool) -> dict:
    done, kept, missing, bad = [], [], [], []
    t0, nbytes = time.time(), 0
    for rel in rels:
        files = [f for f in remote_tree(gw, rel) if not names or any(fnmatch.fnmatch(Path(f["path"]).name, g) for g in names)]
        if not files and require:
            missing.append(rel)
        for f in files:
            dst = root / f["path"]
            if dst.is_file() and dst.stat().st_size == f["size"]:
                kept.append(f["path"])
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            if not fetch(gw, f["path"], f["size"], tmp):
                tmp.unlink(missing_ok=True)
                bad.append(f["path"])
                continue
            if verify:
                code, s = jcall(gw, "GET", "sha", f["path"], timeout=3600)
                if code != 200 or s["sha256"] != sha_file(tmp):
                    tmp.unlink()
                    bad.append(f["path"])
                    continue
            os.replace(tmp, dst)
            done.append(f["path"])
            nbytes += f["size"]
    s = time.time() - t0
    return {"op": "pull", "pulled": done, "kept": kept, "missing": missing, "bad": bad, "bytes": nbytes,
            "seconds": round(s, 1), "mb_s": round(nbytes / 1e6 / max(s, 1e-3), 1)}


def push_file(gw: str, root: Path, rel: str, until: float, streams: int, chunk: int) -> None:
    p = root / rel
    size = p.stat().st_size
    offs = list(range(0, size, chunk))
    hashes = {}

    def one(off: int) -> None:
        if time.time() > until:
            raise Deadline()
        with open(p, "rb") as fh:
            fh.seek(off)
            b = fh.read(min(chunk, size - off))
        h = hashlib.sha256(b).hexdigest()
        for _ in range(3):
            code, body = call(gw, "PUT", "f", rel, f"offset={off}&size={size}&sha256={h}", data=b)
            if code == 200:
                hashes[off] = h
                return
            if code in (409, 507):
                raise RuntimeError(f"HTTP {code} {body[:200]!r}")
        raise RuntimeError(f"chunk {off}: HTTP {code} {body[:200]!r}")

    with ThreadPoolExecutor(max(1, min(streams, len(offs)))) as ex:
        list(ex.map(one, offs))
    total = hashlib.sha256("".join(hashes[o] for o in offs).encode()).hexdigest()
    code, body = call(gw, "POST", "commit", rel, f"size={size}&sha256={total}", data=b"")
    if code != 200:
        raise RuntimeError(f"commit: HTTP {code} {body[:200]!r}")


def push(gw: str, root: Path, rels, until: float, streams: int, chunk: int) -> dict:
    files = []
    for rel in rels:
        p = root / rel
        cand = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else []
        files += [(f.stat().st_size, str(f.relative_to(root))) for f in cand if not f.name.endswith(".part")]
    files.sort()
    remote = {}
    for rel in rels:
        remote.update({f["path"]: f["size"] for f in remote_tree(gw, rel)})
    done, skipped, conflicts, missing, errors = [], [], [], [], []
    t0, nbytes = time.time(), 0
    for size, rel in files:
        if rel in remote:
            (skipped if remote[rel] == size else conflicts).append(rel)
            continue
        if time.time() > until:
            missing.append(rel)
            continue
        try:
            push_file(gw, root, rel, until, streams, chunk)
            done.append(rel)
            nbytes += size
        except Deadline:
            missing.append(rel)
        except (RuntimeError, OSError) as e:
            errors.append({"path": rel, "error": str(e)[:300]})
            missing.append(rel)
    s = time.time() - t0
    return {"op": "push", "pushed": len(done), "bytes": nbytes, "seconds": round(s, 1),
            "mb_s": round(nbytes / 1e6 / max(s, 1e-3), 1), "skipped": skipped, "conflicts": conflicts,
            "missing": missing, "errors": errors[:20]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("op", choices=["pull", "push", "health"])
    ap.add_argument("rels", nargs="*")
    ap.add_argument("--gw", default=os.environ.get("GW", ""))
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("VOL_ROOT", "/workspace")))
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--require", action="store_true")
    ap.add_argument("--until", type=float, default=float("inf"))
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--chunk-mb", type=float, default=64)
    a = ap.parse_args(argv)
    if not a.gw:
        raise SystemExit("no gateway (--gw or env GW)")
    if a.op == "health":
        code, h = jcall(a.gw, "GET", "health", timeout=30)
        print(json.dumps({"op": "health", "http": code, **(h or {})}))
        return 0 if code == 200 else 2
    if a.op == "pull":
        r = pull(a.gw, a.root, a.rels, a.name, a.verify, a.require)
        print(json.dumps(r))
        return 2 if r["missing"] or r["bad"] else 0
    r = push(a.gw, a.root, a.rels, a.until, a.streams, max(1, int(a.chunk_mb * (1 << 20))))
    print(json.dumps(r))
    return 2 if r["missing"] or r["conflicts"] else 0


if __name__ == "__main__":
    sys.exit(main())
