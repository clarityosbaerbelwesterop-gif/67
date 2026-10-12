"""Add this machine to the VGPU pool: register, then answer proof-of-compute challenges.

  python3 scripts/vgpu_worker.py --headcenter http://HOST:8067 --token TOKEN --label "MacBook M4" [--kind apple]
      [--every 1800] [--once]

stdlib + numpy (on macOS numpy uses Accelerate, i.e. the Apple matrix units). The
coordinator times every proof itself and spot-checks the result (headcenter/backend/
vgpu.py), so what this machine adds is verified, never self-reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "headcenter"))
from backend.vgpu import solve  # noqa: E402


def api(base: str, token: str, method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310 - operator-given headcenter URL
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path}: HTTP {e.code} {e.read()[:300]!r}") from None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--headcenter", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--kind", default="cpu", choices=["cpu", "gpu", "webgpu", "apple"])
    ap.add_argument("--claimed-tflops", type=float, default=0.2, help="first guess; later challenges use the verified speed")
    ap.add_argument("--every", type=float, default=1800.0, help="seconds between proofs (capacity counts for 6 h)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    node = api(a.headcenter, a.token, "POST", "/api/vgpu/register",
               {"label": a.label, "kind": a.kind, "claimed_tflops": a.claimed_tflops})["node_id"]
    print(json.dumps({"registered": node}), flush=True)
    while True:
        ch = api(a.headcenter, a.token, "GET", f"/api/vgpu/challenge?node_id={node}")
        rows = solve(ch["seed"], ch["n"], ch["k"])
        res = api(a.headcenter, a.token, "POST", "/api/vgpu/proof", {"challenge_id": ch["challenge_id"], "rows": rows})
        print(json.dumps({"n": ch["n"], "k": ch["k"], **res}), flush=True)
        if a.once:
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
