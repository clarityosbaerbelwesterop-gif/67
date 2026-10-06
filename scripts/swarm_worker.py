"""Join a forge DiLoCo group through the headcenter rendezvous and train.

Run on every real machine that should contribute (only real machines add
compute). The headcenter assigns rank, world and peers once the group is full;
this script then runs `forge diloco` with exactly that assignment and reports
its progress back.

  python3 scripts/swarm_worker.py --headcenter http://HOST:8067 --token TOKEN \\
      --group quasnir-g2 --host THIS_MACHINE_IP --port 47610 [--threads 0] [--no-bench]

stdlib only. The forge output is also written to runs/<group>-w<rank>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def api(base: str, token: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(base.rstrip("/") + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - operator-given headcenter URL
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def bench(forge: str, threads: int) -> float | None:
    try:
        out = subprocess.run([forge, "bench", "--threads", str(threads or os.cpu_count() or 1)], capture_output=True, text=True, timeout=600).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    best = [max(r["gflops"] for r in d["results"]) for d in map(json.loads, (l for l in out.splitlines() if l.startswith("{"))) if d.get("type") == "gemm"]
    return max(best) if best else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headcenter", required=True)
    ap.add_argument("--token", default=os.environ.get("HEADCENTER_TOKEN"))
    ap.add_argument("--group", required=True)
    ap.add_argument("--host", required=True, help="address the other workers can reach this machine at")
    ap.add_argument("--port", type=int, default=47610)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--forge", default=str(ROOT / "target" / "release" / "forge"))
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--log-dir", type=Path, default=ROOT / "runs", help="where <group>-w<rank>.jsonl is written")
    a = ap.parse_args()
    if not a.token:
        ap.error("--token or HEADCENTER_TOKEN is required")
    gflops = None if a.no_bench else bench(a.forge, a.threads)
    st, j = api(a.headcenter, a.token, "POST", "/api/swarm/join",
                {"group": a.group, "host": a.host, "port": a.port, "threads": a.threads or os.cpu_count() or 1, "gflops": gflops})
    if st != 200:
        print(f"join failed ({st}): {j}", file=sys.stderr)
        return 2
    wid = j["worker_id"]
    print(f"joined {a.group} as {wid[:6]} ({j['joined']}/{j['world']}), measured {gflops} GFLOP/s", flush=True)
    while True:
        st, asg = api(a.headcenter, a.token, "GET", f"/api/swarm/{a.group}/assignment?worker_id={wid}")
        if st == 200:
            break
        if st != 202:
            print(f"assignment failed ({st}): {asg}", file=sys.stderr)
            return 2
        time.sleep(a.poll)
    rank, world = asg["rank"], asg["world"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(asg["config_json"] if isinstance(asg["config_json"], str) else json.dumps(asg["config_json"]))
        cfg_path = f.name
    cmd = [a.forge, "diloco", "--config", cfg_path, "--rank", str(rank), "--world", str(world), "--peers", ",".join(asg["peers"]),
           "--listen", f"0.0.0.0:{a.port}", "--inner-steps", str(asg["inner_steps"]), "--compression", asg["compression"]]
    log = a.log_dir / f"{a.group}-w{rank}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"rank {rank}/{world}: {' '.join(cmd)}", flush=True)
    api(a.headcenter, a.token, "POST", f"/api/swarm/{a.group}/report", {"worker_id": wid, "state": "running"})
    with log.open("a") as out:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, cwd=ROOT)
        for line in proc.stdout:
            out.write(line)
            out.flush()
            sys.stdout.write(line)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "diloco":
                api(a.headcenter, a.token, "POST", f"/api/swarm/{a.group}/report",
                    {"worker_id": wid, "state": "running", "round": rec["round"], "rounds": rec["rounds"], "step": rec["step"], "loss": rec["loss"]})
        rc = proc.wait()
    api(a.headcenter, a.token, "POST", f"/api/swarm/{a.group}/report",
        {"worker_id": wid, "state": "done" if rc == 0 else "failed", "exit_code": rc})
    os.unlink(cfg_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
