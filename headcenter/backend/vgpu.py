"""VGPU: the swarm as one virtual accelerator, with verified capacity.

Every machine that should add compute (Mac, Linux box, server, later a browser
via WebGPU) registers here and proves its throughput with a proof of compute:

  challenge  the coordinator draws a 32-bit seed and sizes n x n matrices A0, B with
             entries in {-1, 0, 1} (from a 32-bit hash every node type computes the
             same way: numpy, JavaScript, WGSL) and k rounds so the work takes ~target seconds at the
             node's claimed speed.
  work       C_r = A_r @ B (float32; |C| <= n <= 2^24 keeps every entry exact),
             A_{r+1} = (C_r mod 3) - 1. Round r+1 needs all of round r, so nothing
             can be precomputed or skipped.
  proof      the sha256 of every row of C_k (rows are independent chains).
  verify     the coordinator recomputes `samples` random rows (n^2 per row and round,
             cheap) and compares; it times the work with ITS OWN clock, from issuing
             the challenge to receiving the proof, so a node can neither fake results
             (a node that skips half the rows is caught with p = 1 - 0.5^8) nor speed.

verified TFLOP/s = 2 n^3 k / server-measured seconds. A node's capacity is the
median of its last proofs; it counts toward the pool while its newest proof is
younger than `max_age`. The pool's capacity is reported in B300 equivalents
(2,500 dense BF16 TFLOP/s each, the forge reference). This measures matrix
throughput, an upper bound for training; training-effective throughput is what
DiLoCo telemetry reports once the pool trains. Nothing is simulated: only
machines that answer challenges count.

State: runs/vgpu/pool.json (atomic); every event in runs/vgpu/pool.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

B300_TFLOPS = 2500.0
LABEL_RE = re.compile(r"^[\w .:@()+/-]{1,64}$")
KINDS = ("cpu", "gpu", "webgpu", "apple")


class VGPUError(ValueError):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


# --------------------------------------------------------------------------- proof of compute (pure)

def mix32(x: np.ndarray) -> np.ndarray:
    """lowbias32 (C. Wellons): a 32-bit integer hash that numpy, JavaScript and WGSL
    compute identically, so every kind of node can generate the challenge itself."""
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def entries(seed: int, which: int, rows: np.ndarray, n: int) -> np.ndarray:
    """Rows `rows` of matrix `which` (0 = A0, 1 = B): (mix32(mix32(i*n + j) ^ s) mod 3) - 1."""
    s = np.uint32((seed ^ (which * 0x9E3779B9)) & 0xFFFFFFFF)
    idx = (rows.astype(np.uint32)[:, None] * np.uint32(n) + np.arange(n, dtype=np.uint32)[None, :])
    with np.errstate(over="ignore"):
        return (mix32(mix32(idx) ^ s) % np.uint32(3)).astype(np.float32) - 1.0


def matrices(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    allrows = np.arange(n)
    return entries(seed, 0, allrows, n), entries(seed, 1, allrows, n)


def step(c: np.ndarray) -> np.ndarray:
    return (np.mod(c.astype(np.int64), 3) - 1).astype(np.float32)


def row_digest(row: np.ndarray) -> str:
    return hashlib.sha256(row.astype(np.int32).tobytes()).hexdigest()


def solve(seed: int, n: int, k: int) -> list[str]:
    """The worker's side: the k chained products, then one digest per row of C_k."""
    a, b = matrices(seed, n)
    c = a
    for r in range(k):
        c = a @ b
        if r + 1 < k:
            a = step(c)
    return [row_digest(c[i]) for i in range(n)]


def check_rows(seed: int, n: int, k: int, rows: list[int]) -> dict[int, str]:
    """The coordinator's side: the true digests of a few rows, recomputed together
    (one small (rows x n) @ (n x n) product per round)."""
    b = entries(seed, 1, np.arange(n), n)
    x = entries(seed, 0, np.asarray(rows), n)
    for r in range(k):
        c = x @ b
        if r + 1 < k:
            x = step(c)
    return {i: row_digest(c[j]) for j, i in enumerate(rows)}


# --------------------------------------------------------------------------- the pool

class Pool:
    def __init__(self, runs_dir: Path, max_age: float = 6 * 3600, samples: int = 8, target_seconds: float = 20.0,
                 challenge_ttl: float = 900.0, n_max: int = 8192):
        self.dir = Path(runs_dir) / "vgpu"
        self.path, self.logp = self.dir / "pool.json", self.dir / "pool.jsonl"
        self.max_age, self.samples, self.target = max_age, samples, target_seconds
        self.challenge_ttl, self.n_max = challenge_ttl, n_max
        self.lock = threading.Lock()
        self.nodes: dict[str, dict] = {}
        self.challenges: dict[str, dict] = {}
        try:
            d = json.loads(self.path.read_text())
            self.nodes, self.challenges = d.get("nodes", {}), d.get("challenges", {})
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".pool.")
        with os.fdopen(fd, "w") as f:
            json.dump({"nodes": self.nodes, "challenges": self.challenges}, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _log(self, **rec) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.logp, "a") as f:
            f.write(json.dumps({"ts": time.time(), **rec}) + "\n")

    def register(self, body: Any) -> dict:
        if not isinstance(body, dict):
            raise VGPUError("body must be an object")
        label, kind = str(body.get("label", "")), body.get("kind", "cpu")
        if not LABEL_RE.match(label) or kind not in KINDS:
            raise VGPUError(f"label must match {LABEL_RE.pattern}, kind one of {KINDS}")
        claim = body.get("claimed_tflops")
        claim = float(claim) if isinstance(claim, (int, float)) and 0 < claim < 1e5 else 0.1
        with self.lock:
            nid = secrets.token_hex(8)
            max_n = body.get("max_n")
            max_n = int(max_n) if isinstance(max_n, int) and 256 <= max_n <= self.n_max else self.n_max
            self.nodes[nid] = {"id": nid, "label": label, "kind": kind, "claimed_tflops": claim, "registered": time.time(),
                               "max_n": max_n, "proofs": [], "failed": 0}
            self._save()
        self._log(event="register", node=nid, label=label, kind=kind, claimed_tflops=claim)
        return {"node_id": nid}

    def challenge(self, node_id: str) -> dict:
        with self.lock:
            node = self.nodes.get(node_id)
            if not node:
                raise VGPUError("unknown node", 404)
            speed = self._capacity(node) or node["claimed_tflops"]
            n = int(min(node.get("max_n", self.n_max), self.n_max,
                        max(256, 2 ** round(np.log2(max(256.0, (speed * 1e12 * self.target / 16) ** (1 / 3)))))))
            k = int(max(2, min(64, round(speed * 1e12 * self.target / (2 * n ** 3)))))  # bounds the check
            cid = secrets.token_hex(8)
            ch = {"id": cid, "node": node_id, "seed": secrets.randbits(32), "n": n, "k": k, "issued": time.time()}
            now = time.time()
            self.challenges = {c: v for c, v in self.challenges.items() if now - v["issued"] < self.challenge_ttl}
            self.challenges[cid] = ch
            self._save()
        return {"challenge_id": cid, "seed": ch["seed"], "n": n, "k": k}

    def proof(self, body: Any) -> dict:
        received = time.time()
        if not isinstance(body, dict):
            raise VGPUError("body must be an object")
        with self.lock:
            ch = self.challenges.pop(str(body.get("challenge_id")), None)
            if not ch:
                raise VGPUError("unknown or expired challenge", 404)
            self._save()
        digests = body.get("rows")
        if not isinstance(digests, list) or len(digests) != ch["n"]:
            return self._fail(ch, "wrong number of row digests")
        rows = sorted(secrets.SystemRandom().sample(range(ch["n"]), min(self.samples, ch["n"])))
        truth = check_rows(ch["seed"], ch["n"], ch["k"], rows)
        if any(digests[i] != d for i, d in truth.items()):
            return self._fail(ch, "row digests do not match the recomputed rows")
        secs = max(received - ch["issued"], 1e-6)
        tflops = 2 * ch["n"] ** 3 * ch["k"] / secs / 1e12
        with self.lock:
            node = self.nodes[ch["node"]]
            node["proofs"] = (node["proofs"] + [{"ts": received, "tflops": tflops, "n": ch["n"], "k": ch["k"],
                                                 "seconds": secs}])[-5:]
            self._save()
        self._log(event="proof", node=ch["node"], tflops=tflops, n=ch["n"], k=ch["k"], seconds=secs, checked_rows=len(rows))
        return {"verified": True, "tflops": tflops, "seconds": secs, "checked_rows": len(rows)}

    def _fail(self, ch: dict, why: str) -> dict:
        with self.lock:
            node = self.nodes.get(ch["node"])
            if node:
                node["failed"] = node.get("failed", 0) + 1
                self._save()
        self._log(event="proof-failed", node=ch["node"], reason=why)
        raise VGPUError(f"proof rejected: {why}", 422)

    def _capacity(self, node: dict, now: float | None = None) -> float:
        p = [x for x in node.get("proofs", []) if (now or time.time()) - x["ts"] < self.max_age]
        return statistics.median(x["tflops"] for x in p) if p else 0.0

    def overview(self) -> dict:
        now = time.time()
        with self.lock:
            nodes = [{"id": n["id"], "label": n["label"], "kind": n["kind"], "verified_tflops": self._capacity(n, now),
                      "last_proof": max((x["ts"] for x in n.get("proofs", [])), default=None), "failed": n.get("failed", 0)}
                     for n in self.nodes.values()]
        total = sum(n["verified_tflops"] for n in nodes)
        return {"verified_tflops": total, "b300_equivalents": total / B300_TFLOPS, "b300_tflops": B300_TFLOPS,
                "live_nodes": sum(1 for n in nodes if n["verified_tflops"] > 0), "nodes": nodes, "max_age_hours": self.max_age / 3600,
                "basis": "proof-of-compute matrix throughput (fp32, exact), timed by the coordinator"}
