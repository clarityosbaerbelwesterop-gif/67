"""Swarm rendezvous: real machines join a DiLoCo group, get a rank and the peer list.

    POST /api/swarm/{group}/open   {"world", "config", "inner_steps", "compression"}   (operator)
    POST /api/swarm/join           {"group", "host", "port", "threads", "gflops"}     -> {"worker_id", ...}
    GET  /api/swarm/{group}/assignment?worker_id=...  -> 202 while the group fills, then
         {"rank", "world", "peers": ["host:port", ...], "config", "config_sha256",
          "config_json", "inner_steps", "compression"}
    POST /api/swarm/{group}/report {"worker_id", "state", "round", "rounds", "step", "loss", "exit_code"}
    GET  /api/swarm                groups and workers (operator)
    DELETE /api/swarm/{group}      forget a group (operator)

Ranks follow join order and are assigned once, when the W-th worker joins, so
every worker sees the same rank and peer list. A worker that stops polling
before its group is full is dropped after `ttl` seconds, so a dead machine never
blocks a group. Only machines that call this API are counted; nothing here is
simulated, and `gflops` is whatever the worker measured (scripts/swarm_worker.py
runs `forge bench`) or null. The config is snapshotted with its sha256 when the
group opens and handed to every rank, so all ranks train the identical config.

State survives restarts (`runs/headcenter/swarm.json`); every change is logged
to `runs/headcenter/swarm.jsonl`.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .control import RUN_RE

COMPRESSION_RE = re.compile(r"^(none|bf16|int8(:[1-9][0-9]{0,5})?)$")
LABEL_RE = re.compile(r"^[\w .:@()+/-]{1,64}$")
HOSTNAME_RE = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$")
STATES = ("starting", "running", "done", "failed")
MAX_WORLD = 256
MAX_GROUPS = 64
MAX_CONFIG_BYTES = 1 << 20


class SwarmError(ValueError):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def _int(v: Any, name: str, lo: int, hi: int) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise SwarmError(f'"{name}" must be an integer in {lo}..{hi}')
    return v


def check_host(host: Any) -> str:
    """An IP literal (canonical form) or a DNS name; never unspecified/multicast/broadcast."""
    if not isinstance(host, str) or not host or len(host) > 253:
        raise SwarmError('"host" must be an IP address or a DNS name')
    raw = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        ip = None
    if ip is not None:
        bad = ip.is_unspecified or ip.is_multicast or ip.is_reserved or (ip.version == 4 and int(ip) == 0xFFFFFFFF)
        if bad or (ip.version == 6 and ip.scope_id):
            raise SwarmError(f'"host" {host!r} cannot be a peer address')
        return ip.compressed
    if HOSTNAME_RE.match(host) and not all(p.isdigit() for p in host.split(".")):
        return host.lower()
    raise SwarmError('"host" must be an IP address or a DNS name')


def peer_addr(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _gflops(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 1e7:
        raise SwarmError('"gflops" must be null or a measured positive number')
    return float(v)


def _opt_num(v: Any, name: str) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise SwarmError(f'"{name}" must be a finite number or null')
    return v


class Swarm:
    def __init__(self, runs_dir: Path, config_roots: list[Path], repo_root: Path, ttl: float = 90.0):
        self.runs_dir = Path(runs_dir)
        self.config_roots = [Path(r).resolve() for r in config_roots]
        self.repo_root = Path(repo_root)
        self.ttl = ttl
        self.lock = threading.Lock()
        self.state_path = self.runs_dir / "headcenter" / "swarm.json"
        self.log_path = self.runs_dir / "headcenter" / "swarm.jsonl"
        self.groups: dict[str, dict] = {}
        self._load()

    # ---- persistence -----------------------------------------------------

    def _load(self) -> None:
        try:
            doc = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return
        now = time.time()
        for name, g in (doc.get("groups") or {}).items() if isinstance(doc, dict) else ():
            if RUN_RE.match(name) and isinstance(g, dict) and isinstance(g.get("workers"), list):
                for w in g["workers"]:
                    w["last_seen"] = now  # grace period after a headcenter restart
                self.groups[name] = g

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.state_path.parent, prefix=".swarm-", suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump({"groups": self.groups}, f)
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    def _log(self, **rec) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), **rec}) + "\n")
        except OSError:
            pass

    # ---- helpers ---------------------------------------------------------

    def _group(self, name: Any) -> dict:
        if not isinstance(name, str) or not RUN_RE.match(name):
            raise SwarmError("invalid group name")
        g = self.groups.get(name)
        if g is None:
            raise SwarmError(f"group {name} is not open", 404)
        return g

    def _prune(self, g: dict, now: float) -> None:
        if g["state"] != "open":
            return
        keep = [w for w in g["workers"] if now - w["last_seen"] <= self.ttl]
        for w in g["workers"]:
            if w not in keep:
                self._log(event="evicted", group=g["name"], worker=w["worker_id"][:6], host=w["host"], port=w["port"])
        g["workers"] = keep

    def load_config(self, config: Any) -> tuple[str, dict, str]:
        if not isinstance(config, str) or not config.endswith(".json") or "\x00" in config:
            raise SwarmError('"config" must be the path of a training config (.json)')
        p = Path(config)
        p = (p if p.is_absolute() else self.repo_root / p).resolve()
        if not any(root == p.parent or root in p.parents for root in self.config_roots):
            raise SwarmError('"config" must be inside ' + " or ".join(str(r) for r in self.config_roots))
        try:
            raw = p.read_bytes()
        except OSError:
            raise SwarmError(f"config {config} not found", 404) from None
        if len(raw) > MAX_CONFIG_BYTES:
            raise SwarmError("config too large")
        try:
            cfg = json.loads(raw)
        except ValueError as e:
            raise SwarmError(f"config is not JSON: {e}") from None
        if not isinstance(cfg, dict) or not isinstance(cfg.get("model"), dict) or not isinstance(cfg.get("run"), str):
            raise SwarmError('config needs "run" and "model"')
        _int(cfg.get("max_steps"), "max_steps", 1, 10**9)
        return config, cfg, hashlib.sha256(raw).hexdigest()

    # ---- API -------------------------------------------------------------

    def open(self, name: Any, body: Any) -> dict:
        if not isinstance(body, dict):
            raise SwarmError("body must be a JSON object")
        if not isinstance(name, str) or not RUN_RE.match(name):
            raise SwarmError("invalid group name")
        world = _int(body.get("world"), "world", 2, MAX_WORLD)
        inner = _int(body.get("inner_steps", 50), "inner_steps", 1, 10**6)
        comp = body.get("compression", "none")
        if not isinstance(comp, str) or not COMPRESSION_RE.match(comp):
            raise SwarmError('"compression" must be none, bf16 or int8[:block]')
        config, cfg, sha = self.load_config(body.get("config"))
        with self.lock:
            now = time.time()
            old = self.groups.get(name)
            if old is not None and old["state"] not in ("open", "done", "failed"):
                raise SwarmError(f"group {name} is {old['state']}; delete it first", 409)
            if old is None and len(self.groups) >= MAX_GROUPS:
                raise SwarmError("too many groups", 409)
            workers = old["workers"] if old is not None and old["state"] == "open" else []
            if len(workers) > world:
                raise SwarmError(f"{len(workers)} workers already joined, more than world {world}", 409)
            g = {"name": name, "state": "open", "world": world, "config": config, "config_sha256": sha,
                 "config_json": cfg, "inner_steps": inner, "compression": comp, "opened_ts": now,
                 "assigned_ts": None, "peers": None, "workers": workers}
            self.groups[name] = g
            self._log(event="open", group=name, world=world, config=config, config_sha256=sha, inner_steps=inner, compression=comp)
            self._maybe_assign(g, now)
            self._save()
            return self.public(g)

    def join(self, body: Any) -> dict:
        if not isinstance(body, dict):
            raise SwarmError("body must be a JSON object")
        host = check_host(body.get("host"))
        port = _int(body.get("port"), "port", 1024, 65535)
        threads = _int(body.get("threads"), "threads", 1, 1024)
        gflops = _gflops(body.get("gflops"))
        label = body.get("label")
        if label is not None and (not isinstance(label, str) or not LABEL_RE.match(label)):
            raise SwarmError('"label" must be up to 64 plain characters')
        source = body.get("gflops_source")
        if source is not None and (not isinstance(source, str) or not LABEL_RE.match(source)):
            raise SwarmError('"gflops_source" must be up to 64 plain characters')
        with self.lock:
            g = self._group(body.get("group"))
            now = time.time()
            self._prune(g, now)
            if g["state"] != "open":
                raise SwarmError(f"group {g['name']} is full ({g['state']})", 409)
            addr = peer_addr(host, port)
            if any(peer_addr(w["host"], w["port"]) == addr for w in g["workers"]):
                raise SwarmError(f"{addr} already joined {g['name']}", 409)
            w = {"worker_id": secrets.token_urlsafe(16), "host": host, "port": port, "threads": threads,
                 "gflops": gflops, "gflops_source": source if gflops is not None else None, "label": label,
                 "joined_ts": now, "last_seen": now, "rank": None, "run_state": None}
            g["workers"].append(w)
            self._log(event="join", group=g["name"], worker=w["worker_id"][:6], host=host, port=port, threads=threads, gflops=gflops)
            self._maybe_assign(g, now)
            self._save()
            return {"worker_id": w["worker_id"], "group": g["name"], "joined": len(g["workers"]), "world": g["world"],
                    "state": g["state"]}

    def _maybe_assign(self, g: dict, now: float) -> None:
        if g["state"] != "open" or len(g["workers"]) != g["world"]:
            return
        for rank, w in enumerate(g["workers"]):
            w["rank"] = rank
        g["peers"] = [peer_addr(w["host"], w["port"]) for w in g["workers"]]
        g["state"] = "assigned"
        g["assigned_ts"] = now
        self._log(event="assigned", group=g["name"], peers=g["peers"])

    def assignment(self, name: Any, worker_id: Any) -> tuple[int, dict]:
        with self.lock:
            g = self._group(name)
            now = time.time()
            self._prune(g, now)
            w = next((w for w in g["workers"] if isinstance(worker_id, str) and secrets.compare_digest(w["worker_id"], worker_id)), None)
            if w is None:
                raise SwarmError("unknown worker (never joined or dropped after missing polls): join again", 404)
            w["last_seen"] = now
            if g["state"] == "open":
                return 202, {"status": "waiting", "group": g["name"], "joined": len(g["workers"]), "world": g["world"]}
            return 200, {"status": "assigned", "group": g["name"], "rank": w["rank"], "world": g["world"],
                         "peers": list(g["peers"]), "config": g["config"], "config_sha256": g["config_sha256"],
                         "config_json": g["config_json"], "inner_steps": g["inner_steps"], "compression": g["compression"]}

    def report(self, name: Any, body: Any) -> dict:
        if not isinstance(body, dict):
            raise SwarmError("body must be a JSON object")
        state = body.get("state")
        if state not in STATES:
            raise SwarmError(f'"state" must be one of {STATES}')
        upd = {"run_state": state}
        for k in ("round", "rounds", "step", "exit_code"):
            v = body.get(k)
            if v is not None:
                upd[k] = _int(v, k, -(2**31), 2**62)
        upd["loss"] = _opt_num(body.get("loss"), "loss")
        with self.lock:
            g = self._group(name)
            worker_id = body.get("worker_id")
            w = next((w for w in g["workers"] if isinstance(worker_id, str) and secrets.compare_digest(w["worker_id"], worker_id)), None)
            if w is None:
                raise SwarmError("unknown worker", 404)
            if g["state"] == "open":
                raise SwarmError("group not assigned yet", 409)
            now = time.time()
            w.update(upd, last_seen=now)
            states = [x.get("run_state") for x in g["workers"]]
            if any(s == "failed" for s in states):
                g["state"] = "failed"
            elif all(s == "done" for s in states):
                g["state"] = "done"
            elif g["state"] == "assigned" and any(s == "running" for s in states):
                g["state"] = "running"
            if state in ("done", "failed"):
                self._log(event=state, group=g["name"], rank=w["rank"], exit_code=upd.get("exit_code"), step=upd.get("step"))
            self._save()
            return {"ok": True, "group_state": g["state"]}

    def delete(self, name: Any) -> None:
        with self.lock:
            g = self._group(name)
            del self.groups[g["name"]]
            self._log(event="delete", group=g["name"])
            self._save()

    @staticmethod
    def public(g: dict) -> dict:
        workers = [{k: v for k, v in w.items() if k != "worker_id"} | {"id": w["worker_id"][:6]} for w in g["workers"]]
        reported = [w["gflops"] for w in g["workers"] if w.get("gflops") is not None]
        return {k: v for k, v in g.items() if k not in ("workers", "config_json")} | {
            "workers": workers, "joined": len(workers), "run": g["config_json"].get("run"),
            "max_steps": g["config_json"].get("max_steps"),
            "gflops_reported": sum(reported) if reported else None, "gflops_reporting": len(reported)}

    def overview(self) -> dict:
        with self.lock:
            now = time.time()
            for g in self.groups.values():
                self._prune(g, now)
            return {"groups": {n: self.public(g) for n, g in sorted(self.groups.items())}, "ttl_s": self.ttl, "now": now}
