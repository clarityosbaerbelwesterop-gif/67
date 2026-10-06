#!/usr/bin/env python3
"""Multi-day continuous training orchestrator ("durchgängig trainieren").

A resumable state machine. The state lives in runs/longrun/state.json and every
write is atomic (tmp file + fsync + rename). Every step is idempotent, so after a
crash, a reboot or a STOP the next start continues where the last one stopped.
An interrupted `forge train` resumes from <run>/latest.json.

One cycle (generation g = 2, 3, ...; generation 1 is base-s / rouge-1 /
quasnir-1 / darus-1 from scripts/train_all.sh):

  base       base-g<g>: continue pretraining from the current base FINAL on
             base-v2 (lr re-warm + cosine)                          ~55 % of the cycle
  base_gate  base-g<g> becomes the current base only if base-v2 val loss does
             not regress beyond the tolerance
  quasnir    quasnir-g<g> (coding): init_from the base on code-v2   ~14 %
  quasnir_rsi controlled RSI, family code (training.rsi.loop)       ~5 %
  quasnir_gate
  rouge_mix  base-v2 + general-v2 mix store (re-sliced documents, built once)
  rouge      rouge-g<g> (Rouge 1, restricted): init_from the base   ~15 %
  rouge_rsi  controlled RSI, family text                            ~3 %
  rouge_gate
  darus_merge evolutionary TIES/linear merge search (scripts/merge_search.py)
  darus_rsi  controlled RSI, family mixed            merge + RSI    ~5 %
  darus_gate
  report     forge eval of all four models on the v2 val stores,
             HumanEval/MBPP for Darus and Quasnir -> report-g<g>.json   ~3 %
  hygiene    keep FINAL + the last N step checkpoints per finished run

Step counts come from the wall budget (--days), the cycle length and a measured
throughput: `forge bench --config <base config> --steps 3` once at the start;
later cycles use the median tokens/s of the previous base run's telemetry.

Live control: every `forge train` reads commands from the FIFO runs/<run>.ctl
(opened read-write, like scripts/train_all.sh) and its stdout is appended to
runs/<run>.jsonl, so the headcenter shows and controls it. Create
runs/longrun/STOP to stop gracefully: {"cmd":"checkpoint"} and {"cmd":"stop"}
go through the FIFO and longrun exits with code 75. Remove STOP and start again
to continue. SIGTERM/SIGINT do the same without the file.

Usage:
  python3 scripts/longrun.py --dry-run [--days 3] [--json]   plan only, runs nothing
  python3 scripts/longrun.py [--days 3 | --forever] [--allow-v1]
  python3 scripts/longrun.py --status
  python3 scripts/longrun.py --prepare      plan the next cycle, write its base config, exit
Exit codes: 0 finished, 1 step failed (retry later), 3 fatal (fix the cause),
4 another instance is running, 75 stopped (STOP file or signal).
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EX_OK, EX_FAIL, EX_FATAL, EX_LOCKED, EX_STOPPED = 0, 1, 3, 4, 75
EOS = 8190
EOS_BYTES = EOS.to_bytes(2, "little")
STEP_RE = re.compile(r"^step-(\d{6,})$")
TRAIN_FAMILIES = ("base", "quasnir", "rouge")
FAMILIES = ("base", "quasnir", "rouge", "darus")
CYCLE_STEPS = ("base", "base_gate", "quasnir", "quasnir_rsi", "quasnir_gate", "rouge_mix", "rouge", "rouge_rsi",
               "rouge_gate", "darus_merge", "darus_rsi", "darus_gate", "report", "hygiene")
CTL_STOP = b'{"cmd":"checkpoint"}\n{"cmd":"stop"}\n'

# Built-in defaults; training/configs/longrun.json holds the same values (a test
# keeps the two identical) and is deep-merged over them.
DEFAULTS: dict = {
    "days": 3,
    "forever": False,
    "cycle_hours": 24,
    "min_cycle_fraction": 0.25,
    "max_cycles": None,
    "first_generation": 2,
    "base_config": "training/configs/base-s.json",
    "forge": "target/release/forge",
    "nice": 10,
    "max_retries": 2,
    "retry_backoff_seconds": 60,
    "min_free_gb": 5,
    "min_steps": 20,
    "throughput": {"bench_steps": 3, "assumed_tokens_per_s": 1700, "telemetry_min_samples": 10},
    "shares": {"base": 0.55, "quasnir": 0.14, "quasnir_rsi": 0.05, "rouge": 0.15, "rouge_rsi": 0.03,
               "darus": 0.05, "report": 0.03},
    "names": {"base": "base-g{g}", "quasnir": "quasnir-g{g}", "rouge": "rouge-g{g}", "darus": "darus-g{g}"},
    "initial": {"base": "runs/base-s/FINAL", "quasnir": "runs/quasnir-1/FINAL", "rouge": "runs/rouge-1/FINAL",
                "darus": "runs/darus-1/FINAL"},
    "train": {
        "defaults": {"ckpt_every": 250, "eval_every": 500, "eval_batches": 16, "log_every": 10, "threads": 0},
        "base": {"lr": 0.0004, "min_lr": 0.00004, "warmup_steps": 200, "seed": 1000},
        "quasnir": {"lr": 0.0002, "min_lr": 0.00002, "warmup_steps": 50, "seed": 2000},
        "rouge": {"lr": 0.0002, "min_lr": 0.00002, "warmup_steps": 50, "seed": 3000},
    },
    "corpus": {
        "v2": {"base": "data/out/base-v2/train.meta.json", "code": "data/out/code-v2/train.meta.json",
               "general": "data/out/general-v2/train.meta.json"},
        "v1": {"base": "data/stores/base/train.meta.json", "code": "data/stores/code/train.meta.json",
               "general": "data/stores/general/train.meta.json"},
        "build_cmd": ["{python}", "data/build_corpus_v2.py"],
        "build_args": [],
        "reference_tokenizer": "data/stores/base/tokenizer.json",
        "vocab_size": 8192,
    },
    "rouge_mix": {"weights": {"general": 0.6, "base": 0.4}, "max_tokens": 100000000, "block_tokens": 1048576,
                  "out": "runs/longrun/stores"},
    "rsi": {
        "enabled": True,
        "cmd": ["{python}", "-m", "training.rsi.loop"],
        "module_file": "training/rsi/loop.py",
        "rounds": 4,
        "tasks": 64,
        "samples": 4,
        "steps_per_round": 40,
        "min_wall_hours": 0.1,
        # Every promotion needs a human approval; without one within 6 h the
        # candidate is rejected and the champion stays (fail-safe, the run continues).
        "extra_args": ["--approval-timeout-hours", "6"],
        "models": {
            "quasnir": {"family": "code", "gates": ["code", "general"], "replay": "code", "require_approval": True},
            "rouge": {"family": "text", "gates": ["general", "base"], "replay": "rouge_mix", "require_approval": True},
            "darus": {"family": "mixed", "gates": ["general", "code"], "replay": "base", "require_approval": True},
        },
    },
    "merge": {"cmd": ["{python}", "scripts/merge_search.py"], "method": "ties", "density": 0.5, "lambda": 1.0,
              "acceptance": {"max_regression_vs_parents": 0.02},
              "search": {"budget": 16, "select_seed": 777, "select_batches": 16, "rng_seed": 67}},
    "gate": {"tolerance": 0.01, "eval": {"batch": 8, "seq": 256, "batches": 32, "seed": 4242},
             "stores": {"base": ["base"], "quasnir": ["code"], "rouge": ["general"], "darus": ["general", "code"]}},
    "report": {"stores": ["general", "code", "base"],
               "benchmarks": {"darus": ["humaneval", "mbpp"], "quasnir": ["humaneval", "mbpp"]},
               "humaneval_cmd": ["{python}", "scripts/humaneval.py"], "humaneval_args": []},
    "hygiene": {"keep_last": 2, "keep_last_during_run": 3, "keep_last_superseded": 0, "prune_rsi_rounds": True},
    "control": {"poll_seconds": 2.0},
    "wait_for": {"paths": ["runs/base-s/FINAL"],
                 "procs": ["scripts/train_all.sh", "scripts/darus_finish.sh", "data/build_corpus_v2.py"],
                 "poll_seconds": 60},
    "restricted_models": ["rouge"],
}


class Fatal(Exception):
    """Cannot continue without a human (missing corpus, bad config)."""


class Stopped(Exception):
    """STOP file or signal: graceful exit, resumable."""


class StepFailed(Exception):
    """A step failed after its retries; a later start retries it."""


# --------------------------------------------------------------------------- helpers

def now() -> float:
    return time.time()


def iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now() if ts is None else ts))


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def strip_comments(obj):
    if isinstance(obj, dict):
        return {k: strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_comments(v) for v in obj]
    return obj


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False) + "\n")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0: existence check only, nothing is delivered
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_cmdline(pid: int) -> str:
    p = Path(f"/proc/{pid}/cmdline")
    if p.exists():
        try:
            return p.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return ""
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    except OSError:
        return ""


def procs_matching(pattern: str) -> list[int]:
    """PIDs whose command line contains `pattern` (this process excluded)."""
    pids = []
    if shutil.which("pgrep"):
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
        pids = [int(x) for x in out.split() if x.isdigit()]
    elif Path("/proc").is_dir():
        for d in Path("/proc").iterdir():
            if d.name.isdigit() and pattern in pid_cmdline(int(d.name)):
                pids.append(int(d.name))
    return [p for p in pids if p not in (os.getpid(), os.getppid())]


def model_sha(ckpt: Path) -> str | None:
    st = read_json(ckpt / "state.json")
    return st.get("model_sha256") if isinstance(st, dict) else None


def tail(path: Path, n: int = 20) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def tokens_per_step(train_cfg: dict) -> int:
    return int(train_cfg["batch"]) * int(train_cfg["seq_len"]) * int(train_cfg.get("grad_accum", 1))


# --------------------------------------------------------------------------- planning (pure)

def plan_cycle(cfg: dict, g: int, seconds: float, tps: float, tps_source: str, tok_step: int) -> dict:
    """Step counts and hours for one cycle of `seconds` wall time at `tps` tokens/s."""
    sh = cfg["shares"]
    phases = {}
    for fam in TRAIN_FAMILIES:
        sec = sh[fam] * seconds
        steps = max(int(cfg["min_steps"]), int(math.floor(sec * tps / tok_step + 1e-9)))
        phases[fam] = {"run": cfg["names"][fam].format(g=g), "kind": "train", "share": sh[fam], "steps": steps,
                       "tokens": steps * tok_step, "hours": steps * tok_step / tps / 3600}
    for key, kind in (("quasnir_rsi", "rsi"), ("rouge_rsi", "rsi"), ("darus", "merge+rsi"), ("report", "eval")):
        phases[key] = {"kind": kind, "share": sh[key], "hours": sh[key] * seconds / 3600}
    phases["darus"]["run"] = cfg["names"]["darus"].format(g=g)
    return {"generation": g, "seconds": seconds, "hours": seconds / 3600, "tokens_per_s": tps,
            "tps_source": tps_source, "tokens_per_step": tok_step, "phases": phases}


def plan_budget(cfg: dict, days: float, tps: float, tps_source: str, tok_step: int, start_g: int,
                forever: bool = False, max_cycles: int | None = None, cycle_hours: float | None = None) -> dict:
    ch = float(cycle_hours or cfg["cycle_hours"])
    cyc = ch * 3600
    min_sec = cfg["min_cycle_fraction"] * cyc
    total = None if forever else days * 86400
    cycles, remaining, g = [], total, start_g
    while True:
        if max_cycles is not None and len(cycles) >= max_cycles:
            break
        if forever:
            if cycles and max_cycles is None:
                break  # one representative cycle; --forever repeats it
            sec = cyc
        else:
            if remaining < min_sec - 1e-6:
                break
            sec = min(cyc, remaining)
            remaining -= sec
        cycles.append(plan_cycle(cfg, g, sec, tps, tps_source, tok_step))
        g += 1
    totals = {"steps": {f: sum(c["phases"][f]["steps"] for c in cycles) for f in TRAIN_FAMILIES},
              "tokens": {f: sum(c["phases"][f]["tokens"] for c in cycles) for f in TRAIN_FAMILIES},
              "hours": sum(c["hours"] for c in cycles)}
    totals["tokens"]["all"] = sum(totals["tokens"][f] for f in TRAIN_FAMILIES)
    return {"days": None if forever else days, "forever": forever, "cycle_hours": ch, "tokens_per_s": tps,
            "tps_source": tps_source, "tokens_per_step": tok_step, "tokens_per_day": tps * 86400,
            "cycles": cycles, "totals": totals}


# --------------------------------------------------------------------------- mix store

def _find_eos(raw: bytes, lo: int, hi: int, last: bool = False) -> int:
    """Token index of the first/last eos in raw[2*lo : 2*hi] (aligned), or -1."""
    a, b = 2 * lo, 2 * hi
    while a < b:
        i = raw.rfind(EOS_BYTES, a, b) if last else raw.find(EOS_BYTES, a, b)
        if i < 0:
            return -1
        if i % 2 == 0:
            return i // 2
        if last:
            b = i + 1  # misaligned hit (impossible for vocab <= 8192, kept for safety)
        else:
            a = i + 1
    return -1


def _count_docs(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            n += chunk.count(EOS_BYTES)
    return n


def take_tokens(src: Path, n_tokens: int, out, block_tokens: int) -> int:
    """Append about `n_tokens` whole documents of `src` to `out`, sampled as
    evenly spaced, document-aligned blocks. Returns the tokens written."""
    total = src.stat().st_size // 2
    if n_tokens >= total:
        with open(src, "rb") as f:
            shutil.copyfileobj(f, out, 1 << 22)
        return total
    k = max(1, math.ceil(n_tokens / block_tokens))
    stride = total // k
    written = 0
    with open(src, "rb") as f:
        for i in range(k):
            want = min(block_tokens, n_tokens - written)
            if want <= 0:
                break
            start = i * stride
            f.seek(start * 2)
            raw = f.read((want + block_tokens) * 2)
            ntok = len(raw) // 2
            first = 0
            if start > 0:
                e = _find_eos(raw, 0, ntok)
                if e < 0:
                    continue
                first = e + 1
            last = _find_eos(raw, first, min(ntok, first + want), last=True)
            if last < 0:
                continue
            out.write(raw[2 * first: 2 * (last + 1)])
            written += last + 1 - first
    return written


def build_mix_store(root: Path, sources: dict, weights: dict, max_tokens: int, block_tokens: int,
                    out_dir: Path, ref_tokenizer: Path) -> dict:
    """SCP CorpusStore that mixes whole documents of existing stores by token
    weight. No new text enters: documents are re-sliced from stores that were
    already decontaminated by their builder."""
    metas = {n: read_json(root / sources[n]) for n in weights}
    dirs = {n: (root / sources[n]).parent for n in weights}
    ref_sha = sha256_file(ref_tokenizer)
    for n, d in dirs.items():
        if sha256_file(d / "tokenizer.json") != ref_sha:
            raise Fatal(f"{d}/tokenizer.json differs from {ref_tokenizer}: checkpoints would be incompatible")
    wsum = sum(weights.values())
    w = {n: weights[n] / wsum for n in weights}
    avail = {n: int(metas[n]["tokens"]) for n in w}
    # Largest total that respects the weights, the available tokens and the cap.
    total = min([max_tokens] + [avail[n] / w[n] for n in w if w[n] > 0])
    take = {n: int(total * w[n]) for n in w}
    vavail = {n: int(metas[n]["val_tokens"]) for n in w}
    vtotal = min([vavail[n] / w[n] for n in w if w[n] > 0])
    vtake = {n: int(vtotal * w[n]) for n in w}
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    got, vgot = {}, {}
    with open(tmp / "train.bin", "wb") as f:
        for n in w:
            got[n] = take_tokens(dirs[n] / metas[n]["bin"], take[n], f, block_tokens)
    with open(tmp / "train.val.bin", "wb") as f:
        for n in w:
            vgot[n] = take_tokens(dirs[n] / metas[n]["val_bin"], vtake[n], f, block_tokens)
    shutil.copyfile(ref_tokenizer, tmp / "tokenizer.json")
    tokens, vtokens = sum(got.values()), sum(vgot.values())
    docs = _count_docs(tmp / "train.bin")
    meta = {"tokens": tokens, "val_tokens": vtokens, "dtype": "uint16", "vocab_size": int(metas[next(iter(w))]["vocab_size"]),
            "bin": "train.bin", "val_bin": "train.val.bin",
            "mix": {f"{n}-store": got[n] for n in w}, "mix_fractions": {f"{n}-store": got[n] / max(1, tokens) for n in w},
            "mix_unit": "tokens (documents are whole; see manifest)"}
    atomic_write_json(tmp / "train.meta.json", meta)
    licences = {}
    for n, d in dirs.items():
        m = read_json(d / "manifest.json", {}) or {}
        licences[n] = m.get("licences", m.get("licenses", "see source manifest"))
    manifest = {
        "kind": "mix", "created_by": "scripts/longrun.py", "created_at": iso(), "data_kind": "corpus (no model-generated data)",
        "sources": {n: {"meta": sources[n], "meta_sha256": sha256_file(root / sources[n]),
                        "manifest_sha256": sha256_file(dirs[n] / "manifest.json") if (dirs[n] / "manifest.json").exists() else None,
                        "weight": w[n], "train_tokens": got[n], "val_tokens": vgot[n]} for n in w},
        "documents": docs,
        "decontamination": "inherited: every document comes unchanged from the source stores, which their builder "
                           "decontaminated against data/stores/evals (13-gram overlap); this store only re-slices them",
        "licences": licences,
        "files": {name: sha256_file(tmp / name) for name in ("train.bin", "train.val.bin", "tokenizer.json", "train.meta.json")},
    }
    atomic_write_json(tmp / "manifest.json", manifest)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return {"meta": meta, "manifest": manifest}


# --------------------------------------------------------------------------- orchestrator

class LongRun:
    def __init__(self, root: Path, cfg: dict, args: argparse.Namespace):
        self.root = root
        self.cfg = cfg
        self.args = args
        self.dir = root / "runs" / "longrun"
        self.state_path = self.dir / "state.json"
        self.stop_file = self.dir / "STOP"
        self.finished_file = self.dir / "FINISHED"
        self.signal_stop = False
        self.state: dict = read_json(self.state_path) or {}
        self.base_cfg = read_json(root / cfg["base_config"])
        if not isinstance(self.base_cfg, dict):
            raise Fatal(f"base config {cfg['base_config']} not found or invalid")
        self.tok_step = tokens_per_step(self.base_cfg)
        forge = args.forge or os.environ.get("FORGE") or cfg["forge"]
        self.forge = forge if os.path.isabs(forge) else str(root / forge)
        self.poll = float(cfg["control"]["poll_seconds"])
        self.lock_fd = None

    # ---- small utilities
    def p(self, rel: str | Path) -> Path:
        rel = Path(rel)
        return rel if rel.is_absolute() else self.root / rel

    def rel(self, path: Path | str) -> str:
        path = Path(path)
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def name(self, fam: str, g: int) -> str:
        return self.cfg["names"][fam].format(g=g)

    def expand(self, cmd: list) -> list:
        return [sys.executable if c == "{python}" else c for c in cmd]

    def nice(self, cmd: list) -> list:
        n = int(self.cfg.get("nice") or 0)
        return (["nice", "-n", str(n)] + cmd) if n and shutil.which("nice") else cmd

    def log(self, msg: str, **kw) -> None:
        line = f"[longrun {iso()}] {msg}"
        if kw:
            line += " " + json.dumps(kw, default=str)
        print(line, flush=True)
        if not self.args.dry_run:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "events.jsonl", "a") as f:
                f.write(json.dumps({"ts": now(), "msg": msg, **kw}, default=str) + "\n")

    def save(self) -> None:
        self.state["updated_at"] = now()
        atomic_write_json(self.state_path, self.state)

    def status_file(self, **kw) -> None:
        atomic_write_json(self.dir / "status.json", {"ts": now(), "pid": os.getpid(), **kw})

    def stop_requested(self) -> bool:
        return self.signal_stop or self.stop_file.exists()

    def sleep(self, seconds: float) -> None:
        end = now() + seconds
        while now() < end:
            if self.stop_requested():
                raise Stopped("STOP requested while waiting")
            time.sleep(min(self.poll, max(0.0, end - now())))

    # ---- lifecycle
    def acquire(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock_fd = open(self.dir / "longrun.lock", "a+")
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            other = (self.dir / "longrun.pid").read_text().strip() if (self.dir / "longrun.pid").exists() else "?"
            raise SystemExit(self._locked(other))
        atomic_write_text(self.dir / "longrun.pid", f"{os.getpid()}\n")

    def _locked(self, other: str) -> int:
        print(f"[longrun] another longrun holds runs/longrun/longrun.lock (pid {other})", file=sys.stderr)
        return EX_LOCKED

    def release(self) -> None:
        pidf = self.dir / "longrun.pid"
        if pidf.exists() and pidf.read_text().strip() == str(os.getpid()):
            pidf.unlink()

    def init_state(self) -> None:
        s = self.state
        if not s:
            s.update({"version": 1, "created_at": now(), "status": "new", "cycles": {}, "runs": {}, "eval_cache": {},
                      "throughput": {}, "budget_started_at": None, "cycle": None})
            s["current"] = {}
            for fam, path in self.cfg["initial"].items():
                if path:
                    parts = Path(path).parts
                    s["current"][fam] = {"gen": 1, "run": parts[1] if len(parts) > 1 else path, "dir": path,
                                         "source": "initial"}
        s["config_path"] = self.args.config
        s["config_sha256"] = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        if self.args.days is not None:
            s["days"] = self.args.days
        s.setdefault("days", self.cfg["days"])
        if self.args.forever:
            s["forever"] = True
        s.setdefault("forever", bool(self.cfg["forever"]))
        if self.args.max_cycles is not None:
            s["max_cycles"] = self.args.max_cycles
        s.setdefault("max_cycles", self.cfg["max_cycles"])
        if self.args.cycle_hours is not None:
            s["cycle_hours"] = self.args.cycle_hours
        s.setdefault("cycle_hours", self.cfg["cycle_hours"])
        s["pid"] = os.getpid()

    def adopt_orphans(self) -> None:
        """A previous longrun may have died while its child kept running (e.g.
        SIGKILL). Ask that child to checkpoint and stop through its control
        channel and wait for it, so two trainers never write one run."""
        for g, cyc in self.state.get("cycles", {}).items():
            for name, st in cyc.get("steps", {}).items():
                pid = st.get("pid")
                if not pid or not pid_alive(pid):
                    st.pop("pid", None)
                    continue
                cmd = pid_cmdline(pid)
                if not any(k in cmd for k in ("forge", "rsi", "merge_search", "humaneval", "build_corpus")):
                    st.pop("pid", None)
                    continue
                self.log("previous child still running; asking it to stop", pid=pid, step=name, cycle=g)
                ctl = st.get("ctl")
                if ctl and self.p(ctl).exists():
                    try:
                        fd = os.open(self.p(ctl), os.O_WRONLY | os.O_NONBLOCK)
                        os.write(fd, CTL_STOP)
                        os.close(fd)
                    except OSError as e:
                        self.log("could not write to control FIFO", error=str(e))
                if st.get("rsi_out"):
                    self.rsi_stop(self.p(st["rsi_out"]), st["rsi_model"])
                t0 = now()
                while pid_alive(pid) and pid in procs_or_self(pid):
                    if now() - t0 > 3600:
                        raise Fatal(f"previous child {pid} ({cmd.strip()[:80]}) did not exit within 1 h; stop it by hand")
                    time.sleep(min(self.poll, 2.0))
                st.pop("pid", None)
        self.save()

    def run(self) -> int:
        if self.stop_file.exists():
            self.log("runs/longrun/STOP is present; remove it to continue")
            return EX_STOPPED
        self.acquire()
        try:
            self.init_state()
            self.state["status"] = "running"
            if self.finished_file.exists():
                self.finished_file.unlink()
            self.save()
            self.adopt_orphans()
            self.wait_prerequisites()
            self.ensure_corpus()
            self.check_lineage()
            self.ensure_bench()
            while True:
                g = self.next_cycle()
                if g is None:
                    return EX_OK
                if self.args.prepare:
                    cyc = self.state["cycles"][str(g)]
                    path = self.write_train_config("base", g, cyc)
                    self.log("prepared", cycle=g, base_config=path, plan=cyc["plan"]["phases"]["base"])
                    self.state["status"] = "prepared"
                    self.save()
                    return EX_OK
                self.run_cycle(g)
        except Stopped as e:
            self.state["status"] = "stopped"
            self.state["message"] = str(e)
            self.save()
            self.log("stopped", reason=str(e))
            return EX_STOPPED
        except Fatal as e:
            self.state["status"] = "fatal"
            self.state["message"] = str(e)
            self.save()
            self.log("FATAL: " + str(e))
            return EX_FATAL
        except StepFailed as e:
            self.state["status"] = "failed"
            self.state["message"] = str(e)
            self.save()
            self.log("step failed: " + str(e))
            return EX_FAIL
        finally:
            self.release()

    # ---- prerequisites, corpus, throughput
    def wait_prerequisites(self) -> None:
        if self.args.no_wait:
            return
        wf = self.cfg["wait_for"]
        announced = None
        while True:
            missing = [p for p in wf.get("paths", []) if not self.p(p).exists()]
            busy = {pat: procs_matching(pat) for pat in wf.get("procs", [])}
            busy = {k: v for k, v in busy.items() if v}
            if not missing and not busy:
                if announced is not None:
                    self.log("prerequisites satisfied")
                return
            why = {"missing": missing, "running": busy}
            if why != announced:
                self.log("waiting for prerequisites", **why)
                self.status_file(status="waiting", **why)
                announced = why
            self.state["status"] = "waiting"
            self.save()
            self.sleep(float(wf.get("poll_seconds", 60)))

    def corpus_status(self, allow_v1: bool) -> dict:
        c = self.cfg["corpus"]
        missing = [m for m in c["v2"].values() if not self.p(m).exists()]
        return {"v2_missing": missing, "would_build": bool(missing),
                "build_cmd": self.expand(c["build_cmd"]) + list(c["build_args"]), "allow_v1": allow_v1}

    def validate_store(self, meta_path: str) -> dict:
        c = self.cfg["corpus"]
        mp = self.p(meta_path)
        meta = read_json(mp)
        if not isinstance(meta, dict):
            raise Fatal(f"{meta_path}: missing or not JSON")
        d = mp.parent
        for k in ("bin", "val_bin"):
            if not (d / meta.get(k, "?")).exists():
                raise Fatal(f"{meta_path}: {k} file {meta.get(k)} missing")
        if meta.get("dtype") != "uint16" or int(meta.get("vocab_size", 0)) != int(c["vocab_size"]):
            raise Fatal(f"{meta_path}: dtype/vocab_size {meta.get('dtype')}/{meta.get('vocab_size')} != uint16/{c['vocab_size']}")
        ref = self.p(c["reference_tokenizer"])
        if sha256_file(d / "tokenizer.json") != sha256_file(ref):
            raise Fatal(f"{d}/tokenizer.json is not byte-identical to {c['reference_tokenizer']}")
        return {"meta": meta_path, "tokens": int(meta["tokens"]), "val_tokens": int(meta["val_tokens"]),
                "meta_sha256": sha256_file(mp)}

    def ensure_corpus(self) -> None:
        c = self.cfg["corpus"]
        allow_v1 = bool(self.args.allow_v1)
        missing = [m for m in c["v2"].values() if not self.p(m).exists()]
        build = None
        if missing:
            cmd = self.expand(c["build_cmd"]) + list(c["build_args"])
            script = next((x for x in cmd[1:] if x.endswith(".py")), None)
            if script and not self.p(script).exists():
                build = {"cmd": cmd, "rc": None, "error": f"{script} not found"}
            else:
                self.log("corpus v2 missing; building", missing=missing, cmd=cmd)
                logp = self.dir / "logs" / "corpus.log"
                rc = self.run_logged(cmd, logp, stop_kill=True)
                build = {"cmd": cmd, "rc": rc, "log": self.rel(logp)}
            missing = [m for m in c["v2"].values() if not self.p(m).exists()]
        if not missing:
            stores = {k: self.validate_store(v) for k, v in c["v2"].items()}
            version = "v2"
        elif allow_v1:
            self.log("WARNING: corpus v2 unavailable; falling back to the v1 stores (--allow-v1)", missing=missing, build=build)
            stores = {k: self.validate_store(v) for k, v in c["v1"].items()}
            version = "v1-fallback"
        else:
            detail = f"; build {build['cmd']} -> rc {build['rc']}" + (f" ({build.get('error')})" if build.get("error") else
                                                                     f", log {build.get('log')}") if build else ""
            raise Fatal("corpus v2 missing: " + ", ".join(missing) + detail +
                        ". Fix data/build_corpus_v2.py (args in longrun.json corpus.build_args) or pass --allow-v1.")
        self.state["corpus"] = {"version": version, "stores": stores, "checked_at": now(), "build": build}
        self.save()

    def store_meta(self, name: str) -> str | None:
        if name == "rouge_mix":
            return (self.state.get("mix") or {}).get("meta")
        return self.state["corpus"]["stores"][name]["meta"]

    def check_lineage(self) -> None:
        cur = self.state["current"].get("base")
        if not cur or not self.p(cur["dir"]).exists():
            return
        st = read_json(self.p(cur["dir"]) / "state.json") or {}
        have = (st.get("config") or {}).get("model") or {}
        want = self.base_cfg["model"]
        keys = ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "ffn_hidden", "max_seq_len")
        diff = {k: (have.get(k), want.get(k)) for k in keys if have.get(k) != want.get(k)}
        if diff:
            raise Fatal(f"current base {cur['dir']} has a different model than {self.cfg['base_config']}: {diff}. "
                        "Set initial.* to null in longrun.json to start from scratch with this config.")

    def ensure_bench(self) -> None:
        tp = self.state.setdefault("throughput", {})
        if tp.get("bench") or self.args.tok_per_s:
            return
        cmd = self.nice([self.forge, "bench", "--config", self.cfg["base_config"],
                         "--steps", str(self.cfg["throughput"]["bench_steps"])])
        load = os.getloadavg() if hasattr(os, "getloadavg") else None
        self.log("measuring throughput", cmd=cmd, loadavg=load)
        r = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
        rec = None
        for line in r.stdout.splitlines():
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if v.get("type") == "train_bench":
                rec = v
        if r.returncode != 0 or rec is None:
            self.log("WARNING: forge bench failed; planning with the assumed throughput", rc=r.returncode,
                     stderr=r.stderr[-500:])
            tp["bench"] = None
            tp["bench_error"] = r.stderr[-500:]
        else:
            tp["bench"] = {"tokens_per_s": rec["tokens_per_s"], "step_ms": rec.get("step_ms"), "params": rec.get("params"),
                           "steps": rec.get("steps"), "config": self.cfg["base_config"], "measured_at": now(),
                           "loadavg_before": load}
            self.log("throughput measured", **tp["bench"])
        self.save()

    def telemetry_tps(self, run: str) -> tuple[float | None, int]:
        vals = []
        path = self.root / "runs" / f"{run}.jsonl"
        if not path.exists():
            return None, 0
        with open(path) as f:
            for line in f:
                if '"telemetry"' not in line:
                    continue
                try:
                    v = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if v.get("type") == "telemetry" and isinstance(v.get("tokens_per_s"), (int, float)) and v["tokens_per_s"] > 0:
                    vals.append(float(v["tokens_per_s"]))
        vals = vals[-500:]
        return (statistics.median(vals) if vals else None), len(vals)

    def choose_tps(self, g: int) -> tuple[float, str]:
        if self.args.tok_per_s:
            return float(self.args.tok_per_s), "--tok-per-s"
        tp = self.state.get("throughput", {})
        prev = self.state.get("cycles", {}).get(str(g - 1))
        if prev:
            med, n = self.telemetry_tps(self.name("base", g - 1))
            if med and n >= self.cfg["throughput"]["telemetry_min_samples"]:
                tp.setdefault("telemetry", {})[self.name("base", g - 1)] = {"median_tokens_per_s": med, "samples": n}
                return med, f"median telemetry of {self.name('base', g - 1)} ({n} samples)"
        if tp.get("bench"):
            return float(tp["bench"]["tokens_per_s"]), f"forge bench {iso(tp['bench']['measured_at'])}"
        return float(self.cfg["throughput"]["assumed_tokens_per_s"]), "assumed (longrun.json throughput.assumed_tokens_per_s)"

    # ---- cycles
    def next_cycle(self) -> int | None:
        s = self.state
        g = s.get("cycle")
        if g is not None and s["cycles"].get(str(g), {}).get("status") != "done":
            return g
        g = self.cfg["first_generation"] if g is None else g + 1
        done = sum(1 for c in s["cycles"].values() if c.get("status") == "done")
        if s.get("max_cycles") is not None and done >= s["max_cycles"]:
            return self.finish(f"max_cycles {s['max_cycles']} reached")
        if s.get("budget_started_at") is None:
            s["budget_started_at"] = now()
        cyc_sec = float(s["cycle_hours"]) * 3600
        if s.get("forever"):
            sec = cyc_sec
        else:
            left = s["budget_started_at"] + float(s["days"]) * 86400 - now()
            if left < self.cfg["min_cycle_fraction"] * cyc_sec:
                return self.finish(f"wall budget of {s['days']} days used (left {left / 3600:.2f} h)")
            sec = min(cyc_sec, left)
        tps, src = self.choose_tps(g)
        plan = plan_cycle(self.cfg, g, sec, tps, src, self.tok_step)
        s["cycles"][str(g)] = {"status": "running", "started_at": now(), "plan": plan, "steps": {}}
        s["cycle"] = g
        self.save()
        self.log(f"cycle g{g} planned", hours=round(plan["hours"], 3), tokens_per_s=round(tps, 1), source=src,
                 steps={f: plan["phases"][f]["steps"] for f in TRAIN_FAMILIES})
        return g

    def finish(self, reason: str) -> None:
        self.state["status"] = "finished"
        self.state["message"] = reason
        self.save()
        atomic_write_json(self.finished_file, {"ts": now(), "reason": reason})
        self.status_file(status="finished", reason=reason)
        self.log("finished: " + reason)
        return None

    def run_cycle(self, g: int) -> None:
        cyc = self.state["cycles"][str(g)]
        for name in CYCLE_STEPS:
            st = cyc["steps"].setdefault(name, {"status": "pending"})
            if st["status"] in ("done", "skipped", "error"):
                continue
            if self.stop_requested():
                raise Stopped("STOP requested between steps")
            st["status"] = "running"
            st.setdefault("started_at", now())
            self.save()
            self.status_file(status="running", cycle=g, step=name)
            self.log(f"g{g} {name}: start")
            t0 = now()
            try:
                res = getattr(self, f"step_{name}")(g, cyc, st) or {}
            finally:
                st["wall_seconds"] = st.get("wall_seconds", 0.0) + now() - t0
                self.save()
            st.update(res)
            st["status"] = res.get("status", "done")
            st["finished_at"] = now()
            self.save()
            self.log(f"g{g} {name}: {st['status']}", **{k: v for k, v in res.items() if k in ("decision", "reason", "final", "model_dir")})
        cyc["status"] = "done"
        cyc["finished_at"] = now()
        self.save()

    # ---- training
    def write_train_config(self, fam: str, g: int, cyc: dict, init_from: str | None = None, data: str | None = None) -> str:
        run = self.name(fam, g)
        path = self.dir / "configs" / f"{run}.json"
        if path.exists():
            return self.rel(path)
        if fam == "base":
            cur = self.state["current"].get("base")
            init_from = cur["dir"] if cur and self.p(cur["dir"]).exists() else None
            if cur and init_from is None:
                raise Fatal(f"current base {cur['dir']} does not exist")
            data = self.store_meta("base")
        steps = cyc["plan"]["phases"][fam]["steps"]
        b = self.base_cfg
        t = self.cfg["train"]
        d, ph = t["defaults"], t[fam]
        scratch = init_from is None
        lr = b["lr"] if scratch else ph["lr"]
        min_lr = b["min_lr"] if scratch else ph["min_lr"]
        warm = b.get("warmup_steps", ph["warmup_steps"]) if scratch else ph["warmup_steps"]
        cfg = {"run": run, "model": b["model"], "data": data, "batch": b["batch"], "seq_len": b["seq_len"],
               "grad_accum": b.get("grad_accum", 1), "max_steps": steps, "lr": lr, "min_lr": min(min_lr, lr),
               "warmup_steps": max(1, min(warm, steps // 10)), "weight_decay": b.get("weight_decay", 0.1),
               "grad_clip": b.get("grad_clip", 1.0), "seed": int(ph["seed"]) + g, "out_dir": "runs",
               "log_every": d["log_every"], "eval_every": d["eval_every"], "eval_batches": d["eval_batches"],
               "ckpt_every": d["ckpt_every"], "threads": d["threads"]}
        if init_from:
            cfg["init_from"] = init_from
        for k in ("hbvm_mib", "autotune"):
            if k in b:
                cfg[k] = b[k]
        atomic_write_json(path, cfg)
        return self.rel(path)

    def train(self, run: str, cfg_path: str, st: dict, restricted: bool = False) -> str:
        cfg = read_json(self.p(cfg_path))
        max_steps = int(cfg["max_steps"])
        run_dir = self.root / "runs" / run
        if restricted:
            run_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(run_dir, 0o700)
        self.state["runs"].setdefault(run, {"config": cfg_path, "finished": False, "family": st.get("family")})
        attempts = 0
        while True:
            latest = read_json(run_dir / "latest.json")
            if latest and int(latest.get("step", 0)) >= max_steps:
                return self.finalize(run, latest)
            if self.stop_requested():
                raise Stopped(f"STOP before {run}")
            self.check_disk()
            cmd = [self.forge, "train", "--config", cfg_path]
            if latest:
                cmd += ["--resume", latest["dir"]]
            rc, stopped = self.forge_train(run, cmd, st, restricted)
            if stopped:
                raise Stopped(f"{run} checkpointed and stopped")
            latest = read_json(run_dir / "latest.json")
            if rc == 0 and latest and int(latest.get("step", 0)) >= max_steps:
                return self.finalize(run, latest)
            if rc == 0:
                self.log(f"{run} exited before max_steps (stop via control FIFO?); resuming", step=(latest or {}).get("step"))
                continue
            attempts += 1
            st["failures"] = st.get("failures", 0) + 1
            err = tail(self.dir / "logs" / f"{run}.stderr", 5)
            if attempts > int(self.cfg["max_retries"]):
                raise StepFailed(f"forge train {run} failed {attempts}x (rc {rc}): {err}")
            wait = float(self.cfg["retry_backoff_seconds"]) * attempts
            self.log(f"{run} failed (rc {rc}); retrying in {wait:.0f}s from its newest checkpoint", stderr=err)
            self.sleep(wait)

    def forge_train(self, run: str, cmd: list, st: dict, restricted: bool) -> tuple[int, bool]:
        runs = self.root / "runs"
        fifo = runs / f"{run}.ctl"
        if fifo.exists() and not fifo.is_fifo():
            raise Fatal(f"{fifo} exists and is not a FIFO")
        if not fifo.exists():
            os.mkfifo(fifo)
        jl_path = runs / f"{run}.jsonl"
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        fd = os.open(fifo, os.O_RDWR)  # like `<>` in train_all.sh: never blocks, never sees EOF
        stopped = threading.Event()
        done = threading.Event()

        def watch():
            while not done.is_set():
                if self.stop_requested() and not stopped.is_set():
                    try:
                        os.write(fd, CTL_STOP)
                        self.log(f"{run}: sent checkpoint + stop through {self.rel(fifo)}")
                    except OSError as e:
                        self.log(f"{run}: could not write control FIFO", error=str(e))
                    stopped.set()
                done.wait(self.poll)

        with open(self.dir / "logs" / f"{run}.stderr", "ab") as err, open(jl_path, "a") as jl:
            if restricted:
                os.chmod(jl_path, 0o600)
            self.log(f"{run}: " + " ".join(cmd))
            proc = subprocess.Popen(self.nice(cmd), cwd=self.root, stdin=fd, stdout=subprocess.PIPE, stderr=err,
                                    text=True, bufsize=1, start_new_session=True)
            st.update(pid=proc.pid, ctl=self.rel(fifo))
            self.save()
            w = threading.Thread(target=watch, daemon=True)
            w.start()
            try:
                for line in proc.stdout:
                    jl.write(line if line.endswith("\n") else line + "\n")
                    jl.flush()
                    self.on_train_line(run, line)
                rc = proc.wait()
            finally:
                done.set()
                w.join()
                os.close(fd)
        st.pop("pid", None)
        self.save()
        return rc, stopped.is_set()

    def on_train_line(self, run: str, line: str) -> None:
        if '"type":"telemetry"' in line or '"type": "telemetry"' in line:
            return
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            return
        t = v.get("type")
        if t == "checkpoint":
            self.prune_run(run, int(self.cfg["hygiene"]["keep_last_during_run"]))
        if t in ("eval", "checkpoint", "done") or (t == "event" and v.get("level") in ("warn", "error")):
            self.log(f"{run}: {t}", **{k: v[k] for k in ("step", "val_loss", "val_acc", "path", "msg") if k in v})

    def finalize(self, run: str, latest: dict) -> str:
        run_dir = self.root / "runs" / run
        target = self.p(latest["dir"]).resolve()
        if target.parent != run_dir.resolve() or not (target / "model.safetensors").exists():
            raise StepFailed(f"{run}: latest.json points to {latest['dir']}, not a checkpoint of this run")
        link = run_dir / "FINAL"
        tmp = run_dir / ".FINAL.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(target.name, tmp)  # relative: survives moving the repository to other hardware
        os.replace(tmp, link)
        info = self.state["runs"].setdefault(run, {})
        info.update(finished=True, final=f"runs/{run}/FINAL", step=int(latest["step"]), final_target=target.name,
                    sha256=latest.get("model_sha256"))
        self.save()
        self.prune_run(run, int(self.cfg["hygiene"]["keep_last"]))
        return f"runs/{run}/FINAL"

    def check_disk(self) -> None:
        need = float(self.cfg.get("min_free_gb") or 0) * 1e9
        if not need:
            return
        if shutil.disk_usage(self.root).free < need:
            self.hygiene_all(superseded_only=True)
            free = shutil.disk_usage(self.root).free
            if free < need:
                raise StepFailed(f"only {free / 1e9:.1f} GB free (< min_free_gb {self.cfg['min_free_gb']}); free disk space")

    # ---- disk hygiene
    def protected(self) -> set:
        """Real paths that must never be deleted: every FINAL target, every
        current model, every RSI champion (and its history), every latest.json."""
        keep = set()
        runs = self.root / "runs"
        for link in runs.glob("*/FINAL"):
            keep.add(link.resolve())
        for latest in runs.glob("*/latest.json"):
            v = read_json(latest)
            if v and v.get("dir"):
                keep.add(self.p(v["dir"]).resolve())
        for cur in self.state.get("current", {}).values():
            if cur and cur.get("dir"):
                keep.add(self.p(cur["dir"]).resolve())
        for cj in runs.glob("rsi/*/champion.json"):
            v = read_json(cj) or {}
            for h in [v] + list(v.get("history") or []):
                if h.get("dir"):
                    keep.add(self.p(h["dir"]).resolve())
        return keep

    def _prune_dir(self, run_dir: Path, keep_last: int, protect: set) -> list:
        steps = sorted((int(m.group(1)), d) for d in run_dir.iterdir()
                       if (m := STEP_RE.match(d.name)) and d.is_dir() and not d.is_symlink())
        victims = steps[:-keep_last] if keep_last > 0 else steps
        removed = []
        for _, d in victims:
            if d.resolve() in protect or d.name == "FINAL":
                continue
            shutil.rmtree(d)
            removed.append(self.rel(d))
        return removed

    def prune_run(self, run: str, keep_last: int) -> list:
        run_dir = self.root / "runs" / run
        if run not in self.state.get("runs", {}) or not run_dir.is_dir():
            return []  # only runs this orchestrator created are ever pruned
        removed = self._prune_dir(run_dir, keep_last, self.protected())
        if removed:
            self.log(f"{run}: pruned old checkpoints", removed=removed)
        return removed

    def hygiene_all(self, superseded_only: bool = False) -> list:
        h = self.cfg["hygiene"]
        cur_runs = {c.get("run") for c in self.state.get("current", {}).values() if c}
        g_now = self.state.get("cycle")
        removed = []
        for run, info in self.state.get("runs", {}).items():
            if not info.get("finished"):
                continue
            m = re.search(r"-g(\d+)$", run)
            gen = int(m.group(1)) if m else None
            superseded = run not in cur_runs and gen is not None and g_now is not None and gen < g_now
            if superseded_only and not superseded:
                continue
            removed += self.prune_run(run, int(h["keep_last_superseded"] if superseded else h["keep_last"]))
        return removed

    def prune_rsi(self, model: str) -> list:
        out = self.root / "runs" / "rsi" / model
        if not self.cfg["hygiene"]["prune_rsi_rounds"] or not (out / "rounds").is_dir():
            return []
        protect = self.protected()
        removed = []
        for rd in sorted((out / "rounds").iterdir()):
            if rd.is_dir():
                removed += self._prune_dir(rd, 0, protect)
        if removed:
            self.log(f"rsi {model}: pruned non-champion round checkpoints", removed=len(removed))
        return removed

    # ---- evaluation and gates
    def eval_ckpt(self, ckpt: str, store: str) -> dict:
        meta = self.store_meta(store)
        e = self.cfg["gate"]["eval"]
        sha = model_sha(self.p(ckpt)) or "?"
        key = "|".join([sha, meta, sha256_file(self.p(meta)), str(e["batch"]), str(e["seq"]), str(e["batches"]), str(e["seed"])])
        cache = self.state.setdefault("eval_cache", {})
        if key in cache:
            return cache[key]
        cmd = self.nice([self.forge, "eval", "--ckpt", ckpt, "--data", meta, "--split", "val", "--batch", str(e["batch"]),
                         "--seq", str(e["seq"]), "--batches", str(e["batches"]), "--seed", str(e["seed"])])
        r = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
        if r.returncode != 0:
            raise StepFailed(f"forge eval {ckpt} on {store}: rc {r.returncode}: {r.stderr[-300:]}")
        v = json.loads(r.stdout.strip().splitlines()[-1])
        res = {k: v.get(k) for k in ("loss", "perplexity", "next_token_acc")}
        res.update(ckpt=ckpt, sha256=sha, store=store, meta=meta, measured_at=now(), split="val",
                   settings={k: e[k] for k in ("batch", "seq", "batches", "seed")})
        cache[key] = res
        self.save()
        return res

    def gate(self, fam: str, g: int, run: str, candidate: str, source: str) -> dict:
        tol = float(self.cfg["gate"]["tolerance"])
        prev = self.state["current"].get(fam)
        if prev and not self.p(prev["dir"]).exists():
            prev = None
        stores, worst = {}, None
        for s in self.cfg["gate"]["stores"][fam]:
            c = self.eval_ckpt(candidate, s)
            row = {"candidate": c["loss"]}
            if prev:
                pv = self.eval_ckpt(prev["dir"], s)
                row["previous"] = pv["loss"]
                row["relative_change"] = c["loss"] / pv["loss"] - 1
                worst = row["relative_change"] if worst is None else max(worst, row["relative_change"])
            stores[s] = row
        accept = prev is None or worst <= tol
        decision = {"family": fam, "generation": g, "decision": "accept" if accept else "reject",
                    "candidate": {"run": run, "dir": candidate, "sha256": model_sha(self.p(candidate)), "source": source},
                    "previous": prev, "stores": stores, "tolerance": tol, "worst_relative_change": worst,
                    "rule": "candidate val loss <= previous * (1 + tolerance) on every gate store",
                    "reason": "no previous model" if prev is None else
                    (f"worst change {worst:+.4f} within {tol}" if accept else f"worst change {worst:+.4f} exceeds {tol}"),
                    "ts": now()}
        if accept:
            self.state["current"][fam] = {"gen": g, "run": run, "dir": candidate, "sha256": decision["candidate"]["sha256"],
                                          "source": source}
            atomic_write_json(self.dir / "current.json", {"ts": now(), "current": self.state["current"]})
        self.save()
        return decision

    # ---- RSI
    def rsi_stop(self, out: Path, model: str) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "STOP").touch()
        (self.dir / f"rsi-stop-{model}").touch()  # marker: longrun created this STOP and removes it on restart

    def rsi_rounds_done(self, out: Path) -> int:
        n = 0
        audit = out / "audit.jsonl"
        if audit.exists():
            for line in audit.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "round" in r and r.get("decision") != "stopped":
                    n += 1
        return n

    def rsi_summary(self, out: Path, champion: str) -> dict:
        recs = []
        if (out / "audit.jsonl").exists():
            for line in (out / "audit.jsonl").read_text().splitlines():
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        cj = read_json(out / "champion.json") or {}
        model_dir = champion
        if cj.get("dir") and (self.p(cj["dir"]) / "model.safetensors").exists():
            model_dir = self.rel(self.p(cj["dir"]))
        return {"model_dir": model_dir, "promoted": model_dir != champion, "rounds": len(recs),
                "decisions": dict(Counter(r.get("decision") for r in recs)),
                "kinds": dict(Counter(r.get("kind") for r in recs)),
                "self_improvement_rounds": sum(1 for r in recs if r.get("self_improvement")),
                "heldout_pass": [(r.get("round"), r.get("heldout_pass_before"), r.get("heldout_pass_after")) for r in recs],
                "audit": self.rel(out / "audit.jsonl")}

    def rsi(self, fam: str, g: int, champion: str, hours: float, st: dict) -> dict:
        rc_cfg = self.cfg["rsi"]
        mc = rc_cfg["models"][fam]
        model = self.name(fam, g)
        out_rel = f"runs/rsi/{model}"
        out = self.p(out_rel)
        if not rc_cfg["enabled"]:
            return {"status": "skipped", "reason": "rsi disabled in longrun.json", "model_dir": champion}
        if rc_cfg.get("module_file") and not self.p(rc_cfg["module_file"]).exists():
            return {"status": "skipped", "reason": f"{rc_cfg['module_file']} not found", "model_dir": champion}
        marker = self.dir / f"rsi-stop-{model}"
        if marker.exists():
            if (out / "STOP").exists():
                (out / "STOP").unlink()
            marker.unlink()
        remaining = int(rc_cfg["rounds"]) - self.rsi_rounds_done(out)
        left = hours - st.get("spent_hours", 0.0)
        if remaining <= 0 or left < float(rc_cfg["min_wall_hours"]):
            return {**self.rsi_summary(out, champion), "budget_hours": hours, "spent_hours": st.get("spent_hours", 0.0),
                    "reason": "rounds done" if remaining <= 0 else "wall budget used"}
        if self.stop_requested():
            raise Stopped(f"STOP before RSI {model}")
        cmd = self.expand(rc_cfg["cmd"]) + ["--model", model, "--champion", champion, "--family", mc["family"]]
        for gname in mc["gates"]:
            cmd += ["--gate", f"{gname}={self.store_meta(gname)}"]
        replay = self.store_meta(mc["replay"]) if mc.get("replay") else None
        if replay:
            cmd += ["--replay", replay]
        cmd += ["--rounds", str(remaining), "--tasks", str(rc_cfg["tasks"]), "--samples", str(rc_cfg["samples"]),
                "--steps-per-round", str(rc_cfg["steps_per_round"]), "--max-wall-hours", f"{left:.4f}", "--out", out_rel]
        if mc.get("require_approval"):
            cmd.append("--require-approval")
        cmd += list(rc_cfg.get("extra_args") or [])
        restricted = fam in self.cfg["restricted_models"]
        if restricted:
            out.mkdir(parents=True, exist_ok=True)
            os.chmod(out, 0o700)
        logp = self.dir / "logs" / f"rsi-{model}.log"
        st.update(rsi_out=out_rel, rsi_model=model, cmd=cmd)
        t0 = now()
        rc = self.run_logged(cmd, logp, st=st, on_stop=lambda: self.rsi_stop(out, model))
        st["spent_hours"] = st.get("spent_hours", 0.0) + (now() - t0) / 3600
        self.save()
        if self.stop_requested():
            raise Stopped(f"RSI {model} stopped")
        res = {**self.rsi_summary(out, champion), "budget_hours": hours, "spent_hours": st["spent_hours"],
               "require_approval": bool(mc.get("require_approval")), "log": self.rel(logp)}
        if restricted and out.exists():
            os.chmod(out, 0o700)
        if rc != 0:
            res.update(status="error", rc=rc, error=tail(logp, 8))
            self.log(f"RSI {model} exited with rc {rc}; continuing with {res['model_dir']}")
        return res

    def run_logged(self, cmd: list, logp: Path, st: dict | None = None, on_stop=None, stop_kill: bool = False) -> int:
        """Run a long command with output appended to `logp`; on STOP call
        `on_stop` (or terminate the command's process group if `stop_kill`)."""
        logp.parent.mkdir(parents=True, exist_ok=True)
        with open(logp, "ab") as lf:
            lf.write(f"\n# {iso()} {' '.join(cmd)}\n".encode())
            lf.flush()
            proc = subprocess.Popen(self.nice(cmd), cwd=self.root, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            if st is not None:
                st["pid"] = proc.pid
                self.save()
            sent = False
            while proc.poll() is None:
                if not sent and self.stop_requested():
                    sent = True
                    if on_stop:
                        on_stop()
                    elif stop_kill:
                        os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(min(self.poll, 1.0))
            if st is not None:
                st.pop("pid", None)
            return proc.returncode

    # ---- cycle steps
    def step_base(self, g, cyc, st):
        st["family"] = "base"
        path = self.write_train_config("base", g, cyc)
        init = read_json(self.p(path)).get("init_from")
        return {"final": self.train(self.name("base", g), path, st), "config": path, "init_from": init}

    def step_base_gate(self, g, cyc, st):
        run = self.name("base", g)
        d = self.gate("base", g, run, f"runs/{run}/FINAL", "final")
        cyc["parent_base"] = self.state["current"]["base"]["dir"]
        return d

    def _specialist(self, fam, g, cyc, st, data):
        st["family"] = fam
        parent = cyc.get("parent_base") or self.state["current"]["base"]["dir"]
        path = self.write_train_config(fam, g, cyc, init_from=parent, data=data)
        restricted = fam in self.cfg["restricted_models"]
        return {"final": self.train(self.name(fam, g), path, st, restricted), "config": path, "init_from": parent}

    def step_quasnir(self, g, cyc, st):
        return self._specialist("quasnir", g, cyc, st, self.store_meta("code"))

    def step_quasnir_rsi(self, g, cyc, st):
        return self.rsi("quasnir", g, f"runs/{self.name('quasnir', g)}/FINAL", cyc["plan"]["phases"]["quasnir_rsi"]["hours"], st)

    def _gate_after_rsi(self, fam, g, cyc, rsi_step):
        run = self.name(fam, g)
        final = f"runs/{run}/FINAL"
        r = cyc["steps"].get(rsi_step, {})
        cand = r.get("model_dir") or final
        return self.gate(fam, g, run, cand, "rsi-champion" if cand != final else "final")

    def step_quasnir_gate(self, g, cyc, st):
        return self._gate_after_rsi("quasnir", g, cyc, "quasnir_rsi")

    def step_rouge_mix(self, g, cyc, st):
        m = self.cfg["rouge_mix"]
        sources = {n: self.store_meta(n) for n in m["weights"]}
        key = hashlib.sha256(json.dumps({"inputs": {n: sha256_file(self.p(p)) for n, p in sources.items()},
                                         "weights": m["weights"], "max_tokens": m["max_tokens"],
                                         "block": m["block_tokens"]}, sort_keys=True).encode()).hexdigest()
        out = self.p(m["out"]) / f"rouge-mix-{key[:12]}"
        cur = self.state.get("mix") or {}
        if cur.get("key") == key and (out / "train.meta.json").exists():
            return {"meta": cur["meta"], "reused": True}
        self.log("building the Rouge mix store", out=self.rel(out), weights=m["weights"])
        res = build_mix_store(self.root, sources, m["weights"], int(m["max_tokens"]), int(m["block_tokens"]), out,
                              self.p(self.cfg["corpus"]["reference_tokenizer"]))
        self.state["mix"] = {"key": key, "meta": self.rel(out / "train.meta.json"), "tokens": res["meta"]["tokens"],
                             "val_tokens": res["meta"]["val_tokens"], "built_at": now()}
        self.save()
        return {"meta": self.state["mix"]["meta"], "tokens": res["meta"]["tokens"], "reused": False}

    def step_rouge(self, g, cyc, st):
        return self._specialist("rouge", g, cyc, st, self.store_meta("rouge_mix"))

    def step_rouge_rsi(self, g, cyc, st):
        return self.rsi("rouge", g, f"runs/{self.name('rouge', g)}/FINAL", cyc["plan"]["phases"]["rouge_rsi"]["hours"], st)

    def step_rouge_gate(self, g, cyc, st):
        return self._gate_after_rsi("rouge", g, cyc, "rouge_rsi")

    def step_darus_merge(self, g, cyc, st):
        run = self.name("darus", g)
        children = [f"runs/{self.name('rouge', g)}/FINAL", f"runs/{self.name('quasnir', g)}/FINAL"]
        base = cyc.get("parent_base") or self.state["current"]["base"]["dir"]
        out = f"runs/{run}/FINAL"
        m = self.cfg["merge"]
        mcfg = {"name": run, "base": base, "children": children, "method": m["method"], "density": m["density"],
                "lambda": m["lambda"], "out": out, "suites": {"general": self.store_meta("general"), "code": self.store_meta("code")},
                "acceptance": {**m["acceptance"], "suites": ["general", "code"]}, "search": m["search"]}
        path = self.dir / "configs" / f"{run}.merge.json"
        if not path.exists():
            atomic_write_json(path, mcfg)
        search = self.root / "runs" / run / "search.json"
        if not (search.exists() and (self.p(out) / "state.json").exists()):
            logp = self.dir / "logs" / f"{run}-search.log"
            rc = self.run_logged(self.expand(m["cmd"]) + ["--config", self.rel(path)], logp, st=st)
            if rc != 0 or not (self.p(out) / "state.json").exists():
                return {"status": "error", "rc": rc, "error": tail(logp, 8), "config": self.rel(path)}
        s = read_json(search) or {}
        best = s.get("best") or {}
        return {"final": out, "config": self.rel(path), "search": self.rel(search),
                "best": {k: best.get(k) for k in ("method", "density", "lambda", "objective", "losses")}}

    def step_darus_rsi(self, g, cyc, st):
        merge = cyc["steps"].get("darus_merge", {})
        if merge.get("status") != "done":
            return {"status": "skipped", "reason": "darus merge did not succeed"}
        hours = cyc["plan"]["phases"]["darus"]["hours"] - merge.get("wall_seconds", 0.0) / 3600
        return self.rsi("darus", g, merge["final"], hours, st)

    def step_darus_gate(self, g, cyc, st):
        if cyc["steps"].get("darus_merge", {}).get("status") != "done":
            return {"status": "skipped", "reason": "darus merge did not succeed"}
        return self._gate_after_rsi("darus", g, cyc, "darus_rsi")

    def generation_models(self, g, cyc) -> dict:
        models = {"base": f"runs/{self.name('base', g)}/FINAL"}
        for fam, rsi_step in (("quasnir", "quasnir_rsi"), ("rouge", "rouge_rsi"), ("darus", "darus_rsi")):
            final = (cyc["steps"].get("darus_merge", {}).get("final") if fam == "darus" else f"runs/{self.name(fam, g)}/FINAL")
            cand = cyc["steps"].get(rsi_step, {}).get("model_dir") or final
            if cand and (self.p(cand) / "state.json").exists():
                models[fam] = cand
        return models

    def step_report(self, g, cyc, st):
        models = self.generation_models(g, cyc)
        evals, benches, errors = {}, {}, []
        for fam, ckpt in models.items():
            evals[fam] = {}
            for s in self.cfg["report"]["stores"]:
                try:
                    r = self.eval_ckpt(ckpt, s)
                    evals[fam][s] = {k: r[k] for k in ("loss", "perplexity", "next_token_acc")}
                except StepFailed as e:
                    errors.append(str(e))
        prev_bench = st.get("benchmarks") or {}
        for fam, suites in self.cfg["report"]["benchmarks"].items():
            if fam not in models:
                continue
            for suite in suites:
                key = f"{fam}/{suite}"
                if key in prev_bench:
                    benches.setdefault(fam, {})[suite] = prev_bench[key]
                    continue
                if self.stop_requested():
                    raise Stopped("STOP during report")
                cmd = self.nice(self.expand(self.cfg["report"]["humaneval_cmd"]) + [models[fam], "--suite", suite]
                                + list(self.cfg["report"]["humaneval_args"]))
                r = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
                lines = [x for x in r.stdout.splitlines() if x.startswith("{")]
                if r.returncode != 0 or not lines:
                    errors.append(f"{key}: rc {r.returncode} {r.stderr[-200:]}")
                    continue
                res = json.loads(lines[-1])
                res["measured_at"] = now()
                benches.setdefault(fam, {})[suite] = res
                prev_bench[key] = res
                st["benchmarks"] = prev_bench
                self.save()
        gates = {f: cyc["steps"].get(f"{f}_gate", {}) for f in FAMILIES}
        rsi = {f: {k: cyc["steps"].get(f"{f}_rsi", {}).get(k) for k in
                   ("status", "rounds", "decisions", "kinds", "self_improvement_rounds", "promoted", "model_dir", "reason")}
               for f in ("quasnir", "rouge", "darus")}
        med, n = self.telemetry_tps(self.name("base", g))
        if med:
            self.state.setdefault("throughput", {}).setdefault("telemetry", {})[self.name("base", g)] = {
                "median_tokens_per_s": med, "samples": n}
        tok_step = cyc["plan"]["tokens_per_step"]
        trained = {}
        for fam in TRAIN_FAMILIES:
            info = self.state["runs"].get(self.name(fam, g), {})
            if info.get("step") is not None:
                trained[fam] = {"run": self.name(fam, g), "steps": info["step"], "tokens": info["step"] * tok_step}
        du = shutil.disk_usage(self.root)
        report = {
            "cycle": g, "generated_at": now(), "generated_at_iso": iso(),
            "note": "every number below was measured by forge eval / scripts/humaneval.py / telemetry; nothing is projected",
            "corpus": {"version": self.state["corpus"]["version"],
                       "stores": {k: v["meta"] for k, v in self.state["corpus"]["stores"].items()},
                       "rouge_mix": (self.state.get("mix") or {}).get("meta")},
            "plan": cyc["plan"], "models": models,
            "restricted": {f: True for f in self.cfg["restricted_models"] if f in models},
            "evaluations": evals, "eval_settings": self.cfg["gate"]["eval"], "benchmarks": benches,
            "gates": gates, "rsi": rsi, "merge_search": cyc["steps"].get("darus_merge", {}).get("best"),
            "current": self.state["current"], "trained": trained,
            "throughput": {"planned_tokens_per_s": cyc["plan"]["tokens_per_s"], "planned_source": cyc["plan"]["tps_source"],
                           "measured_base_median_tokens_per_s": med, "samples": n},
            "wall_seconds": {k: v.get("wall_seconds") for k, v in cyc["steps"].items()},
            "disk_free_bytes": du.free, "errors": errors,
        }
        path = self.dir / f"report-g{g}.json"
        atomic_write_json(path, report)
        hist = self.dir / "history.jsonl"
        already = hist.exists() and any(json.loads(x).get("cycle") == g for x in hist.read_text().splitlines() if x.strip())
        if not already:
            line = {"cycle": g, "ts": now(), "wall_hours": (now() - cyc["started_at"]) / 3600,
                    "decisions": {f: gates[f].get("decision") for f in FAMILIES}, "current": {f: (c or {}).get("run")
                                                                                            for f, c in self.state["current"].items()},
                    "val_loss": {f: {s: r["loss"] for s, r in e.items()} for f, e in evals.items()},
                    "benchmarks": {f: {s: r.get("pass@1") for s, r in b.items()} for f, b in benches.items()},
                    "tokens": {f: t["tokens"] for f, t in trained.items()}, "report": self.rel(path)}
            with open(hist, "a") as f:
                f.write(json.dumps(line) + "\n")
        return {"report": self.rel(path), "errors": errors}

    def step_hygiene(self, g, cyc, st):
        removed = self.hygiene_all()
        for fam in ("quasnir", "rouge", "darus"):
            removed += self.prune_rsi(self.name(fam, g))
        return {"removed": len(removed), "disk_free_bytes": shutil.disk_usage(self.root).free}


def procs_or_self(pid: int) -> list:
    """[pid] while `pid` is alive (indirection keeps adopt_orphans testable)."""
    return [pid] if pid_alive(pid) else []


# --------------------------------------------------------------------------- CLI

def load_config(path: Path) -> dict:
    over = read_json(path)
    if not isinstance(over, dict):
        raise Fatal(f"config {path} not found or not JSON")
    cfg = deep_merge(DEFAULTS, strip_comments(over))
    total = sum(cfg["shares"].values())
    if abs(total - 1.0) > 1e-6:
        raise Fatal(f"shares in {path} sum to {total}, not 1")
    return cfg


def dry_run(root: Path, cfg: dict, args) -> int:
    state = read_json(root / "runs" / "longrun" / "state.json") or {}
    base_cfg = read_json(root / cfg["base_config"])
    if not isinstance(base_cfg, dict):
        print(f"base config {cfg['base_config']} not found", file=sys.stderr)
        return EX_FATAL
    tok_step = tokens_per_step(base_cfg)
    if args.tok_per_s:
        tps, src = float(args.tok_per_s), "--tok-per-s"
    elif (state.get("throughput") or {}).get("bench"):
        tps, src = float(state["throughput"]["bench"]["tokens_per_s"]), "forge bench (state.json)"
    else:
        tps, src = float(cfg["throughput"]["assumed_tokens_per_s"]), "assumed (longrun.json throughput.assumed_tokens_per_s)"
    days = args.days if args.days is not None else state.get("days", cfg["days"])
    forever = bool(args.forever or state.get("forever") or cfg["forever"])
    g0 = cfg["first_generation"]
    if state.get("cycle") is not None:
        g0 = state["cycle"] + (1 if state["cycles"].get(str(state["cycle"]), {}).get("status") == "done" else 0)
    max_cycles = args.max_cycles if args.max_cycles is not None else cfg["max_cycles"]
    plan = plan_budget(cfg, float(days), tps, src, tok_step, g0, forever, max_cycles, args.cycle_hours)
    c = cfg["corpus"]
    missing = [m for m in c["v2"].values() if not (root / m).exists()]
    plan["corpus"] = {"v2_missing": missing, "would_build": bool(missing),
                      "build_cmd": [sys.executable if x == "{python}" else x for x in c["build_cmd"]] + list(c["build_args"]),
                      "fallback_v1": bool(args.allow_v1)}
    stores = {}
    for k, m in (c["v1"] if missing and args.allow_v1 else c["v2"]).items():
        meta = read_json(root / m)
        if isinstance(meta, dict):
            stores[k] = {"meta": m, "tokens": meta.get("tokens")}
    plan["stores"] = stores
    for name, s in stores.items():
        fam = {"base": "base", "code": "quasnir"}.get(name)
        if fam and s.get("tokens"):
            s["epochs_planned"] = plan["totals"]["tokens"][fam] / s["tokens"]
    wf = cfg["wait_for"]
    plan["prerequisites"] = {"missing_paths": [p for p in wf.get("paths", []) if not (root / p).exists()],
                             "running": {pat: procs_matching(pat) for pat in wf.get("procs", []) if procs_matching(pat)}}
    plan["initial"] = state.get("current") or cfg["initial"]
    plan["model"] = base_cfg["model"]
    if args.json:
        print(json.dumps(plan, indent=2))
        return EX_OK
    t = plan["totals"]
    print("longrun plan (dry run: nothing is executed or written)")
    print(f"  throughput   {tps:,.0f} tok/s ({src}); {tok_step:,} tokens/step -> {tps * 86400 / 1e6:,.1f} M tokens/day")
    print(f"  budget       {'forever' if forever else f'{days:g} days'}; cycles of {plan['cycle_hours']:g} h"
          f" -> {len(plan['cycles'])} cycle(s), generations "
          f"{plan['cycles'][0]['generation'] if plan['cycles'] else '-'}..{plan['cycles'][-1]['generation'] if plan['cycles'] else '-'}")
    if missing:
        print(f"  corpus       v2 missing ({', '.join(missing)}) -> would run: {' '.join(plan['corpus']['build_cmd'])}"
              + (" (fallback v1 allowed)" if args.allow_v1 else " (hard stop if that fails; --allow-v1 to fall back)"))
    else:
        print("  corpus       v2 present: " + ", ".join(f"{k} {v.get('tokens', 0) / 1e6:,.1f} M tok" for k, v in stores.items()))
    pre = plan["prerequisites"]
    if pre["missing_paths"] or pre["running"]:
        print(f"  waits for    missing {pre['missing_paths']} running {pre['running']}")
    for cyc in plan["cycles"]:
        ph = cyc["phases"]
        print(f"  cycle g{cyc['generation']} ({cyc['hours']:.2f} h)")
        for fam in TRAIN_FAMILIES:
            x = ph[fam]
            print(f"    {x['run']:<14} train {x['steps']:>7,} steps {x['tokens'] / 1e6:>8.1f} M tok {x['hours']:>6.2f} h")
            if fam != "base":
                k = f"{fam}_rsi"
                print(f"    {'':<14} rsi   {ph[k]['hours']:>6.2f} h wall budget")
        print(f"    {ph['darus']['run']:<14} merge search + rsi {ph['darus']['hours']:.2f} h")
        print(f"    {'report':<14} {ph['report']['hours']:.2f} h")
    print(f"  totals       base {t['tokens']['base'] / 1e6:,.1f} M tok, quasnir {t['tokens']['quasnir'] / 1e6:,.1f} M, "
          f"rouge {t['tokens']['rouge'] / 1e6:,.1f} M; {t['hours']:.1f} h wall")
    for k, s in stores.items():
        if "epochs_planned" in s:
            print(f"  data reuse   {k}: {s['epochs_planned']:.2f} passes over {s['tokens'] / 1e6:,.1f} M tokens")
    print("  estimates assume the planned throughput; evals, RSI sampling and the report take wall time too")
    return EX_OK


def print_status(root: Path) -> int:
    s = read_json(root / "runs" / "longrun" / "state.json")
    if not s:
        print("no runs/longrun/state.json yet")
        return EX_OK
    g = s.get("cycle")
    cyc = s.get("cycles", {}).get(str(g), {}) if g is not None else {}
    print(f"status {s.get('status')} {s.get('message', '')}".rstrip())
    if s.get("budget_started_at"):
        if s.get("forever"):
            print("budget forever")
        else:
            left = s["budget_started_at"] + float(s["days"]) * 86400 - now()
            print(f"budget {s['days']} days, {left / 3600:.1f} h left")
    if g is not None:
        print(f"cycle g{g} ({cyc.get('status')})")
        for name in CYCLE_STEPS:
            st = cyc.get("steps", {}).get(name)
            if st:
                extra = st.get("decision") or st.get("reason") or ""
                print(f"  {name:<13} {st.get('status'):<8} {st.get('wall_seconds', 0) / 3600:6.2f} h {extra}")
    for fam, c in (s.get("current") or {}).items():
        print(f"current {fam:<8} {c.get('run')} {c.get('dir')}")
    return EX_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="multi-day continuous training orchestrator")
    ap.add_argument("--config", default="training/configs/longrun.json")
    ap.add_argument("--root", default=str(REPO), help="repository root (default: this checkout)")
    ap.add_argument("--days", type=float, default=None, help="total wall budget in days (default from config/state)")
    ap.add_argument("--forever", action="store_true", help="keep starting cycles; no budget end")
    ap.add_argument("--cycle-hours", type=float, default=None)
    ap.add_argument("--max-cycles", type=int, default=None, help="stop after this many completed cycles")
    ap.add_argument("--allow-v1", action="store_true", help="fall back to the v1 stores if corpus v2 is unavailable")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (step counts, hours); run nothing")
    ap.add_argument("--json", action="store_true", help="with --dry-run: machine-readable plan")
    ap.add_argument("--tok-per-s", type=float, default=None, help="override the planning throughput")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--prepare", action="store_true", help="plan the next cycle and write its base config, then exit")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for wait_for prerequisites")
    ap.add_argument("--forge", default=None, help="forge binary (default: config 'forge' or $FORGE)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    cfg_path = Path(args.config)
    cfg_path = cfg_path if cfg_path.is_absolute() else root / cfg_path
    if args.status:
        return print_status(root)
    try:
        cfg = load_config(cfg_path)
    except Fatal as e:
        print(f"[longrun] FATAL: {e}", file=sys.stderr)
        return EX_FATAL
    if args.dry_run:
        return dry_run(root, cfg, args)
    try:
        lr = LongRun(root, cfg, args)
    except Fatal as e:
        print(f"[longrun] FATAL: {e}", file=sys.stderr)
        return EX_FATAL

    def on_signal(signum, _frame):
        lr.signal_stop = True
        print(f"[longrun] signal {signum}: stopping after a checkpoint", flush=True)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        return lr.run()
    except SystemExit as e:
        return int(e.code or 0)


if __name__ == "__main__":
    sys.exit(main())
