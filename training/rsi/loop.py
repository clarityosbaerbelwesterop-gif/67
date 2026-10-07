"""Controlled recursive self-improvement loop (verifier-gated expert iteration / STaR).

One round r for model M with champion checkpoint C:
  1. sample N tasks at the current difficulty from the TRAIN seed space
     (held-out problems and eval-set overlaps excluded);
  2. generate K candidates per task with C (`forge generate --prompts-file`,
     temperature/top-k sampling, a distinct seed per candidate, stop at eos);
  3. verify (sandbox for code, exact match for text); keep passing,
     deduplicated, decontaminated candidates;
  4. accepted < min_accepted -> "no-signal"; with --warmstart auto the round
     falls back to the generator's reference solutions and is labelled
     "supervised-warmstart" (never reported as self-improvement);
  5. build the round store (accepted docs + replay) and fine-tune with
     `forge train` (init_from C, small lr) into runs/rsi/M/rounds/;
  6. gates on the candidate vs C: held-out pass rate (fixed held-out set,
     disjoint seed space) must not drop and must rise by >= delta; validation
     loss on every --gate store must not regress by more than tol (relative);
     the round store must be contamination-free; the candidate must descend
     from C (state.json parent_sha256);
  7. decision promote / reject / pending-approval (--require-approval: write
     pending.json and poll for approve-<r>.json {"approved": bool, "by": str});
  8. curriculum: raise the difficulty when the champion's held-out pass rate at
     the current level reaches --advance-at.
Control: a STOP file in the output directory ends the loop at the next check;
--rounds, --max-steps-per-round and --max-wall-hours are hard limits; a lock
file prevents two loops on one directory; every write goes through a guard
that only allows the output directory (runs/rsi/M by default) and the round
telemetry files <telemetry-dir>/rsi-M-r<k>.jsonl. One audit line per round in
audit.jsonl; champion.json + the `champion` symlink name the current champion.

Usage:
  python3 -m training.rsi.loop --model quasnir-1 --champion runs/quasnir-1/FINAL --family code \
      --gate general=data/stores/general/train.meta.json --gate code=data/stores/code/train.meta.json \
      --replay data/stores/code/train.meta.json --rounds 3 --require-approval
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import tasks as T
from .store import EOS, FORGE, TOKENIZER, build_store, forge_tokenize, sha256_file
from .verify import Tokenizer, normalize_answer, verify

ROOT = T.ROOT
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
DECISIONS = ("promote", "reject", "pending-approval", "skip", "stopped", "error")
KINDS = ("self-generated", "supervised-warmstart", "no-signal")
AUDIT_FIELDS = ("round", "model", "family", "difficulty", "tasks", "samples", "accepted", "kind", "signal", "data_sha256",
                "heldout_pass_before", "heldout_pass_after", "val_loss_before", "val_loss_after", "decision", "approver",
                "champion", "candidate", "self_improvement", "difficulty_next", "ts")


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (Path.cwd() / p)


# --------------------------------------------------------------------------- gates (pure functions)

def evaluate_gates(*, heldout_before: float | None, heldout_after: float | None, val_before: dict, val_after: dict,
                   contamination: int, lineage_ok: bool, delta: float, tol: float, require_improvement: bool = True) -> dict:
    """All promotion gates of one candidate against the champion."""
    val = {}
    for name, b in val_before.items():
        a = val_after.get(name)
        ok_nums = all(isinstance(x, (int, float)) and math.isfinite(x) for x in (a, b)) and b > 0
        rel_change = (a - b) / b if ok_nums else None
        val[name] = {"before": b, "after": a, "rel_change": rel_change, "ok": rel_change is not None and rel_change <= tol}
    have = heldout_before is not None and heldout_after is not None
    no_drop = have and heldout_after >= heldout_before
    improved = have and heldout_after - heldout_before >= delta - 1e-12
    gates = {
        "heldout_no_drop": no_drop, "heldout_improved": improved, "heldout_delta": delta,
        "val_loss": val, "val_loss_ok": all(v["ok"] for v in val.values()), "val_tol": tol,
        "contamination": contamination, "contamination_free": contamination == 0,
        "lineage_ok": lineage_ok, "require_improvement": require_improvement,
    }
    reasons = []
    if not no_drop:
        reasons.append("held-out pass rate dropped" if have else "held-out pass rate missing")
    if require_improvement and not improved:
        reasons.append(f"held-out pass rate did not rise by >= {delta}")
    for name, v in val.items():
        if not v["ok"]:
            reasons.append(f"val loss on {name} regressed beyond {tol:.2%}" if v["rel_change"] is not None else f"val loss on {name} missing")
    if contamination:
        reasons.append(f"{contamination} contaminated training units")
    if not lineage_ok:
        reasons.append("candidate does not descend from the champion")
    gates["reasons"] = reasons
    gates["passed"] = not reasons
    return gates


def decide(gates: dict, require_approval: bool) -> str:
    if not gates["passed"]:
        return "reject"
    return "pending-approval" if require_approval else "promote"


def read_approval(path: Path, candidate_sha: str | None = None) -> dict | None:
    """None while absent; {"invalid": reason} for a malformed file; else the approval.

    Schema {"approved": bool, "by": non-empty str}; an optional "candidate_sha256"
    must match the pending candidate."""
    try:
        d = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        return {"invalid": f"unreadable: {e}"}
    if not isinstance(d, dict) or not isinstance(d.get("approved"), bool) or not isinstance(d.get("by"), str) or not d["by"].strip():
        return {"invalid": 'expected {"approved": bool, "by": non-empty str}'}
    if candidate_sha is not None and d.get("candidate_sha256") != candidate_sha:
        return {"invalid": "candidate_sha256 missing or not the pending candidate's"}
    return d


def wait_for_approval(path: Path, *, stop_check, deadline: float | None, poll: float, log=lambda *a, **k: None,
                      sleep=time.sleep, candidate_sha: str | None = None) -> tuple[str, str | None]:
    """Poll for the approval file. Returns (approved|rejected|stopped|timeout, approver)."""
    warned = None
    while True:
        if stop_check():
            return "stopped", None
        if deadline is not None and time.time() >= deadline:
            return "timeout", None
        a = read_approval(path, candidate_sha)
        if a is not None and "invalid" not in a:
            return ("approved" if a["approved"] else "rejected"), a["by"].strip()
        if a is not None and a["invalid"] != warned:
            warned = a["invalid"]
            log("approval-file-invalid", path=str(path), reason=warned)
        sleep(poll)


# --------------------------------------------------------------------------- settings and loop

@dataclass
class Settings:
    model: str
    champion: str
    family: str
    gates: dict
    replay: str | None
    rounds: int
    tasks: int = 64
    samples: int = 4
    steps_per_round: int = 40
    max_steps_per_round: int = 200
    lr: float = 1e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 5
    weight_decay: float = 0.01
    batch: int = 8
    seq: int = 256
    grad_accum: int = 1
    require_approval: bool = False
    warmstart: str = "auto"
    max_wall_hours: float = 24.0
    out: str | None = None
    telemetry_dir: str | None = None
    level: int = 0
    heldout: int = 64
    delta: float = 0.03
    tol: float = 0.01
    advance_at: float = 0.7
    require_improvement: bool = True
    min_accepted: int = 8
    max_per_task: int = 2
    temperature: float = 0.8
    top_k: int = 40
    max_new_code: int = 160
    max_new_text: int = 24
    replay_ratio: float = 0.5
    threads: int = 0
    gate_batches: int = 8
    gate_batch: int = 8
    gate_seq: int = 256
    gate_seed: int = 1234
    exec_timeout: float = 5.0
    poll_seconds: float = 10.0
    approval_timeout_hours: float | None = None
    seed: int = 0
    reset_champion: bool = False
    autotune: bool = False
    forge: str = str(FORGE)
    tokenizer: str = str(TOKENIZER)


@dataclass
class Ckpt:
    dir: Path
    sha: str
    lineage_ok: bool = True
    info: dict = field(default_factory=dict)


class Halt(Exception):
    """STOP file or wall-clock limit reached inside a round."""


class RSILoop:
    def __init__(self, s: Settings, out_stream=sys.stdout):
        if not NAME_RE.match(s.model):
            raise ValueError("--model must match [A-Za-z0-9_-]+ (it names forge runs)")
        if s.family not in T.FAMILIES:
            raise ValueError(f"--family must be one of {T.FAMILIES}")
        if s.warmstart not in ("auto", "never", "always"):
            raise ValueError("--warmstart must be auto, never or always")
        if not s.gates:
            raise ValueError("at least one --gate NAME=META is required")
        if s.level not in T.LEVELS:
            raise ValueError("--level must be in 0..9")
        if s.rounds < 0 or s.tasks < 1 or s.samples < 1 or s.max_steps_per_round < 1 or s.steps_per_round < 1:
            raise ValueError("--rounds >= 0 and --tasks, --samples, --steps-per-round, --max-steps-per-round >= 1")
        self.s = s
        self.stream = out_stream
        self.out = _abs(s.out or ROOT / "runs" / "rsi" / s.model).resolve()
        self.tele_dir = _abs(s.telemetry_dir).resolve() if s.telemetry_dir else self.out.parent.parent
        self.forge = _abs(s.forge)
        self.tokenizer = _abs(s.tokenizer)
        self.replay = _abs(s.replay) if s.replay else None
        self.gates = {k: _abs(v) for k, v in s.gates.items()}
        self.t0 = time.time()
        self.deadline = self.t0 + s.max_wall_hours * 3600.0
        self.steps = min(s.steps_per_round, s.max_steps_per_round)
        self.decon = T.default_decontaminator()
        self._heldout: dict[int, list[T.Task]] = {}
        self._heldout_all: list[T.Task] | None = None
        self._ids: dict[str, list[int]] = {}
        self._decoder = None
        self._lock = None
        self.champion: Ckpt | None = None
        self.model_cfg: dict = {}

    # ---------------------------------------------------------------- guarded writes
    def guard(self, path: Path, *, replace_link: bool = False) -> Path:
        """The path if the loop may write it, else PermissionError. Parent directories
        are resolved; a final symlink is followed too (no writing through a link to
        elsewhere) unless the link itself is being replaced (`replace_link`)."""
        q = _abs(path)
        p = q.parent.resolve() / q.name
        targets = [p] if replace_link or not p.is_symlink() else [p, p.resolve()]
        for t in targets:
            inside = t == self.out or self.out in t.parents
            tele = t.parent == self.tele_dir and re.fullmatch(rf"rsi-{re.escape(self.s.model)}-r\d+\.jsonl", t.name)
            if not (inside or tele):
                raise PermissionError(f"RSI loop may not write {t} (allowed: {self.out} and {self.tele_dir}/rsi-{self.s.model}-r<k>.jsonl)")
        return p

    def write_json(self, path: Path, obj) -> None:
        p = self.guard(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, p)

    def write_jsonl(self, path: Path, rows) -> None:
        p = self.guard(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def append_jsonl(self, path: Path, row: dict) -> None:
        p = self.guard(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(row) + "\n")

    def mkdir(self, path: Path) -> Path:
        p = self.guard(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def log(self, event: str, **kw) -> None:
        print(json.dumps({"type": "rsi", "event": event, "model": self.s.model, "ts": time.time(), **kw}), file=self.stream, flush=True)

    def status(self, phase: str, **kw) -> None:
        self.write_json(self.out / "status.json", {"model": self.s.model, "phase": phase, "pid": os.getpid(), "ts": time.time(),
                                                   "champion": rel(self.champion.dir) if self.champion else None, **kw})

    # ---------------------------------------------------------------- control
    def stop_requested(self) -> bool:
        return (self.out / "STOP").exists()

    def out_of_time(self) -> bool:
        return time.time() >= self.deadline

    def check(self, where: str) -> None:
        if self.stop_requested():
            raise Halt(f"STOP file present ({where})")
        if self.out_of_time():
            raise Halt(f"--max-wall-hours {self.s.max_wall_hours} reached ({where})")

    def remaining(self) -> float:
        return max(1.0, self.deadline - time.time())

    def acquire_lock(self) -> None:
        self.mkdir(self.out)
        f = open(self.guard(self.out / "loop.lock"), "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            raise RuntimeError(f"another RSI loop holds {self.out / 'loop.lock'}")
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
        self._lock = f

    def release_lock(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    # ---------------------------------------------------------------- champion
    def _ckpt(self, d: Path) -> Ckpt:
        d = _abs(d).resolve()
        state_p = d / "state.json"
        if not state_p.exists() or not (d / "model.safetensors").exists():
            raise FileNotFoundError(f"{d} is not a forge checkpoint (model.safetensors + state.json)")
        st = json.loads(state_p.read_text())
        return Ckpt(dir=d, sha=st["model_sha256"], info={"step": st.get("step"), "parent_sha256": st.get("parent_sha256"),
                                                          "model": st["config"]["model"]})

    def model_file_sha(self, d: Path) -> str:
        """sha256 of model.safetensors (what forge records as model_sha256)."""
        return sha256_file(Path(d) / "model.safetensors")

    def load_champion(self) -> None:
        cj = self.out / "champion.json"
        if cj.exists() and not self.s.reset_champion:
            c = json.loads(cj.read_text())
            ck = self._ckpt(ROOT / c["dir"] if not Path(c["dir"]).is_absolute() else Path(c["dir"]))
            if ck.sha != c["model_sha256"]:
                raise RuntimeError(f"champion.json sha {c['model_sha256']} != {ck.sha} of {ck.dir}")
            self.champion = ck
            self.log("champion-resumed", dir=rel(ck.dir), sha256=ck.sha, round=c.get("round"))
        else:
            ck = self._ckpt(_abs(self.s.champion))
            self.champion = ck
            self._write_champion(ck, round_no=0, source="initial", approver=None)
            self.log("champion-initial", dir=rel(ck.dir), sha256=ck.sha)
        if self.model_file_sha(self.champion.dir) != self.champion.sha:
            raise RuntimeError(f"{self.champion.dir}/model.safetensors does not match model_sha256 {self.champion.sha}")
        self.model_cfg = self.champion.info["model"]

    def _write_champion(self, ck: Ckpt, *, round_no: int, source: str, approver: str | None) -> None:
        prev = None
        cj = self.out / "champion.json"
        if cj.exists():
            prev = json.loads(cj.read_text())
        history = (prev or {}).get("history", [])
        if prev:
            history = history + [{k: prev.get(k) for k in ("dir", "model_sha256", "round", "source", "approver", "ts")}]
        self.write_json(cj, {"model": self.s.model, "dir": rel(ck.dir), "model_sha256": ck.sha, "round": round_no, "source": source,
                             "approver": approver, "ts": time.time(), "history": history})
        link = self.guard(self.out / "champion", replace_link=True)
        tmp = self.guard(self.out / "champion.tmp-link", replace_link=True)
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(ck.dir, tmp)
        os.replace(tmp, link)

    # ---------------------------------------------------------------- tasks and generation
    @property
    def ctx(self) -> int:
        return int(self.model_cfg["max_seq_len"])

    def max_new(self, task: T.Task) -> int:
        return self.s.max_new_code if task.family == "code" else self.s.max_new_text

    def prompt_ids(self, tasks: list[T.Task]) -> None:
        todo = [t for t in tasks if t.task_id not in self._ids]
        if todo:
            ids = forge_tokenize([t.prompt for t in todo], self.mkdir(self.out / "tmp"), forge=self.forge, tokenizer=self.tokenizer)
            self._ids.update({t.task_id: i for t, i in zip(todo, ids)})

    def fit(self, tasks: list[T.Task]) -> tuple[list[T.Task], int]:
        """Tasks whose prompt plus generation budget fits the model context."""
        self.prompt_ids(tasks)
        kept = [t for t in tasks if len(self._ids[t.task_id]) + self.max_new(t) <= self.ctx]
        return kept, len(tasks) - len(kept)

    def heldout_set(self, level: int) -> list[T.Task]:
        if level not in self._heldout:
            ts, _ = T.heldout_tasks(self.s.family, level, self.s.heldout, self.decon)
            self._heldout[level], _ = self.fit(ts)
        return self._heldout[level]

    def heldout_all(self) -> list[T.Task]:
        if self._heldout_all is None:
            self._heldout_all = [t for lv in T.LEVELS for t in T.heldout_tasks(self.s.family, lv, self.s.heldout, self.decon)[0]]
        return self._heldout_all

    def decode(self, ids: list[int]) -> str:
        if self._decoder is None:
            tok = Tokenizer(self.tokenizer)
            self._decoder = (tok, 256 + len(tok.merges))
        tok, limit = self._decoder
        return tok.decode([i for i in ids if i < limit])

    def generate(self, ckpt: Ckpt, tasks: list[T.Task], k: int, *, temperature: float, top_k: int, seed: int,
                 workdir: Path) -> list[tuple[T.Task, int, str]]:
        """k completions per task; candidate i of the call uses seed + i (forge: seed + line index)."""
        self.prompt_ids(tasks)
        out = []
        offset = 0
        for fam in ("code", "text"):
            group = [t for t in tasks if t.family == fam]
            if not group:
                continue
            lines = [(t, j) for t in group for j in range(k)]
            pf = self.guard(workdir / f"prompts-{fam}.jsonl")
            self.write_jsonl(pf, [self._ids[t.task_id] for t, _ in lines])
            cmd = [str(self.forge), "generate", "--ckpt", str(ckpt.dir), "--prompts-file", str(pf), "--max-new", str(self.max_new(group[0])),
                   "--temperature", str(temperature), "--top-k", str(top_k), "--seed", str(seed + offset), "--stop-id", str(EOS)]
            if self.s.threads:
                cmd += ["--threads", str(self.s.threads)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.remaining(), cwd=ROOT, stdin=subprocess.DEVNULL)
            if res.returncode != 0:
                raise RuntimeError(f"forge generate failed: {res.stderr.strip()[-400:]}")
            recs = {}
            for line in res.stdout.splitlines():
                if line.startswith("{"):
                    r = json.loads(line)
                    if r.get("type") == "generate":
                        recs[r["index"]] = r["ids"]
            if len(recs) != len(lines):
                raise RuntimeError(f"forge generate returned {len(recs)} of {len(lines)} completions")
            out.extend((t, j, self.decode(recs[i])) for i, (t, j) in enumerate(lines))
            offset += len(lines)
        return out

    # ---------------------------------------------------------------- measurements (cached per checkpoint sha)
    def _metrics(self) -> dict:
        p = self.out / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _cache(self, sha: str, kind: str, key: str, value) -> None:
        m = self._metrics()
        m.setdefault(sha, {}).setdefault(kind, {})[key] = value
        self.write_json(self.out / "metrics.json", m)

    def heldout_pass(self, ck: Ckpt, level: int) -> dict:
        key = f"{self.s.family}|L{level}|n{self.s.heldout}|new{self.s.max_new_code},{self.s.max_new_text}|ctx{self.ctx}|t{self.s.exec_timeout}"
        hit = self._metrics().get(ck.sha, {}).get("heldout", {}).get(key)
        if hit is not None:
            return hit
        tasks = self.heldout_set(level)
        wd = self.mkdir(self.out / "heldout" / f"{ck.sha[:16]}-L{level}")
        gens = self.generate(ck, tasks, 1, temperature=0.0, top_k=1, seed=0, workdir=wd) if tasks else []
        rows, passed = [], 0
        for t, _, text in gens:
            v = verify(t, text, self.s.exec_timeout)
            passed += v["status"] == "pass"
            rows.append({"task_id": t.task_id, "status": v["status"], "completion": text[:2000]})
        self.write_jsonl(wd / "results.jsonl", rows)
        res = {"pass_rate": passed / len(tasks) if tasks else 0.0, "passed": passed, "n": len(tasks), "level": level,
               "decoding": "greedy", "ckpt": rel(ck.dir)}
        self._cache(ck.sha, "heldout", key, res)
        self.log("heldout", sha256=ck.sha[:16], **{k: res[k] for k in ("level", "passed", "n", "pass_rate")})
        return res

    def val_losses(self, ck: Ckpt) -> dict:
        seq = min(self.s.gate_seq, self.ctx)
        out = {}
        for name, meta in self.gates.items():
            key = f"{name}|{rel(meta)}|b{self.s.gate_batch}|s{seq}|n{self.s.gate_batches}|seed{self.s.gate_seed}"
            hit = self._metrics().get(ck.sha, {}).get("val", {}).get(key)
            if hit is None:
                cmd = [str(self.forge), "eval", "--ckpt", str(ck.dir), "--data", str(meta), "--split", "val", "--batch", str(self.s.gate_batch),
                       "--seq", str(seq), "--batches", str(self.s.gate_batches), "--seed", str(self.s.gate_seed)]
                if self.s.threads:
                    cmd += ["--threads", str(self.s.threads)]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.remaining(), cwd=ROOT, stdin=subprocess.DEVNULL)
                lines = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
                if res.returncode != 0 or not lines:
                    raise RuntimeError(f"forge eval failed on {meta}: {res.stderr.strip()[-400:]}")
                hit = json.loads(lines[-1])["loss"]
                self._cache(ck.sha, "val", key, hit)
            out[name] = hit
        return out

    # ---------------------------------------------------------------- training
    def train(self, k: int, meta: Path, champ: Ckpt, kind: str, rd: Path) -> Ckpt:
        run = f"rsi-{self.s.model}-r{k}"
        steps = self.steps
        cfg = {
            "run": run, "model": self.model_cfg, "data": str(meta), "batch": self.s.batch, "seq_len": min(self.s.seq, self.ctx),
            "grad_accum": self.s.grad_accum, "max_steps": steps, "lr": self.s.lr, "min_lr": self.s.lr * self.s.min_lr_ratio,
            "warmup_steps": min(self.s.warmup_steps, steps // 2), "weight_decay": self.s.weight_decay, "grad_clip": 1.0,
            "seed": int(hashlib.sha256(f"{self.s.model}|train|{k}|{self.s.seed}".encode()).hexdigest()[:8], 16),
            "init_from": str(champ.dir), "out_dir": str(self.mkdir(self.out / "rounds")), "log_every": max(1, steps // 20),
            "eval_every": 0, "eval_batches": 4, "ckpt_every": 0, "threads": self.s.threads, "autotune": self.s.autotune,
        }
        run_dir = self.out / "rounds" / run
        if run_dir.exists():  # an aborted attempt at this round, possibly the champion's own directory: never overwrite it
            aside = self.guard(self.out / "rounds" / f"{run}.aborted-{time.time_ns()}")
            os.replace(self.guard(run_dir), aside)
            self.log("round-dir-moved-aside", round=k, moved_to=rel(aside))
        cfg_path = self.out / "configs" / f"r{k}.json"
        self.write_json(cfg_path, cfg)
        tele = self.guard(self.tele_dir / f"{run}.jsonl")
        err_path = self.guard(rd / "train.stderr.log")
        self.append_jsonl(tele, {"type": "event", "level": "info", "step": 0, "ts": time.time(),
                                 "msg": f"RSI {self.s.model} round {k} ({kind}): {steps} steps from champion {champ.sha[:12]}"})
        done = None
        with open(err_path, "w") as err, open(tele, "a") as tf:
            proc = subprocess.Popen([str(self.forge), "train", "--config", str(cfg_path)], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=err, text=True, cwd=ROOT)
            timer = threading.Timer(self.remaining(), proc.kill)  # our own child only
            timer.start()
            try:
                for line in proc.stdout:
                    tf.write(line)
                    tf.flush()
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("type") == "done":
                        done = r
            finally:
                timer.cancel()
                rc = proc.wait()
        if rc != 0 or done is None:
            tail = err_path.read_text()[-400:] if err_path.exists() else ""
            raise RuntimeError(f"forge train for round {k} failed (rc {rc}): {tail}")
        latest = json.loads((self.out / "rounds" / run / "latest.json").read_text())
        ck = self._ckpt(Path(latest["dir"]))
        ck.lineage_ok = ck.info.get("parent_sha256") == champ.sha
        ck.info["done"] = done
        return ck

    # ---------------------------------------------------------------- one round
    def _seed(self, k: int, what: str) -> int:
        return int(hashlib.sha256(f"{self.s.model}|{what}|{k}|{self.s.seed}".encode()).hexdigest()[:7], 16)

    def run_round(self, k: int, level: int) -> dict:
        s = self.s
        rec = {f: None for f in AUDIT_FIELDS}
        rec.update(round=k, model=s.model, family=s.family, difficulty=level, tasks=0, samples=0, accepted=0, kind=None, signal=None,
                   decision=None, approver=None, champion=rel(self.champion.dir), champion_sha256=self.champion.sha,
                   self_improvement=False, difficulty_next=level, ts_start=time.time(), reasons=[])
        rd = self.mkdir(self.out / "data" / f"r{k}")
        champ = self.champion
        try:
            self.status("heldout-before", round=k, level=level)
            hb = self.heldout_pass(champ, level)
            rec["heldout_pass_before"] = hb["pass_rate"]
            rec["heldout_n"] = hb["n"]
            self.check("after held-out baseline")

            tasks, tstats = T.sample_tasks(s.family, level, s.tasks, f"{s.model}|r{k}|{s.seed}",
                                           exclude={t.key for t in self.heldout_all()}, decon=self.decon)
            tasks, too_long = self.fit(tasks)
            tstats["dropped_too_long"] = too_long
            rec["tasks"], rec["task_stats"] = len(tasks), tstats
            self.write_jsonl(rd / "tasks.jsonl", [t.to_json() for t in tasks])

            accepted, statuses = [], Counter()
            if s.warmstart != "always" and tasks:
                self.status("sampling", round=k, level=level, tasks=len(tasks), samples=s.samples)
                gens = self.generate(champ, tasks, s.samples, temperature=s.temperature, top_k=s.top_k,
                                     seed=self._seed(k, "sample"), workdir=rd)
                rec["samples"] = len(gens)
                self.check("after sampling")
                self.status("verifying", round=k, candidates=len(gens))
                per_task: dict[str, set] = {}
                rows = []
                for t, j, text in gens:
                    v = verify(t, text, s.exec_timeout)
                    statuses[v["status"]] += 1
                    note = None
                    if v["status"] == "pass":
                        norm = normalize_answer(v["cut"]) if t.family == "text" else "\n".join(x.rstrip() for x in v["cut"].splitlines())
                        doc = t.prompt + v["cut"]
                        seen = per_task.setdefault(t.task_id, set())
                        if norm in seen:
                            note = "duplicate"
                        elif len(seen) >= s.max_per_task:
                            note = "over max-per-task"
                        elif self.decon.contaminated(doc):
                            note = "contaminated"
                        else:
                            seen.add(norm)
                            accepted.append({"text": doc, "source": "self-generated", "task_id": t.task_id, "level": t.level, "sample": j})
                            note = "accepted"
                    rows.append({"task_id": t.task_id, "sample": j, "status": v["status"], "note": note, "completion": text[:2000],
                                 **({"reason": v["reason"]} if v.get("reason") else {})})
                self.write_jsonl(rd / "candidates.jsonl", rows)
            rec["accepted"] = len(accepted)
            rec["verify_status"] = dict(statuses)
            rec["sample_pass_rate"] = statuses["pass"] / rec["samples"] if rec["samples"] else None
            self.check("after verification")

            if s.warmstart == "always":
                kind, signal = "supervised-warmstart", "not-sampled"
            elif len(accepted) >= s.min_accepted:
                kind, signal = "self-generated", "ok"
            elif s.warmstart == "auto":
                kind, signal = "supervised-warmstart", "no-signal"
            else:
                kind, signal = "no-signal", "no-signal"
            rec["kind"], rec["signal"] = kind, signal
            if kind == "no-signal":
                rec["decision"] = "skip"
                rec["reasons"].append(f"accepted {len(accepted)} < --min-accepted {s.min_accepted} and --warmstart never")
                return self._finish(rec, level, hb["pass_rate"])

            docs = list(accepted)
            if kind == "supervised-warmstart":
                docs += [{"text": t.prompt + t.reference, "source": "supervised-reference", "task_id": t.task_id, "level": t.level}
                         for t in tasks]
            self.status("building-store", round=k, kind=kind, documents=len(docs))
            seq = min(s.seq, self.ctx)
            manifest = build_store(self.mkdir(rd / "store"), docs, name=f"rsi-{s.model}-r{k}", replay_meta=self.replay,
                                   replay_ratio=s.replay_ratio if self.replay else 0.0, seed=self._seed(k, "store"),
                                   min_split_tokens=2 * (seq + 1), decon=self.decon, forbidden=[t.prompt for t in self.heldout_all()],
                                   tokenizer=self.tokenizer, forge=self.forge,
                                   min_train_tokens=self.steps * s.batch * s.grad_accum * (seq + 1))
            rec["data"] = rel(rd / "store")
            rec["data_sha256"] = {k2: manifest["sha256"][k2] for k2 in ("train.bin", "train.val.bin")}
            rec["data_tokens"] = {"train": manifest["tokens"], "val": manifest["val_tokens"], "by_source": manifest["token_mix"]}
            rec["documents_by_source"] = {src: sum(1 for d in docs if d["source"] == src) for src in ("self-generated", "supervised-reference")}
            self.check("after building the store")

            self.status("training", round=k, steps=self.steps)
            cand = self.train(k, rd / "store" / "train.meta.json", champ, kind, rd)
            rec["candidate"], rec["candidate_sha256"], rec["steps"] = rel(cand.dir), cand.sha, self.steps
            rec["train_done"] = cand.info.get("done")
            self.check("after training")

            self.status("gating", round=k)
            ha = self.heldout_pass(cand, level)
            vb, va = self.val_losses(champ), self.val_losses(cand)
            rec.update(heldout_pass_after=ha["pass_rate"], val_loss_before=vb, val_loss_after=va)
            gates = evaluate_gates(heldout_before=hb["pass_rate"], heldout_after=ha["pass_rate"], val_before=vb, val_after=va,
                                   contamination=manifest["decontamination"]["remaining_overlaps"], lineage_ok=cand.lineage_ok,
                                   delta=s.delta, tol=s.tol, require_improvement=s.require_improvement)
            rec["gates"] = gates
            rec["reasons"] += gates["reasons"]
            self.check("after gating")
            decision = decide(gates, s.require_approval)
            if decision == "pending-approval":
                decision, approver = self._await_approval(k, rec, cand)
                rec["approver"] = approver
            if decision == "promote" and self.stop_requested():
                raise Halt("STOP file present (before promotion)")
            if decision == "promote" and self.model_file_sha(cand.dir) != cand.sha:
                decision = "reject"
                rec["reasons"].append("candidate model.safetensors changed after gating (sha256 mismatch)")
            rec["decision"] = decision
            rate_now = hb["pass_rate"]
            if decision == "promote":
                self.champion = cand
                self._write_champion(cand, round_no=k, source=kind, approver=rec["approver"])
                rate_now = ha["pass_rate"]
                self.log("promoted", round=k, dir=rel(cand.dir), sha256=cand.sha)
            rec["self_improvement"] = kind == "self-generated" and decision == "promote" and gates["heldout_improved"]
            self.append_jsonl(self.tele_dir / f"rsi-{s.model}-r{k}.jsonl", {
                "type": "event", "level": "info" if decision == "promote" else "warn", "step": self.steps, "ts": time.time(),
                "msg": f"RSI {s.model} round {k} ({kind}): {decision}; held-out {hb['pass_rate']:.3f} -> {ha['pass_rate']:.3f}"
                       + (f"; {'; '.join(gates['reasons'])}" if gates["reasons"] else "")})
            return self._finish(rec, level, rate_now)
        except Halt as h:
            rec["decision"] = rec["decision"] or "stopped"
            rec["reasons"].append(str(h))
            rec["halt"] = str(h)
            return self._finish(rec, level, None)
        except Exception as e:  # noqa: BLE001 - recorded in the audit, the loop halts for a human to look
            if isinstance(e, subprocess.TimeoutExpired) or self.out_of_time():
                rec["decision"] = rec["decision"] or "stopped"
                rec["halt"] = f"--max-wall-hours {self.s.max_wall_hours} reached ({type(e).__name__})"
            else:
                rec["decision"] = "error"
                rec["halt"] = "error"
            rec["reasons"].append(f"{type(e).__name__}: {e}")
            return self._finish(rec, level, None)

    def _await_approval(self, k: int, rec: dict, cand: Ckpt) -> tuple[str, str | None]:
        approve = self.guard(self.out / f"approve-{k}.json")
        if approve.exists():  # left over from an aborted attempt at this round: never let it approve a new candidate
            stale = self.guard(self.out / f"approve-{k}.stale-{time.time_ns()}.json")
            os.replace(approve, stale)
            self.log("approval-file-stale", round=k, moved_to=rel(stale))
        pending = {"model": self.s.model, "round": k, "family": self.s.family, "difficulty": rec["difficulty"], "kind": rec["kind"],
                   "candidate": rel(cand.dir), "candidate_sha256": cand.sha, "champion": rel(self.champion.dir),
                   "champion_sha256": self.champion.sha, "heldout_pass_before": rec["heldout_pass_before"],
                   "heldout_pass_after": rec["heldout_pass_after"], "val_loss_before": rec["val_loss_before"],
                   "val_loss_after": rec["val_loss_after"], "gates": rec["gates"], "data_sha256": rec["data_sha256"],
                   "approve_file": rel(approve), "approve_schema": {"approved": "bool", "by": "str"}, "created": time.time()}
        self.write_json(self.out / "pending.json", pending)
        self.status("awaiting-approval", round=k, approve_file=rel(approve))
        self.log("pending-approval", round=k, approve_file=rel(approve))
        deadline = self.deadline
        if self.s.approval_timeout_hours is not None:
            deadline = min(deadline, time.time() + self.s.approval_timeout_hours * 3600.0)
        st, by = wait_for_approval(approve, stop_check=self.stop_requested, deadline=deadline, poll=self.s.poll_seconds, log=self.log,
                                   candidate_sha=cand.sha)
        if st in ("approved", "rejected"):
            self.guard(self.out / "pending.json").unlink(missing_ok=True)
            if st == "rejected":
                rec["reasons"].append(f"rejected by {by}")
            return ("promote" if st == "approved" else "reject"), by
        if st == "timeout" and not self.out_of_time():
            self.guard(self.out / "pending.json").unlink(missing_ok=True)
            rec["reasons"].append("no approval before --approval-timeout-hours")
            return "reject", None
        rec["halt"] = "STOP file present while awaiting approval" if st == "stopped" else "--max-wall-hours reached while awaiting approval"
        rec["reasons"].append(rec["halt"])
        # Withdraw the request so nobody can approve a candidate that no loop is waiting for.
        pend = self.guard(self.out / "pending.json")
        if pend.exists():
            os.replace(pend, self.guard(self.out / f"pending-r{k}.withdrawn-{time.time_ns()}.json"))
        return "pending-approval", None

    def _finish(self, rec: dict, level: int, rate_now: float | None) -> dict:
        if rate_now is not None and rate_now >= self.s.advance_at and level < max(T.LEVELS):
            rec["difficulty_next"] = level + 1
        else:
            rec["difficulty_next"] = level
        rec["champion"] = rel(self.champion.dir)
        rec["champion_sha256_after"] = self.champion.sha
        rec["ts"] = time.time()
        rec["seconds"] = round(rec["ts"] - rec["ts_start"], 1)
        return rec

    # ---------------------------------------------------------------- main
    def audit(self) -> list[dict]:
        p = self.out / "audit.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

    def run(self) -> int:
        self.acquire_lock()
        self.load_champion()
        past = self.audit()
        last = max((r["round"] for r in past), default=0)
        cj = self.out / "champion.json"
        if cj.exists():
            last = max(last, int(json.loads(cj.read_text()).get("round") or 0))
        pend = self.out / "pending.json"
        if pend.exists():  # a previous loop halted while awaiting approval; nobody waits for that request any more
            old = json.loads(pend.read_text())
            moved = self.guard(self.out / f"pending-r{old.get('round')}.expired-{time.time_ns()}.json")
            os.replace(self.guard(pend), moved)
            self.log("pending-expired", round=old.get("round"), moved_to=rel(moved))
        level = past[-1]["difficulty_next"] if past else self.s.level
        self.log("start", out=rel(self.out), champion=rel(self.champion.dir), level=level, first_round=last + 1, rounds=self.s.rounds,
                 steps_per_round=self.steps, require_approval=self.s.require_approval, warmstart=self.s.warmstart)
        exit_code = 0
        try:
            if not past and not self.stop_requested():  # curriculum for the initial champion
                while level < max(T.LEVELS) and self.heldout_pass(self.champion, level)["pass_rate"] >= self.s.advance_at:
                    level += 1
            for i in range(self.s.rounds):
                k = last + 1 + i
                if self.stop_requested():
                    self.log("stop", reason="STOP file present", before_round=k)
                    break
                if self.out_of_time():
                    self.log("stop", reason="max wall time reached", before_round=k)
                    break
                self.log("round-start", round=k, level=level)
                rec = self.run_round(k, level)
                self.append_jsonl(self.out / "audit.jsonl", rec)
                self.log("round-end", round=k, decision=rec["decision"], kind=rec["kind"], accepted=rec["accepted"],
                         heldout_before=rec["heldout_pass_before"], heldout_after=rec["heldout_pass_after"], reasons=rec["reasons"])
                level = rec["difficulty_next"]
                if rec.get("halt"):
                    exit_code = 1 if rec["decision"] == "error" else 0
                    break
        finally:
            self.status("idle", level=level, last_round=max((r["round"] for r in self.audit()), default=0))
            self.release_lock()
        return exit_code


def parse_args(argv=None) -> Settings:
    d = Settings(model="", champion="", family="code", gates={}, replay=None, rounds=0)
    ap = argparse.ArgumentParser(prog="python3 -m training.rsi.loop", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", required=True, help="model name (runs/rsi/<model>)")
    ap.add_argument("--champion", required=True, help="initial champion checkpoint dir (champion.json in --out wins on restart)")
    ap.add_argument("--family", required=True, choices=T.FAMILIES)
    ap.add_argument("--gate", action="append", default=[], metavar="NAME=META", help="gate store (val loss must not regress), repeatable")
    ap.add_argument("--replay", default=None, help="store meta for replay mixing")
    ap.add_argument("--replay-ratio", type=float, default=d.replay_ratio, help="replay share of training tokens")
    ap.add_argument("--rounds", type=int, required=True, help="hard limit: rounds in this invocation")
    ap.add_argument("--tasks", type=int, default=d.tasks, help="tasks per round (N)")
    ap.add_argument("--samples", type=int, default=d.samples, help="candidates per task (K)")
    ap.add_argument("--steps-per-round", type=int, default=d.steps_per_round)
    ap.add_argument("--max-steps-per-round", type=int, default=d.max_steps_per_round, help="hard limit on optimizer steps per round")
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--min-lr-ratio", type=float, default=d.min_lr_ratio)
    ap.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--batch", type=int, default=d.batch)
    ap.add_argument("--seq", type=int, default=d.seq)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--require-approval", action="store_true", help="a human must approve every promotion (approve-<round>.json)")
    ap.add_argument("--warmstart", choices=("auto", "never", "always"), default=d.warmstart)
    ap.add_argument("--max-wall-hours", type=float, default=d.max_wall_hours, help="hard limit on wall time")
    ap.add_argument("--out", default=None, help="output dir (default runs/rsi/<model>)")
    ap.add_argument("--telemetry-dir", default=None, help="where rsi-<model>-r<k>.jsonl go (default: two levels above --out, i.e. runs/)")
    ap.add_argument("--level", type=int, default=d.level, help="starting difficulty 0..9 (resumed from the audit)")
    ap.add_argument("--heldout", type=int, default=d.heldout, help="held-out tasks per level")
    ap.add_argument("--delta", type=float, default=d.delta, help="held-out pass-rate rise that counts as improvement")
    ap.add_argument("--tol", type=float, default=d.tol, help="max relative val-loss regression per gate store")
    ap.add_argument("--advance-at", type=float, default=d.advance_at, help="held-out pass rate that raises the difficulty")
    ap.add_argument("--promote-on", choices=("improvement", "no-regression"), default="improvement",
                    help="'no-regression' drops the delta requirement (still recorded in the audit)")
    ap.add_argument("--min-accepted", type=int, default=d.min_accepted)
    ap.add_argument("--max-per-task", type=int, default=d.max_per_task, help="distinct accepted solutions kept per task")
    ap.add_argument("--temperature", type=float, default=d.temperature)
    ap.add_argument("--top-k", type=int, default=d.top_k)
    ap.add_argument("--max-new-code", type=int, default=d.max_new_code)
    ap.add_argument("--max-new-text", type=int, default=d.max_new_text)
    ap.add_argument("--threads", type=int, default=d.threads, help="forge threads (0 = forge default)")
    ap.add_argument("--gate-batches", type=int, default=d.gate_batches)
    ap.add_argument("--gate-batch", type=int, default=d.gate_batch)
    ap.add_argument("--gate-seq", type=int, default=d.gate_seq)
    ap.add_argument("--gate-seed", type=int, default=d.gate_seed)
    ap.add_argument("--exec-timeout", type=float, default=d.exec_timeout, help="sandbox wall timeout per program (s)")
    ap.add_argument("--poll-seconds", type=float, default=d.poll_seconds)
    ap.add_argument("--approval-timeout-hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--reset-champion", action="store_true", help="ignore champion.json in --out and start from --champion")
    ap.add_argument("--autotune", action="store_true", help="run the forge JIT autotuner before each round's training")
    ap.add_argument("--forge", default=d.forge)
    ap.add_argument("--tokenizer", default=d.tokenizer)
    a = ap.parse_args(argv)
    gates = {}
    for g in a.gate:
        name, sep, meta = g.partition("=")
        if not sep or not name or not meta:
            ap.error(f"--gate expects NAME=META, got {g!r}")
        gates[name] = meta
    if not gates:
        ap.error("at least one --gate NAME=META is required")
    vals = {f.name: getattr(a, f.name) for f in fields(Settings) if hasattr(a, f.name)}
    vals.update(gates=gates, require_improvement=a.promote_on == "improvement")
    return Settings(**vals)


def main(argv=None) -> int:
    s = parse_args(argv)
    loop = RSILoop(s)
    return loop.run()


if __name__ == "__main__":
    sys.exit(main())
