"""RSI control and the long-run view.

Reads what `training/rsi/loop.py` writes under `runs/rsi/<model>/`:

    audit.jsonl     one line per finished round
    pending.json    a candidate that passed the gates and awaits a human
    champion.json   the current champion (with history)
    status.json     the live phase
    STOP            the loop halts at its next check while this exists

and writes only the two files the loop polls for:

    approve-<round>.json  {"approved": bool, "by": "headcenter", "ts", "candidate_sha256"}
                          only while pending.json names that round; created
                          exclusively (a decision is never overwritten) and
                          bound to the pending candidate's sha256
    STOP                  {"by": "headcenter", "ts"}

Every decision is also appended to `runs/headcenter/rsi.jsonl`.

The long-run view returns `runs/longrun/state.json` and the tail of
`runs/longrun/history.jsonl` as written by the orchestrator (never recomputed).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from .control import RUN_RE

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ROUND_FIELDS = (
    "round", "model", "family", "difficulty", "kind", "signal", "tasks", "samples", "accepted",
    "heldout_n", "heldout_pass_before", "heldout_pass_after", "val_loss_before", "val_loss_after",
    "decision", "approver", "self_improvement", "reasons", "halt", "candidate", "difficulty_next", "ts", "seconds",
)
PENDING_FIELDS = (
    "model", "round", "family", "difficulty", "kind", "candidate", "candidate_sha256", "champion",
    "heldout_pass_before", "heldout_pass_after", "val_loss_before", "val_loss_after", "gates", "created",
)
CHAMPION_FIELDS = ("model", "dir", "model_sha256", "round", "source", "approver", "ts")


class RSIError(ValueError):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def tail_jsonl(path: Path, n: int, max_bytes: int = 1 << 20) -> list[dict]:
    """The last n JSON object lines of a (possibly growing, possibly huge) file."""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    lines = data.split(b"\n")
    if size > max_bytes:
        lines = lines[1:]  # the first one is probably cut
    out: list[dict] = []
    for line in reversed(lines):
        if len(out) >= n:
            break
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    out.reverse()
    return out


def _pick(d: Any, fields: tuple) -> dict | None:
    return {k: d.get(k) for k in fields if k in d} if isinstance(d, dict) else None


def _write_exclusive(path: Path, obj: dict) -> None:
    """Create `path` with complete content, atomically, failing if it exists."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
            f.write("\n")
        try:
            os.link(tmp, path)  # atomic, never replaces an existing decision
        except FileExistsError:
            raise RSIError(f"{path.name} already exists: this round was already decided", 409) from None
    finally:
        Path(tmp).unlink(missing_ok=True)


class RSI:
    def __init__(self, runs_dir: Path, rounds: int = 30):
        self.runs_dir = Path(runs_dir)
        self.rounds = rounds
        self.audit = self.runs_dir / "headcenter" / "rsi.jsonl"

    @property
    def root(self) -> Path:
        return self.runs_dir / "rsi"

    def model_dir(self, model: Any, must_exist: bool = True) -> Path:
        if not isinstance(model, str) or not RUN_RE.match(model):
            raise RSIError("invalid model name")
        d = self.root / model
        if must_exist and not d.is_dir():
            raise RSIError(f"no RSI directory for {model}", 404)
        if d.exists() and self.root.resolve() not in d.resolve().parents:
            raise RSIError("model directory outside runs/rsi", 400)
        return d

    def model(self, name: str) -> dict:
        d = self.model_dir(name)
        rounds = [_pick(r, ROUND_FIELDS) for r in tail_jsonl(d / "audit.jsonl", self.rounds)]
        pending = _pick(load_json(d / "pending.json"), PENDING_FIELDS)
        if pending is not None:
            k = pending.get("round")
            pending["decided"] = isinstance(k, int) and (d / f"approve-{k}.json").exists()
        return {
            "name": name,
            "rounds": rounds,
            "pending": pending,
            "champion": _pick(load_json(d / "champion.json"), CHAMPION_FIELDS),
            "status": load_json(d / "status.json"),
            "stop": (d / "STOP").exists(),
        }

    def overview(self) -> dict:
        models = {}
        if self.root.is_dir():
            for d in sorted(self.root.iterdir()):
                if d.is_dir() and RUN_RE.match(d.name):
                    try:
                        models[d.name] = self.model(d.name)
                    except RSIError:
                        continue
        return {"models": models}

    def _log(self, rec: dict) -> None:
        try:
            self.audit.parent.mkdir(parents=True, exist_ok=True)
            with self.audit.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def decide(self, model: Any, round_no: Any, approved: Any, candidate_sha256: Any = None) -> dict:
        d = self.model_dir(model)
        if not isinstance(round_no, int) or isinstance(round_no, bool) or not 1 <= round_no <= 10**9:
            raise RSIError('"round" must be a positive integer')
        if not isinstance(approved, bool):
            raise RSIError('"approved" must be true or false')
        pending = load_json(d / "pending.json")
        if not isinstance(pending, dict):
            raise RSIError(f"{model} has no pending approval", 409)
        if pending.get("round") != round_no:
            raise RSIError(f"{model} awaits approval for round {pending.get('round')}, not {round_no}", 409)
        sha = pending.get("candidate_sha256")
        if not (isinstance(sha, str) and SHA_RE.match(sha)):
            raise RSIError(f"{model}: pending request has no valid candidate_sha256", 409)
        if candidate_sha256 is not None and candidate_sha256 != sha:
            raise RSIError("the candidate changed since you reviewed it; reload and decide again", 409)
        # The loop accepts the decision only for exactly this candidate.
        rec = {"approved": approved, "by": "headcenter", "ts": time.time(), "candidate_sha256": sha}
        _write_exclusive(d / f"approve-{round_no}.json", rec)
        self._log({"type": "rsi-approval", "model": model, "round": round_no, **rec})
        return rec

    def stop(self, model: Any) -> dict:
        d = self.model_dir(model)
        rec = {"by": "headcenter", "ts": time.time()}
        (d / "STOP").write_text(json.dumps(rec) + "\n")
        self._log({"type": "rsi-stop", "model": model, **rec})
        return rec

    def clear_stop(self, model: Any) -> bool:
        d = self.model_dir(model)
        existed = (d / "STOP").exists()
        (d / "STOP").unlink(missing_ok=True)
        self._log({"type": "rsi-stop-cleared", "model": model, "by": "headcenter", "ts": time.time(), "existed": existed})
        return existed


def read_longrun(runs_dir: Path, n: int = 200) -> dict:
    d = Path(runs_dir) / "longrun"
    return {
        "present": d.is_dir(),
        "state": load_json(d / "state.json"),
        "history": tail_jsonl(d / "history.jsonl", n),
        "stop": (d / "STOP").exists(),
    }
