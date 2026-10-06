"""Telemetry ingestion: incremental tailing of `runs/*.jsonl` and per-run state.

Every line `forge train` / `forge merge` writes is one JSON record (see
docs/DESIGN.md, "Telemetry"). The tailer reads only complete new lines, so a
run that is still writing is safe to follow; a truncated file restarts.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Same reference as `forge bench`: dense BF16 per B300 GPU.
B300_DENSE_BF16_TFLOPS = 2500.0

HISTORY_COLUMNS = ["step", "loss", "tokens_per_s", "tflops", "lr", "grad_norm", "step_ms", "ts"]


def num(x: Any) -> float | None:
    """Finite float or None (serde_json writes NaN/inf as null)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


class Tail:
    """Complete new lines of a growing file; a shrunk file is re-read from the start."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.partial = b""

    def read(self) -> list[str]:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self.offset:
            self.offset, self.partial = 0, b""
        if size == self.offset:
            return []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        self.offset += len(data)
        *lines, self.partial = (self.partial + data).split(b"\n")
        return [ln.decode("utf-8", "replace") for ln in lines if ln.strip()]


def parse(line: str) -> dict | None:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    return rec if isinstance(rec, dict) else None


@dataclass
class RunState:
    name: str
    max_steps: int | None = None
    status: str = "idle"  # idle | running | paused | done
    step: int = 0
    last: dict | None = None
    last_ts: float | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=20000))
    evals: list = field(default_factory=list)
    events: deque = field(default_factory=lambda: deque(maxlen=200))
    checkpoints: deque = field(default_factory=lambda: deque(maxlen=50))
    autotune: dict = field(default_factory=dict)
    intervals: deque = field(default_factory=lambda: deque(maxlen=50))

    def ingest(self, rec: dict) -> None:
        kind = rec.get("type")
        ts = num(rec.get("ts"))
        if kind == "telemetry":
            if ts is not None and self.last_ts is not None and ts > self.last_ts:
                self.intervals.append(ts - self.last_ts)
            self.last = rec
            self.step = int(rec.get("step", self.step))
            self.status = "paused" if rec.get("paused") else "running"
            self.history.append(
                [self.step]
                + [num(rec.get(k)) for k in ("loss", "tokens_per_s", "tflops", "lr", "grad_norm", "step_ms")]
                + [ts]
            )
        elif kind == "eval":
            self.evals.append([int(rec.get("step", 0)), num(rec.get("val_loss")), num(rec.get("val_acc"))])
        elif kind == "autotune":
            self.autotune[json.dumps([rec.get("shape"), rec.get("trans")])] = rec
        elif kind == "checkpoint":
            self.checkpoints.append(rec)
        elif kind == "done":
            self.status = "done"
            self.step = int(rec.get("step", self.step))
        else:
            self.events.append(rec)
        if ts is not None:
            self.last_ts = ts if self.last_ts is None else max(self.last_ts, ts)

    def median_interval(self) -> float | None:
        return statistics.median(self.intervals) if self.intervals else None

    def eta_s(self) -> float | None:
        if self.max_steps is None or self.status == "done":
            return None
        ms = [h[6] for h in list(self.history)[-20:] if h[6] is not None]
        if not ms:
            return None
        return max(0, self.max_steps - self.step) * statistics.median(ms) / 1e3

    def summary(self) -> dict:
        tflops = num(self.last.get("tflops")) if self.last else None
        return {
            "name": self.name,
            "status": self.status,
            "step": self.step,
            "max_steps": self.max_steps,
            "eta_s": self.eta_s(),
            "last": self.last,
            "last_ts": self.last_ts,
            "b300_fraction": None if tflops is None else tflops / B300_DENSE_BF16_TFLOPS,
            "history_columns": HISTORY_COLUMNS,
            "history": list(self.history),
            "evals": self.evals,
            "events": list(self.events),
            "checkpoints": list(self.checkpoints),
            "autotune": list(self.autotune.values()),
        }


def load_max_steps(configs_dir: Path | None, run: str) -> int | None:
    if configs_dir is None:
        return None
    try:
        cfg = json.loads((configs_dir / f"{run}.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    v = cfg.get("max_steps") if isinstance(cfg, dict) else None
    return v if isinstance(v, int) and not isinstance(v, bool) else None
