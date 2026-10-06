"""Control channel: validated commands written to a run's FIFO (`runs/<run>.ctl`).

`scripts/train_all.sh` starts every `forge train` with its stdin opened
read-write on that FIFO, so a write succeeds exactly while the run is alive.
Only the commands of the trainer protocol (docs/DESIGN.md) pass validation,
in canonical form, as one line below PIPE_BUF (atomic).
"""

from __future__ import annotations

import errno
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

RUN_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
VARIANTS = ("scalar", "avx2-fma", "avx512-f32", "amx-bf16")
SIMPLE = ("pause", "resume", "stop", "autotune", "checkpoint")


class CommandError(ValueError):
    """The command or run name is invalid."""


class NotLive(RuntimeError):
    """The run has no live reader on its control channel."""


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate(cmd: Any) -> dict:
    """Canonical form of a trainer command, or CommandError."""
    if not isinstance(cmd, dict):
        raise CommandError("command must be a JSON object")
    name = cmd.get("cmd")
    if name in SIMPLE:
        return {"cmd": name}
    if name == "set_lr":
        v = cmd.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v < 1:
            raise CommandError("set_lr needs 0 < value < 1")
        return {"cmd": "set_lr", "value": float(v)}
    if name == "set_threads":
        v = cmd.get("value")
        if not _is_int(v) or not 1 <= v <= 1024:
            raise CommandError("set_threads needs an integer 1..=1024")
        return {"cmd": "set_threads", "value": v}
    if name == "swap_kernel":
        variant, op = cmd.get("variant"), cmd.get("op", "MATMUL")
        if variant not in VARIANTS:
            raise CommandError(f"swap_kernel variant must be one of {VARIANTS}")
        if op not in ("MATMUL", "MATMUL_BATCHED"):
            raise CommandError("swap_kernel op must be MATMUL or MATMUL_BATCHED")
        return {"cmd": "swap_kernel", "op": op, "variant": variant}
    raise CommandError(f"unknown command {name!r}")


class Controller:
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)

    def send(self, run: str, cmd: Any) -> dict:
        if not isinstance(run, str) or not RUN_RE.match(run):
            raise CommandError("invalid run name")
        canon = validate(cmd)
        path = self.runs_dir / f"{run}.ctl"
        try:
            st = os.stat(path)
        except FileNotFoundError:
            raise NotLive(f"{run} has no control channel") from None
        if not stat.S_ISFIFO(st.st_mode):
            raise CommandError(f"{path.name} is not a FIFO")
        line = (json.dumps(canon, separators=(",", ":")) + "\n").encode()
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            if e.errno == errno.ENXIO:
                raise NotLive(f"{run} is not running") from None
            raise
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
        return canon
