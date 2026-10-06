"""Verification of candidate solutions.

Code: the approach of scripts/humaneval.py, whose `limits()` is imported and
used unchanged (RLIMIT_CPU 10 s, RLIMIT_AS 1 GiB, RLIMIT_FSIZE 1 MiB,
RLIMIT_NPROC 64). Each program runs in a fresh `python -I` subprocess with an
empty environment, inside its own temporary directory and process group, with
a wall-clock timeout (the whole group is killed on expiry). stdout/stderr go to
files in that directory, so the FSIZE limit also bounds output floods.

On top of that, two guards against reward hacking:
  * a random per-run marker is printed after the hidden asserts and must appear
    in stdout, so `exit(0)` before the tests cannot pass;
  * a static filter rejects completions that import outside a small allow-list
    or touch process/file/introspection facilities (os, sys, open, exec,
    dunder names, class definitions, ...).
This is a sandbox for model-written snippets on a single trusted host, not a
security boundary against an adversary (no seccomp/namespace isolation).

Text: exact match after normalisation (case, surrounding whitespace and quotes,
trailing period, spacing around commas).
"""

from __future__ import annotations

import importlib.util
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .tasks import Task

ROOT = Path(__file__).resolve().parents[2]


def _load_humaneval():
    spec = importlib.util.spec_from_file_location("rsi_scripts_humaneval", ROOT / "scripts" / "humaneval.py")
    mod = importlib.util.module_from_spec(spec)
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # never write into scripts/
    try:
        spec.loader.exec_module(mod)  # module level: imports, constants and function definitions only
    finally:
        sys.dont_write_bytecode = prev
    return mod


HUMANEVAL = _load_humaneval()
limits = HUMANEVAL.limits
Tokenizer = HUMANEVAL.Tokenizer

DEFAULT_TIMEOUT = 5.0
ALLOWED_IMPORTS = {"math", "itertools", "functools", "collections", "string", "re", "heapq", "bisect", "operator"}
_BANNED = re.compile(
    r"__\w+__"
    r"|\b(?:exit|quit|eval|exec|compile|open|input|globals|locals|vars|getattr|setattr|delattr|breakpoint|help|memoryview)\s*\("
    r"|\b(?:os|sys|subprocess|socket|shutil|ctypes|signal|builtins|importlib|pathlib|threading|multiprocessing|resource|inspect|gc)\b"
    r"|\bclass\s"
)
_IMPORT = re.compile(r"^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import|import[ \t]+([\w., \t]+))", re.M)
_TOPLEVEL = re.compile(r"\n(?=\S)")


def static_check(code: str) -> str | None:
    """Reason for rejecting a completion before execution, or None."""
    for m in _IMPORT.finditer(code):
        mods = [m.group(1)] if m.group(1) else [x.strip().split(" ")[0] for x in m.group(2).split(",")]
        for mod in mods:
            if mod.split(".")[0] not in ALLOWED_IMPORTS:
                return f"import of {mod!r} not allowed"
    m = _BANNED.search(code)
    if m:
        return f"banned construct {m.group(0).strip()!r}"
    return None


def cut_code(completion: str) -> str:
    """The function body: everything before the first line that starts at column 0.

    Code prompts end at the closing docstring quotes, so a body starts with a newline."""
    m = _TOPLEVEL.search(completion)
    body = completion[: m.start()] if m else completion
    body = body.rstrip()
    return body + "\n" if body.strip() else ""


def cut_text(completion: str) -> str:
    return completion.split("\n", 1)[0].rstrip()


def normalize_answer(s: str) -> str:
    s = s.strip().rstrip(".").strip().strip("\"'`").strip()
    s = re.sub(r"\s+", " ", s.lower())
    s = re.sub(r"\s*,\s*", ", ", s)
    return s


def run_program(program: str, timeout: float = DEFAULT_TIMEOUT, marker: str | None = None) -> dict:
    """Execute `program` in the sandbox. status: pass | wrong | error | syntax | timeout | crash."""
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rsi-sbx-") as d:
        f = Path(d) / "t.py"
        f.write_text(program)
        with open(Path(d) / "stdout", "wb") as out, open(Path(d) / "stderr", "wb") as err:
            p = subprocess.Popen([sys.executable, "-I", str(f)], cwd=d, env={}, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 preexec_fn=limits, start_new_session=True)
            # Wait without reaping (WNOWAIT): while the leader is unreaped its pid,
            # and so the process-group id, cannot be reused, so killpg below can
            # only hit this sandbox's own group (any children it spawned included).
            deadline = time.monotonic() + timeout
            timed_out = False
            while os.waitid(os.P_PID, p.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                time.sleep(0.005)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = p.wait()
        stdout = (Path(d) / "stdout").read_bytes()[-4096:].decode("utf-8", "replace")
        stderr = (Path(d) / "stderr").read_bytes()[-4096:].decode("utf-8", "replace")
    res = {"seconds": round(time.monotonic() - t0, 3), "returncode": rc, "stderr_tail": stderr[-400:]}
    if timed_out:
        return {**res, "status": "timeout", "returncode": None}
    if rc is not None and rc < 0:
        return {**res, "status": "crash", "signal": -rc}
    if rc == 0:
        if marker is not None and marker not in stdout:
            return {**res, "status": "error", "reason": "exited before the hidden tests completed"}
        return {**res, "status": "pass"}
    last = stderr.strip().splitlines()[-1] if stderr.strip() else ""
    if last.startswith(("SyntaxError", "IndentationError", "TabError")):
        return {**res, "status": "syntax"}
    if last.startswith("AssertionError"):
        return {**res, "status": "wrong"}
    return {**res, "status": "error", "reason": last[:200]}


def check_code(task: Task, completion: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    body = cut_code(completion)
    if not body.strip():
        return {"status": "empty", "cut": body}
    reason = static_check(body)
    if reason:
        return {"status": "rejected", "reason": reason, "cut": body}
    marker = "RSI-TESTS-DONE-" + secrets.token_hex(8)
    program = task.prompt + body + "\n\n" + task.tests + f"\nprint({marker!r})\n"
    return {**run_program(program, timeout, marker), "cut": body}


def check_text(task: Task, completion: str) -> dict:
    ans = cut_text(completion)
    if not ans.strip():
        return {"status": "empty", "cut": ans}
    ok = normalize_answer(ans) == normalize_answer(task.answer)
    return {"status": "pass" if ok else "wrong", "cut": ans}


def verify(task: Task, completion: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Check one completion; `cut` is the part that would enter training data."""
    if task.family == "code":
        return check_code(task, completion, timeout)
    if task.family == "text":
        return check_text(task, completion)
    raise ValueError(f"unknown task family {task.family}")
