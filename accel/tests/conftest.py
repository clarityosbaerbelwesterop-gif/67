"""Shared fixtures for the forge <-> accel parity tests.

Run:  nice -n 19 /home/user/accel-venv/bin/python -m pytest accel/tests -q
Every forge and accel process runs single-threaded (forge --threads 1 / threads: 1, XLA's CPU
client without its Eigen pool) under nice 19, so the tests stay small next to a live run.
Measured numbers are collected in one dict and written to $ACCEL_PARITY_REPORT (JSON) when set.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Before jax is imported anywhere: one CPU thread, deterministic reductions.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FORGE = Path(os.environ.get("FORGE_REAL", ROOT / "target" / "release" / "forge"))
PY = sys.executable
NICE = ["nice", "-n", "19"]
REPORT: dict = {}


def child_env(**extra) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run(cmd: list, env: dict | None = None, cwd=ROOT) -> str:
    r = subprocess.run(NICE + [str(c) for c in cmd], cwd=cwd, env=env or child_env(), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    if r.returncode:
        raise AssertionError(f"{' '.join(map(str, cmd))} failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def lines(out: str) -> list[dict]:
    return [json.loads(x) for x in out.splitlines() if x.startswith("{")]


def forge(*args) -> str:
    return run([FORGE, *args])


def forge_json(*args) -> dict:
    return lines(forge(*args, "--threads", "1"))[-1]


def accel(module: str, *args, env: dict | None = None) -> str:
    return run([PY, "-m", f"accel.{module}", *args], env=env)


def accel_json(module: str, *args) -> dict:
    return lines(accel(module, *args))[-1]


def record(key: str, value) -> None:
    REPORT[key] = value
    print(f"\n[parity] {key}: {json.dumps(value)}")


@pytest.fixture(scope="session")
def forge_bin() -> Path:
    if not FORGE.is_file() or not os.access(FORGE, os.X_OK):
        pytest.skip(f"forge binary not found at {FORGE}")
    return FORGE


def need_ckpt(rel: str) -> Path:
    p = ROOT / rel
    if not (p / "model.safetensors").exists() or not (p / "state.json").exists():
        pytest.skip(f"checkpoint {rel} not available")
    return p


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("ACCEL_PARITY_REPORT")
    if path and REPORT:
        Path(path).write_text(json.dumps(REPORT, indent=1, sort_keys=True))
