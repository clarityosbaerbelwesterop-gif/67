"""Agents that watch every run and act through the same validated control
channel a human uses. Each agent runs in one of three modes:

- ``act``: its actions are sent to the run;
- ``advise``: decisions are recorded and shown, nothing is sent;
- ``off``: not consulted.

Every decision, executed or not, is broadcast and appended to the audit log
``runs/headcenter/decisions.jsonl``. Conditions are edge-triggered: a
decision is raised when a condition becomes true, not on every line while it
holds. History replayed at start-up primes the agents without decisions, so
an old incident is never acted on again.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from .telemetry import RunState, num

MODES = ("act", "advise", "off")


@dataclass
class Decision:
    agent: str
    run: str
    severity: str  # info | warn | critical
    reason: str
    action: dict | None = None
    step: int | None = None
    mode: str = "advise"
    executed: bool = False
    error: str | None = None
    ts: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return {"type": "decision", **asdict(self)}


class Agent:
    name = "agent"

    def __init__(self, mode: str = "act"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self._active: set[tuple[str, str]] = set()

    def _edge(self, run: str, cond: str, on: bool) -> bool:
        """True exactly when `cond` switches from off to on for `run`."""
        key = (run, cond)
        if not on:
            self._active.discard(key)
            return False
        if key in self._active:
            return False
        self._active.add(key)
        return True

    def observe(self, run: RunState, rec: dict, live: bool) -> list[Decision]:
        return []

    def tick(self, runs: dict[str, RunState], now: float) -> list[Decision]:
        return []


class _WatchState:
    def __init__(self) -> None:
        self.obs = 0
        self.ema: float | None = None
        self.spike_lines = 0
        self.peak_lr = 0.0
        self.lr_action_step: int | None = None
        self.grad = deque(maxlen=50)


class SiliconWatchdog(Agent):
    """Health of the run: numerics, divergence, memory, liveness.

    - non-finite loss → pause (critical); the newest checkpoint stays intact;
    - loss above 1.5× its EMA on 2 consecutive lines → halve the LR peak
      (`set_lr`), at most once per 200 steps;
    - gradient norm above 20× its recent median → warn;
    - HBVM peak above 97 % of capacity → warn; fragmentation above 0.5 → info;
    - no telemetry for max(180 s, 6× the usual interval) while running → warn.
    """

    name = "silicon-watchdog"
    SPIKE, SPIKE_LINES, MIN_OBS, LR_GAP = 1.5, 2, 20, 200
    GRAD_X, MEM_FRAC, FRAG, STALL_MIN_S = 20.0, 0.97, 0.5, 180.0

    def __init__(self, mode: str = "act"):
        super().__init__(mode)
        self.state: dict[str, _WatchState] = {}

    def observe(self, run: RunState, rec: dict, live: bool) -> list[Decision]:
        st = self.state.setdefault(run.name, _WatchState())
        out: list[Decision] = []
        kind = rec.get("type")

        def emit(severity: str, reason: str, action: dict | None = None) -> None:
            if live:
                out.append(Decision(self.name, run.name, severity, reason, action, run.step))

        if kind == "event" and rec.get("level") == "error" and "non-finite" in str(rec.get("msg", "")):
            if self._edge(run.name, "nonfinite", True):
                emit("critical", f"trainer reported: {rec.get('msg')}", {"cmd": "pause"})
            return out
        if kind != "telemetry":
            return out

        loss = num(rec.get("loss"))
        if self._edge(run.name, "nonfinite", loss is None):
            emit("critical", f"non-finite loss at step {run.step}", {"cmd": "pause"})
        if loss is None:
            return out

        lr = num(rec.get("lr"))
        if lr is not None:
            st.peak_lr = max(st.peak_lr, lr)
        st.obs += 1
        spiking = st.ema is not None and st.obs > self.MIN_OBS and loss > self.SPIKE * st.ema
        st.spike_lines = st.spike_lines + 1 if spiking else 0
        if not spiking:
            st.ema = loss if st.ema is None else 0.9 * st.ema + 0.1 * loss
        sustained = st.spike_lines >= self.SPIKE_LINES
        if self._edge(run.name, "spike", sustained):
            recent = st.lr_action_step is not None and run.step - st.lr_action_step < self.LR_GAP
            if st.peak_lr > 0 and not recent:
                st.lr_action_step = run.step
                st.peak_lr *= 0.5
                emit(
                    "warn",
                    f"loss {loss:.4f} > {self.SPIKE}× EMA {st.ema:.4f} for {st.spike_lines} lines: halving LR peak",
                    {"cmd": "set_lr", "value": st.peak_lr},
                )
            else:
                emit("warn", f"loss {loss:.4f} > {self.SPIKE}× EMA {st.ema:.4f}; LR already lowered recently")

        gn = num(rec.get("grad_norm"))
        if gn is not None:
            exploding = len(st.grad) >= self.MIN_OBS and gn > self.GRAD_X * statistics.median(st.grad)
            if self._edge(run.name, "grad", exploding):
                emit("warn", f"gradient norm {gn:.3g} > {self.GRAD_X:g}× recent median {statistics.median(st.grad):.3g}")
            if not exploding:
                st.grad.append(gn)

        h = rec.get("hbvm") or {}
        cap, peak, frag = num(h.get("capacity")), num(h.get("peak")), num(h.get("fragmentation"))
        full = bool(cap) and peak is not None and peak / cap > self.MEM_FRAC
        if self._edge(run.name, "mem", full):
            emit("warn", f"HBVM peak at {peak / cap:.1%} of capacity")
        if self._edge(run.name, "frag", frag is not None and frag > self.FRAG):
            emit("info", f"HBVM fragmentation {frag:.2f}")
        return out

    def tick(self, runs: dict[str, RunState], now: float) -> list[Decision]:
        out = []
        for run in runs.values():
            if run.last_ts is None:
                continue
            limit = max(self.STALL_MIN_S, 6 * (run.median_interval() or 0.0))
            stalled = run.status == "running" and now - run.last_ts > limit
            if self._edge(run.name, "stall", stalled):
                out.append(
                    Decision(
                        self.name, run.name, "warn",
                        f"no telemetry for {now - run.last_ts:.0f} s (limit {limit:.0f} s)", None, run.step,
                    )
                )
        return out


class _JitState:
    def __init__(self) -> None:
        self.base: list[float] = []
        self.low_lines = 0
        self.last_action: float | None = None
        self.last_profile: float | None = None


class JitOptimizer(Agent):
    """Throughput of the run.

    The MATMUL rate achieved in each telemetry window (op FLOPs / op time) is
    compared with the run's own baseline (median of its first 5 windows). Below
    75 % of it for 3 consecutive windows, the agent re-runs the trainer's
    autotuner, which re-measures every GEMM variant per shape on the live
    machine and hot-swaps the faster kernels in without stopping the run (at
    most once per 30 min; the baseline is then re-learnt). Every 10 min it
    reports where the step time goes.
    """

    name = "jit-optimizer"
    BASE_LINES, DROP, DROP_LINES, COOLDOWN_S, PROFILE_S = 5, 0.75, 3, 1800.0, 600.0

    def __init__(self, mode: str = "act"):
        super().__init__(mode)
        self.state: dict[str, _JitState] = {}

    def observe(self, run: RunState, rec: dict, live: bool) -> list[Decision]:
        if rec.get("type") != "telemetry":
            return []
        st = self.state.setdefault(run.name, _JitState())
        ts = num(rec.get("ts")) or time.time()
        ops = rec.get("ops") or {}
        mm = ops.get("MATMUL") or {}
        flops, ns = num(mm.get("flops")), num(mm.get("ns"))
        out: list[Decision] = []
        if flops and ns:
            g = flops / ns  # FLOP per ns = GFLOP/s
            if len(st.base) < self.BASE_LINES:
                st.base.append(g)
            else:
                ref = statistics.median(st.base)
                st.low_lines = st.low_lines + 1 if g < self.DROP * ref else 0
                cooled = st.last_action is None or ts - st.last_action >= self.COOLDOWN_S
                if st.low_lines >= self.DROP_LINES and cooled:
                    st.last_action, st.low_lines, st.base = ts, 0, []
                    if live:
                        out.append(
                            Decision(
                                self.name, run.name, "warn",
                                f"MATMUL at {g:.0f} GFLOP/s, below {self.DROP:.0%} of baseline {ref:.0f}: re-autotuning kernels",
                                {"cmd": "autotune"}, run.step,
                            )
                        )
        if live and (st.last_profile is None or ts - st.last_profile >= self.PROFILE_S):
            st.last_profile = ts
            report = profile(ops)
            if report:
                out.append(Decision(self.name, run.name, "info", report, None, run.step))
        return out


def profile(ops: dict) -> str | None:
    """Top op time shares with achieved rates, and the non-GEMM share."""
    rows = []
    for op, s in ops.items():
        ns, flops = num((s or {}).get("ns")) or 0.0, num((s or {}).get("flops")) or 0.0
        if ns > 0:
            rows.append((ns, op, flops / ns))
    total = sum(r[0] for r in rows)
    if total <= 0:
        return None
    rows.sort(reverse=True)
    top = ", ".join(f"{op} {ns / total:.0%}" + (f" @ {g:.0f} GFLOP/s" if g > 0 else "") for ns, op, g in rows[:3])
    gemm = sum(ns for ns, op, _ in rows if op.startswith("MATMUL")) / total
    note = f"; non-GEMM ops take {1 - gemm:.0%} (fusion candidates)" if 1 - gemm > 0.3 else ""
    return f"step time: {top}{note}"
