"""Devices, data-parallel sharding and output for the accel commands.

Data parallelism: one 1-D mesh ("data") over the devices. Parameters and AdamW moments are
replicated; every batch is split along its batch axis, so XLA all-reduces the gradients
inside the compiled step. Works the same on one CPU, a GPU box, a single TPU VM (v3-8,
v4-8, v5e-8, Kaggle/Colab TPU) and, with ACCEL_DISTRIBUTED=1, a multi-host TPU slice
(every process samples the same global batch from the same seed and feeds only its own
shards; process 0 prints and writes checkpoints).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_initialised = False


def init_distributed() -> None:
    global _initialised
    if _initialised:
        return
    _initialised = True
    if os.environ.get("ACCEL_DISTRIBUTED", "") not in ("", "0"):
        import jax
        jax.distributed.initialize()


def is_main() -> bool:
    import jax
    return jax.process_index() == 0


def now() -> float:
    return time.time()


def emit(obj: dict) -> None:
    """One telemetry line, exactly as forge's stdout sink prints a serde_json Value."""
    from .forge_io import dumps
    if is_main():
        sys.stdout.write(dumps(obj) + "\n")
        sys.stdout.flush()


def event(level: str, msg: str, step: int) -> dict:
    return {"type": "event", "level": level, "msg": msg, "step": step, "ts": now()}


class Mesh:
    """A 1-D data-parallel mesh over the first `n` devices."""

    def __init__(self, n: int = 0, batch: int | None = None, warn=None):
        import jax
        from jax.sharding import Mesh as JMesh, NamedSharding, PartitionSpec as P
        devs = jax.devices()
        n = len(devs) if n <= 0 else min(n, len(devs))
        if batch is not None and batch % n:
            if jax.process_count() > 1:
                raise ValueError(f"batch {batch} is not divisible by the {n} devices of this slice")
            m = max(k for k in range(1, n + 1) if batch % k == 0)
            if warn:
                warn(f"batch {batch} is not divisible by {n} devices; using {m} (make batch a multiple of {n})")
            n = m
        self.devices = devs[:n]
        self.n = n
        self.platform = self.devices[0].platform
        self.kind = getattr(self.devices[0], "device_kind", self.platform)
        self.mesh = JMesh(np.array(self.devices), ("data",))
        self.repl = NamedSharding(self.mesh, P())
        self.rows = NamedSharding(self.mesh, P("data"))           # [B, ...]
        self.micro_rows = NamedSharding(self.mesh, P(None, "data"))  # [A, B, ...]

    def put(self, a: np.ndarray, sharding):
        import jax
        a = np.asarray(a)
        return jax.make_array_from_callback(a.shape, sharding, lambda idx: a[idx])

    def put_tree(self, tree: dict, sharding) -> dict:
        return {k: self.put(v, sharding) for k, v in tree.items()}


def to_host(x) -> np.ndarray:
    """A (possibly sharded, possibly multi-host) array as one numpy array."""
    if getattr(x, "is_fully_addressable", True):
        return np.asarray(x)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def replicated_to_host(x) -> np.ndarray:
    """A replicated array: every device holds all of it, so one local shard is the whole value."""
    if getattr(x, "is_fully_addressable", True):
        return np.asarray(x)
    return np.asarray(x.addressable_data(0))


def memory_stats(dev) -> dict:
    """forge's `hbvm` block from the device allocator (zeros where the backend has no stats)."""
    try:
        s = dev.memory_stats() or {}
    except Exception:  # noqa: BLE001 - CPU and some plugins raise or return None
        s = {}
    return {"capacity": int(s.get("bytes_limit", 0)), "used": int(s.get("bytes_in_use", 0)),
            "peak": int(s.get("peak_bytes_in_use", 0)), "live": int(s.get("num_allocs", 0)),
            "allocs": int(s.get("num_allocs", 0)), "frees": 0, "fragmentation": 0.0}
