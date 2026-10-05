"""DiLoCo: low-communication data-parallel training (Douillard et al. 2023, arXiv 2311.08105).

K workers each train a full replica with their own inner optimiser (AdamW) on their own data
shard for H steps, without talking to anyone. Then each sends its parameter delta
(global - worker); the deltas are averaged into an "outer gradient" and an outer optimiser
(SGD with Nesterov momentum, lr 0.7, momentum 0.9 in the paper) updates the global parameters,
which every worker copies before the next H steps. Communication drops by a factor of H against
synchronous data parallelism; every worker still needs memory for the whole model.

Codecs compress what a worker sends. Top-k keeps an error-feedback residual on the worker, so
what is dropped in one round is sent in a later one (Stich et al. 2018).
"""

from __future__ import annotations

import math

import torch


def flatten(params) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).float() for p in params])


def assign(params, vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(vector[offset:offset + n].view_as(p).to(p.dtype))
            offset += n


class Outer:
    """SGD with Nesterov momentum on the averaged pseudo-gradient (torch.optim.SGD semantics)."""

    def __init__(self, global_vec: torch.Tensor, lr: float = 0.7, momentum: float = 0.9, nesterov: bool = True):
        self.theta = global_vec.clone()
        self.lr, self.momentum, self.nesterov = lr, momentum, nesterov
        self.buf = torch.zeros_like(self.theta)

    def step(self, outer_grad: torch.Tensor) -> torch.Tensor:
        if self.momentum:
            self.buf.mul_(self.momentum).add_(outer_grad)
            update = outer_grad + self.momentum * self.buf if self.nesterov else self.buf
        else:
            update = outer_grad
        self.theta.add_(update, alpha=-self.lr)
        return self.theta

    def state(self) -> dict:
        return {"theta": self.theta, "buf": self.buf, "lr": self.lr, "momentum": self.momentum, "nesterov": self.nesterov}

    @classmethod
    def load(cls, state: dict) -> "Outer":
        outer = cls(state["theta"], state["lr"], state["momentum"], state["nesterov"])
        outer.buf = state["buf"].clone()
        return outer


class Codec:
    """Compresses one worker's delta. encode() returns (payload, bytes on the wire); decode() restores a dense vector."""

    def __init__(self, spec: str = "fp32", chunk: int = 4096):
        self.spec, self.chunk = spec, chunk
        self.residual: torch.Tensor | None = None
        if spec.startswith("topk:"):
            self.fraction = float(spec.split(":", 1)[1])
            assert 0 < self.fraction <= 1, spec
        elif spec not in ("fp32", "fp16", "int8"):
            raise ValueError(f"unknown codec {spec}")

    def encode(self, delta: torch.Tensor) -> tuple[dict, int]:
        n = delta.numel()
        if self.spec == "fp32":
            return {"dense": delta.clone()}, 4 * n
        if self.spec == "fp16":
            return {"dense": delta.half()}, 2 * n
        if self.spec == "int8":
            pad = (-n) % self.chunk
            blocks = torch.nn.functional.pad(delta, (0, pad)).view(-1, self.chunk)
            scale = blocks.abs().amax(dim=1).clamp_min(1e-12) / 127.0
            q = torch.round(blocks / scale[:, None]).clamp(-127, 127).to(torch.int8)
            return {"q": q, "scale": scale, "n": n}, q.numel() + 4 * scale.numel()
        # top-k with error feedback: send the k largest entries of (delta + residual), keep the rest
        acc = delta if self.residual is None else delta + self.residual
        k = max(1, math.ceil(self.fraction * n))
        idx = acc.abs().topk(k).indices
        values = acc[idx].half()
        self.residual = acc.clone()
        self.residual[idx] -= values.float()
        return {"idx": idx.int(), "values": values, "n": n}, k * (4 + 2)

    @staticmethod
    def decode(payload: dict) -> torch.Tensor:
        if "dense" in payload:
            return payload["dense"].float()
        if "q" in payload:
            return (payload["q"].float() * payload["scale"][:, None]).reshape(-1)[:payload["n"]]
        out = torch.zeros(payload["n"])
        out[payload["idx"].long()] = payload["values"].float()
        return out


def outer_gradient(global_vec: torch.Tensor, worker_vecs: list[torch.Tensor], codecs: list[Codec]) -> tuple[torch.Tensor, int]:
    """Average of the decoded worker deltas, and the bytes each worker sent (max over workers)."""
    total, sent = torch.zeros_like(global_vec), 0
    for vec, codec in zip(worker_vecs, codecs):
        payload, nbytes = codec.encode(global_vec - vec)
        total += Codec.decode(payload)
        sent = max(sent, nbytes)
    return total / len(worker_vecs), sent


def average_payloads(payloads: list[dict], expected: int, min_workers: int) -> tuple[torch.Tensor, int]:
    """Outer gradient from the deltas that arrived. A round survives lost workers down to a quorum:
    the global state lives with the aggregator, so a worker that drops out only costs its share of
    this round (it rejoins from the next global state)."""
    if len(payloads) > expected:
        raise ValueError(f"{len(payloads)} deltas for {expected} workers")
    if len(payloads) < min_workers:
        raise SystemExit(f"only {len(payloads)} of {expected} deltas arrived; quorum is {min_workers}")
    return sum(Codec.decode(p) for p in payloads) / len(payloads), len(payloads)
