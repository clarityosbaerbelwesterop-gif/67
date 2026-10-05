"""T1/T2 pilot: DiLoCo against synchronous data parallelism at equal tokens, simulated in one process.

Every arm sees the same model (Osirus's native Rouge model, byte-level), the same data
(enwik8; K disjoint training shards, one per worker) and the same number of training tokens:
- ddp: one replica, batch K*B per step, gradients averaged every step (identical to synchronous
  DDP over K workers);
- diloco: K replicas of batch B, each with its own AdamW, synchronised every H steps through the
  outer Nesterov step on the averaged delta, compressed by a codec.
Bytes are what one worker would send: DDP all-reduces a full fp32 gradient every step.

    python -m fabric.pilot --osirus ../Osirus --arm diloco-h250-fp32 --seed 0 --out results/t1/x.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from fabric.diloco import Codec, Outer, assign, flatten, outer_gradient

HERE = Path(__file__).resolve().parents[1]


def load_osirus(root: Path):
    sys.path[:0] = [str(root / "training" / "rouge"), str(root / "research" / "rouge-architecture")]
    from benchmarks.text import Corpus
    from native.config import RougeConfig
    from native.model import RougeModel
    return Corpus, RougeConfig, RougeModel


class Shard:
    """Random windows from one worker's contiguous 1/K slice of the training bytes."""

    def __init__(self, data: torch.Tensor, worker: int, workers: int, seq: int, batch: int, seed: int):
        size = len(data) // workers
        self.data, self.lo, self.hi = data, worker * size, (worker + 1) * size
        self.seq, self.batch = seq, batch
        self.gen = torch.Generator().manual_seed(1_000_003 * seed + worker)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(self.lo, self.hi - self.seq - 1, (self.batch,), generator=self.gen)
        rows = torch.stack([self.data[s:s + self.seq + 1] for s in starts.tolist()]).long()
        return rows[:, :-1], rows[:, 1:]


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def adamw(model, lr: float, wd: float):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8)


def inner_step(model, opt, x, y, lr: float, clip: float) -> float:
    for group in opt.param_groups:
        group["lr"] = lr
    model.train()
    loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return loss.detach().item()


@torch.no_grad()
def val_bpb(model, windows: torch.Tensor) -> float:
    model.eval()
    total = 0.0
    for chunk in windows.split(32):
        total += float(model(chunk[:, :-1].long(), chunk[:, 1:].long())) * len(chunk)
    return total / len(windows) / math.log(2)


def run(arm: dict, spec: dict, seed: int, osirus: Path) -> dict:
    Corpus, RougeConfig, RougeModel = load_osirus(osirus)
    torch.set_num_threads(os.cpu_count() or 1)
    corpus = Corpus()
    train, valid = corpus.splits["train"], corpus.splits["valid"]
    seq, batch, workers, steps = spec["seq"], spec["batch"], spec["workers"], spec["steps"]
    windows = torch.stack([valid[i * (seq + 1):(i + 1) * (seq + 1)] for i in range(spec["eval_windows"])])
    cfg = RougeConfig(**spec["model"])
    torch.manual_seed(seed)
    global_model = RougeModel(cfg)
    params = sum(p.numel() for p in global_model.parameters())
    shards = [Shard(train, w, workers, seq, batch, seed) for w in range(workers)]
    curve, sent, start = [], 0, time.time()
    peak, warmup, wd, clip = spec["lr"], spec["warmup"], spec["weight_decay"], spec["clip"]

    if arm["kind"] == "ddp":
        opt = adamw(global_model, peak, wd)
        for step in range(steps):
            parts = [s.next() for s in shards]
            x, y = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
            inner_step(global_model, opt, x, y, lr_at(step, steps, peak, warmup), clip)
            sent += 4 * params
            if (step + 1) % spec["eval_every"] == 0 or step + 1 == steps:
                curve.append([step + 1, round(val_bpb(global_model, windows), 5)])
        final = global_model
    else:
        h = arm["h"]
        replicas = [RougeModel(cfg) for _ in range(workers)]
        opts = [adamw(m, peak, wd) for m in replicas]
        codecs = [Codec(arm.get("codec", "fp32")) for _ in range(workers)]
        outer = Outer(flatten(global_model.parameters()), arm.get("outer_lr", 0.7), arm.get("outer_momentum", 0.9))
        step = 0
        while step < steps:
            todo = min(h, steps - step)
            for m in replicas:
                assign(list(m.parameters()), outer.theta)
            for w, (m, opt) in enumerate(zip(replicas, opts)):
                for i in range(todo):
                    x, y = shards[w].next()
                    inner_step(m, opt, x, y, lr_at(step + i, steps, peak, warmup), clip)
            step += todo
            grad, nbytes = outer_gradient(outer.theta, [flatten(m.parameters()) for m in replicas], codecs)
            outer.step(grad)
            sent += nbytes
            if step % spec["eval_every"] == 0 or step == steps or todo < h:
                assign(list(global_model.parameters()), outer.theta)
                curve.append([step, round(val_bpb(global_model, windows), 5)])
        assign(list(global_model.parameters()), outer.theta)
        final = global_model

    return {
        "schema": "fabric.pilot/1", "arm": arm["name"], "kind": arm["kind"], "seed": seed,
        "h": arm.get("h", 1), "codec": arm.get("codec", "fp32-allreduce"), "workers": workers, "steps": steps,
        "tokens": steps * workers * batch * seq, "params": params,
        "final_val_bpb": round(val_bpb(final, windows), 5), "curve": curve,
        "bytes_sent_per_worker": sent, "wall_s": round(time.time() - start, 1),
        "corpus_sha256": corpus.sha256, "torch": torch.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osirus", required=True, type=Path, help="checkout of clarityosbaerbelwesterop-gif/Osirus")
    ap.add_argument("--experiment", type=Path, default=HERE / "experiments" / "t1-diloco.json")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, help="override (smoke runs only)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    exp = json.loads(args.experiment.read_text())
    spec = dict(exp["spec"], **({"steps": args.steps, "eval_every": args.steps} if args.steps else {}))
    arm = next(a for a in exp["arms"] if a["name"] == args.arm)
    result = run(arm, spec, args.seed, args.osirus)
    result["experiment"] = exp["id"]
    result["smoke"] = bool(args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print("FABRIC_RESULT", json.dumps({k: result[k] for k in ("arm", "seed", "final_val_bpb", "bytes_sent_per_worker", "wall_s")}))


if __name__ == "__main__":
    main()
