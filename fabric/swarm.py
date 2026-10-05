"""DiLoCo across real machines: every worker is its own job, and only files travel between them.

    init       global parameters + outer optimiser state          -> state.pt
    work       one worker: copy global params, train H steps on its shard, send its delta
               (compressed by the codec); its AdamW state and top-k residual stay with the worker
    aggregate  average the deltas, outer Nesterov step, validation BPB -> new state.pt

A GitHub Actions run passes these files as artifacts (.github/workflows/swarm.yml); any other
transport (object storage, scp, a USB stick) works the same way. Same model, data and schedule
as fabric/pilot.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from fabric.diloco import Codec, Outer, assign, flatten
from fabric.pilot import HERE, Shard, adamw, inner_step, load_osirus, lr_at, val_bpb


def setup(args):
    Corpus, RougeConfig, RougeModel = load_osirus(args.osirus)
    torch.set_num_threads(os.cpu_count() or 1)
    exp = json.loads(args.experiment.read_text())
    spec = exp["spec"]
    return exp, spec, Corpus(), RougeConfig(**spec["model"]), RougeModel


def cmd_init(args) -> None:
    exp, spec, corpus, cfg, RougeModel = setup(args)
    torch.manual_seed(args.seed)
    model = RougeModel(cfg)
    outer = Outer(flatten(model.parameters()), args.outer_lr, args.outer_momentum)
    torch.save({"outer": outer.state(), "round": 0, "step": 0, "log": [], "seed": args.seed,
                "corpus_sha256": corpus.sha256}, args.out)
    print(f"init: {sum(p.numel() for p in model.parameters())} parameters -> {args.out}")


def cmd_work(args) -> None:
    exp, spec, corpus, cfg, RougeModel = setup(args)
    state = torch.load(args.state, weights_only=False)
    model = RougeModel(cfg)
    assign(list(model.parameters()), state["outer"]["theta"])
    opt = adamw(model, spec["lr"], spec["weight_decay"])
    private = torch.load(args.private, weights_only=False) if args.private and args.private.exists() else None
    if private:
        opt.load_state_dict(private["opt"])
    codec = Codec(args.codec)
    codec.residual = private["residual"] if private else None
    shard = Shard(corpus.splits["train"], args.worker, spec["workers"], spec["seq"], spec["batch"],
                  seed=state["seed"] * 1000 + state["round"])
    total = args.h * args.rounds
    for i in range(args.h):
        x, y = shard.next()
        inner_step(model, opt, x, y, lr_at(state["step"] + i, total, spec["lr"], spec["warmup"]), spec["clip"])
    payload, nbytes = codec.encode(state["outer"]["theta"] - flatten(model.parameters()))
    torch.save({"payload": payload, "bytes": nbytes, "worker": args.worker, "round": state["round"]}, args.out)
    args.private_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"opt": opt.state_dict(), "residual": codec.residual}, args.private_out)
    print(f"work: worker {args.worker} round {state['round']} sent {nbytes} bytes")


def cmd_aggregate(args) -> None:
    exp, spec, corpus, cfg, RougeModel = setup(args)
    state = torch.load(args.state, weights_only=False)
    deltas = [torch.load(p, weights_only=False) for p in sorted(args.deltas.rglob("delta-*.pt"))]
    if len(deltas) != spec["workers"]:
        raise SystemExit(f"expected {spec['workers']} deltas, found {len(deltas)}")
    outer = Outer.load(state["outer"])
    grad = sum(Codec.decode(d["payload"]) for d in deltas) / len(deltas)
    outer.step(grad)
    model = RougeModel(cfg)
    assign(list(model.parameters()), outer.theta)
    valid, seq = corpus.splits["valid"], spec["seq"]
    windows = torch.stack([valid[i * (seq + 1):(i + 1) * (seq + 1)] for i in range(spec["eval_windows"])])
    bpb = round(val_bpb(model, windows), 5)
    entry = {"round": state["round"] + 1, "step": state["step"] + args.h, "val_bpb": bpb,
             "bytes_per_worker": max(d["bytes"] for d in deltas)}
    state.update(outer=outer.state(), round=state["round"] + 1, step=state["step"] + args.h, log=state["log"] + [entry])
    torch.save(state, args.out)
    print("SWARM_ROUND", json.dumps(entry))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["init", "work", "aggregate"])
    ap.add_argument("--osirus", required=True, type=Path)
    ap.add_argument("--experiment", type=Path, default=HERE / "experiments" / "t1-diloco.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--state", type=Path)
    ap.add_argument("--worker", type=int)
    ap.add_argument("--h", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3, help="rounds in the whole run (sets the LR schedule length)")
    ap.add_argument("--codec", default="fp32")
    ap.add_argument("--private", type=Path, help="this worker's optimiser state from its previous round")
    ap.add_argument("--private-out", type=Path)
    ap.add_argument("--deltas", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    {"init": cmd_init, "work": cmd_work, "aggregate": cmd_aggregate}[args.command](args)


if __name__ == "__main__":
    main()
