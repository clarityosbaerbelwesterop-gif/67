"""Evolutionary search for the Darus merge (CPU only, no gradients).

Darus = merge(θ0; Rouge 1, Quasnir). Instead of fixed merge settings, this
searches method (TIES or linear task arithmetic), TIES density and the
task-vector scale λ, measuring every candidate on both validation suites.

Objective (minimised), the same quantity the acceptance rule bounds:
    worst-case relative regression = max over suites of loss_s / best_parent_loss_s − 1
Ties are broken by the mean loss.

Honesty: candidates are selected on validation windows drawn with
`select_seed`; `scripts/report.py` then measures with its own seed (4242) on
the same validation split. Selection and report share the split, which is the
standard hyperparameter-selection setting; both seeds are recorded in
runs/darus-1/search.json.

Usage: python3 scripts/merge_search.py [--budget 24]
"""
import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

FORGE = "target/release/forge"
SUITES = {"general": "data/stores/general/train.meta.json", "code": "data/stores/code/train.meta.json"}


def evaluate(ckpt: str, data: str, seed: int, batches: int, seq: int = 256) -> float:
    out = subprocess.run(
        [FORGE, "eval", "--ckpt", ckpt, "--data", data, "--batch", "8", "--seq", str(seq),
         "--batches", str(batches), "--seed", str(seed)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])["loss"]


def merge(spec: dict, base: str, children: list, out: str) -> None:
    cmd = [FORGE, "merge", "--base", base, "--method", spec["method"], "--lambda", f"{spec['lambda']:.4f}", "--out", out]
    for c in children:
        cmd += ["--child", c]
    if spec["method"] == "ties":
        cmd += ["--density", f"{spec['density']:.4f}"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def key(spec: dict) -> tuple:
    return (spec["method"], round(spec.get("density") or 0, 2), round(spec["lambda"], 2))


def mutate(spec: dict, rng: random.Random) -> dict:
    s = dict(spec)
    if rng.random() < 0.15:
        s["method"] = "linear" if s["method"] == "ties" else "ties"
    if s["method"] == "ties":
        s["density"] = min(1.0, max(0.05, (s.get("density") or 0.5) * math.exp(rng.gauss(0, 0.35))))
    else:
        s["density"] = None
    s["lambda"] = min(2.0, max(0.3, s["lambda"] * math.exp(rng.gauss(0, 0.2))))
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/darus-1.merge.json")
    ap.add_argument("--budget", type=int, default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    suites = cfg.get("suites", SUITES)
    search = cfg.get("search", {})
    budget = a.budget or search.get("budget", 24)
    seed, batches, seq = search.get("select_seed", 777), search.get("select_batches", 16), search.get("seq", 256)
    base, children, final = cfg["base"], cfg["children"], Path(cfg["out"])
    work = final.parent / "search"
    work.mkdir(parents=True, exist_ok=True)

    parents = {c: {s: evaluate(c, d, seed, batches, seq) for s, d in suites.items()} for c in children}
    best_parent = {s: min(p[s] for p in parents.values()) for s in suites}
    print(json.dumps({"type": "parents", "losses": parents, "best": best_parent}), flush=True)

    def score(spec: dict, idx: int) -> dict:
        out = str(work / f"cand-{idx:03d}")
        merge(spec, base, children, out)
        losses = {s: evaluate(out, d, seed, batches, seq) for s, d in suites.items()}
        shutil.rmtree(out)
        reg = {s: losses[s] / best_parent[s] - 1 for s in suites}
        r = {**spec, "losses": losses, "regression": reg,
             "objective": max(reg.values()), "mean_loss": sum(losses.values()) / len(losses)}
        print(json.dumps({"type": "candidate", "index": idx, **r}), flush=True)
        return r

    seeds = [
        {"method": "ties", "density": cfg.get("density", 0.5), "lambda": cfg.get("lambda", 1.0)},
        {"method": "ties", "density": 0.2, "lambda": 1.0},
        {"method": "ties", "density": 0.8, "lambda": 1.0},
        {"method": "ties", "density": 0.5, "lambda": 0.7},
        {"method": "ties", "density": 0.5, "lambda": 1.4},
        {"method": "linear", "density": None, "lambda": 1.0},
        {"method": "linear", "density": None, "lambda": 1.6},
        {"method": "ties", "density": 0.3, "lambda": 1.6},
    ]
    rng = random.Random(search.get("rng_seed", 67))
    seen, results = set(), []
    queue = list(seeds)
    while len(results) < budget:
        if not queue:
            elite = sorted(results, key=lambda r: (r["objective"], r["mean_loss"]))[:3]
            queue = [mutate(rng.choice(elite), rng) for _ in range(4)]
        spec = queue.pop(0)
        if key(spec) in seen:
            continue
        seen.add(key(spec))
        results.append(score(spec, len(results)))

    best = min(results, key=lambda r: (r["objective"], r["mean_loss"]))
    tmp = final.parent / "FINAL.search"
    if tmp.exists():
        shutil.rmtree(tmp)
    merge(best, base, children, str(tmp))
    default = final.parent / "FINAL-default"
    if final.exists() and not default.exists():
        final.rename(default)
    elif final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    shutil.rmtree(work, ignore_errors=True)
    record = {"selection": {"split": "val", "seed": seed, "batches": batches, "seq": seq, "budget": budget},
              "parents": parents, "best_parent": best_parent, "candidates": results, "best": best,
              "final": str(final), "previous_default": str(default) if default.exists() else None}
    json.dump(record, open(final.parent / "search.json", "w"), indent=2)
    print(json.dumps({"type": "best", **best}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
