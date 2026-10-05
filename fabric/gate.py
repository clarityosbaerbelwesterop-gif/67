"""Applies the pre-registered gate of experiments/t1-diloco.json to the committed result files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def verdict(exp: dict, results: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for r in results:
        if r.get("experiment") == exp["id"] and not r.get("smoke"):
            by_arm.setdefault(r["arm"], []).append(r)
    missing = [(a["name"], s) for a in exp["arms"] for s in exp["seeds"]
               if s not in {r["seed"] for r in by_arm.get(a["name"], [])}]
    table = {arm: {"mean_bpb": round(statistics.mean(r["final_val_bpb"] for r in rs), 5),
                   "sd_bpb": round(statistics.stdev(r["final_val_bpb"] for r in rs), 5) if len(rs) > 1 else None,
                   "bytes_per_worker": rs[0]["bytes_sent_per_worker"], "seeds": sorted(r["seed"] for r in rs)}
             for arm, rs in sorted(by_arm.items())}
    out = {"experiment": exp["id"], "arms": table, "missing": missing}
    if missing:
        out["t1"] = "INCOMPLETE"
        return out
    ddp, dl = table["ddp"], table["diloco-h250-fp32"]
    ratio = ddp["bytes_per_worker"] / dl["bytes_per_worker"]
    out["bytes_ratio_ddp_over_diloco_h250"] = round(ratio, 1)
    out["bpb_ratio_diloco_h250_over_ddp"] = round(dl["mean_bpb"] / ddp["mean_bpb"], 4)
    out["t1"] = "PASS" if dl["mean_bpb"] <= ddp["mean_bpb"] * 1.03 and ratio >= 100 else "FAIL"
    t2 = {}
    for arm in ("diloco-h250-int8", "diloco-h250-topk1"):
        t2[arm] = {"bpb_cost_vs_fp32": round(table[arm]["mean_bpb"] / dl["mean_bpb"] - 1, 4),
                   "bytes_vs_fp32": round(table[arm]["bytes_per_worker"] / dl["bytes_per_worker"], 4)}
    t2["int8_lossless"] = t2["diloco-h250-int8"]["bpb_cost_vs_fp32"] <= 0.01
    out["t2"] = t2
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, default=HERE / "experiments" / "t1-diloco.json")
    ap.add_argument("--results", type=Path, default=HERE / "results" / "t1")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "t1" / "VERDICT.json")
    args = ap.parse_args()
    exp = json.loads(args.experiment.read_text())
    results = [json.loads(p.read_text()) for p in sorted(args.results.glob("*.json")) if p.name != "VERDICT.json"]
    out = verdict(exp, results)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
