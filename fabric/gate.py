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
        out["verdict"] = "INCOMPLETE"
        return out
    gate = exp["gate"]
    if "primary_arm" not in gate:                      # t1-diloco: its gate was written before this generalisation
        gate = {"baseline_arm": "ddp", "primary_arm": "diloco-h250-fp32", "max_bpb_ratio": 1.03, "min_bytes_ratio": 100}
    base, prim = table[gate["baseline_arm"]], table[gate["primary_arm"]]
    ratio = base["bytes_per_worker"] / prim["bytes_per_worker"]
    bpb_ratio = prim["mean_bpb"] / base["mean_bpb"]
    out["primary_arm"], out["bytes_ratio"], out["bpb_ratio"] = gate["primary_arm"], round(ratio, 1), round(bpb_ratio, 4)
    passed = bpb_ratio <= gate["max_bpb_ratio"] and ratio >= gate["min_bytes_ratio"]
    out["verdict"] = "PASS" if passed else "FAIL"
    out["vs_baseline"] = {arm: round(r["mean_bpb"] / base["mean_bpb"] - 1, 4) for arm, r in table.items()}
    if exp["id"] == "t1-diloco":                       # field names of the committed t1 verdict
        out["bytes_ratio_ddp_over_diloco_h250"], out["bpb_ratio_diloco_h250_over_ddp"], out["t1"] = out["bytes_ratio"], out["bpb_ratio"], out["verdict"]
        dl = prim
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
    ap.add_argument("--results", type=Path, help="default: results/<experiment id>")
    ap.add_argument("--out", type=Path, help="default: <results>/VERDICT.json")
    args = ap.parse_args()
    exp = json.loads(args.experiment.read_text())
    args.results = args.results or HERE / "results" / ("t1" if exp["id"] == "t1-diloco" else exp["id"])
    args.out = args.out or args.results / "VERDICT.json"
    results = [json.loads(p.read_text()) for p in sorted(args.results.glob("*.json")) if p.name != "VERDICT.json"]
    out = verdict(exp, results)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
