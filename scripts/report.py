"""Evaluate base, Rouge 1, Quasnir and Darus on both validation sets with the
same windows (fixed seed) and apply the Darus acceptance rule. Writes
runs/report.json and prints a table. Numbers are measured, never edited."""
import json
import os
import subprocess
import sys

FORGE = "target/release/forge"
MODELS = {"base": "runs/base-s/FINAL", "rouge-1": "runs/rouge-1/FINAL", "quasnir-1": "runs/quasnir-1/FINAL", "darus-1": "runs/darus-1/FINAL"}
SUITES = {"general": "data/stores/general/train.meta.json", "code": "data/stores/code/train.meta.json"}


# Post-training quantisation of Darus: weight matrices fake-quantised, norms f32.
QUANT = ["bf16", "fp8-e4m3", "mx-fp8-e4m3", "mx-fp4-e2m1", "fp4-e2m1"]


def evaluate(ckpt: str, data: str, quant: str | None = None) -> dict:
    cmd = [FORGE, "eval", "--ckpt", ckpt, "--data", data, "--batch", "8", "--seq", "256", "--batches", "32", "--seed", "4242"]
    out = subprocess.run(cmd + (["--quant", quant] if quant else []), check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main() -> int:
    if os.path.isdir("runs/darus-1/FINAL-default"):  # fixed settings, before scripts/merge_search.py
        MODELS["darus-1-default"] = "runs/darus-1/FINAL-default"
    merge = json.load(open("training/configs/darus-1.merge.json"))
    tol = merge["acceptance"]["max_regression_vs_parents"]
    rows = {m: {s: evaluate(c, d) for s, d in SUITES.items()} for m, c in MODELS.items()}
    verdict = []
    for s in SUITES:
        best = min(rows["rouge-1"][s]["loss"], rows["quasnir-1"][s]["loss"])
        darus = rows["darus-1"][s]["loss"]
        # Lower loss is better: Darus may exceed the best parent's loss by at most `tol` (relative).
        ok = darus <= best * (1 + tol)
        verdict.append({"suite": s, "darus_loss": darus, "best_parent_loss": best, "within_tolerance": ok})
    quant = {}
    for q in QUANT:
        r = {s: evaluate(MODELS["darus-1"], d, q) for s, d in SUITES.items()}
        quant[q] = {"weight_bytes": next(iter(r.values()))["weight_bytes"],
                    **{s: {"loss": r[s]["loss"], "delta_vs_f32": r[s]["loss"] - rows["darus-1"][s]["loss"]} for s in SUITES}}
    report = {"evaluations": rows, "darus_acceptance": verdict, "accepted": all(v["within_tolerance"] for v in verdict),
              "darus_quantized": {"f32_weight_bytes": rows["darus-1"][next(iter(SUITES))]["weight_bytes"], "formats": quant}}
    json.dump(report, open("runs/report.json", "w"), indent=2)
    print(f"{'model':<10} " + " ".join(f"{s+' loss':>12} {s+' ppl':>10} {s+' acc':>9}" for s in SUITES))
    for m, r in rows.items():
        print(f"{m:<10} " + " ".join(f"{r[s]['loss']:>12.4f} {r[s]['perplexity']:>10.2f} {r[s]['next_token_acc']:>9.4f}" for s in SUITES))
    print("Darus accepted:", report["accepted"], verdict)
    print(f"{'Darus as':<12} {'MB':>7} " + " ".join(f"{s+' loss':>12} {'Δ':>8}" for s in SUITES))
    print(f"{'f32':<12} {report['darus_quantized']['f32_weight_bytes'] / 1e6:>7.1f} " + " ".join(f"{rows['darus-1'][s]['loss']:>12.4f} {0:>8.4f}" for s in SUITES))
    for q, r in quant.items():
        print(f"{q:<12} {r['weight_bytes'] / 1e6:>7.1f} " + " ".join(f"{r[s]['loss']:>12.4f} {r[s]['delta_vs_f32']:>8.4f}" for s in SUITES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
