"""Evaluate base, Rouge 1, Quasnir and Darus on both validation sets with the
same windows (fixed seed) and apply the Darus acceptance rule. Writes
runs/report.json and prints a table. Numbers are measured, never edited."""
import json
import subprocess
import sys

FORGE = "target/release/forge"
MODELS = {"base": "runs/base-s/FINAL", "rouge-1": "runs/rouge-1/FINAL", "quasnir-1": "runs/quasnir-1/FINAL", "darus-1": "runs/darus-1/FINAL"}
SUITES = {"general": "data/out/general/train.meta.json", "code": "data/out/code/train.meta.json"}


def evaluate(ckpt: str, data: str) -> dict:
    out = subprocess.run([FORGE, "eval", "--ckpt", ckpt, "--data", data, "--batch", "8", "--seq", "256", "--batches", "32", "--seed", "4242"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main() -> int:
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
    report = {"evaluations": rows, "darus_acceptance": verdict, "accepted": all(v["within_tolerance"] for v in verdict)}
    json.dump(report, open("runs/report.json", "w"), indent=2)
    print(f"{'model':<10} " + " ".join(f"{s+' loss':>12} {s+' ppl':>10} {s+' acc':>9}" for s in SUITES))
    for m, r in rows.items():
        print(f"{m:<10} " + " ".join(f"{r[s]['loss']:>12.4f} {r[s]['perplexity']:>10.2f} {r[s]['next_token_acc']:>9.4f}" for s in SUITES))
    print("Darus accepted:", report["accepted"], verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
