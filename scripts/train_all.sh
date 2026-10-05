#!/usr/bin/env bash
# Full pipeline: shared base θ0 → Rouge 1 (general) and Quasnir (code/security),
# both continued from θ0 → Darus = TIES(θ0; Rouge 1, Quasnir) → evaluation of all
# four checkpoints on both validation sets. Every number lands in runs/report.json.
set -euo pipefail
cd "$(dirname "$0")/.."
FORGE=${FORGE:-target/release/forge}
cargo build --release -p forge
final() { # link the newest checkpoint of a run as FINAL
  local dir; dir=$(python3 -c "import json;print(json.load(open('runs/$1/latest.json'))['dir'])")
  ln -sfn "$(realpath "$dir")" "runs/$1/FINAL"
}
train() { "$FORGE" train --config "training/configs/$1.json" < /dev/null | tee "runs/$1.jsonl"; final "$1"; }
mkdir -p runs
[ -e runs/base-s/FINAL ] || train base-s
[ -e runs/rouge-1/FINAL ] || train rouge-1
[ -e runs/quasnir-1/FINAL ] || train quasnir-1
M=training/configs/darus-1.merge.json
read -r BASE C1 C2 OUT DENS LAM < <(python3 -c "import json;m=json.load(open('$M'));print(m['base'],*m['children'],m['out'],m['density'],m['lambda'])")
"$FORGE" merge --base "$BASE" --child "$C1" --child "$C2" --method ties --density "$DENS" --lambda "$LAM" --out "$OUT" | tee runs/darus-1.jsonl
python3 scripts/report.py
