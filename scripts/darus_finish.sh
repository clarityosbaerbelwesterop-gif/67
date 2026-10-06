#!/usr/bin/env bash
# Completes Darus after scripts/train_all.sh: evolutionary merge search
# (scripts/merge_search.py), the four-model report, then HumanEval and MBPP
# pass@1 with real, sandboxed execution for every model. Every number lands in
# runs/: darus-1/search.json, report.json, benchmarks.jsonl.
#   scripts/darus_finish.sh [PID to wait for, e.g. the running train_all.sh]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "${1:-}" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
for r in base-s rouge-1 quasnir-1 darus-1; do
  [ -e "runs/$r/FINAL" ] || { echo "runs/$r/FINAL missing" >&2; exit 1; }
done
python3 scripts/merge_search.py > runs/darus-1-search.log
python3 scripts/report.py | tee runs/report.txt
: > runs/benchmarks.jsonl
for m in darus-1 quasnir-1 rouge-1 base-s; do
  for s in humaneval mbpp; do
    python3 scripts/humaneval.py "runs/$m/FINAL" --suite "$s" | tee -a runs/benchmarks.jsonl
  done
done
echo '{"type":"finished"}' >> runs/benchmarks.jsonl
