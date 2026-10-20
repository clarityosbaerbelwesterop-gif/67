#!/usr/bin/env bash
# Job "bench-1b" (runpod-job workflow in Osirus), Auftrag 1 with the owner's yes of 2026-10-20: public benchmarks of
# the s2-run2 step-7500 export. Greedy on HumanEval, MBPP and the hard variants, then thinking (best-of-8, a second
# round, self-tests, repair, oracle) while the time budget lasts. result.json is rewritten after every suite, so the
# runner gets partial results at its deadline. Held-out sets are not used here; the weights stay on the volume.
set -uo pipefail
OUT=$1; cd "$(dirname "$0")/.."
. jobs/common.sh
T0=$(date +%s); LIMIT=$(( ${BENCH_MAX_MIN:-200} * 60 ))
left() { echo $(( LIMIT - ($(date +%s) - T0) )); }
collect() {
  python3 - "$OUT" "$(( $(date +%s) - T0 ))" <<'PY'
import json, sys
from pathlib import Path
out, secs = Path(sys.argv[1]), int(sys.argv[2])
suites = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("s_*.json")) if p.stat().st_size}
(out / "result.json").write_text(json.dumps({"seconds": secs, "suites": suites}, indent=1))
PY
}
ensure_1b_ckpt "$OUT" || exit $?
if ! [ -x target/release/forge ]; then
  command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal; . "$HOME/.cargo/env"; }
  cargo build --release -q -p forge || exit 5
fi
python3 -c "import sys; sys.path.insert(0, 'data'); import build_corpus as bc; bc.get_evals()" || exit 6
mkdir -p data/stores/evals && cp data/out/evals/*.jsonl data/stores/evals/
H="python3 scripts/humaneval.py $SRC --tokenizer data/tokenizer.json --cache-dir /root/gens"
for s in humaneval mbpp humaneval-hard mbpp-hard; do
  $H --suite $s > "$OUT/s_$s.json"; collect
done
T="--samples 8 --rounds 2 --self-tests 8 --repair --oracle"
t1=$(date +%s)
[ "$(left)" -gt 1800 ] && { $H --suite humaneval $T > "$OUT/s_humaneval-think.json"; collect; }
think=$(( $(date +%s) - t1 ))
[ "$(left)" -gt 300 ] && [ -s "$OUT/s_humaneval-think.json" ] && { $H --suite humaneval-hard $T > "$OUT/s_humaneval-hard-think.json"; collect; }
# MBPP has 500 instead of 164 tasks: about 3x the HumanEval thinking time
if [ -s "$OUT/s_humaneval-think.json" ] && [ "$(left)" -gt $(( think * 3 + 600 )) ]; then
  $H --suite mbpp $T > "$OUT/s_mbpp-think.json"; collect
  $H --suite mbpp-hard $T > "$OUT/s_mbpp-hard-think.json"; collect
fi
collect
