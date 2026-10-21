#!/usr/bin/env bash
# Job "eval-1" (runpod-job workflow in Osirus, one GPU, gpu=...@X.XX/any with the gateway at the volume). Measures what
# run train-1-f produced, before POD_DEADLINE:
#   1. setup:   JAX for CUDA, forge for the tokenizer
#   2. pull:    the run's record (logs), the exports of the old 1B (run2 step 7500), the new 1B and the 10B, and the
#               data store, through the gateway (jobs/volsync.py)
#   3. curves:  train loss every 10 steps from the run's logs (1B and 10B), and val loss of all three models on the
#               same validation split of the store (accel.eval, forge's windows)
#   4. bench:   HumanEval, MBPP and their -hard variants with thinking (scripts/humaneval.py: greedy + samples over
#               rounds, self-tests, repair, selection by visible tests; --oracle gives the any-candidate ceiling that
#               the RSI rule reads), decoded on the GPU (accel.generate); new 1B and 10B first, the old 1B last
#   5. push:    OUT (results, logs) to runs/$RUN_ID/record on the volume
# Only exports and logs are read; nothing on the volume is changed or deleted. Held-out sets are not used here.
set -uo pipefail
OUT=$1; cd "$(dirname "$0")/.."
V=${VOL_ROOT:-/workspace}; RUN=${RUN_ID:-eval-1}; SRC=${SRC_RUN:-train-1-f}
T0=$(date +%s); DEADLINE=${POD_DEADLINE:-$(( T0 + 120 * 60 ))}
PY=${PY:-python3}; JPY=${JPY:-$PY}
mkdir -p "$OUT" "$V/ckpt"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
st() { python3 jobs/status.py "$OUT" "$@"; }
fail() { st "$1" failed "error=$2"; log "FAILED $1: $2"; exit 1; }
vsync() { $PY jobs/volsync.py "$@" --root "$V"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
[ -n "${GW:-}" ] || fail plan "no gateway (GW); eval-1 runs with gpu=...@X.XX/any"
st plan ok "deadline=$DEADLINE" "minutes=$(( (DEADLINE - T0) / 60 ))" "cpus=$(nproc)" "source_run=$SRC"

# 1. setup ------------------------------------------------------------------------------------------------------
if [ -z "${SKIP_SETUP:-}" ]; then
  log setup
  $PY -m pip install -q numpy pyarrow zstandard ml_dtypes >/dev/null 2>&1
  smoke() { $JPY - <<'PY'
import jax, jax.numpy as jnp
devs = jax.devices()
assert devs and devs[0].platform == "gpu", devs
x = jax.device_put(jnp.ones((4096, 4096), jnp.bfloat16), devs[0])
assert float((x @ x)[0, 0]) == 4096.0
print(len(devs), devs[0].device_kind)
PY
  }
  ok=""
  for spec in "jax[cuda13]" "jax[cuda12]"; do
    $PY -m pip install -q -U "$spec" >/dev/null 2>&1 && GPUINFO=$(smoke 2>/dev/null) && { ok=$spec; break; }
  done
  [ -n "$ok" ] || fail setup "no JAX build runs on this GPU ($(nvidia-smi -L 2>&1 | head -1))"
  if ! [ -x target/release/forge ]; then
    command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal; . "$HOME/.cargo/env"; }
    cargo build --release -q -p forge || fail setup "forge build"
  fi
  st setup ok "jax=$ok" "gpus=$GPUINFO" "seconds=$(( $(date +%s) - T0 ))"
fi

# 2. pull -------------------------------------------------------------------------------------------------------
t=$(date +%s); log pull
rec=$(vsync pull "runs/$SRC/record" | tail -1)
new=$(vsync pull "runs/$SRC/1b/export" --verify --require | tail -1) || fail pull "new 1B export: $new"
ten=$(vsync pull "runs/$SRC/10b/export" --verify | tail -1)
old=$(vsync pull quasnir/run2/export --name 'weights-bf16-step007500.safetensors' --name EXPORT.json | tail -1)
store=$(vsync pull data/train-1/store | tail -1)
st pull ok "seconds=$(( $(date +%s) - t ))" "record=$rec" "new=$new" "ten=$ten" "old=$old" "store=$store"
declare -A EXP=(
  [new-1b]=$(find "$V/runs/$SRC/1b/export" -name 'weights-bf16-*.safetensors' 2>/dev/null | sort | tail -1)
  [10b]=$(find "$V/runs/$SRC/10b/export" -name 'weights-bf16-*.safetensors' 2>/dev/null | sort | tail -1)
  [old-1b]=$(find "$V/quasnir/run2/export" -name 'weights-bf16-step007500.safetensors' 2>/dev/null | head -1))
MODELS=""
for m in new-1b 10b old-1b; do
  [ -n "${EXP[$m]}" ] || { st "import-$m" skipped "reason=export not on the volume"; continue; }
  $PY scripts/import_export.py "${EXP[$m]}" "$V/ckpt/eval-$m" --max-seq-len 512 > "$OUT/import-$m.json" 2>&1 \
    && MODELS="$MODELS $m" || st "import-$m" failed "error=$(tail -c 300 "$OUT/import-$m.json")"
done
[ -n "$MODELS" ] || fail pull "no model to measure"

# 3. curves -----------------------------------------------------------------------------------------------------
$PY - "$V/runs/$SRC/record" > "$OUT/curves.json" <<'PY'
import json, sys
from pathlib import Path
out = {}
for name in ("1b", "10b"):
    p = Path(sys.argv[1]) / f"{name}.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text(errors="replace").splitlines() if l.startswith("{")]
    steps = [r for r in rows if r.get("event") == "step"]
    pick = steps[:: max(1, len(steps) // 40)] + steps[-1:]
    out[name] = {"steps_logged": len(steps), "start": next((r for r in rows if r.get("event") == "start"), None),
                 "plan": next((r for r in rows if r.get("event") == "plan"), None),
                 "curve": [[r["step"], r["loss"]] for r in pick],
                 "first_loss": steps[0]["loss"] if steps else None, "last_loss": steps[-1]["loss"] if steps else None,
                 "mean_last_50": round(sum(r["loss"] for r in steps[-5:]) / max(1, len(steps[-5:])), 5) if steps else None}
print(json.dumps(out))
PY
st curves ok "result=$(cat "$OUT/curves.json")"
for m in $MODELS; do
  [ "$(left)" -gt 1800 ] || break
  timeout 900 $JPY -m accel.eval --ckpt "$V/ckpt/eval-$m" --data "$V/data/train-1/store/train.meta.json" --split val \
    --batch 8 --seq 512 --batches 40 --precision bf16 > "$OUT/val-$m.json" 2>"$OUT/val-$m.err" \
    && st "val-$m" ok "result=$(tail -1 "$OUT/val-$m.json")" || st "val-$m" failed "error=$(tail -c 300 "$OUT/val-$m.err")"
done

# 4. bench (thinking, decoded on the GPU) -----------------------------------------------------------------------
mkdir -p data/stores/evals && cp data/out/evals/*.jsonl data/stores/evals/ 2>/dev/null
$PY -c "import sys; sys.path.insert(0, 'data'); import build_corpus as bc; bc.get_evals()" >/dev/null 2>&1
mkdir -p data/stores/evals && cp data/out/evals/*.jsonl data/stores/evals/ 2>/dev/null
export GEN_CMD="$JPY -m accel.generate --batch ${GEN_BATCH:-128} --precision bf16"
THINK=${THINK:---samples 8 --rounds 2 --self-tests 4 --repair --oracle}
for m in $MODELS; do
  for s in humaneval mbpp humaneval-hard mbpp-hard; do
    [ "$(left)" -gt 600 ] || { st "bench-$m-$s" skipped "reason=deadline"; continue; }
    t=$(date +%s)
    timeout $(( $(left) - 300 )) $PY scripts/humaneval.py "$V/ckpt/eval-$m" --suite "$s" --tokenizer data/tokenizer.json \
      $THINK --jobs "$(nproc)" --cache-dir "$OUT/gens-$m" > "$OUT/bench-$m-$s.json" 2>"$OUT/bench-$m-$s.err" \
      && st "bench-$m-$s" ok "seconds=$(( $(date +%s) - t ))" "result=$(tail -1 "$OUT/bench-$m-$s.json")" \
      || st "bench-$m-$s" failed "seconds=$(( $(date +%s) - t ))" "error=$(tail -c 300 "$OUT/bench-$m-$s.err")"
  done
done

# 5. push the record --------------------------------------------------------------------------------------------
st done ok "wall_min=$(( ($(date +%s) - T0) / 60 ))"
mkdir -p "$V/runs/$RUN/record" && find "$OUT" -maxdepth 1 -type f -exec cp {} "$V/runs/$RUN/record/" \;
vsync push "runs/$RUN/record" --until $(( DEADLINE - 60 )) > "$OUT/push.json" 2>&1
st push "$( [ $? = 0 ] && echo ok || echo partial)" "result=$(tail -1 "$OUT/push.json")"
