#!/usr/bin/env bash
# Job "train-1" (runpod-job workflow in Osirus, GPU pod with the network volume at /workspace). One run that ends
# before POD_DEADLINE (unix time, set by the runner; at most 300 min after the pod starts):
#   1. setup: JAX for CUDA (pinned, with fallbacks), forge for the tokenizer, a smoke test on every GPU
#   2. data:  corpus v6 store from jobs/train-1-sources.json (verified reasoning traces, a slice of Stack-Edu,
#             FineMath-4+), filtered, decontaminated, deduplicated; kept on the volume, so a second run reuses it
#   3. 1b:    premium tokens on quasnir/run2 step 7500 (accel.run_s2, re-warm and WSD decay)
#   4. grow:  the new 1B to about 10B (scripts/grow.py, function-preserving)
#   5. 10b:   training until the deadline minus the time for the final checkpoint and export
#   bench:    meanwhile the pod's CPUs run the public benchmarks (forge, greedy) of the old and the new 1B
# OUT/result.json is rewritten after every phase. A phase whose output already exists is skipped. Nothing on the
# volume is deleted or overwritten. Held-out sets are not used on the pod.
set -uo pipefail
OUT=$1; cd "$(dirname "$0")/.."
V=${VOL_ROOT:-/workspace}; RUN=${RUN_NAME:-train-1}
T0=$(date +%s); DEADLINE=${POD_DEADLINE:-$(( T0 + 280 * 60 ))}
PY=${PY:-python3}; JPY=${JPY:-$PY}
W="$V/runs/$RUN"; D="$V/data/$RUN"; G="$V/ckpt/$RUN-10b-grown"
INIT_1B=${INIT_1B:-$V/quasnir/run2/step-007500}
NCPU=$(nproc)
mkdir -p "$W" "$D" "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
st() { python3 jobs/status.py "$OUT" "$@"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
fail() { st "$1" failed "error=$2"; log "FAILED $1: $2"; exit 1; }
runstatus() { python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" "$1" 2>/dev/null; }
dims() {  # grown dim, ffn and layers for the 1B in state.json $1 (head grouping kept, about 10B)
  $PY - "$1" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["config"]["model"]
step = (m["dim"] // m["n_heads"]) * (m["n_heads"] // m["n_kv_heads"])
dim = max(m["dim"], round(4096 / step) * step)
print(dim, max(m["ffn_hidden"], round(dim * 3.5 / 256) * 256), max(m["n_layers"], 44))
PY
}
st plan ok "deadline=$DEADLINE" "minutes=$(( (DEADLINE - T0) / 60 ))" "cpus=$NCPU" "volume_free_gb=$(df -BG --output=avail "$V" | tail -1 | tr -dc 0-9)"

# 1. setup ------------------------------------------------------------------------------------------------------
if [ -z "${SKIP_SETUP:-}" ]; then
  log setup
  $PY -m pip install -q numpy pyarrow zstandard ml_dtypes >/dev/null 2>&1
  smoke() { $JPY - <<'PY'
import jax, jax.numpy as jnp
devs = jax.devices()
assert devs and devs[0].platform == "gpu", devs
for d in devs:
    x = jax.device_put(jnp.ones((4096, 4096), jnp.bfloat16), d)
    assert float((x @ x)[0, 0]) == 4096.0
print(len(devs), devs[0].device_kind)
PY
  }
  ok=""
  for spec in "jax[cuda12]==0.11.2" "jax[cuda12]" "jax[cuda13]"; do
    $PY -m pip install -q -U "$spec" >/dev/null 2>&1 && GPUINFO=$(smoke 2>/dev/null) && { ok=$spec; break; }
  done
  [ -n "$ok" ] || fail setup "no JAX build runs on these GPUs ($(nvidia-smi -L 2>&1 | head -2 | tr '\n' ' '))"
  if ! [ -x target/release/forge ]; then
    command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal; . "$HOME/.cargo/env"; }
    cargo build --release -q -p forge || fail setup "forge build"
  fi
  st setup ok "jax=$ok" "gpus=$GPUINFO" "seconds=$(( $(date +%s) - T0 ))"
fi
NGPU=$($JPY -c "import jax; print(jax.device_count())")
MESH=${MESH:-1x$NGPU}

# 2. data -------------------------------------------------------------------------------------------------------
if [ ! -f "$D/store/train.meta.json" ]; then
  t=$(date +%s); log data
  $PY -c "import sys; sys.path.insert(0, 'data'); import build_corpus as bc; bc.get_evals()" || fail data "eval sets"
  if [ ! -s "$D/reasoning/docs-train.jsonl" ]; then
    $PY data/build_reasoning.py --docs-only --synthetic-tokens "${REASONING_TOKENS:-1.5e8}" --workers "$NCPU" >"$OUT/reasoning.log" 2>&1 \
      || fail data "reasoning traces"
    mkdir -p "$D/reasoning" && cp data/cache/reasoning/docs-train.jsonl "$D/reasoning/"
  fi
  sed "s#@D@#$D#g" "${TRAIN_SOURCES:-jobs/train-1-sources.json}" > "$D/sources.json"
  [ -f "$D/plan.json" ] || $PY data/build_corpus_v6.py plan --out "$D" --config "$D/sources.json" --tiers A,C || fail data plan
  for w in $(seq 0 $(( NCPU - 1 ))); do
    $PY data/build_corpus_v6.py filter --out "$D" --worker "$w" --workers "$NCPU" --min-free-gb "${MIN_FREE_GB:-5}" >"$D/filter-$w.log" 2>&1 &
  done; wait
  $PY data/build_corpus_v6.py dedup --out "$D" --partitions 32 >"$OUT/dedup.log" 2>&1 || fail data dedup
  for w in $(seq 0 $(( NCPU - 1 ))); do
    $PY data/build_corpus_v6.py tokenize --out "$D" --worker "$w" --workers "$NCPU" >"$D/tok-$w.log" 2>&1 &
  done; wait
  $PY data/build_corpus_v6.py pack --out "$D" >"$OUT/pack.log" 2>&1 || fail data pack
  TOK=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tokens'])" "$D/store/train.meta.json")
  [ "$TOK" -gt 0 ] || fail data "empty store ($(tail -c 300 "$D/filter-0.log"))"
  st data ok "seconds=$(( $(date +%s) - t ))" "store=$(cat "$D/store/manifest.json" 2>/dev/null | head -c 3000)"
fi
TRAIN="--data $D/store/train.meta.json --clean-tsv $D/store/kept.npy --seq ${SEQ:-512} --batch ${BATCH:-128} --accum ${ACCUM:-4}"
TRAIN="$TRAIN --mesh $MESH --attn ${ATTN:-cudnn} --precision ${PRECISION:-bf16} --cap 1000000"

# bench (CPU, in the background while the GPUs train) ------------------------------------------------------------
bench() {  # $1 name, $2 export safetensors
  local ck="$V/ckpt/bench-$1"
  [ -f "$ck/model.safetensors" ] || $PY scripts/import_export.py "$2" "$ck" --max-seq-len 512 >/dev/null || return
  for s in humaneval mbpp humaneval-hard mbpp-hard; do
    nice -n 10 $PY scripts/humaneval.py "$ck" --suite "$s" --tokenizer data/tokenizer.json --cache-dir "$OUT/gens-$1" \
      > "$OUT/bench-$1-$s.json" 2>/dev/null && st "bench-$1-$s" ok "result=$(cat "$OUT/bench-$1-$s.json")"
  done
}
if [ -z "${SKIP_BENCH:-}" ]; then
  mkdir -p data/stores/evals && cp data/out/evals/*.jsonl data/stores/evals/ 2>/dev/null
  OLD=$(find "$V/quasnir/run2/export" -name 'weights-bf16-step007500.safetensors' 2>/dev/null | head -1)
  [ -n "$OLD" ] && { bench old-1b "$OLD" > "$OUT/bench-old.log" 2>&1 & BENCH_PIDS="${BENCH_PIDS:-} $!"; }
fi

# room for the 10B? grown f32 (4 B/param) + its final checkpoint with AdamW (12 B) + bf16 export (2 B) + 15 GB margin
read -r DIM FFN LAYERS < <(dims "$INIT_1B/state.json")
DIM=${GROW_DIM:-$DIM}; FFN=${GROW_FFN:-$FFN}; LAYERS=${GROW_LAYERS:-$LAYERS}
N10=$($PY scripts/grow.py "$INIT_1B" "$OUT/.dry" --dim "$DIM" --ffn-hidden "$FFN" --n-layers "$LAYERS" --dry-run \
      | $PY -c "import json,sys; print(json.load(sys.stdin)['grown_params'])")
PER=18; [ -f "$G/model.safetensors" ] && PER=14
NEED=$(( N10 * PER + 15000000000 )); FREE=$(df -B1 --output=avail "$V" | tail -1 | tr -dc 0-9)
TEN=1; WHY=
if [ "$FREE" -lt "$NEED" ]; then
  TEN=0; KEEP_10B_MIN=0; WHY=storage
  st storage short "free_gb=$(( FREE / 1000000000 ))" "need_gb=$(( NEED / 1000000000 ))" "action=1B only, no growth"
fi
# does the 10B fit on these GPUs? f32 params, grads and AdamW (16 B/param) sharded over the GPUs + 30 GB activations,
# at most 90 % of one GPU (one B200: no; two: yes)
GMIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
PERGPU=$(( N10 * 16 / NGPU + 30000000000 ))
if [ "$TEN" = 1 ] && [ -n "$GMIB" ] && [ "$PERGPU" -gt $(( GMIB * 1048576 * 9 / 10 )) ]; then
  TEN=0; KEEP_10B_MIN=0; WHY=gpu-memory
  st gpu-memory short "gpus=$NGPU" "gpu_gb=$(( GMIB * 1048576 / 1000000000 ))" \
    "need_gb_per_gpu=$(( PERGPU / 1000000000 ))" "action=1B only, no growth"
fi

# 3. 1B premium tokens -------------------------------------------------------------------------------------------
B1="$W/1b"
case "$(runstatus "$B1/DONE.json")" in ok|deadline) ;; *)
  t=$(date +%s); log 1b
  END1=$(( t + ${P1B_MIN:-60} * 60 ))
  KEEP=$(( ${KEEP_10B_MIN:-150} * 60 ))  # time kept for grow + 10B
  [ "$TEN" = 1 ] || END1=$(( DEADLINE - 60 ))
  [ "$END1" -lt $(( DEADLINE - KEEP )) ] || END1=$(( DEADLINE - KEEP ))
  $JPY -m accel.run_s2 $TRAIN --init "$INIT_1B" --out "$B1" --deadline "$END1" --reserve-s "${RESERVE_1B:-300}" \
    --lr "${LR_1B:-1e-4}" --min-lr "${MINLR_1B:-1e-5}" > "$OUT/1b.jsonl" 2>"$OUT/1b.err"
  case "$(runstatus "$B1/DONE.json")" in ok|deadline) ;; *) fail 1b "$(tail -c 400 "$OUT/1b.jsonl") $(tail -c 400 "$OUT/1b.err")";; esac
  st 1b "$(runstatus "$B1/DONE.json")" \
    "seconds=$(( $(date +%s) - t ))" --json "$OUT/1b.jsonl"
esac
NEW1B=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['final_ckpt'])" "$B1/DONE.json")
NEWEXP=$(find "$B1/export" -name 'weights-bf16-*.safetensors' 2>/dev/null | sort | tail -1)
if [ -z "${SKIP_BENCH:-}" ] && [ -n "$NEWEXP" ]; then bench new-1b "$NEWEXP" > "$OUT/bench-new.log" 2>&1 & BENCH_PIDS="${BENCH_PIDS:-} $!"; fi

# 4. grow to about 10B -------------------------------------------------------------------------------------------
if [ "$TEN" = 1 ] && [ ! -f "$G/model.safetensors" ]; then
  t=$(date +%s); log grow
  $PY scripts/grow.py "$NEW1B" "$G" --dim "$DIM" --ffn-hidden "$FFN" --n-layers "$LAYERS" > "$OUT/grow.json" || fail grow "grow.py"
  st grow ok "seconds=$(( $(date +%s) - t ))" "result=$(cat "$OUT/grow.json")"
fi

# 5. 10B -------------------------------------------------------------------------------------------------------
B10="$W/10b"
[ "$TEN" = 1 ] || st 10b skipped "reason=$WHY"
[ "$TEN" = 1 ] && case "$(runstatus "$B10/DONE.json")" in ok|deadline) ;; *)
  t=$(date +%s); log 10b
  $JPY -m accel.run_s2 $TRAIN --init "$G" --out "$B10" --deadline "$(( DEADLINE - 60 ))" --reserve-s "${RESERVE_10B:-1500}" \
    --lr "${LR_10B:-6e-5}" --min-lr "${MINLR_10B:-1e-5}" > "$OUT/10b.jsonl" 2>"$OUT/10b.err"
  case "$(runstatus "$B10/DONE.json")" in ok|deadline) ;; *) fail 10b "$(tail -c 400 "$OUT/10b.jsonl") $(tail -c 400 "$OUT/10b.err")";; esac
  st 10b "$(runstatus "$B10/DONE.json")" \
    "seconds=$(( $(date +%s) - t ))" --json "$OUT/10b.jsonl"
esac
[ -n "${BENCH_PIDS:-}" ] && kill $BENCH_PIDS 2>/dev/null; pkill -f scripts/humaneval.py 2>/dev/null; pkill -f "forge generate" 2>/dev/null
wait 2>/dev/null
st done ok "wall_min=$(( ($(date +%s) - T0) / 60 ))" "volume_free_gb=$(df -BG --output=avail "$V" | tail -1 | tr -dc 0-9)"
