#!/usr/bin/env bash
# Job "train-final" (Osirus runpod-job, gpu=B200x2@X.XX/any, long run with hours=N; owner order of 22.10.2026):
# finish the 1B, fine-tune it, measure it, then fine-tune the 10B if time is left. Every phase saves its result
# to the volume before the next one starts, so a run that ends early loses only the phase it was in.
#   0. setup:  JAX for CUDA, forge for the tokenizer
#   1. pull:   the train-1-f 1B (weights), the train-1 store (replay and validation) and the old 1B's export
#   2. data:   two corpus-v6 stores on the pod's CPUs (jobs/train-final-*-sources.json): pre-training (Stack-Edu with
#              free licences, FineMath-4+, OpenCoder's algorithmic code, verified reasoning traces) and fine-tuning
#              (OpenCodeInstruct solutions that pass all of their tests, traces, a little replay); decontaminated
#   p0. warm:  meanwhile the GPUs train the 1B on the train-1 store at a constant lr (second epoch, context 1024)
#   3. pt:     the 1B on the new pre-training store (re-warm, WSD decay to the end of its window)
#   4. sft:    fine-tuning of the 1B on the fine-tuning store, at most 2 epochs
#   5. eval:   validation loss of the old 1B, the train-1-f 1B and both new ones on the same split of the train-1
#              store; HumanEval, MBPP and their -hard variants with thinking for the new ones (GPU decoding)
#   6. 10b:    if at least TEN_MIN (230) minutes are left: the 10B (train-1-f bf16 export) fine-tuned on the same
#              store, then measured; only its bf16 export goes to the volume
#   7. done:   record and DONE.json to runs/$RUN_ID; the gateway pod ends itself once DONE.json is on the volume
# Nothing on the volume is deleted or overwritten (jobs/volsync.py refuses conflicts). Held-out sets are not used.
set -uo pipefail
OUT=$1; cd "$(dirname "$0")/.."
V=${VOL_ROOT:-/workspace}; RUN=${RUN_ID:-train-final}
T0=$(date +%s); DEADLINE=${POD_DEADLINE:-$(( T0 + 280 * 60 ))}
PY=${PY:-python3}; JPY=${JPY:-$PY}
SRC=${SRC_RUN:-train-1-f}; W="$V/runs/$RUN"; DPT="$V/data/final-pt"; DSFT="$V/data/final-sft"; D1="$V/data/train-1"
NCPU=$(nproc); DW=$(( NCPU > 8 ? NCPU - 4 : NCPU ))
SEQ=${SEQ:-1024}; BATCH=${BATCH:-64}; ACCUM=${ACCUM:-4}
mkdir -p "$W" "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
st() { python3 jobs/status.py "$OUT" "$@"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
vsync() { $PY jobs/volsync.py "$@" --root "$V"; }
runstatus() { python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" "$1" 2>/dev/null; }
final_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['final_ckpt'])" "$1" 2>/dev/null; }
export_of() { find "$1/export" -name 'weights-bf16-*.safetensors' 2>/dev/null | sort | tail -1; }
push() {  # files or directories (relative to $V or absolute) to the volume, in the background unless SYNC=1
  local rels=(); for p in "$@"; do rels+=("${p#"$V"/}"); done
  [ -n "${GW:-}" ] || return 0
  vsync push "${rels[@]}" --until $(( DEADLINE - 60 )) >> "$OUT/push.jsonl" 2>>"$OUT/push.err"
}
ckpt_files() { echo "$1/state.json" "$1/model.safetensors"; }   # no optimizer state: every phase starts AdamW anew
wait_for() {  # $1 file, $2 latest unix time: true once the file exists (the data build runs in the background)
  while [ ! -f "$1" ]; do
    kill -0 "$DATA_PID" 2>/dev/null || { [ -f "$1" ]; return; }
    [ "$(date +%s)" -lt "$2" ] || return 1
    sleep 30
  done
}
finish() {
  [ -n "${DATA_PID:-}" ] && { kill "$DATA_PID" 2>/dev/null; pkill -f build_corpus_v6 2>/dev/null; }
  st done "$1" "wall_min=$(( ($(date +%s) - T0) / 60 ))"
  mkdir -p "$W/record" && find "$OUT" -maxdepth 1 -type f -exec cp {} "$W/record/" \;
  python3 -c "import json,sys,time; print(json.dumps({'status': sys.argv[1], 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}))" "$1" > "$W/DONE.json"
  wait 2>/dev/null
  push "$W/record"; push "$W/DONE.json"   # DONE.json last: the gateway pod ends itself when it sees it
  exit "${2:-0}"
}
fail() { st "$1" failed "error=$2"; log "FAILED $1: $2"; finish failed 1; }
[ -n "${GW:-}" ] || fail plan "no gateway (GW); train-final runs with gpu=...@X.XX/any"
# a long run needs pods that can end themselves (jobs/pod_guard.sh, checked by the pod's boot script); without that
# the run stops here, after a few minutes, instead of spending hours of two GPUs on a schedule it cannot finish
[ "${LONG:-0}" = 1 ] && [ "${SELFTERM_OK:-0}" = 1 ] || fail plan "long run not available (LONG=${LONG:-0}, SELFTERM_OK=${SELFTERM_OK:-0})"
[ $(( (DEADLINE - T0) / 60 )) -ge 600 ] || fail plan "train-final needs at least 600 minutes (hours=N in JOB)"
st plan ok "deadline=$DEADLINE" "minutes=$(( (DEADLINE - T0) / 60 ))" "cpus=$NCPU" "seq=$SEQ" "source_run=$SRC" \
  "long_run=${LONG:-0}"

# 0. setup ------------------------------------------------------------------------------------------------------
t=$(date +%s); log setup
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
NGPU=$($JPY -c "import jax; print(jax.device_count())"); MESH=${MESH:-1x$NGPU}
st setup ok "jax=$ok" "gpus=$GPUINFO" "seconds=$(( $(date +%s) - t ))"

# fixed time after the pre-training: fine-tuning, measurements, the 10B block (if any), pushes, reserve
TOTAL_MIN=$(( $(left) / 60 ))
SFT_MIN=${SFT_MIN:-$([ "$TOTAL_MIN" -ge 900 ] && echo 120 || echo 90)}; EVAL_MIN=${EVAL_MIN:-75}; TEN_MIN=${TEN_MIN:-230}
TAIL_MIN=55
TEN=0; [ "$TOTAL_MIN" -ge $(( ${MIN_PT_FOR_TEN:-480} + SFT_MIN + EVAL_MIN + TEN_MIN + TAIL_MIN + 150 )) ] && TEN=1
PT_END=$(( DEADLINE - (SFT_MIN + EVAL_MIN + TAIL_MIN + TEN * TEN_MIN) * 60 ))
st schedule ok "total_min=$TOTAL_MIN" "ten=$TEN" "pt_end=$PT_END" "sft_min=$SFT_MIN" "eval_min=$EVAL_MIN"

# 1. pull -------------------------------------------------------------------------------------------------------
t=$(date +%s); log pull
vsync pull "runs/$SRC/1b/DONE.json" --require > /dev/null || fail pull "runs/$SRC/1b/DONE.json"
TF1=$(final_of "$V/runs/$SRC/1b/DONE.json"); TF1="$V/${TF1#*/workspace/}"
r=$(vsync pull "${TF1#"$V"/}" --name model.safetensors --name state.json --verify --require | tail -1) || fail pull "1B: $r"
s=$(vsync pull "${D1#"$V"/}/store" --verify --require | tail -1) || fail pull "train-1 store: $s"
o=$(vsync pull quasnir/run2/export --name 'weights-bf16-step007500.safetensors' | tail -1)
for d in "$DPT" "$DSFT"; do vsync pull "${d#"$V"/}/store" > /dev/null 2>&1; done   # stores of an earlier attempt
GWFREE=$(vsync health | $PY -c "import json,sys; print(json.load(sys.stdin).get('free', 0))")
st pull ok "seconds=$(( $(date +%s) - t ))" "init=${TF1#"$V"/}" "store=$s" "old=$o" "volume_free_gb=$(( GWFREE / 1000000000 ))"
BIGPUSH=0; [ "$GWFREE" -gt $(( ${VOL_MIN_FREE_GB:-100} * 1000000000 )) ] && BIGPUSH=1   # stores only with room to spare
[ "$TEN" = 1 ] && { vsync pull "runs/$SRC/10b/export" --verify > "$OUT/pull-10b.json" 2>&1 & TEN_PULL=$!; }

# 2. data (CPUs, in the background) ------------------------------------------------------------------------------
store_ok() {  # the store in $1 is complete: train.bin holds the tokens train.meta.json counts
  python3 - "$1/store" <<'PY' 2>/dev/null
import json, os, sys
m = json.load(open(os.path.join(sys.argv[1], "train.meta.json")))
sys.exit(0 if os.path.getsize(os.path.join(sys.argv[1], "train.bin")) == 2 * m["tokens"]
         and os.path.exists(os.path.join(sys.argv[1], "kept.npy")) else 1)
PY
}
build() {  # $1 data dir, $2 sources file
  local D=$1
  store_ok "$D" && return 0
  rm -rf "$D/store"   # an incomplete copy from an earlier attempt (local disk only)
  mkdir -p "$D/reasoning" && cp data/cache/reasoning/docs-train.jsonl "$D/reasoning/" || return 1
  sed "s#@D@#$D#g" "$2" > "$D/sources.json"
  [ -f "$D/plan.json" ] || $PY data/build_corpus_v6.py plan --out "$D" --config "$D/sources.json" --tiers A,C || return 1
  for w in $(seq 0 $(( DW - 1 ))); do
    $PY data/build_corpus_v6.py filter --out "$D" --worker "$w" --workers "$DW" --min-free-gb 20 >"$D/filter-$w.log" 2>&1 &
  done; wait
  [ -f "$D/dedup/drop.npy" ] || $PY data/build_corpus_v6.py dedup --out "$D" --partitions 32 >"$D/dedup.log" 2>&1 || return 1
  for w in $(seq 0 $(( DW - 1 ))); do
    $PY data/build_corpus_v6.py tokenize --out "$D" --worker "$w" --workers "$DW" >"$D/tok-$w.log" 2>&1 &
  done; wait
  $PY data/build_corpus_v6.py pack --out "$D" >"$D/pack.log" 2>&1
}
data_phase() {
  local t=$(date +%s)
  $PY -c "import sys; sys.path.insert(0, 'data'); import build_corpus as bc; bc.get_evals()" > /dev/null 2>&1 || { st data failed "error=eval sets"; return 1; }
  [ -s data/cache/reasoning/docs-train.jsonl ] || $PY data/build_reasoning.py --docs-only --synthetic-tokens "${REASONING_TOKENS:-1.5e8}" \
    --workers "$DW" > "$OUT/reasoning.log" 2>&1 || { st data failed "error=reasoning traces"; return 1; }
  for pair in "$DPT:jobs/train-final-pt-sources.json" "$DSFT:jobs/train-final-sft-sources.json"; do
    local D=${pair%%:*} name; name=$(basename "${pair%%:*}")
    build "$D" "${pair#*:}" || { st "data-$name" failed "error=$(cat "$D"/*.log 2>/dev/null | tail -c 300)"; return 1; }
    touch "$D/.ready"
    st "data-$name" ok "seconds=$(( $(date +%s) - t ))" \
      "tokens=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tokens'])" "$D/store/train.meta.json")" \
      "manifest=$(head -c 3000 "$D/store/manifest.json" 2>/dev/null)"
    [ "$BIGPUSH" = 1 ] && push "$D/store" &
  done
  wait
}
data_phase > "$OUT/data.log" 2>&1 &
DATA_PID=$!

train() {  # $1 name, $2 init, $3 store dir, $4 deadline, $5 lr, $6 min lr, [$7 step cap]
  local out="$W/$1" t=$(date +%s)
  case "$(runstatus "$out/DONE.json")" in ok|deadline) return 0;; esac
  $JPY -m accel.run_s2 --data "$3/store/train.meta.json" --clean-tsv "$3/store/kept.npy" --seq "$SEQ" --batch "$BATCH" \
    --accum "$ACCUM" --mesh "$MESH" --attn "${ATTN:-cudnn}" --precision bf16 --cap "${7:-1000000}" \
    --init "$2" --out "$out" --deadline "$4" --reserve-s "${RESERVE:-300}" --lr "$5" --min-lr "$6" \
    > "$OUT/$1.jsonl" 2>"$OUT/$1.err"
  case "$(runstatus "$out/DONE.json")" in ok|deadline) ;; *) return 1;; esac
  st "$1" "$(runstatus "$out/DONE.json")" "seconds=$(( $(date +%s) - t ))" --json "$OUT/$1.jsonl"
  local fin; fin=$(final_of "$out/DONE.json")
  case "${PUSH_MODE:-all}" in   # all: weights (no AdamW) and export; export: bf16 export only; none: stays on the pod
    all) push $(ckpt_files "$fin") "$out/export" "$out/DONE.json" &;;
    export) push "$out/export" "$out/DONE.json" &;;
  esac
}
epochs_cap() {  # steps for $2 epochs of the store in $1
  python3 -c "import json,sys,math; t=json.load(open(sys.argv[1]))['tokens']; print(max(50, math.ceil(float(sys.argv[2]) * t / ($BATCH * $ACCUM * $SEQ))))" "$1/store/train.meta.json" "$2"
}

# p0. warm: the train-1 store at a constant lr while the data is built --------------------------------------------
P0_END=$(( $(date +%s) + ${DATA_EST_MIN:-150} * 60 ))
[ "$P0_END" -lt $(( PT_END - 3600 )) ] || P0_END=$(( PT_END - 3600 ))
INIT=$TF1
if [ "$P0_END" -gt $(( $(date +%s) + 1200 )) ]; then
  log p0
  if PUSH_MODE=none train p0-warm "$TF1" "$D1" "$P0_END" "${LR_PT:-1e-4}" "${LR_PT:-1e-4}"; then INIT=$(final_of "$W/p0-warm/DONE.json")
  else st p0-warm failed "error=$(tail -c 300 "$OUT/p0-warm.err")"; fi
fi
log "wait for the pre-training store"
wait_for "$DPT/.ready" $(( PT_END - 1800 )) || fail data "pre-training store missing ($(tail -c 400 "$OUT/data.log"))"

# 3. pt -----------------------------------------------------------------------------------------------------------
log pt
[ "$PT_END" -gt $(( $(date +%s) + 1800 )) ] || PT_END=$(( $(date +%s) + 1800 ))
train pt-1b "$INIT" "$DPT" "$PT_END" "${LR_PT:-1e-4}" "${MINLR_PT:-1e-5}" || fail pt-1b "$(tail -c 400 "$OUT/pt-1b.err")"
PT1=$(final_of "$W/pt-1b/DONE.json")

# 4. sft ----------------------------------------------------------------------------------------------------------
SFT1=""
if wait_for "$DSFT/.ready" $(( $(date +%s) + 600 )); then
  log sft
  if train sft-1b "$PT1" "$DSFT" $(( $(date +%s) + SFT_MIN * 60 )) "${LR_SFT:-3e-5}" "${MINLR_SFT:-3e-6}" "$(epochs_cap "$DSFT" 2)"; then
    SFT1=$(final_of "$W/sft-1b/DONE.json")
  else st sft-1b failed "error=$(tail -c 300 "$OUT/sft-1b.err")"; fi
else
  st sft-1b skipped "reason=fine-tuning store missing"
fi

# 5. eval ---------------------------------------------------------------------------------------------------------
log eval
mkdir -p data/stores/evals && cp data/out/evals/*.jsonl data/stores/evals/ 2>/dev/null
OLD=$(find "$V/quasnir/run2/export" -name 'weights-bf16-step007500.safetensors' 2>/dev/null | head -1)
[ -n "$OLD" ] && $PY scripts/import_export.py "$OLD" "$V/ckpt/final-old-1b" --max-seq-len "$SEQ" > /dev/null 2>&1
declare -A CK=([old-1b]="$V/ckpt/final-old-1b" [tf-1b]="$TF1" [pt-1b]="$PT1" [sft-1b]="$SFT1")
for m in old-1b tf-1b pt-1b sft-1b; do
  [ -n "${CK[$m]}" ] && [ -f "${CK[$m]}/model.safetensors" ] || continue
  timeout 900 $JPY -m accel.eval --ckpt "${CK[$m]}" --data "$D1/store/train.meta.json" --split val --batch 8 --seq 512 \
    --batches 40 --precision bf16 > "$OUT/val-$m.json" 2>"$OUT/val-$m.err" \
    && st "val-$m" ok "result=$(tail -1 "$OUT/val-$m.json")" || st "val-$m" failed "error=$(tail -c 300 "$OUT/val-$m.err")"
done
THINK=${THINK:---samples 8 --rounds 2 --self-tests 4 --repair --oracle}
bench() {  # $1 name, $2 ckpt, $3 end (unix time), [$4 decode batch]
  local GEN_CMD="$JPY -m accel.generate --batch ${4:-128} --precision bf16"; export GEN_CMD
  for s in humaneval mbpp humaneval-hard mbpp-hard; do
    [ $(( $3 - $(date +%s) )) -gt 300 ] || { st "bench-$1-$s" skipped "reason=time"; continue; }
    timeout $(( $3 - $(date +%s) )) $PY scripts/humaneval.py "$2" --suite "$s" --tokenizer data/tokenizer.json $THINK \
      --jobs "$NCPU" --cache-dir "$OUT/gens-$1" > "$OUT/bench-$1-$s.json" 2>"$OUT/bench-$1-$s.err" \
      && st "bench-$1-$s" ok "result=$(tail -1 "$OUT/bench-$1-$s.json")" \
      || st "bench-$1-$s" failed "error=$(tail -c 300 "$OUT/bench-$1-$s.err")"
  done
}
EV_END=$(( $(date +%s) + (EVAL_MIN - 15) * 60 ))
for m in sft-1b pt-1b; do
  [ -n "${CK[$m]}" ] || continue
  exp=$(export_of "$W/$m"); [ -n "$exp" ] || continue
  $PY scripts/import_export.py "$exp" "$V/ckpt/final-$m" --max-seq-len "$SEQ" > /dev/null 2>&1 || continue
  bench "$m" "$V/ckpt/final-$m" "$EV_END"
done

# 6. 10b ----------------------------------------------------------------------------------------------------------
if [ "$TEN" = 1 ] && [ -f "$DSFT/.ready" ] && [ "$(left)" -gt $(( (TEN_MIN + TAIL_MIN - 10) * 60 )) ]; then
  t=$(date +%s); log 10b
  [ -n "${TEN_PULL:-}" ] && wait "$TEN_PULL"; r=$(tail -1 "$OUT/pull-10b.json" 2>/dev/null)
  TEN_EXP=$(export_of "$V/runs/$SRC/10b")
  if [ -n "$TEN_EXP" ] && $PY scripts/import_export.py "$TEN_EXP" "$V/ckpt/final-10b-init" --max-seq-len "$SEQ" > "$OUT/import-10b.json" 2>&1; then
    st 10b-init ok "seconds=$(( $(date +%s) - t ))" "pull=$r"
    T10_END=$(( DEADLINE - (TAIL_MIN + 70) * 60 ))
    RESERVE=1500 PUSH_MODE=export train sft-10b "$V/ckpt/final-10b-init" "$DSFT" "$T10_END" "${LR_SFT10:-2e-5}" "${MINLR_SFT10:-2e-6}" \
      "$(epochs_cap "$DSFT" 1)" || st sft-10b failed "error=$(tail -c 300 "$OUT/sft-10b.err")"
    exp=$(export_of "$W/sft-10b")
    if [ -n "$exp" ] && $PY scripts/import_export.py "$exp" "$V/ckpt/final-sft-10b" --max-seq-len "$SEQ" > /dev/null 2>&1; then
      bench sft-10b "$V/ckpt/final-sft-10b" $(( DEADLINE - (TAIL_MIN + 5) * 60 )) 32
    fi
  else
    st 10b-init failed "pull=$r" "error=$(tail -c 300 "$OUT/import-10b.json")"
  fi
else
  st 10b skipped "reason=time or data (ten=$TEN, left_min=$(( $(left) / 60 )))"
fi

# 7. done ---------------------------------------------------------------------------------------------------------
finish ok 0
