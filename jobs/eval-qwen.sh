#!/usr/bin/env bash
# Job "eval-qwen" (runpod-job workflow in Osirus, 1-2 GPUs, gpu=...@X.XX/any with the gateway at the volume). Step 1 of
# the Qwen path (owner approval 22.10.2026): measure Qwen3.8-27B as published, before any fine-tuning.
#   1. setup:    vLLM (pinned) for CUDA
#   2. download: the model from Hugging Face onto the pod's local disk (public, Apache-2.0); the resolved commit is
#                recorded
#   3. bench:    HumanEval, MBPP and their -hard variants, thinking off and on (scripts/qwen_eval.py)
#   4. push:     OUT (results, generations, logs) to runs/$RUN_ID/record on the volume
# Nothing on the volume is changed or deleted. Benchmark generations are measurement only, never training data.
set -uo pipefail
OUT=$1; cd "$(dirname "$0")/.."
V=${VOL_ROOT:-/workspace}; RUN=${RUN_ID:-eval-qwen}
MODEL=${QWEN_MODEL:-Qwen/Qwen3.8-27B}; VLLM=${VLLM_VERSION:-0.32.0}
T0=$(date +%s); DEADLINE=${POD_DEADLINE:-$(( T0 + 120 * 60 ))}
PY=${PY:-python3}
HFD=${HF_DIR:-$([ -n "${GW:-}" ] && echo "$V/hf" || echo /root/hf)}   # with GW, $V is the pod's local disk
mkdir -p "$OUT" "$HFD"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
st() { python3 jobs/status.py "$OUT" "$@"; }
fail() { st "$1" failed "error=$2"; log "FAILED $1: $2"; push; exit 1; }
left() { echo $(( DEADLINE - $(date +%s) )); }
push() {
  mkdir -p "$V/runs/$RUN/record" && find "$OUT" -maxdepth 1 -type f -exec cp {} "$V/runs/$RUN/record/" \;
  if [ -n "${GW:-}" ]; then
    $PY jobs/volsync.py push "runs/$RUN/record" --root "$V" --until $(( DEADLINE - 60 )) > "$OUT/push.json" 2>&1
    st push "$( [ $? = 0 ] && echo ok || echo partial)" "result=$(tail -1 "$OUT/push.json")"
  else
    st push ok "result=written to the network volume"
  fi
}
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
st plan ok "deadline=$DEADLINE" "minutes=$(( (DEADLINE - T0) / 60 ))" "gpus=$NGPU" "model=$MODEL" "vllm=$VLLM"
[ "$NGPU" -ge 1 ] || fail plan "no GPU"

# 1. setup ------------------------------------------------------------------------------------------------------
t=$(date +%s); log setup
$PY -m pip install -q "vllm==$VLLM" "huggingface_hub[hf_transfer]" >"$OUT/pip.log" 2>&1 || fail setup "pip: $(tail -c 300 "$OUT/pip.log")"
GPUINFO=$($PY -c "import torch, vllm; assert torch.cuda.is_available(); print(torch.cuda.device_count(), torch.cuda.get_device_name(0), 'torch', torch.__version__, 'vllm', vllm.__version__)" 2>&1) \
  || fail setup "cuda: $GPUINFO"
st setup ok "seconds=$(( $(date +%s) - t ))" "gpus=$GPUINFO"

# 2. download ---------------------------------------------------------------------------------------------------
t=$(date +%s); log download
MP=$(HF_HUB_ENABLE_HF_TRANSFER=1 $PY - "$MODEL" "$HFD" 2>"$OUT/download.err" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], cache_dir=sys.argv[2],
                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken", "LICENSE*"]))
PY
) || fail download "$(tail -c 300 "$OUT/download.err")"
REV=$(basename "$MP")
st download ok "seconds=$(( $(date +%s) - t ))" "revision=$REV" "gb=$(du -sL "$MP" | awk '{printf "%.1f", $1/1048576}')"

# 3. bench ------------------------------------------------------------------------------------------------------
log bench
$PY -c "import sys; sys.path.insert(0, 'data'); import build_corpus as bc; bc.get_evals()" >/dev/null 2>&1
mkdir -p data/stores/evals && cp data/out/evals/*.jsonl data/stores/evals/ 2>/dev/null
for s in humaneval mbpp; do [ -s "data/stores/evals/$s.jsonl" ] || fail bench "eval set $s missing"; done
t=$(date +%s)
[ "$(left)" -gt 1200 ] || fail bench "less than 20 min left after setup and download"
timeout $(( $(left) - 240 )) $PY scripts/qwen_eval.py --model "$MP" --modes "${MODES:-instruct,think}" --out "$OUT" \
  --tp "$NGPU" --deadline $(( DEADLINE - 1800 )) --jobs "$(nproc)" ${QWEN_EVAL_ARGS:-} \
  > "$OUT/qwen-eval.jsonl" 2> "$OUT/qwen-eval.err"
rc=$?
while IFS= read -r line; do
  [ -n "$line" ] || continue
  name=$(printf '%s' "$line" | $PY -c "import json,sys; r=json.load(sys.stdin); print(f\"bench-{r.get('mode')}-{r.get('suite','all')}\")")
  st "$name" ok "result=$line"
done < "$OUT/qwen-eval.jsonl"
[ $rc = 0 ] && st bench ok "seconds=$(( $(date +%s) - t ))" "revision=$REV" \
            || st bench failed "seconds=$(( $(date +%s) - t ))" "rc=$rc" "error=$(tail -c 400 "$OUT/qwen-eval.err")"

# 4. push the record --------------------------------------------------------------------------------------------
st done ok "wall_min=$(( ($(date +%s) - T0) / 60 ))"
push
