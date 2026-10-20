# Shared by the pod jobs: find the s2-run2 step-7500 export on the volume and turn it into a forge checkpoint
# once (scripts/import_export.py hard-links the weights; nothing is deleted).
V=${VOL_ROOT:-/workspace}
SRC="$V/ckpt/quasnir-1b-s7500"

ensure_1b_ckpt() {  # $1 = job output dir for an error report
  python3 -c "import numpy" 2>/dev/null || python3 -m pip install -q numpy
  [ -f "$SRC/model.safetensors" ] && return 0
  local exp sha seq
  exp=$(find "$V" -maxdepth 8 -name 'weights-bf16-step007500.safetensors' -path '*export*' | sort | head -1)
  [ -n "$exp" ] || { echo '{"error": "export weights-bf16-step007500.safetensors not found"}' > "$1/result.json"; return 3; }
  sha=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['export']['sha256'])" "$(dirname "$exp")/EXPORT.json")
  seq=$(python3 - "$exp" <<'PY'
import json, sys
from pathlib import Path
for p in Path(sys.argv[1]).parents:
    for s in sorted(p.glob("step-007500/state.json")) + sorted(p.glob("state.json")):
        try:
            print(json.loads(s.read_text())["config"]["model"]["max_seq_len"]); sys.exit()
        except (KeyError, ValueError):
            pass
print(512)
PY
)
  python3 scripts/import_export.py "$exp" "$SRC" --expect-sha256 "$sha" --max-seq-len "$seq"
}
