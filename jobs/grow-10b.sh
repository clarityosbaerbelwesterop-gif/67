#!/usr/bin/env bash
# Job "grow-10b" (runpod-job workflow in Osirus): the s2-run2 step-7500 export on the volume -> forge checkpoint
# -> function-preserving growth to about 10B (scripts/grow.py). Writes only new directories under
# /workspace/ckpt; nothing is overwritten or deleted.
set -euo pipefail
OUT=$1; cd "$(dirname "$0")/.."
. jobs/common.sh
ensure_1b_ckpt "$OUT" || exit $?
DST="$V/ckpt/quasnir-10b-grown-s7500"
read -r DIM FFN LAYERS < <(python3 - "$SRC/state.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["config"]["model"]
hd, rep = m["dim"] // m["n_heads"], m["n_heads"] // m["n_kv_heads"]
step = hd * rep
dim = max(m["dim"], round(4096 / step) * step)
ffn = max(m["ffn_hidden"], round(dim * 3.5 / 256) * 256)
print(dim, ffn, max(m["n_layers"], 44))
PY
)
DIM=${GROW_DIM:-$DIM}; FFN=${GROW_FFN:-$FFN}; LAYERS=${GROW_LAYERS:-$LAYERS}  # overrides for tests
NEED=$(python3 scripts/grow.py "$SRC" "$DST" --dim "$DIM" --ffn-hidden "$FFN" --n-layers "$LAYERS" --dry-run \
       | python3 -c "import json,sys; print(json.load(sys.stdin)['grown_bytes_f32'])")
FREE=$(python3 -c "import shutil,sys; print(shutil.disk_usage(sys.argv[1]).free)" "$V")
if [ "$FREE" -lt $(( NEED + 20000000000 )) ]; then
  echo "{\"error\": \"volume free $FREE bytes, need $NEED + 20 GB\"}" > "$OUT/result.json"; exit 4
fi
python3 scripts/grow.py "$SRC" "$DST" --dim "$DIM" --ffn-hidden "$FFN" --n-layers "$LAYERS" > "$OUT/grow.json"
python3 - "$OUT" "$SRC" "$DST" <<'PY'
import json, shutil, sys
from pathlib import Path
out, src, dst = sys.argv[1:]
g = json.loads(Path(out, "grow.json").read_text())
du = shutil.disk_usage(dst)
Path(out, "result.json").write_text(json.dumps({"source_ckpt": src, "grown_ckpt": dst, **g,
                                                "volume_free_after": du.free}, indent=1))
PY
