"""jobs/train-final.sh end to end with a fake python that stands in for pip, JAX, the corpus builder, the trainer,
the gateway client and the benchmarks: every phase runs in order, the schedule follows the time left, DONE.json is
the last file pushed, and a run without self-termination stops at once."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FAKE = r'''#!/usr/bin/env bash
# fake interpreter for jobs/train-final.sh; logs every call
echo "$*" >> "$FAKE_LOG"
case "$1" in
  -m)
    case "$2" in
      pip) exit 0;;
      accel.run_s2)
        out=""; while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
        mkdir -p "$out/step-000100" "$out/export"
        echo '{"config": {"model": {}}}' > "$out/step-000100/state.json"; echo w > "$out/step-000100/model.safetensors"
        echo e > "$out/export/weights-bf16-step000100.safetensors"
        echo '{"event": "step", "step": 10, "loss": 1.5}'
        printf '{"status": "ok", "final_ckpt": "%s"}' "$out/step-000100" > "$out/DONE.json"; exit 0;;
      accel.eval) echo '{"val_loss": 1.9}'; exit 0;;
    esac;;
  -) cat > /dev/null; echo "2 NVIDIA B200"; exit 0;;
  -c)
    case "$2" in
      *jax.device_count*) echo 2; exit 0;;
      *get_evals*) exit 0;;
    esac
    exec python3 "$@";;
  jobs/volsync.py)
    op=$2; shift 2; root=""; rels=()
    while [ $# -gt 0 ]; do case "$1" in --root) root=$2; shift 2;; --*) [ "$1" = --verify ] || [ "$1" = --require ] && shift || shift 2;; *) rels+=("$1"); shift;; esac; done
    case $op in
      health) echo '{"op": "health", "ok": true, "free": 300000000000}';;
      pull)
        for r in "${rels[@]}"; do
          case "$r" in
            runs/train-1-f/1b/DONE.json) mkdir -p "$root/runs/train-1-f/1b"
              echo '{"status": "ok", "final_ckpt": "/workspace/runs/train-1-f/1b/step-002704"}' > "$root/$r";;
            runs/train-1-f/1b/step-002704) mkdir -p "$root/$r"; echo '{}' > "$root/$r/state.json"; echo w > "$root/$r/model.safetensors";;
            data/train-1/store) mkdir -p "$root/$r"; echo '{"tokens": 4}' > "$root/$r/train.meta.json"; printf 'abcdefgh' > "$root/$r/train.bin"; echo k > "$root/$r/kept.npy";;
            runs/train-1-f/10b/export) mkdir -p "$root/$r"; echo t > "$root/$r/weights-bf16-step000973.safetensors";;
          esac
        done
        echo '{"op": "pull", "pulled": [], "missing": [], "bad": []}';;
      push) printf '%s\n' "${rels[@]}" >> "$FAKE_PUSH"; echo '{"op": "push", "pushed": 1, "missing": [], "conflicts": []}';;
    esac; exit 0;;
  data/build_corpus_v6.py)
    out=""; for ((i=1; i<=$#; i++)); do [ "${!i}" = --out ] && { j=$((i+1)); out=${!j}; }; done
    case $2 in
      plan) echo '{}' > "$out/plan.json";;
      dedup) mkdir -p "$out/dedup"; echo d > "$out/dedup/drop.npy";;
      pack) mkdir -p "$out/store"; echo '{"tokens": 3}' > "$out/store/train.meta.json"; printf 'abcdef' > "$out/store/train.bin"
            echo k > "$out/store/kept.npy"; echo '{"documents": 3}' > "$out/store/manifest.json";;
    esac; exit 0;;
  data/build_reasoning.py) mkdir -p data/cache/reasoning; echo '{"t": "x"}' > data/cache/reasoning/docs-train.jsonl; exit 0;;
  scripts/import_export.py) mkdir -p "$3"; echo m > "$3/model.safetensors"; echo '{}' > "$3/state.json"; exit 0;;
  scripts/humaneval.py) echo '{"pass@1": 0.02, "oracle_any_pass": 0.05}'; exit 0;;
esac
echo "fake python: unhandled $*" >&2; exit 9
'''


def setup(tmp_path):
    repo = tmp_path / "repo"
    (repo / "jobs").mkdir(parents=True)
    for f in ("train-final.sh", "status.py", "train-final-pt-sources.json", "train-final-sft-sources.json"):
        shutil.copy(ROOT / "jobs" / f, repo / "jobs" / f)
    (repo / "target/release").mkdir(parents=True)
    (repo / "target/release/forge").write_text("#!/bin/sh\n")
    (repo / "target/release/forge").chmod(0o755)
    (repo / "data/out/evals").mkdir(parents=True)
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    return repo, fake


def run(tmp_path, minutes, **env):
    repo, fake = setup(tmp_path)
    out, vol = tmp_path / "out", tmp_path / "vol"
    vol.mkdir()
    e = {**os.environ, "PY": str(fake), "VOL_ROOT": str(vol), "RUN_ID": "train-final-t", "GW": "http://gw/tok",
         "POD_DEADLINE": str(int(time.time()) + minutes * 60), "FAKE_LOG": str(tmp_path / "calls.log"),
         "FAKE_PUSH": str(tmp_path / "push.log"), **env}
    r = subprocess.run(["bash", str(repo / "jobs/train-final.sh"), str(out)], capture_output=True, text=True, env=e,
                       timeout=300)
    res = json.loads((out / "result.json").read_text())["phases"]
    pushes = (tmp_path / "push.log").read_text().split() if (tmp_path / "push.log").exists() else []
    calls = (tmp_path / "calls.log").read_text() if (tmp_path / "calls.log").exists() else ""
    return r, res, pushes, calls, vol


def test_full_long_run(tmp_path):
    r, res, pushes, calls, vol = run(tmp_path, 1320, LONG="1", SELFTERM_OK="1")
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    want = ["plan", "setup", "schedule", "pull", "data-final-pt", "data-final-sft", "p0-warm", "pt-1b", "sft-1b",
            "val-tf-1b", "val-pt-1b", "val-sft-1b", "bench-sft-1b-humaneval", "bench-pt-1b-mbpp-hard", "10b-init",
            "sft-10b", "bench-sft-10b-humaneval", "done"]
    for name in want:
        assert res.get(name, {}).get("status") in ("ok",), (name, res.get(name))
    assert res["schedule"]["ten"] == 1 and res["schedule"]["sft_min"] == 120
    # training order and inits: p0 from the train-1-f 1B, pt from p0, sft from pt, the 10B from its export
    runs = [l for l in calls.splitlines() if "accel.run_s2" in l]
    assert [l.split("--out ")[1].split()[0].rsplit("/", 1)[1] for l in runs] == ["p0-warm", "pt-1b", "sft-1b", "sft-10b"]
    assert "runs/train-1-f/1b/step-002704" in runs[0] and "p0-warm/step-000100" in runs[1] and "pt-1b/step-000100" in runs[2]
    assert "data/train-1/store" in runs[0] and "data/final-pt/store" in runs[1] and "data/final-sft/store" in runs[2]
    assert "--lr 1e-4 --min-lr 1e-4" in runs[0] and "--seq 1024" in runs[1]
    # pushes: no optimizer state, the 10B without its f32 weights, DONE.json last
    assert pushes[-1] == "runs/train-final-t/DONE.json" and "runs/train-final-t/record" in pushes
    assert not any("optim" in p for p in pushes)
    assert not any(p.startswith("runs/train-final-t/sft-10b/step-") for p in pushes)
    assert "runs/train-final-t/sft-1b/step-000100/model.safetensors" in pushes
    assert not any(p.startswith("runs/train-final-t/p0-warm") for p in pushes)
    assert json.loads((vol / "runs/train-final-t/DONE.json").read_text())["status"] == "ok"


def test_without_self_termination_it_stops_at_once(tmp_path):
    r, res, pushes, calls, _ = run(tmp_path, 1320, LONG="1", SELFTERM_OK="0")
    assert r.returncode == 1 and res["plan"]["status"] == "failed" and "accel.run_s2" not in calls
    assert pushes[-1] == "runs/train-final-t/DONE.json"


def test_short_window_is_refused(tmp_path):
    r, res, _, calls, _ = run(tmp_path, 300, LONG="1", SELFTERM_OK="1")
    assert r.returncode == 1 and "600 minutes" in res["plan"]["error"] and "pip" not in calls


def test_no_10b_when_time_is_short(tmp_path):
    r, res, _, calls, _ = run(tmp_path, 700, LONG="1", SELFTERM_OK="1")
    assert r.returncode == 0, r.stderr[-2000:]
    assert res["schedule"]["ten"] == 0 and res["10b"]["status"] == "skipped" and "sft-10b" not in calls
