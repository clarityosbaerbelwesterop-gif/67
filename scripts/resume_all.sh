#!/usr/bin/env bash
# After a container or machine restart: (re)start whatever is not running.
# Idempotent; every component resumes from its own checkpoints/state.
#   generation 1  scripts/train_all.sh + scripts/darus_finish.sh (until runs/report.json exists)
#   long run      scripts/supervise.sh --days ${LONGRUN_DAYS:-10}
#   headcenter    python3 -m headcenter.backend (loopback, port 8067)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
running() { pgrep -f "$1" > /dev/null; }
if [ ! -e runs/benchmarks.jsonl ] || ! grep -q '"finished"' runs/benchmarks.jsonl 2>/dev/null; then
  if ! running "bash scripts/train_all.sh"; then
    setsid nohup bash -c 'echo $$ > runs/pipeline.pid; exec bash scripts/train_all.sh' >> runs/pipeline.log 2>&1 < /dev/null &
    sleep 1
    echo "started generation-1 pipeline"
  fi
  if ! running "bash scripts/darus_finish.sh"; then
    setsid nohup bash scripts/darus_finish.sh "$(cat runs/pipeline.pid 2>/dev/null)" > runs/darus_finish.log 2>&1 < /dev/null &
    echo "started darus finish chain"
  fi
fi
if [ ! -e runs/longrun/STOP ] && ! running "scripts/supervise.sh"; then
  setsid nohup scripts/supervise.sh --days "${LONGRUN_DAYS:-10}" > /dev/null 2>&1 < /dev/null &
  echo "started long-run supervisor"
fi
if ! running "headcenter.backend"; then
  setsid nohup nice -n 5 python3 -m headcenter.backend --runs runs --configs training/configs --port 8067 > runs/headcenter.log 2>&1 < /dev/null &
  echo "started headcenter"
fi
