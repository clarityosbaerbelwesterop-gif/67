#!/usr/bin/env bash
# Keeps scripts/longrun.py running: starts it when it is not running, restarts it
# after a failure with backoff, and stops when it finishes (exit 0) or reports a
# fatal error (3) or a STOP (75). Safe to call repeatedly (one instance via a lock).
#   scripts/supervise.sh [longrun.py args...]       e.g. --days 3   or   --forever
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/longrun
exec 9>runs/longrun/supervise.lock
flock -n 9 || { echo "supervise: already running"; exit 0; }
log() { echo "[$(date -u +%FT%TZ)] $*" >> runs/longrun/supervise.log; }
backoff=60
while true; do
  log "starting longrun $*"
  python3 scripts/longrun.py "$@" >> runs/longrun/longrun.out 2>&1
  rc=$?
  log "longrun exited with $rc"
  case $rc in
    0)  log "finished"; exit 0 ;;
    3)  log "fatal error: fix the cause (see runs/longrun/longrun.out), then start again"; exit 3 ;;
    75) log "stopped (STOP file or signal)"; exit 75 ;;
    4)  log "another longrun instance holds the lock"; sleep 300 ;;
    *)  sleep "$backoff"; backoff=$(( backoff < 1800 ? backoff * 2 : 1800 )) ;;
  esac
done
