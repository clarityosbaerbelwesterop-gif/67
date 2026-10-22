#!/usr/bin/env bash
# Self-termination of RunPod pods for long runs (Osirus runpod-job, JOB line hours=N). RunPod gives every pod
# RUNPOD_POD_ID and RUNPOD_API_KEY, a key limited to that pod. With them a pod can end itself, so a run may outlive
# the GitHub job that started it without running on after its deadline.
#
#   pod_guard.sh check                            exit 0 if this pod can read itself with its own key; prints JSON
#   pod_guard.sh watch DEADLINE [DONE [GRACE_S]]  ends this pod at DEADLINE (unix time), or GRACE_S (default 600) s
#                                                 after the file DONE appears; retries every 60 s until it is gone
#   pod_guard.sh publish SRC DST                  every 60 s copies SRC/result.json and SRC/exit_code to DST with
#                                                 every string in $REDACT (space separated) replaced by ***
#   pod_guard.sh end                              ends this pod now
#
# Ending tries runpodctl, then the GraphQL mutation podTerminate, then REST DELETE. Terminate, not stop: a stopped
# pod keeps its disk and is still billed. Network volumes are never touched. GUARD_ONCE=1 runs one loop pass (tests).
set -uo pipefail
GQL=${RUNPOD_GQL_URL:-https://api.runpod.io/graphql}
REST=${RUNPOD_REST_URL:-https://rest.runpod.io/v1}
ID=${RUNPOD_POD_ID:-}
log() { echo "[guard $(date -u +%H:%M:%S)] $*" >&2; }

gql() {  # $1 = query; prints the response
  python3 - "$1" <<'PY' | curl -sS -m 30 -X POST "$GQL" -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
    -H "Content-Type: application/json" --data-binary @-
import json, sys
print(json.dumps({"query": sys.argv[1]}))
PY
}

check() {
  [ -n "$ID" ] && [ -n "${RUNPOD_API_KEY:-}" ] || { echo '{"ok": false, "why": "RUNPOD_POD_ID or RUNPOD_API_KEY missing"}'; return 1; }
  [[ "$ID" =~ ^[a-z0-9]{6,40}$ ]] || { echo '{"ok": false, "why": "unexpected pod id"}'; return 1; }
  if gql "query { pod(input: {podId: \"$ID\"}) { id desiredStatus } }" 2>/dev/null \
       | python3 -c "import json,sys,os; sys.exit(0 if (json.load(sys.stdin).get('data') or {}).get('pod', {}).get('id') == os.environ['RUNPOD_POD_ID'] else 1)" 2>/dev/null; then
    echo '{"ok": true, "how": "graphql"}'; return 0
  fi
  if command -v runpodctl >/dev/null && runpodctl get pod "$ID" >/dev/null 2>&1; then
    echo '{"ok": true, "how": "runpodctl"}'; return 0
  fi
  echo '{"ok": false, "why": "the pod key cannot read this pod"}'; return 1
}

end() {
  [ -n "$ID" ] || { log "no RUNPOD_POD_ID"; return 1; }
  log "ending pod $ID"
  if command -v runpodctl >/dev/null && runpodctl remove pod "$ID" >&2; then return 0; fi
  if gql "mutation { podTerminate(input: {podId: \"$ID\"}) }" 2>/dev/null | python3 -c "import json,sys; sys.exit(1 if json.load(sys.stdin).get('errors') else 0)" 2>/dev/null; then
    return 0
  fi
  c=$(curl -sS -m 30 -o /dev/null -w '%{http_code}' -X DELETE "$REST/pods/$ID" -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" 2>/dev/null || echo 000)
  case $c in 2*|404) return 0;; esac
  log "could not end the pod (rest http $c)"; return 1
}

watch() {  # $1 deadline, $2 done file, $3 grace seconds
  local deadline=$1 done=${2:-} grace=${3:-600}
  while :; do
    now=$(date +%s); why=""
    [ "$now" -ge "$deadline" ] && why="deadline"
    if [ -z "$why" ] && [ -n "$done" ] && [ -e "$done" ]; then
      age=$(( now - $(stat -c %Y "$done" 2>/dev/null || echo "$now") ))
      [ "$age" -ge "$grace" ] && why="done"
    fi
    if [ -n "$why" ]; then
      log "reason: $why"
      end && return 0
    fi
    [ -n "${GUARD_ONCE:-}" ] && return 2
    sleep 60
  done
}

publish() {  # $1 src dir, $2 public dir
  mkdir -p "$2"
  while :; do
    for f in result.json exit_code selfterm.json; do
      [ -f "$1/$f" ] || continue
      python3 - "$1/$f" "$2/.$f.tmp" <<'PY' && mv "$2/.$f.tmp" "$2/$f"
import os, sys
text = open(sys.argv[1], errors="replace").read()
for s in os.environ.get("REDACT", "").split():
    if len(s) >= 8:
        text = text.replace(s, "***")
open(sys.argv[2], "w").write(text)
PY
    done
    [ -n "${GUARD_ONCE:-}" ] && return 0
    sleep 60
  done
}

case "${1:-}" in
  check) check;;
  end) end;;
  watch) shift; watch "$@";;
  publish) shift; publish "$@";;
  *) echo "usage: pod_guard.sh check|end|watch DEADLINE [DONE [GRACE_S]]|publish SRC DST" >&2; exit 64;;
esac
