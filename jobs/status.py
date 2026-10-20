"""Merge one phase into OUT/result.json of a pod job: python3 jobs/status.py OUT PHASE STATUS [key=value ...] [--json FILE]
Values that parse as JSON are stored as JSON. --json adds the last JSON object line of FILE (a run log) under "last"."""
import fcntl
import json
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    out, phase, status, rest = Path(argv[0]), argv[1], argv[2], argv[3:]
    res_p = out / "result.json"
    lock = open(out / ".result.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)  # the bench runs in the background and writes here too
    try:
        res = json.loads(res_p.read_text())
    except (OSError, ValueError):
        res = {"phases": {}}
    ph = res["phases"].setdefault(phase, {})
    ph.update(status=status, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    i = 0
    while i < len(rest):
        if rest[i] == "--json":
            lines = [l for l in Path(rest[i + 1]).read_text(errors="replace").splitlines() if l.startswith("{")]
            if lines:
                try:
                    ph["last"] = json.loads(lines[-1])
                except ValueError:
                    pass
            i += 2
            continue
        k, _, v = rest[i].partition("=")
        try:
            ph[k] = json.loads(v)
        except ValueError:
            ph[k] = v
        i += 1
    tmp = res_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(res_p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
