"""Read-only inventory of a RunPod network volume: sizes per directory, large files, and the small checkpoint,
export and data manifests (their JSON is copied into the report). Nothing is changed, nothing is followed
through symlinks, and no other file content is read.

Usage: python3 jobs/inventory.py ROOT OUT_DIR   (writes OUT_DIR/result.json)
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

MANIFESTS = {"EXPORT.json", "latest.json", "state.json", "manifest.json", "report.json", "train.meta.json",
             "dedup.json", "scan.json", "plan.json", "DONE", "STOP", "RUN", "FETCH"}
MAX_MANIFEST = 64 << 10
BIG = 100 << 20
DIR_DEPTH = 3
LIMIT_S = 1200


def inventory(root: Path) -> dict:
    t0, files, dirs, big, manifests, partial = time.time(), 0, {}, [], {}, False
    for dp, dns, fns in os.walk(root, followlinks=False):
        if time.time() - t0 > LIMIT_S:
            partial = True
            break
        rel = Path(dp).relative_to(root)
        dns.sort()
        for fn in fns:
            p = Path(dp) / fn
            try:
                st = p.lstat()
            except OSError:
                continue
            if not p.is_file() or p.is_symlink():
                continue
            files += 1
            parts = (rel / fn).parts
            for d in range(1, min(DIR_DEPTH, len(parts) - 1) + 1):
                k = "/".join(parts[:d])
                dirs.setdefault(k, [0, 0])
                dirs[k][0] += st.st_size
                dirs[k][1] += 1
            if st.st_size >= BIG and len(big) < 3000:
                big.append({"path": str(rel / fn), "bytes": st.st_size, "mtime": int(st.st_mtime)})
            if fn in MANIFESTS and st.st_size <= MAX_MANIFEST and len(manifests) < 1000:
                text = p.read_text(errors="replace")
                try:
                    manifests[str(rel / fn)] = json.loads(text)
                except ValueError:
                    manifests[str(rel / fn)] = text.strip()[:2000]
    du = shutil.disk_usage(root)
    return {"root": str(root), "files": files, "partial": partial, "seconds": round(time.time() - t0, 1),
            "disk": {"total": du.total, "used": du.used, "free": du.free},
            "dirs": {k: {"bytes": v[0], "files": v[1]} for k, v in sorted(dirs.items())},
            "big_files": sorted(big, key=lambda b: b["path"]), "manifests": manifests}


def main(argv: list[str]) -> int:
    root, out = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(inventory(root), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
