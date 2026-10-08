"""Encrypted resume pack: carry the 10-day run into a new session or machine.

The run's checkpoints live only on this container's disk (runs/ is not in git).
`save` collects what a fresh clone needs to continue instead of starting over:
  * the weights (model.safetensors + state.json) of every checkpoint the
    long-run state, the RSI champion files and the latest.json pointers refer to;
  * the full latest checkpoint (incl. optimizer) of a training step in progress;
  * the small metadata: runs/longrun state/configs/logs, RSI champion/metrics/
    audit/held-out files, champions and reports.
It tars them, encrypts with AES-256 (openssl, PBKDF2 200k iterations; the key
is read from $FORGE_PACK_KEY and never written anywhere) and splits the result
into 90 MB parts under resume/pack/ so they can be pushed to git (GitHub's
per-file limit is 100 MB). The repository is public: without the key the parts
are unreadable, which keeps the restricted Rouge weights confidential.
Corpora (data/out) are not packed; rebuild them with data/build_corpus_v2.py and
data/build_corpus_v3.py.

Usage:
  FORGE_PACK_KEY=... python3 scripts/resume_pack.py save [--dry-run]
  FORGE_PACK_KEY=... python3 scripts/resume_pack.py restore
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "resume" / "pack"
PART = 90 * 1024 * 1024
META_GLOBS = ["runs/longrun/*.json", "runs/longrun/*.jsonl", "runs/longrun/*.out", "runs/longrun/configs/*",
              "runs/champions.json", "runs/report.json", "runs/benchmarks.jsonl", "runs/*.jsonl", "runs/rsi/*/*.json", "runs/rsi/*/*.jsonl", "runs/rsi/*/heldout/**/*",
              "runs/rsi/*/configs/*"]
TRAIN_STEPS = {"base", "quasnir", "rouge"}


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


def ckpt_dir(s: str) -> Path | None:
    p = ROOT / s
    return p if s.startswith("runs/") and (p / "model.safetensors").is_file() else None


def collect() -> list[Path]:
    files: set[Path] = set()

    def add_ckpt(d: Path, full: bool = False) -> None:
        for real in {d, d.resolve()}:
            if real.is_symlink():
                files.add(real)  # the link itself; its target is added via resolve()
                continue
            for f in real.iterdir():
                if f.is_file() and (full or f.name != "optim.safetensors"):
                    files.add(f)

    # Weights needed to continue: the current champions, the bases they derive
    # from (for the cross-cycle rebase), everything the latest cycle refers to and
    # the champion/anchor of that cycle's RSI runs. Older generations are history.
    state = json.loads((ROOT / "runs/longrun/state.json").read_text())
    cycles = state.get("cycles", {})
    last = max(cycles, key=lambda g: int(str(g).lstrip("g") or 0)) if cycles else None
    refs = list(strings(state.get("current", {})))
    if last is not None:
        refs += list(strings(cycles[last]))
        for f in ROOT.glob(f"runs/rsi/*-g{str(last).lstrip('g')}/champion.json"):
            refs += list(strings(json.loads(f.read_text())))
    for cur in (state.get("current") or {}).values():
        cfg = ROOT / "runs/longrun/configs" / f"{(cur or {}).get('run')}.json"
        if cfg.is_file():
            refs.append(json.loads(cfg.read_text()).get("init_from") or "")
    for s in refs:
        d = ckpt_dir(s.rstrip("/"))
        if d and "rebased-init-" not in s:
            add_ckpt(d)
    # A training step in progress resumes from its newest checkpoint, optimizer included.
    for cyc in state.get("cycles", {}).values():
        for name, st in cyc.get("steps", {}).items():
            if name in TRAIN_STEPS and st.get("status") == "running":
                steps = sorted((ROOT / "runs").glob(f"{st.get('run') or '*'}/step-*"))
                if steps:
                    add_ckpt(steps[-1], full=True)
    for f in list(files):  # the latest.json pointer of every run whose weights are packed
        rel = f.relative_to(ROOT).parts
        if len(rel) > 2 and rel[0] == "runs" and (ROOT / "runs" / rel[1] / "latest.json").is_file():
            files.add(ROOT / "runs" / rel[1] / "latest.json")
    for g in META_GLOBS:
        files |= {f for f in ROOT.glob(g) if f.is_file()}
    return sorted(files)


def key() -> str:
    k = os.environ.get("FORGE_PACK_KEY", "")
    if len(k) < 32:
        sys.exit("set FORGE_PACK_KEY (>= 32 characters)")
    return k


def save(dry: bool) -> int:
    files = collect()
    total = sum(f.lstat().st_size for f in files)
    print(f"{len(files)} files, {total / 1e6:.0f} MB before encryption")
    if dry:
        for f in files:
            print(f"  {f.relative_to(ROOT)}  {f.lstat().st_size / 1e6:.1f} MB")
        return 0
    k = key()
    PACK.mkdir(parents=True, exist_ok=True)
    for old in PACK.glob("part-*"):
        old.unlink()
    lst = PACK / ".files"
    lst.write_text("\n".join(str(f.relative_to(ROOT)) for f in files) + "\n")
    tar = subprocess.Popen(["tar", "-C", str(ROOT), "-cf", "-", "-T", str(lst)], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-salt", "-pass", "env:FORGE_PACK_KEY"],
                           stdin=tar.stdout, stdout=subprocess.PIPE, env={**os.environ, "FORGE_PACK_KEY": k})
    tar.stdout.close()
    parts, n, h = [], 0, hashlib.sha256()
    while chunk := enc.stdout.read(PART):
        p = PACK / f"part-{n:03d}"
        p.write_bytes(chunk)
        h.update(chunk)
        parts.append({"file": p.name, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
        n += 1
    if enc.wait() or tar.wait():
        sys.exit("tar/openssl failed")
    lst.unlink()
    state = json.loads((ROOT / "runs/longrun/state.json").read_text())
    (PACK / "MANIFEST.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": len(files), "plain_bytes": total,
        "cipher": "openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt (key: $FORGE_PACK_KEY)",
        "parts": parts, "sha256": h.hexdigest(),
        "current": {k2: (v or {}).get("dir") for k2, v in state.get("current", {}).items()}}, indent=2) + "\n")
    print(f"wrote {n} parts to {PACK.relative_to(ROOT)}")
    return 0


def restore() -> int:
    k = key()
    man = json.loads((PACK / "MANIFEST.json").read_text())
    for p in man["parts"]:
        if hashlib.sha256((PACK / p["file"]).read_bytes()).hexdigest() != p["sha256"]:
            sys.exit(f"{p['file']} is corrupt")
    cat = subprocess.Popen(["cat"] + [str(PACK / p["file"]) for p in man["parts"]], stdout=subprocess.PIPE)
    dec = subprocess.Popen(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-pass", "env:FORGE_PACK_KEY"],
                           stdin=cat.stdout, stdout=subprocess.PIPE, env={**os.environ, "FORGE_PACK_KEY": k})
    cat.stdout.close()
    tar = subprocess.run(["tar", "-C", str(ROOT), "-xpf", "-"], stdin=dec.stdout)
    if dec.wait() or tar.returncode:
        sys.exit("decryption or extraction failed (wrong key?)")
    print(f"restored {man['files']} files from the pack of {man['created']}; champions: {man['current']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["save", "restore"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.exit(save(a.dry_run) if a.cmd == "save" else restore())
