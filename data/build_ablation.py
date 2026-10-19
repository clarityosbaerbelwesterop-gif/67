"""Value gate for new data: the second arm of a replacement ablation (docs/PLAN_100B_MOE.md, "Wert-Tor").

A new source earns a place in the training mix only if a token of it is worth at least as much as a token of the data
it replaces. The test is a replacement ablation at equal tokens:

  arm A   the baseline store as it is
  arm B   this script: --tokens tokens, of which --frac are whole documents of the candidate store and the rest whole
          documents of the baseline store, both drawn by a seeded shuffle

Both arms are trained from the same checkpoint with the same seed, schedule and steps (accel/run_s2.py --init ...),
and arm A once more with a second seed to measure the noise. The candidate passes if arm B beats arm A on the held-out
code val by more than twice the seed noise and is not worse than that noise on general val. Its weight in the mix
follows the measured gain, not its size.

Output is a store the trainer's fail-closed CleanStore reads: train.bin, train.val.bin (empty), train.meta.json,
kept.npy (doc, start, len without eos), report.json (sha256 of train.bin), manifest.json.

  python3 data/build_ablation.py --base-meta BASE/train.meta.json --base-index BASE_KEPT.tsv|.npy \\
      --cand-meta /vol/data/v6/store/train.meta.json --cand-index /vol/data/v6/store/kept.npy \\
      --frac 0.3 --tokens 1e9 --seed 1 --out /vol/data/ablation/stack-python
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EOS = 8190


def load_index(path: Path) -> np.ndarray:
    """(doc, start, len) rows; len excludes the eos at start + len (the CleanStore contract)."""
    if str(path).endswith(".npy"):
        a = np.load(path).astype(np.int64)
    else:
        a = np.loadtxt(path, dtype=np.int64, delimiter="\t", skiprows=1, ndmin=2)
    return a.reshape(-1, 3)


def load_bin(meta: Path) -> np.memmap:
    m = json.loads(meta.read_text())
    if m.get("dtype", "uint16") != "uint16":
        raise SystemExit(f"{meta}: only uint16 stores")
    return np.memmap(meta.parent / Path(m["bin"]).name, "<u2", "r")


def pick(index: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    """Whole documents in a seeded random order until their tokens (incl. eos) reach target."""
    order = rng.permutation(len(index))
    lens = index[order, 2] + 1
    n = int(np.searchsorted(np.cumsum(lens), target)) + 1
    if n > len(order):
        raise SystemExit(f"store too small: {int(lens.sum()):,} tokens < {target:,}")
    return index[order[:n]]


def build(base_meta: Path, base_index: Path, cand_meta: Path, cand_index: Path, frac: float, tokens: int, seed: int,
          out: Path) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists; nothing is overwritten")
    if not 0 < frac < 1:
        raise SystemExit("--frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    parts = {"candidate": (load_bin(cand_meta), pick(load_index(cand_index), int(tokens * frac), rng)),
             "baseline": (load_bin(base_meta), pick(load_index(base_index), int(tokens * (1 - frac)), rng))}
    docs = [(name, int(s), int(ln)) for name, (_, idx) in parts.items() for s, ln in idx[:, 1:]]
    docs = [docs[i] for i in rng.permutation(len(docs))]
    stage = out.with_name(f".{out.name}-staging")
    stage.mkdir(parents=True)
    kept, pos, counts = [], 0, {"candidate": 0, "baseline": 0}
    h = hashlib.sha256()
    with open(stage / "train.bin", "wb") as f:
        for name, s, ln in docs:
            seg = np.asarray(parts[name][0][s:s + ln + 1], "<u2")
            if len(seg) != ln + 1 or int(seg[-1]) != EOS:
                raise SystemExit(f"{name}: document at {s} does not end on eos; index and bin disagree")
            b = seg.tobytes()
            f.write(b)
            h.update(b)
            kept.append((len(kept), pos, ln))
            pos += ln + 1
            counts[name] += ln + 1
    (stage / "train.val.bin").write_bytes(b"")
    np.save(stage / "kept.npy", np.array(kept, np.int64).reshape(-1, 3))
    (stage / "report.json").write_text(json.dumps({"train_bin_sha256_expected": h.hexdigest(), "kept_docs": len(kept)}))
    (stage / "train.meta.json").write_text(json.dumps({"tokens": pos, "val_tokens": 0, "dtype": "uint16",
                                                      "vocab_size": 8192, "bin": "train.bin",
                                                      "val_bin": "train.val.bin", "mix": counts}, indent=1))
    man = {"arm": "B (replacement ablation)", "frac_requested": frac, "frac_actual": counts["candidate"] / pos,
           "tokens": pos, "tokens_per_part": counts, "seed": seed, "documents": len(kept),
           "baseline": {"meta": str(base_meta), "index": str(base_index)},
           "candidate": {"meta": str(cand_meta), "index": str(cand_index)}, "train_bin_sha256": h.hexdigest()}
    (stage / "manifest.json").write_text(json.dumps(man, indent=1))
    stage.rename(out)
    return man


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for k in ("base-meta", "base-index", "cand-meta", "cand-index", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--tokens", type=float, default=1e9)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    man = build(a.base_meta, a.base_index, a.cand_meta, a.cand_index, a.frac, int(a.tokens), a.seed, a.out)
    print(json.dumps({k: man[k] for k in ("tokens", "frac_actual", "documents", "train_bin_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
