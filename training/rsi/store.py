"""SCP CorpusStore writer for RSI rounds.

Accepted samples (prompt + verified completion, or programmatic reference
solutions in a warm-start round) are tokenised with `forge tokenize --jsonl`
(the shared SCP BPE, no eos appended), each document gets eos 8190, and the
documents are mixed with a replay sample from an existing store's train.bin
(default 50 % of the tokens) to limit forgetting. Output layout (STORE
CONTRACT): tokenizer.json (byte-identical copy of the shared tokenizer),
train.bin / train.val.bin (headerless little-endian uint16), train.meta.json,
manifest.json (provenance, sha256 of every file, licences, decontamination),
plus samples.jsonl (the exact documents and their source labels).

Source labels never mix: "self-generated" (model output that passed the
verifier), "supervised-reference" (generator reference solutions; warm-start
only) and "replay" (chunks of an existing store).
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from array import array
from pathlib import Path

from .tasks import ROOT, Decontaminator

FORGE = ROOT / "target" / "release" / "forge"
TOKENIZER = ROOT / "data" / "stores" / "base" / "tokenizer.json"
BOS, EOS, PAD = 8189, 8190, 8191
VOCAB = 8192
SOURCES = ("self-generated", "supervised-reference")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def forge_tokenize(texts: list[str], workdir: Path, *, forge: Path = FORGE, tokenizer: Path = TOKENIZER) -> list[list[int]]:
    """Token ids per text (no eos) via `forge tokenize --jsonl`."""
    if not texts:
        return []
    workdir.mkdir(parents=True, exist_ok=True)
    inp = workdir / "tokenize-in.jsonl"
    inp.write_text("".join(json.dumps(t) + "\n" for t in texts))
    try:
        out = subprocess.run([str(forge), "tokenize", "--tokenizer", str(tokenizer), "--jsonl", str(inp)],
                             check=True, capture_output=True, text=True).stdout
    finally:
        inp.unlink(missing_ok=True)
    ids = [json.loads(line) for line in out.splitlines() if line.strip()]
    if len(ids) != len(texts):
        raise RuntimeError(f"forge tokenize returned {len(ids)} rows for {len(texts)} texts")
    return ids


def _to_bytes(ids: list[int]) -> bytes:
    a = array("H", ids)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def read_tokens(path: Path, start: int = 0, count: int | None = None) -> list[int]:
    with open(path, "rb") as f:
        f.seek(2 * start)
        data = f.read(-1 if count is None else 2 * count)
    a = array("H")
    a.frombytes(data[: len(data) // 2 * 2])
    if sys.byteorder == "big":
        a.byteswap()
    return a.tolist()


def store_bins(meta_path: Path) -> tuple[dict, Path, Path]:
    meta = json.loads(Path(meta_path).read_text())
    if meta.get("dtype", "uint16") != "uint16":
        raise ValueError(f"{meta_path}: only uint16 stores are supported")
    d = Path(meta_path).parent
    return meta, d / Path(meta["bin"]).name, d / Path(meta["val_bin"]).name


def sample_replay(meta_path: Path, n_tokens: int, rng: random.Random, *, split: str = "train", chunk: int = 256) -> list[list[int]]:
    """Random chunks of an existing store, each aligned to start after an eos when
    one is near the start, and terminated by eos."""
    if n_tokens <= 0:
        return []
    _, train_bin, val_bin = store_bins(meta_path)
    path = train_bin if split == "train" else val_bin
    total = path.stat().st_size // 2
    if total < 2:
        return []
    chunk = min(chunk, total)
    out, got = [], 0
    while got < n_tokens:
        a = rng.randrange(0, total - chunk + 1)
        toks = read_tokens(path, a, chunk)
        head = toks[: max(1, chunk // 4)]
        if EOS in head:
            toks = toks[head.index(EOS) + 1:]
        need = n_tokens - got
        if len(toks) > need:  # the last chunk is trimmed so small stores still hit the requested ratio
            toks = toks[: max(1, need - 1)]
        if not toks:
            continue
        if toks[-1] != EOS:
            toks.append(EOS)
        out.append(toks)
        got += len(toks)
    return out


class _Decoder:
    """Lazy SCP BPE decoder (scripts/humaneval.py Tokenizer) for re-checking written units."""

    def __init__(self, tokenizer: Path):
        from .verify import Tokenizer

        self.tok = Tokenizer(Path(tokenizer))
        self.limit = 256 + len(self.tok.merges)

    def __call__(self, ids: list[int]) -> str:
        return self.tok.decode([i for i in ids if i < self.limit])


def build_store(out_dir: Path, docs: list[dict], *, name: str, replay_meta: Path | None = None, replay_ratio: float = 0.5,
                seed: int = 0, val_fraction: float = 0.1, min_split_tokens: int = 600, chunk_tokens: int = 256,
                decon: Decontaminator | None = None, forbidden: list[str] | tuple = (), tokenizer: Path = TOKENIZER,
                forge: Path = FORGE) -> dict:
    """Write an SCP store from `docs` ({"text", "source", ...}) plus replay; returns the manifest.

    Documents overlapping the eval sets (13-gram) or containing a forbidden
    string (held-out prompts) are dropped before tokenisation; replay chunks are
    decoded and checked the same way. After writing, every unit is decoded again
    and re-checked; the count lands in manifest["decontamination"]["remaining_overlaps"].
    """
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("replay_ratio must be in [0, 1)")
    if replay_ratio > 0 and replay_meta is None:
        raise ValueError("replay_ratio > 0 needs a replay store")
    from .tasks import default_decontaminator

    decon = decon or default_decontaminator()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_bytes = Path(tokenizer).read_bytes()
    if replay_meta is not None:
        rtok = Path(replay_meta).parent / "tokenizer.json"
        if rtok.exists() and rtok.read_bytes() != tok_bytes:
            raise ValueError(f"replay store tokenizer {rtok} differs from {tokenizer}")
    rng = random.Random(f"store|{name}|{seed}")
    stats = {"input_documents": len(docs), "dropped_contaminated": 0, "dropped_heldout": 0, "dropped_duplicate": 0,
             "replay_dropped_contaminated": 0}

    kept, seen = [], set()
    for d in docs:
        if d.get("source") not in SOURCES:
            raise ValueError(f"document source must be one of {SOURCES}, got {d.get('source')!r}")
        text = d["text"]
        if text in seen:
            stats["dropped_duplicate"] += 1
            continue
        if decon.contaminated(text):
            stats["dropped_contaminated"] += 1
            continue
        if any(f and f in text for f in forbidden):
            stats["dropped_heldout"] += 1
            continue
        seen.add(text)
        kept.append(d)
    rng.shuffle(kept)
    n_val = max(1, round(val_fraction * len(kept))) if len(kept) >= 10 and val_fraction > 0 else 0
    ids = forge_tokenize([d["text"] for d in kept], out_dir, forge=forge, tokenizer=tokenizer)
    units = {"train": [], "val": []}  # (source, ids incl. eos, doc or None)
    for i, d in enumerate(kept):
        units["val" if i < n_val else "train"].append((d["source"], ids[i] + [EOS], d))

    decode = _Decoder(tokenizer)
    replay_info = {"meta": str(replay_meta) if replay_meta else None, "ratio_target": replay_ratio, "chunk_tokens": chunk_tokens,
                   "chunks": {"train": 0, "val": 0}, "tokens": {"train": 0, "val": 0}, "topped_up_tokens": {"train": 0, "val": 0}}
    for split in ("train", "val"):
        new_tokens = sum(len(u[1]) for u in units[split])
        want = round(new_tokens * replay_ratio / (1.0 - replay_ratio)) if replay_ratio > 0 else 0
        short = max(0, min_split_tokens - new_tokens - want)
        if replay_meta is None:
            # No replay store: a small split repeats its own documents; an empty val
            # split is filled with copies of train documents (flagged in the manifest).
            if not units[split] and split == "val":
                units["val"] = list(units["train"])
                replay_info["val_is_train_copy"] = True
            if short and units[split]:
                base, i = list(units[split]), 0
                while sum(len(u[1]) for u in units[split]) < min_split_tokens:
                    units[split].append(base[i % len(base)])
                    i += 1
                replay_info["topped_up_tokens"][split] = sum(len(u[1]) for u in units[split]) - new_tokens
            continue
        target = want + short  # `short` tops a small split up to min_split_tokens with replay
        got = tries = 0
        while got < target and tries < 1000:
            for ch in sample_replay(replay_meta, target - got, rng, split=split, chunk=chunk_tokens):
                if decon.contaminated(decode(ch)):
                    stats["replay_dropped_contaminated"] += 1
                    continue
                units[split].append(("replay", ch, None))
                got += len(ch)
                replay_info["chunks"][split] += 1
            tries += 1
        replay_info["tokens"][split] = got
        replay_info["topped_up_tokens"][split] = short

    remaining = 0  # re-check of exactly what is written: decoded token ids of every unit
    for split in units:
        rng.shuffle(units[split])
        for src, u, _ in units[split]:
            text = decode(u)
            if decon.contaminated(text) or (src != "replay" and any(f and f in text for f in forbidden)):
                remaining += 1

    files = {"train": out_dir / "train.bin", "val": out_dir / "train.val.bin"}
    counts = {}
    for split, path in files.items():
        flat = [t for _, u, _ in units[split] for t in u]
        if len(flat) < max(min_split_tokens, 3):
            raise ValueError(f"{split} split has only {len(flat)} tokens (need {min_split_tokens})")
        if max(flat) >= VOCAB:
            raise ValueError("token id outside the vocabulary")
        path.write_bytes(_to_bytes(flat))
        counts[split] = len(flat)
    (out_dir / "tokenizer.json").write_bytes(tok_bytes)

    mix: dict[str, int] = {}
    token_mix: dict[str, int] = {}
    for src, u, _ in units["train"]:
        mix[src] = mix.get(src, 0) + 1
        token_mix[src] = token_mix.get(src, 0) + len(u)
    meta = {"tokens": counts["train"], "val_tokens": counts["val"], "dtype": "uint16", "vocab_size": VOCAB,
            "bin": "train.bin", "val_bin": "train.val.bin", "mix": mix,
            "mix_fractions": {k: v / max(1, sum(mix.values())) for k, v in mix.items()}}
    (out_dir / "train.meta.json").write_text(json.dumps(meta, indent=2))
    with open(out_dir / "samples.jsonl", "w") as f:
        for split in ("train", "val"):
            for src, u, d in units[split]:
                if d is not None:
                    f.write(json.dumps({"split": split, "source": src, "tokens": len(u),
                                        **{k: v for k, v in d.items() if k != "source"}}) + "\n")
    replay_manifest = None
    if replay_meta is not None and (Path(replay_meta).parent / "manifest.json").exists():
        rm = json.loads((Path(replay_meta).parent / "manifest.json").read_text())
        replay_manifest = {"store": rm.get("store"), "sha256": rm.get("sha256"), "pipeline": rm.get("pipeline")}
    manifest = {
        "store": name, "created": time.time(), "pipeline": "training/rsi/store.py (forge tokenize --jsonl + eos 8190, SCP CorpusStore layout)",
        "documents": {"train": sum(1 for s, _, _ in units["train"] if s != "replay"), "val": sum(1 for s, _, _ in units["val"] if s != "replay")},
        **meta, "token_mix": token_mix,
        "replay_token_fraction": token_mix.get("replay", 0) / max(1, counts["train"]),
        "sha256": {"train.bin": sha256_file(files["train"]), "train.val.bin": sha256_file(files["val"]),
                   "tokenizer.json": hashlib.sha256(tok_bytes).hexdigest(), "samples.jsonl": sha256_file(out_dir / "samples.jsonl")},
        "replay": {**replay_info, "source_manifest": replay_manifest},
        "filters": stats,
        "decontamination": {"method": "13-gram word overlap (lower-cased \\w+), as data/build_corpus.py", "against": decon.sources,
                            "heldout_strings": len(forbidden), "remaining_overlaps": remaining},
        "licences": {
            "self-generated": "model outputs produced and verified in this RSI round; prompts procedurally generated by training/rsi/tasks.py",
            "supervised-reference": "reference solutions procedurally generated by training/rsi/tasks.py (project-owned)",
            "replay": "inherited from the replay store (see its manifest.json and data/manifests/sources.json)",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
