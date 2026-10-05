"""HumanEval pass@1 (greedy) for forge checkpoints, with real execution.

Prompts are encoded with the shared SCP tokenizer, completions come from
`forge generate --prompts-file` (weights loaded once), and every completion runs
against the official tests in an isolated Python subprocess (`-I`, empty env,
temp dir, CPU/memory/file-size limits, 10 s timeout). Results are measured,
including the expected zero for small models.

Usage: python3 scripts/humaneval.py runs/quasnir-1/FINAL [--limit 164] [--max-new 192]
"""
import argparse
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCP = Path(os.environ.get("SCP_ROOT", "/home/user/swarm-compute-protocol-")) / "model" / "scp_model"
STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert"]


def tokenizer(path: Path):
    pkg = types.ModuleType("scp_model")
    pkg.__path__ = [str(SCP)]
    sys.modules.setdefault("scp_model", pkg)
    spec = importlib.util.spec_from_file_location("scp_model.bpe", SCP / "bpe.py")
    bpe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpe)
    return bpe.BPETokenizer.load(str(path))


def limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def passes(program: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, env={}, capture_output=True, timeout=15, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--limit", type=int, default=164)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/stores/base/tokenizer.json"))
    args = ap.parse_args()
    tok = tokenizer(Path(args.tokenizer))
    probs = [json.loads(l) for l in (ROOT / "data/stores/evals/humaneval.jsonl").read_text().splitlines() if l.strip()][: args.limit]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for p in probs:
            f.write(json.dumps(tok.encode(p["prompt"])) + "\n")
        prompts = f.name
    eos = len(tok.merges) + 256 + 1
    out = subprocess.run([str(ROOT / "target/release/forge"), "generate", "--ckpt", args.ckpt, "--prompts-file", prompts,
                          "--max-new", str(args.max_new), "--stop-id", str(eos)], check=True, capture_output=True, text=True).stdout
    gens = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    solved = 0
    for p, g in zip(probs, gens):
        text = tok.decode([i for i in g["ids"] if i < 256 + len(tok.merges)])
        cut = min([text.find(s) for s in STOPS if s in text] + [len(text)])
        program = p["prompt"] + text[:cut] + "\n\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
        solved += passes(program)
    result = {"ckpt": args.ckpt, "suite": "humaneval", "problems": len(probs), "passed": solved, "pass@1": solved / max(1, len(probs)),
              "decoding": "greedy", "max_new_tokens": args.max_new}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
