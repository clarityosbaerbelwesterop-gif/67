"""HumanEval / MBPP pass@1 (greedy) for forge checkpoints, with real execution.

Prompts are encoded with the shared SCP tokenizer (`forge tokenize`), completions come from
`forge generate --prompts-file` (weights loaded once), and every completion runs
against the official tests in an isolated Python subprocess (`-I`, empty env,
temp dir, CPU/memory/file-size limits, 10 s timeout). Results are measured,
including the expected zero for small models.

MBPP uses the standard test split (task ids 11-510) and the prompt of Austin
et al. (2021): the task text and the first test inside a docstring.

Usage: python3 scripts/humaneval.py runs/quasnir-1/FINAL [--suite humaneval|mbpp] [--limit N] [--max-new 192]
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STOPS = {
    "humaneval": ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert"],
    "mbpp": ["\nassert", '\n"""', "\nif __name__", "\nprint(", "\n#"],
}


def problems(suite: str) -> list:
    rows = [json.loads(l) for l in (ROOT / f"data/stores/evals/{suite}.jsonl").read_text().splitlines() if l.strip()]
    if suite == "humaneval":
        return [{"prompt": p["prompt"], "tail": "\n\n" + p["test"] + f"\ncheck({p['entry_point']})\n"} for p in rows]
    out = []
    for p in rows:
        if 11 <= p["task_id"] <= 510:
            prompt = f'"""\n{p["text"]}\n{p["test_list"][0]}\n"""\n'
            tail = "\n\n" + p.get("test_setup_code", "") + "\n" + "\n".join(p["test_list"]) + "\n"
            out.append({"prompt": prompt, "tail": tail})
    return out


class Tokenizer:
    """SCP BPE: encoding by `forge tokenize` (Rust, parity-checked against
    scp_model.bpe), decoding from the merge table."""

    def __init__(self, path: Path):
        self.path = path
        data = json.loads(path.read_text())
        self.merges = sorted(data["merges"], key=lambda m: m[2])
        self.vocab = {i: bytes([i]) for i in range(256)}
        for a, b, idx in self.merges:
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    def encode_many(self, texts: list) -> list:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("".join(json.dumps(t) + "\n" for t in texts))
        out = subprocess.run([str(ROOT / "target/release/forge"), "tokenize", "--tokenizer", str(self.path), "--jsonl", f.name],
                             check=True, capture_output=True, text=True).stdout
        os.unlink(f.name)
        return [json.loads(l) for l in out.splitlines()]

    def decode(self, ids: list) -> str:
        return b"".join(self.vocab[i] for i in ids if i in self.vocab).decode("utf-8", errors="replace")


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
    ap.add_argument("--suite", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/stores/base/tokenizer.json"))
    args = ap.parse_args()
    tok = Tokenizer(Path(args.tokenizer))
    probs = problems(args.suite)[: args.limit]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for ids in tok.encode_many([p["prompt"] for p in probs]):
            f.write(json.dumps(ids) + "\n")
        prompts = f.name
    eos = len(tok.merges) + 256 + 1
    out = subprocess.run([str(ROOT / "target/release/forge"), "generate", "--ckpt", args.ckpt, "--prompts-file", prompts,
                          "--max-new", str(args.max_new), "--stop-id", str(eos)], check=True, capture_output=True, text=True).stdout
    gens = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    solved = 0
    for p, g in zip(probs, gens):
        text = tok.decode([i for i in g["ids"] if i < 256 + len(tok.merges)])
        cut = min([text.find(s) for s in STOPS[args.suite] if s in text] + [len(text)])
        program = p["prompt"] + text[:cut] + p["tail"]
        solved += passes(program)
    result = {"ckpt": args.ckpt, "suite": args.suite, "problems": len(probs), "passed": solved, "pass@1": solved / max(1, len(probs)),
              "decoding": "greedy", "max_new_tokens": args.max_new}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
