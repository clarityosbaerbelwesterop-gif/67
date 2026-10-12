"""HumanEval / MBPP pass@1 (greedy) for forge checkpoints, with real execution.

Prompts are encoded with the shared SCP tokenizer (`forge tokenize`), completions come from
`forge generate --prompts-file` (weights loaded once), and every completion runs
against the official tests in an isolated Python subprocess (`-I`, empty env,
temp dir, CPU/memory/file-size limits, 10 s timeout). Results are measured,
including the expected zero for small models.

MBPP uses the standard test split (task ids 11-510) and the prompt of Austin
et al. (2021): the task text and the first test inside a docstring.

ARC-AGI-2 (`--suite arc-agi-2`) is the public evaluation set (120 tasks, every
test output scored by exact match, pass@2 as in the ARC Prize: one greedy and
one sampled attempt). The data is the arcagi2 file of the Apache-2.0 PyPI
package arckit==1.0.1, fetched once into data/benchmarks/. Grids are written as
rows of digits; a task whose prompt does not fit the model's context window
counts as failed and is reported under "not_fit".

Usage: python3 scripts/humaneval.py runs/quasnir-1/FINAL [--suite humaneval|mbpp] [--limit N] [--max-new 192]
"""
import argparse
import hashlib
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


ARC_FILE = ROOT / "data/benchmarks/arc-agi-2-eval.json"
ARCKIT = "arckit==1.0.1"


def arc_tasks() -> list:
    """[(task_id, test_index, prompt, expected_text)] of the ARC-AGI-2 public evaluation set."""
    if not ARC_FILE.exists():
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, "-m", "pip", "download", ARCKIT, "--no-deps", "-q", "-d", d], check=True)
            whl = next(Path(d).glob("arckit-*.whl"))
            data = json.loads(zipfile.ZipFile(whl).read("arckit/data/arcagi2_f3283f7.json"))
        ARC_FILE.parent.mkdir(parents=True, exist_ok=True)
        ARC_FILE.write_text(json.dumps(data["eval"]))
    grid = lambda g: "\n".join("".join(str(c) for c in row) for row in g)
    out = []
    for tid, t in sorted(json.loads(ARC_FILE.read_text()).items()):
        shots = "".join(f"in:\n{grid(e['input'])}\nout:\n{grid(e['output'])}\n\n" for e in t["train"])
        for i, e in enumerate(t["test"]):
            out.append((tid, i, shots + f"in:\n{grid(e['input'])}\nout:\n", grid(e["output"])))
    return out


def arc_grid(text: str) -> str:
    """The first grid of a completion: digit rows up to the first blank or non-digit line."""
    rows = []
    for line in text.split("\n"):
        if line and line.isdigit():
            rows.append(line)
        elif rows or line:
            break
    return "\n".join(rows)


def arc_main(args, tok) -> int:
    tasks = arc_tasks()[: args.limit]
    ctx = int(json.loads((Path(args.ckpt) / "state.json").read_text())["config"]["model"]["max_seq_len"])
    ids = tok.encode_many([t[2] for t in tasks])
    fit = [(t, p) for t, p in zip(tasks, ids) if len(p) + 16 <= ctx]
    eos = len(tok.merges) + 256 + 1
    hits = {}
    if fit:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("".join(json.dumps(p) + "\n" for _, p in fit))
            prompts = f.name
        max_new = max(16, min(args.max_new, ctx - max(len(p) for _, p in fit)))
        for attempt, extra in enumerate((["--temperature", "0"], ["--temperature", "0.7", "--seed", "1"])):
            out = subprocess.run([str(ROOT / "target/release/forge"), "generate", "--ckpt", args.ckpt, "--prompts-file", prompts,
                                  "--max-new", str(max_new), "--stop-id", str(eos)] + extra,
                                 check=True, capture_output=True, text=True).stdout
            gens = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
            for ((tid, i, _, want), _), g in zip(fit, gens):
                got = arc_grid(tok.decode([x for x in g["ids"] if x < 256 + len(tok.merges)]))
                hits.setdefault((tid, i), [False, False])[attempt] = got == want
        os.unlink(prompts)
    n = len(tasks)
    p1 = sum(h[0] for h in hits.values())
    p2 = sum(any(h) for h in hits.values())
    print(json.dumps({"ckpt": args.ckpt, "suite": "arc-agi-2", "problems": n, "not_fit": n - len(fit), "context": ctx,
                      "passed": p2, "pass@1": p1 / max(1, n), "pass@2": p2 / max(1, n), "decoding": "greedy + 1 sample (T 0.7)",
                      "data": ARCKIT, "data_sha256": hashlib.sha256(ARC_FILE.read_bytes()).hexdigest()}))
    return 0


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
    ap.add_argument("--suite", choices=["humaneval", "mbpp", "arc-agi-2"], default="humaneval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/stores/base/tokenizer.json"))
    args = ap.parse_args()
    tok = Tokenizer(Path(args.tokenizer))
    if args.suite == "arc-agi-2":
        return arc_main(args, tok)
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
