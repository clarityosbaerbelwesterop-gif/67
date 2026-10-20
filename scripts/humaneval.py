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

HumanEval-hard and MBPP-hard (`--suite humaneval-hard|mbpp-hard`) use the same prompts plus ~37
extra differential tests per task from data/build_hardbench.py (data/benchmarks/*-hard.jsonl);
a task passes only if the official tests and every extra test pass.

Best-of-N (`--samples N`): greedy plus N-1 sampled completions per task. One is picked by the tests the
prompt itself shows (MBPP's assert line, HumanEval's doctest or `f(x) => y` examples; 130 of 164
HumanEval and all MBPP tasks have some), and only that one runs against the hidden tests, so the score
is still a pass@1. `--oracle` also scores every candidate (a ceiling, not a pass@1). `--cache-dir`
keeps completions so the -hard suites reuse those of their base suite.

Usage: python3 scripts/humaneval.py runs/quasnir-1/FINAL [--suite humaneval|mbpp|humaneval-hard|mbpp-hard] [--limit N] [--max-new 192]
       [--samples 8 --temperature 0.8 --seed 1 --oracle --cache-dir DIR]
"""
import argparse
import concurrent.futures
import doctest
import hashlib
import json
import os
import re
import resource
import secrets
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
    if suite.endswith("-hard"):
        return [json.loads(l) for l in (ROOT / f"data/benchmarks/{suite}.jsonl").read_text().splitlines() if l.strip()]
    rows = [json.loads(l) for l in (ROOT / f"data/stores/evals/{suite}.jsonl").read_text().splitlines() if l.strip()]
    if suite == "humaneval":
        return [{"prompt": p["prompt"], "entry_point": p["entry_point"],
                 "tail": "\n\n" + p["test"] + f"\ncheck({p['entry_point']})\n"} for p in rows]
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


CPU_S = 10
VISIBLE_CPU_S = 3  # visible checks are a few small calls; a looping candidate must not stall the selection


def limits(cpu: int | None = None) -> None:
    cpu = cpu or CPU_S
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def run(program: str, cpu: int) -> tuple:
    """(exit code 0, stdout) of `program` in the sandbox; a timeout keeps the output printed so far."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, env={}, capture_output=True, timeout=cpu + 5,
                               preexec_fn=lambda: limits(cpu))
        except subprocess.TimeoutExpired as e:
            return False, (e.stdout or b"").decode(errors="replace")
        return r.returncode == 0, r.stdout.decode(errors="replace")


def passes(program: str, cpu: int | None = None) -> bool:
    return run(program, cpu or CPU_S)[0]


# Best-of-N: the model writes N candidates and picks one by the tests its prompt shows (MBPP's assert line,
# HumanEval's doctest or `f(x) => y` examples). The hidden tests run once, on the picked candidate, so the
# score stays a pass@1. Visible tests are the only signal a user of the model would also have.
_ARROW = re.compile(r"\s*(?:==>|==|=>|->|➞|should return|returns?)\s*(.+?)[\s.,;]*$")


def _compiles(src: str, mode: str) -> bool:
    try:
        compile(src, "<visible>", mode)
        return True
    except (SyntaxError, ValueError):
        return False


def _call_end(s: str) -> int:
    """Index after the call that opens `s`, or -1."""
    depth, quote, i = 0, None, 0
    while i < len(s):
        c = s[i]
        if quote:
            i += c == "\\"
            quote = None if c == quote else quote
        elif c in "'\"":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def visible_tests(prompt: str, entry_point: str | None = None) -> list:
    """[(source, expected)] taken from the prompt text only; expected None: the statement must not raise."""
    out = [(ln.strip(), None) for ln in prompt.splitlines() if ln.strip().startswith("assert ") and _compiles(ln.strip(), "exec")]
    try:
        examples = doctest.DocTestParser().get_examples(prompt)
    except ValueError:
        examples = []
    for e in examples:
        src = e.source.strip()
        want = "\n".join(re.split(r"\n\s*(?:\"\"\"|\'\'\')", "\n" + e.want)[0].splitlines()).strip()  # the docstring may close right after
        if want and _compiles(src, "eval") and _compiles(want, "eval"):
            out.append((src, want))
        elif not want and _compiles(src, "eval") and "==" in src:
            out.append((f"assert {src}", None))  # `>>> f(x) == y` written without an expected line
        elif not want and _compiles(src, "exec"):
            out.append((src, None))
    if entry_point and not out:
        for ln in prompt.splitlines():
            s = ln.strip()
            end = _call_end(s) if s.startswith(entry_point + "(") else -1
            m = _ARROW.match(s[end:]) if end > 0 else None
            if m and _compiles(s[:end], "eval") and _compiles(m.group(1), "eval"):
                out.append((s[:end], m.group(1)))
    return out


CHECK = """


def _forge_visible(_tests, _tag):
    print(_tag, "loaded", flush=True)
    for _src, _want in _tests:
        try:
            if _want is None:
                exec(_src, globals())
                _ok = True
            else:
                _ok = bool(eval(_src, globals()) == eval(_want, globals()))
        except BaseException:
            _ok = False
        if _ok:
            print(_tag, "pass", flush=True)


_forge_visible({tests!r}, {tag!r})
"""


def visible_score(job: tuple) -> tuple:
    """(visible tests passed, module loaded) of one candidate; `tag` is a fresh nonce the candidate cannot print."""
    body, tests, tag = job
    out = run(body + CHECK.format(tests=tests, tag=tag), VISIBLE_CPU_S)[1].splitlines()
    return out.count(f"{tag} pass"), int(f"{tag} loaded" in out)


def choose(scores: list) -> int:
    """Most visible tests passed, then loads cleanly; ties keep the earlier candidate, greedy first."""
    return max(range(len(scores)), key=lambda j: (scores[j], -j))


def hidden_pass(job: tuple) -> bool:
    return passes(*job)


def ckpt_id(ckpt: str, tokenizer: str) -> list:
    files = sorted(Path(ckpt).iterdir()) + [Path(tokenizer)]
    return [str(Path(ckpt).resolve())] + [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files if f.is_file()]


def completions(args, tok, texts: list, temperature: float, seed: int, copies: int) -> list:
    """`copies` decoded completions per prompt (line j of the batch decodes with seed + j). With --cache-dir the
    result is reused by any suite with the same prompts, so the -hard variants cost no extra decoding."""
    key = hashlib.sha256(json.dumps([ckpt_id(args.ckpt, args.tokenizer), texts, args.max_new, temperature, seed,
                                     copies]).encode()).hexdigest()[:32]
    cache = Path(args.cache_dir) / f"{key}.json" if args.cache_dir else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for ids in tok.encode_many(texts):
            f.write((json.dumps(ids) + "\n") * copies)
        prompts = f.name
    eos = len(tok.merges) + 256 + 1
    cmd = [str(ROOT / "target/release/forge"), "generate", "--ckpt", args.ckpt, "--prompts-file", prompts,
           "--max-new", str(args.max_new), "--stop-id", str(eos)]
    if temperature > 0:
        cmd += ["--temperature", str(temperature), "--seed", str(seed)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    os.unlink(prompts)
    gens = [tok.decode([i for i in json.loads(l)["ids"] if i < 256 + len(tok.merges)]) for l in out.splitlines() if l.startswith("{")]
    res = [gens[i * copies:(i + 1) * copies] for i in range(len(texts))]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(res))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--suite", choices=["humaneval", "mbpp", "humaneval-hard", "mbpp-hard", "arc-agi-2"], default="humaneval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/stores/base/tokenizer.json"))
    ap.add_argument("--samples", type=int, default=1, help="candidates per task: greedy plus samples-1 sampled ones")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--oracle", action="store_true", help="also run the hidden tests on every candidate (ceiling, not pass@1)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    tok = Tokenizer(Path(args.tokenizer))
    if args.suite == "arc-agi-2":
        return arc_main(args, tok)
    global CPU_S
    if args.suite.endswith("-hard"):
        CPU_S = 40  # the extra tests run inside the same sandboxed process
    probs = problems(args.suite)[: args.limit]
    texts = [p["prompt"] for p in probs]
    cands = completions(args, tok, texts, 0.0, 0, 1)
    if args.samples > 1:
        cands = [c + more for c, more in zip(cands, completions(args, tok, texts, args.temperature, args.seed, args.samples - 1))]
    stops = STOPS[args.suite.removesuffix("-hard")]
    bodies = [[p["prompt"] + t[:min([t.find(s) for s in stops if s in t] + [len(t)])] for t in c] for p, c in zip(probs, cands)]
    pick = [0] * len(probs)
    with concurrent.futures.ProcessPoolExecutor(args.jobs) as pool:
        if args.samples > 1:
            tests = [visible_tests(p["prompt"], p.get("entry_point")) for p in probs]
            jobs = [(b, t, secrets.token_hex(8)) for bs, t in zip(bodies, tests) for b in bs]
            scores = list(pool.map(visible_score, jobs, chunksize=4))
            pick = [choose(scores[i * args.samples:(i + 1) * args.samples]) for i in range(len(probs))]
        todo = sorted({(i, 0) for i in range(len(probs))} | {(i, pick[i]) for i in range(len(probs))}
                      | ({(i, j) for i in range(len(probs)) for j in range(args.samples)} if args.oracle else set()))
        hidden = dict(zip(todo, pool.map(hidden_pass, [(bodies[i][j] + probs[i]["tail"], CPU_S) for i, j in todo], chunksize=4)))
    solved = sum(hidden[(i, pick[i])] for i in range(len(probs)))
    n = max(1, len(probs))
    result = {"ckpt": args.ckpt, "suite": args.suite, "problems": len(probs), "passed": solved, "pass@1": solved / n,
              "decoding": "greedy", "max_new_tokens": args.max_new}
    if args.samples > 1:
        result.update({"decoding": f"best-of-{args.samples}: greedy + {args.samples - 1} samples (T {args.temperature}, "
                                   "top-k 40), picked by the prompt's own visible tests; hidden tests scored once",
                       "samples": args.samples, "temperature": args.temperature, "seed": args.seed,
                       "greedy_pass@1": sum(hidden[(i, 0)] for i in range(len(probs))) / n,
                       "tasks_with_visible_tests": sum(bool(t) for t in tests),
                       "picked_not_greedy": sum(j > 0 for j in pick)})
    if args.oracle and args.samples > 1:
        result["oracle_any_pass"] = sum(any(hidden[(i, j)] for j in range(args.samples)) for i in range(len(probs))) / n
        result["sampled_pass@1"] = sum(hidden[(i, j)] for i in range(len(probs)) for j in range(1, args.samples)) / (n * (args.samples - 1))
        result["oracle_note"] = "uses the hidden tests: a ceiling for search and RL, not a pass@1"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
