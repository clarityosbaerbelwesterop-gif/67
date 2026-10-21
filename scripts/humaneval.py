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

Thinking (all optional; the defaults are the plain greedy pass@1):
- `--samples N`: greedy plus N-1 sampled candidates per task.
- `--rounds R`: up to R-1 further rounds of N candidates, only for tasks whose visible tests still fail.
- `--self-tests K`: the model, as tester, writes up to K `assert f(...) == ...` lines per task.
- `--repair`: one repair round; the model sees its best failed attempt and the check it fails.
Visible tests are the ones the prompt itself shows (MBPP's assert line, HumanEval's doctest or
`f(x) => y` examples; 130 of 164 HumanEval and all MBPP tasks have some). A candidate is picked by its
visible tests, then by the size of the group of candidates that return the same on every probe call,
weighted by the self-tests that group satisfies (execution agreement as in AlphaCode and CodeT). Only
the picked candidate runs against the hidden tests, so the score stays a pass@1; `ablation_pass@1` adds
greedy and plain best-of-N on the same candidates. `--oracle` also scores every candidate (a ceiling,
not a pass@1). `--cache-dir` keeps completions, so the -hard suites reuse those of their base suite.

Usage: python3 scripts/humaneval.py runs/quasnir-1/FINAL [--suite humaneval|mbpp|humaneval-hard|mbpp-hard] [--limit N] [--max-new 192]
       [--samples 8 --rounds 2 --self-tests 8 --repair --temperature 0.8 --seed 1 --oracle --cache-dir DIR]
"""
import argparse
import ast
import builtins
import concurrent.futures
import doctest
import hashlib
import json
import os
import re
import resource
import secrets
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
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
            out = subprocess.run(generate_cmd(args.ckpt) + ["--prompts-file", prompts, "--max-new", str(max_new),
                                                            "--stop-id", str(eos)] + extra,
                                 check=True, capture_output=True, text=True, cwd=ROOT).stdout
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


def probe_of(src: str, want) -> tuple | None:
    """(call, expected) that a test evaluates: a doctest pair as it is, `assert f(x) == y` split at the ==."""
    if want is not None:
        return src, want
    try:
        node = ast.parse(src).body[0]
    except (SyntaxError, IndexError):
        return None
    t = getattr(node, "test", None)
    if isinstance(node, ast.Assert) and isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
        return ast.get_source_segment(src, t.left), ast.get_source_segment(src, t.comparators[0])
    return None


def callee(src: str) -> str | None:
    try:
        node = ast.parse(src, mode="eval").body
    except SyntaxError:
        return None
    return node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else None


def entry_of(p: dict, vis: list) -> str | None:
    """The function under test: HumanEval names it, MBPP shows it in its visible assert."""
    if p.get("entry_point"):
        return p["entry_point"]
    calls = [probe_of(s, w) for s, w in vis]
    return next((callee(c[0]) for c in calls if c and callee(c[0])), None)


def tester_prompt(p: dict, ep: str) -> str:
    """The tester role: the task, a placeholder body, then a test block the model continues."""
    return p["prompt"] + ("    ...\n" if f"def {ep}(" in p["prompt"] else "") + f"\n\n# Tests\nassert {ep}("


PROBE = """


def _forge_probe(_vis, _probes, _tag):
    print(_tag, "loaded", flush=True)
    for _i, (_src, _want) in enumerate(_vis):
        try:
            if _want is None:
                exec(_src, globals())
                _ok = True
            else:
                _ok = bool(eval(_src, globals()) == eval(_want, globals()))
        except BaseException:
            _ok = False
        print(_tag, "v", _i, int(_ok), flush=True)
    for _i, (_src, _want) in enumerate(_probes):
        try:
            _got = eval(_src, globals())
            _r = repr(_got)[:200]
            _ok = _want is not None and bool(_got == eval(_want, globals()))
        except BaseException as _e:
            _r, _ok = "raised " + type(_e).__name__, False
        print(_tag, "p", _i, int(_ok), repr(_r), flush=True)


_forge_probe({vis!r}, {probes!r}, {tag!r})
"""


def evaluate(job: tuple) -> dict:
    """One candidate in the sandbox: which visible tests pass, and what it returns on every probe call (visible
    calls and self-tests). `tag` is a fresh nonce, so printing from the candidate counts for nothing."""
    body, vis, probes, tag = job
    out = run(body + PROBE.format(vis=vis, probes=probes, tag=tag), VISIBLE_CPU_S)[1].splitlines()
    v, sig, ok = [0] * len(vis), [None] * len(probes), [0] * len(probes)
    for line in out:
        f = line.split(" ", 4)
        try:
            if f[0] == tag and f[1] == "v":
                v[int(f[2])] = int(f[3])
            elif f[0] == tag and f[1] == "p":
                sig[int(f[2])], ok[int(f[2])] = ast.literal_eval(f[4]), int(f[3])
        except (IndexError, ValueError, SyntaxError):
            continue  # a line cut off by the CPU limit
    return {"passed": sum(v), "loaded": int(f"{tag} loaded" in out), "vis": tuple(v), "sig": tuple(sig), "ok": tuple(ok)}


def choose(evals: list, n_self: int = 0, consensus: bool = True) -> int:
    """Most visible tests passed, then loads cleanly, then (with consensus) the largest group of candidates that
    return the same on every probe, weighted by the self-tests that group satisfies. A group whose probes all
    raise gets no weight. Ties keep the earlier candidate, greedy first."""
    groups = Counter(e["sig"] for e in evals if e["loaded"])

    def key(j):
        e = evals[j]
        alive = e["loaded"] and any(r is not None and not r.startswith("raised ") for r in e["sig"])
        c = groups[e["sig"]] * (1 + sum(e["ok"][len(e["ok"]) - n_self:])) if consensus and alive else 0
        return e["passed"], e["loaded"], c, -j

    return max(range(len(evals)), key=key)


def repair_prompt(p: dict, body: str, e: dict, vis: list, vis_probe: list) -> str | None:
    """The repair role: the failed attempt, the first visible check it fails and what it gave, then the task again."""
    i = next((k for k, ok in enumerate(e["vis"]) if not ok), None)
    if i is None:
        return None
    src, want = vis[i]
    note = "# This check fails: " + " ".join((src + (f" == {want}" if want is not None else "")).split())
    j = vis_probe[i]
    if j is not None and e["sig"][j] is not None:
        note += f"\n# It gives: {e['sig'][j]}"
    return body + "\n\n\n" + note + "\n# Fixed version:\n" + p["prompt"]


def hidden_pass(job: tuple) -> bool:
    return passes(*job)


def ckpt_id(ckpt: str, tokenizer: str) -> list:
    files = sorted(Path(ckpt).iterdir()) + [Path(tokenizer)]
    return [str(Path(ckpt).resolve())] + [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files if f.is_file()]


def generate_cmd(ckpt: str) -> list:
    """The decoder: forge generate (CPU), or the command in GEN_CMD that takes the same arguments, e.g.
    GEN_CMD="python3 -m accel.generate --batch 128" for KV-cache decoding on GPUs (run from the repository root)."""
    pre = shlex.split(os.environ.get("GEN_CMD", "")) or [str(ROOT / "target/release/forge"), "generate"]
    return pre + ["--ckpt", str(Path(ckpt).resolve())]


def completions(args, tok, texts: list, temperature: float, seed: int, copies: int, max_new: int | None = None) -> list:
    """`copies` decoded completions per prompt (line j of the batch decodes with seed + j). With --cache-dir the
    result is reused by any suite with the same prompts, so the -hard variants cost no extra decoding."""
    max_new = max_new or args.max_new
    if not texts:
        return []
    key = hashlib.sha256(json.dumps([ckpt_id(args.ckpt, args.tokenizer), texts, max_new, temperature, seed,
                                     copies]).encode()).hexdigest()[:32]
    cache = Path(args.cache_dir) / f"{key}.json" if args.cache_dir else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for ids in tok.encode_many(texts):
            f.write((json.dumps(ids) + "\n") * copies)
        prompts = f.name
    eos = len(tok.merges) + 256 + 1
    cmd = generate_cmd(args.ckpt) + ["--prompts-file", prompts, "--max-new", str(max_new), "--stop-id", str(eos)]
    if temperature > 0:
        cmd += ["--temperature", str(temperature), "--seed", str(seed)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout
    os.unlink(prompts)
    gens = [tok.decode([i for i in json.loads(l)["ids"] if i < 256 + len(tok.merges)]) for l in out.splitlines() if l.startswith("{")]
    res = [gens[i * copies:(i + 1) * copies] for i in range(len(texts))]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(res))
    return res


def self_tests(args, tok, probs: list, eps: list, vis: list) -> list:
    """Up to --self-tests distinct `assert f(...) == ...` lines per task, written by the model as tester. Their
    expected values may be wrong; selection only rewards candidates that agree with them as a group."""
    todo = [i for i, ep in enumerate(eps) if ep]
    gens = completions(args, tok, [tester_prompt(probs[i], eps[i]) for i in todo], args.temperature,
                       args.seed + TESTER * STREAM, 2 * args.self_tests, max_new=64)
    out = [[] for _ in probs]
    for i, g in zip(todo, gens):
        seen = {c[0] for c in (probe_of(s, w) for s, w in vis[i]) if c}
        for t in g:
            c = probe_of(f"assert {eps[i]}(" + t.split("\n", 1)[0].strip(), None)
            if c and callee(c[0]) == eps[i] and c[0] not in seen and _closed(c, eps[i]) and len(out[i]) < args.self_tests:
                seen.add(c[0])
                out[i].append(c)
    return out


def _closed(call: tuple, ep: str) -> bool:
    """A self-test that names nothing but the function under test and builtins (no undefined variables)."""
    try:
        names = {n.id for src in call for n in ast.walk(ast.parse(src, mode="eval")) if isinstance(n, ast.Name)}
    except SyntaxError:
        return False
    return names <= {ep} | set(dir(builtins))


STREAM, TESTER, REPAIR = 1_000_003, 900, 901  # seed streams: round r samples use seed + r * STREAM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--suite", choices=["humaneval", "mbpp", "humaneval-hard", "mbpp-hard", "arc-agi-2"], default="humaneval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/stores/base/tokenizer.json"))
    ap.add_argument("--samples", type=int, default=1, help="candidates per task and round: greedy plus samples-1 in round 1")
    ap.add_argument("--rounds", type=int, default=1, help="further rounds of --samples candidates for tasks not yet solved on their visible tests")
    ap.add_argument("--self-tests", type=int, default=0, help="test lines the model writes per task as tester")
    ap.add_argument("--repair", action="store_true", help="one repair round from the best failed attempt and its failing check")
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
    n = len(probs)
    texts = [p["prompt"] for p in probs]
    stops = STOPS[args.suite.removesuffix("-hard")]
    body = lambda i, t: probs[i]["prompt"] + t[:min([t.find(s) for s in stops if s in t] + [len(t)])]
    thinking = args.samples > 1 or args.rounds > 1 or args.self_tests > 0 or args.repair
    cands = completions(args, tok, texts, 0.0, 0, 1)
    if args.samples > 1:
        cands = [c + more for c, more in zip(cands, completions(args, tok, texts, args.temperature, args.seed, args.samples - 1))]
    bodies = [[body(i, t) for t in c] for i, c in enumerate(cands)]
    vis = [visible_tests(p["prompt"], p.get("entry_point")) for p in probs]
    eps = [entry_of(p, v) for p, v in zip(probs, vis)]
    selft = self_tests(args, tok, probs, eps, vis) if args.self_tests > 0 else [[] for _ in probs]
    vis_probe, probes = [], []
    for v, st in zip(vis, selft):
        calls = [probe_of(s, w) for s, w in v]
        kept = [c for c in calls if c]
        idx = iter(range(len(kept)))
        vis_probe.append([next(idx) if c else None for c in calls])
        probes.append(kept + st)
    evals = [[] for _ in probs]
    repairs = 0
    with concurrent.futures.ProcessPoolExecutor(args.jobs) as pool:
        def evaluate_new():
            todo = [(i, k) for i in range(n) for k in range(len(evals[i]), len(bodies[i]))]
            jobs = [(bodies[i][k], vis[i], probes[i], secrets.token_hex(8)) for i, k in todo]
            for (i, _), e in zip(todo, pool.map(evaluate, jobs, chunksize=4)):
                evals[i].append(e)

        solved = lambda i: bool(vis[i]) and max(e["passed"] for e in evals[i]) == len(vis[i])
        if thinking:
            evaluate_new()
            for r in range(1, args.rounds):  # more thinking only where the visible tests still fail
                todo = [i for i in range(n) if not solved(i)]
                for i, c in zip(todo, completions(args, tok, [texts[i] for i in todo], args.temperature,
                                                  args.seed + r * STREAM, args.samples)):
                    bodies[i] += [body(i, t) for t in c]
                evaluate_new()
            if args.repair:
                todo = [i for i in range(n) if vis[i] and not solved(i)]
                best = {i: choose(evals[i], len(selft[i])) for i in todo}
                rp = {i: repair_prompt(probs[i], bodies[i][best[i]], evals[i][best[i]], vis[i], vis_probe[i]) for i in todo}
                todo = [i for i in todo if rp[i]]
                repairs = len(todo)
                for i, c in zip(todo, completions(args, tok, [rp[i] for i in todo], args.temperature,
                                                  args.seed + REPAIR * STREAM, args.samples)):
                    bodies[i] += [body(i, t) for t in c]
                evaluate_new()
        picks = {"greedy": [0] * n}
        if thinking:
            picks["best_of_n_visible"] = [choose(evals[i][:args.samples], consensus=False) for i in range(n)]
            picks["full"] = [choose(evals[i], len(selft[i])) for i in range(n)]
        todo = sorted({(i, j) for pk in picks.values() for i, j in enumerate(pk)}
                      | ({(i, j) for i in range(n) for j in range(len(bodies[i]))} if args.oracle else set()))
        hidden = dict(zip(todo, pool.map(hidden_pass, [(bodies[i][j] + probs[i]["tail"], CPU_S) for i, j in todo], chunksize=4)))
    rate = {k: sum(hidden[(i, j)] for i, j in enumerate(pk)) for k, pk in picks.items()}
    final = "full" if thinking else "greedy"
    m = max(1, n)
    result = {"ckpt": args.ckpt, "suite": args.suite, "problems": n, "passed": rate[final], "pass@1": rate[final] / m,
              "decoding": "greedy", "max_new_tokens": args.max_new}
    if thinking:
        result.update({"decoding": f"thinking: greedy + {args.samples - 1} samples (T {args.temperature}, top-k 40), "
                                   f"{args.rounds} round(s), {args.self_tests} self-tests, repair {'on' if args.repair else 'off'}; "
                                   "picked by the prompt's visible tests, then agreement with the self-tests; hidden tests scored once",
                       "samples": args.samples, "rounds": args.rounds, "self_tests": args.self_tests, "repair": args.repair,
                       "temperature": args.temperature, "seed": args.seed,
                       "greedy_pass@1": rate["greedy"] / m,
                       "ablation_pass@1": {k: v / m for k, v in rate.items()},
                       "candidates_decoded": sum(map(len, bodies)), "self_tests_kept": sum(map(len, selft)),
                       "repair_prompts": repairs, "tasks_with_visible_tests": sum(map(bool, vis)),
                       "picked_not_greedy": sum(j > 0 for j in picks["full"])})
    if args.oracle and thinking:
        result["oracle_any_pass"] = sum(any(hidden[(i, j)] for j in range(len(bodies[i]))) for i in range(n)) / m
        result["sampled_pass@1"] = (sum(hidden[(i, j)] for i in range(n) for j in range(1, len(bodies[i])))
                                    / max(1, sum(len(b) - 1 for b in bodies)))
        result["oracle_note"] = "uses the hidden tests: a ceiling for search and RL, not a pass@1"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
