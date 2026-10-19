"""Hard coding test: HumanEval-hard and MBPP-hard, differential testing in the style of EvalPlus.

The official HumanEval and MBPP tests are thin (about 7 and 3 asserts per task), so many wrong
programs pass them. This builder adds extra test inputs per task and takes the expected outputs
from the reference solution:

1. seeds: the arguments of every call the official tests make, recorded in a sandboxed run of
   the reference solution;
2. mutation: type-aware variants of the seeds that stay inside the value ranges, characters,
   element pools and lengths the official tests use (ints within [min, max] of the seen ints,
   strings from seen characters, lists from seen elements). This keeps inputs in the task's
   domain; EvalPlus uses hand-written contracts for the same purpose, which we do not have;
3. oracle: the reference solution runs every input twice under a 2 s per-call alarm; only
   inputs with an identical, literal (ast.literal_eval-able) result both times are kept;
4. MBPP also gets its official challenge_test_list.

A candidate passes a hard task only if it passes the official tests AND matches the reference on
every extra input (floats with rel/abs tolerance 1e-6). The same sandbox as scripts/humaneval.py
runs everything.

Validation (`--validate`): the reference solutions must pass 100 %; AST mutants of the reference
(flipped comparisons, +-1 on integer constants, swapped and/or, + <-> -) that pass the official
tests are counted, and the share of them the extra tests kill shows how much harder the test is.

Output: data/benchmarks/{humaneval,mbpp}-hard.jsonl and data/benchmarks/hardbench.manifest.json
(sha256, counts, validation). Usage: python3 data/build_hardbench.py [--per-task 40] [--validate]
"""
import argparse
import ast
import copy
import hashlib
import json
import random
import resource
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "data/stores/evals"
OUT = ROOT / "data/benchmarks"
SEED = 67067


def limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 22, 1 << 22))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def run(program: str, timeout: int = 90) -> tuple:
    """Runs a program isolated (-I, empty env, temp dir, rlimits). Returns (ok, stdout)."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, env={}, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return False, ""
        return r.returncode == 0, r.stdout


def entry_of_mbpp(p: dict) -> str:
    for node in ast.walk(ast.parse(p["test_list"][0])):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
    raise ValueError(f"no call in MBPP task {p['task_id']}")


def tasks() -> list:
    out = []
    for p in (json.loads(l) for l in (EVALS / "humaneval.jsonl").read_text().splitlines() if l.strip()):
        out.append({"suite": "humaneval", "id": p["task_id"], "entry": p["entry_point"], "prompt": p["prompt"],
                    "ref": p["prompt"] + p["canonical_solution"], "setup": "",
                    "tests": p["test"] + f"\ncheck({p['entry_point']})\n", "challenge": []})
    for p in (json.loads(l) for l in (EVALS / "mbpp.jsonl").read_text().splitlines() if l.strip()):
        if 11 <= p["task_id"] <= 510:
            out.append({"suite": "mbpp", "id": p["task_id"], "entry": entry_of_mbpp(p),
                        "prompt": f'"""\n{p["text"]}\n{p["test_list"][0]}\n"""\n',
                        "ref": p["code"].replace("\r\n", "\n"), "setup": p.get("test_setup_code", ""),
                        "tests": "\n".join(p["test_list"]) + "\n", "challenge": p.get("challenge_test_list") or []})
    return out


# --- 1. seeds -------------------------------------------------------------------------------

RECORD = r'''
import ast as _ast, copy as _copy, json as _json
_CALLS = []
def _rec(fn):
    def w(*a):
        try:
            r = repr(a)
            if _ast.literal_eval(r) == a and r not in _CALLS:
                _CALLS.append(r)
        except Exception:
            pass
        return fn(*_copy.deepcopy(a))
    return w
'''


def seeds(t: dict) -> list:
    program = t["ref"] + "\n" + RECORD + f"\n_REF = {t['entry']}\n{t['entry']} = _rec(_REF)\n" + t["setup"] + "\n" + \
        t["tests"] + "\nprint('@@' + _json.dumps(_CALLS[:200]))\n"
    ok, out = run(program)
    line = [l for l in out.splitlines() if l.startswith("@@")]
    return json.loads(line[-1][2:]) if ok and line else []


# --- 2. in-domain mutation ------------------------------------------------------------------

def pools(values: list) -> dict:
    """Value pools per type from all seed arguments (recursively)."""
    pool = {"int": [], "float": [], "chars": set(), "str": [], "len": []}
    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            pool["int"].append(v)
        elif isinstance(v, float):
            pool["float"].append(v)
        elif isinstance(v, str):
            pool["str"].append(v); pool["chars"].update(v); pool["len"].append(len(v))
        elif isinstance(v, (list, tuple, set)):
            pool["len"].append(len(v))
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(k); walk(x)
    for v in values:
        walk(v)
    return pool


def mutate(v, pool: dict, rng: random.Random, depth: int = 0):
    if isinstance(v, bool):
        return (not v) if rng.random() < 0.5 else v
    if isinstance(v, int):
        lo, hi = min(pool["int"]), max(pool["int"])
        choice = rng.random()
        if choice < 0.4:
            return rng.randint(lo, hi)
        if choice < 0.7:
            return max(lo, min(hi, v + rng.choice([-2, -1, 1, 2])))
        return rng.choice(pool["int"])
    if isinstance(v, float):
        lo, hi = min(pool["float"]), max(pool["float"])
        return round(rng.uniform(lo, hi), rng.choice([1, 2, 3])) if hi > lo else v
    if isinstance(v, str):
        if " " in v.strip() and rng.random() < 0.8:
            # whitespace-separated inputs (word lists, paren groups, number strings): recombine whole
            # tokens from the seen strings, which keeps each token valid
            words, seen = v.split(" "), [w for x in pool["str"] for w in x.split(" ") if w]
            op = rng.randrange(4)
            if op == 0 and len(words) > 1:
                words.pop(rng.randrange(len(words)))
            elif op == 1 and seen:
                words.insert(rng.randrange(len(words) + 1), rng.choice(seen))
            elif op == 2 and len(words) > 1:
                rng.shuffle(words)
            elif seen:
                words[rng.randrange(len(words))] = rng.choice(seen)
            return " ".join(words)
        chars = sorted(pool["chars"]) or ["a"]
        s = list(v)
        op = rng.randrange(5)
        if op == 0 and s:
            s.pop(rng.randrange(len(s)))
        elif op == 1:
            s.insert(rng.randrange(len(s) + 1), rng.choice(chars))
        elif op == 2 and s:
            s[rng.randrange(len(s))] = rng.choice(chars)
        elif op == 3 and len(s) > 1:
            rng.shuffle(s)
        else:
            return rng.choice(pool["str"])
        return "".join(s)
    if isinstance(v, (list, tuple)):
        items = [copy.deepcopy(x) for x in v]
        op = rng.randrange(5)
        max_len = max(pool["len"] or [len(items)]) * 2
        if op == 0 and items:
            items.pop(rng.randrange(len(items)))
        elif op == 1 and items and len(items) < max_len:
            items.insert(rng.randrange(len(items) + 1), copy.deepcopy(rng.choice(items)))
        elif op == 2 and len(items) > 1:
            rng.shuffle(items)
        elif items and depth < 3:
            i = rng.randrange(len(items))
            items[i] = mutate(items[i], pool, rng, depth + 1)
        return type(v)(items)
    return v  # dicts, sets, None: kept as seen


def candidates(seed_reprs: list, n: int, rng: random.Random) -> list:
    args = [ast.literal_eval(s) for s in seed_reprs]
    pool = pools([a for t in args for a in t])
    seen, out = set(seed_reprs), []
    for _ in range(n * 20):
        if len(out) >= n * 3:
            break
        base = rng.choice(args)
        new = tuple(mutate(a, pool, rng) if rng.random() < 0.7 else copy.deepcopy(a) for a in base)
        r = repr(new)
        if r not in seen and len(r) <= 2000:
            seen.add(r)
            out.append(r)
    return out


# --- 3. oracle ------------------------------------------------------------------------------

ORACLE = r'''
import ast as _ast, json as _json, math as _math, signal as _signal
def _alarm(*_):
    raise TimeoutError
_signal.signal(_signal.SIGALRM, _alarm)
def _finite(v):
    if isinstance(v, float):
        return _math.isfinite(v)
    if isinstance(v, (list, tuple, set)):
        return all(_finite(x) for x in v)
    if isinstance(v, dict):
        return all(_finite(k) and _finite(x) for k, x in v.items())
    return True
_RES = []
for _a in _json.loads(_INPUTS):
    try:
        _signal.alarm(2)
        _r = _REF(*_ast.literal_eval(_a))
        _signal.alarm(0)
        _s = repr(_r)
        _RES.append([_a, _s] if _finite(_r) and _ast.literal_eval(_s) == _r else None)
    except BaseException:
        _signal.alarm(0)
        _RES.append(None)
print('@@' + _json.dumps(_RES))
'''


def oracle(t: dict, inputs: list) -> list:
    program = t["ref"] + "\n" + t["setup"] + f"\n_REF = {t['entry']}\n_INPUTS = {json.dumps(json.dumps(inputs))}\n" + ORACLE
    outs = []
    for _ in range(2):
        ok, out = run(program, timeout=max(30, 3 * len(inputs)))
        line = [l for l in out.splitlines() if l.startswith("@@")]
        outs.append(json.loads(line[-1][2:]) if ok and line else [None] * len(inputs))
    return [a for a, b in zip(*outs) if a is not None and a == b]


# --- 4. candidate check (appended to every program) ------------------------------------------

CHECK = r'''
import ast as _ast, json as _json, math as _math
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)
def _close(a, b):
    if _num(a) and _num(b) and (isinstance(a, float) or isinstance(b, float)):
        return _math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b
for _a, _o in _json.loads(_HARD):
    _got = ENTRY(*_ast.literal_eval(_a))
    assert _close(_got, _ast.literal_eval(_o)), _a
'''


def hard_tail(t: dict, pairs: list) -> str:
    official = t["setup"] + "\n" + t["tests"] + "\n".join(t["challenge"]) + "\n"
    if t["suite"] == "humaneval":
        official = "\n\n" + t["tests"]
    return official + f"\n_HARD = {json.dumps(json.dumps(pairs))}\n" + CHECK.replace("ENTRY", t["entry"])


def build(t: dict, per_task: int) -> dict:
    rng = random.Random(f"{SEED}:{t['id']}")
    # official challenge asserts that the reference itself fails are dropped
    t = dict(t, challenge=[c for c in t["challenge"] if run(t["ref"] + "\n" + t["setup"] + "\n" + c + "\n", timeout=20)[0]])
    s = seeds(t)
    pairs = oracle(t, candidates(s, per_task, rng))[:per_task] if s else []
    fallback = False
    if not run(t["ref"] + "\n" + hard_tail(t, pairs), timeout=90)[0]:
        # the reference must pass its own hard tail; if it does not (call-order effects), keep the official tests only
        pairs, fallback = [], True
    return {"task_id": t["id"], "entry_point": t["entry"], "prompt": t["prompt"], "tail": hard_tail(t, pairs),
            "seeds": len(s), "extra_tests": len(pairs), "challenge_tests": len(t["challenge"]), "fallback": fallback}


# --- validation -----------------------------------------------------------------------------

class Mutant(ast.NodeTransformer):
    SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq, ast.Add: ast.Sub, ast.Sub: ast.Add, ast.And: ast.Or, ast.Or: ast.And}

    def __init__(self, target: int):
        self.target, self.i = target, 0

    def hit(self) -> bool:
        self.i += 1
        return self.i - 1 == self.target

    def visit_Compare(self, node):
        self.generic_visit(node)
        if type(node.ops[0]) in self.SWAP and self.hit():
            node.ops[0] = self.SWAP[type(node.ops[0])]()
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in self.SWAP and self.hit():
            node.op = self.SWAP[type(node.op)]()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if type(node.op) in self.SWAP and self.hit():
            node.op = self.SWAP[type(node.op)]()
        return node

    def visit_Constant(self, node):
        if type(node.value) is int and self.hit():
            return ast.copy_location(ast.Constant(node.value + 1), node)
        return node


def mutants(ref: str, k: int) -> list:
    out = []
    for target in range(200):
        m = Mutant(target)
        tree = m.visit(ast.parse(ref))
        if m.i <= target:
            break
        out.append(ast.unparse(tree))
    random.Random(SEED).shuffle(out)
    return out[:k]


def validate(t: dict, row: dict, k: int) -> dict:
    official = t["ref"] + "\n" + (("\n\n" + t["tests"]) if t["suite"] == "humaneval" else t["setup"] + "\n" + t["tests"])
    ref_ok = run(t["ref"] + "\n" + row["tail"], timeout=60)[0]
    base_pass = hard_kill = 0
    for m in mutants(t["ref"], k):
        if run(m + "\n" + official.split(t["ref"], 1)[1], timeout=20)[0]:
            base_pass += 1
            hard_kill += not run(m + "\n" + row["tail"], timeout=60)[0]
    return {"ref_ok": ref_ok, "mutants_passing_official": base_pass, "killed_by_hard": hard_kill}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=40)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--mutants", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N tasks per suite (smoke runs)")
    a = ap.parse_args()
    all_tasks = tasks()
    if a.limit:
        all_tasks = [t for s in ("humaneval", "mbpp") for t in [x for x in all_tasks if x["suite"] == s][: a.limit]]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda t: build(t, a.per_task), all_tasks))
        checks = list(ex.map(lambda tr: validate(tr[0], tr[1], a.mutants), zip(all_tasks, rows))) if a.validate else []
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"type": "hardbench", "seed": SEED, "per_task": a.per_task, "suites": {}}
    for suite in ("humaneval", "mbpp"):
        idx = [i for i, t in enumerate(all_tasks) if t["suite"] == suite]
        path = OUT / f"{suite}-hard.jsonl"
        path.write_text("".join(json.dumps(rows[i]) + "\n" for i in idx))
        entry = {"file": str(path.relative_to(ROOT)), "sha256": sha256(path), "tasks": len(idx),
                 "extra_tests": sum(rows[i]["extra_tests"] for i in idx),
                 "tasks_without_extra": sum(rows[i]["extra_tests"] == 0 for i in idx),
                 "fallbacks": sum(rows[i]["fallback"] for i in idx),
                 "challenge_tests": sum(rows[i]["challenge_tests"] for i in idx)}
        if checks:
            c = [checks[i] for i in idx]
            mp = sum(x["mutants_passing_official"] for x in c)
            entry["validation"] = {"reference_pass": sum(x["ref_ok"] for x in c), "mutants_passing_official": mp,
                                   "killed_by_hard": sum(x["killed_by_hard"] for x in c),
                                   "kill_rate": round(sum(x["killed_by_hard"] for x in c) / max(1, mp), 4)}
        manifest["suites"][suite] = entry
    if not a.limit:
        (OUT / "hardbench.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
