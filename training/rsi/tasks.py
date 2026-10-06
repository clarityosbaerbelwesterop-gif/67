"""Procedurally generated, verifiable tasks for controlled RSI.

Families
  code   Python function completion (Quasnir, Darus). The prompt is a signature
         and a docstring with two examples, ending at the closing quotes; a
         candidate is the function body (starting with a newline) and is checked
         by hidden asserts in the sandbox (training/rsi/verify.py).
  text   exact-answer questions (Rouge 1, Darus): multi-digit arithmetic,
         counting, sorting short lists, reversing, unit conversion, simple
         logic. Answers are compared after normalisation.
  mixed  even seeds -> code, odd seeds -> text.

Every task is a pure function of (family, level, seed): the generator draws
from random.Random(f"{family}|{level}|{seed}"), which is stable across
processes and Python hash seeds. Difficulty levels 0..9; higher levels compose
operations or use small algorithms.

Seed spaces: TRAIN_SEEDS = [0, 1e9) and HELDOUT_SEEDS = [1e9, 2e9) never mix,
and training additionally excludes every task whose prompt equals a held-out
prompt (small parameter spaces at low levels can repeat a prompt), so held-out
pass rates are measured on problems that never entered training.

Decontamination: every generated task (prompt + reference + tests) is checked
for 13-gram overlap with data/stores/evals/*.jsonl, with the word n-grams of
data/build_corpus.py (lower-cased \\w+ tokens), hashed with a stable hash, and
dropped on overlap. Missing eval files are an error (fail closed).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = ROOT / "data" / "stores" / "evals"
LEVELS = tuple(range(10))
FAMILIES = ("code", "text", "mixed")
SEED_SPACE = 1_000_000_000
TRAIN_SEEDS = range(0, SEED_SPACE)
HELDOUT_SEEDS = range(SEED_SPACE, 2 * SEED_SPACE)


def space_of(seed: int) -> str:
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in HELDOUT_SEEDS:
        return "heldout"
    raise ValueError(f"seed {seed} outside the train [0, {SEED_SPACE}) and held-out [{SEED_SPACE}, {2 * SEED_SPACE}) spaces")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str  # concrete family: code | text
    level: int
    seed: int
    template: str
    prompt: str
    reference: str  # programmatic reference completion (supervised warm-start only)
    tests: str = ""  # code: hidden asserts
    entry_point: str = ""
    answer: str = ""  # text: expected answer

    @property
    def space(self) -> str:
        return space_of(self.seed)

    @property
    def key(self) -> str:
        """Identity of the problem: the family and the whitespace-normalised prompt."""
        return hashlib.sha256((self.family + "\n" + " ".join(self.prompt.split())).encode()).hexdigest()

    def text_for_decontamination(self) -> str:
        return "\n".join((self.prompt, self.reference, self.tests, self.answer))

    def to_json(self) -> dict:
        return {**asdict(self), "space": self.space, "key": self.key}


# --------------------------------------------------------------------------- decontamination

WORD = re.compile(r"\w+")


def ngrams(text: str, n: int = 13) -> set[int]:
    """Word 13-grams as in data/build_corpus.ngrams, with a process-stable hash."""
    w = WORD.findall(text.lower())
    return {int.from_bytes(hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest(), "little")
            for i in range(len(w) - n + 1)}


def _row_texts(row: dict) -> list[str]:
    parts = []
    for v in row.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(x for x in v if isinstance(x, str))
    return [" ".join(parts)]


class Decontaminator:
    """13-gram overlap test against the benchmark sets."""

    def __init__(self, banned: set[int], sources: dict[str, int]):
        self.banned = banned
        self.sources = sources

    @classmethod
    def from_eval_dir(cls, d: Path = EVALS_DIR) -> "Decontaminator":
        files = sorted(Path(d).glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no eval sets in {d}: refusing to generate training data without decontamination")
        banned: set[int] = set()
        sources = {}
        for f in files:
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            for r in rows:
                for t in _row_texts(r):
                    banned |= ngrams(t)
            sources[f.stem] = len(rows)
        return cls(banned, sources)

    @classmethod
    def from_texts(cls, texts: list[str]) -> "Decontaminator":
        banned: set[int] = set()
        for t in texts:
            banned |= ngrams(t)
        return cls(banned, {"custom": len(texts)})

    def contaminated(self, text: str) -> bool:
        return not self.banned.isdisjoint(ngrams(text))


_DECON: Decontaminator | None = None


def default_decontaminator() -> Decontaminator:
    global _DECON
    if _DECON is None:
        _DECON = Decontaminator.from_eval_dir()
    return _DECON


# --------------------------------------------------------------------------- word lists

WORDS = """apple river stone cloud garden window bottle planet forest candle silver orange marble pencil rocket
summer winter castle dragon button yellow butter coffee ladder mirror pepper rabbit tomato violin wallet anchor
bridge cactus desert engine falcon guitar hammer island jacket kettle lemon magnet needle oyster parrot saddle
tunnel umbrella velvet walnut zipper basket circle doctor flower meadow puzzle shadow tiger maple ocean lantern
""".split()
SHORT_WORDS = "cat dog sun map cup hat pen box fox owl bee ant jar key log net pie rug toy van".split()
NAMES = "Alice Bob Carol Dave Erin Frank Grace Heidi Ivan Judy Oscar Peggy Trent Victor Wendy Zoe".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _mixcase(r: random.Random, w: str) -> str:
    k = r.randrange(3)
    return w if k == 0 else (w.capitalize() if k == 1 else "".join(c.upper() if r.random() < 0.3 else c for c in w))


def _phrase(r: random.Random, k: int) -> str:
    return " ".join(r.sample(WORDS, k))


# --------------------------------------------------------------------------- code family

TESTS_HEADER = (
    "def _rsi_check(got, want):\n"
    "    return type(got) is type(want) and repr(got) == repr(want)\n\n\n"
)


def _code_task(name: str, params: list[str], desc: str, body: list[str], fn: Callable, inputs: list[tuple],
               n_examples: int = 2) -> dict:
    """Render signature + docstring with examples, reference body and hidden asserts."""
    seen, uniq = set(), []
    for a in inputs:
        k = repr(a)
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    outs = [fn(*a) for a in uniq]
    call = lambda a: f"{name}({', '.join(repr(x) for x in a)})"  # noqa: E731
    lines = [f"def {name}({', '.join(params)}):"]
    wrapped = textwrap.wrap(desc, 84)
    lines.append('    """' + wrapped[0])
    lines.extend("    " + w for w in wrapped[1:])
    lines.append("")
    for a, o in list(zip(uniq, outs))[:n_examples]:
        lines.append(f"    >>> {call(a)}")
        lines.append(f"    {o!r}")
    lines.append('    """')
    # The prompt stops at the closing quotes: the newline + indentation that
    # starts the body is one BPE token, so the model has to produce it itself.
    prompt = "\n".join(lines)
    reference = "".join("\n    " + b for b in body) + "\n"
    tests = TESTS_HEADER + "".join(f"assert _rsi_check({call(a)}, {o!r}), {call(a)!r}\n" for a, o in zip(uniq, outs))
    return {"prompt": prompt, "reference": reference, "tests": tests, "entry_point": name}


# Integer pipeline operations: (constant sampler, phrase, expression over {x}, python).
INT_OPS = {
    "add": (lambda r: r.randint(1, 20), "add {c}", "{x} + {c}", lambda x, c: x + c),
    "sub": (lambda r: r.randint(1, 20), "subtract {c}", "{x} - {c}", lambda x, c: x - c),
    "mul": (lambda r: r.randint(2, 9), "multiply by {c}", "{x} * {c}", lambda x, c: x * c),
    "floordiv": (lambda r: r.randint(2, 7), "divide by {c} rounding down (floor division)", "{x} // {c}", lambda x, c: x // c),
    "mod": (lambda r: r.randint(2, 9), "take the remainder of division by {c}", "{x} % {c}", lambda x, c: x % c),
    "square": (None, "square it", "{x} * {x}", lambda x, c: x * x),
    "abs": (None, "take the absolute value", "abs({x})", lambda x, c: abs(x)),
    "neg": (None, "negate it", "-{x}", lambda x, c: -x),
    "cap": (lambda r: r.randint(5, 40), "cap it at {c} (keep the smaller of the value and {c})", "min({x}, {c})", lambda x, c: min(x, c)),
    "atleast": (lambda r: r.randint(-5, 10), "raise it to at least {c} (keep the larger of the value and {c})", "max({x}, {c})",
                lambda x, c: max(x, c)),
    "halve_even": (lambda r: r.randint(1, 9), "halve it if it is even, otherwise add {c}", "{x} // 2 if {x} % 2 == 0 else {x} + {c}",
                   lambda x, c: x // 2 if x % 2 == 0 else x + c),
    "digit_sum": (None, "replace it by the sum of its decimal digits (ignoring the sign)", "sum(int(d) for d in str(abs({x})))",
                  lambda x, c: sum(int(d) for d in str(abs(x)))),
}
INT_BASIC = ["add", "sub", "mul"]
INT_WIDE = INT_BASIC + ["floordiv", "mod", "square", "abs", "neg", "cap", "atleast"]
INT_ALL = INT_WIDE + ["halve_even", "digit_sum"]
INT_NAMES = {"add": "add_constant", "sub": "subtract_constant", "mul": "scale", "floordiv": "floor_divide", "mod": "remainder",
             "square": "square", "abs": "absolute", "neg": "negate", "cap": "cap_value", "atleast": "at_least",
             "halve_even": "halve_or_add", "digit_sum": "digit_sum"}


def _int_pipeline(r: random.Random, pool: list[str], k: int) -> dict:
    keys: list[str] = []
    while len(keys) < k:
        op = r.choice(pool)
        if op == "square" and "square" in keys:
            continue
        if keys and op == keys[-1]:
            continue
        keys.append(op)
    consts = [INT_OPS[o][0](r) if INT_OPS[o][0] else None for o in keys]
    phrases = [INT_OPS[o][1].format(c=c) for o, c in zip(keys, consts)]

    def fn(n: int) -> int:
        x = n
        for o, c in zip(keys, consts):
            x = INT_OPS[o][3](x, c)
        return x

    if k == 1:
        name = INT_NAMES[keys[0]]
        desc = f"Take the integer n, {phrases[0]}, and return the result."
        body = ["return " + INT_OPS[keys[0]][2].format(x="n", c=consts[0])]
    else:
        name = "transform"
        desc = "Take the integer n, " + ", then ".join(phrases) + ", and return the result."
        body = ["x = n"] + ["x = " + INT_OPS[o][2].format(x="x", c=c) for o, c in zip(keys, consts)] + ["return x"]
    lo = -15 if any(o in ("abs", "neg", "atleast", "digit_sum") for o in keys) else 0
    hi = 25 if "square" in keys else 60
    inputs = [(v,) for v in r.sample(range(lo, hi), 6)]
    return _code_task(name, ["n"], desc, body, fn, inputs)


def _two_arg(r: random.Random) -> dict:
    c1, c2 = r.randint(2, 9), r.randint(2, 9)
    kind = r.randrange(3)
    if kind == 0:
        desc, body, fn = f"Return a times {c1} plus b times {c2}.", [f"return a * {c1} + b * {c2}"], lambda a, b: a * c1 + b * c2
    elif kind == 1:
        desc, body, fn = f"Return the sum of a and b, multiplied by {c1}.", [f"return (a + b) * {c1}"], lambda a, b: (a + b) * c1
    else:
        desc, body, fn = f"Return a minus b, plus {c1}.", [f"return a - b + {c1}"], lambda a, b: a - b + c1
    inputs = [(r.randint(0, 30), r.randint(0, 30)) for _ in range(6)]
    return _code_task("combine", ["a", "b"], desc, body, fn, inputs)


# List pipeline operations: (constant sampler, phrase, body lines over xs with {c}, python).
LIST_OPS = {
    "map_add": (lambda r: r.randint(1, 9), "add {c} to every element", ["xs = [v + {c} for v in xs]"], lambda xs, c: [v + c for v in xs]),
    "map_mul": (lambda r: r.randint(2, 5), "multiply every element by {c}", ["xs = [v * {c} for v in xs]"], lambda xs, c: [v * c for v in xs]),
    "map_square": (None, "square every element", ["xs = [v * v for v in xs]"], lambda xs, c: [v * v for v in xs]),
    "map_neg": (None, "negate every element", ["xs = [-v for v in xs]"], lambda xs, c: [-v for v in xs]),
    "map_abs": (None, "replace every element by its absolute value", ["xs = [abs(v) for v in xs]"], lambda xs, c: [abs(v) for v in xs]),
    "map_mod": (lambda r: r.randint(2, 7), "replace every element by its remainder of division by {c}", ["xs = [v % {c} for v in xs]"],
                lambda xs, c: [v % c for v in xs]),
    "keep_even": (None, "keep only the even elements", ["xs = [v for v in xs if v % 2 == 0]"], lambda xs, c: [v for v in xs if v % 2 == 0]),
    "keep_odd": (None, "keep only the odd elements", ["xs = [v for v in xs if v % 2 != 0]"], lambda xs, c: [v for v in xs if v % 2 != 0]),
    "keep_gt": (lambda r: r.randint(0, 10), "keep only the elements greater than {c}", ["xs = [v for v in xs if v > {c}]"],
                lambda xs, c: [v for v in xs if v > c]),
    "keep_lt": (lambda r: r.randint(3, 15), "keep only the elements less than {c}", ["xs = [v for v in xs if v < {c}]"],
                lambda xs, c: [v for v in xs if v < c]),
    "keep_pos": (None, "keep only the positive elements", ["xs = [v for v in xs if v > 0]"], lambda xs, c: [v for v in xs if v > 0]),
    "sort": (None, "sort it in ascending order", ["xs = sorted(xs)"], lambda xs, c: sorted(xs)),
    "sort_desc": (None, "sort it in descending order", ["xs = sorted(xs, reverse=True)"], lambda xs, c: sorted(xs, reverse=True)),
    "reverse": (None, "reverse it", ["xs = xs[::-1]"], lambda xs, c: xs[::-1]),
    "dedupe": (None, "remove duplicates, keeping the first occurrence of each value", ["xs = list(dict.fromkeys(xs))"],
               lambda xs, c: list(dict.fromkeys(xs))),
    "take": (lambda r: r.randint(2, 4), "keep only the first {c} elements", ["xs = xs[:{c}]"], lambda xs, c: xs[:c]),
    "drop": (lambda r: r.randint(1, 3), "drop the first {c} elements", ["xs = xs[{c}:]"], lambda xs, c: xs[c:]),
    "cumsum": (None, "replace it by its running (cumulative) sums",
               ["acc = 0", "out = []", "for v in xs:", "    acc += v", "    out.append(acc)", "xs = out"],
               lambda xs, c: [sum(xs[:i + 1]) for i in range(len(xs))]),
}
LIST_SIMPLE = ["map_add", "map_mul", "map_square", "keep_even", "keep_odd", "keep_gt", "sort", "sort_desc", "reverse"]
LIST_ALL = list(LIST_OPS)
REDUCERS = {
    "sum": (None, "return the sum of the elements", ["return sum(xs)"], lambda xs, c: sum(xs)),
    "len": (None, "return how many elements there are", ["return len(xs)"], lambda xs, c: len(xs)),
    "max": (None, "return the largest element (0 if the list is empty)", ["return max(xs, default=0)"], lambda xs, c: max(xs, default=0)),
    "min": (None, "return the smallest element (0 if the list is empty)", ["return min(xs, default=0)"], lambda xs, c: min(xs, default=0)),
    "count_gt": (lambda r: r.randint(0, 10), "return how many elements are greater than {c}", ["return sum(1 for v in xs if v > {c})"],
                 lambda xs, c: sum(1 for v in xs if v > c)),
    "first": (None, "return the first element (-1 if the list is empty)", ["return xs[0] if xs else -1"], lambda xs, c: xs[0] if xs else -1),
    "product": (None, "return the product of the elements (1 for an empty list)", ["p = 1", "for v in xs:", "    p *= v", "return p"],
                lambda xs, c: math.prod(xs)),
    "list": (None, "return the resulting list", ["return xs"], lambda xs, c: list(xs)),
}


def _list_pipeline(r: random.Random, pool: list[str], k: int, reducers: list[str] | None = None) -> dict:
    keys: list[str] = []
    while len(keys) < k:
        op = r.choice(pool)
        if op in keys:
            continue
        keys.append(op)
    red = r.choice(reducers or list(REDUCERS))
    if red == "product" and any(o in ("map_square", "map_mul", "cumsum") for o in keys):
        red = "sum"
    consts = [LIST_OPS[o][0](r) if LIST_OPS[o][0] else None for o in keys]
    rc = REDUCERS[red][0](r) if REDUCERS[red][0] else None

    def fn(xs: list) -> object:
        for o, c in zip(keys, consts):
            xs = LIST_OPS[o][3](list(xs), c)
        return REDUCERS[red][3](list(xs), rc)

    phrases = [LIST_OPS[o][1].format(c=c) for o, c in zip(keys, consts)]
    final = REDUCERS[red][1].format(c=rc)
    if phrases:
        desc = "Given a list of integers xs, " + ", then ".join(phrases) + "; finally " + final + "."
    else:
        desc = "Given a list of integers xs, " + final + "."
    body = [ln.format(c=c) for o, c in zip(keys, consts) for ln in LIST_OPS[o][2]] + [ln.format(c=rc) for ln in REDUCERS[red][2]]
    lo = -6 if any(o in ("map_neg", "map_abs", "keep_pos", "map_mod") for o in keys) else 0
    inputs = [([r.randint(lo, 20) for _ in range(r.randint(4, 6))],) for _ in range(2)]
    inputs += [([r.randint(lo, 20) for _ in range(r.randint(0, 8))],) for _ in range(3)] + [([],)]
    name = {"sum": "sum_after", "len": "count_after", "max": "max_after", "min": "min_after", "count_gt": "count_greater",
            "first": "first_after", "product": "product_after", "list": "transform_list"}[red] if keys else \
        {"sum": "list_sum", "len": "list_length", "max": "list_max", "min": "list_min", "count_gt": "count_greater",
         "first": "first_element", "product": "list_product", "list": "copy_list"}[red]
    return _code_task(name, ["xs"], desc, body, fn, inputs)


# String pipeline operations: (constant sampler, phrase, body line over s, python).
_VOW = "aeiou"
STR_OPS = {
    "upper": (None, "convert it to upper case", "s = s.upper()", lambda s, c: s.upper()),
    "lower": (None, "convert it to lower case", "s = s.lower()", lambda s, c: s.lower()),
    "reverse": (None, "reverse it", "s = s[::-1]", lambda s, c: s[::-1]),
    "replace": (lambda r: tuple(r.sample("aeiorstn", 2)), "replace every {c0!r} with {c1!r}", "s = s.replace({c0!r}, {c1!r})",
                lambda s, c: s.replace(c[0], c[1])),
    "no_vowels": (None, "remove all vowels (a, e, i, o, u in either case)", "s = ''.join(ch for ch in s if ch.lower() not in 'aeiou')",
                  lambda s, c: "".join(ch for ch in s if ch.lower() not in _VOW)),
    "swapcase": (None, "swap the case of every letter", "s = s.swapcase()", lambda s, c: s.swapcase()),
    "suffix": (lambda r: (r.choice(SHORT_WORDS),), "append {c0!r} to the end", "s = s + {c0!r}", lambda s, c: s + c[0]),
    "prefix": (lambda r: (r.choice(SHORT_WORDS),), "put {c0!r} in front", "s = {c0!r} + s", lambda s, c: c[0] + s),
    "every_other": (None, "keep every other character, starting with the first", "s = s[::2]", lambda s, c: s[::2]),
    "double": (None, "repeat every character twice", "s = ''.join(ch * 2 for ch in s)", lambda s, c: "".join(ch * 2 for ch in s)),
    "caesar": (lambda r: (r.randint(1, 5),), "shift every lowercase letter {c0} places forward in the alphabet, wrapping from 'z' to 'a'",
               "s = ''.join(chr((ord(ch) - 97 + {c0}) % 26 + 97) if 'a' <= ch <= 'z' else ch for ch in s)",
               lambda s, c: "".join(chr((ord(ch) - 97 + c[0]) % 26 + 97) if "a" <= ch <= "z" else ch for ch in s)),
    "dedupe": (None, "remove repeated characters, keeping the first occurrence of each", "s = ''.join(dict.fromkeys(s))",
               lambda s, c: "".join(dict.fromkeys(s))),
}
STR_FINAL = {
    "str": (None, "return the resulting string", "return s", lambda s, c: s),
    "len": (None, "return its length", "return len(s)", lambda s, c: len(s)),
    "count": (lambda r: (r.choice("aeinorst"),), "return how many times {c0!r} occurs in it", "return s.count({c0!r})",
              lambda s, c: s.count(c[0])),
    "vowels": (None, "return how many vowels (a, e, i, o, u in either case) it contains", "return sum(1 for ch in s.lower() if ch in 'aeiou')",
               lambda s, c: sum(1 for ch in s.lower() if ch in _VOW)),
    "palin": (None, "return True if it reads the same forwards and backwards, otherwise False", "return s == s[::-1]",
              lambda s, c: s == s[::-1]),
}
STR_L0 = ["upper", "lower", "reverse"]
STR_L1 = ["replace", "no_vowels", "swapcase", "suffix", "prefix", "every_other", "double"]
STR_ALL = list(STR_OPS)


def _fmt(t: str, c) -> str:
    c = c or ()
    return t.format(c0=c[0] if len(c) > 0 else None, c1=c[1] if len(c) > 1 else None)


def _str_pipeline(r: random.Random, pool: list[str], k: int, finals: list[str]) -> dict:
    keys: list[str] = []
    while len(keys) < k:
        op = r.choice(pool)
        if op in keys:
            continue
        keys.append(op)
    fin = r.choice(finals)
    consts = [STR_OPS[o][0](r) if STR_OPS[o][0] else None for o in keys]
    fc = STR_FINAL[fin][0](r) if STR_FINAL[fin][0] else None

    def fn(s: str) -> object:
        for o, c in zip(keys, consts):
            s = STR_OPS[o][3](s, c)
        return STR_FINAL[fin][3](s, fc)

    phrases = [_fmt(STR_OPS[o][1], c) for o, c in zip(keys, consts)]
    final = _fmt(STR_FINAL[fin][1], fc)
    desc = "Given a string s, " + (", then ".join(phrases) + "; finally " if phrases else "") + final + "."
    body = [_fmt(STR_OPS[o][2], c) for o, c in zip(keys, consts)] + [_fmt(STR_FINAL[fin][2], fc)]
    words = r.sample(WORDS, 6)
    inputs = [(_mixcase(r, w),) for w in words[:4]] + [(" ".join(words[4:6]),), ("",)]
    if fin == "palin":
        inputs[1] = ("level",) if not keys else inputs[1]
    name = "transform_text" if keys else {"str": "identity", "len": "text_length", "count": "count_letter",
                                          "vowels": "count_vowels", "palin": "is_palindrome"}[fin]
    if len(keys) == 1 and fin == "str":
        name = {"upper": "to_upper", "lower": "to_lower", "reverse": "reverse_text", "replace": "replace_letter",
                "no_vowels": "remove_vowels", "swapcase": "swap_case", "suffix": "add_suffix", "prefix": "add_prefix",
                "every_other": "every_other", "double": "double_letters", "caesar": "caesar_shift", "dedupe": "unique_chars"}[keys[0]]
    return _code_task(name, ["s"], desc, body, fn, inputs)


def _ints(r: random.Random, lo: int, hi: int, k: int) -> list[tuple]:
    return [(v,) for v in r.sample(range(lo, hi), k)]


def _cond(r: random.Random) -> dict:
    t = r.randrange(9)
    if t == 0:
        return _code_task("parity", ["n"], "Return 'even' if n is even and 'odd' otherwise.",
                          ["return 'even' if n % 2 == 0 else 'odd'"], lambda n: "even" if n % 2 == 0 else "odd", _ints(r, -20, 60, 6))
    if t == 1:
        return _code_task("sign", ["n"], "Return 'positive' if n > 0, 'negative' if n < 0 and 'zero' if n == 0.",
                          ["if n > 0:", "    return 'positive'", "if n < 0:", "    return 'negative'", "return 'zero'"],
                          lambda n: "positive" if n > 0 else ("negative" if n < 0 else "zero"),
                          [(r.randint(1, 50),), (r.randint(-50, -1),), (0,), (r.randint(-9, 9),), (r.randint(-90, 90),)])
    if t == 2:
        lo = r.randint(-5, 10)
        hi = lo + r.randint(5, 20)
        return _code_task("clamp", ["n"], f"Clamp n into the range [{lo}, {hi}]: return {lo} if n is below {lo}, {hi} if n is above {hi}, "
                          "and n otherwise.", [f"return max({lo}, min({hi}, n))"], lambda n: max(lo, min(hi, n)),
                          [(lo - r.randint(1, 9),), (hi + r.randint(1, 9),), ((lo + hi) // 2,), (lo,), (hi,), (r.randint(-20, 40),)])
    if t == 3:
        big = r.random() < 0.5
        word = "larger" if big else "smaller"
        return _code_task(word, ["a", "b"], f"Return the {word} of the two integers a and b.",
                          [f"return a if a {'>' if big else '<'} b else b"], (max if big else min),
                          [(r.randint(-20, 40), r.randint(-20, 40)) for _ in range(6)])
    if t == 4:
        c = r.randint(40, 70)
        return _code_task("grade", ["score"], f"Return 'pass' if score is at least {c} and 'fail' otherwise.",
                          [f"return 'pass' if score >= {c} else 'fail'"], lambda s: "pass" if s >= c else "fail",
                          [(c,), (c - 1,), (r.randint(0, 100),), (r.randint(0, 100),), (100,), (0,)])
    if t == 5:
        c = r.randint(3, 7)
        return _code_task("fizz", ["n"], f"Return 'fizz' if n is divisible by {c}; otherwise return n converted to a string.",
                          [f"return 'fizz' if n % {c} == 0 else str(n)"], lambda n: "fizz" if n % c == 0 else str(n),
                          [(c * r.randint(1, 9),), (c * r.randint(1, 9) + 1,)] + _ints(r, 1, 80, 4))
    if t == 6:
        return _int_pipeline(r, ["halve_even"], 1)
    if t == 7:
        c = r.randint(3, 6)
        words = r.sample(WORDS + SHORT_WORDS, 6)
        return _code_task("shout", ["s"], f"Return s in upper case if it is longer than {c} characters, otherwise return s unchanged.",
                          [f"return s.upper() if len(s) > {c} else s"], lambda s: s.upper() if len(s) > c else s,
                          [(w,) for w in words])
    lo = r.randint(0, 20)
    hi = lo + r.randint(5, 30)
    return _code_task("in_range", ["n"], f"Return True if n lies between {lo} and {hi} inclusive, otherwise False.",
                      [f"return {lo} <= n <= {hi}"], lambda n: lo <= n <= hi,
                      [(lo,), (hi + 1,), (r.randint(lo, hi),), (lo - 1,), (hi,), (r.randint(-10, 80),)])


def _loop(r: random.Random) -> dict:
    t = r.randrange(8)
    if t == 0:
        return _code_task("sum_to", ["n"], "Return the sum of the integers from 1 to n (0 when n is 0).",
                          ["total = 0", "for i in range(1, n + 1):", "    total += i", "return total"],
                          lambda n: n * (n + 1) // 2, _ints(r, 0, 60, 6))
    if t == 1:
        return _code_task("factorial", ["n"], "Return n factorial (the product 1 * 2 * ... * n, and 1 when n is 0).",
                          ["result = 1", "for i in range(2, n + 1):", "    result *= i", "return result"],
                          math.factorial, _ints(r, 0, 11, 6))
    if t == 2:
        return _code_task("sum_squares", ["n"], "Return the sum of the squares of the integers from 1 to n.",
                          ["total = 0", "for i in range(1, n + 1):", "    total += i * i", "return total"],
                          lambda n: sum(i * i for i in range(1, n + 1)), _ints(r, 0, 30, 6))
    if t == 3:
        c = r.randint(2, 9)
        return _code_task("count_multiples", ["n"], f"Return how many integers from 1 to n are divisible by {c}.",
                          ["count = 0", "for i in range(1, n + 1):", f"    if i % {c} == 0:", "        count += 1", "return count"],
                          lambda n: n // c, _ints(r, 0, 100, 6))
    if t == 4:
        return _code_task("digit_sum", ["n"], "Return the sum of the decimal digits of the non-negative integer n.",
                          ["total = 0", "while n > 0:", "    total += n % 10", "    n //= 10", "return total"],
                          lambda n: sum(int(d) for d in str(n)), _ints(r, 0, 100000, 6))
    if t == 5:
        return _str_pipeline(r, [], 0, ["vowels"])
    if t == 6:
        return _list_pipeline(r, [], 0, ["product"])
    return _str_pipeline(r, ["caesar"], 1, ["str"])


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(math.isqrt(n)) + 1))


def _fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _collatz(n: int) -> int:
    s = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        s += 1
    return s


def _rle(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)


def _look_say(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{j - i}{s[i]}")
        i = j
    return "".join(out)


def _runs_str(r: random.Random, alphabet: str) -> str:
    return "".join(r.choice(alphabet) * r.randint(1, 3) for _ in range(r.randint(2, 5)))


def _algo6(r: random.Random) -> dict:
    t = r.randrange(7)
    if t == 0:
        return _code_task("fib", ["n"], "Return the n-th Fibonacci number, where fib(0) == 0 and fib(1) == 1.",
                          ["a, b = 0, 1", "for _ in range(n):", "    a, b = b, a + b", "return a"], _fib, _ints(r, 0, 26, 6))
    if t == 1:
        return _code_task("gcd", ["a", "b"], "Return the greatest common divisor of the positive integers a and b.",
                          ["while b:", "    a, b = b, a % b", "return a"], math.gcd,
                          [(r.randint(1, 30) * k, r.randint(1, 30) * k) for k in (r.randint(1, 9) for _ in range(6))])
    if t == 2:
        return _code_task("is_prime", ["n"], "Return True if n is a prime number, otherwise False.",
                          ["if n < 2:", "    return False", "i = 2", "while i * i <= n:", "    if n % i == 0:", "        return False",
                           "    i += 1", "return True"], _is_prime, [(r.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]),), (1,)] + _ints(r, 0, 120, 4))
    if t == 3:
        return _code_task("count_ones", ["n"], "Return how many 1 bits the binary representation of the non-negative integer n has.",
                          ["return bin(n).count('1')"], lambda n: bin(n).count("1"), _ints(r, 0, 1024, 6))
    if t == 4:
        words = [r.choice(["level", "Racecar", "noon", "Madam", "stats"])] + r.sample(WORDS, 4) + [r.choice(["Refer", "civic", "kayak"])]
        return _code_task("is_palindrome", ["s"], "Return True if s reads the same forwards and backwards when case is ignored, otherwise False.",
                          ["t = s.lower()", "return t == t[::-1]"], lambda s: s.lower() == s.lower()[::-1], [(w,) for w in words])
    if t == 5:
        return _code_task("dedupe", ["xs"], "Return a new list with the duplicates of xs removed, keeping the first occurrence of each value.",
                          ["return list(dict.fromkeys(xs))"], lambda xs: list(dict.fromkeys(xs)),
                          [([r.randint(0, 6) for _ in range(r.randint(3, 9))],) for _ in range(5)] + [([],)])
    return _code_task("reverse_words", ["s"], "Return the words of the sentence s in reverse order, separated by single spaces.",
                      ["return ' '.join(s.split()[::-1])"], lambda s: " ".join(s.split()[::-1]),
                      [(_phrase(r, r.randint(2, 5)),) for _ in range(6)])


def _algo7(r: random.Random) -> dict:
    t = r.randrange(6)
    if t == 0:
        return _code_task("count_primes_below", ["n"], "Return how many prime numbers are smaller than n.",
                          ["count = 0", "for k in range(2, n):", "    if all(k % d for d in range(2, int(k ** 0.5) + 1)):", "        count += 1",
                           "return count"], lambda n: sum(1 for k in range(2, n) if _is_prime(k)), _ints(r, 0, 200, 6))
    if t == 1:
        return _code_task("collatz_steps", ["n"], "Return how many steps the Collatz process needs to reach 1 from the positive integer n: "
                          "an even number is halved, an odd number becomes 3 * n + 1.",
                          ["steps = 0", "while n != 1:", "    n = n // 2 if n % 2 == 0 else 3 * n + 1", "    steps += 1", "return steps"],
                          _collatz, _ints(r, 1, 60, 6))
    if t == 2:
        return _code_task("lcm", ["a", "b"], "Return the least common multiple of the positive integers a and b.",
                          ["x, y = a, b", "while y:", "    x, y = y, x % y", "return a * b // x"], lambda a, b: a * b // math.gcd(a, b),
                          [(r.randint(1, 25), r.randint(1, 25)) for _ in range(6)])
    if t == 3:
        return _code_task("digital_root", ["n"], "Repeatedly replace the non-negative integer n by the sum of its digits until it has a "
                          "single digit, and return that digit.",
                          ["while n >= 10:", "    n = sum(int(d) for d in str(n))", "return n"],
                          lambda n: 0 if n == 0 else 1 + (n - 1) % 9, _ints(r, 0, 100000, 6))
    if t == 4:
        return _code_task("index_of_max", ["xs"], "Return the index of the first occurrence of the largest element of xs, or -1 if xs is empty.",
                          ["if not xs:", "    return -1", "return xs.index(max(xs))"],
                          lambda xs: xs.index(max(xs)) if xs else -1,
                          [([r.randint(-5, 20) for _ in range(r.randint(3, 8))],) for _ in range(5)] + [([],)])
    return _code_task("count_distinct", ["xs"], "Return how many distinct values the list xs contains.",
                      ["return len(set(xs))"], lambda xs: len(set(xs)),
                      [([r.randint(0, 9) for _ in range(r.randint(2, 9))],) for _ in range(5)] + [([],)])


def _algo8(r: random.Random) -> dict:
    t = r.randrange(7)
    if t == 0:
        return _code_task("run_length", ["s"], "Run-length encode s: write each maximal run of equal characters as the character followed by "
                          "the length of the run, for example 'aaab' becomes 'a3b1'.",
                          ["out = []", "i = 0", "while i < len(s):", "    j = i", "    while j < len(s) and s[j] == s[i]:", "        j += 1",
                           "    out.append(s[i] + str(j - i))", "    i = j", "return ''.join(out)"], _rle,
                          [(_runs_str(r, "abcd"),) for _ in range(5)] + [("",)])
    if t == 1:
        mk = lambda: sorted(r.randint(0, 30) for _ in range(r.randint(0, 5)))  # noqa: E731
        return _code_task("merge_sorted", ["a", "b"], "Merge the two ascending lists a and b into one ascending list without using sort.",
                          ["out = []", "i = j = 0", "while i < len(a) and j < len(b):", "    if a[i] <= b[j]:", "        out.append(a[i])",
                           "        i += 1", "    else:", "        out.append(b[j])", "        j += 1", "return out + a[i:] + b[j:]"],
                          lambda a, b: sorted(a + b), [(mk(), mk()) for _ in range(6)])
    if t == 2:
        return _code_task("rotate_right", ["xs", "k"], "Rotate the list xs to the right by k positions and return the new list "
                          "(an empty list stays empty).",
                          ["if not xs:", "    return []", "k = k % len(xs)", "return xs[-k:] + xs[:-k]"],
                          lambda xs, k: [xs[(i - k) % len(xs)] for i in range(len(xs))] if xs else [],
                          [([r.randint(0, 9) for _ in range(r.randint(2, 6))], r.randint(0, 8)) for _ in range(5)] + [([], 3)])
    if t == 3:
        return _code_task("second_largest", ["xs"], "Return the second largest distinct value in xs, or -1 if there is none.",
                          ["vals = sorted(set(xs), reverse=True)", "return vals[1] if len(vals) > 1 else -1"],
                          lambda xs: sorted(set(xs))[-2] if len(set(xs)) > 1 else -1,
                          [([r.randint(0, 15) for _ in range(r.randint(2, 7))],) for _ in range(4)] + [([4, 4],), ([],)])
    if t == 4:
        return _code_task("most_frequent", ["xs"], "Return the value that occurs most often in the non-empty list xs; on a tie return the "
                          "smallest such value.",
                          ["counts = {}", "for v in xs:", "    counts[v] = counts.get(v, 0) + 1", "best = max(counts.values())",
                           "return min(v for v, c in counts.items() if c == best)"],
                          lambda xs: min(v for v in set(xs) if xs.count(v) == max(xs.count(u) for u in xs)),
                          [([r.randint(0, 5) for _ in range(r.randint(1, 9))],) for _ in range(6)])
    if t == 5:
        def bs_input():
            xs = sorted(r.sample(range(0, 60), r.randint(0, 8)))
            return (xs, r.choice(xs) if xs and r.random() < 0.6 else r.randint(0, 60))
        return _code_task("binary_search", ["xs", "target"], "xs is sorted in ascending order and has distinct values. Return the index of "
                          "target in xs, or -1 if it is absent.",
                          ["lo, hi = 0, len(xs) - 1", "while lo <= hi:", "    mid = (lo + hi) // 2", "    if xs[mid] == target:",
                           "        return mid", "    if xs[mid] < target:", "        lo = mid + 1", "    else:", "        hi = mid - 1",
                           "return -1"], lambda xs, t: xs.index(t) if t in xs else -1, [bs_input() for _ in range(6)])
    w = r.sample(WORDS, 3)
    return _code_task("is_anagram", ["a", "b"], "Return True if the strings a and b contain exactly the same characters with the same "
                      "counts (anagrams), otherwise False.", ["return sorted(a) == sorted(b)"], lambda a, b: sorted(a) == sorted(b),
                      [(w[0], "".join(r.sample(w[0], len(w[0])))), (w[1], w[2]), ("listen", "silent"), (w[2], w[2][:-1]), ("", "")])


def _algo9(r: random.Random) -> dict:
    t = r.randrange(7)
    if t == 0:
        def lir(xs):
            best = cur = 0
            for i, v in enumerate(xs):
                cur = cur + 1 if i and v > xs[i - 1] else 1
                best = max(best, cur)
            return best
        return _code_task("longest_increasing_run", ["xs"], "Return the length of the longest run of consecutive elements of xs that is "
                          "strictly increasing (0 for an empty list).",
                          ["best = cur = 0", "prev = None", "for v in xs:", "    cur = cur + 1 if prev is not None and v > prev else 1",
                           "    best = max(best, cur)", "    prev = v", "return best"], lir,
                          [([r.randint(0, 9) for _ in range(r.randint(1, 10))],) for _ in range(5)] + [([],)])
    if t == 1:
        def bal(s):
            st = []
            for ch in s:
                if ch in "([{":
                    st.append(ch)
                elif ch in ")]}":
                    if not st or st.pop() != {")": "(", "]": "[", "}": "{"}[ch]:
                        return False
            return not st
        def gen():
            s = "".join(r.choice("()[]{}") for _ in range(r.randint(0, 8)))
            return s if r.random() < 0.5 else r.choice(["()", "([])", "{[()]}", "(()", "([)]", "[]{}()"])
        return _code_task("balanced", ["s"], "Return True if every bracket in s ('()', '[]' and '{}') is closed in the right order, "
                          "otherwise False.",
                          ["pairs = {')': '(', ']': '[', '}': '{'}", "stack = []", "for ch in s:", "    if ch in '([{':",
                           "        stack.append(ch)", "    elif ch in pairs:", "        if not stack or stack.pop() != pairs[ch]:",
                           "            return False", "return not stack"], bal, [("{[()]}",), ("([)]",)] + [(gen(),) for _ in range(5)])
    if t == 2:
        def mat():
            rows, cols = r.randint(1, 3), r.randint(1, 4)
            return [[r.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
        return _code_task("transpose", ["m"], "Return the transpose of the rectangular matrix m, given as a non-empty list of rows.",
                          ["return [list(row) for row in zip(*m)]"], lambda m: [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))],
                          [(mat(),) for _ in range(6)])
    if t == 3:
        def kad(xs):
            return max(sum(xs[i:j]) for i in range(len(xs)) for j in range(i + 1, len(xs) + 1))
        return _code_task("max_subarray", ["xs"], "Return the largest sum of a non-empty contiguous sublist of the non-empty list xs.",
                          ["best = cur = xs[0]", "for v in xs[1:]:", "    cur = max(v, cur + v)", "    best = max(best, cur)", "return best"],
                          kad, [([r.randint(-9, 9) for _ in range(r.randint(1, 8))],) for _ in range(6)])
    if t == 4:
        return _code_task("sum_square_primes", ["xs"], "Return the sum of the squares of the prime numbers in xs.",
                          ["total = 0", "for v in xs:", "    if v >= 2 and all(v % d for d in range(2, int(v ** 0.5) + 1)):",
                           "        total += v * v", "return total"], lambda xs: sum(v * v for v in xs if _is_prime(v)),
                          [([r.randint(0, 30) for _ in range(r.randint(1, 7))],) for _ in range(5)] + [([],)])
    if t == 5:
        return _code_task("count_pairs", ["xs", "target"], "Return how many pairs of positions i < j satisfy xs[i] + xs[j] == target.",
                          ["count = 0", "for i in range(len(xs)):", "    for j in range(i + 1, len(xs)):", "        if xs[i] + xs[j] == target:",
                           "            count += 1", "return count"],
                          lambda xs, t: sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[i] + xs[j] == t),
                          [([r.randint(0, 9) for _ in range(r.randint(0, 8))], r.randint(2, 14)) for _ in range(6)])
    return _code_task("look_and_say", ["s"], "Describe the digit string s by runs: each maximal run of equal digits becomes the length of the "
                      "run followed by the digit, for example '1211' becomes '111221'.",
                      ["out = []", "i = 0", "while i < len(s):", "    j = i", "    while j < len(s) and s[j] == s[i]:", "        j += 1",
                       "    out.append(str(j - i) + s[i])", "    i = j", "return ''.join(out)"], _look_say,
                      [("1211",)] + [(_runs_str(r, "123"),) for _ in range(5)])


CODE_LEVELS: dict[int, list[Callable[[random.Random], dict]]] = {
    0: [lambda r: _int_pipeline(r, INT_BASIC, 1), lambda r: _str_pipeline(r, STR_L0, 1, ["str"]),
        lambda r: _str_pipeline(r, [], 0, ["len"])],
    1: [lambda r: _int_pipeline(r, INT_WIDE, 1), _two_arg, lambda r: _str_pipeline(r, STR_L1, 1, ["str"]),
        lambda r: _str_pipeline(r, [], 0, ["count"])],
    2: [_cond],
    3: [lambda r: _list_pipeline(r, LIST_SIMPLE, r.randint(0, 1), ["sum", "len", "max", "min", "list", "count_gt"])],
    4: [_loop],
    5: [lambda r: _int_pipeline(r, INT_ALL, 2), lambda r: _list_pipeline(r, LIST_ALL, 2),
        lambda r: _str_pipeline(r, STR_ALL, 2, list(STR_FINAL))],
    6: [_algo6],
    7: [lambda r: _int_pipeline(r, INT_ALL, 3), lambda r: _list_pipeline(r, LIST_ALL, 3),
        lambda r: _str_pipeline(r, STR_ALL, 3, list(STR_FINAL)), _algo7],
    8: [_algo8, lambda r: _list_pipeline(r, LIST_ALL, 3)],
    9: [_algo9, lambda r: _int_pipeline(r, INT_ALL, 4), lambda r: _list_pipeline(r, LIST_ALL, 4),
        lambda r: _str_pipeline(r, STR_ALL, 4, list(STR_FINAL))],
}


# --------------------------------------------------------------------------- text family

TEXT_PROMPT = "Question: {q}\nAnswer:"


def _num(r: random.Random, digits: int) -> int:
    return r.randint(10 ** (digits - 1) if digits > 1 else 0, 10 ** digits - 1)


def _t_add(r, da, db):
    a, b = _num(r, da), _num(r, db)
    return f"What is {a} + {b}?", str(a + b)


def _t_sub(r, da, db, allow_negative=False):
    a, b = _num(r, da), _num(r, db)
    if not allow_negative and b > a:
        a, b = b, a
    return f"What is {a} - {b}?", str(a - b)


def _t_mul(r, da, db):
    a, b = _num(r, da), _num(r, db)
    return f"What is {a} * {b}?", str(a * b)


def _t_count_letters(r, pool):
    w = r.choice(pool)
    return f'How many letters are in the word "{w}"?', str(len(w))


def _t_reverse_word(r, pool):
    w = r.choice(pool)
    return f'Write the word "{w}" backwards.', w[::-1]


def _t_compare(r, digits):
    a, b = r.sample(range(10 ** (digits - 1), 10 ** digits), 2)
    if r.random() < 0.5:
        return f"Which number is larger, {a} or {b}?", str(max(a, b))
    return f"Is {a} greater than {b}? Answer yes or no.", "yes" if a > b else "no"


UNITS = [("kilometers", "meters", 1000), ("meters", "centimeters", 100), ("hours", "minutes", 60), ("minutes", "seconds", 60),
         ("kilograms", "grams", 1000), ("days", "hours", 24), ("weeks", "days", 7), ("liters", "milliliters", 1000),
         ("years", "months", 12), ("dozens", "items", 12)]


def _t_unit(r, hi):
    big, small, f = r.choice(UNITS)
    k = r.randint(2, hi)
    return f"How many {small} are in {k} {big}?", str(k * f)


def _t_unit2(r):
    big, small, f = r.choice(UNITS[:4] + UNITS[5:7])
    a, b = r.randint(1, 9), r.randint(1, f - 1)
    return f"How many {small} are in {a} {big} and {b} {small}?", str(a * f + b)


def _t_unit_sum(r):
    a, b = r.randint(1, 5), r.randint(0, 59)
    c, d = r.randint(1, 5), r.randint(0, 59)
    return f"How many minutes are {a} hours {b} minutes plus {c} hours {d} minutes?", str(a * 60 + b + c * 60 + d)


def _t_sort_nums(r, k, desc=False):
    xs = r.sample(range(1, 100), k)
    order = "from largest to smallest" if desc else "from smallest to largest"
    return f"Sort these numbers {order}: {', '.join(map(str, xs))}.", ", ".join(map(str, sorted(xs, reverse=desc)))


def _t_sort_words(r, k):
    ws = r.sample(WORDS, k)
    return f"Sort these words alphabetically: {', '.join(ws)}.", ", ".join(sorted(ws))


def _t_reverse_words(r, k):
    ws = r.sample(WORDS, k)
    return f'Reverse the order of the words in "{" ".join(ws)}".', " ".join(ws[::-1])


def _t_reverse_each(r, k):
    ws = r.sample(WORDS, k)
    return f'Reverse the letters of each word in "{" ".join(ws)}".', " ".join(w[::-1] for w in ws)


def _t_count_char(r, k):
    p = _phrase(r, k)
    c = r.choice(sorted(set(p.replace(" ", ""))))
    return f'How many times does the letter "{c}" appear in "{p}"?', str(p.count(c))


def _t_vowels(r, k):
    p = _phrase(r, k)
    return f'How many vowels (a, e, i, o, u) are in "{p}"?', str(sum(ch in "aeiou" for ch in p))


def _t_even_count(r, k):
    xs = [r.randint(1, 60) for _ in range(k)]
    return f"How many of these numbers are even: {', '.join(map(str, xs))}?", str(sum(x % 2 == 0 for x in xs))


def _t_max_list(r, k):
    xs = r.sample(range(1, 200), k)
    return f"What is the largest number in this list: {', '.join(map(str, xs))}?", str(max(xs))


ORDER_REL = [("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest"),
             ("heavier", "heaviest", "lightest")]


def _t_order(r, k, ask):
    names = r.sample(NAMES, k)  # names[0] is the most, names[-1] the least
    rel, most, least = r.choice(ORDER_REL)
    facts = [f"{names[i]} is {rel} than {names[i + 1]}." for i in range(k - 1)]
    r.shuffle(facts)
    if ask == "extreme":
        if r.random() < 0.5:
            q, a = f"Who is the {most}?", names[0]
        else:
            q, a = f"Who is the {least}?", names[-1]
    elif ask == "second":
        q, a = f"Who is the second {most}?", names[1]
    else:  # "third"
        q, a = f"Who is the third {most}?", names[2]
    return " ".join(facts) + " " + q, a


def _t_bool(r, clauses):
    parts, vals = [], []
    for _ in range(clauses):
        a, b = r.randint(1, 50), r.randint(1, 50)
        op = r.choice([">", "<"])
        parts.append(f"({a} {op} {b})")
        vals.append(a > b if op == ">" else a < b)
    ops = [r.choice(["and", "or"]) for _ in range(clauses - 1)]
    expr = parts[0]
    val = vals[0]
    for o, p, v in zip(ops, parts[1:], vals[1:]):  # evaluated left to right, as written with explicit grouping
        expr = f"({expr} {o} {p})" if clauses > 2 else f"{expr} {o} {p}"
        val = (val and v) if o == "and" else (val or v)
    return f"Is it true that {expr}? Answer yes or no.", "yes" if val else "no"


def _t_expr_prec(r):
    a, b, c = r.randint(1, 30), r.randint(2, 12), r.randint(2, 12)
    return f"What is {a} + {b} * {c}?", str(a + b * c)


def _t_expr_paren(r):
    a, b, c, d = r.randint(1, 30), r.randint(1, 30), r.randint(2, 9), r.randint(1, 40)
    return f"What is ({a} + {b}) * {c} - {d}?", str((a + b) * c - d)


def _t_expr4(r):
    a, b, c, d, e = (r.randint(2, 15) for _ in range(5))
    return f"What is {a} * {b} - {c} * {d} + {e}?", str(a * b - c * d + e)


def _t_expr5(r):
    a, b, c, d, e, f = (r.randint(2, 20) for _ in range(6))
    return f"What is ({a} + {b}) * ({c} - {d}) + {e} * {f}?", str((a + b) * (c - d) + e * f)


def _t_remainder(r):
    b = r.randint(3, 12)
    a = r.randint(b + 1, 150)
    if r.random() < 0.5:
        return f"What is the remainder when {a} is divided by {b}?", str(a % b)
    return f"What is {a} divided by {b}, rounded down to a whole number?", str(a // b)


def _t_weekday(r, hi):
    i, k = r.randrange(7), r.randint(1, hi)
    return f"If today is {DAYS[i]}, what day of the week will it be in {k} days?", DAYS[(i + k) % 7]


TEXT_LEVELS: dict[int, list[Callable[[random.Random], tuple[str, str]]]] = {
    0: [lambda r: _t_add(r, 1, 1), lambda r: _t_sub(r, 1, 1), lambda r: _t_count_letters(r, SHORT_WORDS),
        lambda r: _t_reverse_word(r, SHORT_WORDS)],
    1: [lambda r: _t_add(r, 2, 2), lambda r: _t_sub(r, 2, 2), lambda r: _t_reverse_word(r, WORDS), lambda r: _t_compare(r, 2),
        lambda r: _t_count_letters(r, WORDS)],
    2: [lambda r: _t_mul(r, 2, 1), lambda r: _t_unit(r, 9), lambda r: _t_count_char(r, 1), lambda r: _t_sort_nums(r, 3)],
    3: [lambda r: _t_add(r, 3, 3), lambda r: _t_sub(r, 3, 2), lambda r: _t_sort_nums(r, 4, r.random() < 0.5),
        lambda r: _t_sort_words(r, 3), lambda r: _t_max_list(r, 5)],
    4: [lambda r: _t_mul(r, 2, 2), lambda r: _t_unit2(r), lambda r: _t_reverse_words(r, 3), lambda r: _t_even_count(r, 5),
        lambda r: _t_order(r, 3, "extreme")],
    5: [lambda r: _t_mul(r, 2, 2), _t_remainder, lambda r: _t_sort_words(r, 5), lambda r: _t_bool(r, 2),
        lambda r: _t_order(r, 3, "second")],
    6: [_t_expr_prec, lambda r: _t_weekday(r, 30), lambda r: _t_vowels(r, 2), lambda r: _t_order(r, 4, r.choice(["extreme", "second"])),
        lambda r: _t_sort_nums(r, 6, True)],
    7: [_t_expr_paren, lambda r: _t_add(r, 4, 4), lambda r: _t_sub(r, 4, 3, True), lambda r: _t_reverse_each(r, 3), _t_unit_sum],
    8: [lambda r: _t_mul(r, 3, 2), _t_expr4, lambda r: _t_order(r, 5, "third"), lambda r: _t_weekday(r, 400),
        lambda r: _t_count_char(r, 3)],
    9: [lambda r: _t_mul(r, 3, 3), _t_expr5, lambda r: _t_sort_nums(r, 8), lambda r: _t_order(r, 5, "second"),
        lambda r: _t_bool(r, 3)],
}


# --------------------------------------------------------------------------- task construction and sampling

def make_task(family: str, level: int, seed: int) -> Task:
    """Deterministic task for (family, level, seed); `mixed` alternates by seed parity."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if level not in LEVELS:
        raise ValueError(f"level must be in 0..9, got {level}")
    space_of(seed)
    fam = family if family != "mixed" else ("code" if seed % 2 == 0 else "text")
    r = random.Random(f"{fam}|{level}|{seed}")
    if fam == "code":
        gens = CODE_LEVELS[level]
        g = gens[r.randrange(len(gens))]
        d = g(r)
        return Task(task_id=f"code-L{level}-s{seed}", family="code", level=level, seed=seed, template=d["entry_point"],
                    prompt=d["prompt"], reference=d["reference"], tests=d["tests"], entry_point=d["entry_point"])
    gens = TEXT_LEVELS[level]
    gi = r.randrange(len(gens))
    q, a = gens[gi](r)
    return Task(task_id=f"text-L{level}-s{seed}", family="text", level=level, seed=seed, template=f"text{level}.{gi}",
                prompt=TEXT_PROMPT.format(q=q), reference=" " + a, answer=a)


def heldout_tasks(family: str, level: int, n: int, decon: Decontaminator | None = None) -> tuple[list[Task], dict]:
    """The fixed held-out set: the first n clean, distinct tasks of the held-out seed space."""
    decon = decon or default_decontaminator()
    out, keys = [], set()
    stats = {"generated": 0, "dropped_contaminated": 0, "dropped_duplicate": 0}
    for i in range(max(64, 50 * n)):
        t = make_task(family, level, HELDOUT_SEEDS.start + i)
        stats["generated"] += 1
        if t.key in keys:
            stats["dropped_duplicate"] += 1
            continue
        if decon.contaminated(t.text_for_decontamination()):
            stats["dropped_contaminated"] += 1
            continue
        keys.add(t.key)
        out.append(t)
        if len(out) == n:
            break
    return out, stats


def heldout_keys(family: str, n: int, levels=LEVELS, decon: Decontaminator | None = None) -> set[str]:
    keys: set[str] = set()
    for lv in levels:
        keys |= {t.key for t in heldout_tasks(family, lv, n, decon)[0]}
    return keys


def sample_tasks(family: str, level: int, n: int, round_key: str, *, exclude: set[str] | frozenset = frozenset(),
                 decon: Decontaminator | None = None) -> tuple[list[Task], dict]:
    """n clean, distinct training tasks drawn from the train seed space.

    Seeds come from random.Random(f"train|{family}|{level}|{round_key}"), so a
    round is reproducible; tasks whose key is in `exclude` (the held-out keys)
    or that overlap the eval sets are dropped.
    """
    decon = decon or default_decontaminator()
    r = random.Random(f"train|{family}|{level}|{round_key}")
    out, keys, seeds = [], set(), set()
    stats = {"generated": 0, "dropped_contaminated": 0, "dropped_heldout_collision": 0, "dropped_duplicate": 0}
    for _ in range(max(200, 50 * n)):
        s = r.randrange(TRAIN_SEEDS.start, TRAIN_SEEDS.stop)
        if s in seeds:
            continue
        seeds.add(s)
        t = make_task(family, level, s)
        stats["generated"] += 1
        if t.key in exclude:
            stats["dropped_heldout_collision"] += 1
            continue
        if t.key in keys:
            stats["dropped_duplicate"] += 1
            continue
        if decon.contaminated(t.text_for_decontamination()):
            stats["dropped_contaminated"] += 1
            continue
        keys.add(t.key)
        out.append(t)
        if len(out) == n:
            break
    return out, stats
