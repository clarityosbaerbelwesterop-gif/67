"""Thinking in scripts/humaneval.py: candidates are picked by the tests the prompt shows and by agreement with
the model's own tests, never by the hidden tests; a candidate cannot fake a pass; repair prompts name the failing
check."""
import importlib.util
import json
import secrets
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("humaneval", ROOT / "scripts" / "humaneval.py")
he = importlib.util.module_from_spec(spec)
sys.modules["humaneval"] = he  # the process pool pickles functions by module name
spec.loader.exec_module(he)

needs_evals = pytest.mark.skipif(not (ROOT / "data/stores/evals/humaneval.jsonl").exists(), reason="needs data/stores/evals")
PROMPT = 'def add(a: int, b: int) -> int:\n    """Sum.\n    >>> add(2, 3)\n    5\n    >>> add(-1, 1)\n    0\n    """\n'


def evaluate(body, vis, probes=()):
    return he.evaluate((body, vis, list(probes), secrets.token_hex(8)))


def score(body, tests):
    e = evaluate(body, tests)
    return e["passed"], e["loaded"]


def test_visible_tests_parse_doctest_arrows_and_asserts():
    assert he.visible_tests(PROMPT) == [("add(2, 3)", "5"), ("add(-1, 1)", "0")]
    assert he.visible_tests('def f(x):\n    """\n    f(2) ==> 4\n    f(3) => 9.\n    """\n', "f") == [("f(2)", "4"), ("f(3)", "9")]
    assert he.visible_tests('"""\nSquare.\nassert sq(3) == 9\n"""\n') == [("assert sq(3) == 9", None)]
    assert he.visible_tests('def g():\n    """\n    >>> g() == [1]\n    """\n') == [("assert g() == [1]", None)]
    assert he.probe_of("assert sq(3) == 9", None) == ("sq(3)", "9") and he.probe_of("assert sq(3)", None) is None
    assert he.entry_of({"prompt": ""}, [("assert sq(3) == 9", None)]) == "sq"
    assert he._closed(("sq(3)", "9"), "sq") and he._closed(("sq(len('ab'))", "[4]"), "sq")
    assert not he._closed(("sq(x)", "9"), "sq") and not he._closed(("sq(3)", "y"), "sq")


def test_consensus_prefers_the_majority_behaviour_and_self_tests():
    probes = [("add(2, 3)", "5"), ("add(10, 1)", "11")]   # visible call, then one self-test
    right = [PROMPT + "    return a + b\n", PROMPT + "    return b + a\n"]
    odd = PROMPT + "    return 5 if a == 2 else a * b\n"   # passes the visible check, fails the self-test
    raising = PROMPT + "    raise ValueError\n"
    vis = he.visible_tests(PROMPT)[:1]
    ev = [evaluate(b, vis, probes) for b in (odd, raising, raising, raising) + tuple(right)]
    assert [e["passed"] for e in ev] == [1, 0, 0, 0, 1, 1] and ev[4]["sig"] == ("5", "11") and ev[1]["sig"][0] == "raised ValueError"
    assert he.choose(ev, n_self=1) == 4                           # agreeing pair with the self-test beats the first one
    assert he.choose(ev, n_self=1, consensus=False) == 0          # plain best-of-N keeps the earliest
    assert he.choose([ev[1], ev[2], ev[3]], n_self=1) == 0        # all raising: no group weight, greedy kept


def test_repair_prompt_names_the_failing_check():
    vis = he.visible_tests(PROMPT)
    wrong = PROMPT + "    return a - b\n"
    e = evaluate(wrong, vis, [he.probe_of(s, w) for s, w in vis])
    rp = he.repair_prompt({"prompt": PROMPT}, wrong, e, vis, [0, 1])
    assert rp.startswith(wrong) and rp.endswith("# Fixed version:\n" + PROMPT)
    assert "# This check fails: add(2, 3) == 5\n# It gives: -1\n" in rp
    right = PROMPT + "    return a + b\n"
    assert he.repair_prompt({"prompt": PROMPT}, right, evaluate(right, vis), vis, [None, None]) is None


def test_selection_uses_visible_tests_and_cannot_be_faked():
    tests = he.visible_tests(PROMPT)
    wrong = PROMPT + "    return a - b\n"
    right = PROMPT + "    return a + b\n"
    broken = PROMPT + "    return a +\n"
    fake = PROMPT + "    return 0\n\n\nprint('pass')\nprint('loaded')\n"
    scores = [score(b, tests) for b in (wrong, broken, fake, right)]
    assert scores == [(0, 1), (0, 0), (1, 1), (2, 1)], scores  # the fake prints count for nothing; add(-1, 1) == 0 holds for 0
    assert max(range(4), key=lambda j: (scores[j], -j)) == 3


def test_looping_candidate_is_cut_off():
    loop = PROMPT + "    while True:\n        pass\n"
    assert score(loop, he.visible_tests(PROMPT)) == (0, 1)


@needs_evals
def test_visible_tests_come_from_the_prompt_only():
    squash = lambda t: "".join(t.split())
    for suite, least in (("humaneval", 125), ("mbpp", 500)):
        probs = he.problems(suite)
        tests = [he.visible_tests(p["prompt"], p.get("entry_point")) for p in probs]
        assert sum(bool(t) for t in tests) >= least
        for p, ts in zip(probs, tests):
            for src, want in ts:
                assert squash(src.removeprefix("assert ")) in squash(p["prompt"]) and squash(want or "") in squash(p["prompt"])
                assert squash(src) not in squash(p["tail"]) or squash(src) in squash(p["prompt"])


@needs_evals
def test_canonical_solutions_pass_their_visible_tests():
    rows = [json.loads(l) for l in (ROOT / "data/stores/evals/humaneval.jsonl").read_text().splitlines()]
    scores = [score(r["prompt"] + r["canonical_solution"], he.visible_tests(r["prompt"], r["entry_point"])) for r in rows]
    full = sum(s == (len(he.visible_tests(r["prompt"], r["entry_point"])), 1) for s, r in zip(scores, rows))
    assert full >= 159, full  # 5 HumanEval prompts carry wrong examples (median, is_simple_power, is_happy, sort_array, bf)
