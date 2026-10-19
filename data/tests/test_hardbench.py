"""The hard coding test must accept the reference, reject wrong programs that pass the official tests,
and keep its extra inputs inside the domain the official tests use."""
import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("hardbench", ROOT / "data" / "build_hardbench.py")
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)


def _task(suite: str, i: int) -> dict:
    return [t for t in hb.tasks() if t["suite"] == suite][i]


def _extra(row: dict) -> list:
    return json.loads(json.loads(row["tail"].split("_HARD = ", 1)[1].split("\n", 1)[0]))


def test_reference_passes_and_extra_tests_exist():
    for suite, i in (("humaneval", 5), ("mbpp", 1)):
        t = _task(suite, i)
        row = hb.build(t, 20)
        assert row["extra_tests"] > 0 and not row["fallback"]
        assert hb.run(t["ref"] + "\n" + row["tail"])[0]


def test_wrong_program_passing_official_tests_fails_hard_tests():
    # HumanEval/5 intersperse: a version that is wrong only for lists longer than 3 elements still
    # passes the 3 official asserts ([], [5, 6, 3, 2], [2, 2, 2]) if it special-cases them; here it
    # drops the delimiter after the 4th element, which the official tests never reach
    t = _task("humaneval", 5)
    row = hb.build(t, 40)
    wrong = t["prompt"] + (
        "    out = []\n"
        "    for i, n in enumerate(numbers):\n"
        "        out.append(n)\n"
        "        if i < len(numbers) - 1 and i < 3:\n"
        "            out.append(delimeter)\n"
        "    return out\n")
    assert hb.run(wrong + "\n\n" + t["tests"])[0], "the wrong program must pass the official tests"
    assert not hb.run(wrong + "\n" + row["tail"])[0], "the hard tests must catch it"


def test_mutated_inputs_stay_in_the_seen_domain():
    t = _task("humaneval", 5)
    seeds = [ast.literal_eval(s) for s in hb.seeds(t)]
    seen_ints = [x for a in seeds for v in a for x in (v if isinstance(v, list) else [v])]
    for args_repr, _ in _extra(hb.build(t, 40)):
        numbers, delim = ast.literal_eval(args_repr)
        assert all(min(seen_ints) <= x <= max(seen_ints) for x in numbers + [delim])
