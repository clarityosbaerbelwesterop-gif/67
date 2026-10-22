"""scripts/qwen_eval.py: code extraction, the program the hidden tests run, and a full run with a fake engine that
answers with the reference solutions."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import humaneval as he  # noqa: E402
import qwen_eval as qe  # noqa: E402


def rows(name):
    return [json.loads(l) for l in (ROOT / f"data/stores/evals/{name}.jsonl").read_text().splitlines() if l.strip()]


def test_extract_code_takes_last_python_block_after_thinking():
    text = "<think>try ```python\nbad()\n```</think>Here:\n```python\nx = 1\n```\nand\n```py\ndef f():\n    return 2\n```"
    assert qe.extract_code(text) == "def f():\n    return 2\n"
    assert qe.extract_code("no fence at all") == "no fence at all"
    assert qe.extract_code("```\nplain\n```") == "plain\n"
    assert qe.extract_code("```js\nlet a\n```") == "let a\n"


def test_program_humaneval_and_mbpp():
    he_rows, probs = rows("humaneval"), he.problems("humaneval")
    p, ref = probs[0], he_rows[0]["canonical_solution"]
    assert he.passes(qe.program(p, ref) + p["tail"])                       # body only
    assert he.passes(qe.program(p, p["prompt"] + ref) + p["tail"])         # whole function again
    assert not he.passes(qe.program(p, "def has_close_elements(a, b):\n    return None\n") + p["tail"])
    mb = he.problems("mbpp")[0]
    code = next(r["code"] for r in rows("mbpp") if r["text"] in mb["prompt"])
    assert qe.program(mb, code) == code and he.passes(code + mb["tail"])


def test_run_with_reference_answers(tmp_path):
    ref = {r["prompt"]: r["prompt"] + r["canonical_solution"] for r in rows("humaneval")}
    ref.update({r["text"]: r["code"] for r in rows("mbpp")})
    seen = []

    def chat(convs, mode, max_tokens, effort):
        out = []
        for c in convs:
            msg = c[0]["content"]
            code = next(v for k, v in ref.items() if k in msg)
            out.append({"text": f"<think>plan</think>\n```python\n{code}\n```", "finish": "stop", "tokens": 7})
        seen.append((mode, max_tokens, len(convs)))
        return out

    res = qe.run(chat, ["instruct", "think"], tmp_path, {"instruct": 64, "think": 128}, 4, float("inf"), None, 6, "fake")
    by = {(r["mode"], r["suite"]): r for r in res}
    assert len(by) == 8 and seen == [("instruct", 64, 6), ("instruct", 64, 6), ("think", 128, 6), ("think", 128, 6)]
    for mode in ("instruct", "think"):
        assert by[(mode, "humaneval")]["pass@1"] == 1.0 and by[(mode, "mbpp")]["pass@1"] == 1.0
        assert by[(mode, "humaneval")]["truncated"] == 0 and by[(mode, "humaneval")]["mean_output_tokens"] == 7
    assert (tmp_path / "gens-think.jsonl").read_text().count("\n") == 12


def test_deadline_skips_modes(tmp_path):
    res = qe.run(lambda *a: [], ["instruct"], tmp_path, {"instruct": 8, "think": 8}, 1, 0.0, None, 1, "fake")
    assert res == [{"mode": "instruct", "skipped": "deadline"}]
