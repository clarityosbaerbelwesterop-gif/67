"""Measure an instruction-tuned chat model (Qwen3.8-27B as published, before any fine-tuning) on HumanEval, MBPP and
their -hard variants, decoded with vLLM on the pod's GPUs.

The model answers in chat form. Its code is the last ```python block of the answer after any thinking; it is scored
with the sandbox of scripts/humaneval.py (same hidden tests, same CPU limits). The -hard suites share their prompts
with the base suites, so they are scored from the same generations.

Modes, with the sampling settings the model card recommends:
  instruct  thinking off: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5
  think     thinking on:  temperature 1.0, top_p 0.95, top_k 20
One sample per task with a fixed seed, so pass@1 is one draw, not an average.

Generations of benchmark prompts are measurement only and never training data.

  python3 scripts/qwen_eval.py --model /workspace/hf/Qwen3.8-27B --modes instruct,think --out OUT [--tp 2]
Each (mode, suite) prints one JSON line; OUT/gens-<mode>.jsonl keeps every answer.
"""
import argparse
import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import humaneval as he  # noqa: E402

SUITES = ("humaneval", "mbpp", "humaneval-hard", "mbpp-hard")
SAMPLING = {
    "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    "think": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
}
FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)


def chat_prompt(suite: str, p: dict) -> str:
    if suite.startswith("humaneval"):
        return ("Complete the following Python function. Answer with the complete function, including its signature "
                "and any imports it needs, in a single ```python code block.\n\n```python\n" + p["prompt"] + "```")
    task = p["prompt"].strip().strip('"').strip()
    return ("Solve the following task in Python. Your code must pass the test shown. Answer with the complete "
            "solution in a single ```python code block.\n\n" + task)


def extract_code(text: str) -> str:
    """The last python (or untagged) fenced block after the thinking; the answer itself if there is none."""
    answer = text.split("</think>")[-1]
    blocks = FENCE.findall(answer)
    py = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3", "")]
    if py:
        return py[-1]
    if blocks:
        return blocks[-1][1]
    return answer


def program(p: dict, code: str) -> str:
    """The program the hidden tests run. MBPP: the answer alone (its prompt is the task as a docstring). HumanEval:
    the prompt first, so its imports and helper functions exist; an answer that redefines the function at the top
    level replaces the prompt's stub, an answer that only continues the body is appended to it."""
    if p["prompt"].startswith('"""'):
        return code
    ep = p.get("entry_point") or ""
    if ep and re.search(rf"^def {re.escape(ep)}\s*\(", code, re.M):
        return p["prompt"] + "\n" + code
    return p["prompt"] + code


def score(suite: str, problems: list, codes: list, jobs: int) -> list:
    cpu = 40 if suite.endswith("-hard") else he.CPU_S
    progs = [(program(p, c) + p["tail"], cpu) for p, c in zip(problems, codes)]
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        return list(pool.map(he.hidden_pass, progs, chunksize=4))


def engine(model: str, tp: int, max_len: int, seed: int):
    """vLLM's offline engine; returns chat(convs, mode, max_tokens) -> [{text, finish, tokens}]."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_len, gpu_memory_utilization=0.92, seed=seed,
              limit_mm_per_prompt={"image": 0, "video": 0}, trust_remote_code=False)

    def chat(convs: list, mode: str, max_tokens: int, effort: str | None) -> list:
        sp = SamplingParams(max_tokens=max_tokens, seed=seed, **SAMPLING[mode])
        kw = {"enable_thinking": mode == "think"}
        if effort and mode == "think":
            kw["reasoning_effort"] = effort
        outs = llm.chat(convs, sp, chat_template_kwargs=kw, use_tqdm=False)
        return [{"text": o.outputs[0].text, "finish": o.outputs[0].finish_reason, "tokens": len(o.outputs[0].token_ids)}
                for o in outs]

    return chat


def run(chat, modes: list, out: Path, max_tokens: dict, jobs: int, deadline: float, effort: str | None,
        limit: int | None, model: str) -> list:
    out.mkdir(parents=True, exist_ok=True)
    probs = {s: he.problems(s)[:limit] for s in SUITES}
    base = {"humaneval": probs["humaneval"], "mbpp": probs["mbpp"]}
    results = []
    for mode in modes:
        if time.time() > deadline:
            results.append({"mode": mode, "skipped": "deadline"})
            print(json.dumps(results[-1]), flush=True)
            continue
        gens = {}
        t = time.time()
        for b, ps in base.items():
            texts = chat([[{"role": "user", "content": chat_prompt(b, p)}] for p in ps], mode, max_tokens[mode], effort)
            gens.update({p["prompt"]: g for p, g in zip(ps, texts)})
        secs = time.time() - t
        with open(out / f"gens-{mode}.jsonl", "w") as f:
            for prompt, g in gens.items():
                f.write(json.dumps({"prompt": prompt, **g}) + "\n")
        n_tok = sum(g["tokens"] for g in gens.values())
        for s in SUITES:
            ps = probs[s]
            gs = [gens[p["prompt"]] for p in ps]
            ok = score(s, ps, [extract_code(g["text"]) for g in gs], jobs)
            n = max(1, len(ps))
            results.append({
                "model": model, "suite": s, "mode": mode, "problems": len(ps), "passed": sum(ok),
                "pass@1": round(sum(ok) / n, 4), "truncated": sum(g["finish"] == "length" for g in gs),
                "mean_output_tokens": round(sum(g["tokens"] for g in gs) / n, 1),
                "sampling": SAMPLING[mode], "max_tokens": max_tokens[mode], "effort": effort if mode == "think" else None,
                "gen_seconds_mode": round(secs, 1), "tokens_per_s_mode": round(n_tok / max(secs, 1e-3), 1),
                "decoding": "one sample per task (fixed seed), code = last ```python block after thinking",
            })
            print(json.dumps(results[-1]), flush=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="local snapshot directory or hub id")
    ap.add_argument("--modes", default="instruct,think")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (GPUs)")
    ap.add_argument("--max-tokens-instruct", type=int, default=4096)
    ap.add_argument("--max-tokens-think", type=int, default=32768)
    ap.add_argument("--effort", default=None, help="reasoning_effort for think (model default if unset)")
    ap.add_argument("--deadline", type=float, default=float("inf"), help="epoch; no new mode starts after it")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args(argv)
    modes = [m for m in a.modes.split(",") if m]
    bad = [m for m in modes if m not in SAMPLING]
    if bad:
        raise SystemExit(f"unknown mode(s): {bad}")
    max_tokens = {"instruct": a.max_tokens_instruct, "think": a.max_tokens_think}
    chat = engine(a.model, a.tp, max(max_tokens[m] for m in modes) + 4096, a.seed)
    run(chat, modes, a.out, max_tokens, a.jobs, a.deadline, a.effort, a.limit, a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
