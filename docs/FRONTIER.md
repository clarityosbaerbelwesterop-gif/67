# Standard: the hardest AI tests (owner decision, 2026-10-12)

From now on every model is measured against the hardest public AI tests. Every
release reports them; a test that cannot run here is listed with the reason, never
skipped silently or estimated.

## Scorecard

Frontier values are published by vendors or ARC Prize, as found on 2026-10-12
(status in brackets). Ours are measured by `scripts/humaneval.py`; "n/a" means the
test cannot run here, with the reason given.

| Test | Best frontier result | Quasnir 2 | Rouge (g3) | Darus (g2) | Why ours is what it is |
| --- | --- | --- | --- | --- | --- |
| ARC-AGI-2 (public eval, 120 tasks / 167 outputs, pass@2) | GPT-6.1 Sol 94.2 % (ARC Prize, verified); Opus 5.5 93.3 % (ARC Prize, reported) | **0 / 167** | **0 / 167** | **0 / 167** | 148 of 167 prompts exceed the 512-token context; the other 19 are wrong |
| Humanity's Last Exam (with tools) | Opus 5.5 67.7 % (official); GPT-6 Astra 57.2 % (official) | n/a | n/a | n/a | dataset gated on Hugging Face (blocked here); no tool use |
| ARC-AGI-3 (interactive) | GPT-6 Astra 62.7 % standard harness (ARC Prize) | n/a | n/a | n/a | needs an agent that plays games |
| FrontierMath Tier 4 | GPT-6 Astra ~98 % (vendor) | n/a | n/a | n/a | private dataset |
| GPQA Diamond | GPT-6 Astra ~96 % (reported) | n/a | n/a | n/a | gated dataset; chance is 25 % |
| SWE-bench Pro | Opus 5.5 89.9 % (reported) | n/a | n/a | n/a | needs an agent, repositories and Docker |
| Terminal-Bench 4.0 | Sonnet 5.5 70.6 %, Opus 5.5 66.4 % (official) | n/a | n/a | n/a | needs an agent in a terminal |
| HumanEval / MBPP (entry level) | saturated | 0 / 164, 0 / 500 | – | 0 / 164, 0 / 500 | 12.6M parameters |

Base 2 (base-g3) also scores 0 / 167 on ARC-AGI-2 (148 do not fit).

## What follows from it

- ARC-AGI-2 now runs in every cycle's report for all four families
  (`report.benchmarks`). The data is the public evaluation set from the Apache-2.0
  package `arckit==1.0.1`, fetched into `data/benchmarks/` (gitignored), sha256 in
  every result.
- The release rule (2x, `docs/LONGRUN.md`) keeps using the held-out suites as long
  as the hardest tests are at the floor: 0 against 0 cannot show 2x. As soon as a
  hardest test has passes, `scripts/releases.py` can gate on it with the same
  odds-ratio rule.
- The first hard limit is architectural: 89 % of ARC-AGI-2 prompts do not fit
  into 512 tokens. A longer context (RoPE extension to 2048+) belongs to the next
  model growth step.
- HLE and GPQA can be added as soon as huggingface.co is allowed in the network
  settings and access to the gated datasets is granted; FrontierMath is not public.
