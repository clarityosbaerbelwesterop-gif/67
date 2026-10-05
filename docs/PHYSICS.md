# Physics of a pure-code accelerator — what 67 Forge can and cannot be

Date: 2026-10-05. Every number below is either measured in this repository
(command given) or cited. Estimates are marked as estimates.

## 1. The target: 100 × NVIDIA B300

| Quantity (one B300, dense) | Value | Source |
| --- | --- | --- |
| BF16 | 2.25–2.5 PFLOPS | NVIDIA HGX B300 / GB300 NVL72 pages |
| FP8 | 4.5–5 PFLOPS | same |
| FP4 | 13.5–15 PFLOPS | same |
| HBM3e | 288 GB at 8 TB/s | same |

100 × B300 ≈ **225–250 PFLOPS BF16**, 1.35–1.5 EFLOPS FP4, ~28 TB HBM,
~800 TB/s, joined by NVLink at 1.8 TB/s per GPU.

## 2. What code runs on, measured

Software does not create FLOPs; it only decides how much of the silicon
underneath is used. "Virtual silicon" that emulates a GPU on a CPU is slower
than the CPU itself, so 67 Forge does the opposite: it drives the matrix
hardware that commodity processors already contain.

| Silicon | Measured / cited | How |
| --- | --- | --- |
| Intel AMX tile unit, 1 core (this sandbox, Xeon Sapphire Rapids) | **1.94 TFLOPS BF16** register-bound | `tdpbf16ps` loop, measured 2026-10-05 |
| `forge bench` end to end | see `docs/BENCHMARKS.md` | measured by the CLI, never typed in |
| Apple M4 / M5 iPad GPU, FP32 peak | ~4.3 / ~5.1 TFLOPS (third-party estimates) | eatyourbytes.com |
| Safari WebGPU matmul efficiency | ~25 % of peak FP32, ~47 % f16 | AnswerDotAI/gpu.cpp PR #39 |
| GitHub-hosted runner (public repo) | 4 vCPU AMD EPYC 7763, ~0.16–0.22 TFLOPS FP32 peak (estimate) | github/docs, Geekbench |

## 3. The gap, in numbers

| Pool | Effective throughput | Gap to 100 × B300 BF16 |
| --- | --- | --- |
| This sandbox, 4 cores AMX (best case) | ≤ ~7.7 TFLOPS peak | ≥ ~30,000× |
| One iPad, practical WebGPU | ~1.1–2 TFLOPS | ~150,000× |
| 20 GitHub runners (terms forbid this use, see §5) | ~2–4.5 TFLOPS | ~56,000–120,000× |
| 1,000 volunteer iPads, ideal | ~1.5–5 PFLOPS | ~44–167× |

Every algorithmic lever was checked against its primary source:

| Lever | Real effect | Source |
| --- | --- | --- |
| FP8 / FP4 arithmetic | 2–6× only on hardware with FP8/FP4 units (B300 yes; iPad and AVX2 no) | DeepSeek-V3 report |
| BitNet b1.58 ternary weights | inference memory and energy; ~1× training FLOPs (latent weights stay full precision) | arXiv 2402.17764, 2504.12285 |
| GaLore / LoRA | optimizer memory −65 % to −82 %, not FLOPs | arXiv 2403.03507 |
| DiLoCo | 400–500× less communication, which makes swarms of weak devices feasible; adds 0 FLOPs | arXiv 2311.08105, 2407.07852 |
| μP | cheaper hyperparameter search (~7 % of a 6.7B run) | arXiv 2203.03466 |
| Distillation from a stronger teacher | the largest lever for narrow skills (phi-1: 50.6 % HumanEval at ~4e20 FLOP) | arXiv 2306.11644 |
| Test-time compute + verifiers | ≥4× over best-of-N; can match a ~14× larger model, only where the model already succeeds sometimes | arXiv 2408.03314 |

**Combined, the levers buy about 10²–10³ of effective compute on narrow tasks.
The gap is 10⁴–10⁵ in throughput and 10⁵·⁷–10⁸·⁷ in frontier training compute
(GPT-5 ≈ 5e25 FLOP, Grok 4 ≈ 5e26 FLOP, Epoch AI). It cannot be closed by code.**
67 Forge therefore never claims to be "as strong as 100 B300"; `forge bench`
prints the measured fraction instead.

## 4. What this means for Rouge 1, Quasnir and Darus

- Chinchilla-optimal sizes for budgets we can reach: 1e18 FLOP → ~91M
  parameters on ~1.8B tokens; 1e17 → ~29M on ~0.6B tokens (arXiv 2203.15556).
- nanochat's 4e19 FLOP model is "a bit like talking to a kindergartener"
  (karpathy/nanochat README); SmolLM2-135M alone used ~1.6e21 FLOP.
- Frontier models (Gemini 3.x Pro: GPQA 91.9–94.3 %, SWE-bench Verified
  76–81 %) are 5–9 orders of magnitude more training compute away. Claims of
  beating them would be fabricated; the project reports measured perplexity
  and small verifiable tasks only.
- Darus is the TIES merge of Rouge 1 and Quasnir. Both are continued from one
  shared base checkpoint, verified by SHA-256 before merging.

## 5. Platform terms

GitHub's additional product terms forbid using Actions for "any activity that
places a burden on our servers ... disproportionate to the benefits" and, on
hosted runners, "any other activity unrelated to the production, testing,
deployment, or publication of the software project" (github/site-policy). 67
Forge therefore uses Actions for build, tests, short smoke training and
evaluation only. Real training runs on hardware the owner controls, such as a
Mac, a Linux box or a Codespace within its quota.

## 6. The SCP "compute swarm"

`swarm-compute-protocol-/backend/core/protocol.py:147-177` spawns its 10,000
nodes from a random-number generator; no route registers real workers, and
work packages are random `blob://` strings. It contributes 0 FLOPs. 67 Forge
uses the parts of SCP that are real: its model definition (weight-compatible),
its BPE tokenizer and token-store format, and its data generators and filters.
