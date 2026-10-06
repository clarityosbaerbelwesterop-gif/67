# How far Darus and Rouge 1 are from the frontier, and how long closing it would take

Date: 2026-10-06. The frontier numbers below are third-party reports, cited at
the end; Gemini 4 Pro's are leaks and not official. Our numbers are measured
in this repository (`docs/BENCHMARKS.md`, `runs/report.json`,
`runs/benchmarks.jsonl`).

## The reference models (reported)

| Model | Reported results |
| --- | --- |
| Claude Opus 5.5 | Humanity's Last Exam 67.7 % with tools, 64.4 % without. Top of the Artificial Analysis Intelligence Index (58). |
| GPT-6.1 Sol (29 Sep 2026) | DeepSWE v1.1 75.2 %. Artificial Analysis Intelligence Index 52. |
| Gemini 4 Pro (leaked, unreleased) | DeepSWE v1.1 88 %, Terminal-bench 2.1 95.3 %, OSWorld-2.0 86.8 %, MMLU 94.7 %. |
| GPT-7/8, Fable 5.5/6 | Not released. Nothing exists to measure against. |

## What decides the gap: compute

| Quantity | Value |
| --- | --- |
| Frontier training run in 2026 (Epoch AI–based estimates) | ~1e27 FLOP (median scenario 9.5e26) |
| This host, sustained in training (measured) | 0.14 TFLOPS = 1.4e11 FLOP/s |
| 10 days on this host | 1.2e17 FLOP |
| Ratio | about 8,000,000,000× (≈10^10) |

Time to a frontier-sized pretraining run at the measured rate:

    1e27 FLOP / 1.4e11 FLOP/s = 7.1e15 s ≈ 226 million years

The same run in 10 days would need about 1.2e21 FLOP/s sustained. At ~40 %
utilisation of a B300 (2.5 PFLOPS dense BF16, ~1 PFLOPS effective), that is
about 1.2 million B300 for 10 days. The rate quoted earlier in this project
(100 B300 ≈ €19,000/day, ≈ €7.9 per GPU-hour) puts the compute alone at about
2.8e8 GPU-hours ≈ €2.2 billion. With 100 B300 the run would take about 317 years.

Data is the second gap. Frontier pretraining uses tens of trillions of tokens.
Corpus v2 has 0.4 billion; v3 with FineWeb-Edu would reach roughly 1–1.5 billion.

## What RSI and long training change

- **Controlled RSI** (`training/rsi/`) raises a model on task families it can
  verify. It does not create FLOPs. Each round costs sampling plus
  fine-tuning compute. At frontier labs, RL and self-improvement
  post-training runs themselves take 1e25 FLOP and more.
- **Longer training** of a model this size improves validation loss along a
  power law with diminishing returns. The Chinchilla fit shows 10 vs 20 days
  of this host differing by a few percent in loss, not by orders of magnitude.
- **More machines** through `forge diloco` scale throughput linearly. Even
  10,000 hosts like this one are 1.4 PFLOPS, still about 10^6× short of a
  10-day frontier run.

## Answer to "how long until Darus beats Gemini 4 Pro / GPT-7 / Fable 6?"

On this hardware it never happens in any meaningful timeframe; the arithmetic
gives about 10^8 years for compute alone. The gap is physical: FLOPs, data
and capital. Code, training duration or RSI cannot close it. What this
project can do and measure:

- the best model **for its size and budget** (validation loss vs compute,
  measured);
- verifiable progress on controlled tasks (RSI held-out pass rates, audited);
- honest public-benchmark numbers (HumanEval, MBPP pass@1 with real
  execution).

For this model size these are expected to stay near 0 %. They are reported
as measured.

The frontier benchmarks above (HLE, DeepSWE, OSWorld, Terminal-bench) need
agentic tool use and knowledge far beyond a 12.6M-parameter model. They are
not run here, because a score of 0 on them would carry no information.

## Sources

- [Claude Opus 5.5 benchmarks explained (Vellum)](https://www.vellum.ai/md/blog/claude-opus-5-5-benchmarks-explained)
- [Claude Opus 5.5: benchmarks, pricing (MetricNexus)](https://metricnexus.ai/blog/claude-opus-5-5-benchmarks-pricing)
- [GPT-6.1 Sol benchmarks explained (Vellum)](https://vellum.ai/blog/gpt-6-1-sol-benchmarks-explained)
- [GPT-6.1 Sol benchmarks (Emergent)](https://emergent.sh/learn/gpt-6-1-sol-benchmarks)
- [Gemini 4 Pro leaked benchmarks (KuCoin News)](https://www.kucoin.com/news/flash/gemini-4-pro-leaks-outperforms-astra-and-fable-in-benchmarks)
- [Gemini 4 Pro: 88 % on DeepSWE (BornCity)](https://borncity.com/news/google-gemini-4-pro-88-prozent-bei-deepswe-test-erreicht/)
- [Trends in frontier AI model count, forecast to 2028 (arXiv 2504.16138)](https://arxiv.org/html/2504.16138v1)
- [Epoch AI: model counts and compute thresholds](https://epoch.ai/blog/model-counts-compute-thresholds)
- [Frontier training compute chart (Dror Poleg)](https://data.drorpoleg.com/charts/frontier-compute/)
