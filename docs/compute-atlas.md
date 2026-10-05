# Compute Atlas (as of 2026-10-05)

Every source of training or serving compute the program can reach, what it delivers, and its limits. Secondary sources are marked; re-check before relying on a number. Speeds are vendor peak values unless measured here.

## 1. The reference: 100 × NVIDIA B300

| | One B300 | 100 × B300 |
|---|---|---|
| HBM3e | 288 GB | 28.8 TB |
| Memory bandwidth | 8 TB/s | 800 TB/s |
| NVLink | 1.8 TB/s | — |
| Dense FP4 | 15 PF | 1.5 EF |
| Dense FP8 | about 7 PF | about 700 PF |
| Dense BF16 | about 2.5–3.5 PF (sources differ) | 250–350 PF |

Sources: The Register (2025-03-18), vast.ai, datacrunch.io B300-vs-B200. To be verified against NVIDIA's datasheet before any number is quoted outside this repo.

## 2. Sources we can reach

| Source | Hardware | Quota / price | Peak BF16/FP16 | How the fabric uses it | Limits |
|---|---|---|---|---|---|
| GitHub-hosted runners (this repo) | 4 vCPU, 16 GB, no GPU | free for private repos within the account's minutes; public repos free | ≈ 0.1–0.3 TF (CPU) | tests, pilots (T1, swarm proof) | ToS: only this repository's own software project; no serverless, no disproportionate burden |
| Kaggle Notebooks | 1 × P100 16 GB or 2 × T4 16 GB | about 30 GPU-h per week, sessions ≤ 12 h | T4 ≈ 65 TF FP16 tensor (peak) | small training and eval jobs via the Kaggle API (`kernels push`) | phone-verified account; one account; secondary source (gpuperhour.com, 2026-09) |
| Kaggle TPU | TPU v5e-8, 8 × 16 GB HBM | sessions ≤ 9 h; weekly quota to be verified | ≈ 8 × 197 TF ≈ 1.6 PF BF16 | the strongest free source: JAX or torch-xla training of native Rouge models | XLA port needed; quota and terms to verify |
| Google Colab (free) | T4 | 15–30 h per week, varies | ≈ 65 TF | manual experiments only | no stable automation API; usage terms restrict unattended use |
| Hugging Face ZeroGPU | RTX Pro 6000 Blackwell, 48/96 GB | 5 min per day (free account) | — | demos of serving, not training | too small for training |
| SageMaker Studio Lab | T4 | free with limits | ≈ 65 TF | manual experiments | account approval |
| Modal | various GPUs | small monthly credits | — | short GPU jobs | credit-bound |
| Lightning AI (teamspace "Rouge") | H200, B200, 8-GPU machines | paid; balance 1.27 credits on 2026-09-30 | H200 ≈ 1 PF BF16 dense | Rouge 1 RSI iterations, teacher, Darus later | owner approval (`rouge-gpu`), ceiling 128.66 USD |
| Owner's devices (iPad, iPhone) | Apple silicon | — | — | serving small models (exo/MLX, llama.cpp), not training | owner: not a priority now |
| IONOS H200 (named in the SCP runbook) | 1 × H200 141 GB | unknown | ≈ 1 PF | — | not confirmed to exist; owner to confirm |

Free sources together, running non-stop, ≈ 0.01 % of 100 × B300. In bursts the Kaggle TPU v5e-8 alone is about half of one B300 at peak.

## 3. Where the limits really are

**Training memory.** Full-parameter AdamW needs about 16 bytes per parameter: bf16 weights and gradients, fp32 master weights and two moments.

| Model | Memory per replica |
|---|---|
| Rouge 1 (27B) | ≈ 430 GB |
| DeepSeek-V4-Flash-class Darus base (291B) | ≈ 4.7 TB |

Low-communication methods (DiLoCo, DisTrO) keep one full replica per worker. They remove the network bottleneck, not the memory one.

**Ternary weights** (BitNet b1.58):
- Weights are about 10× smaller than bf16. Our R1.29b measured 2.6× smaller checkpoints at equal quality, with only the MLP ternary.
- Training still uses high-precision latent weights and optimiser states, so it gains little memory.
- Inference on CPU: 2.37–6.17× on x86 and 1.37–5.07× on ARM (bitnet.cpp, arXiv 2410.16144).
- Not 10–50×, and not for backprop.

**Communication.**

| Method | Result |
|---|---|
| DiLoCo (arXiv 2311.08105) | synchronises every H ≈ 500 steps, so H× less traffic |
| OpenDiLoCo (arXiv 2407.07852) | 1.1B parameters across 2 continents at 90–95 % utilisation |
| Nous DisTrO/DeMo | up to 10,000× less bandwidth (vendor claim); a 15B test run over the internet |
| prime-rl (Apache-2.0) | asynchronous RL from 1 node to more than 1,000 GPUs (INTELLECT-3, 106B MoE) |

## 4. The ledger

**Unit: B300-equivalent hours delivered** = measured useful FLOPs ÷ (2.5 PF × 3600 s).

**What counts as useful:** forward + backward FLOPs of the job (6 × params × tokens), only when the job finished and its result was used.

`results/ledger.json` (from the first real jobs on) records, per job:
- source;
- hardware;
- wall time;
- tokens;
- measured MFU;
- B300-equivalent hours.

**Effective-compute multipliers** (DiLoCo, codecs, ternary, MoE) are recorded separately. A multiplier is only entered after a pre-registered test passes.
