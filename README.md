# 67: Rouge compute fabric ("Swarm-vGPU")

The GPU repository of the Osirus/Rouge program. Its job is to give Rouge 1, Quesnir and Darus as much useful training and serving compute as the hardware we can actually reach allows, and to measure every step.

## The honest target

**Software cannot create FLOPs or memory.**

| | 100 × B300 | Free compute we can reach |
|---|---|---|
| Memory | 28.8 TB HBM | — |
| Memory bandwidth | about 800 TB/s | — |
| Compute | about 250–350 PF BF16 dense, 1.5 EF FP4 dense | about 0.01 % of that, running continuously |

The free compute is GitHub CPU runners, Kaggle T4/P100/TPU v5e-8, Colab T4 and small credits; see [`docs/compute-atlas.md`](docs/compute-atlas.md). A "pure-code GPU as strong as 100 B300" therefore does not exist.

What does exist:

1. **Effective compute.** Reach the same model quality with less hardware: low-communication training (DiLoCo), compressed updates, ternary weights, sparse experts, distillation, better data. Every factor is measured here, not assumed.
2. **Reach.** One fabric that sends each job to the cheapest machine able to run it:
   - free CPU runners for tests and pilots;
   - Kaggle GPUs/TPUs;
   - Lightning H200/B200 (paid, owner-approved);
   - the owner's devices for serving (exo, llama.cpp, bitnet.cpp).

Full-parameter training of Rouge 1 (27B), Quesnir and Darus needs real GPUs: at 16 bytes/param, 27B ≈ 430 GB per replica. The fabric orchestrates that; it does not replace it.

## What is here

| Path | What |
|---|---|
| `fabric/diloco.py` | DiLoCo engine: outer Nesterov step, delta codecs (fp32, fp16, int8, top-k with error feedback) and their byte counts |
| `fabric/pilot.py` | T1/T2 pilot: DiLoCo vs synchronous data parallelism at equal tokens, simulated in one process |
| `fabric/swarm.py` | the same algorithm across real machines: `init` / `work` / `aggregate`, only files travel |
| `fabric/gate.py` | applies the pre-registered gate to committed results |
| `experiments/t1-diloco.json` | pre-registration: arms, seeds, gate |
| `.github/workflows/t1-diloco.yml` | 5 arms × 3 seeds on free runners; commits results and verdict |
| `.github/workflows/swarm.yml` | 3 DiLoCo rounds, 4 workers on 4 runners, deltas as artifacts |
| `docs/compute-atlas.md` | every GPU/TPU source, the limits, the ledger |
| `docs/gemini-27-steps.md` | the Gemini plan, step by step: kept, adapted, dropped, and why |
| `docs/scp-audit.md` | read-only audit of `swarm-compute-protocol-` |

The model and data come from a pinned Osirus commit: the native Rouge model in `training/rouge/native` and the enwik8 loader in `research/rouge-architecture/benchmarks`.

## Rules

- **Repository and secrets:**
  - This repository stays private.
  - No weights, datasets or secrets in git; weights live in the private Lightning registry.
  - Secrets are referenced by name only.
- **Providers:**
  - One account per provider; quotas are never multiplied.
  - GitHub-hosted runners run only this repository's own tests and pilots (GitHub Additional Product Terms, Actions).
- **Money and gates:**
  - Every paid run needs the owner's approval and stays under the program ceiling.
  - No merge to `main` without the owner's OK.
  - No claim beyond a measurement.

## Run

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
python -m unittest discover -s tests -t .
python -m fabric.pilot --osirus ../Osirus --arm diloco-h250-fp32 --seed 0 --steps 50 --out /tmp/smoke.json
```
