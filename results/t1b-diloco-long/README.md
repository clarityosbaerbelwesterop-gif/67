# T1b result: DiLoCo vs DDP with longer training (run 37273497357, 2026-10-05)

**T1b: FAIL** under its pre-registered gate (`experiments/t1b-diloco-long.json`).

The gate required DiLoCo with H = 100 to be no more than 5 % worse than DDP while each worker sends at least 100× fewer bytes. It is 7.6 % worse.

Setup: as T1 (3.28M-parameter native Rouge model, enwik8, 4 workers, same schedule shape), but 5,000 steps instead of 2,000 and new seeds 10–12. Free CPU runners.

| Arm | Validation BPB (mean ± sd, 3 seeds) | Bytes sent per worker | vs DDP |
|---|---|---|---|
| ddp (gradients every step) | 1.701 ± 0.005 | 65,582 MB | — |
| diloco-h100 (Nesterov, lr 0.7, m 0.9) | 1.829 ± 0.002 | 656 MB (100× less) | +7.6 % |
| diloco-h100-avg (plain averaging) | 1.847 ± 0.010 | 656 MB (100× less) | +8.6 % |
| diloco-h100-int8 | 1.832 ± 0.003 | 164 MB (399× less) | +7.7 % |
| diloco-h250 | 1.877 ± 0.005 | 262 MB (250× less) | +10.4 % |

## What the numbers say

- **The gap closes with longer training, but not far enough.** At H = 250 the gap fell from 28.6 % (T1, 8 outer steps) to 10.4 % (T1b, 20 outer steps). At H = 100 it is 7.6 %.
- **The instability of T1 is gone.** With 20–50 outer steps, every DiLoCo arm has a seed spread as small as DDP's (sd ≤ 0.010; T1 H = 250 had 0.214).
- **The outer Nesterov step helps.** Plain averaging is 1.0 % worse than Nesterov at the same bytes.
- **int8 compression is lossless (T2 confirmed).** It costs +0.15 % against fp32 at a quarter of the bytes, so 399× less traffic than DDP.
- **Token efficiency.** DiLoCo H = 100 after 5,000 steps (1.829) reaches about what DDP reached after 2,000 steps in T1 (1.826; other seeds, same setup). At this model size, DiLoCo therefore needs roughly 2.5× the tokens for DDP's quality, in exchange for 100–400× less communication.

## What this means for the fabric

DiLoCo pays only where bandwidth, not compute, is the bottleneck.

- **Inside one GPU node** (8×H200 with NVLink), synchronous FSDP/DDP stays the method. Nothing in T1/T1b argues for DiLoCo there.
- **Across poorly connected machines** (different providers, home devices), DiLoCo with int8 deltas is the working protocol: communication drops 400×, at a measured quality cost of about 8 % at equal tokens on this small model.
- No claim is made for 27B. Published DiLoCo results come from much larger models and runs; this repository has not tested that scale.

## Provenance

- All 15 arm jobs of run 37273497357 succeeded.
- The `verdict` job, which collects the artifacts and commits them, was not started. GitHub reported: "recent account payments have failed or your spending limit needs to be increased" (Actions billing of this private repository).
- The result files here were therefore reconstructed from the `FABRIC_RESULT` line that each arm printed to its job log. Each file names its job id. The validation curves, parameter count and corpus hash remain in the run's artifacts.
- `VERDICT.json` was produced by `fabric/gate.py` on these files, unchanged.
