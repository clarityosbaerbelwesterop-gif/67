# T1/T2 result: DiLoCo vs synchronous data parallelism (run 37266101375, 2026-10-05)

**T1: FAIL** under its pre-registered gate (`experiments/t1-diloco.json`).

Setup: 3.28M-parameter native Rouge model, enwik8, 4 workers, 2000 steps, 3 seeds per arm, free CPU runners. The gate required DiLoCo with H = 250 to be no more than 3 % worse than DDP while sending at least 100× fewer bytes.

| Arm | Validation BPB (mean ± sd, 3 seeds) | Bytes sent per worker | vs DDP |
|---|---|---|---|
| ddp (gradients every step) | 1.826 ± 0.024 | 26,233 MB | — |
| diloco-h50-fp32 | 2.034 ± 0.002 | 525 MB (50× less) | +11.4 % |
| diloco-h250-fp32 | 2.348 ± 0.214 | 105 MB (250× less) | +28.6 % |
| diloco-h250-int8 | 2.365 ± 0.238 | 26 MB (1,000× less) | +29.5 % |
| diloco-h250-topk1 | 2.963 ± 0.156 | 1.6 MB (16,700× less) | +62.3 % |

- **Communication:** cut exactly as designed (H× for fp32, 4× more with int8).
- **Quality:** fell short, and more so the longer the rounds: H = 50 cost 11 %, H = 250 cost 29 %.
- **H = 250 is unstable:**
  - With only 8 outer steps in 2,000 inner steps, the outer Nesterov step (lr 0.7, momentum 0.9) overshoots.
  - Validation BPB rises again in later rounds on 2 of 3 seeds.
  - The seeds spread ten times wider than DDP's.
- **T2:**
  - int8 compression of the delta is lossless relative to fp32 (+0.7 %, inside seed noise) at a quarter of the bytes.
  - Top-k 1 % costs a further 26 %.

**What this does not show.** DiLoCo's published results (arXiv 2311.08105, 2407.07852) come from far longer runs with many more outer steps. This test cannot say whether the gap closes with longer training. That question is T1b, pre-registered separately; this verdict stands as it is.
