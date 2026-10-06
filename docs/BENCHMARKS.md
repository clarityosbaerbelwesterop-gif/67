# Measured benchmarks

Every number here was printed by a tool in this repository on the host named
in the heading. Nothing here is a projection. Re-measure with
`forge bench [--config run.json]`.

## Host of 2026-10-06

The sandbox host:

- **CPU:** Intel Xeon @ 2.80 GHz, 4 vCPUs, 15 GB RAM.
- **Instruction sets:** AVX-512F and AVX2/FMA.
- **No AMX:** CPUID leaf 7 EDX bits 22 and 24 are clear, and XCR0 tile
  state is off. forge therefore offers `scalar`, `avx2-fma` and `avx512-f32`.
  The AMX engine is verified here through its SDM-exact emulator test.
- **Earlier host:** the sandbox of 2026-10-05 exposed AMX and measured
  1.94 TFLOPS BF16 on one core (`docs/PHYSICS.md`).

### GEMM

`forge bench`, 4 threads, GFLOP/s:

| m × n × k (trans) | avx512-f32 | avx2-fma | scalar |
| --- | ---: | ---: | ---: |
| 4096 × 512 × 512 (N,T) | 320 | 176 | 64 |
| 4096 × 1408 × 512 (N,T) | 227 | 170 | 54 |
| 4096 × 512 × 1408 (N,N) | 266 | 184 | 57 |
| 512 × 512 × 4096 (T,N) | 183 | 164 | 56 |
| 2048 × 2048 × 2048 (N,N) | 351 | 219 | 77 |

### Training, end to end

`forge bench --config …`: forward, backward and AdamW, with 8,192 tokens per
step.

| Model | Params | Step | Tokens/s | Sustained TFLOPS | Share of one B300 (2,500 TFLOPS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| d512 L8 H8 KV4 (first base-s) | 27.8 M | 9.04 s | 906 | 0.163 | 0.0065 % |
| d384 L6 H6 KV2 (base-s as trained) | 12.6 M | 4.94 s | 1,660 | 0.137 | 0.0055 % |

Where the step time goes in base-s, from live telemetry:

| Op | Share of step time | Rate |
| --- | ---: | ---: |
| MATMUL | 72 % | ~205–217 GFLOP/s |
| MATMUL_BATCHED (attention) | 11 % | ~130 GFLOP/s |
| All element-wise ops together | ~17 % | memory-bound |

### Inference and tooling

| What | Measured |
| --- | --- |
| Generation, 12.6 M model, KV-cache decoder (`forge generate`) | 9 ms/token, while training runs on the same cores |
| Generation, same model, window recomputation (`--decoder window`) | ~220 ms/token, same conditions |
| Tokenizer, 3.65 M characters | Rust 0.6 s vs Python reference 12.7 s, identical tokens |
| forge-comm all-reduce, 2 ranks on localhost, 102 k values, int8 | 2.4 ms |

## The gap

At 0.137–0.163 TFLOPS sustained, this host is about 1/15,000 to 1/18,000 of
one B300 and about 1/1,500,000 of 100 B300. Software tuning moves the first
number by small factors, not by orders of magnitude. More silicon is what
scales it: AMX hosts, more machines through `forge diloco`, and GPUs reached
through WebGPU.
