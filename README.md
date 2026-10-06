# 67 Forge — a pure-code AI accelerator stack

Every layer between the model and the silicon is our own source code: number
formats, tensor instruction set, memory manager, compiler, kernels, collective
communication, training loop and the iPad mission control. There is no CUDA, no
cuBLAS, no NCCL and no PyTorch. The "systolic arrays" are real: forge drives
the matrix units that commodity processors already contain (Intel AMX tiles,
AVX-512/AVX2 vector units). The iPad is the control center.

**What it is not:** as strong as 100 NVIDIA B300. Code does not create FLOPs.
`forge bench` measures this machine and prints the measured fraction of one B300
and of 100 B300. `docs/PHYSICS.md` documents the gap with sources, along with
every lever (FP8/FP4, ternary weights, DiLoCo, distillation, test-time compute)
and what each one really buys.

```
iPad (Safari) ── WebSocket/HTTPS ──► headcenter/  FastAPI + agents
                                         │  stdin: JSON commands     ▲ stdout: JSON telemetry
                                         ▼                           │
                                   forge train ──► forge-isa program (bytecode F67I)
                                         │            compiler: fusion · DCE · HBVM placement
                                         ▼
                 interpreter ──► forge-kernels: AMX-BF16 │ AVX-512 │ AVX2 │ scalar (autotuned per shape)
                      │          forge-formats: FP8 E4M3/E5M2 · FP4 E2M1 · MX block · BF16/FP16 (bit-exact)
                      ▼
                 forge-hbvm (aligned arena, live fragmentation) · forge-comm (ring all-reduce over TCP)
```

## Repository

| Path | Contents |
| --- | --- |
| `core_engine/formats` | bit-exact low-precision formats per OCP OFP8 / MX v1.0 |
| `core_engine/kernels` | GEMM engines with runtime detection and per-shape autotuning |
| `core_engine/hbvm` | High-Bandwidth Virtual Memory: tensor arena with telemetry |
| `core_engine/isa` | 27-opcode tensor ISA, register file, verifier, bytecode, compiler, interpreter |
| `core_engine/comm` | ring all-reduce / broadcast over TCP with CRC-checked frames |
| `core_engine/data` | SCP tokenizer (BPE v2) and token-store reader |
| `core_engine/train` | SCP-1 transformer as ISA programs, AdamW, checkpoints, TIES merge, sampler |
| `core_engine/cli` | `forge` binary |
| `headcenter/` | iPad mission control: FastAPI, WebSocket telemetry, Silicon Watchdog and JIT agents, live control, WebGPU model hand-off |
| `data/` | corpus build on top of the SCP pipelines (licensed sources, provenance manifests) |
| `training/configs/` | run configs for `base-s`, `rouge-1`, `quasnir-1` and the `darus-1` merge |
| `docs/DESIGN.md` | binding interface contract |
| `docs/PHYSICS.md` | measured and cited limits |

## Bootstrap protocol

1. **Synthesize virtual silicon:** `cargo build --release -p forge`. `forge engines`
   lists the matrix engines this CPU exposes, for example `amx-bf16`.
2. **Launch bus and memory:** the HBVM pool is sized from the compiled
   programs (`forge disasm --config …` prints the plan: fused instructions,
   transient peak versus naive).
3. **Spin up the headcenter:** `python -m headcenter.backend --host 0.0.0.0 --new-token`
   (see `headcenter/README.md`) and open the printed URL on the iPad.
4. **Execute the training graph:** `scripts/train_all.sh` runs base θ0 →
   Rouge 1 and Quasnir (both continued from θ0) → Darus = TIES(θ0; Rouge 1,
   Quasnir), then evaluates all of them. Every run resumes from its newest
   checkpoint after an interruption and takes live commands from the iPad
   (pause, LR, threads, kernel swap, autotune, checkpoint) through
   `runs/<run>.ctl`.

Run sizes follow the measured budget, not wishes: on the 4-core AVX-512 host
used here one training step of 8,192 tokens at 12.6M parameters takes ~4.7 s
(~1,700 tok/s, 0.14 TFLOPS sustained), so `base-s` is the compute-optimal size
(Chinchilla) for roughly four hours of this machine.

## The three models

| Model | Data | Origin |
| --- | --- | --- |
| Rouge 1 | general prose (`data/out/general`) | continued from θ0 |
| Quasnir | code + secure-coding sources (`data/out/code`) | continued from θ0 |
| Darus | — | TIES merge of Rouge 1 and Quasnir on θ0; accepted only if no suite regresses more than 2 % against the best parent |

Architecture: SCP-1 (RMSNorm, RoPE, GQA, SwiGLU, tied head) with weights in
safetensors under SCP's PyTorch parameter names, so checkpoints load into
`swarm-compute-protocol-/model`.

## Verification

- `cargo test --workspace --release` runs these checks:
  - every backward instruction against central differences;
  - the whole transformer's gradient end to end;
  - AdamW against the torch formula;
  - SHA-256 against the FIPS vectors;
  - bytecode round trips for all opcodes;
  - convergence on a synthetic stream, with greedy sampling reproducing it.
- `python -m pytest headcenter/tests` covers the headcenter: incremental
  ingestion, command validation, the FIFO control channel, both agents in
  every mode (including that replayed history never triggers an action),
  token auth, the WebSocket protocol and checkpoint serving confined to `runs/`.
- `runs/report.json` holds measured losses, perplexities and accuracies of
  every model. No number in this repository is a projection.
