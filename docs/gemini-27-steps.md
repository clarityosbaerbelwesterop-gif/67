# The Gemini "Swarm-vGPU" plan, step by step

Source: the owner's screenshots of a Gemini conversation (2026-10-05): a 27-step plan "from the foundation to the Swarm-vGPU", claiming 100 × B300 performance. This file says, for every step seen, what the fabric does with it.

## Corrections to the claims behind the plan

| Gemini claim | What is true | Evidence |
|---|---|---|
| Ternary (BitNet) cuts memory 10–15×; the model fits in L2/L3 cache | Weights shrink about 10× vs bf16. A 27B ternary model is still about 5.3 GB, while caches are MBs | R1.29b (Osirus): 2.6× smaller checkpoint at equal quality |
| No FP multiplications, so 10–50× compute; "10 TFLOPS chip does 50–200 TOPS" | Ternary matmuls become additions only at inference: 1.37–6.17× on CPUs. Backprop keeps floating point | bitnet.cpp, arXiv 2410.16144 |
| DiLoCo makes internet latency irrelevant | True for latency and bandwidth. But each worker holds a full replica (27B ≈ 430 GB to train) | DiLoCo arXiv 2311.08105; OpenDiLoCo arXiv 2407.07852 |
| Top-k 0.1 % + INT8 saves 99.9 % bandwidth | Plausible for bytes; the quality cost must be measured | T2 in `experiments/t1-diloco.json` |

## The steps

| # | Gemini step | Decision | Why / how |
|---|---|---|---|
| 1 | Monorepo, Git LFS with S3/IPFS backends for weights | **Adapted** | Git holds code, configs and hashes only; weights and data live in the private Lightning registry with sha256 manifests (already in Osirus) |
| 2 | Rust workspace (core, kernel, p2p, cli) | **Deferred** | Python + PyTorch first; Rust only where a profile shows the coordinator is the bottleneck |
| 3 | WebRTC + Kademlia DHT P2P | **Deferred** | Workers are known machines; a public P2P swarm needs Byzantine-robust aggregation first |
| 4 | GitHub ARC on Kubernetes/bare metal | **Later, owner hardware only** | Self-hosted runners are allowed; GitHub-hosted runners as a compute swarm are not |
| 5 | Zero-copy binary schema (Protobuf/FlatBuffers) | **Adapted** | Deltas travel as torch/safetensors files with byte counts; schema when a network transport exists |
| 6 | Zero-trust WASM sandbox for untrusted nodes | **Deferred** | Only with untrusted nodes |
| 7 | BitNet b1.58 SIMD kernel (AVX-512, NEON, AMX) | **Adapted** | Start with bitnet.cpp (Microsoft, MIT; licence to re-check); own kernels only after measuring it |
| 8 | WebGPU/WGSL shaders, browsers join the swarm | **Inference demo at most** | Browsers cannot hold a training replica |
| 9 | DiLoCo local-SGD engine | **Kept: the core** | `fabric/diloco.py`, `fabric/swarm.py`; test T1 |
| 10 | Top-k sparsifier + INT8 | **Kept** | `Codec("topk:…")` with error feedback, `Codec("int8")`; test T2 |
| 11 | Ring attention over hundreds of nodes | **Dropped** | Needs a per-layer high-bandwidth interconnect; only inside one machine |
| 12 | Work-stealing scheduler | **Kept (later)** | Job queue of the fabric router |
| 13 | Out-of-band control plane; Actions only for macro events | **Kept** | Coordinator outside Actions once jobs leave GitHub runners |
| 14 | Thin workflows that start a CLI | **Kept** | Workflows call `python -m fabric.*` |
| 15 | 60-s synthetic pre-flight check before heavy compute | **Kept** | `--steps` smoke mode; Osirus already runs smoke tests before paid jobs |
| 16 | Speculative branches for hyperparameter hypotheses | **Adapted** | Matrix jobs, not git branches |
| 17 | Auto-merge the branch with the steepest loss drop into main | **Dropped** | Replaced by pre-registered gates and the owner's OK |
| 18 | Commit only checksums and metadata | **Kept** | Results JSON and sha256, never weights |
| 19 | Cron deleting branches and tags every 24 h | **Dropped** | Risk of losing work; artifacts expire by retention instead |
| 20 | Live telemetry (Prometheus/Grafana/W&B) | **Adapted** | JSONL/JSON results and a ledger first |
| 21 | Cache-blocking for L1/L2/L3 | **Experiment** | Only measured kernels (T5) |
| 22 | Ternary arithmetic "10 TF → 50–200 TOPS" | **Corrected** | See above; measured in T5 |
| 23 | 1000-step isolation scaling | **Experiment** | H ∈ {50, 250} in T1; larger H next |
| 24 | Massive heterogeneous swarm: idle Actions runners, cloud, Macs, browsers; "10,000 standard nodes beat 100 B300 in effective BitNet FLOPs" | **Corrected, deferred** | 10,000 CPU nodes at about 0.2 TF ≈ 2 PF. Even with bitnet.cpp's measured 6× (inference only) that is about 12 PF against 1,500 PF FP4: a factor of about 100 short. Training needs a full replica per node, which consumer nodes lack. "Idle Actions runners" of other repositories are not ours to use (GitHub terms) |
| 25 | Lossless asynchronous checkpointing (TorchSnapshot over P2P); half the workers fail, training continues | **Adapted** | The global state lives with the aggregator and is checkpointed every round (`state.pt`). A round now survives lost workers down to a quorum (`average_payloads`, `--min-workers 3` of 4); a worker that misses a round rejoins from the next global state. Checkpoints use `torch.save`; sharded DCP as in the Osirus trainer when models grow |
| 26 | Decentralised vector/shard database; GitHub keeps only state, data streams out of band | **Adapted** | Data shards with sha256 manifests in the private Lightning registry (built in Osirus); GitHub holds code, configs and result hashes. No vector DB is needed for training |
| 27 | End-to-end benchmark, MFU, "verify exceeding 1.5 ExaFLOPS FP4" | **Kept as measurement, not as a promise** | The ledger (`docs/compute-atlas.md` §4) records measured MFU and B300-equivalent hours per job. The expected value with free compute is far below 1.5 EF, and the report will say so |

## The four "anti-overload" rules

| Rule | Decision | How |
|---|---|---|
| 1. Workflows at most 15 lines, no logic, only a CLI call | **Adapted** | All logic lives in `python -m fabric.*`; workflows only set up and call it. Setup steps (checkout, Python, cache) stay; a shared composite action can shorten them later. 15 lines is not a goal in itself |
| 2. Communication out of band (P2P/WebSockets), never through GitHub logs or the API | **Kept for scale** | Pilot rounds pass deltas as Actions artifacts (fine for 4 workers and MBs). Beyond pilots, deltas go to object storage (Lightning registry), not the GitHub API. GitHub stores only results and hashes |
| 3. Monorepo crate isolation (Rust) | **Deferred** | Python packages are isolated by module; Rust only after profiling |
| 4. Ephemeral runners: start, compute one bounded work package, send the delta, exit | **Kept, already true** | Each `swarm work` job trains exactly H steps, uploads one delta and its private optimiser state, and ends |
