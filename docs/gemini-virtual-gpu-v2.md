# Gemini's "virtual GPU" plans, round 2 (owner screenshots, 2026-10-05)

Two further 10-step plans: (A) a virtual layer over a real H100/H200 cluster, and (B) a purely virtual cloud GPU on GitHub runners. Steps seen so far, checked:

## A: layer over a real GPU cluster

| Step | Claim | Verdict | What is real instead |
|---|---|---|---|
| A1 Virtual Silicon Emulation Layer | PyTorch is shown "100 virtual B300" instead of the real H100/H200 | **No effect or harm.** A label changes no FLOPs; a framework that believes in memory that is not there runs out of it or spills to the CPU and slows down | The real cards, used well: FSDP2 sharding (Rouge 1 already does this on 8 × H200) |
| A2 Virtual Tensor Core Sharding with Triton kernels | Darus's maths split into fragments on the real tensor cores "without overhead" | **Partly real.** Triton/fused kernels (FlashAttention, fused cross-entropy, fused RMSNorm) give typically 10–30 % more throughput or less memory, not a different GPU class | Measure fused kernels on the H200 job before adopting them |
| A3 Infinite Virtual VRAM | — (text cut off) | **Real form exists:** ZeRO-Infinity / FSDP offload keeps optimiser states and parameters in CPU RAM or on NVMe, so a model larger than GPU memory still trains, much slower | Option for Darus-scale models on few GPUs; cost measured per step |

## B: purely virtual GPU on GitHub runners

| Step | Claim | Verdict |
|---|---|---|
| B1 Distributed Runner Orchestration | Matrix jobs start hundreds of runners as a "serverless network" of vGPUs | **Not allowed for this purpose.** GitHub's terms forbid serverless use of Actions and, on hosted runners, work unrelated to the repository's own software project. Matrix jobs for this repository's own tests and pilots (T1, swarm proof) are fine. A runner is 4 CPUs, no GPU |
| B2 Virtual Execution Kernels | C++/Rust kernels "emulate B300 matmul" in runner RAM | **A CPU matmul stays a CPU matmul.** Good BLAS reaches about 0.1–0.3 TFLOPS on 4 cores, against about 2,500 TFLOPS BF16 on one B300 |
| B3 Git-sharded model weights | Each runner loads a tiny shard of Darus, "no out-of-memory possible" | **Breaks two ways.** (1) Weights never go into git (program rule; GitHub file limits). (2) Splitting one model across machines (tensor/pipeline parallelism) sends activations between the parts at every layer of every step. Over the internet that is about 10,000× slower than NVLink. Data parallelism with DiLoCo (our T1) avoids that, but needs a full replica per worker |
| B4 Serverless Interconnect (WebSockets/gRPC between workflows) | Real-time gradient streaming between runners | Hosted runners accept no inbound connections; a relay would carry about 0.1–1 GB/s against 1,800 GB/s NVLink. Usable only for rare, compressed DiLoCo deltas |
| B5 Webhook-driven training loops, 24/7 | Each finished job triggers the next, autonomously around the clock | **Excluded:** a continuous training service on hosted runners is the serverless use GitHub's terms forbid. Long training belongs on GPUs the program pays for, or on self-hosted runners on our own hardware |
| B6 Virtual Precision Emulation (FP4/FP8 on CPU) | FP4/FP8 computed "mathematically" on CPUs, "ten times faster" | CPUs have no FP8/FP4 units. Emulation saves memory and transfer bytes, but computes slower than native fp32/bf16. FP8 pays off only on Hopper/Blackwell tensor cores |
| B7 Zero-hardware failover | A watchdog restarts a failed runner elsewhere; Rouge 1's training never stops | **Adapted, and already built:** a DiLoCo round survives lost workers down to a quorum, and the global state is saved every round. A failed job is re-run once at most. A watchdog that restarts jobs on its own would be the 24/7 loop of B5. Rouge 1 itself trains on 8 × H200 with checkpoints and exact resume (Osirus `full.py`). Runners cannot be placed "elsewhere in the world" |
| B8 Chunked data preprocessing workflows | Pipelines tokenise, clean and chunk the training data, separate from training | **Real and already built** in Osirus: `rouge-data.yml`, the streaming corpus build, sha256 manifests, decontamination, private registry. Preparing this repository's own data is a legitimate Actions use |
| B9 Automated security and bug audit after each run | Check code and computed weights for logic errors and security holes | **Code:** CI, tests and dependency and secret scans are real; this repository runs its tests on every push. **Weights:** a checkpoint has no "security holes" to scan. What matters: (1) integrity: sha256 manifests (Osirus); (2) safe loading: never unpickle untrusted files. `fabric/swarm.py` now loads every delta and state with `torch.load(weights_only=True)`, so a tampered delta cannot run code; (3) behaviour: pre-registered evals and gates before any promotion |
| B10 Serverless API gateway with Stripe | External developers send prompts to a GitHub workflow, billed through Stripe | **Excluded.** Answers would take minutes (job start), it is the serverless use GitHub's terms forbid, and it would resell free compute. A paid API belongs in Osirus (Vercel) with models served on rented GPUs, plus the legal pages (AGB, privacy, imprint) |
| "Parallel circuitry": B300 cores → thousands of workflows | The GPU's thousands of cores mapped onto thousands of workflows | GPU cores share on-chip memory with nanosecond latency. Workflows need minutes to start and talk over the internet (milliseconds, MB/s). Only work that rarely synchronises (DiLoCo rounds, independent eval shards) maps onto many machines |
| Pipelining / streaming | Keep data permanently moving | **Real principle:** prefetching and overlapping communication with compute. It is used inside each job, not between workflows |

## Gemini's two questions

1. **FP8 or FP16 for Darus?**
   - **Neither as a gut decision.**
   - Today's standard on H100/H200/B200 is BF16 compute with FP32 master weights, which is what Rouge 1's FSDP2 trainer does. BF16 is safer than FP16 (no loss scaling).
   - FP8 training (DeepSeek-V3 style, fine-grained scaling) gives roughly 1.3–2× throughput on Hopper/Blackwell. That is a pre-registered experiment on a real H200 against BF16 at equal tokens, before Darus.
   - On CPU runners, FP8 is pointless (see B6).
2. **An API so other developers train on "the code GPU"?** Not now:
   - There is no compute of our own to sell.
   - Free tiers (GitHub, Kaggle, Colab) may not be resold or shared.
   - Running strangers' training code is a security project of its own.
   - Possible later as an Osirus product on paid, owned or rented GPUs.

## Round 3: "software intelligence instead of emulation" (2026-10-05)

Gemini now agrees that emulating 100 × B300 is physically impossible. Instead it proposes four levers that reach a similar output with less compute. This direction is right; the numbers need correcting.

| Lever | Claim | What is established | Where it already lives / next step |
|---|---|---|---|
| Data density ("golden tokens") | 100 GB of perfect synthetic data = 15 TB of web text, 99 % saving | Data quality is a real multiplier (filtered corpora such as FineWeb-Edu; textbook-style data). The 150× figure is unproven. Breadth of knowledge needs broad data, and purely synthetic training risks narrowing | Osirus `rft-v1`: only answers checked by code are kept; teacher data from self-hosted DeepSeek-V4-Pro. Outputs of closed APIs are excluded, so Gemini's "/data_synthesis via API" is not used |
| Test-time compute (MCTS + process reward models) | A 14–32B model searching for 10 s matches a 10× larger model | Real within limits: compute-optimal test-time scaling can beat a 14× larger model on problems the base model already solves sometimes (Snell et al. 2024, arXiv 2408.03314). It costs more compute per answer and does not help where the model has no chance | Fits Rouge 1 (27B), whose verifiers already exist. Proposal: measure majority-of-8 and verifier best-of-8 against greedy on the 300 primary held-out items in `rouge-1-rl-001` |
| Fine-grained MoE | Hundreds of billions of parameters, under 5 % active per token | Real: DeepSeek-V3 has 671B total and 37B active (256 routed experts, 8 active plus 1 shared; arXiv 2412.19437). It saves compute per token, not memory | Native candidate D; R1.22 measured 3.3× from sparse dispatch; Darus's candidate base V4-Flash is MoE |
| Evolutionary merging and LoRA stacking | The model grows without retraining | Merging is real (Sakana AI, arXiv 2403.13187; MergeKit). A merge does not add parameters or capacity, and adapters can interfere | Quesnir as a code/security specialist of Rouge 1, adapter or full fine-tune; the merge measured against keeping separate experts. The LoRA pipeline exists in Osirus (M58) |
