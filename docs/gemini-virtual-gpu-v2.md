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
| B6 Virtual Precision Emulation (FP4/FP8 on CPU) | FP4/FP8 computed "mathematically" on CPUs | CPUs have no FP8/FP4 units. Emulation saves memory and transfer bytes, but computes slower than native fp32/bf16. FP8 pays off only on Hopper/Blackwell tensor cores |

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
