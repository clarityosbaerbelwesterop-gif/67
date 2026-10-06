# 67 Forge — design contract

Status: binding interface contract for every module. Change it only together
with all affected modules.

## 0. Principle

Code does not create FLOPs; silicon does. "Pure code" here means: every layer
of the accelerator stack — number formats, instruction set, memory manager,
scheduler, kernels, collective communication, compiler, training loop — is our
own source code, with no vendor driver stack (no CUDA, no cuBLAS, no NCCL, no
PyTorch). The "systolic arrays" are driven on matrix silicon that already
exists in commodity processors:

- Intel AMX (TMUL tiles; a 16x16 systolic array per core) via inline assembly,
- AVX-512 / AVX2 FMA vector units as the vector engine,
- (stage 2) Apple GPUs on iPad/Mac through WebGPU.

Every performance number the project states is measured by `forge bench` on
the machine it refers to. Nothing is projected. See `docs/PHYSICS.md`.

## 1. Workspace

```
core_engine/formats   forge-formats   bit-exact BF16, FP16, FP8 (E4M3/E5M2), FP4 (E2M1), MX block formats
core_engine/kernels   forge-kernels   GEMM engines (AMX/AVX-512/AVX2/scalar), thread pool, autotune
core_engine/hbvm      forge-hbvm      High-Bandwidth Virtual Memory: tensor arena + telemetry
core_engine/isa       forge-isa       tensor ISA, bytecode, graph IR, fusion compiler, interpreter
core_engine/comm      forge-comm      ring all-reduce over TCP (no NCCL)
core_engine/data      forge-data      SCP tokenizer (BPE v2) + SCP token store reader + RNG
core_engine/train     forge-train     SCP-compatible transformer, AdamW, checkpoints, TIES merge, DiLoCo
core_engine/cli       forge (bin)     bench | train | merge | eval | generate
headcenter/                           iPad mission control: FastAPI + WebSocket + agents + touch UI
data/                                 corpus build on top of the SCP pipelines
training/configs/                     declarative run configs for rouge-1, quasnir-1, darus-1
```

Rust stable (edition 2021). Allowed dependencies are those in the root
`[workspace.dependencies]`; anything else needs a reason in the PR.

## 2. forge-formats (no dependencies)

Normative references: OCP 8-bit Floating Point Specification (OFP8) rev 1.0;
OCP Microscaling Formats (MX) v1.0; IEEE 754-2019.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding { NearestEven, TowardZero, Stochastic(u32) } // Stochastic carries 32 random bits

pub mod bf16 { pub fn from_f32(x: f32) -> u16; pub fn to_f32(h: u16) -> f32; }     // RNE, NaN stays NaN (quiet)
pub mod fp16 { pub fn from_f32(x: f32) -> u16; pub fn to_f32(h: u16) -> f32; }     // IEEE binary16
pub mod fp8 {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Kind { E4M3, E5M2 }
    pub fn decode(kind: Kind, code: u8) -> f32;
    /// saturate=true clamps finite overflow to ±max (448 / 57344); false maps to NaN (E4M3) / Inf (E5M2).
    pub fn encode(kind: Kind, x: f32, r: super::Rounding, saturate: bool) -> u8;
    pub fn max_finite(kind: Kind) -> f32;
}
pub mod fp4 { pub fn decode(code: u8) -> f32; pub fn encode(x: f32, r: super::Rounding) -> u8; } // E2M1, saturating, low nibble
pub mod mx {
    pub const BLOCK: usize = 32;
    #[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Elem { Fp8E4M3, Fp8E5M2, Fp4E2M1, Int8 }
    pub struct MxTensor { pub elem: Elem, pub len: usize, pub scales: Vec<u8> /* E8M0 */, pub data: Vec<u8> /* FP4: 2 per byte, low nibble first */ }
    pub fn quantize(x: &[f32], elem: Elem, r: super::Rounding) -> MxTensor;   // scale = 2^(floor(log2(amax)) - emax_elem)
    pub fn dequantize(t: &MxTensor) -> Vec<f32>;
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format { F32, Bf16, Fp16, Fp8(fp8::Kind), Fp4, Mx(mx::Elem) }
/// Fake-quantize in place (quantize then dequantize): for low-precision simulation and QAT.
pub fn fake_quant(x: &mut [f32], f: Format, r: Rounding);
pub fn bits_per_element(f: Format) -> f32; // storage cost incl. MX scale amortisation
```

Tests: exhaustive over every FP8/FP4 code (decode table, encode∘decode = id for
non-NaN, monotonic, nearest), BF16/FP16 against bit tricks over a dense sweep,
MX error bounds, stochastic rounding unbiased within statistical tolerance.

## 3. forge-kernels

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Trans { N, T }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmVariant { Scalar, Avx2Fma, Avx512F32, AmxBf16 }
impl GemmVariant { pub fn name(self) -> &'static str; pub fn from_name(s: &str) -> Option<Self>; }
/// Runtime detection. AmxBf16 requires CPUID amx-bf16 + a granted XTILEDATA permission (arch_prctl).
pub fn available() -> Vec<GemmVariant>;
pub fn best_available() -> GemmVariant;

pub struct Pool { /* rayon::ThreadPool */ }
impl Pool { pub fn new(threads: usize) -> Pool; pub fn threads(&self) -> usize;
            pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R; }

/// Row-major. C[m×n] = alpha·op(A)·op(B) + beta·C, op(A) is m×k, op(B) is k×n.
/// lda/ldb/ldc are row strides of the stored (untransposed) matrices.
/// beta == 0 must ignore (overwrite) existing C, including NaN.
pub fn gemm(pool: &Pool, v: GemmVariant, ta: Trans, tb: Trans, m: usize, n: usize, k: usize,
            alpha: f32, a: &[f32], lda: usize, b: &[f32], ldb: usize, beta: f32, c: &mut [f32], ldc: usize);
/// Strided batch of independent GEMMs (attention heads).
pub fn gemm_batched(pool: &Pool, v: GemmVariant, ta: Trans, tb: Trans, batch: usize, m: usize, n: usize, k: usize,
            alpha: f32, a: &[f32], lda: usize, stride_a: usize, b: &[f32], ldb: usize, stride_b: usize,
            beta: f32, c: &mut [f32], ldc: usize, stride_c: usize);
pub struct TuneResult { pub variant: GemmVariant, pub gflops: f64 }
/// Measure every available variant on this shape; sorted fastest first.
pub fn autotune(pool: &Pool, ta: Trans, tb: Trans, m: usize, n: usize, k: usize) -> Vec<TuneResult>;
```

Accuracy: fp32 variants within 1e-4 relative (vs f64 reference, scaled by
sqrt(k)); AmxBf16 rounds inputs to bf16 and accumulates in f32 — tolerance
consistent with bf16 inputs. All four Trans combinations, ragged edges.

## 4. forge-hbvm

A tensor arena ("High-Bandwidth Virtual Memory"): one 64-byte-aligned pool,
size-class free lists, handles instead of pointers, live telemetry.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub struct Buf(pub u32);
pub struct Hbvm;  // new(capacity_floats), alloc(len) -> Buf, free(Buf), slice(Buf) -> &[f32], slice_mut(Buf) -> &mut [f32]
pub struct HbvmStats { pub capacity: usize, pub used: usize, pub peak: usize, pub live: usize, pub allocs: u64, pub frees: u64, pub fragmentation: f64 }
```

## 5. forge-isa — tensor instruction set

Coarse-grained tensor ISA (TPU-style): one instruction = one tensor primitive
over HBVM buffers. A program is a verified bytecode sequence; the compiler
fuses adjacent primitives; the interpreter dispatches to kernels and counts
FLOPs, bytes and nanoseconds per opcode.

Bytecode: little-endian, header `b"F67I" u16 version u16 flags u32 n_instr`,
then per instruction `u8 opcode, u8 n_operands, u16 attr_len, u32 operands[n], attr bytes`.

Opcodes (forward and backward): MATMUL (shape/trans attrs, variant slot),
MATMUL_BATCHED, ADD, ADD_INPLACE, MUL, SCALE, EMBED, EMBED_BWD, RMSNORM,
RMSNORM_BWD, ROPE, ROPE_BWD, SOFTMAX_CAUSAL, SOFTMAX_BWD, SILU_MUL (fused
SwiGLU gate), SILU_MUL_BWD, XENT (fused softmax+cross-entropy+grad), COPY,
ZERO, ADAMW (fused optimizer step), QUANT (fake-quant to a forge-formats
Format), FUSED_* (emitted only by the compiler).

## 6. forge-comm — collectives without NCCL

```rust
pub enum Compression { None, Bf16, Int8Block { block: usize } }
pub struct Ring { pub rank: usize, pub world: usize, /* ... */ }
impl Ring {
    /// Listen on `listen`, connect to peers[(rank+1)%world], accept from the left neighbour.
    pub fn connect(rank: usize, world: usize, listen: std::net::SocketAddr, peers: &[std::net::SocketAddr],
                   timeout: std::time::Duration) -> std::io::Result<Ring>;
    pub fn all_reduce_sum(&mut self, buf: &mut [f32], c: Compression) -> std::io::Result<()>; // reduce-scatter + all-gather
    pub fn broadcast(&mut self, buf: &mut [f32], root: usize) -> std::io::Result<()>;
    pub fn barrier(&mut self) -> std::io::Result<()>;
    pub fn stats(&self) -> CommStats; // bytes_sent, bytes_received, ops, last_ms
}
pub struct CommStats { pub bytes_sent: u64, pub bytes_received: u64, pub ops: u64, pub last_ms: f64 }
/// Test helper: `world` rings on 127.0.0.1 ephemeral ports, connected in threads.
pub fn local_rings(world: usize) -> std::io::Result<Vec<Ring>>;
```

Wire frames: `b"F67C" u32 seq u32 len_bytes u32 crc32 payload`. A CRC mismatch is an error.
A hello frame `[0x4F4C4C45, rank, world]` on every new connection rejects a
miswired ring. Each ring step sends and receives concurrently (no deadlock on
full socket buffers). Compression is wire-only: partial sums stay f32, and the
all-gather forwards the received bytes unchanged, so every rank decodes the
same encoding and ends bit-identical. `Int8Block` sends one f32 scale
(max|x|/127) plus i8 codes per block.

DiLoCo transport: `forge_train::diloco::RingCollective { ring, compression }`
implements `Collective`; the initial synchronisation is an exact `broadcast`
from rank 0, and only the pseudo-gradients are compressed. CLI:
`forge diloco --config C --rank R --world W --peers h0:p,…,hW-1:p
[--inner-steps H] [--outer-lr 0.7] [--momentum 0.9] [--compression none|bf16|int8[:B]]`
(rank r samples with seed + r; rank 0 writes the checkpoint).

## 7. forge-data — SCP contract

Reads the SCP CorpusStore layout unchanged (`swarm-compute-protocol-/model/scp_model/store.py`):
`tokenizer.json` (v2: `{version:2, pattern, specials:["<bos>","<eos>","<pad>"], merges:[[a,b,idx]]}`;
ids 0..255 bytes, merge k → 256+k, then bos/eos/pad = V-3/V-2/V-1), `train.bin` /
`train.val.bin` (headerless LE uint16 or uint32 per `train.meta.json` `dtype`), `manifest.json`.
Pre-tokenisation regex is SCP's `GPT_SPLIT_PATTERN` with Python-`re` Unicode semantics.

```rust
pub struct Tokenizer; // load(path), encode(&str) -> Vec<u32>, decode(&[u32]) -> String, vocab_size(), bos(), eos(), pad()
pub struct TokenStore; // open(meta_path), train_len(), val_len(), vocab_size(), batch(&mut Rng, Split, b, t) -> (Vec<u32>, Vec<u32>)
pub enum Split { Train, Val }
pub struct Rng; // xoshiro256**: new(seed), next_u64(), next_f32(), state() -> [u64;4], from_state([u64;4])
```

Batch semantics (SCP `data_pipeline.py`): window start uniform in [0, N-T-1],
x = w[0..T], y = w[1..T+1]; windows cross document boundaries.

## 8. Model (forge-train) — SCP-1 equations

Llama family exactly as `scp_model/model.py`: RMSNorm (eps 1e-5), interleaved
RoPE (theta 1e4), GQA, SwiGLU `w2(silu(w1 x) * w3 x)`, tied embeddings, AdamW
(0.9, 0.95, eps 1e-8), warmup + cosine to min_lr, global-norm clip 1.0.
Checkpoints: safetensors (f32, SCP parameter names) + JSON state (step,
optimizer moments, RNG state, config, tokenizer sha256, parent sha256).
Darus = θ0 + TIES(τ_rouge, τ_quasnir); both children must share θ0 by sha256.

## 9. Telemetry and control protocol (trainer ⇄ headcenter)

The trainer writes one JSON object per line on stdout and reads one JSON
command per line on stdin.

Telemetry (`"type":"telemetry"`, every `log_every` steps):
```json
{"type":"telemetry","run":"rouge-1","step":120,"tokens":983040,"loss":4.21,"val_loss":null,
 "lr":0.0006,"grad_norm":0.91,"step_ms":812.4,"tokens_per_s":10084,"tflops":1.37,
 "threads":4,"paused":false,
 "ops":{"MATMUL":{"calls":96,"ns":610000000,"flops":8.3e11,"bytes":1.2e9}},
 "kernels":{"MATMUL":"amx-bf16"},
 "hbvm":{"capacity":0,"used":0,"peak":0,"live":0,"allocs":0,"frees":0,"fragmentation":0.0},
 "comm":null,"ts":1759660000.123}
```
Events: `{"type":"event","level":"info|warn|error","msg":"...","step":n,"ts":...}`;
`{"type":"autotune","op":"MATMUL","shape":[m,n,k],"results":[{"variant":"amx-bf16","gflops":1234.5}],"chosen":"amx-bf16"}`;
`{"type":"checkpoint","path":"...","sha256":"...","step":n}`; `{"type":"done","step":n,"loss":x}`.

Commands: `{"cmd":"pause"}`, `{"cmd":"resume"}`, `{"cmd":"stop"}`,
`{"cmd":"set_lr","value":0.0003}` (overrides schedule peak), `{"cmd":"set_threads","value":2}`,
`{"cmd":"swap_kernel","op":"MATMUL","variant":"avx512-f32"}`, `{"cmd":"autotune"}`,
`{"cmd":"checkpoint"}`. Unknown commands produce a warn event, never a crash.
