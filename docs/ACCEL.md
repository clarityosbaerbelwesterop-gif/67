# accel: forge on TPUs, GPUs and any CPU

`accel/` is a JAX port of forge's model and trainer (`core_engine/`). It uses the same run
config, the same algorithm, the same stdout JSON lines and the same checkpoint files.
`accel/bin/forge-accel` stands in for the forge binary: `train`, `eval` and `logits` run on
JAX devices, and every other subcommand goes to the real forge.

Checkpoints move freely in both directions:
- accel can start from a forge checkpoint, resume it and evaluate it;
- forge can do the same with a checkpoint written by accel.

| Piece | File |
| --- | --- |
| forge's configs, LR schedules (f32, libm `cosf`), `num_params`, FLOPs per token | `accel/config.py` |
| Checkpoint files and the RNG (see the list below) | `accel/forge_io.py` |
| Model: params dict keyed by forge's names, `forward`, `row_losses`, `loss`, bit-exact init, `bf16` GEMM path | `accel/model.py` |
| `python -m accel.train --config RUN.json` (= `forge train`) | `accel/train.py` |
| `python -m accel.eval --ckpt DIR --data META ...` (= `forge eval`) | `accel/eval.py` |
| `python -m accel.logits --ckpt DIR --prompt-ids ...` (= `forge logits`) | `accel/logits.py` |
| Device mesh, data-parallel sharding, multi-host | `accel/runtime.py` |
| Drop-in binary for `cluster/fabric.py` and `scripts/longrun.py` | `accel/bin/forge-accel` |
| Parity tests against the forge binary | `accel/tests/test_parity.py` |

`accel/forge_io.py` covers:
- safetensors files, read and written through `scripts/rebase.py`;
- `state.json`, `latest.json`, `TRUNK` and `FINAL`, written byte for byte as forge writes them;
- forge's xoshiro256** RNG;
- the store reader and the window sampler;
- the `model_sha256` / `parent_sha256` lineage.

## What is the same as forge

| Part | Match |
| --- | --- |
| **Initialisation** | Bit for bit. accel replays forge's generator: `Rng(seed ^ 0x5EED1417)`, Box-Muller, f32 std. |
| **Data** | The same windows. accel uses forge's `Rng(seed)` and `below(N-T-1)` sampler. `rng_state` in `state.json` is equal after every step. |
| **LR** | Bit for bit, for every schedule (cosine, wsd, constant) and the warmup. Computed in f32, with the cosine through libm `cosf` like Rust's `f32::cos`. |
| **Bias correction** | Bit for bit. `t = optim_offset + step` and `0.9f.powf(t)`, through libm `powf`. |
| **AdamW** | Same operation order and the same f32 constants. `1 - 0.9f` is 0.100000024, not 0.1. Weight decay applies to ndim >= 2. |
| **Clipping** | Same: `min(clip / (norm + 1e-6), 1)` on the global pre-clip norm, and `grad_clip = 0` gives pure decay. |
| **Model** | Same as forge:<br>• RMSNorm without a mean;<br>• interleaved RoPE with forge's own f32 table;<br>• GQA with contiguous head groups;<br>• causal softmax and SwiGLU;<br>• tied head;<br>• no biases. |
| **Loss** | Same: f32 CE per row, rows summed in f64 per batch into an f32 register.<br>`xent="forge"` sums the softmax denominator strictly in order, as forge's `Xent` loop does. This is the default for every eval; see below. |
| **Gradient accumulation** | `sum(CE) * f32(1/(B*T*accum))`, with micro-batch gradients summed in order. |
| **Files** | Byte-identical to forge's:<br>• `model.safetensors` and `optim.safetensors` (m under `<name>`, v under `v.<name>`), written via `.tmp` and then renamed;<br>• `state.json` (sorted keys, ryu floats, 2-space indent, no trailing newline);<br>• `latest.json`;<br>• `TRUNK`, saved once at `max_steps - D` for wsd. |
| **Lineage** | `model_sha256` and `parent_sha256` (the `init_from` model, copied on resume) follow forge's rules. `optim_offset` is set by `init_optim`. |
| **Telemetry** | The same `event`, `telemetry`, `eval`, `checkpoint`, `trunk` and `done` lines, with the same keys.<br>`telemetry` also carries `devices`, `backend` and `precision`. |
| **stdin commands** | `pause`, `resume`, `stop`, `checkpoint` and `set_lr` are handled.<br>`autotune`, `set_threads` and `swap_kernel` are acknowledged and ignored, because they have no meaning on XLA. |

## Measured parity

Every number here comes from this host:
- forge: `target/release/forge`, default engine `avx512-f32` (this host has no AMX);
- accel: JAX 0.4.38 on CPU;
- both single-threaded under `nice -n 19` while the long run was training.

The command is `ACCEL_PARITY_REPORT=report.json nice -n 19 taskset -c 3 /home/user/accel-venv/bin/python -m pytest accel/tests -q`. It ran 20 tests, all passing, in 197 s.

### Forward pass and eval (`runs/base-g3/FINAL`, `runs/base-s/FINAL`)

| Check | forge | accel | Difference |
| --- | --- | --- | --- |
| `logits` base-g3, 9 ids, `--ctx 16` | argmax 46 | argmax 46 | max abs 1.10e-5 (max abs logit 7.03) |
| `logits` base-g3, ids 5..14, `--ctx 512` | argmax 13 | argmax 13 | max abs 2.22e-5 (max abs logit 8.38) |
| `logits` base-g3, 300 random ids, `--ctx 256` | argmax 287 | argmax 287 | max abs 8.11e-6 (max abs logit 8.27) |
| `logits` base-s, 40 random ids, `--ctx 64` | argmax 49 | argmax 49 | max abs 7.63e-6 (max abs logit 8.42) |
| `eval` base-g3, `data/stores/base`, val, b2 t64 ×3, seed 4242 | 2.834432601928711 / 0.4765625 | 2.834432601928711 / 0.4765625 | **0** |
| `eval` base-g3, `data/out/base-v2`, same settings | 2.471578359603882 / 0.5625 | 2.471578359603882 / 0.5625 | **0** |
| `eval` base-s, `data/stores/base`, train, b3 t48 ×2, seed 9 | 2.7626500129699707 / 0.4548611 | 2.7626500129699707 / 0.4548611 | **0** |

**Where the 1.1e-5 eval gap came from.** With a tree-summed softmax denominator (`--xent fast`), accel's eval loss was 1.14e-5 above forge's on both base-g3 cases (2.834444046 vs 2.834432602) and 1.05e-5 on base-s.

The cause is forge's `Xent` loop. It adds `exp(l - max)` over the 8192 logits one at a time in f32, and that sequential sum comes out about 4e-6 low. Replaying it on accel's own logits gives forge's f32 loss exactly (2.834432601928711).

So the `forge` mode of the loss replays that loop:
- it is a vocab-length `lax.scan`, used only for the value;
- the gradient is unchanged.

Every eval uses it. Training steps use it with `--xent forge`; the default for training steps is the faster tree sum, whose value is off by about 1e-5.

### Training

**Tiny model, fresh init.** V 64, d 32, L 2, H 4, KV 2, `synthetic:17`, seed 5. Run with batch 2, seq 16, accum 2.

**1. Init check: lr = 0, 1 step.**
- 0 of 32,928 init weights differ.
- `model.safetensors` is byte-identical: sha256 `7afa2f88…` from both.
- `state.json` is equal in every field except the step loss, which is 1 ulp apart (4.200462818 vs 4.200463295).
- forge's `state.json` and `latest.json` re-serialise byte for byte through accel's writer.

**2. Cosine run: 4 steps, warmup 2, clip 0.5 (active), wd 0.1.**
- **lr:** bit-identical at every step (0.005, 0.01, 0.01, 0.0055).
- **Loss:** within 4.8e-7.
- **grad_norm:** within 2.4e-7.
- **`rng_state` and `tokens`:** equal after every step.
- **After 4 steps:** weights within 2.77e-6 (median 3.7e-9, 0 elements above 1e-5); Adam m and v within 6.2e-6 and 5.4e-6 of each tensor's max.
- **Final eval:** forge 3.0297720432281494 / 0.734375, accel 3.0297720432281494 / 0.734375.

**3. wsd: 10 steps.**
- All 10 lrs are bit-identical: 0.005, then 0.01 ×7, 0.0055, 0.001.
- The `trunk` line appears at step 8 in both, and `TRUNK/state.json` has `"trunk": true`.

**Base model, one-window store.** This is `init_from runs/base-s/FINAL` (12,587,904 parameters).
- The store has seq + 2 = 18 tokens per split. forge's sampler `below(N - T - 1) = below(1)` therefore always picks window 0, so the batch is deterministic.
- Run settings: 2 updates, batch 2, seq 16, accum 2, lr 1e-3, warmup 2, clip 1.

| | step-1 loss | step-2 loss | grad_norm (steps 1, 2) | weights after 2 steps |
| --- | --- | --- | --- | --- |
| `init_optim` (warm moments, `optim_offset` 2800 in both) | 11.560369491577148 in both | 8.730643273 vs 8.730642319 | 53.738216 vs 53.738182; 34.378838 vs 34.378845 | max abs 1.19e-7 against a max update of 1.41e-3; 0 elements above 1e-6; m and v within 3.3e-6 and 4.3e-6 relative |
| fresh moments | 11.560369491577148 in both | 6.827474594 vs 6.827520847 | 53.738216 vs 53.738182; 29.708687 vs 29.709433 | max abs 2.70e-4; 582 of 12.6 M elements above 1e-5; median 1.4e-8 |

The fresh-moment gap is AdamW, not the port. With zero moments the first update is `lr·g/(|g| + eps')`, which is close to `lr·sign(g)`. For gradients that are essentially zero, the last-bit f32 noise decides the sign, so a few hundred weights move by a sizeable fraction of lr in different directions. The spec found the same thing with an independent reference.

With warm moments the update is a smooth function of g. The two implementations then agree to one f32 ulp of the weights.

**Round trips.**

| Direction | What was done | Result |
| --- | --- | --- |
| accel → forge | forge read an accel checkpoint with `eval` and `logits` | Eval loss and accuracy identical (3.045487403869629 / 0.703125). Logits max abs 1.19e-7, argmax equal. |
| accel → forge | `forge train --resume <accel step-2>` ran to step 4 | Matches forge's uninterrupted run: weights within 2.77e-6, step-4 loss identical (3.2456345558166504), `rng_state` equal. |
| forge → accel | Rewrote forge's `model.safetensors` and `optim.safetensors` with accel's writer | Same sha256. |
| forge → accel | `accel.train --resume <forge step-2>` ran to step 4 | Matches forge's uninterrupted run: weights within 1.19e-7, loss identical. |
| accel → accel | Resumed an accel run | Bit-identical to the uninterrupted run (same `model.safetensors` sha256). |

**Devices, precision and the cluster.**

| Check | Result |
| --- | --- |
| 2 devices (forced on the host) vs 1 device, tiny run | Losses within 4.8e-7; weights within 8.9e-7. The batch is split and the gradients are all-reduced. |
| `--remat` vs plain | Losses identical; weights within 4.3e-7. |
| `--precision bf16` vs f32, tiny run | Losses within 4.9e-4 over 4 steps (bf16 has 8 mantissa bits). |
| `cluster/fabric.py` with `FORGE=accel/bin/forge-accel` | `init --scratch` (accel) is forge's exact init to 1.8e-12; that is the one lr-1e-12 step. Then `work` ×2, `aggregate` and `eval` (val 4.166696548461914). `forge eval` of the same global model gives the identical 4.166696548461914. |

### Gaps that remain, and why

1. **Logits: ≤ 2.2e-5.** Both sides are true f32. They differ in GEMM summation order: forge's AVX-512 blocking with FMA versus XLA/Eigen. They also differ in `exp` and `rsqrt` implementations. Averaged into an eval loss this vanishes; the eval losses above are identical.

2. **grad_norm: ≤ 6.4e-7 relative.**
   - forge sums the squared gradients in f64.
   - accel sums in f32 with 128-wide levels (`pairwise_sum`), because f64 is slow or absent on TPUs.
   - Part of the difference is gradient noise from the GEMMs.

3. **Fresh-moment AdamW amplifies f32 noise** in near-zero gradients (see the table above). Use loss, grad_norm and moments as the parity metrics for fresh runs, as the spec says.

4. **forge itself is not bit-reproducible across thread counts.** `RmsNormBwd` dw is an f32 rayon fold. Autotune also picks GEMM engines by timing. Parity tests therefore pin forge to `--threads 1` / `autotune: false`.

5. **AMX / bf16 is not tested against forge.** `--precision bf16` implements amx-bf16's arithmetic: both GEMM inputs rounded to bf16 (RNE) with f32 accumulation, in forward and backward. But this host has no AMX, so it could not be compared with forge's `amx-bf16` engine. Runs trained on AMX hosts (base-g3, per `scripts/longrun.py`) used bf16 GEMMs for some shapes. Which shapes is in their telemetry logs, which were not read for this work.

## Running it

accel needs Python ≥ 3.10, `numpy` and `jax`. It was tested with jax 0.4.38 and uses only stable APIs: `jax.jit` with `NamedSharding`, `make_array_from_callback`, `lax.scan` and `custom_vjp`.

The commands take the same arguments as forge:

```sh
python -m accel.train  --config training/configs/base-s.json            # = forge train --config ...
python -m accel.train  --config RUN.json --resume runs/RUN/step-001000   # = forge train --resume
python -m accel.eval   --ckpt runs/base-g3/FINAL --data data/stores/base/train.meta.json \
                       --split val --batch 8 --seq 256 --batches 32 --seed 4242
python -m accel.logits --ckpt runs/base-g3/FINAL --prompt-ids 1,200,3000 --ctx 512
```

**accel-only flags** (`python -m accel.train --help`):

| Flag | Effect |
| --- | --- |
| `--precision f32` | True f32 GEMMs (`Precision.HIGHEST`). This is forge's f32 engines and the parity mode. |
| `--precision bf16` | bf16 GEMM inputs with f32 accumulation. This is forge's amx-bf16 arithmetic and the fast path on TPU MXUs and GPU tensor cores. Parameters, AdamW state and element-wise math stay f32. |
| `--remat` | Recompute each layer in the backward pass: less memory, roughly 1/3 more compute. |
| `--devices N` | Use the first N devices. |
| `--xent forge` | Report forge's exact loss value. |
| `--init fast` | A numpy init. The default replays forge's generator, which takes about 1.7 s per million weights. |
| `--no-final` | Do not write `FINAL`. |

**Environment variables** are honoured by `forge-accel`:

| Variable | Meaning |
| --- | --- |
| `ACCEL_PRECISION` | Default for `--precision`. |
| `ACCEL_REMAT` | Turns on `--remat`. |
| `ACCEL_DEVICES` | Default for `--devices`. |
| `ACCEL_XENT` | Default for `--xent`. |
| `ACCEL_PYTHON` | The Python with jax. |
| `FORGE_REAL` | The real forge binary for the other subcommands. |
| `ACCEL_DISTRIBUTED=1` | Multi-host. |

**Data parallelism.**
- Parameters and moments are replicated. Each batch, `[accum, batch, seq]`, is split along `batch` over a 1-D `data` mesh, and XLA all-reduces the gradients inside the compiled step.
- `batch` should be a multiple of the device count. Otherwise accel uses the largest device count that divides it and emits a warn event.
- The math does not depend on the device count; this was checked above with 1 and 2 devices.

### Any CPU

```sh
python3 -m venv ~/accel-venv && ~/accel-venv/bin/pip install "jax==0.4.38" numpy pytest
cd /path/to/67 && ~/accel-venv/bin/python -m accel.train --config RUN.json
```

On a CPU, forge is the faster trainer. accel on CPU is for parity tests and for machines without a Rust toolchain.

### TPU VM (Cloud TPU v4-8, v5e-8, ...)

```sh
pip install "jax[tpu]==0.4.38" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html numpy
git clone <this repository> 67 && cd 67          # or copy accel/, scripts/rebase.py, the store and the init checkpoint
python -c "import jax; print(jax.devices())"     # expect TpuDevice entries
ACCEL_PRECISION=bf16 python -m accel.train --config RUN.json > runs/RUN.jsonl
```

What a run needs on the machine:
- `accel/`;
- `scripts/rebase.py` (the safetensors reader and writer);
- the data store (`train.meta.json` plus the two `.bin` files);
- the `init_from` checkpoint directory, if the config has one.

**Multi-host slices (v4-32, v5e-16, ...).**
- Run the same command on every host with `ACCEL_DISTRIBUTED=1` (`jax.distributed.initialize()`).
- Every host needs the store. All hosts draw the same global batch from the same seed, and each feeds only its own shards.
- Process 0 prints and writes checkpoints, so `out_dir` must be on host 0's disk or on shared storage.
- This path is written but untested here; there was no slice available.

### Kaggle / Colab TPU

The TPU runtimes ship with `jax[tpu]`. Check `jax.devices()` first, then:

```python
!git clone <this repository> 67
%cd 67
!ACCEL_PRECISION=bf16 python -m accel.train --config RUN.json
```

Copy the checkpoint directory back out before the session ends; notebook sessions are time-limited.

### GPU (CUDA 12)

```sh
pip install "jax[cuda12]==0.4.38" numpy
ACCEL_PRECISION=bf16 python -m accel.train --config RUN.json     # bf16 tensor cores
python -m accel.train --config RUN.json                           # true f32 (no TF32)
```

## The cluster and the long run

**`cluster/fabric.py`** calls `$FORGE train` and `$FORGE eval` and reads `latest.json`, the telemetry `loss` and `tokens_per_s`, and the eval `loss`. Point it at the wrapper:

```sh
export FORGE=$PWD/accel/bin/forge-accel ACCEL_PYTHON=$(which python) ACCEL_PRECISION=bf16
python3 cluster/fabric.py work --state S --worker 3 --steps 300 --data META --out D
```

This was tested end to end (`init --scratch`, `work`, `aggregate`, `eval`) in `test_fabric_through_forge_accel`. TPU/GPU workers and CPU forge workers can be mixed in one round because their deltas have the same format. In the workflow (`.github/workflows/forge-cluster.yml`), a `forge`-labelled self-hosted runner with an accelerator would set `FORGE` the same way.

**`scripts/longrun.py`** takes `--forge` (or `$FORGE`). Started with `--forge accel/bin/forge-accel` on an accelerator host:
- `train` (the base, specialist and RSI training runs), the gate's `eval` (`--batch 8 --seq 256 --batches 32 --seed 4242`) and `logits` run on accel;
- `bench`, `merge`, `generate` and `tokenize` go to the real forge, so the first cycle's throughput plan comes from `forge bench`;
- later cycles plan from the telemetry median of `tokens_per_s`, which accel reports.

The running long run was not touched by this work. Moving it to an accelerator means stopping it at a checkpoint and restarting it on that host with `--forge accel/bin/forge-accel`. That is an operator decision. accel reads every checkpoint it wrote, and `forge train --resume` reads accel's.

## Throughput: what to expect

**Measured on this host.** This is one core under `nice -n 19` while the long run was busy on the others (load average about 4).
- Setting: base model (12.6 M params), batch 1, seq 256, `init_from runs/base-s/FINAL`, steps 2–6.
- forge: median 650 tok/s.
- accel f32: median 405 tok/s.
- accel bf16: 367 tok/s. bf16 is emulated on this CPU.

So on a CPU, forge is about 1.6× faster than XLA. accel is for accelerators.

**Estimated for accelerators. These numbers were not measured here.**

For the base config at seq 256, one token costs `6·N + 12·L·d·T` = **82.6 MFLOP**. That is `ModelConfig.flops_per_token`, the same count forge's telemetry uses for `tflops`.

On one accelerator:

`tokens/s ≈ MFU × peak FLOP/s ÷ 82.6 MFLOP`

A model this small (d = 384, matrices of 384×1024) keeps systolic arrays and tensor cores poorly fed. Assume an MFU of 10–30 % with `--precision bf16` and a batch large enough to fill the device.

| Device (vendor bf16 peak) | 10 % MFU | 30 % MFU |
| --- | --- | --- |
| TPU v5e chip (197 TFLOP/s) | ~240 k tok/s | ~720 k tok/s |
| TPU v4 chip (275 TFLOP/s) | ~330 k tok/s | ~1.0 M tok/s |
| A100 (312 TFLOP/s) | ~380 k tok/s | ~1.1 M tok/s |

These are upper-bound arithmetic, not promises. forge on this machine class does about 2,000 tok/s on 4 cores (`docs/CLUSTER.md`), so even the 10 % column is about 100× forge on CPU.

**What decides the real number:**
- **Per-step work.** Base runs use batch 8 × seq 256 × accum 4 = 8,192 tokens per update. That is about 0.7 TFLOP per update, a few milliseconds on a TPU chip. Fixed per-step costs then dominate: dispatch, host-side window sampling and the loss fetch, which happens one step behind. To use an accelerator well, raise `batch`, for example to 64–512 rows. More tokens per step is a change to the recipe; LR and warmup should be re-tuned, and that is a training decision, not part of the port.
- **`--precision f32` on TPU** is several times slower than bf16, because XLA runs f32 matmuls as multi-pass bf16. Use bf16 to train and f32 for parity checks.
- **The first telemetry line includes XLA compilation** (seconds to a minute). Steady-state `tokens_per_s` comes from the later lines; `scripts/longrun.py` uses a median.

The first run on a real TPU or GPU reports the true number in its telemetry `tokens_per_s` and `tflops`, which are measured, not assumed.

## Tests

```sh
nice -n 19 taskset -c 3 /home/user/accel-venv/bin/python -m pytest accel/tests -q   # -s prints every measured number
```

The tests skip themselves when the forge binary or the base checkpoints are missing. Every process runs on one CPU thread. The tolerance of each assertion and the reason for it are next to the assertion in `accel/tests/test_parity.py`.

## Not in v1

- `--quant` for eval. Use `forge eval --quant`.
- `forge bench`, `merge`, `generate`, `tokenize`, `diloco` and `disasm`. They are passed through to forge.
- Per-op telemetry: `ops` is `{}`, and `hbvm` holds the device allocator's numbers where the backend reports them.
- `step_ms` is the mean step time since the previous telemetry line, because dispatch is asynchronous. forge reports the time of the last step.
- Sequence or tensor parallelism. Data parallelism covers every model in this repository. Models larger than one device's memory would need parameter sharding over the mesh (FSDP), which is a small change in `accel/runtime.py` and `build_step`.
