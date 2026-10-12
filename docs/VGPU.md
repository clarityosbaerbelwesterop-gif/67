# VGPU: the swarm as one virtual accelerator

The "code GPU" of forge does not create compute; it gathers it. Every machine that
joins (CPU box, Mac, PC, iPad or phone in a browser) adds its processor or GPU
to one pool. The pool's capacity is **verified**, never self-reported, and counted in
B300 equivalents (2,500 dense BF16 TFLOP/s each, the forge reference). Once the pool
holds the target capacity, the long run moves base training onto it.

## How capacity is verified (proof of compute)

1. The coordinator (`headcenter/backend/vgpu.py`) draws a 32-bit seed and sizes n x n
   matrices A0, B with entries in {-1, 0, 1} and k rounds (about 20 s of work at the
   node's speed). The entries come from the lowbias32 hash, which numpy, JavaScript
   and WGSL compute bit for bit the same way.
2. The node computes C_r = A_r @ B and A_{r+1} = (C_r mod 3) - 1 for k rounds. Every
   entry stays an exact integer in float32 (|C| <= n), and round r+1 needs all of
   round r: nothing can be precomputed or skipped.
3. The node returns the sha256 of every row of C_k. The coordinator recomputes 8
   random rows and compares them. A node that skipped half the rows is caught with
   probability 1 - 0.5^8 = 99.6 %.
4. The time is measured by the coordinator's clock, from issuing the challenge to
   receiving the proof. Verified TFLOP/s = 2 n^3 k / seconds.
5. A node's capacity is the median of its last proofs; it counts while the newest one
   is younger than 6 h.

This measures matrix throughput, an upper bound for training. Training-effective
throughput comes from the DiLoCo telemetry once the pool trains.

## Joining

| Device | How |
| --- | --- |
| Any machine with Python | `python3 scripts/vgpu_worker.py --headcenter URL --token T --label "name" [--kind apple]` (numpy; on macOS numpy uses Accelerate and the Apple matrix units) |
| Browser (iPad/iPhone Safari 26+, Chrome, Edge) | open `URL/webgpu/vgpu` served by the headcenter, enter the token, "Beitreten" |
| Node.js + Dawn | `serve()` from `webgpu/vgpu-node.mjs` |

API (token required): `POST /api/vgpu/register`, `GET /api/vgpu/challenge?node_id=`,
`POST /api/vgpu/proof`, `GET /api/vgpu`.

The coordinator must be reachable by the nodes. The cloud container accepts no
inbound connections, so for real devices the headcenter runs on an always-on
machine of the owner (see the self-host plan) or another reachable server.

## The switch to the pool

`training/configs/longrun.json` `vgpu`: `min_b300` 2.0, `sustain_hours` 6. At every
cycle start the run reads the pool, tracks the verified capacity
(`python3 scripts/longrun.py --status`, line `vgpu`) and marks it **ready** once it
has held >= 2 B300 for 6 hours. Training then runs as DiLoCo across the pool:
`forge diloco` with the swarm rendezvous (`/api/swarm`), each node training locally
and synchronising only every H steps.

## Measured (2026-10-12)

| Node | Verified | B300 equivalents |
| --- | --- | --- |
| cloud Xeon, 4 vCPU, sharing the CPU with training | 0.049 TFLOP/s | 0.00002 |

2 B300 = 5,000 TFLOP/s. The number of devices needed follows from their verified
speed: for example 500 devices at 10 TFLOP/s each.

## Tests

- `headcenter/tests/test_vgpu.py`: exactness, rejection of skipped work, the
  coordinator's timing, expiry, API flow.
- `webgpu/test/vgpu.mjs`: the WebGPU node gives identical row digests to the Python
  coordinator (run headless with `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`).
- `scripts/tests/test_longrun.py::test_vgpu_gate_needs_sustained_verified_capacity`.
