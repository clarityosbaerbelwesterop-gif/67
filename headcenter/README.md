# headcenter — iPad mission control

A single small process next to the trainer. It reads what `forge` writes
(`runs/*.jsonl`), runs two agents over it, and sends commands back through
each run's control FIFO (`runs/<run>.ctl`, created by `scripts/train_all.sh`).
It never touches the compute path, so it costs the training run no FLOPs.

```
iPad Safari ──HTTPS/WSS──► headcenter (FastAPI) ──reads──► runs/<run>.jsonl  ◄── forge train
                                  │                                              ▲
                                  └──── validated commands ──► runs/<run>.ctl ───┘
```

## Run

```bash
pip install -r headcenter/requirements.txt
python -m headcenter.backend                                  # loopback only, no token needed
python -m headcenter.backend --host 0.0.0.0 --new-token       # LAN/iPad: prints http://…/#token=…
```

Binding beyond loopback refuses to start without a token (`--token`,
`HEADCENTER_TOKEN` or `--new-token`); the API can pause, stop and retune
runs. For access from outside the LAN, put it behind a TLS tunnel or reverse
proxy and open `https://host/#token=…`. The page stores the token in the
browser and removes it from the address bar.

## What the iPad shows and does

- **Run state:** step and progress, ETA, train and validation loss, tokens/s,
  measured TFLOPS and its share of one B300 (2,500 TFLOPS dense BF16), the
  kernels in use.
- **Loss chart:** pinch to zoom, drag to pan, double-tap to reset.
- **Profiling and memory:** step time by op with achieved GFLOP/s, and HBVM
  memory (used, peak, capacity, fragmentation).
- **Controls:** pause/resume, checkpoint, autotune, LR peak, threads and a
  kernel swap. Stop needs a 1.5 s press.
- **Run on this iPad (WebGPU):** opens `/webgpu/` with the run's FINAL (else
  newest) checkpoint and the tokenizer prefilled, then generates text on the
  iPad's own GPU.

## Agents

Each agent works in one of three modes, switchable live from the iPad:

- `act` (the default): actions are sent to the run.
- `advise`: decisions are shown and logged, but nothing is sent.
- `off`: the agent is not consulted.

Every decision, executed or not, is appended to
`runs/headcenter/decisions.jsonl`. Conditions are edge-triggered. History
read at start-up only primes the agents, so an old incident is never acted on
again.

**silicon-watchdog** watches numerics, divergence, memory and liveness:

| Condition | Response |
| --- | --- |
| Non-finite loss | `pause` (critical); the newest checkpoint stays intact |
| Loss above 1.5× its EMA for 2 consecutive lines | Halve the LR peak (`set_lr`), at most once per 200 steps |
| Gradient norm above 20× its recent median | Warning |
| HBVM peak above 97 % of capacity | Warning |
| HBVM fragmentation above 0.5 | Info |
| No telemetry for max(180 s, 6× the usual interval) | Warning |

**jit-optimizer** watches throughput:

| Condition | Response |
| --- | --- |
| MATMUL GFLOP/s below 75 % of the run's own baseline for 3 windows | `autotune`: the trainer re-measures every GEMM variant per shape on the live machine and hot-swaps the faster kernels in without stopping. At most once per 30 min, then the baseline is re-learnt. |
| Every 10 min | Reports where the step time goes, and flags non-GEMM ops as fusion candidates when they exceed 30 % |

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | The iPad UI. |
| `GET /webgpu/` | The WebGPU runtime. |
| `GET /models/{token}/{run}/{file}` | Checkpoint files (`model.safetensors`, `state.json`, `tokenizer.json`). The token is in the path because the runtime fetches from a base URL. Only files inside `runs/` are served. |
| `GET /api/health` | Health check, no auth. |
| `GET /api/runs` | Snapshot of runs, agent modes and recent decisions. |
| `GET /api/models` | Runs that have a servable checkpoint. |
| `POST /api/runs/{run}/cmd` | Send a trainer command (`docs/DESIGN.md`, "Commands"). |
| `POST /api/agents/{agent}` | Set an agent's mode, e.g. `{"mode":"advise"}`. |
| `WS /ws?token=…` | Sends the snapshot, then every record and decision live. Accepts `{"type":"cmd",…}` and `{"type":"mode",…}`, and acknowledges each with an `ack` carrying the same `id`. |

Commands are validated against a whitelist and written in canonical form as a
single atomic line. A run that is not alive answers `409`.

## Tests

`python -m pytest -q headcenter/tests`. CI (`.github/workflows/deploy-headcenter.yml`)
additionally runs the headcenter against a live smoke training run. A manual
dispatch with `pages: true` publishes the UI and the WebGPU runtime to GitHub
Pages. The UI then connects to a backend given as `#api=wss://host&token=…`.
