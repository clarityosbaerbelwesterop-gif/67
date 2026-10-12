# Forge cluster: training as a workflow

The cluster turns any set of machines into one training job. The machines only
exchange files: no network between workers and no inbound ports. GitHub Actions is
the scheduler and artifact store.

```
prepare ──► work × W (parallel) ──► aggregate ──► (next round)
 build forge   download own shard     outer Nesterov step
 load/create   train H steps          eval, cluster-state artifact
 global model  upload delta (fp16)    dispatch the next round (self-hosted)
```

| Piece | File |
| --- | --- |
| Round workflow | `.github/workflows/forge-cluster.yml` (on `main`; runs the code of `ref`) |
| DiLoCo round: init (also from scratch), work, aggregate, eval | `cluster/fabric.py` |
| Per-worker data shard from pinned public packages | `cluster/shard.py` |
| Tokenizer (shared by every store) | `data/tokenizer.json` |

## Data: every worker loads its own share

Worker i of W takes every W-th package of `data/manifests/v3/sources.json`, a list of
3,884 licence-checked packages with archive URL and sha256. An archive whose hash
changed since the manifest was built is skipped and never trained on. Every
document then passes:

- the corpus quality rules: v2 rules, the generated-code marker and the v4 high-quality filter;
- exact dedup;
- 13-gram decontamination against HumanEval, MBPP and GSM8K.

About 1 % goes to val. No secret and no private data ever reaches a runner. Each
shard is cached per shard, so it is built once.

## Runners

| `runner` | What it is | Limits |
| --- | --- | --- |
| `ubuntu-latest` | GitHub-hosted, 4 vCPU each | A bounded test of the software: at most 4 workers, 200 steps, no chaining. GitHub's terms do not allow a training farm on hosted runners, and an account suspension would take every repository with it. |
| `forge` | The owner's self-hosted runners, label `forge` | Real training. With `continue` set, every round dispatches the next, bounded by `rounds`. The cluster then runs without any Claude session. |

To add a machine, any Linux, macOS or Windows box or rented GPU server:

1. Go to repository Settings → Actions → Runners → New self-hosted runner, and run the
   three commands shown there.
2. Add the label `forge`.
3. Make sure the machine has Python 3 with numpy. The workflow installs Rust itself.

The repository is public, so keep these settings:

- Only this workflow uses `forge`.
- It runs only on `workflow_dispatch`, which forks cannot trigger.
- Under Settings → Actions → General, require approval for outside contributors.

## Running it

From the Actions tab, run "Forge cluster round (DiLoCo)":

| Goal | Inputs |
| --- | --- |
| New model from scratch | `from_run` empty, `model` `base-s.json` |
| Continue a model | `from_run` = the run id of the last round; its `cluster-state` artifact is the model, momentum and history |
| Endless training on self-hosted runners | `runner` `forge`, `continue` true, `rounds` e.g. 1000 |

Each round's summary shows workers, tokens, the slowest worker, cluster tok/s and
val loss.

## What it buys

DiLoCo at H = 100 needs about 2.5× the tokens of synchronous training for the same
quality. This was measured on `gpu/fabric-v0`. So W workers give about W / 2.5 of
one machine's progress, with H times less communication.

The cluster's speed is the sum of its machines, nothing more. Its capacity is
verified by the VGPU pool (`docs/VGPU.md`) in B300 equivalents. Once the pool
holds 2 B300 for 6 hours, the long run moves base training onto it.

## Measured

| Run | Workers | Steps | tok/s per worker | Val loss |
| --- | --- | --- | --- | --- |
| Local smoke from scratch (cloud Xeon, shared with training) | 2 | 8 | ~850 | 9.01 → 7.93 after one round |
| GitHub run 38264700716, `ubuntu-latest`, from scratch (2026-10-12) | 4 | 100 (8,192 tok/step) | ~1,800 | 9.01 → 5.49 after one round (3.28 M tokens) |

In that run, the whole round took 10 minutes:

| Phase | Time |
| --- | --- |
| Build | 35 s (cached) |
| Data shard per worker | about 45 s |
| Training | about 7.5 min |
| Aggregate and eval | 30 s |

One hosted runner is about as fast as the cloud container (~2,000 tok/s). Four
runners give about 7,400 tok/s raw, or about 3,000 tok/s training-effective after
the DiLoCo factor 2.5.
