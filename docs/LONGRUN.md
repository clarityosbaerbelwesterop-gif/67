# Long run: continuous training of Quasnir, Rouge 1 and Darus

`scripts/longrun.py` trains in cycles until its wall budget (`--days`) is spent,
or indefinitely with `--forever`. Every step is resumable: an interrupted
`forge train` continues from `<run>/latest.json`, and a restarted orchestrator
continues from `runs/longrun/state.json`.

## One cycle (generation g = 2, 3, …)

| Phase | What happens | Share of the cycle |
| --- | --- | --- |
| base-g | Continue pretraining the current base on corpus v2 (code + technical prose) | 55 % |
| quasnir-g | Coding model: continue from the base on code-v2, then controlled RSI on verifiable programming tasks | 14 % + 5 % |
| rouge-g | Rouge 1, the restricted model: continue from the base on a general/base mix, then controlled RSI on exact-answer tasks | 15 % + 3 % |
| darus-g | Flagship: evolutionary merge search over Rouge and Quasnir (TIES or linear), then controlled RSI on mixed tasks | 5 % |
| report | Validation losses of all models; HumanEval and MBPP pass@1 for Darus and Quasnir | 3 % |

A model replaces its predecessor only if its validation loss does not regress
by more than 1 %. Rouge checkpoints are served only to companies listed in
`headcenter/acl.json`; see `headcenter/README.md`.

## Controlled RSI (`training/rsi/`)

RSI here is verifier-gated expert iteration. Each round:
1. The model samples solutions to procedurally generated tasks.
2. Only solutions that pass sandboxed tests or exact-match checks become training data.
3. The model is fine-tuned on that data, mixed with replay.
4. A promotion gate checks the result.

Every gate must hold before a candidate is promoted:
- the held-out pass rate does not drop (seeds disjoint from training);
- no validation store regresses by more than 1 %;
- every task is decontaminated against HumanEval, MBPP and GSM8K;
- **a human approves** (iPad: Controlled RSI → Approve). With no decision
  within 6 hours the candidate is rejected and the previous champion stays.

`runs/rsi/<model>/STOP` (iPad: Stop RSI) halts the loop. Round counts, steps
and wall time are hard-limited, and every round is written to
`runs/rsi/<model>/audit.jsonl`.

Small models start at 0 % on these tasks, so the first rounds are
supervised warm-starts. These are labelled as such and never counted as
self-improvement.

## Budget on the 4-core AVX-512 host (~1,700 tokens/s measured)

| Days | Tokens trained | Base passes over corpus v2 (~400 M tokens) |
| ---: | ---: | ---: |
| 3 | ~440 M (~240 M base) | < 1 |
| 10 | ~1.5 G | ~2 |
| 20 | ~2.9 G | ~4 |

For 10–20 days, `training/configs/base-m.json` (39.6 M parameters, compute-optimal
for that budget) is the better base. It starts from scratch.

More machines add compute through `forge diloco`. Each real machine runs
`scripts/swarm_worker.py` and joins a group that the headcenter opens
(`POST /api/swarm/<group>/open`). Simulated SCP "agents" add no compute.

## Running it

```bash
python3 data/build_corpus_v2.py            # corpus v2, once (~10 min here)
python3 scripts/longrun.py --dry-run --days 3
scripts/supervise.sh --days 3              # restarts longrun after failures
touch runs/longrun/STOP                    # graceful stop (checkpoint first)
```

The cloud session container is not a durable multi-day host. It is ephemeral,
and today it restarted twice, which killed all running processes. Checkpoints
survive such restarts, and the next start resumes. For uninterrupted 10–20 day
runs, use your own machine with `deploy/forge-longrun.service` (systemd) or
`deploy/com.forge.longrun.plist` (launchd).

No result here is a promise. Every number in `runs/longrun/report-g*.json` is
measured.
