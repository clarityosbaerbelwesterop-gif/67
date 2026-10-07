# Long run: continuous training of Quasnir, Rouge 1 and Darus

`scripts/longrun.py` trains in cycles until its wall budget (`--days`) is spent,
or indefinitely with `--forever`. Every step is resumable: an interrupted
`forge train` continues from `<run>/latest.json`, and a restarted orchestrator
continues from `runs/longrun/state.json`.

## One cycle (generation g = 2, 3, …)

| Phase | What happens | Share of the cycle |
| --- | --- | --- |
| base-g | Continue pretraining the current base on corpus v2 (code + technical prose) | 47 % |
| quasnir-g | Coding model: fine-tune on code-v2 from the best init (see below), then controlled RSI on verifiable programming tasks | 12 % + 8 % |
| rouge-g | Rouge 1, the restricted model: fine-tune on a general/base mix from the best init, then controlled RSI on exact-answer tasks | 13 % + 6 % |
| darus-g | Flagship: evolutionary merge search over Rouge and Quasnir (TIES or linear), then controlled RSI on mixed tasks | 8 % |
| champions | Champion arbiter for Quasnir, Rouge and Darus (`scripts/champions.py`) | 3 % |
| report | Validation losses of all models; HumanEval and MBPP pass@1 for Darus and Quasnir | 3 % |

Rouge checkpoints are served only to companies listed in `headcenter/acl.json`
(see `headcenter/README.md`). Every new directory that holds Rouge weights
(run, RSI and rebased-init directories) is created with mode 0700.

## Promotion: one rule

Promotion is automatic for every model whenever the gates pass. Nobody has to
approve it (owner decision, 2026-10-07). `rsi.models.<model>.require_approval:
true` in `training/configs/longrun.json` turns the human approval back on for
that model. With approval on, a candidate gets no decision within
`approval_timeout_hours` (capped at half of the RSI phase) is rejected.

There is one rule, and it is applied at two levels:

- **Within an RSI run, every round** (`training/rsi/loop.py`). A candidate is
  promoted only if all of these hold:
  - its held-out pass rate rises by at least 3 points;
  - val loss on every gate store of the model is at most 1 % above the current
    RSI champion's;
  - val loss stays within `anchor_tol` (2 %) of the run's initial champion,
    which bounds the drift over many rounds;
  - every task is decontaminated against HumanEval, MBPP and GSM8K.
- **Once per cycle, the champion arbiter** (step `champions`, after the darus
  phase and before the report). For each family the candidates are:
  - the current champion (the incumbent);
  - this cycle's FINAL (for Darus, the merge);
  - this cycle's RSI champion.

  Each candidate is measured the same way: val loss on the model's gate stores,
  and greedy held-out pass rate at levels 0–2 (128 tasks each). The model's gate
  stores (`rsi.models.<model>.gates`) are code + general for Quasnir, general +
  base for Rouge, and general + code for Darus. The val loss uses
  `gate.eval`'s `forge eval` flags. The arbiter then applies
  `champions.select`:
  1. A candidate is feasible if its val loss is within 1 % of the best
     candidate's on every gate store.
  2. A feasible challenger replaces the incumbent only if its mean held-out
     pass rate is at least 3 points higher, or if it is within 3 points and its
     mean relative val loss is at least 0.5 % lower.
  3. If there is no incumbent, or the incumbent is infeasible, the feasible
     candidate with the best held-out rate wins.

  The winner becomes the current champion right away. The per-family gate steps
  (`quasnir_gate`, `rouge_gate`, `darus_gate`) only record a candidate and defer
  to the arbiter. They decide by the old rule, "val loss ≤ previous × 1.01",
  only when `champions.enabled` is false or `scripts/champions.py` cannot be
  imported.

A failed measurement never crowns a candidate:
- A challenger that cannot be measured drops out.
- An incumbent that cannot be measured stays champion.
- A checkpoint whose files are missing, or whose `model.safetensors` does not
  hash to its `state.json`, is never a candidate.

The base (`base_gate`) is a lineage, not a choice between alternatives. A new
base replaces the current one when it passes the same feasibility test (val
loss on `base-v2` within 1 % of the current base). Base has no held-out tasks,
so this is the base-family case of the rule. The newer base wins a tie, so that
pretraining keeps compounding.

Every decision, with all candidates, metrics and reasons, is appended to
`runs/longrun/champions.jsonl`. `runs/champions.json` holds the current champion
of every family with its metrics, its base, the cycle that decided it and a
timestamp. The headcenter shows it in the Long run card. The measurements are
cached per model sha256 in `runs/longrun/champions-cache.json`. Held-out
completions are kept in the 0700 work directory next to that file.

## Gains accumulate across cycles (rebase)

Each specialist is derived from a base. Its task vector is
`champion − the base it derives from`. From the second cycle on, Quasnir and
Rouge do not simply restart from the new base. For each λ in `rebase.lams`
(1.0 and 0.5), `scripts/rebase.py` builds

    runs/<model>-g<g>/rebased-init-<champion sha12>-lam<λ> = new base + λ · (current champion − its base)

The plain new base and the rebased inits are then measured on the family's
`gate.stores` (code for Quasnir, general for Rouge; val loss only by default,
held-out too with `rebase.heldout_levels`). The best one by `champions.select`
is the `init_from` of the cycle's fine-tune. So last cycle's specialist and RSI
gains are carried into the new base instead of being discarded. The arbiter
then checks that the result really is better.

The choice and its measurements are recorded in the step's `init_choice`. Each
run's base is recorded in `state.json` (`runs.<run>.base` and
`current.<model>.base`). For state written by earlier versions, the base is
derived from the run's train config (`init_from`) or from the checkpoint lineage
(`parent_sha256`). If the base cannot be determined, the plain base is used.

## Controlled RSI (`training/rsi/`)

RSI here is verifier-gated expert iteration. Each round:
1. The model samples solutions to procedurally generated tasks.
2. Only solutions that pass sandboxed tests or exact-match checks become training data.
3. The model is fine-tuned on that data, mixed with replay.
4. The promotion gate above checks the result.

`runs/rsi/<model>/STOP` (iPad: Stop RSI) halts the loop. Round counts, steps
and wall time are hard-limited, and every round is written to
`runs/rsi/<model>/audit.jsonl`.

Small models start at 0 % on these tasks, so the first rounds are supervised
warm-starts. These are labelled as such and never counted as self-improvement.

## Disk hygiene

The `hygiene` step keeps every FINAL and the last N step checkpoints of each
finished run. It never deletes any of these:
- a current champion, or the base it derives from;
- anything named in `runs/champions.json`;
- an RSI champion or its history;
- a `latest.json` target;
- a rebased init that an unfinished run still starts from.

Rebased inits of finished runs are removed.

## Config, corpus and restarts

- The config file is re-read at every phase boundary when its sha256 changes, so
  edits take effect without a restart. An edit that fails validation (shares
  must sum to 1; the model must not change) is logged and ignored, and the run
  keeps its config.
- Corpus v3 (`data/out/{base,code,general}-v3`) is adopted only between cycles,
  and only when all three stores exist and validate (dtype, vocabulary,
  tokenizer). Otherwise the run stays on v2. A restart in the middle of a cycle
  keeps the corpus that cycle started with.
- A restart (SIGTERM or `runs/longrun/STOP` → exit 75 → start again) resumes
  the cycle in progress from `state.json`, including state written by earlier
  versions. A cycle planned before the `champions` share existed gets its
  arbiter time from the configured share.

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

No result here is a promise. Every number in `runs/longrun/report-g*.json` and
`runs/champions.json` is measured.
