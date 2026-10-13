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
| release | Release ladder: Quasnir 2, Rouge 1.5, ... only at >= 2x (see below) | (in champions) |
| report | Validation losses of all models; HumanEval and MBPP pass@1 for Darus and Quasnir | 3 % |

Rouge checkpoints are served only to companies listed in `headcenter/acl.json`
(see `headcenter/README.md`). Every new directory that holds Rouge weights
(run, RSI and rebased-init directories) is created with mode 0700.

## Open-ended operation (2026-10-12)

- **Horizon.** `scripts/resume_all.sh` starts the run with `--days 100` (override
  with `LONGRUN_DAYS`). The run finishes earlier on a **plateau**: when the last
  `plateau_cycles` (4) finished cycles brought no promotion, no release and a base
  gain below `plateau_base_gain` (1 %), it stops instead of burning compute.
- **Base generations run warmup-stable-decay** (`train.base.schedule: "wsd"`).
  `forge train` writes `runs/base-g<n>/TRUNK` (weights and AdamW moments) after the
  last stable step; base-g<n+1> continues that TRUNK with `init_optim` and a 1-step
  warmup, and only the last `decay_frac` (20 %) of its steps decay to `min_lr`.
  With cosine and a fresh warmup per generation, base-g3 needed 63 % of its steps
  to get back below base-g2.
- **Champion tolerance.** The arbiter's val-loss feasibility uses `champions.tol`
  (2 %, the RSI loop's anchor tolerance). In g3 the 1 % base-gate tolerance
  discarded quasnir RSI r8 (held-out 101/384 vs 3/384 for the FINAL) for +1.3 %
  val loss. `python3 scripts/longrun.py --rearbitrate G` decides a finished cycle
  again with the current settings (run it while the long run is stopped).
- **Darus** merges the cycle's RSI champions of Rouge and Quasnir when their
  lineage leads back to the base (`forge merge --via` checks every hop: RSI rounds,
  FINAL, rebased init), otherwise the FINALs.
- **RSI** blocks a promotion that forgets a lower level (exact McNemar test on the
  paired held-out tasks, `--forget-alpha 0.05`) and excludes the first 128
  held-out tasks per level from training (`--exclude-heldout`), the set the
  arbiter scores.
- A step that ended in an error is retried once within its cycle (`step_attempts`).
- The planning throughput comes from the most recent run of the previous cycle,
  i.e. from the host the run is on now.

## Releases: version N+1 must be 2x as good

Step `release` (after the arbiter, `scripts/releases.py`) compares each family's
champion with its last major release, both measured on the same stores and
held-out tasks:

| Family | Major release (Quasnir 2, Darus 2) | Half step (Rouge 1.5) | Blocked by |
| --- | --- | --- | --- |
| quasnir, rouge, darus | odds of solving a held-out task at least 2x (95 % lower bound of the odds ratio, pooled levels) | >= sqrt(2)x | a level with a significant loss (McNemar p < 0.05); val loss > +2 % |
| base | perplexity halved on `base` (val loss -ln 2) | perplexity / sqrt(2) | other stores > +1 % |

Version 1 is the initial model (`initial`). A major release always needs 2x over
version k itself, so half steps never compound. Released checkpoints are hard
links in `runs/releases/<family>-v<version>/` (registry `runs/releases.json`),
outside checkpoint hygiene; Rouge releases are mode 0700. Out of 384 held-out
tasks, a release needs at least:

| Old version's passes | 0 | 3 | 11 | 36 | 101 |
| --- | --- | --- | --- | --- | --- |
| Major release | 11 | 18 | 38 | 88 | 185 |
| Half step | 8 | 13 | 28 | 67 | 152 |

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

### Low disk

Before every `forge train` start, the run checks that `min_free_gb` is free. If not,
it first runs the hygiene it already applies, least deletion first, and re-checks
after each stage:
1. superseded runs down to `hygiene.keep_last_superseded`;
2. rebased inits of finished runs;
3. finished runs down to `hygiene.keep_last`;
4. non-champion RSI round checkpoints of RSI steps that will not run again.

Every stage keeps FINAL and everything listed above. `runs/longrun/events.jsonl`
records what each stage removed and how much it freed (`disk hygiene (...)`). The
step fails (exit 1; `scripts/supervise.sh` retries with backoff) only when the
disk is still below `min_free_gb` after all four stages. No other kind of file is
ever deleted automatically.

### Storage report (read-only)

```
python3 scripts/longrun.py --storage-report [--json] [--include-restricted] [--root DIR]
```

This lists reclaimable space by category. For each item it shows its size, how much
deleting it would free (a file hard-linked by a release frees nothing), and whether it
is protected, with the reasons. It creates, changes and deletes nothing; the owner
decides.

| Category | What it lists |
| --- | --- |
| `hygiene` | Exactly what the low-disk hygiene above would remove (its own functions, dry run). |
| `optim` | `optim.safetensors` of kept checkpoints of finished runs. Only resuming that checkpoint reads it, or `init_optim` from the current base run's TRUNK (`optim_needed`). |
| `unreferenced` | `step-*` dirs of finished runs that no state.json, releases.json, champions.json, config, champion.json, latest.json or FINAL names. |
| `data` | `data/out`, `data/stores`, `data/cache`, `runs/longrun/stores`. The corpus in use, configured corpora (v2 is rebuilt when missing), the reference tokenizer, eval sets and the Rouge mix in use are protected. |
| `tmp` | `pytest-of-*`, humaneval `tmp*.jsonl`, `.pytest_cache`, interrupted merge-search candidates and atomic writes (protected while possibly in use). |
| `build` | `target/` (the directory with the forge binary is protected). |
| `worktrees` | `.claude/worktrees` (protected: review, then `git worktree remove`). |
| `kept` | Every protected checkpoint and why: champions, releases, FINAL targets, TRUNKs, the latest checkpoint of an unfinished run, init_from of the latest cycle's runs, the open darus merge's `--via` chain and children, current champions' lineage, and RSI champions with their history. |

Without `--include-restricted`, the report never opens, lists or measures anything of a
restricted family (`restricted_models`: `runs/rouge-*`, `runs/rsi/rouge-*`,
`runs/releases/rouge-*`, ...). It names those entries under "not inspected".

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
