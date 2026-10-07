# Controlled RSI: verifier-gated expert iteration

`training/rsi` lets a model improve on tasks whose answers can be checked by a
program. Each round the champion checkpoint attempts tasks. Only verified
solutions become training data, the model is fine-tuned on them, and the new
checkpoint replaces the champion only if it passes measured gates and, when
required, a human approves it. This is STaR / expert iteration with a sandboxed
verifier. Nothing here edits code or configs, and a model never trains on its
own unverified output.

```
python3 -m training.rsi.loop --model quasnir-1 --champion runs/quasnir-1/FINAL --family code \
    --gate general=data/stores/general/train.meta.json --gate code=data/stores/code/train.meta.json \
    --replay data/stores/code/train.meta.json --rounds 5 --require-approval --max-wall-hours 24
```

Typical families: Quasnir uses `code`, Rouge 1 uses `text`, and Darus uses
`mixed`. Run the loop under `nice -n 15` next to other training jobs.

## What "controlled" means here

| control | where |
|---|---|
| **Verifier, not self-judgement.** Code runs in the sandbox against hidden asserts. The sandbox follows `scripts/humaneval.py` and uses its `limits()`: `python -I`, empty env, temp dir, RLIMIT CPU/AS/FSIZE/NPROC, a wall timeout and a process-group kill. Text answers are compared exactly after normalisation. A random marker must be printed after the asserts (so `exit(0)` cannot pass), and a static filter rejects imports outside a small allow-list, `os`/`sys`/`open`/`exec`, dunder names and `class` (reward-hacking guards). | `verify.py` |
| **Honest held-out set.** Train seeds come from [0, 1e9) and held-out seeds from [1e9, 2e9). Any training task whose prompt equals a held-out prompt at any level is dropped as well. Every task and every training document is checked for 13-gram overlap with `data/stores/evals/*.jsonl` (HumanEval, MBPP, GSM8K) and dropped on overlap. Missing eval files are an error. | `tasks.py`, `store.py` |
| **Gates** (all must pass to promote): (1) the held-out pass rate (greedy, fixed set at the current level) must not drop and must rise by at least `--delta`; (2) validation loss on every `--gate` store may regress by at most `--tol` (1 % relative by default); (3) the round store has no contamination left (each written unit is decoded and re-checked); (4) the candidate descends from the champion (`state.json` `parent_sha256`). `--promote-on no-regression` drops the delta requirement, and the audit records that choice. | `loop.evaluate_gates` |
| **Anchor guard.** Without `--require-approval` every candidate that passes the gates is promoted, so eight rounds that each stay within `--tol` could still add up to about 8 % val-loss drift. The run's anchor is its initial champion (`--champion` at round 0). Its sha256 and directory are kept in `champion.json` as `"anchor"` and survive restarts; `--reset-champion` starts a new anchor, and a `champion.json` written before anchors existed uses its latest round-0 history entry. A candidate also needs val loss ≤ anchor × (1 + `--anchor-tol`) on every gate store (2 % by default; ≤ 0 disables). The anchor's losses come from the same cached gate evals (`metrics.json`, per sha). The audit records `anchor_sha256`, `val_loss_anchor` and per store `gates.anchor` (`anchor`, `after`, `rel_change`, `ok`) plus `gates.anchor_ok`. | `loop.evaluate_gates`, `loop.RSILoop.load_champion` |
| **Human approval.** With `--require-approval`, a candidate that passes the gates is written to `pending.json`, and the loop polls for `approve-<round>.json` = `{"approved": bool, "by": "name"}` (the headcenter writes it; an optional `"candidate_sha256"` must match the pending candidate). Rejected candidates are never promoted. A malformed approval file is ignored, logged, and polling continues. An approval file that already existed before the request (from an aborted attempt) is moved aside as stale. A `pending.json` left by a loop that halted is expired on restart. | `loop.wait_for_approval` |
| **Integrity.** The champion's `model.safetensors` must hash to its recorded `model_sha256` when it is loaded. A candidate is hashed again right before promotion, so a file that changed while awaiting approval is rejected. | `loop.RSILoop.model_file_sha` |
| **STOP.** If `runs/rsi/<model>/STOP` exists, the loop stops at the next check: before each round, between phases, and while waiting for approval. | `loop.RSILoop.check` |
| **Hard limits.** `--rounds` (per invocation), `--max-steps-per-round` (caps `--steps-per-round`), `--max-wall-hours` (also bounds every forge subprocess) and `--approval-timeout-hours`. A lock file stops a second loop from running on the same directory. | `loop.py` |
| **Write guard.** Every write goes through `RSILoop.guard`. It allows only `runs/rsi/<model>/` (data, configs, checkpoints, audit) and the round telemetry `runs/rsi-<model>-r<k>.jsonl`. The champion checkpoint and other models' runs are only read. | `loop.RSILoop.guard` |
| **Replay.** Each round store holds at least the round's training token budget (steps × batch × grad_accum × (seq+1)); new documents are stored `--new-repeats` times (default 8) and replay chunks from an existing store (`--replay`, at least `--replay-ratio`) fill the rest, so fine-tuning never cycles a tiny store. Defaults lr 2e-5 and grad_accum 4 match the champions' final lr and batch. Measured on quasnir-g2, one round (0/64 held-out before): 1 repeat → 1/64, val +0.4 %; 16 repeats → 42/64, code val +1.15 % (over the 1 % gate). The previous setting (lr 1e-4, batch 8, 14.5k-token store) reached ~49/64 but regressed val loss by 8–13 %. | `store.py` |
| **Audit.** `audit.jsonl` holds one JSON line per round (fields below). `champion.json` (with history and the anchor) and the `champion` symlink name the current champion. `metrics.json` caches measured held-out and val numbers per checkpoint sha. `status.json` gives the live phase. | `loop.py` |

## Data kinds are never mixed up

* `self-generated`: model output that passed the verifier. Only a round whose
  data is self-generated and that is promoted with a measured held-out gain sets
  `"self_improvement": true` in the audit.
* `supervised-warmstart`: if fewer than `--min-accepted` candidates pass, the
  round records `"signal": "no-signal"`. With `--warmstart auto` (the default) it
  then trains on the generator's reference solutions (`supervised-reference`
  documents). Such rounds are supervised fine-tuning on synthetic data and are
  never reported as self-improvement. `--warmstart always` skips sampling and
  always uses references. `--warmstart never` records `kind: "no-signal"`,
  `decision: "skip"`, and does no training.
* `replay`: chunks of an existing licensed store.

Every store has a `manifest.json` (sha256 of every file, token mix per source,
replay provenance, decontamination counts, licences) and a `samples.jsonl` with
the exact documents and their labels.

## Honest expectation

The checkpoints in this repository are small (base-s: 12.6 M parameters, about
40 M training tokens), and they **start at or near 0 %** on these tasks.
Measured on this host on 2026-10-06 with `heldout_pass` (greedy, 32 held-out
tasks per family, level 0, max-new 160/24), checkpoint `runs/base-s/step-002500`:

| family | passed | verdicts |
|---|---|---|
| code L0 | 0 / 32 | 13 syntax, 13 error, 6 wrong |
| text L0 | 0 / 32 | 26 wrong, 6 empty |

The first useful rounds are therefore warm-start rounds (supervised on reference
solutions). Self-generated rounds only become possible once sampling yields at
least `--min-accepted` verified solutions.
Improvement is **measured, not assumed**: a candidate that does not raise the
held-out pass rate by `--delta` is rejected. The audit says which rounds were
supervised, which had no signal, and which (if any) improved on self-generated
data. These tasks are procedurally generated and in-distribution with
their own training data. Gains here say nothing about HumanEval/MBPP/GSM8K;
those stay separate benchmarks (`scripts/humaneval.py`), and they are never
trained on.

## Round, step by step

1. Held-out baseline of the champion at the current level (cached per sha).
2. Sample `--tasks` tasks from the train seed space at the current difficulty.
   Drop held-out collisions, overlaps and prompts too long for the model context.
3. `forge generate --prompts-file` with `--samples` candidates per task
   (temperature 0.8, top-k 40, a distinct seed per candidate, stop at eos 8190).
4. Verify. Keep passing candidates, deduplicated per task (at most
   `--max-per-task`) and decontaminated.
5. Choose the kind (above). Build the store `runs/rsi/<m>/data/r<k>/store`
   (SCP CorpusStore: tokenizer.json byte-identical to `data/stores/base`, uint16
   bins, eos after every document, plus replay).
6. `forge train` with `init_from` = champion and a small lr. The config goes to
   `runs/rsi/<m>/configs/r<k>.json`, checkpoints to
   `runs/rsi/<m>/rounds/rsi-<m>-r<k>/`, and the telemetry stream to
   `runs/rsi-<m>-r<k>.jsonl` (the headcenter shows it live).
7. Gates, then the decision (`promote`, `reject` or `pending-approval`). Other
   values are `skip` (no signal), `stopped` (STOP or wall limit) and `error`
   (the loop halts for a human).
8. Curriculum: once the champion's held-out pass rate at the current level
   reaches `--advance-at` (default 0.7), the next round uses level + 1. The
   difficulty resumes from the audit on restart.

Audit fields (always present): `round, model, family, difficulty, tasks,
samples, accepted, kind, signal, data_sha256, heldout_pass_before,
heldout_pass_after, val_loss_before, val_loss_after, decision, approver,
champion, candidate, self_improvement, difficulty_next, ts`. Detail fields
include `gates`, `reasons`, `anchor_sha256`, `val_loss_anchor`, `verify_status`,
`task_stats`, `data_tokens`, `documents_by_source`, `train_done` and `seconds`.

## Tasks

`tasks.make_task(family, level, seed)` is a pure function. Levels 0..9:

* **code:** 0 single integer/string operations; 1 wider operations, two-argument
  arithmetic; 2 conditionals; 3 list map/filter + reduce; 4 loops; 5 two composed
  operations; 6 small algorithms (fib, gcd, primes, ...); 7 three composed
  operations, Collatz, lcm, ...; 8 run-length, merge, rotate, binary search, ...;
  9 Kadane, brackets, transpose, look-and-say, four composed operations.
  Prompts are `def` + docstring with two doctest examples. Hidden asserts compare
  type and repr.
* **text:** arithmetic from 1-digit addition up to 3x3-digit multiplication and
  nested expressions; counting letters, vowels and even numbers; sorting
  numbers and words; reversing words; unit conversions; ordering logic; boolean
  comparisons; weekday arithmetic.
* **mixed:** even seeds give code tasks, odd seeds give text tasks.

## Files and tests

`tasks.py`, `verify.py`, `store.py` and `loop.py` hold the code. The tests in
`tests/test_rsi.py` cover: determinism and seed-space disjointness;
decontamination; sandbox verdicts (wrong, timeout, crash, syntax, exit hacks,
output floods) and that reference solutions pass for every family and level; a
store round trip through `forge eval`; the gate logic, including the anchor
guard; approval and STOP; stubbed promote/reject/pending flows; anchor drift
that compounds over rounds and across a restart; and one real micro round
with a freshly trained tiny model.

```
nice -n 15 python3 -m pytest -q training/rsi/tests/test_rsi.py
```
