#!/usr/bin/env python3
"""Champion arbiter: measure the candidate checkpoints of one model family the
same way and pick the true champion by one fixed rule.

Families: base (no held-out tasks), quasnir (RSI task family code), rouge (text),
darus (mixed); code, text and mixed name a task family directly.

Measurements (evaluate), cached per model sha256 in a JSON file the caller names
(layout of the RSI loop's metrics.json: {sha: {"val": {key: loss}, "heldout":
{key: result}, "dirs": [...]}}):
  val       `forge eval --split val` on every gate store with the RSI loop's gate
            flags (--batch 8, --seq min(256, ctx), --batches B, --seed S; see
            RSILoop.val_losses), deterministic for fixed B and S. The cache key
            holds the sha256 of the store's meta file and val bin.
  held-out  greedy pass rate per difficulty level on the task family's fixed
            held-out set (tasks.heldout_tasks: held-out seed space, disjoint from
            every training seed, decontaminated), measured by RSILoop.heldout_pass,
            i.e. the loop's batched `forge generate --prompts-file` with greedy
            decoding and the verify.py sandbox. Completions stay in the work dir
            (default <cache stem>.work/ next to the cache, mode 0700).
A checkpoint whose model.safetensors does not hash to its state.json
model_sha256 is refused.

Selection rule (select); every candidate must be measured on the same gate
stores and held-out levels:
  1. best_s = lowest val loss on gate store s among all candidates;
     rel_s(c) = loss_s(c) / best_s - 1, mean_rel(c) = mean of rel_s(c) over s.
  2. feasible = candidates with loss_s(c) <= best_s * (1 + tol) on every gate
     store (a missing or non-finite loss is infeasible).
  3. H(c) = mean held-out pass rate over the levels.
  4. Feasible incumbent: a feasible challenger replaces it only if
       H(c) - H(inc) >= margin, or
       |H(c) - H(inc)| < margin and mean_rel(c) <= mean_rel(inc) - 0.005.
     Of several such challengers the highest H wins (tie -> lower mean_rel);
     without one the incumbent stays.
  5. No incumbent, or an infeasible one: the feasible candidate with the highest
     H (tie -> lower mean_rel).
  6. Family base has no held-out: the same rule on mean_rel alone, i.e. a
     challenger replaces a feasible incumbent only if its mean_rel is lower by
     >= 0.005, else the lowest mean_rel among the feasible candidates.
  7. Nobody feasible (possible with two or more gate stores): the incumbent
     stays; without one the smallest worst-case rel_s wins (tie -> higher H).
The decision names the chosen candidate, ranks all candidates with every metric
(feasible first, then H, then mean_rel; hysteresis can keep an incumbent that
ranks below a challenger) and states the reasons in words.

Usage:
  python3 scripts/champions.py --family quasnir --candidate g2=runs/quasnir-g2/FINAL \\
      --candidate rsi=runs/rsi/quasnir-g2/champion --incumbent g2 \\
      --gate code=data/out/code-v2/train.meta.json --gate general=data/out/general-v2/train.meta.json \\
      [--levels 0,1,2] [--n 128] [--batches 16] --cache runs/champions/cache.json --out runs/champions/quasnir.json
Run it under `nice -n 15` next to training. Any evaluation error aborts without
a decision (exit 1), so a failed measurement can never crown a candidate.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.rsi import loop as L  # noqa: E402
from training.rsi import tasks as T  # noqa: E402
from training.rsi.store import FORGE, TOKENIZER, sha256_file  # noqa: E402

TASKS = {"base": None, "quasnir": "code", "rouge": "text", "darus": "mixed", "code": "code", "text": "text", "mixed": "mixed"}
VAL_MARGIN = 0.005
EVAL_TIMEOUT = 3600.0
EPS = 1e-9
RULE = ("feasible: val loss <= best candidate's * (1 + tol) on every gate store; a feasible challenger replaces a "
        "feasible incumbent only if its mean held-out pass rate is >= margin higher, or within margin and its mean "
        "relative val loss is >= 0.5 % lower; no or infeasible incumbent: highest held-out, tie -> lower mean "
        "relative val loss; family base: mean relative val loss alone with the same 0.5 % hysteresis")
rel = L.rel


# --------------------------------------------------------------------------- cache

def read_cache(path) -> dict:
    try:
        c = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    return c if isinstance(c, dict) else {}


def cache_put(path, sha: str, kind: str, key: str, value, ckpt: str) -> None:
    """Locked read-modify-write, replaced atomically (concurrent arbiters keep each other's entries)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_name(p.name + ".lock"), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        c = read_cache(p)
        e = c.setdefault(sha, {})
        e.setdefault(kind, {})[key] = value
        if ckpt not in e.setdefault("dirs", []):
            e["dirs"].append(ckpt)
        tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(c, indent=1))
        os.replace(tmp, p)


_STORE_SHA: dict = {}


def store_sha(meta: Path) -> str:
    """sha256 over a store's meta file and val bin (what `forge eval --split val` reads)."""
    files = [meta]
    vb = json.loads(meta.read_text()).get("val_bin")
    if vb:
        files.append(meta.parent / Path(vb).name)  # forge resolves the bin by file name next to the meta
    sig = tuple((str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files)
    if sig not in _STORE_SHA:
        _STORE_SHA[sig] = hashlib.sha256("".join(sha256_file(f) for f in files).encode()).hexdigest()
    return _STORE_SHA[sig]


# --------------------------------------------------------------------------- measurements

def checkpoint(ckpt_dir) -> L.Ckpt:
    """A forge checkpoint whose model.safetensors hashes to state.json's model_sha256."""
    d = Path(ckpt_dir).resolve()
    if not (d / "model.safetensors").is_file() or not (d / "state.json").is_file():
        raise FileNotFoundError(f"{ckpt_dir} is not a forge checkpoint (model.safetensors + state.json)")
    st = json.loads((d / "state.json").read_text())
    sha = sha256_file(d / "model.safetensors")
    if st.get("model_sha256") != sha:
        raise RuntimeError(f"{ckpt_dir}/model.safetensors has sha256 {sha}, state.json records {st.get('model_sha256')}")
    return L.Ckpt(dir=d, sha=sha, info={"step": st.get("step"), "parent_sha256": st.get("parent_sha256"),
                                        "model": st["config"]["model"]})


def forge_eval(forge, ckpt: Path, meta: Path, *, batch: int, seq: int, batches: int, seed: int, threads: int):
    """Val loss from `forge eval` with the RSI loop's gate flags (RSILoop.val_losses); None if not finite."""
    cmd = [str(forge), "eval", "--ckpt", str(ckpt), "--data", str(meta), "--split", "val", "--batch", str(batch),
           "--seq", str(seq), "--batches", str(batches), "--seed", str(seed)]
    if threads:
        cmd += ["--threads", str(threads)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=EVAL_TIMEOUT, cwd=ROOT, stdin=subprocess.DEVNULL)
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
    if res.returncode != 0 or not lines:
        raise RuntimeError(f"forge eval failed on {meta}: {res.stderr.strip()[-400:]}")
    return json.loads(lines[-1])["loss"]


def evaluate(ckpt_dir, family: str, gate_stores: dict, levels=(0, 1, 2), n_tasks: int = 128, batches: int = 16,
             seed: int = L.Settings.gate_seed, *, cache=None, workdir=None, threads: int = 0, forge=FORGE,
             tokenizer=TOKENIZER, log_stream=None) -> dict:
    """Val loss per gate store and greedy held-out pass rate per level of one checkpoint.

    Results are cached per model sha256 in `cache` (a JSON file; None disables caching)."""
    if family not in TASKS:
        raise ValueError(f"family must be one of {sorted(TASKS)}")
    if not gate_stores:
        raise ValueError("at least one gate store is required")
    if n_tasks < 1 or batches < 1:
        raise ValueError("n_tasks and batches must be >= 1")
    tf = TASKS[family]
    levels = [] if tf is None else sorted({int(lv) for lv in levels})
    if any(lv not in T.LEVELS for lv in levels):
        raise ValueError(f"levels must be in {T.LEVELS[0]}..{T.LEVELS[-1]}")
    metas = {}
    for name, meta in gate_stores.items():
        metas[name] = Path(meta).resolve()
        if not metas[name].is_file():
            raise FileNotFoundError(f"gate store {name}: {meta} not found")
    ck = checkpoint(ckpt_dir)
    ctx = int(ck.info["model"]["max_seq_len"])
    batch, seq = L.Settings.gate_batch, min(L.Settings.gate_seq, ctx)
    have = read_cache(cache).get(ck.sha, {}) if cache else {}

    def put(kind: str, key: str, value) -> None:
        if cache:
            cache_put(cache, ck.sha, kind, key, value, rel(ck.dir))

    val = {}
    for name, meta in metas.items():
        key = f"{rel(meta)}|{store_sha(meta)[:16]}|b{batch}|s{seq}|n{batches}|seed{seed}"
        if key in have.get("val", {}):
            val[name] = have["val"][key]
            continue
        val[name] = forge_eval(forge, ck.dir, meta, batch=batch, seq=seq, batches=batches, seed=seed, threads=threads)
        put("val", key, val[name])

    heldout, loop = {}, None
    for lv in levels:
        key = f"{tf}|L{lv}|n{n_tasks}|new{L.Settings.max_new_code},{L.Settings.max_new_text}|ctx{ctx}|t{L.Settings.exec_timeout}"
        if key in have.get("heldout", {}):
            heldout[str(lv)] = have["heldout"][key]
            continue
        if loop is None:
            work = Path(workdir) if workdir else (Path(cache).with_name(Path(cache).stem + ".work") if cache
                                                  else Path(tempfile.mkdtemp(prefix="champions-")))
            work.mkdir(parents=True, exist_ok=True)
            os.chmod(work, 0o700)  # completions of restricted models (rouge) stay private
            s = L.Settings(model="champions", champion=str(ck.dir), family=tf, gates={k: str(v) for k, v in metas.items()},
                           replay=None, rounds=0, out=str(work), heldout=n_tasks, threads=threads, forge=str(forge),
                           tokenizer=str(tokenizer))
            loop = L.RSILoop(s, out_stream=log_stream or sys.stderr)
            loop.model_cfg = ck.info["model"]
        r = loop.heldout_pass(ck, lv)
        heldout[str(lv)] = {k: r[k] for k in ("pass_rate", "passed", "n", "level", "decoding")}
        heldout[str(lv)]["results"] = rel(loop.out / "heldout" / f"{ck.sha[:16]}-L{lv}" / "results.jsonl")
        put("heldout", key, heldout[str(lv)])
    rates = [heldout[str(lv)]["pass_rate"] for lv in levels]
    return {"dir": rel(ck.dir), "sha256": ck.sha, "family": family, "tasks": tf, "val": val, "heldout": heldout,
            "heldout_mean": sum(rates) / len(rates) if rates else None,
            "settings": {"levels": levels, "n_tasks": n_tasks, "batch": batch, "seq": seq, "batches": batches, "seed": seed,
                         "gate_stores": {k: rel(v) for k, v in metas.items()}}}


# --------------------------------------------------------------------------- selection (pure)

def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _rate(v) -> float | None:
    return _num(v.get("pass_rate") if isinstance(v, dict) else v)


def select(candidates: list[dict], tol: float = 0.01, margin: float = 0.03, val_margin: float = VAL_MARGIN) -> dict:
    """The champion among candidates [{name, dir, metrics, incumbent}] by the rule in the module docstring."""
    if not candidates:
        raise ValueError("no candidates")
    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"candidate names must be unique: {names}")
    if sum(bool(c.get("incumbent")) for c in candidates) > 1:
        raise ValueError("at most one candidate can be the incumbent")
    store_sets = {tuple(sorted(c["metrics"].get("val") or {})) for c in candidates}
    level_sets = {tuple(sorted(c["metrics"].get("heldout") or {})) for c in candidates}
    if len(store_sets) != 1 or not next(iter(store_sets)):
        raise ValueError("every candidate must be measured on the same, non-empty set of gate stores")
    if len(level_sets) != 1:
        raise ValueError("every candidate must be measured on the same held-out levels")
    stores, levels = list(next(iter(store_sets))), list(next(iter(level_sets)))
    with_heldout = bool(levels)

    best = {}
    for s in stores:
        vals = [v for c in candidates if (v := _num(c["metrics"]["val"][s])) is not None and v > 0]
        best[s] = min(vals) if vals else None
    rows = []
    for c in candidates:
        m = c["metrics"]
        val = {s: _num(m["val"][s]) for s in stores}
        rv = {s: val[s] / best[s] - 1 if val[s] is not None and val[s] > 0 and best[s] else None for s in stores}
        why = [f"val loss on {s} missing or not finite" if rv[s] is None else
               f"val loss on {s} {val[s]:.4f} is {rv[s]:+.2%} vs the best {best[s]:.4f} (tol {tol:.2%})"
               for s in stores if rv[s] is None or rv[s] > tol + EPS]
        rates = {lv: _rate(m["heldout"][lv]) for lv in levels}
        h = sum(rates.values()) / len(levels) if with_heldout and None not in rates.values() else None
        if with_heldout and h is None:
            why.append("held-out pass rate missing")
        full = None not in rv.values()
        rows.append({"name": c["name"], "dir": c.get("dir") or m.get("dir"), "sha256": m.get("sha256"),
                     "incumbent": bool(c.get("incumbent")), "feasible": not why, "infeasible_because": why,
                     "heldout_mean": h, "heldout": rates, "val": val, "rel_val": rv,
                     "mean_rel_val": sum(rv.values()) / len(rv) if full else None,
                     "worst_rel_val": max(rv.values()) if full else None, "metrics": m})

    def rank(r: dict) -> tuple:  # smaller is better: feasible, higher held-out, lower mean relative val loss
        return (not r["feasible"], -(r["heldout_mean"] or 0.0),
                math.inf if r["mean_rel_val"] is None else r["mean_rel_val"],
                math.inf if r["worst_rel_val"] is None else r["worst_rel_val"])

    def challenge(r: dict, inc: dict) -> tuple[bool, str]:
        dv = r["mean_rel_val"] - inc["mean_rel_val"]
        vtxt = f"mean relative val loss {r['mean_rel_val']:+.2%} vs {inc['mean_rel_val']:+.2%}"
        val_better = dv <= -val_margin + EPS
        if not with_heldout:
            if val_better:
                return True, f"{r['name']} qualifies: {vtxt}, lower by >= {val_margin:.1%}"
            return False, f"{r['name']} does not replace {inc['name']}: {vtxt}, not lower by >= {val_margin:.1%}"
        dh = r["heldout_mean"] - inc["heldout_mean"]
        htxt = f"held-out {r['heldout_mean']:.3f} vs {inc['heldout_mean']:.3f} ({dh:+.3f})"
        if dh >= margin - EPS:
            return True, f"{r['name']} qualifies: {htxt}, at least the margin {margin}"
        if dh > -margin + EPS and val_better:
            return True, f"{r['name']} qualifies: {htxt} ties within the margin {margin} and {vtxt}, lower by >= {val_margin:.1%}"
        if dh > -margin + EPS:
            return False, (f"{r['name']} does not replace {inc['name']}: {htxt} ties within the margin {margin}, "
                           f"but {vtxt} is not lower by >= {val_margin:.1%}")
        return False, f"{r['name']} does not replace {inc['name']}: {htxt}, below the incumbent by at least the margin {margin}"

    feasible = [r for r in rows if r["feasible"]]
    inc = next((r for r in rows if r["incumbent"]), None)
    reasons = ["best val loss per gate store: " + ", ".join(f"{s} {best[s]:.4f}" if best[s] else f"{s} none" for s in stores)]
    reasons += [f"{r['name']} is infeasible: " + "; ".join(r["infeasible_because"]) for r in rows if not r["feasible"]]
    if not feasible:
        if inc is not None:
            chosen = inc
            reasons.append(f"no candidate is feasible on every gate store: the incumbent {inc['name']} stays")
        else:
            chosen = min(rows, key=lambda r: (math.inf if r["worst_rel_val"] is None else r["worst_rel_val"], -(r["heldout_mean"] or 0.0)))
            reasons.append(f"no candidate is feasible on every gate store and there is no incumbent: {chosen['name']} has the "
                           f"smallest worst-case relative val loss")
    elif inc is None or not inc["feasible"]:
        chosen = min(feasible, key=rank)
        what = ("the highest mean held-out pass rate (tie -> lower mean relative val loss)" if with_heldout
                else "the lowest mean relative val loss")
        lead = "no incumbent" if inc is None else f"the incumbent {inc['name']} is infeasible"
        reasons.append(f"{lead}: {chosen['name']} has {what} among the feasible candidates")
    else:
        qualified = []
        for r in sorted(feasible, key=rank):
            if r is not inc:
                ok, why = challenge(r, inc)
                reasons.append(why)
                if ok:
                    qualified.append(r)
        chosen = min(qualified, key=rank) if qualified else inc
        reasons.append(f"{chosen['name']} replaces the incumbent {inc['name']}" if chosen is not inc
                       else f"the incumbent {inc['name']} stays")
    ranking = sorted(rows, key=rank)
    for i, r in enumerate(ranking, 1):
        r["rank"], r["chosen"] = i, r is chosen
    return {"type": "champion-decision", "mode": "heldout" if with_heldout else "val-only", "rule": RULE, "tol": tol,
            "margin": margin, "val_margin": val_margin, "stores": stores, "levels": levels, "best_val": best,
            "incumbent": inc["name"] if inc else None, "chosen": {k: chosen[k] for k in ("name", "dir", "sha256")},
            "changed": inc is None or chosen is not inc, "ranking": ranking, "reasons": reasons}


# --------------------------------------------------------------------------- CLI

def _pairs(ap: argparse.ArgumentParser, items: list[str], flag: str) -> dict:
    out = {}
    for it in items:
        name, sep, val = it.partition("=")
        if not sep or not name or not val:
            ap.error(f"{flag} expects NAME=VALUE, got {it!r}")
        if name in out:
            ap.error(f"{flag} {name} given twice")
        out[name] = val
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 scripts/champions.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(TASKS), help="model family (base has no held-out tasks)")
    ap.add_argument("--candidate", action="append", required=True, metavar="NAME=DIR", help="candidate checkpoint, repeatable")
    ap.add_argument("--incumbent", default=None, metavar="NAME", help="the current champion among the candidates")
    ap.add_argument("--gate", action="append", required=True, metavar="NAME=META", help="gate store, repeatable")
    ap.add_argument("--levels", default="0,1,2", help="held-out difficulty levels")
    ap.add_argument("--n", type=int, default=128, help="held-out tasks per level")
    ap.add_argument("--batches", type=int, default=16, help="forge eval batches per gate store")
    ap.add_argument("--seed", type=int, default=L.Settings.gate_seed, help="forge eval seed")
    ap.add_argument("--threads", type=int, default=0, help="forge threads (0 = forge default)")
    ap.add_argument("--tol", type=float, default=0.01, help="feasibility: val loss <= best * (1 + tol) per gate store")
    ap.add_argument("--margin", type=float, default=0.03, help="held-out pass-rate lead a challenger needs")
    ap.add_argument("--cache", required=True, help="JSON cache of measurements per model sha256")
    ap.add_argument("--workdir", default=None, help="held-out prompts and completions (default <cache stem>.work)")
    ap.add_argument("--forge", default=str(FORGE))
    ap.add_argument("--out", required=True, help="decision JSON")
    a = ap.parse_args(argv)
    cands, gates = _pairs(ap, a.candidate, "--candidate"), _pairs(ap, a.gate, "--gate")
    if a.incumbent is not None and a.incumbent not in cands:
        ap.error(f"--incumbent {a.incumbent} is not a --candidate")
    try:
        levels = [int(x) for x in a.levels.split(",") if x.strip()]
    except ValueError:
        ap.error("--levels expects comma-separated integers")
    try:
        rows = []
        for name, d in cands.items():
            m = evaluate(d, a.family, gates, levels, a.n, a.batches, a.seed, cache=a.cache, workdir=a.workdir,
                         threads=a.threads, forge=a.forge)
            rows.append({"name": name, "dir": m["dir"], "metrics": m, "incumbent": name == a.incumbent})
        dec = select(rows, tol=a.tol, margin=a.margin)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}), file=sys.stderr)
        return 1
    dec.update(family=a.family, settings={"levels": rows[0]["metrics"]["settings"]["levels"], "n": a.n, "batches": a.batches,
                                          "seed": a.seed, "gates": gates, "cache": a.cache}, ts=time.time())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(dec, indent=2) + "\n")
    os.replace(tmp, out)
    print(json.dumps(dec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
