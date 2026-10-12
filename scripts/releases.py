"""Release ladder: a model family gets a new version only when it is clearly better.

Owner rule (2026-10-12): version N+1 of a model must be at least twice as good as
version N; half steps (e.g. Rouge 1.5) are allowed. "Twice as good" is measured, on
the same stores and held-out task sets for both versions:

  task families (quasnir, rouge, darus): the odds of solving a held-out task at least
      double, at 95 % confidence: the one-sided lower bound of the odds ratio
      new/old over the pooled levels (Haldane +0.5 correction, Woolf standard error)
      is >= 2 for a major release, >= sqrt(2) for a half step. Near zero this means
      the pass rate doubles, near the top that the failure rate halves.
      Blocked when any level loses significantly (exact one-sided McNemar test on
      the paired per-task outcomes, p < mcnemar_alpha) or a val store regresses by
      more than val_tol.
  base: perplexity halves on the primary store (val loss -ln 2) for a major release,
      drops by sqrt(2) (-ln 2 / 2) for a half step; other stores within base_tol.

Versions are numbers: 1, 1.5, 2, 2.5, ... A candidate is always compared with the
last MAJOR release k: >= 2x -> k+1, else >= sqrt(2) and no k.5 yet -> k.5. A major
release therefore needs 2x over version k itself; half steps never compound.

Registry runs/releases.json:
  {"version": 1, "families": {fam: {"current": v, "releases": [{version, name, dir,
   sha256, cycle, ts, metrics, gate, restricted}]}}}
Every release is materialised under runs/releases/<fam>-v<version>/ (hard links of
model.safetensors and state.json, plus release.json), so checkpoint hygiene can
never remove a released model. Restricted families (rouge) get mode 0700/0600.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

Z95 = 1.6448536269514722  # one-sided 95 %
DEFAULTS = {"mcnemar_alpha": 0.05, "val_tol": 0.02, "base_tol": 0.01, "base_store": "base",
            "major": 2.0, "half": math.sqrt(2.0)}
NAMES = {"base": "Base", "quasnir": "Quasnir", "rouge": "Rouge", "darus": "Darus"}


# --------------------------------------------------------------------------- statistics

def lor_lb(a: int, b: int, n: int, z: float = Z95) -> float:
    """Lower confidence bound of the odds ratio new/old: a old passes, b new passes, of n each."""
    lo = math.log((b + .5) / (n - b + .5)) - math.log((a + .5) / (n - a + .5))
    se = math.sqrt(1 / (a + .5) + 1 / (n - a + .5) + 1 / (b + .5) + 1 / (n - b + .5))
    return math.exp(lo - z * se)


def mcnemar_loss_p(lost: int, gained: int) -> float:
    """Exact one-sided p-value that the new model is not worse, given the discordant
    pairs (lost: old passed, new failed; gained: the reverse). Small p = real loss."""
    n = lost + gained
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(lost, n + 1)) / 2 ** n


def outcomes(results: Path) -> dict:
    """{task_id: passed} from a held-out results.jsonl."""
    out = {}
    with open(results) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["task_id"]] = r.get("status") == "pass"
    return out


def paired_counts(old: dict, new: dict) -> tuple[int, int, int]:
    """(lost, gained, n_paired) over the task ids both outcome maps share."""
    ids = old.keys() & new.keys()
    lost = sum(1 for t in ids if old[t] and not new[t])
    gained = sum(1 for t in ids if new[t] and not old[t])
    return lost, gained, len(ids)


# --------------------------------------------------------------------------- the gate

def gate(family: str, ref: dict, cand: dict, root: Path | None = None, cfg: dict | None = None) -> dict:
    """ref/cand: champions.evaluate results (val {store: loss}, heldout {level: {passed,
    n, results}}). Returns {decision: major|half|none, reasons, evidence}."""
    c = {**DEFAULTS, **(cfg or {})}
    reasons, ev = [], {}
    rv, cv = ref.get("val") or {}, cand.get("val") or {}
    if set(rv) != set(cv) or not rv:
        return {"decision": "none", "reasons": ["not comparable: different val stores"], "evidence": {}}
    if family == "base":
        prim = c["base_store"] if c["base_store"] in rv else sorted(rv)[0]
        delta = cv[prim] - rv[prim]
        ev = {"store": prim, "delta_loss": delta, "ppl_ratio": math.exp(-delta)}
        for s in rv:
            if s != prim and cv[s] > rv[s] * (1 + c["base_tol"]):
                reasons.append(f"{s} val {cv[s]:.4f} > {rv[s]:.4f} * (1 + {c['base_tol']})")
        if reasons:
            return {"decision": "none", "reasons": reasons, "evidence": ev}
        dec = "major" if delta <= -math.log(c["major"]) else "half" if delta <= -math.log(c["half"]) else "none"
        return {"decision": dec, "reasons": [f"perplexity x{math.exp(-delta):.3f} on {prim}"], "evidence": ev}
    rh, ch = ref.get("heldout") or {}, cand.get("heldout") or {}
    if set(rh) != set(ch) or not rh or any(rh[k].get("n") != ch[k].get("n") for k in rh):
        return {"decision": "none", "reasons": ["not comparable: different held-out levels or sizes"], "evidence": {}}
    for s in rv:
        if cv[s] > rv[s] * (1 + c["val_tol"]):
            reasons.append(f"{s} val {cv[s]:.4f} > {rv[s]:.4f} * (1 + {c['val_tol']})")
    per_level = {}
    for lv in sorted(rh, key=int):
        rec = {"old": rh[lv].get("passed"), "new": ch[lv].get("passed"), "n": rh[lv].get("n")}
        paths = [(rh[lv].get("results")), (ch[lv].get("results"))]
        if root is not None and all(paths) and all((Path(root) / p).exists() for p in paths):
            lost, gained, n = paired_counts(outcomes(Path(root) / paths[0]), outcomes(Path(root) / paths[1]))
            p = mcnemar_loss_p(lost, gained)
            rec.update(lost=lost, gained=gained, paired=n, p_loss=p)
            if p < c["mcnemar_alpha"]:
                reasons.append(f"level {lv}: significant loss ({lost} lost, {gained} gained, p={p:.3g})")
        per_level[lv] = rec
    a = sum(int(rh[k]["passed"]) for k in rh)
    b = sum(int(ch[k]["passed"]) for k in ch)
    n = sum(int(rh[k]["n"]) for k in rh)
    lb = lor_lb(a, b, n)
    ev = {"old_passed": a, "new_passed": b, "n": n, "odds_ratio_lb95": lb, "levels": per_level}
    if reasons:
        return {"decision": "none", "reasons": reasons, "evidence": ev}
    dec = "major" if lb >= c["major"] else "half" if lb >= c["half"] else "none"
    return {"decision": dec, "reasons": [f"held-out {a} -> {b} of {n}, odds ratio >= {lb:.2f} (95 %)"], "evidence": ev}


# --------------------------------------------------------------------------- versions

def fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def display_name(family: str, version: float) -> str:
    return f"{NAMES.get(family, family.capitalize())} {fmt(version)}"


def last_major(fam_reg: dict) -> dict | None:
    majors = [r for r in fam_reg.get("releases", []) if float(r["version"]).is_integer()]
    return max(majors, key=lambda r: r["version"]) if majors else None


def next_version(fam_reg: dict, decision: str) -> float | None:
    """The version a candidate gets from its gate decision against the last major release."""
    lm = last_major(fam_reg)
    if lm is None or decision == "none":
        return None
    k = float(lm["version"])
    if decision == "major":
        return k + 1
    have = {float(r["version"]) for r in fam_reg.get("releases", [])}
    return k + 0.5 if decision == "half" and (k + 0.5) not in have else None


def choose_levels(ref: dict, levels: list, advance_at: float = 0.7, width: int = 3, max_level: int = 5) -> list:
    """Once the reference solves >= advance_at of the pooled held-out tasks, measure both
    versions one window of `width` levels higher (never across windows)."""
    h = ref.get("heldout") or {}
    n = sum(int(v.get("n", 0)) for v in h.values())
    p = sum(int(v.get("passed", 0)) for v in h.values()) / n if n else 0.0
    if p >= advance_at and max(levels) + width <= max_level:
        return [lv + width for lv in levels]
    return list(levels)


# --------------------------------------------------------------------------- registry

def load_registry(path: Path) -> dict:
    try:
        reg = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        reg = {}
    reg.setdefault("version", 1)
    reg.setdefault("families", {})
    return reg


def _write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_release(root: Path, registry_path: Path, family: str, version: float, src_dir: str, sha256: str,
                   metrics: dict, gate_result: dict | None, cycle: int | None, restricted: bool = False) -> dict:
    """Materialise runs/releases/<family>-v<version>/ and add it to the registry
    (idempotent: an existing release of that version is returned unchanged)."""
    root = Path(root)
    reg = load_registry(registry_path)
    fam = reg["families"].setdefault(family, {"current": None, "releases": []})
    for r in fam["releases"]:
        if float(r["version"]) == float(version):
            return r
    out = root / "runs" / "releases" / f"{family}-v{fmt(version)}"
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    src = root / src_dir
    for name in ("model.safetensors", "state.json"):
        try:
            os.link(src / name, tmp / name)
        except OSError:
            shutil.copy2(src / name, tmp / name)
    if _sha256(tmp / "model.safetensors") != sha256:
        shutil.rmtree(tmp)
        raise ValueError(f"{src_dir}: model.safetensors does not hash to {sha256}")
    rec = {"version": float(version), "name": display_name(family, version), "dir": str(out.relative_to(root)),
           "source": src_dir, "sha256": sha256, "cycle": cycle, "ts": time.time(),
           "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "metrics": metrics,
           "gate": gate_result, "restricted": restricted}
    _write_json(tmp / "release.json", rec)
    if restricted:
        for f in tmp.iterdir():
            os.chmod(f, 0o600)
        os.chmod(tmp, 0o700)
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    fam["releases"].append(rec)
    fam["current"] = max(float(r["version"]) for r in fam["releases"])
    _write_json(registry_path, reg)
    return rec
