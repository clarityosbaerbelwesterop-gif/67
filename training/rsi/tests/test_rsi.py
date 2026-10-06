"""Tests for training/rsi (tiny scale). Outputs go under RSI_TEST_DIR
(default: the session scratchpad .../scratchpad/rsi, else a pytest tmp dir)."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.rsi import loop as L  # noqa: E402
from training.rsi import store as St  # noqa: E402
from training.rsi import tasks as T  # noqa: E402
from training.rsi import verify as V  # noqa: E402

FORGE = ROOT / "target" / "release" / "forge"
SCRATCHPAD = Path("/tmp/claude-0/-home-user/d103ac1f-f13d-5616-bc4d-c936a331fc71/scratchpad/rsi")
CODE_META = ROOT / "data" / "stores" / "code" / "train.meta.json"
GENERAL_META = ROOT / "data" / "stores" / "general" / "train.meta.json"
needs_forge = pytest.mark.skipif(not FORGE.exists() or not CODE_META.exists(), reason="forge binary or data/stores missing")
TINY_MODEL = {"vocab_size": 8192, "dim": 32, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "max_seq_len": 128}


@pytest.fixture(scope="session")
def scratch(tmp_path_factory) -> Path:
    base = Path(os.environ["RSI_TEST_DIR"]) if os.environ.get("RSI_TEST_DIR") else (
        SCRATCHPAD if SCRATCHPAD.parent.exists() else tmp_path_factory.mktemp("rsi"))
    d = base / "tests"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


# --------------------------------------------------------------------------- tasks

def test_tasks_deterministic_across_processes():
    for fam in T.FAMILIES:
        for lv in T.LEVELS:
            assert T.make_task(fam, lv, 123) == T.make_task(fam, lv, 123)
    assert T.make_task("code", 3, 1).prompt != T.make_task("code", 3, 2).prompt
    # A different hash seed in another interpreter yields byte-identical tasks.
    code = ("import json,sys; sys.path.insert(0, %r); from training.rsi import tasks as T; "
            "print(json.dumps([T.make_task(f, l, 77).to_json() for f in ('code','text','mixed') for l in range(10)]))") % str(ROOT)
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": h}).stdout for h in ("1", "999")]
    assert outs[0] == outs[1]
    assert json.loads(outs[0]) == [T.make_task(f, lv, 77).to_json() for f in ("code", "text", "mixed") for lv in range(10)]
    a, _ = T.sample_tasks("mixed", 2, 12, "round-a")
    b, _ = T.sample_tasks("mixed", 2, 12, "round-a")
    c, _ = T.sample_tasks("mixed", 2, 12, "round-b")
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert [t.task_id for t in a] != [t.task_id for t in c]
    assert {t.family for t in a} == {"code", "text"}


def test_seed_spaces_disjoint_and_heldout_excluded():
    assert T.TRAIN_SEEDS.start == 0 and T.TRAIN_SEEDS.stop <= T.HELDOUT_SEEDS.start
    assert T.space_of(T.TRAIN_SEEDS.stop - 1) == "train" and T.space_of(T.HELDOUT_SEEDS.start) == "heldout"
    with pytest.raises(ValueError):
        T.make_task("code", 0, T.HELDOUT_SEEDS.stop)
    with pytest.raises(ValueError):
        T.make_task("code", 0, -1)
    for fam in ("code", "text"):
        held, _ = T.heldout_tasks(fam, 0, 16)
        assert len(held) == 16 and all(t.space == "heldout" for t in held)
        keys = {t.key for t in held}
        train, stats = T.sample_tasks(fam, 0, 120, "disjoint", exclude=keys)
        assert all(t.space == "train" for t in train)
        assert keys.isdisjoint({t.key for t in train})
        if fam == "text":  # small parameter space at level 0: collisions occur and are dropped
            assert stats["dropped_heldout_collision"] > 0
    # held-out sets are fixed
    assert [t.task_id for t in T.heldout_tasks("text", 4, 8)[0]] == [t.task_id for t in T.heldout_tasks("text", 4, 8)[0]]


def test_decontamination_drops_overlaps(scratch):
    decon = T.default_decontaminator()
    assert set(decon.sources) == {"humaneval", "mbpp", "gsm8k"}
    he = json.loads((T.EVALS_DIR / "humaneval.jsonl").read_text().splitlines()[0])
    assert decon.contaminated("some preamble\n" + he["prompt"])
    gsm = json.loads((T.EVALS_DIR / "gsm8k.jsonl").read_text().splitlines()[5])
    assert decon.contaminated(gsm["question"])
    assert not decon.contaminated(T.make_task("code", 5, 3).text_for_decontamination())
    # A decontaminator banning one would-be-sampled task makes the sampler skip it.
    first, _ = T.sample_tasks("code", 6, 3, "decon")
    banned = T.Decontaminator.from_texts([first[0].prompt])  # (the shared assert header alone would ban every code task)
    again, stats = T.sample_tasks("code", 6, 3, "decon", decon=banned)
    assert stats["dropped_contaminated"] >= 1 and first[0].key not in {t.key for t in again} and len(again) == 3
    held, _ = T.heldout_tasks("code", 7, 2)
    held2, hstats = T.heldout_tasks("code", 7, 2, decon=T.Decontaminator.from_texts([held[0].prompt]))
    assert hstats["dropped_contaminated"] >= 1 and held[0].key not in {t.key for t in held2} and len(held2) == 2
    # The store drops contaminated documents and held-out prompts.
    if FORGE.exists():
        t = T.make_task("code", 1, 4)
        h = T.heldout_tasks("code", 1, 1)[0][0]
        docs = [{"text": t.prompt + t.reference, "source": "supervised-reference"},
                {"text": "def f():\n" + he["prompt"], "source": "self-generated"},
                {"text": h.prompt + h.reference, "source": "supervised-reference"}]
        m = St.build_store(scratch / "decon-store", docs, name="decon", replay_meta=None, replay_ratio=0.0, min_split_tokens=40,
                           forbidden=[h.prompt])
        assert m["filters"]["dropped_contaminated"] == 1 and m["filters"]["dropped_heldout"] == 1
        assert m["decontamination"]["remaining_overlaps"] == 0 and m["mix"] == {"supervised-reference": m["documents"]["train"]}


# --------------------------------------------------------------------------- sandbox and checkers

def test_sandbox_catches_wrong_timeout_crash():
    t = T.make_task("code", 0, 11)
    assert V.verify(t, t.reference)["status"] == "pass"
    bad = {
        "wrong": ("\n    return 'not it'\n", {"wrong"}),
        "timeout": ("\n    while True:\n        pass\n", {"timeout"}),
        "exception": ("\n    raise RuntimeError('boom')\n", {"error"}),
        "syntax": ("\n    return (\n", {"syntax"}),
        "exit-builtin": ("\n    exit(0)\n", {"rejected"}),
        "exit-systemexit": ("\n    raise SystemExit(0)\n", {"error"}),  # exits 0 before the marker
        "import-os": ("\n    import os\n    os._exit(0)\n", {"rejected"}),
        "dunder": ("\n    return __builtins__\n", {"rejected"}),
        "memory": ("\n    x = [0] * (10 ** 10)\n    return x\n", {"error", "crash"}),
        "output-flood": ("\n    while True:\n        print('x' * 65536)\n", {"error", "crash", "timeout"}),
        "empty": ("\ndef other():\n    pass\n", {"empty"}),
    }
    for name, (completion, want) in bad.items():
        r = V.verify(t, completion, timeout=1.5)
        assert r["status"] in want, (name, r)
    # The sandbox itself: a process killed by a signal is a crash; children are killed with the group.
    r = V.run_program("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n", timeout=3)
    assert r["status"] == "crash" and r["signal"] == 9
    r = V.run_program("import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                      "import time\ntime.sleep(60)\n", timeout=1.0)
    assert r["status"] == "timeout"
    r = V.run_program("import os\nprint(sorted(os.environ))\nopen('f', 'w').write('x' * (2 << 20))\n", timeout=3)
    assert r["status"] == "error" and "File too large" in r["reason"]


def test_reference_solutions_pass_every_family_and_level():
    n = 0
    for fam in ("code", "text", "mixed"):
        for lv in T.LEVELS:
            seeds = [0, 1, 2, T.HELDOUT_SEEDS.start + 5] if fam != "mixed" else [10, 11]
            for s in seeds:
                t = T.make_task(fam, lv, s)
                r = V.verify(t, t.reference)
                assert r["status"] == "pass", (t.task_id, r, t.prompt, t.reference)
                n += 1
                if t.family == "text":
                    assert V.verify(t, " " + t.answer + "x")["status"] == "wrong"
    assert n == 2 * 10 * 4 + 10 * 2
    t = T.make_task("text", 3, 1)
    assert t.answer == "dragon, kettle, mirror"
    assert V.verify(t, ' "Dragon ,kettle,  Mirror".\nQuestion: next')["status"] == "pass"
    assert V.verify(t, " mirror, kettle, dragon")["status"] == "wrong"


# --------------------------------------------------------------------------- store

def _base_ckpt() -> Path | None:
    p = ROOT / "runs" / "base-s" / "latest.json"
    if not p.exists():
        return None
    d = Path(json.loads(p.read_text())["dir"])
    return d if d.is_absolute() else ROOT / d


@needs_forge
def test_store_roundtrip_forge_eval(scratch):
    tasks = [T.make_task("mixed", lv, s) for lv in (0, 3, 6) for s in range(12)]
    docs = [{"text": t.prompt + t.reference, "source": "supervised-reference", "task_id": t.task_id} for t in tasks]
    out = scratch / "store"
    m = St.build_store(out, docs, name="roundtrip", replay_meta=CODE_META, replay_ratio=0.5, seed=1, min_split_tokens=200)
    meta = json.loads((out / "train.meta.json").read_text())
    assert (out / "tokenizer.json").read_bytes() == (ROOT / "data/stores/base/tokenizer.json").read_bytes()
    assert meta["dtype"] == "uint16" and meta["vocab_size"] == 8192 and meta["bin"] == "train.bin" and meta["val_bin"] == "train.val.bin"
    assert (out / "train.bin").stat().st_size == 2 * meta["tokens"] and (out / "train.val.bin").stat().st_size == 2 * meta["val_tokens"]
    for f in ("train.bin", "train.val.bin", "tokenizer.json"):
        assert m["sha256"][f] == St.sha256_file(out / f)
    assert set(meta["mix"]) == {"supervised-reference", "replay"} and abs(sum(meta["mix_fractions"].values()) - 1) < 1e-9
    assert 0.35 < m["replay_token_fraction"] < 0.65
    train = St.read_tokens(out / "train.bin")
    val = St.read_tokens(out / "train.val.bin")
    assert train[-1] == St.EOS and val[-1] == St.EOS and max(train + val) < 8192
    units = m["documents"]["train"] + m["replay"]["chunks"]["train"]
    assert train.count(St.EOS) >= units
    # Every document is stored as its own tokens followed by eos.
    samples = [json.loads(x) for x in (out / "samples.jsonl").read_text().splitlines()]
    ids = St.forge_tokenize([s["text"] for s in samples[:5]], scratch)
    flat = {"train": train, "val": val}
    for s, i in zip(samples[:5], ids):
        seq, hay = i + [St.EOS], flat[s["split"]]
        assert any(hay[j:j + len(seq)] == seq for j in range(len(hay) - len(seq) + 1)), s["task_id"]
    ck = _base_ckpt()
    if ck is None:
        pytest.skip("runs/base-s/latest.json missing")
    r = subprocess.run(["nice", "-n", "15", str(FORGE), "eval", "--ckpt", str(ck), "--data", str(out / "train.meta.json"),
                        "--batches", "1", "--seq", "64", "--threads", "2"], capture_output=True, text=True, check=True, cwd=ROOT)
    ev = json.loads(r.stdout.strip().splitlines()[-1])
    assert ev["type"] == "eval" and math.isfinite(ev["loss"]) and 0 < ev["loss"] < 20


# --------------------------------------------------------------------------- gates, approval, STOP

def _gates(**kw):
    base = dict(heldout_before=0.25, heldout_after=0.31, val_before={"general": 3.0, "code": 2.0},
                val_after={"general": 3.01, "code": 1.99}, contamination=0, lineage_ok=True, delta=0.03, tol=0.01)
    base.update(kw)
    return L.evaluate_gates(**base)


def test_gate_logic_with_injected_metrics():
    g = _gates()
    assert g["passed"] and L.decide(g, False) == "promote" and L.decide(g, True) == "pending-approval"
    assert L.decide(_gates(heldout_after=0.20), True) == "reject"  # drop
    g = _gates(heldout_after=0.26)  # no drop but below delta
    assert not g["passed"] and g["heldout_no_drop"] and not g["heldout_improved"]
    assert L.decide(_gates(heldout_after=0.26, require_improvement=False), False) == "promote"
    assert L.decide(_gates(heldout_after=0.20, require_improvement=False), False) == "reject"
    assert L.decide(_gates(heldout_after=0.28), False) == "promote"  # exactly +delta counts
    g = _gates(val_after={"general": 3.04, "code": 1.9})  # +1.33 % on general
    assert not g["passed"] and not g["val_loss"]["general"]["ok"] and g["val_loss"]["code"]["ok"]
    assert _gates(val_after={"general": 3.029, "code": 2.0})["passed"]  # +0.97 %
    assert L.decide(_gates(val_after={"general": float("nan"), "code": 2.0}), False) == "reject"
    assert L.decide(_gates(val_after={"code": 2.0}), False) == "reject"  # missing gate store
    assert L.decide(_gates(contamination=1), False) == "reject"
    assert L.decide(_gates(lineage_ok=False), False) == "reject"
    assert L.decide(_gates(heldout_after=None), False) == "reject"


def test_approval_file_and_stop(scratch):
    d = scratch / "approval"
    d.mkdir()
    ap = d / "approve-3.json"
    stop = d / "STOP"
    no_stop = lambda: stop.exists()  # noqa: E731
    assert L.wait_for_approval(ap, stop_check=no_stop, deadline=time.time() - 1, poll=0.01) == ("timeout", None)
    ap.write_text(json.dumps({"approved": True, "by": "alice"}))
    assert L.wait_for_approval(ap, stop_check=no_stop, deadline=None, poll=0.01) == ("approved", "alice")
    ap.write_text(json.dumps({"approved": False, "by": "bob"}))
    assert L.wait_for_approval(ap, stop_check=no_stop, deadline=None, poll=0.01) == ("rejected", "bob")
    for bad in ('{"approved": "yes", "by": "x"}', '{"approved": true}', '{"approved": true, "by": " "}', "not json", "[]"):
        ap.write_text(bad)
        assert "invalid" in L.read_approval(ap)
    # Malformed now, valid later: the poller logs once and keeps waiting.
    ap.write_text("{broken")
    logged = []
    threading.Timer(0.3, lambda: ap.write_text(json.dumps({"approved": True, "by": "carol"}))).start()
    assert L.wait_for_approval(ap, stop_check=no_stop, deadline=time.time() + 10, poll=0.02,
                               log=lambda e, **k: logged.append(e)) == ("approved", "carol")
    assert logged == ["approval-file-invalid"]
    ap.unlink()
    threading.Timer(0.2, stop.touch).start()
    assert L.wait_for_approval(ap, stop_check=no_stop, deadline=time.time() + 10, poll=0.02) == ("stopped", None)


# --------------------------------------------------------------------------- loop flows with stubbed measurements

def _fake_ckpt(d: Path, sha: str, parent: str | None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(b"")
    (d / "state.json").write_text(json.dumps({"model_sha256": sha, "parent_sha256": parent, "step": 1,
                                              "config": {"model": TINY_MODEL}}))
    return d


class StubLoop(L.RSILoop):
    """Real sampling/verification/store/guard/approval logic; injected forge results and metrics."""

    heldout_rates: dict = {}
    val: dict = {}
    solve_fraction = 1.0
    tampered: set = set()

    def model_file_sha(self, d):  # fake checkpoints carry symbolic shas; a "tampered" one no longer matches
        sha = json.loads((Path(d) / "state.json").read_text())["model_sha256"]
        return "tampered" if sha in self.tampered else sha

    def generate(self, ckpt, tasks, k, *, temperature, top_k, seed, workdir):
        out = []
        for i, t in enumerate(tasks):
            for j in range(k):
                good = i < int(self.solve_fraction * len(tasks))
                out.append((t, j, (t.reference if j == 0 else t.reference + "\n") if good else "\n    return None\n"))
        return out

    def heldout_pass(self, ck, level):
        return {"pass_rate": self.heldout_rates[ck.sha], "passed": 0, "n": 8, "level": level}

    def val_losses(self, ck):
        return dict(self.val[ck.sha])

    def train(self, k, meta, champ, kind, rd):
        assert meta.exists()
        d = _fake_ckpt(self.mkdir(self.out / "rounds" / f"rsi-{self.s.model}-r{k}" / "step-000003"), f"cand{k}", champ.sha)
        ck = self._ckpt(d)
        ck.lineage_ok = ck.info["parent_sha256"] == champ.sha
        return ck


def _settings(out: Path, champ: Path, **kw) -> L.Settings:
    s = L.Settings(model="stub", champion=str(champ), family="code", gates={"general": str(GENERAL_META)}, replay=str(CODE_META),
                   rounds=1, tasks=8, samples=2, steps_per_round=3, heldout=8, min_accepted=4, max_new_code=16, max_new_text=8,
                   out=str(out), poll_seconds=0.02, max_wall_hours=0.5, seq=64)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


@needs_forge
def test_loop_flows_promote_reject_pending_stop(scratch):
    import io

    champ = _fake_ckpt(scratch / "flows" / "champ", "champ0", None)

    def run(name, rates, val, **kw):
        out = scratch / "flows" / name / "rsi" / "stub"
        StubLoop.heldout_rates = rates
        StubLoop.val = val
        loop = StubLoop(_settings(out, champ, **kw), out_stream=io.StringIO())
        return loop, out

    good_val = {"champ0": {"general": 3.0}, "cand1": {"general": 2.99}}
    def when_pending(out, action):
        def wait():
            for _ in range(3000):
                if (out / "pending.json").exists():
                    return action()
                time.sleep(0.01)
        th = threading.Thread(target=wait, daemon=True)
        th.start()
        return th

    def approve(out, k, ok, by, **extra):
        return lambda: (out / f"approve-{k}.json").write_text(json.dumps({"approved": ok, "by": by, **extra}))

    # 1. approval required -> pending.json, a human approves -> promote, champion moves, self-improvement recorded.
    #    A stale approve-1.json from an aborted attempt is moved aside and does not count.
    loop, out = run("approve", {"champ0": 0.25, "cand1": 0.5}, good_val, require_approval=True)
    out.mkdir(parents=True)
    (out / "approve-1.json").write_text(json.dumps({"approved": True, "by": "stale"}))
    when_pending(out, approve(out, 1, True, "alice", candidate_sha256="cand1"))
    assert loop.run() == 0
    assert len(list(out.glob("approve-1.stale-*.json"))) == 1
    rec = json.loads((out / "audit.jsonl").read_text().splitlines()[-1])
    assert rec["decision"] == "promote" and rec["approver"] == "alice" and rec["kind"] == "self-generated" and rec["self_improvement"]
    assert rec["accepted"] == 8 and rec["documents_by_source"] == {"self-generated": 8, "supervised-reference": 0}
    cj = json.loads((out / "champion.json").read_text())
    assert cj["model_sha256"] == "cand1" and cj["round"] == 1 and cj["approver"] == "alice" and cj["history"][0]["model_sha256"] == "champ0"
    assert (out / "champion").resolve() == (out / "rounds" / "rsi-stub-r1" / "step-000003").resolve()
    assert not (out / "pending.json").exists()
    tele = [json.loads(x) for x in (scratch / "flows" / "approve" / "rsi-stub-r1.jsonl").read_text().splitlines()]
    assert tele[-1]["type"] == "event" and "round 1 (self-generated): promote" in tele[-1]["msg"]
    samples = [json.loads(x) for x in (out / "data" / "r1" / "store" / "samples.jsonl").read_text().splitlines()]
    assert {s["source"] for s in samples} == {"self-generated"} and len(samples) == 8  # duplicates per task removed
    # curriculum: 0.5 < advance_at 0.7 keeps the level
    assert rec["difficulty_next"] == 0

    # 2. human rejects -> champion unchanged.
    loop, out = run("rejected", {"champ0": 0.25, "cand1": 0.5}, good_val, require_approval=True)
    out_rejected = out
    when_pending(out, approve(out, 1, False, "bob", candidate_sha256="cand1"))
    loop.run()
    rec = json.loads((out / "audit.jsonl").read_text().splitlines()[-1])
    assert rec["decision"] == "reject" and rec["approver"] == "bob" and not rec["self_improvement"]
    assert json.loads((out / "champion.json").read_text())["model_sha256"] == "champ0"

    # 3. STOP while waiting -> pending-approval recorded, loop halts, and the request is withdrawn so that
    #    nobody can approve a candidate no loop is waiting for.
    loop, out = run("pending", {"champ0": 0.25, "cand1": 0.5}, good_val, require_approval=True, rounds=3)
    when_pending(out, lambda: (out / "approve-1.json").write_text(json.dumps({"approved": True, "by": "x", "candidate_sha256": "other"})))
    stopper = when_pending(out, lambda: (time.sleep(0.2), (out / "STOP").touch()))
    loop.run()
    stopper.join()
    lines = (out / "audit.jsonl").read_text().splitlines()
    rec = json.loads(lines[-1])
    assert len(lines) == 1 and rec["decision"] == "pending-approval" and "STOP" in rec["halt"]  # mismatched sha never approves
    assert not (out / "pending.json").exists()
    withdrawn = list(out.glob("pending-r1.withdrawn-*.json"))
    assert len(withdrawn) == 1
    pend = json.loads(withdrawn[0].read_text())
    assert pend["round"] == 1 and pend["candidate_sha256"] == "cand1" and pend["approve_file"].endswith("approve-1.json")
    # A restart finds no open request (and STOP still holds the loop); the champion is unchanged.
    loop2, _ = run("pending", {"champ0": 0.25, "cand1": 0.5}, good_val, require_approval=True)
    loop2.run()
    assert not (out / "pending.json").exists()
    assert len((out / "audit.jsonl").read_text().splitlines()) == 1

    # 3b. the candidate file changes while awaiting approval -> approval cannot promote it.
    loop, out = run("tamper", {"champ0": 0.25, "cand1": 0.5}, good_val, require_approval=True)
    when_pending(out, lambda: (StubLoop.tampered.add("cand1"), approve(out, 1, True, "dave", candidate_sha256="cand1")()))
    loop.run()
    StubLoop.tampered.clear()
    rec = json.loads((out / "audit.jsonl").read_text().splitlines()[-1])
    assert rec["decision"] == "reject" and rec["approver"] == "dave" and any("sha256 mismatch" in r for r in rec["reasons"])
    assert json.loads((out / "champion.json").read_text())["model_sha256"] == "champ0"

    # 4. STOP before the first round -> nothing runs.
    loop, out = run("stopped", {"champ0": 0.25}, good_val)
    out.mkdir(parents=True)
    (out / "STOP").touch()
    loop.run()
    assert not (out / "audit.jsonl").exists()

    # 5. gates fail (val regression) -> reject without asking anyone.
    loop, out = run("valreg", {"champ0": 0.25, "cand1": 0.5}, {"champ0": {"general": 3.0}, "cand1": {"general": 3.2}},
                    require_approval=True)
    loop.run()
    rec = json.loads((out / "audit.jsonl").read_text().splitlines()[-1])
    assert rec["decision"] == "reject" and rec["approver"] is None and not (out / "pending.json").exists()
    assert any("val loss on general" in r for r in rec["reasons"])

    # 6. no signal: warmstart never -> skip; warmstart auto -> supervised-warmstart, never self-improvement.
    StubLoop.solve_fraction = 0.0
    try:
        loop, out = run("nosignal", {"champ0": 0.0, "cand1": 0.5}, good_val, warmstart="never")
        loop.run()
        rec = json.loads((out / "audit.jsonl").read_text().splitlines()[-1])
        assert rec["kind"] == "no-signal" and rec["decision"] == "skip" and rec["data_sha256"] is None
        loop, out = run("warm", {"champ0": 0.0, "cand1": 0.75}, good_val, warmstart="auto")
        loop.run()
        rec = json.loads((out / "audit.jsonl").read_text().splitlines()[-1])
        assert rec["kind"] == "supervised-warmstart" and rec["signal"] == "no-signal" and rec["decision"] == "promote"
        assert rec["self_improvement"] is False and rec["difficulty_next"] == 1  # 0.75 >= advance_at
        samples = [json.loads(x) for x in (out / "data" / "r1" / "store" / "samples.jsonl").read_text().splitlines()]
        assert {s["source"] for s in samples} == {"supervised-reference"}
        assert json.loads((out / "champion.json").read_text())["source"] == "supervised-warmstart"
    finally:
        StubLoop.solve_fraction = 1.0

    # 7. the write guard refuses anything outside the loop's directory.
    with pytest.raises(PermissionError):
        loop.write_json(ROOT / "training" / "configs" / "evil.json", {})
    with pytest.raises(PermissionError):
        loop.guard(champ / "model.safetensors")
    rej = StubLoop(_settings(out_rejected, champ), out_stream=io.StringIO())
    assert (out_rejected / "champion").resolve() == champ.resolve()
    with pytest.raises(PermissionError):  # through the champion symlink into the (external) champion checkpoint
        rej.guard(out_rejected / "champion" / "state.json")
    with pytest.raises(PermissionError):
        rej.write_json(out_rejected / "champion", {})
    assert rej.guard(out_rejected / "champion", replace_link=True)
    with pytest.raises(PermissionError):
        loop.guard(out / "data" / ".." / ".." / ".." / "escape.json")
    assert loop.guard(out.parent.parent / "rsi-stub-r9.jsonl")
    with pytest.raises(PermissionError):
        loop.guard(out.parent.parent / "rsi-other-r1.jsonl")
    # 8. a second loop on the same directory is refused while the first holds the lock
    loop.acquire_lock()
    other = StubLoop(_settings(out, champ), out_stream=io.StringIO())
    with pytest.raises(RuntimeError):
        other.acquire_lock()
    loop.release_lock()
    other.acquire_lock()
    other.release_lock()


# --------------------------------------------------------------------------- end-to-end micro round (real forge)

@needs_forge
def test_end_to_end_micro_round(scratch):
    d = scratch / "e2e"
    tasks = [T.make_task("code", lv, s) for lv in (0, 1) for s in range(30)]
    St.build_store(d / "tiny-store", [{"text": t.prompt + t.reference, "source": "supervised-reference"} for t in tasks],
                   name="tiny", replay_meta=CODE_META, seed=2, min_split_tokens=300)
    cfg = {"run": "tiny", "model": TINY_MODEL, "data": str(d / "tiny-store" / "train.meta.json"), "batch": 4, "seq_len": 64,
           "grad_accum": 1, "max_steps": 10, "lr": 3e-3, "min_lr": 3e-4, "warmup_steps": 2, "seed": 5, "out_dir": str(d / "ckpt"),
           "log_every": 5, "eval_every": 0, "eval_batches": 2, "ckpt_every": 0, "threads": 2, "autotune": False}
    (d / "tiny.json").write_text(json.dumps(cfg))
    subprocess.run(["nice", "-n", "15", str(FORGE), "train", "--config", str(d / "tiny.json")], check=True, capture_output=True,
                   stdin=subprocess.DEVNULL, cwd=ROOT)
    champ = Path(json.loads((d / "ckpt" / "tiny" / "latest.json").read_text())["dir"])
    out = d / "runs" / "rsi" / "tiny"
    cmd = ["nice", "-n", "15", sys.executable, "-m", "training.rsi.loop", "--model", "tiny", "--champion", str(champ), "--family", "mixed",
           "--gate", f"general={GENERAL_META}", "--gate", f"code={CODE_META}", "--replay", str(CODE_META), "--rounds", "1",
           "--tasks", "4", "--samples", "2", "--steps-per-round", "3", "--heldout", "4", "--gate-batches", "1", "--gate-seq", "64",
           "--batch", "4", "--seq", "64", "--max-new-code", "16", "--max-new-text", "8", "--threads", "2", "--out", str(out),
           "--max-wall-hours", "0.25", "--exec-timeout", "3"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    events = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
    assert events[-1]["event"] == "round-end"
    lines = (out / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    for f in L.AUDIT_FIELDS:
        assert f in rec, f
    assert rec["round"] == 1 and rec["model"] == "tiny" and rec["family"] == "mixed" and rec["difficulty"] == 0
    assert rec["tasks"] == 4 and rec["samples"] == 8 and rec["kind"] in L.KINDS and rec["decision"] in ("promote", "reject")
    assert rec["kind"] == ("self-generated" if rec["accepted"] >= 8 else "supervised-warmstart")
    store = out / "data" / "r1" / "store"
    assert rec["data_sha256"] == {"train.bin": St.sha256_file(store / "train.bin"), "train.val.bin": St.sha256_file(store / "train.val.bin")}
    assert 0.0 <= rec["heldout_pass_before"] <= 1.0 and 0.0 <= rec["heldout_pass_after"] <= 1.0
    assert set(rec["val_loss_before"]) == {"general", "code"} and all(math.isfinite(v) for v in rec["val_loss_after"].values())
    assert rec["gates"]["lineage_ok"] and rec["gates"]["contamination_free"] and rec["steps"] == 3
    cand = ROOT / rec["candidate"] if not Path(rec["candidate"]).is_absolute() else Path(rec["candidate"])
    assert json.loads((cand / "state.json").read_text())["parent_sha256"] == rec["champion_sha256"]
    tele = [json.loads(x) for x in (out.parent.parent / "rsi-tiny-r1.jsonl").read_text().splitlines()]
    done = [x for x in tele if x.get("type") == "done"]
    assert len(done) == 1 and done[0]["step"] == 3
    assert tele[-1]["type"] == "event" and f"round 1 ({rec['kind']}): {rec['decision']}" in tele[-1]["msg"]
    assert json.loads((out / "champion.json").read_text())["model_sha256"] == rec["champion_sha256_after"]
    assert json.loads((out / "status.json").read_text())["phase"] == "idle"


def test_approval_must_name_the_pending_candidate(scratch):
    ap = scratch / "approve-bind.json"
    ap.write_text(json.dumps({"approved": True, "by": "alice"}))
    assert "invalid" in L.read_approval(ap, "cand1"), "an approval without the candidate hash must not count"
    ap.write_text(json.dumps({"approved": True, "by": "alice", "candidate_sha256": "other"}))
    assert "invalid" in L.read_approval(ap, "cand1")
    ap.write_text(json.dumps({"approved": True, "by": "alice", "candidate_sha256": "cand1"}))
    assert L.read_approval(ap, "cand1")["approved"] is True
    # Past the deadline an approval no longer counts, even if it is already on disk.
    assert L.wait_for_approval(ap, stop_check=lambda: False, deadline=time.time() - 1, poll=0.01, candidate_sha="cand1") == ("timeout", None)
