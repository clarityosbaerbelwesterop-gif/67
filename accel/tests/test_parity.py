"""forge <-> accel parity.

  a) logits and eval of the base checkpoints (runs/base-g3/FINAL, runs/base-s/FINAL)
  b) training: bit-exact init and file bytes; a tiny fresh run (cosine, clipping, accumulation);
     wsd + TRUNK; one-window store from init_from with and without init_optim (the batch is
     deterministic: N = seq + 2 tokens => forge's below(N - seq - 1) = below(1) = 0)
  c) round trips: accel checkpoints read by forge eval / forge logits / forge train --resume, and
     forge checkpoints re-written by accel byte for byte
  plus data parallelism over 2 (forced) CPU devices (one all-reduce per update), a simulated
  2-host run (stdin commands reach every process), the bf16 path, accel resume, AdamW bit for bit,
  out-of-range token ids, FINAL, and cluster/fabric.py driven through accel/bin/forge-accel.

Tolerances (why they are what they are) are next to each assertion; measured values are printed
with a [parity] tag (pytest -s) and collected into $ACCEL_PARITY_REPORT.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import (FORGE, NICE, PY, ROOT, accel, accel_json, child_env, forge, forge_json, lines, need_ckpt,
                      record, run)

from accel import forge_io as fio
from accel.config import ModelConfig, TrainConfig
from accel.model import init_params

TINY = {"vocab_size": 64, "dim": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "max_seq_len": 32}


# ---------------------------------------------------------------------------------------------- helpers
def tiny_cfg(run_name: str, out_dir: Path, **kw) -> dict:
    c = {"run": run_name, "model": TINY, "data": "synthetic:17", "batch": 2, "seq_len": 16, "grad_accum": 2,
         "max_steps": 4, "lr": 1e-2, "min_lr": 1e-3, "warmup_steps": 2, "weight_decay": 0.1, "grad_clip": 0.5,
         "seed": 5, "out_dir": str(out_dir), "log_every": 1, "eval_every": 0, "eval_batches": 2, "ckpt_every": 1,
         "threads": 1, "autotune": False}
    c.update(kw)
    return c


def train_both(cfg: dict, tmp: Path, accel_args=("--xent", "forge"), env=None) -> tuple[Path, Path, list, list]:
    """Run forge train and accel train on the same config (different out_dirs). Returns
    (forge run dir, accel run dir, forge lines, accel lines)."""
    fcfg = dict(cfg, out_dir=str(tmp / "forge"))
    acfg = dict(cfg, out_dir=str(tmp / "accel"))
    (tmp / "f.json").write_text(json.dumps(fcfg))
    (tmp / "a.json").write_text(json.dumps(acfg))
    fl = lines(forge("train", "--config", tmp / "f.json"))
    al = lines(run([PY, "-m", "accel.train", "--config", tmp / "a.json", *accel_args], env=env))
    return tmp / "forge" / cfg["run"], tmp / "accel" / cfg["run"], fl, al


def by_type(ls: list, t: str) -> list:
    return [x for x in ls if x.get("type") == t]


def tel(ls: list) -> dict:
    return {x["step"]: x for x in by_type(ls, "telemetry")}


def ckpt_diff(fd: Path, ad: Path) -> dict:
    fw, _, _ = fio.read_safetensors(fd / "model.safetensors")
    aw, _, _ = fio.read_safetensors(ad / "model.safetensors")
    fo, _, _ = fio.read_safetensors(fd / "optim.safetensors")
    ao, _, _ = fio.read_safetensors(ad / "optim.safetensors")
    assert sorted(fw) == sorted(aw) and sorted(fo) == sorted(ao)
    dw = {k: np.abs(fw[k].astype(np.float64) - aw[k]) for k in fw}
    allw = np.concatenate([d.ravel() for d in dw.values()])

    def rel(k):  # max |diff| relative to the tensor's max |value|
        return float(np.abs(fo[k].astype(np.float64) - ao[k]).max() / max(float(np.abs(fo[k]).max()), 1e-30))

    return {"w_max_abs": float(allw.max()), "w_median_abs": float(np.median(allw)),
            "w_n_over_1e-6": int((allw > 1e-6).sum()), "w_n_over_1e-5": int((allw > 1e-5).sum()), "w_n": int(allw.size),
            "m_max_rel": max(rel(k) for k in fo if not k.startswith("v.")),
            "v_max_rel": max(rel(k) for k in fo if k.startswith("v."))}


def state_without(d: Path, *drop) -> dict:
    s = json.loads((d / "state.json").read_text())
    s["config"] = {k: v for k, v in s["config"].items() if k not in drop}
    return s


def one_window_store(d: Path, vocab: int, seq: int, seed: int = 0) -> Path:
    """A store whose train and val bins hold exactly seq + 2 tokens: one possible window each."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in ("train.bin", "val.bin"):
        rng.integers(0, vocab, seq + 2, dtype=np.uint16).tofile(d / name)
    (d / "train.meta.json").write_text(json.dumps({"tokens": seq + 2, "val_tokens": seq + 2, "dtype": "uint16",
                                                   "vocab_size": vocab, "bin": "train.bin", "val_bin": "val.bin"}))
    return d / "train.meta.json"


def logits_diff(ckpt: Path, ids: list, ctx: int) -> dict:
    f = np.asarray(forge_json("logits", "--ckpt", ckpt, "--prompt-ids", ",".join(map(str, ids)), "--ctx", str(ctx))["logits"],
                   np.float32)
    a = np.asarray(accel_json("logits", "--ckpt", str(ckpt), "--prompt-ids", ",".join(map(str, ids)), "--ctx", str(ctx))["logits"],
                   np.float32)
    return {"max_abs_diff": float(np.abs(f - a).max()), "max_abs_logit": float(np.abs(f).max()),
            "argmax_forge": int(f.argmax()), "argmax_accel": int(a.argmax())}


# ---------------------------------------------------------------------------------------------- a) forward / eval
LOGIT_CASES = [("runs/base-g3/FINAL", [1, 200, 3000, 45, 8000, 17, 99, 512, 7], 16),
               ("runs/base-g3/FINAL", list(range(5, 15)), 512),
               ("runs/base-g3/FINAL", [int(x) for x in np.random.RandomState(11).randint(0, 8192, 300)], 256),
               ("runs/base-s/FINAL", [int(x) for x in np.random.RandomState(3).randint(0, 8192, 40)], 64)]


@pytest.mark.parametrize("ckpt,ids,ctx", LOGIT_CASES, ids=lambda x: None)
def test_logits_parity(forge_bin, ckpt, ids, ctx):
    p = need_ckpt(ckpt)
    r = logits_diff(p, ids, ctx)
    record(f"logits {ckpt} n={len(ids)} ctx={ctx}", r)
    # Both sides compute in f32; they differ only in GEMM summation order (forge's avx512 blocking with
    # FMA vs XLA's Eigen kernels) and libm vs XLA exp/rsqrt. Over 6 layers with K <= 1024 that is
    # ~1e-5 at |logit| ~ 8 (measured <= 3.1e-5 by the independent reference). 1e-4 = ~12 ppm of the
    # logit range; any real mismatch (RoPE pairing, GQA grouping, eps, scaling) moves logits by >= 1e-2.
    assert r["max_abs_diff"] <= 1e-4
    assert r["argmax_forge"] == r["argmax_accel"]


EVAL_CASES = [("runs/base-g3/FINAL", "data/stores/base/train.meta.json", "val", 2, 64, 3, 4242),
              ("runs/base-g3/FINAL", "data/out/base-v2/train.meta.json", "val", 2, 64, 3, 4242),
              ("runs/base-s/FINAL", "data/stores/base/train.meta.json", "train", 3, 48, 2, 9)]


@pytest.mark.parametrize("ckpt,data,split,b,t,k,seed", EVAL_CASES, ids=lambda x: None)
def test_eval_parity(forge_bin, ckpt, data, split, b, t, k, seed):
    need_ckpt(ckpt)
    if not (ROOT / data).exists():
        pytest.skip(f"{data} not available")
    args = ["--ckpt", ckpt, "--data", data, "--split", split, "--batch", str(b), "--seq", str(t), "--batches", str(k),
            "--seed", str(seed)]
    f = forge_json("eval", *args)
    a = accel_json("eval", *args)
    record(f"eval {ckpt} {data} {split} b{b} t{t} x{k} seed{seed}",
           {"forge": [f["loss"], f["next_token_acc"]], "accel": [a["loss"], a["next_token_acc"]],
            "loss_diff": abs(f["loss"] - a["loss"])})
    for key in ("ckpt", "data", "sha256", "split", "tokens", "quant", "weight_bytes", "type"):
        assert f[key] == a[key], key
    # Same windows (forge's Rng and sampler) and forge's sequential softmax sum: what is left is the
    # logit-level f32 noise above, which averages out over the rows (measured: identical f32 values).
    # With a tree-summed denominator (--xent fast) the gap is 1.1e-5 (forge's sum runs ~4e-6 low).
    assert abs(f["loss"] - a["loss"]) <= 2e-6
    assert abs(f["next_token_acc"] - a["next_token_acc"]) <= 1.0 / (b * t * k) + 1e-9  # at most one near-tie flips
    # perplexity = f32 loss.exp(), glibc expf on both sides (numpy's f32 exp is 1 ulp off for ~40 % of inputs)
    assert f["perplexity"] == float(fio.expf(f["loss"])) and a["perplexity"] == float(fio.expf(a["loss"]))
    if f["loss"] == a["loss"]:
        assert f["perplexity"] == a["perplexity"]


# ---------------------------------------------------------------------------------------------- b) training
def test_init_bitwise_and_files(forge_bin, tmp_path):
    """lr = 0: the step-1 weights are the initialisation. forge's and accel's model.safetensors must be
    the same bytes (bit-exact init + identical file format), state.json identical up to out_dir."""
    cfg = tiny_cfg("t-init", tmp_path, lr=0.0, min_lr=0.0, max_steps=1)
    fd, ad, fl, al = train_both(cfg, tmp_path)
    fsha = fio.sha256_file(fd / "step-000001" / "model.safetensors")
    asha = fio.sha256_file(ad / "step-000001" / "model.safetensors")
    ref = init_params(ModelConfig.from_dict(TINY), 5, exact=True)
    w, _, _ = fio.read_safetensors(ad / "step-000001" / "model.safetensors")
    n_diff = sum(int((w[k] != ref[k]).sum()) for k in ref)
    ft, at = tel(fl)[1], tel(al)[1]
    fe, ae = by_type(fl, "eval")[-1], by_type(al, "eval")[-1]
    record("init tiny lr=0", {"model_sha_equal": fsha == asha, "init_elements_differing": n_diff,
                              "loss": [ft["loss"], at["loss"]], "grad_norm": [ft["grad_norm"], at["grad_norm"]],
                              "val_loss": [fe["val_loss"], ae["val_loss"]], "val_acc": [fe["val_acc"], ae["val_acc"]]})
    assert n_diff == 0
    assert fsha == asha
    sf, sa = state_without(fd / "step-000001", "out_dir"), state_without(ad / "step-000001", "out_dir")
    assert abs(sf.pop("loss") - sa.pop("loss")) <= 1e-6  # the train loss of step 1 (f32 noise, see below)
    assert sf == sa  # config, rng_state, step, tokens, val_loss, lineage, model_sha256
    # accel's JSON writer reproduces serde_json byte for byte (sorted keys, ryu floats, 2-space pretty
    # state.json without trailing newline, compact latest.json)
    for name, pretty in (("step-000001/state.json", True), ("latest.json", False)):
        text = (fd / name).read_text()
        assert fio.dumps(json.loads(text), pretty=pretty) == text, name
    lf = json.loads((fd / "latest.json").read_text())
    la = json.loads((ad / "latest.json").read_text())
    assert lf["model_sha256"] == la["model_sha256"] and lf["step"] == la["step"]
    assert abs(ft["loss"] - at["loss"]) <= 1e-6
    assert abs(fe["val_loss"] - ae["val_loss"]) <= 1e-6 and fe["val_acc"] == ae["val_acc"]
    # the files forge writes and nothing else: forge train never writes FINAL (scripts/longrun.py links it)
    assert sorted(os.listdir(fd)) == sorted(os.listdir(ad)) == ["latest.json", "step-000001"]


def test_tiny_training_parity(forge_bin, tmp_path):
    """Fresh init, 4 cosine steps, accum 2, warmup 2, clipping active (clip 0.5 < grad norm)."""
    cfg = tiny_cfg("t-cos", tmp_path)
    fd, ad, fl, al = train_both(cfg, tmp_path)
    ft, at = tel(fl), tel(al)
    rows = []
    for s in range(1, 5):
        sf = json.loads((fd / f"step-{s:06d}" / "state.json").read_text())
        sa = json.loads((ad / f"step-{s:06d}" / "state.json").read_text())
        rows.append({"step": s, "lr": [ft[s]["lr"], at[s]["lr"]], "loss": [ft[s]["loss"], at[s]["loss"]],
                     "grad_norm": [ft[s]["grad_norm"], at[s]["grad_norm"]], "rng_equal": sf["rng_state"] == sa["rng_state"],
                     "tokens": [sf["tokens"], sa["tokens"]]})
    d = ckpt_diff(fd / "step-000004", ad / "step-000004")
    fe, ae = by_type(fl, "eval")[-1], by_type(al, "eval")[-1]
    record("tiny cosine 4 steps", {"steps": rows, "final": d, "eval": [[fe["val_loss"], fe["val_acc"]],
                                                                       [ae["val_loss"], ae["val_acc"]]]})
    for r in rows:
        assert np.float32(r["lr"][0]) == np.float32(r["lr"][1])  # LR schedule is bit-exact (f32, libm cosf)
        assert abs(r["loss"][0] - r["loss"][1]) <= 2e-6  # measured <= 4.8e-7 (a few f32 ulps of ~4)
        assert abs(r["grad_norm"][0] - r["grad_norm"][1]) <= 1e-5 * r["grad_norm"][0]
        assert r["rng_equal"] and r["tokens"][0] == r["tokens"][1]
    # Fresh moments make the first AdamW steps ~lr*sign(g); f32 noise on |g| ~ 0 can then move a weight by
    # up to ~lr. On this tiny model the measured max is ~1e-6; allow 1e-4 (1 % of lr).
    assert d["w_max_abs"] <= 1e-4
    assert d["m_max_rel"] <= 1e-4 and d["v_max_rel"] <= 1e-4
    assert abs(fe["val_loss"] - ae["val_loss"]) <= 1e-5 and fe["val_acc"] == ae["val_acc"]


def test_wsd_schedule_and_trunk(forge_bin, tmp_path):
    cfg = tiny_cfg("t-wsd", tmp_path, lr_schedule="wsd", max_steps=10, ckpt_every=0)
    fd, ad, fl, al = train_both(cfg, tmp_path)
    flr = [tel(fl)[s]["lr"] for s in range(1, 11)]
    alr = [tel(al)[s]["lr"] for s in range(1, 11)]
    want = [float(TrainConfig.from_dict(cfg).lr_at(s)) for s in range(10)]
    record("wsd lrs", {"forge": flr, "accel": alr})
    assert [np.float32(x) for x in flr] == [np.float32(x) for x in alr] == [np.float32(x) for x in want]
    assert [x["step"] for x in by_type(fl, "trunk")] == [x["step"] for x in by_type(al, "trunk")] == [8]
    st = json.loads((ad / "TRUNK" / "state.json").read_text())
    assert st["step"] == 8 and st["trunk"] is True and not (ad / "TRUNK.tmp").exists()
    assert ckpt_diff(fd / "TRUNK", ad / "TRUNK")["w_max_abs"] <= 1e-4


@pytest.mark.parametrize("init_optim", [True, False], ids=["init_optim", "fresh_moments"])
def test_one_window_init_from_parity(forge_bin, tmp_path, init_optim):
    """init_from runs/base-s/FINAL on a store with exactly one window: every row of every micro-batch is
    the same, independent of any RNG. 2 updates, batch 2, seq 16, accum 2, warmup 2, clip 1."""
    base = need_ckpt("runs/base-s/FINAL")
    mc = json.loads((base / "state.json").read_text())["config"]["model"]
    meta = one_window_store(tmp_path / "store", mc["vocab_size"], 16, seed=7)
    cfg = {"run": "ow", "model": mc, "data": str(meta), "batch": 2, "seq_len": 16, "grad_accum": 2, "max_steps": 2,
           "lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 2, "weight_decay": 0.1, "grad_clip": 1.0, "seed": 77,
           "init_from": str(base), "log_every": 1, "eval_every": 0, "eval_batches": 1, "ckpt_every": 1, "threads": 1,
           "autotune": False}
    if init_optim:
        cfg["init_optim"] = True
    fd, ad, fl, al = train_both(cfg, tmp_path)
    ft, at = tel(fl), tel(al)
    d = ckpt_diff(fd / "step-000002", ad / "step-000002")
    sf, sa = state_without(fd / "step-000002", "out_dir"), state_without(ad / "step-000002", "out_dir")
    key = "one-window base-s " + ("init_optim" if init_optim else "fresh moments")
    w0, _, _ = fio.read_safetensors(base / "model.safetensors")
    wf, _, _ = fio.read_safetensors(fd / "step-000002" / "model.safetensors")
    max_update = max(float(np.abs(wf[k].astype(np.float64) - w0[k]).max()) for k in w0)
    record(key, {"loss": [[ft[s]["loss"], at[s]["loss"]] for s in (1, 2)],
                 "grad_norm": [[ft[s]["grad_norm"], at[s]["grad_norm"]] for s in (1, 2)],
                 "lr": [[ft[s]["lr"], at[s]["lr"]] for s in (1, 2)], "weights_after_2": d, "max_update": max_update,
                 "optim_offset": [sf.get("optim_offset"), sa.get("optim_offset")]})
    for s in (1, 2):
        assert np.float32(ft[s]["lr"]) == np.float32(at[s]["lr"])
    # Step 1 runs on identical weights and identical windows: with forge's softmax sum (--xent forge) only
    # logit-level f32 noise is left (measured: identical f32 loss).
    assert abs(ft[1]["loss"] - at[1]["loss"]) <= 5e-6
    assert abs(ft[1]["grad_norm"] - at[1]["grad_norm"]) <= 2e-5 * ft[1]["grad_norm"]
    # Step 2 runs on the weights after one update: tight with warm moments; with fresh moments the
    # first update is ~lr*sign(g), which turns f32 noise in near-zero gradients into lr-sized steps on a
    # few hundred of the 12.6M weights (counted below), so step 2's loss moves by a few 1e-5.
    tol2 = 5e-6 if init_optim else 2e-4
    assert abs(ft[2]["loss"] - at[2]["loss"]) <= tol2
    assert abs(ft[2]["grad_norm"] - at[2]["grad_norm"]) <= (2e-5 if init_optim else 1e-4) * ft[2]["grad_norm"]
    for k in ("step", "tokens", "rng_state", "parent_sha256", "loss", "val_loss", "optim_offset", "config"):
        if k in ("loss", "val_loss"):
            assert abs(sf[k] - sa[k]) <= tol2, k  # the loss of step 2 and the final eval after it
        else:
            assert sf.get(k) == sa.get(k), k
    assert sa["parent_sha256"] == fio.sha256_file(base / "model.safetensors")
    if init_optim:
        bs = json.loads((base / "state.json").read_text())
        assert sa["optim_offset"] == bs["step"] + bs.get("optim_offset", 0)  # 2800 for base-s
        # Warm moments: AdamW's denominator is far from 0, so the update is a smooth function of the
        # gradient and f32 noise stays f32 noise (spec measured 3.3e-7 vs a max update of 1.4e-3).
        assert d["w_max_abs"] <= 5e-6
        assert d["m_max_rel"] <= 1e-3 and d["v_max_rel"] <= 1e-3
    else:
        # Fresh moments: step 1 is p -= lr*g/(|g|+eps*...) ~ lr*sign(g); elements with |g| near f32 noise
        # may flip, so only the bulk is compared tightly.
        assert d["w_n_over_1e-5"] <= 1e-3 * d["w_n"]
        assert d["w_max_abs"] <= 2.5e-3  # bounded by ~2 * lr (one sign flip over two steps)


# ---------------------------------------------------------------------------------------------- c) round trips
def test_round_trip_accel_to_forge(forge_bin, tmp_path):
    """accel writes a checkpoint; forge reads it (eval, logits, train --resume) and agrees with accel."""
    cfg = tiny_cfg("t-rt", tmp_path, max_steps=4, ckpt_every=2)
    fd, ad, fl, al = train_both(cfg, tmp_path)
    ack = ad / "step-000004"
    # forge eval / logits of the accel checkpoint vs accel's own
    args = ["--ckpt", str(ack), "--data", "synthetic:17", "--split", "val", "--batch", "4", "--seq", "16",
            "--batches", "3", "--seed", "4242"]
    fe, ae = forge_json("eval", *args), accel_json("eval", *args)
    lg = logits_diff(ack, [1, 2, 3, 4, 5, 6, 7], 32)
    assert fe["sha256"] == ae["sha256"] == fio.sha256_file(ack / "model.safetensors")
    # forge resumes the accel run at step 2 and finishes it: same as forge's own uninterrupted run
    rcfg = dict(cfg, out_dir=str(tmp_path / "resumed"))
    (tmp_path / "r.json").write_text(json.dumps(rcfg))
    rl = lines(forge("train", "--config", tmp_path / "r.json", "--resume", ad / "step-000002"))
    rd = tmp_path / "resumed" / "t-rt"
    d = ckpt_diff(fd / "step-000004", rd / "step-000004")
    rs = json.loads((rd / "step-000004" / "state.json").read_text())
    fs = json.loads((fd / "step-000004" / "state.json").read_text())
    record("round trip accel->forge", {"eval": [[fe["loss"], fe["next_token_acc"]], [ae["loss"], ae["next_token_acc"]]],
                                       "logits": lg, "forge_resume_of_accel_vs_forge": d,
                                       "resume_loss": [tel(rl)[4]["loss"], tel(fl)[4]["loss"]]})
    assert abs(fe["loss"] - ae["loss"]) <= 1e-5 and fe["next_token_acc"] == ae["next_token_acc"]
    assert lg["max_abs_diff"] <= 1e-5 and lg["argmax_forge"] == lg["argmax_accel"]
    assert rs["rng_state"] == fs["rng_state"] and rs["tokens"] == fs["tokens"] and rs["step"] == 4
    assert abs(tel(rl)[4]["loss"] - tel(fl)[4]["loss"]) <= 1e-5
    assert d["w_max_abs"] <= 1e-4


def test_round_trip_forge_to_accel(forge_bin, tmp_path):
    """forge writes checkpoints; accel re-writes them byte for byte and resumes them."""
    cfg = tiny_cfg("t-fa", tmp_path, max_steps=4, ckpt_every=2)
    fd, ad, fl, al = train_both(cfg, tmp_path)
    for name in ("model.safetensors", "optim.safetensors"):
        src = fd / "step-000002" / name
        t, meta, sha = fio.read_safetensors(src)
        out = tmp_path / f"rewritten-{name}"
        assert fio.write_safetensors(out, t, meta) == sha  # same header bytes, same order, same data
        assert not out.with_suffix(".tmp").exists()
    # accel resumes forge's step 2 and finishes: same as accel's own uninterrupted run (bit-identical on CPU)
    rcfg = dict(cfg, out_dir=str(tmp_path / "resumed"))
    (tmp_path / "r.json").write_text(json.dumps(rcfg))
    rl = lines(accel("train", "--config", str(tmp_path / "r.json"), "--resume", str(fd / "step-000002")))
    rd = tmp_path / "resumed" / "t-fa"
    d_forge = ckpt_diff(fd / "step-000004", rd / "step-000004")
    rs = json.loads((rd / "step-000004" / "state.json").read_text())
    record("round trip forge->accel resume", {"vs_forge_uninterrupted": d_forge,
                                              "loss": [tel(rl)[4]["loss"], tel(fl)[4]["loss"]]})
    assert rs["config"]["resume_from"] == str(fd / "step-000002")
    assert rs["rng_state"] == json.loads((fd / "step-000004" / "state.json").read_text())["rng_state"]
    assert d_forge["w_max_abs"] <= 1e-4
    assert abs(tel(rl)[4]["loss"] - tel(fl)[4]["loss"]) <= 1e-5


def test_accel_resume_is_exact(tmp_path):
    """accel interrupted at step 2 and resumed == accel uninterrupted, bit for bit (CPU, 1 thread)."""
    cfg = tiny_cfg("t-ar", tmp_path / "full", max_steps=4, ckpt_every=2)
    (tmp_path / "full.json").write_text(json.dumps(cfg))
    accel("train", "--config", str(tmp_path / "full.json"))
    full = tmp_path / "full" / "t-ar"
    rcfg = dict(cfg, out_dir=str(tmp_path / "res"))
    (tmp_path / "res.json").write_text(json.dumps(rcfg))
    accel("train", "--config", str(tmp_path / "res.json"), "--resume", str(full / "step-000002"))
    a = fio.sha256_file(full / "step-000004" / "model.safetensors")
    b = fio.sha256_file(tmp_path / "res" / "t-ar" / "step-000004" / "model.safetensors")
    assert a == b


# ---------------------------------------------------------------------------------------------- devices / precision
def test_data_parallel_two_devices(tmp_path):
    """The same run on 1 and on 2 (forced host) devices: the batch is split, gradients all-reduced."""
    cfg = tiny_cfg("t-dp", tmp_path, max_steps=4, ckpt_every=0)
    outs = {}
    for n in (1, 2):
        c = dict(cfg, out_dir=str(tmp_path / f"d{n}"))
        (tmp_path / f"d{n}.json").write_text(json.dumps(c))
        env = child_env(XLA_FLAGS=f"--xla_force_host_platform_device_count={n} --xla_cpu_multi_thread_eigen=false "
                                  "intra_op_parallelism_threads=1")
        outs[n] = lines(run([PY, "-m", "accel.train", "--config", tmp_path / f"d{n}.json"], env=env))
    assert by_type(outs[2], "telemetry")[0]["devices"] == 2
    d = ckpt_diff(tmp_path / "d1" / "t-dp" / "step-000004", tmp_path / "d2" / "t-dp" / "step-000004")
    l1 = [x["loss"] for x in by_type(outs[1], "telemetry")]
    l2 = [x["loss"] for x in by_type(outs[2], "telemetry")]
    record("data parallel 1 vs 2 devices", {"loss_1": l1, "loss_2": l2, "weights": d})
    assert max(abs(a - b) for a, b in zip(l1, l2)) <= 1e-6
    assert d["w_max_abs"] <= 1e-5


def test_bf16_precision(tmp_path):
    """bf16 GEMM inputs (forge's amx-bf16 arithmetic): same run, losses within bf16 rounding of f32."""
    cfg = tiny_cfg("t-bf", tmp_path, max_steps=4, ckpt_every=0)
    res = {}
    for p in ("f32", "bf16"):
        c = dict(cfg, out_dir=str(tmp_path / p))
        (tmp_path / f"{p}.json").write_text(json.dumps(c))
        res[p] = [x["loss"] for x in by_type(lines(accel("train", "--config", str(tmp_path / f"{p}.json"),
                                                         "--precision", p)), "telemetry")]
    record("bf16 vs f32 losses (tiny)", res)
    assert all(np.isfinite(res["bf16"]))
    assert max(abs(a - b) for a, b in zip(res["f32"], res["bf16"])) <= 2e-2  # bf16 has 8 mantissa bits


# ---------------------------------------------------------------------------------------------- drop-in
def test_fabric_through_forge_accel(forge_bin, tmp_path):
    """cluster/fabric.py init --scratch / work / aggregate / eval with FORGE=accel/bin/forge-accel."""
    sys_py = shutil.which("python3")
    if not sys_py or subprocess.run([sys_py, "-c", "import numpy"], capture_output=True).returncode:
        pytest.skip("system python3 with numpy needed for cluster/fabric.py")
    model = dict(TINY, max_seq_len=256)  # fabric eval uses --seq 256
    (tmp_path / "tiny.json").write_text(json.dumps({"model": model}))
    store = tmp_path / "store"
    store.mkdir()
    rng = np.random.default_rng(1)
    for n in ("train.bin", "val.bin"):
        rng.integers(0, 64, 1 << 14, dtype=np.uint16).tofile(store / n)
    (store / "train.meta.json").write_text(json.dumps({"vocab_size": 64, "bin": "train.bin", "val_bin": "val.bin",
                                                       "dtype": "uint16"}))
    env = child_env(FORGE=ROOT / "accel" / "bin" / "forge-accel", ACCEL_PYTHON=PY, FORGE_REAL=FORGE)
    fab = [sys_py, ROOT / "cluster" / "fabric.py"]
    S = tmp_path / "state"
    out = lines(run(fab + ["init", "--scratch", tmp_path / "tiny.json", "--state", S, "--seed", 3], env=env))
    ref = init_params(ModelConfig.from_dict(model), 3, exact=True)
    g, _, _ = fio.read_safetensors(S / "global" / "model.safetensors")
    init_gap = max(float(np.abs(g[k] - ref[k]).max()) for k in ref)
    recs = []
    for w in (0, 1):
        recs.append(lines(run(fab + ["work", "--state", S, "--worker", w, "--steps", 3, "--data", store / "train.meta.json",
                                     "--out", tmp_path / f"w{w}", "--batch", 2, "--seq", 16, "--accum", 1, "--threads", 1,
                                     "--lr", 1e-3], env=env))[-1])
    agg = lines(run(fab + ["aggregate", "--state", S, "--deltas", tmp_path / "w0", tmp_path / "w1"], env=env))[-1]
    ev = lines(run(fab + ["eval", "--state", S, "--data", store / "train.meta.json", "--batches", 2], env=env))[-1]
    record("fabric via forge-accel", {"init_vs_exact_init_max_abs": init_gap, "workers": recs, "aggregate": agg,
                                      "eval": ev})
    assert init_gap <= 1e-9  # one constant-lr 1e-12 step on top of forge's exact init
    assert agg["round"] == 1 and agg["workers"] == 2
    assert all(r["loss_last"] is not None and r["tokens_per_s"] > 0 for r in recs)
    assert np.isfinite(ev["val_loss"])
    # forge itself evaluates the cluster's accel-trained global model
    fe = forge_json("eval", "--ckpt", S / "global", "--data", store / "train.meta.json", "--split", "val", "--batch", "8",
                    "--seq", "256", "--batches", "2", "--seed", "4242")
    record("fabric global model: forge eval vs accel eval", {"forge": fe["loss"], "accel": ev["val_loss"],
                                                             "diff": abs(fe["loss"] - ev["val_loss"])})
    assert abs(fe["loss"] - ev["val_loss"]) <= 1e-5


def test_remat_same_result(tmp_path):
    """--remat recomputes each layer in the backward pass; the arithmetic is the same."""
    cfg = tiny_cfg("t-rm", tmp_path, max_steps=2, ckpt_every=0)
    res = {}
    for flag in ("", "--remat"):
        c = dict(cfg, out_dir=str(tmp_path / (flag or "plain")))
        (tmp_path / "c.json").write_text(json.dumps(c))
        res[flag] = [x["loss"] for x in by_type(lines(accel("train", "--config", str(tmp_path / "c.json"),
                                                            *([flag] if flag else []))), "telemetry")]
    d = ckpt_diff(tmp_path / "plain" / "t-rm" / "step-000002", tmp_path / "--remat" / "t-rm" / "step-000002")
    record("remat vs plain", {"loss": res, "weights": d})
    assert max(abs(a - b) for a, b in zip(res[""], res["--remat"])) <= 1e-6 and d["w_max_abs"] <= 1e-6


def test_control_commands(tmp_path, capsys):
    """forge's stdin commands (trainer.rs:563-608) on an in-process Trainer."""
    from accel.train import Trainer
    cfg = TrainConfig.from_dict(tiny_cfg("t-ctl", tmp_path, max_steps=3, ckpt_every=0, warmup_steps=0, lr=1e-2))
    tr = Trainer(cfg, final_link=False)
    tr.handle('{"cmd":"pause"}')
    assert tr.paused
    tr.handle('{"cmd":"resume"}')
    assert not tr.paused
    tr.handle('{"cmd":"set_lr","value":0.005}')
    assert np.float32(tr.lr_now()) == np.float32(np.float32(0.005) * cfg.lr_at(0) / cfg.lr)  # value * lr_at / lr
    tr.handle('{"cmd":"set_lr","value":2}')
    tr.handle('{"cmd":"set_threads","value":0}')
    tr.handle('{"cmd":"set_threads","value":4}')
    tr.handle('{"cmd":"swap_kernel","variant":"avx2-fma"}')
    tr.handle('{"cmd":"nope"}')
    tr.handle("not json")
    tr.handle('{"cmd":"checkpoint"}')
    assert (tmp_path / "t-ctl" / "step-000000" / "model.safetensors").exists()
    tr.handle('{"cmd":"stop"}')
    tr.run()
    out = lines(capsys.readouterr().out)
    msgs = [(x["level"], x["msg"]) for x in by_type(out, "event")]
    # forge's handle() (trainer.rs:563-608) emits "applied <cmd>" after a rejected value too
    i = msgs.index(("warn", "set_lr needs 0 < value < 1"))
    assert msgs[i:i + 5] == [("warn", "set_lr needs 0 < value < 1"), ("info", "applied set_lr"),
                             ("warn", "set_threads needs 1..=1024"), ("info", "applied set_threads"),
                             ("info", "applied set_threads")]
    assert ("warn", 'unknown command "nope"') in msgs
    assert any(lv == "warn" and m.startswith("unparseable command") for lv, m in msgs)
    assert ("info", "applied pause") in msgs and ("info", "applied stop") in msgs
    assert [x["step"] for x in by_type(out, "done")] == [0]  # stopped before the first update, like forge


# ---------------------------------------------------------------------------------------------- review fixes
def test_adamw_bitwise_on_identical_inputs(forge_bin, tmp_path):
    """grad_clip 0 makes the clip coefficient 0, so an update depends only on the loaded weights and moments
    (init_from + init_optim of a forge checkpoint) and the f32 hyper-parameters, which are bit-equal. forge's
    loop rounds every product; XLA's CPU backend contracts a*b + c into one fma unless accel stops it (without
    the guard: 14,241 of these 32,928 weights 1 ulp apart; 5.1M of base-s's 12.6M after 2 updates). Weights
    and moments after 3 updates must be the same bits."""
    src = tiny_cfg("t-src", tmp_path / "src", max_steps=2, ckpt_every=0)
    (tmp_path / "src.json").write_text(json.dumps(src))
    forge("train", "--config", tmp_path / "src.json")
    base = tmp_path / "src" / "t-src" / "step-000002"
    cfg = tiny_cfg("t-aw", tmp_path, max_steps=3, ckpt_every=0, grad_clip=0.0, init_from=str(base), init_optim=True)
    fd, ad, fl, al = train_both(cfg, tmp_path)
    d = ckpt_diff(fd / "step-000003", ad / "step-000003")
    record("adamw clip 0 init_optim tiny", d)
    assert [x["lr"] for x in by_type(fl, "telemetry")] == [x["lr"] for x in by_type(al, "telemetry")]
    assert d["w_max_abs"] == 0.0 and d["m_max_rel"] == 0.0 and d["v_max_rel"] == 0.0
    assert fio.sha256_file(fd / "step-000003" / "model.safetensors") == fio.sha256_file(ad / "step-000003" / "model.safetensors")


def bad_token_store(d: Path, vocab: int = 8192) -> Path:
    """A store whose meta says vocab 8192 but every 7th token is 9000..9499 (forge: BadToken)."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for name in ("train.bin", "val.bin"):
        a = rng.integers(0, vocab, 4096).astype(np.uint16)
        a[::7] = 9000 + rng.integers(0, 500, a[::7].size)
        a.tofile(d / name)
    (d / "train.meta.json").write_text(json.dumps({"vocab_size": vocab, "bin": "train.bin", "val_bin": "val.bin",
                                                   "dtype": "uint16", "tokens": 4096, "val_tokens": 4096}))
    return d / "train.meta.json"


def run_rc(cmd: list) -> tuple[int, str]:
    r = subprocess.run(NICE + [str(c) for c in cmd], cwd=ROOT, env=child_env(), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    return r.returncode, (r.stderr.strip().splitlines() or [""])[-1]


def test_bad_token_ids_fail_like_forge(forge_bin, tmp_path):
    """Token ids >= the model's vocab: forge stops with BadToken (exit 1). On a device XLA's gather would
    return NaN instead, so accel checks the ids on the host, with forge's message, and writes nothing."""
    import re
    meta = bad_token_store(tmp_path / "store")
    strip = lambda m: re.sub(r"^(forge: pc \d+: |accel: )", "", m)  # noqa: E731
    cfg = {"run": "bad", "model": {"vocab_size": 8192, "dim": 16, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1,
                                   "max_seq_len": 32},
           "data": str(meta), "batch": 2, "seq_len": 16, "max_steps": 3, "lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 1,
           "log_every": 1, "eval_batches": 1, "autotune": False, "threads": 1, "seed": 2}
    msgs = {}
    for kind in ("forge", "accel"):
        c = dict(cfg, out_dir=str(tmp_path / kind))
        (tmp_path / f"{kind}.json").write_text(json.dumps(c))
        cmd = [FORGE, "train"] if kind == "forge" else [PY, "-m", "accel.train"]
        msgs[kind] = run_rc(cmd + ["--config", tmp_path / f"{kind}.json"])
    record("bad token ids: train", msgs)
    assert msgs["forge"][0] != 0 and msgs["accel"][0] != 0
    assert strip(msgs["forge"][1]) == strip(msgs["accel"][1]) and "is not an integer id below 8192" in msgs["accel"][1]
    run_dir = tmp_path / "accel" / "bad"
    assert not run_dir.exists() or not any(run_dir.iterdir())  # no NaN checkpoint, no FINAL
    if (ROOT / "runs/base-s/FINAL/model.safetensors").exists():
        args = ["eval", "--ckpt", "runs/base-s/FINAL", "--data", meta, "--batch", "2", "--seq", "32", "--batches", "2"]
        f = run_rc([FORGE, *args, "--threads", "1"])
        a = run_rc([PY, "-m", "accel.eval", *args[1:]])
        record("bad token ids: eval", {"forge": f, "accel": a})
        assert f[0] != 0 and a[0] != 0 and strip(f[1]) == strip(a[1])


def test_final_only_for_finished_runs(tmp_path):
    """forge train writes no FINAL; accel writes it only with --final, and only once max_steps is reached."""
    from accel.train import Trainer
    cfg = TrainConfig.from_dict(tiny_cfg("t-fin", tmp_path, max_steps=2, ckpt_every=0))
    tr = Trainer(cfg, final_link=True)
    tr.handle('{"cmd":"stop"}')
    tr.run()
    assert tr.step == 0 and not (tmp_path / "t-fin" / "FINAL").exists()  # a stopped run is not finished
    Trainer(cfg, final_link=True).run()
    assert str((tmp_path / "t-fin" / "FINAL").readlink()) == "step-000002"
    cfg2 = TrainConfig.from_dict(tiny_cfg("t-nofin", tmp_path, max_steps=1, ckpt_every=0))
    Trainer(cfg2).run()  # the default, as `python -m accel.train` without --final
    assert sorted(os.listdir(tmp_path / "t-nofin")) == ["latest.json", "step-000001"]


HLO_PROBE = r"""
import re, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import jax, jax.numpy as jnp
from accel.config import ModelConfig
from accel.runtime import Mesh
from accel.train import build_step
A = 3
mc = ModelConfig.from_dict({"vocab_size": 64, "dim": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "max_seq_len": 32})
mesh = Mesh(2, 4)
P = {n: jax.ShapeDtypeStruct(s, jnp.float32, sharding=mesh.repl) for n, s in mc.param_specs()}
xs = jax.ShapeDtypeStruct((A, 4, 16), jnp.int32, sharding=mesh.micro_rows)
hy = jax.ShapeDtypeStruct((4,), jnp.float32, sharding=mesh.repl)
txt = build_step(mc, 4, 16, A, 1.0, mesh, "f32", False).lower(P, P, P, xs, xs, hy).compile().as_text()
out = {"devices": mesh.n, "params": mc.num_params(), "A": A, "entry": 0, "other": 0, "bytes": 0}
for block in txt.split("\n\n"):
    for line in block.splitlines():
        m = re.search(r"= (.*?) all-reduce(?:-start)?\(", line)
        if not m:
            continue
        out["entry" if block.lstrip().startswith("ENTRY") else "other"] += 1
        out["bytes"] += sum(4 * int(np.prod([int(d) for d in dims.split(",") if d])) for dims in re.findall(r"f32\[([\d,]*)\]", m.group(1)))
print(out)
"""


def test_one_all_reduce_per_update(tmp_path):
    """Data parallel with grad_accum 3 on 2 devices: the gradients are all-reduced once per update, in the
    step's entry computation, not once per micro-batch inside the accumulation loop; the bytes reduced are
    the parameters plus the A micro-batch loss sums (tok_emb's two gradient parts merged before it)."""
    (tmp_path / "probe.py").write_text(HLO_PROBE)
    env = child_env(XLA_FLAGS="--xla_force_host_platform_device_count=2 --xla_cpu_multi_thread_eigen=false "
                              "intra_op_parallelism_threads=1")
    out = eval(run([PY, tmp_path / "probe.py", ROOT], env=env).strip().splitlines()[-1])  # noqa: S307 - our own dict
    record("all-reduce per update (tiny, 2 devices, accum 3)", out)
    assert out["devices"] == 2 and out["other"] == 0 and out["entry"] >= 1
    assert out["bytes"] == 4 * (out["params"] + out["A"])


MH_LAUNCH = r"""
import hashlib, sys
sys.path.insert(0, sys.argv[1])
import jax
jax.config.update("jax_cpu_collectives_implementation", "gloo")
import numpy as np
import accel.train as T
_run = T.Trainer.run
def run(self, *a, **k):
    r = _run(self, *a, **k)
    h = hashlib.sha256()
    for n in sorted(self.params):  # this process's own copy of the replicated parameters
        h.update(np.asarray(self.params[n].addressable_data(0)).tobytes())
    print(f"PARAMHASH {jax.process_index()} {self.step} {h.hexdigest()}", file=sys.stderr, flush=True)
    return r
T.Trainer.run = run
sys.exit(T.main(sys.argv[2:]))
"""


def test_multihost_commands_reach_every_process(tmp_path):
    """A simulated 2-host run (2 processes x 2 CPU devices, gloo collectives, coordinator on loopback),
    commands written to process 0's stdin only, as scripts/longrun.py does: set_lr must change every
    process's update (identical replicated weights at the end), and checkpoint + stop must end both
    processes at the same step (before the fix: process 0 entered the final eval alone and both hung)."""
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    cfg = tiny_cfg("t-mh", tmp_path, batch=4, max_steps=400, ckpt_every=0, log_every=1)
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    (tmp_path / "launch.py").write_text(MH_LAUNCH)
    procs, logs = [], []
    try:
        for i in range(2):
            env = child_env(ACCEL_DISTRIBUTED=1, XLA_FLAGS="--xla_force_host_platform_device_count=2",
                            JAX_COORDINATOR_ADDRESS=f"127.0.0.1:{port}", SLURM_JOB_ID="4242",
                            SLURM_STEP_NODELIST="localhost", SLURM_NTASKS=2, SLURM_PROCID=i, SLURM_LOCALID=0,
                            SLURM_STEP_NUM_NODES=1, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
            logs.append(open(tmp_path / f"p{i}.err", "w"))
            procs.append(subprocess.Popen(NICE + [PY, str(tmp_path / "launch.py"), str(ROOT), "--config", str(tmp_path / "c.json")],
                                          cwd=ROOT, env=env, text=True, stdin=subprocess.PIPE if i == 0 else subprocess.DEVNULL,
                                          stdout=subprocess.PIPE if i == 0 else subprocess.DEVNULL,
                                          stderr=logs[-1]))
        p0 = procs[0]
        out0 = []

        def feed():  # the driver: reads process 0's stdout, writes commands to process 0 only
            first = None
            for line in p0.stdout:
                out0.append(line)
                if '"telemetry"' not in line:
                    continue
                step = json.loads(line)["step"]
                if first is None:
                    first = step
                    p0.stdin.write('{"cmd":"set_lr","value":0.001}\n')
                    p0.stdin.flush()
                elif step == first + 6:
                    p0.stdin.write('{"cmd":"checkpoint"}\n{"cmd":"stop"}\n')
                    p0.stdin.close()

        t0 = time.time()
        threading.Thread(target=feed, daemon=True).start()
        errs = []
        for i, p in enumerate(procs):
            try:
                p.wait(timeout=max(10.0, 300 - (time.time() - t0)))
                err = ""
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                err = "TIMEOUT\n"
            errs.append(err + (tmp_path / f"p{i}.err").read_text())
    finally:  # a hung slice (the bug this guards against) is killed, which also ends the reader thread
        for p in procs:
            if p.poll() is None:
                p.kill()
        for f in logs:
            f.close()
    out0 = lines("".join(out0))
    hashes = [next((x.split()[2:] for x in e.splitlines() if x.startswith("PARAMHASH")), None) for e in errs]  # step, sha
    tl = tel(out0)
    tcfg = TrainConfig.from_dict(cfg)
    changed = [s for s in sorted(tl) if np.float32(tl[s]["lr"]) != np.float32(tcfg.lr_at(s - 1))]
    done = by_type(out0, "done")
    record("multi-host 2x2 commands", {"rc": [p.returncode for p in procs], "hashes": hashes,
                                       "done_step": done[0]["step"] if done else None, "lr_changed_from": changed[:1],
                                       "stderr_tail": [e.strip().splitlines()[-1:] for e in errs]})
    assert [p.returncode for p in procs] == [0, 0], errs
    assert hashes[0] is not None and hashes[0] == hashes[1]  # same step, same bits on both processes
    assert done and done[0]["step"] < 400 and changed  # stopped early; set_lr took effect
