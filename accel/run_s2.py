"""s2_run1 pod runner with time-budget guard.

python -m accel.run_s2 --data META --clean-tsv TSV --init S2_DIR --out OUT --deadline EPOCH [--attn cudnn] ...

* init from S2_DIR/model.safetensors, each device memmaps only its own shard (lazy_sharded); or resume from the
  newest OUT/step-* (final ckpt of an earlier run: weights + AdamW + step + the planned schedule in its state.json).
* plan: after PLAN_AT (20) steps the mean step time of steps 6..20 sets
      total = min(cap, steps_done + floor((deadline - now - reserve) / step_time))
  and the schedule is recomputed for `total`: warmup 300 fixed (round(total/2) if total < 600), wsd decay over the
  last round(0.2*total) steps to min_lr; warmup = min(300, int(0.3*total)) (>= 1). The plan is stored in the
  checkpoint; a resumed run never re-plans.
* hard guard: before each step, if now + step_time + reserve > deadline -> stop early.
* end (normal or guard): finish_run -> ONE full f32 ckpt OUT/step-N (model+optim+state, ~11.5 GB for s2) + bf16
  weights-only export + EXPORT.json; OUT/DONE.json. No periodic checkpoints (pod has no volume, so they could not
  be used for resume anyway) and nothing is ever deleted by this script. JSONL log OUT/train.jsonl with tok/s.
Exit 0 = finished (status ok|deadline); nonzero = error (DONE.json status error written when possible).
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np


def plan_schedule(total: int, cap: int = 1000) -> dict:
    total = max(1, min(int(total), cap))
    warm = max(1, min(300, int(0.3 * total)))
    return {"max_steps": total, "warmup_steps": warm, "decay_steps": 0}   # decay = round(0.2*total) in lr_at


def plan_total(steps_done: int, step_s: float, now: float, deadline: float, reserve_s: float, cap: int) -> int:
    left = deadline - now - reserve_s
    return max(steps_done, min(cap, steps_done + int(left // step_s))) if step_s > 0 else cap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--clean-tsv", required=True)
    ap.add_argument("--init", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, required=True, help="unix time; run must be fully saved before it")
    ap.add_argument("--reserve-s", type=float, default=600.0, help="time kept for final ckpt + export")
    ap.add_argument("--cap", type=int, default=1000); ap.add_argument("--plan-at", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512); ap.add_argument("--seed", type=int, default=67102)
    ap.add_argument("--precision", default="bf16"); ap.add_argument("--attn", default="cudnn")
    ap.add_argument("--remat", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None, help="peak lr (default: the s2_run1 preset, 2e-4)")
    ap.add_argument("--min-lr", type=float, default=None, help="final lr (default: the s2_run1 preset, 3e-5)")
    ap.add_argument("--mesh", default="", help="DATAxFSDP, default 1x<all devices>")
    ap.add_argument("--no-verify-sha", action="store_true", help="tests only")
    ap.add_argument("--tiny", action="store_true", help="tests only: tiny random model instead of --init")
    ap.add_argument("--fake-step-s", type=float, default=0.0, help="tests only: add sleep per step")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.jsonl", "a")
    def log(**kv):
        kv["t"] = round(time.time(), 3); line = json.dumps(kv); print(line, flush=True); logf.write(line + "\n"); logf.flush()
    def done(status, **kv):
        (out / "DONE.json").write_text(json.dumps({"status": status, **kv}, indent=1)); log(event="done", status=status, **kv)
    try:
        import jax
        from accel import mgpu as G, model as M
        from accel.config import ModelConfig
        t_start = time.time()
        ndev = jax.device_count()
        d, f = (int(x) for x in a.mesh.split("x")) if a.mesh else (1, ndev)
        mesh = G.make_mesh(d, f)
        hp = G.HP.preset("s2_run1", batch=a.batch, accum=a.accum, seq=a.seq, seed=a.seed, precision=a.precision,
                         attn=a.attn, remat=bool(a.remat), data=a.data, clean_tsv=a.clean_tsv, verify_sha=not a.no_verify_sha,
                         **{k: v for k, v in (("lr", a.lr), ("min_lr", a.min_lr)) if v is not None})
        data = G.ShardedData(hp, mesh, 8192)        # fail-closed clean store (raises DataSafetyError)
        cks = sorted([c for c in out.glob("step-*") if (c / "state.json").exists()],
                     key=lambda c: int(c.name.split("-")[1]))
        planned = None
        if cks:
            st = json.loads((cks[-1] / "state.json").read_text())
            mc = ModelConfig.from_dict(st["config"]["model"])
            p, m, v, t, _ = G.load(cks[-1], mc, mesh)
            planned = st.get("plan"); src = f"resume {cks[-1].name}"
        else:
            if a.tiny:
                mc = ModelConfig(vocab_size=8192, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=128,
                                 max_seq_len=a.seq, rope_theta=10000.0, norm_eps=1e-5)
                p, m, v, t = G.init_state(mc, mesh, M.init_params(mc, 7, exact=False)); src = "tiny random"
            else:
                ini = Path(a.init); st = json.loads((ini / "state.json").read_text())
                mc = ModelConfig.from_dict(st["config"]["model"]); sh = G.shardings(mc, mesh)
                hdr = G.st_header(ini / "model.safetensors")
                p = {k: G.lazy_sharded(ini / "model.safetensors", k, sh[k], hdr) for k, _ in mc.param_specs()}
                z = jax.jit(lambda q: jax.tree.map(jax.numpy.zeros_like, q), out_shardings=sh)
                m, v, t = z(p), z(p), jax.numpy.zeros((), jax.numpy.int32); src = f"init {ini}"
        if planned:
            hp = G.HP(**{**hp.__dict__, **planned})
        else:
            hp = G.HP(**{**hp.__dict__, **plan_schedule(a.cap, a.cap)})
        log(event="start", src=src, devices=ndev, mesh=f"{d}x{f}", params=mc.num_params(), step=int(t),
            tokens_per_step=a.batch * a.accum * a.seq, deadline=a.deadline, secs_to_deadline=round(a.deadline - time.time(), 1),
            plan=planned, setup_s=round(time.time() - t_start, 1))
        step_fn = G.build_step(mc, hp, mesh); rep = G.NamedSharding(mesh, G.P())
        tok = a.batch * a.accum * a.seq
        step = int(t); times = []; status = "ok"; step_s = None
        def plan_now():
            return {"max_steps": hp.max_steps, "warmup_steps": hp.warmup_steps, "decay_steps": hp.decay_steps}
        while step < hp.max_steps:
            if step_s is not None and time.time() + step_s + a.reserve_s > a.deadline:
                status = "deadline"; log(event="deadline_guard", step=step, step_s=round(step_s, 3)); break
            xs, ys = data.global_batch(step)
            t0 = time.time()
            p, m, v, loss, gn = step_fn(p, m, v, xs, ys, jax.device_put(G.hyper_at(hp, step), rep))
            loss = float(loss)                       # blocks until the step is done
            if a.fake_step_s: time.sleep(a.fake_step_s)
            dt = time.time() - t0; step += 1; times.append(dt)
            if len(times) > 5: step_s = float(np.mean(times[5:]))
            elif step_s is None and len(times) >= 2: step_s = float(np.max(times[1:]))
            if step % 10 == 0 or step <= 3:
                log(event="step", step=step, loss=round(loss, 5), gnorm=round(float(gn), 4), lr=float(hp.lr_at(step - 1)),
                    step_s=round(dt, 4), tok_s=round(tok / dt, 1))
            if not planned and step == a.plan_at:
                st_s = float(np.mean(times[5:])) if len(times) > 5 else float(np.mean(times[1:]))
                total = plan_total(step, st_s, time.time(), a.deadline, a.reserve_s, a.cap)
                hp = G.HP(**{**hp.__dict__, **plan_schedule(total, a.cap)}); planned = plan_now()
                log(event="plan", step=step, step_s=round(st_s, 4), tok_s=round(tok / st_s, 1), total=total, **planned,
                    est_train_end_s=round(total * st_s, 1))
        t1 = time.time()
        res = G.finish_run(out, mc, hp, p, m, v, step)
        sj = json.loads((Path(res["ckpt"]) / "state.json").read_text()); sj["plan"] = plan_now()
        (Path(res["ckpt"]) / "state.json").write_text(json.dumps(sj, indent=1))
        tr = sum(times)
        done(status, step=step, planned=plan_now(), export=res.get("export"), final_ckpt=res["ckpt"],
             train_s=round(tr, 1), mean_tok_s=round(tok * len(times) / tr, 1) if tr else None,
             finish_s=round(time.time() - t1, 1), wall_s=round(time.time() - t_start, 1))
        return 0
    except Exception as e:
        try: done("error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        except Exception: pass
        return 3 if type(e).__name__ == "DataSafetyError" else 1


if __name__ == "__main__":
    sys.exit(main())
