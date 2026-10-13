"""python -m accel.train --config RUN.json [--resume DIR] [--precision f32|bf16] [--remat] [--devices N]

`forge train` on JAX devices. Same run config (core_engine/train/src/config.rs), same
algorithm (trainer.rs; spec in docs/ACCEL.md):

  * initialisation: forge's seeded generator bit for bit (fresh), or init_from (+ init_optim);
    resume_from / --resume restores weights, moments, step, tokens, RNG and lineage;
  * data: forge's Rng(seed) and window sampler, so the batches are forge's batches;
  * per update: grad_accum micro-batches, gradients of sum(row CE) * 1/(B*T*accum) summed in
    order, global-norm clip (pre-clip norm in telemetry), AdamW (beta 0.9/0.95, eps 1e-8,
    decoupled decay on ndim >= 2) with bias correction at optim_offset + step, all in f32;
  * LR schedules cosine | wsd (with TRUNK) | constant with linear warmup, in f32;
  * in-train eval with Rng(seed ^ 0xE7A1) on the val split;
  * stdout: forge's JSON lines (event, telemetry, eval, checkpoint, trunk, done), sorted keys;
  * files: <out_dir>/<run>/step-NNNNNN/{model,optim}.safetensors + state.json, latest.json and
    TRUNK for wsd -- the files forge writes. Like forge, no FINAL: scripts/longrun.py links FINAL
    itself once latest.json reaches max_steps. --final links FINAL when the run reaches max_steps
    (never for a stopped run);
  * stdin commands: pause, resume, stop, checkpoint, set_lr{value} (the CPU-only ones --
    autotune, set_threads, swap_kernel -- are acknowledged and ignored), with forge's events.
    Multi-host (ACCEL_DISTRIBUTED=1): process 0 reads stdin and every process applies its
    commands at the same update, one update after they arrive.

Differences from forge, by design: the batch is split over all JAX devices (data parallel);
`threads`, `hbvm_mib` and `autotune` have no meaning on XLA and are accepted silently; other
unknown keys get a warn event; telemetry step_ms is the mean step time since the last line
(dispatch is asynchronous), `ops` is empty and `hbvm` holds the device allocator's numbers.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import forge_io as fio
from .config import ConfigError, TrainConfig, ModelConfig
from .forge_io import Rng, Store, f32, powf
from .runtime import Mesh, emit, event, init_distributed, is_main, memory_stats, now, replicated_to_host, to_host

B1, B2, EPS = f32(0.9), f32(0.95), f32(1e-8)
OMB1, OMB2 = f32(f32(1.0) - B1), f32(f32(1.0) - B2)  # 0.100000024, 0.050000012 as forge computes them


# ---------------------------------------------------------------------------------------------- compiled programs
def build_step(mc: ModelConfig, B: int, T: int, A: int, grad_clip, mesh: Mesh, precision: str, remat: bool,
               xent: str = "fast"):
    """The train step + update program as one jitted function:
    (params, m, v, xs[A,B,T], ys[A,B,T], hyper[step_size, bc2s, decay, 0]) -> (params, m, v, ce_sums[A], grad_norm).

    Data parallel: under shard_map every device runs the micro-batch loop on its rows of each
    micro-batch and accumulates its own partial gradients; one psum over the mesh after the loop
    gives the full gradient. So an update all-reduces the parameter size once, whatever grad_accum
    is (an all-reduce per micro-batch otherwise), and tok_emb's two gradient parts (embedding
    scatter and tied head) are added on the device before that one reduction."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as PS
    from . import model as M

    gs = f32(1.0) / f32(B * T * A)  # model.rs:397, f32 division
    gc = f32(grad_clip)
    names = [n for n, _ in mc.param_specs()]
    decays = {n: len(s) >= 2 for n, s in mc.param_specs()}

    def loss_fn(params, x, y):
        ce, _ = M.row_losses(params, mc, x, y, precision, remat, xent)
        return jnp.sum(ce) * gs, M.pairwise_sum(ce)

    vg = jax.value_and_grad(loss_fn, has_aux=True)

    def local_grads(params, xs, ys):
        """One device: its rows of every micro-batch (xs [A, B/n, T]) -> summed gradients and per-micro-batch
        CE sums, all-reduced once over the mesh."""
        if A == 1:
            (_, cs), grads = vg(params, xs[0], ys[0])
            ce_sums = cs[None]
        else:
            def micro(gacc, xy):
                (_, cs), g = vg(params, xy[0], xy[1])
                return {k: gacc[k] + g[k] for k in gacc}, cs  # beta = 1 accumulation, in order
            grads, ce_sums = jax.lax.scan(micro, {k: jnp.zeros_like(p) for k, p in params.items()}, (xs, ys))
        return jax.lax.psum((grads, ce_sums), "data")

    all_grads = shard_map(local_grads, mesh=mesh.mesh, in_specs=(PS(), PS(None, "data"), PS(None, "data")),
                          out_specs=(PS(), PS()))

    def step(params, m, v, xs, ys, hyper):
        step_size, bc2s, decay = hyper[0], hyper[1], hyper[2]
        zero = jax.lax.bitcast_convert_type(hyper[3], jnp.int32)  # 0 at run time, unknown to the compiler

        def rounded(x):
            """x, rounded to f32 on its own: XLA's CPU and GPU backends contract a*b + c into one fma
            (one rounding); forge's Rust loop rounds the product first. An integer xor with a run-time
            zero is a no-op the compiler cannot see through, so no fma forms across it."""
            return jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(x, jnp.int32) ^ zero, jnp.float32)

        grads, ce_sums = all_grads(params, xs, ys)
        sumsq = f32(0.0)
        for n in names:  # SumSq over every gradient, tok_emb included (machine.rs:983)
            sumsq = sumsq + M.pairwise_sum(grads[n] * grads[n])
        reg = jnp.where(jnp.isnan(sumsq), f32(0.0), jnp.maximum(sumsq, f32(0.0)))  # Rust f32::max(NaN, 0) = 0
        norm = jnp.sqrt(reg)
        clip = jnp.minimum(gc / (norm + f32(1e-6)), f32(1.0))  # ClipCoef
        P, Mo, V = {}, {}, {}
        for n in names:  # AdamW, machine.rs:1000-1040: same operation order, every product rounded (no fma)
            gi = grads[n] * clip
            mm = rounded(B1 * m[n]) + rounded(OMB1 * gi)
            vv = rounded(B2 * v[n]) + rounded((OMB2 * gi) * gi)
            dec = decay if decays[n] else f32(1.0)  # 1 - lr*0.0 == 1 exactly
            P[n] = rounded(params[n] * dec) - (step_size * mm) / (jnp.sqrt(vv) / bc2s + EPS)
            Mo[n], V[n] = mm, vv
        return P, Mo, V, ce_sums, norm

    r = mesh.repl
    return jax.jit(step, in_shardings=(r, r, r, mesh.micro_rows, mesh.micro_rows, r),
                   out_shardings=(r, r, r, r, r), donate_argnums=(0, 1, 2))


def build_eval(mc: ModelConfig, mesh: Mesh, precision: str, xent: str = "forge"):
    """(params, x[B,T], y[B,T]) -> (row CE [B,T], hit [B,T]) -- forge's Eval program."""
    import jax
    from . import model as M

    def ev(params, x, y):
        return M.row_losses(params, mc, x, y, precision, xent=xent)

    return jax.jit(ev, in_shardings=(mesh.repl, mesh.rows, mesh.rows), out_shardings=(mesh.rows, mesh.rows))


def eval_windows(ev, params, mesh: Mesh, data: Store, rng: Rng, split: str, b: int, t: int, batches: int):
    """forge's LOSS / CORRECT / LOSS_COUNT registers: per batch the row losses summed in f64,
    added to an f32 register; returns (LOSS/COUNT, CORRECT/COUNT) in f32."""
    loss = cor = cnt = f32(0.0)
    for _ in range(max(batches, 1)):
        x, y = data.batch(rng, split, b, t)
        ce, hit = ev(params, mesh.put(x, mesh.rows), mesh.put(y, mesh.rows))
        ce, hit = to_host(ce), to_host(hit)
        loss = f32(loss + f32(np.sum(ce.astype(np.float64))))
        cor = f32(cor + f32(int(np.count_nonzero(hit))))
        cnt = f32(cnt + f32(b * t))
    return f32(loss / cnt), f32(cor / cnt)


def _check_adamw_state(st: dict, d, what: str) -> None:
    """forge refuses to mix optimisers on resume / init_optim; accel runs AdamW only, so the checkpoint's
    optimiser state must be AdamW too (init_from without init_optim takes the weights of any checkpoint)."""
    cfg = st.get("config") or {}
    theirs = cfg.get("optimizer", cfg.get("optimiser", "adamw"))
    if theirs != "adamw":
        raise ConfigError(f"{what}: {d} holds {theirs} optimiser state but accel runs optimizer \"adamw\"; "
                          "mixing optimisers is refused (drop init_optim to start fresh state from the weights)")


# ---------------------------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg: TrainConfig, precision: str = "f32", remat: bool = False, devices: int = 0,
                 init: str = "auto", final_link: bool = False, xent: str = "fast", eval_xent: str = "forge"):
        import jax
        self.cfg, self.precision, self.remat, self.final_link = cfg, precision, remat, final_link
        mc = cfg.model
        self.data = Store(cfg.data, mc.vocab_size)
        if self.data.vocab != mc.vocab_size:
            raise ConfigError(f"data vocab {self.data.vocab} != model vocab {mc.vocab_size}")
        self.mesh = Mesh(devices, cfg.batch, warn=lambda m: emit(event("warn", m, 0)))
        if cfg.ignored:
            emit(event("warn", f"ignoring unknown config keys {cfg.ignored}", 0))
        self.specs = mc.param_specs()
        self.rng = Rng(cfg.seed)
        self.step, self.tokens, self.optim_offset = 0, 0, 0
        self.parent_sha: str | None = None
        self.last_loss, self.last_val = f32(np.nan), None
        self.lr_override: float | None = None
        self.paused = self.stop = False
        self._pending = None  # (step, ce_sums) of an update whose loss has not been fetched yet
        # Multi-process: commands change the requested state on process 0; every process takes it from
        # the control bus at the same loop iteration (see sync_control).
        self._bus = ControlBus() if jax.process_count() > 1 else None
        self._req = {"paused": False, "stop": False, "lr_override": None}
        self._bus_pending = None

        params, m, v = self._init_params(init)
        if cfg.resume_from:
            params, m, v = self._resume(cfg.resume_from)
        if cfg.lr_schedule == "wsd" and cfg.warmup_steps + cfg.wsd_decay_steps() > cfg.max_steps:
            emit(event("warn", f"wsd: warmup {cfg.warmup_steps} + decay {cfg.wsd_decay_steps()} exceed max_steps "
                               f"{cfg.max_steps}; the decay starts before warmup ends", self.step))
        r = self.mesh.repl
        self.params = self.mesh.put_tree(params, r)
        self.m = self.mesh.put_tree(m, r)
        self.v = self.mesh.put_tree(v, r)
        self._step = build_step(mc, cfg.batch, cfg.seq_len, cfg.grad_accum, cfg.grad_clip, self.mesh, precision, remat,
                                xent)
        self._eval = build_eval(mc, self.mesh, precision, eval_xent)
        emit(event("info", f"compiled: accel (jax {jax.__version__}) on {self.mesh.n} {self.mesh.platform} device(s) "
                           f"[{self.mesh.kind}], {jax.process_count()} process(es); precision {precision}; remat {remat}; "
                           f"xent {xent}; params {mc.num_params()}; batch {cfg.batch} x seq {cfg.seq_len} x accum "
                           f"{cfg.grad_accum}", 0))

    # ------------------------------------------------------------------ init / resume (trainer.rs:171-330)
    def _init_params(self, init: str):
        cfg, mc = self.cfg, self.cfg.model
        zeros = {n: np.zeros(s, f32) for n, s in self.specs}
        if cfg.init_from:
            path = Path(cfg.init_from) / "model.safetensors"
            w, _, sha = fio.read_safetensors(path)
            params = fio.check_weights(w, self.specs, str(path))
            self.parent_sha = sha
            emit(event("info", f"initialised from {path} (sha256 {sha})", 0))
            m, v = zeros, {n: np.zeros(s, f32) for n, s in self.specs}
            if cfg.init_optim and not cfg.resume_from:
                m, v = self._init_moments(Path(cfg.init_from))
            return params, m, v
        from .model import init_params
        exact = None if init == "auto" else (init == "exact")
        params = init_params(mc, cfg.seed, exact)
        return params, zeros, {n: np.zeros(s, f32) for n, s in self.specs}

    def _init_moments(self, d: Path):
        st = json.loads((d / "state.json").read_text())
        _check_adamw_state(st, d, "init_optim")
        path = d / "optim.safetensors"
        opt, _, _ = fio.read_safetensors(path)
        m, v = {}, {}
        for n, s in self.specs:
            for dst, key in ((m, n), (v, f"v.{n}")):
                if key not in opt:
                    raise ConfigError(f"{path}: missing {key}")
                if tuple(opt[key].shape) != tuple(s):
                    raise ConfigError(f"{path}: {key} has shape {list(opt[key].shape)}, model wants {list(s)}")
                dst[n] = opt[key]
        self.optim_offset = int(st.get("step") or 0) + int(st.get("optim_offset") or 0)
        emit(event("info", f"AdamW moments from {path} ({self.optim_offset} steps behind them)", 0))
        return m, v

    def _resume(self, d: str):
        dp = Path(d)
        st = json.loads((dp / "state.json").read_text())
        if ModelConfig.from_dict(st["config"]["model"]) != self.cfg.model:
            raise ConfigError(f"{d}: model config differs from this run")
        _check_adamw_state(st, d, "resume")
        w, _, _ = fio.read_safetensors(dp / "model.safetensors")
        opt, _, _ = fio.read_safetensors(dp / "optim.safetensors")
        params, m, v = {}, {}, {}
        for n, _s in self.specs:
            for dst, src, key in ((params, w, n), (m, opt, n), (v, opt, f"v.{n}")):
                if key not in src:
                    raise ConfigError(f"{d}: missing tensor for {key}")
                dst[n] = src[key]
        self.step = int(st.get("step") or 0)
        self.tokens = int(st.get("tokens") or 0)
        self.rng = Rng(state=st["rng_state"])
        self.parent_sha = st.get("parent_sha256")
        self.optim_offset = int(st.get("optim_offset") or 0)
        emit(event("info", f"resumed from {d} at step {self.step}", self.step))
        return params, m, v

    # ------------------------------------------------------------------ one update (trainer.rs:387-410)
    def lr_now(self) -> np.float32:
        base = self.cfg.lr_at(self.step)
        if self.lr_override is None:
            return base
        return f32(f32(self.lr_override) * base / max(self.cfg.lr, f32(1e-12)))

    def train_step(self):
        cfg = self.cfg
        lr = self.lr_now()
        xs, ys = [], []
        for _ in range(cfg.grad_accum):
            x, y = self.data.batch(self.rng, "train", cfg.batch, cfg.seq_len)
            xs.append(x)
            ys.append(y)
        xs, ys = np.stack(xs), np.stack(ys)
        self.step += 1
        t = f32(self.optim_offset + self.step)
        bc1 = f32(f32(1.0) - powf(B1, t))
        bc2s = np.sqrt(f32(f32(1.0) - powf(B2, t)), dtype=f32)
        hyper = np.array([f32(lr / bc1), bc2s, f32(f32(1.0) - lr * cfg.weight_decay), 0.0], f32)
        self.params, self.m, self.v, ce_sums, norm = self._step(
            self.params, self.m, self.v, self.mesh.put(xs, self.mesh.micro_rows), self.mesh.put(ys, self.mesh.micro_rows),
            self.mesh.put(hyper, self.mesh.repl))
        self.tokens += cfg.batch * cfg.seq_len * cfg.grad_accum
        return lr, ce_sums, norm

    def loss_of(self, ce_sums) -> np.float32:
        """LOSS / LOSS_COUNT: micro-batch sums added to an f32 register, divided by the f32 row count."""
        cs = replicated_to_host(ce_sums)
        loss = cnt = f32(0.0)
        for s in cs:
            loss = f32(loss + f32(s))
            cnt = f32(cnt + f32(self.cfg.batch * self.cfg.seq_len))
        return f32(loss / cnt)

    def _flush(self) -> None:
        """Fetch the pending update's loss (last_loss, and forge's non-finite-loss error event)."""
        if self._pending is None:
            return
        step, ce_sums = self._pending
        self._pending = None
        self.last_loss = self.loss_of(ce_sums)
        if not np.isfinite(self.last_loss):
            emit(event("error", f"non-finite loss {_rust_f32(self.last_loss)} at step {step}", step))

    def evaluate(self):
        cfg = self.cfg
        rng = Rng((cfg.seed ^ 0xE7A1) & fio.M64)  # a fresh eval stream every time; the train RNG is untouched
        return eval_windows(self._eval, self.params, self.mesh, self.data, rng, "val", cfg.batch, cfg.seq_len,
                            cfg.eval_batches)

    # ------------------------------------------------------------------ files (trainer.rs:473-561)
    def run_dir(self) -> str:
        return fio.join(self.cfg.out_dir, self.cfg.run)

    def save(self, d: str, trunk: bool = False) -> str:
        self._flush()
        p = {n: replicated_to_host(x) for n, x in self.params.items()}
        m = {n: replicated_to_host(x) for n, x in self.m.items()}
        v = {n: replicated_to_host(x) for n, x in self.v.items()}
        if not is_main():
            return ""
        state = {"step": self.step, "config": self.cfg.to_dict(), "rng_state": self.rng.state(),
                 "parent_sha256": self.parent_sha, "loss": float(self.last_loss),
                 "val_loss": None if self.last_val is None else float(self.last_val), "tokens": self.tokens}
        if self.optim_offset > 0:
            state["optim_offset"] = self.optim_offset
        if trunk:
            state["trunk"] = True
        return fio.save_checkpoint(d, self.step, p, m, v, state)

    def checkpoint(self) -> str:
        d = fio.join(self.run_dir(), f"step-{self.step:06d}")
        sha = self.save(d)
        if is_main():
            fio.write_latest(self.run_dir(), d, self.step, sha)
        emit({"type": "checkpoint", "path": d, "sha256": sha, "step": self.step, "ts": now()})
        return d

    def save_trunk(self) -> None:
        trunk = fio.join(self.run_dir(), "TRUNK")
        if os.path.exists(trunk):
            return
        tmp = fio.join(self.run_dir(), "TRUNK.tmp")
        if is_main() and os.path.exists(tmp):
            shutil.rmtree(tmp)
        self.save(tmp, trunk=True)
        if is_main():
            os.rename(tmp, trunk)
        emit({"type": "trunk", "path": trunk, "step": self.step, "ts": now()})

    # ------------------------------------------------------------------ control (trainer.rs:563-608)
    def _request(self, key: str, value) -> None:
        """A control change: applied now (one process), or handed to the control bus (multi-process)."""
        if self._bus is None:
            setattr(self, key, value)
        else:
            self._req[key] = value

    def handle(self, line: str) -> None:
        """forge's handle(): the same events, including "applied <cmd>" after a rejected value."""
        try:
            v = json.loads(line)
        except json.JSONDecodeError as e:
            return emit(event("warn", f"unparseable command: {e}", self.step))
        cmd = v.get("cmd", "") if isinstance(v, dict) else ""
        cmd = cmd if isinstance(cmd, str) else ""
        x = v.get("value") if isinstance(v, dict) else None
        if cmd == "pause":
            self._request("paused", True)
        elif cmd == "resume":
            self._request("paused", False)
        elif cmd == "stop":
            self._request("stop", True)
        elif cmd == "set_lr":
            if isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 < x < 1.0:  # as_f64()
                self._request("lr_override", float(f32(x)))
            else:
                emit(event("warn", "set_lr needs 0 < value < 1", self.step))
        elif cmd == "set_threads":  # forge's validation and events; the thread count itself means nothing to XLA
            if not (isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 1024):  # as_u64()
                emit(event("warn", "set_threads needs 1..=1024", self.step))
        elif cmd in ("autotune", "swap_kernel"):
            emit(event("info", f"{cmd} has no meaning on XLA devices; ignored", self.step))
        elif cmd == "checkpoint":
            try:
                self.checkpoint()
            except OSError as e:
                emit(event("error", f"checkpoint failed: {e}", self.step))
        else:
            return emit(event("warn", f"unknown command {json.dumps(cmd)}", self.step))
        emit(event("info", f"applied {cmd}", self.step))

    def sync_control(self) -> None:
        """Multi-process only: send process 0's requested state (pause, stop, set_lr) through the control
        bus and apply the state sent one iteration earlier. Every process calls this at the top of every
        loop iteration, so all of them apply the same state at the same iteration and step: stop ends
        every process's loop before the same update (the final eval is a collective), and set_lr changes
        the replicated weights' update everywhere. The fetched record was queued behind the update
        dispatched two iterations ago, so the host never waits for the update in flight."""
        out = self._bus.launch(self._req if is_main() else None)
        if self._bus_pending is not None:
            self.paused, self.stop, self.lr_override = self._bus.fetch(self._bus_pending)
        self._bus_pending = out

    # ------------------------------------------------------------------ loop (trainer.rs:621-671)
    def run(self, control: "queue.Queue[str] | None" = None) -> float:
        cfg = self.cfg
        mc = cfg.model
        flops_tok = mc.flops_per_token(cfg.seq_len)
        trunk_step = cfg.trunk_step()
        last_log_t, last_log_tok, last_log_step = time.perf_counter(), self.tokens, self.step
        lr = f32(0.0)
        norm = None
        while self.step < cfg.max_steps and not self.stop:
            while control is not None:
                try:
                    self.handle(control.get_nowait())
                except queue.Empty:
                    break
            if self._bus is not None:
                self.sync_control()
                if self.stop:
                    break
            if self.paused:
                time.sleep(0.2)
                continue
            lr, ce_sums, norm = self.train_step()
            log_now = self.step % max(cfg.log_every, 1) == 0 or self.step == 1
            eval_now = cfg.eval_every > 0 and self.step % cfg.eval_every == 0
            ckpt_now = cfg.ckpt_every > 0 and self.step % cfg.ckpt_every == 0
            trunk_now = trunk_step is not None and trunk_step > 0 and self.step == trunk_step
            # A step's loss is fetched after the next step is queued, so the device never waits for the
            # host's batch sampling; steps that print or save something wait for their own loss.
            self._flush()
            self._pending = (self.step, ce_sums)
            if log_now or eval_now or ckpt_now or trunk_now:
                self._flush()
            if trunk_now:
                self.save_trunk()
            if eval_now:
                vl, acc = self.evaluate()
                self.last_val = vl
                emit({"type": "eval", "step": self.step, "val_loss": float(vl), "val_acc": float(acc), "ts": now()})
            if log_now:
                gn = float(f32(replicated_to_host(norm)))
                t1 = time.perf_counter()
                secs = max(t1 - last_log_t, 1e-9)
                toks = self.tokens - last_log_tok
                nsteps = max(self.step - last_log_step, 1)
                emit({"type": "telemetry", "run": cfg.run, "step": self.step, "tokens": self.tokens,
                      "loss": float(self.last_loss), "val_loss": None if self.last_val is None else float(self.last_val),
                      "lr": float(lr), "grad_norm": gn, "step_ms": secs * 1e3 / nsteps, "tokens_per_s": toks / secs,
                      "tflops": flops_tok * toks / secs / 1e12, "threads": self.mesh.n, "paused": self.paused,
                      "ops": {}, "kernels": {"MATMUL": f"xla-{self.mesh.platform}-{self.precision}"},
                      "hbvm": memory_stats(self.mesh.devices[0]), "comm": None, "ts": now(),
                      "devices": self.mesh.n, "backend": self.mesh.platform, "precision": self.precision})
                last_log_t, last_log_tok, last_log_step = time.perf_counter(), self.tokens, self.step
            if ckpt_now:
                self.checkpoint()
        self._flush()
        vl, acc = self.evaluate()
        self.last_val = vl
        emit({"type": "eval", "step": self.step, "val_loss": float(vl), "val_acc": float(acc), "ts": now()})
        d = self.checkpoint()
        if self.final_link and is_main() and self.step >= cfg.max_steps:  # finished runs only, never a stop
            fio.set_final(self.run_dir(), d)
        emit({"type": "done", "step": self.step, "loss": float(self.last_loss), "val_loss": float(vl), "ts": now()})
        return float(self.last_loss)


def _rust_f32(x) -> str:
    x = float(x)
    return "NaN" if x != x else ("inf" if x > 0 else "-inf")


class ControlBus:
    """Process 0 -> every process, through the devices (multi-host runs). A record is int32
    [paused, stop, has_lr, lr bits]; process 0 contributes it, the others zeros, and a sum over a
    [processes, 4] array sharded by process hands every process the same record."""

    def __init__(self):
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh as JMesh, NamedSharding, PartitionSpec as PS
        devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
        mesh = JMesh(np.array(devs).reshape(jax.process_count(), -1), ("proc", "local"))
        self.shape = (jax.process_count(), 4)
        self.index = jax.process_index()
        self.sharding = NamedSharding(mesh, PS("proc"))
        self._sum = jax.jit(lambda a: jnp.sum(a, axis=0), out_shardings=NamedSharding(mesh, PS()))

    @staticmethod
    def encode(req: dict | None) -> np.ndarray:
        if req is None:
            return np.zeros(4, np.int32)
        lr = req["lr_override"]
        bits = int(np.array(lr if lr is not None else 0.0, np.float32).view(np.int32))
        return np.array([bool(req["paused"]), bool(req["stop"]), lr is not None, bits], np.int32)

    def launch(self, req: dict | None):
        import jax
        full = np.zeros(self.shape, np.int32)
        full[self.index] = self.encode(req)
        return self._sum(jax.make_array_from_callback(self.shape, self.sharding, lambda idx: full[idx]))

    @staticmethod
    def fetch(out) -> tuple[bool, bool, float | None]:
        r = np.asarray(out.addressable_data(0))
        lr = float(np.array(r[3], np.int32).view(np.float32)) if r[2] else None
        return bool(r[0]), bool(r[1]), lr


def stdin_control() -> "queue.Queue[str]":
    """forge's stdin reader: one JSON command per line, applied before the next update."""
    q: "queue.Queue[str]" = queue.Queue()

    def reader():
        try:
            for line in sys.stdin:
                q.put(line.strip())
        except (OSError, ValueError):
            pass

    threading.Thread(target=reader, daemon=True).start()
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="accel.train", description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="a forge run config (JSON)")
    ap.add_argument("--resume", help="resume from this checkpoint dir of the run (forge train --resume)")
    ap.add_argument("--precision", choices=("f32", "bf16"), default=os.environ.get("ACCEL_PRECISION", "f32"),
                    help="GEMM arithmetic: f32 (forge's f32 engines) or bf16 inputs with f32 accumulation "
                         "(forge's amx-bf16 engine; the fast path on TPUs and tensor cores). Default $ACCEL_PRECISION or f32")
    ap.add_argument("--remat", action="store_true", default=os.environ.get("ACCEL_REMAT", "") not in ("", "0"),
                    help="recompute each layer in the backward pass (less memory, ~1/3 more compute)")
    ap.add_argument("--devices", type=int, default=int(os.environ.get("ACCEL_DEVICES", "0") or 0),
                    help="use the first N devices (0 = all)")
    ap.add_argument("--init", choices=("auto", "exact", "fast"), default="auto",
                    help="fresh init: forge's generator bit for bit (exact; ~1.7 s per million weights), numpy (fast), "
                         "or exact up to 150M parameters (auto)")
    ap.add_argument("--xent", choices=("fast", "forge"), default=os.environ.get("ACCEL_XENT", "fast"),
                    help="train-step loss value: tree-summed softmax denominator (fast) or forge's sequential f32 sum "
                         "(forge; the reported loss then matches forge's to the logit noise). Same gradient either way; "
                         "evals always use forge's")
    ap.add_argument("--final", action="store_true",
                    help="point <run>/FINAL at the last checkpoint when the run reaches max_steps (never for a "
                         "stopped run). Default off, like forge: scripts/longrun.py links FINAL itself")
    ap.add_argument("--no-final", action="store_true", help="do not write FINAL (the default; overrides --final)")
    ap.add_argument("--threads", help=argparse.SUPPRESS)  # forge CLI flag, meaningless here
    a = ap.parse_args(argv)
    try:
        init_distributed()
        raw = json.loads(Path(a.config).read_text())
        cfg = TrainConfig.from_dict(raw)
        if a.resume:
            cfg.resume_from = a.resume
        tr = Trainer(cfg, a.precision, a.remat, a.devices, a.init, final_link=a.final and not a.no_final, xent=a.xent)
        tr.run(stdin_control() if is_main() else None)  # multi-host: process 0 reads the commands
    except KeyboardInterrupt:
        return 130
    except (ConfigError, ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"accel: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
