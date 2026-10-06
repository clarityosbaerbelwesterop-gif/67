//! Training loop with telemetry (stdout JSON lines) and live control (stdin
//! JSON commands). Protocol: docs/DESIGN.md §9.

use crate::ckpt::{self, Tensor};
use crate::config::TrainConfig;
use crate::data::{self, Batches, Split};
use crate::model::{self, Head};
use crate::rng::Rng;
use forge_isa::{compile, regs, CompileOptions, Loaded, Machine, OpStats, Opcode};
use forge_kernels::GemmVariant;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub type Sink = Box<dyn FnMut(Value) + Send>;

/// Line-oriented stdout sink.
pub fn stdout_sink() -> Sink {
    Box::new(|v: Value| {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{v}");
        let _ = out.flush();
    })
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub struct Trainer {
    pub cfg: TrainConfig,
    pub machine: Machine,
    step_lp: Loaded,
    update_lp: Loaded,
    eval_lp: Loaded,
    data: Box<dyn Batches>,
    pub rng: Rng,
    pub step: usize,
    lr_override: Option<f32>,
    paused: bool,
    stop: bool,
    sink: Sink,
    parent_sha: Option<String>,
    last_stats: HashMap<Opcode, OpStats>,
    last_log: Instant,
    /// `tokens_seen` at the last telemetry line (throughput is measured, not assumed).
    log_tokens: u64,
    tokens_seen: u64,
    pub last_loss: f32,
    pub last_val: Option<f32>,
}

fn event(level: &str, msg: impl Into<String>, step: usize) -> Value {
    json!({"type": "event", "level": level, "msg": msg.into(), "step": step, "ts": now()})
}

impl Trainer {
    pub fn new(cfg: TrainConfig, sink: Sink) -> Result<Trainer, String> {
        cfg.validate()?;
        let data = data::open(&cfg.data, cfg.model.vocab_size)?;
        if data.vocab_size() != cfg.model.vocab_size {
            return Err(format!(
                "data vocab {} != model vocab {}",
                data.vocab_size(),
                cfg.model.vocab_size
            ));
        }
        let m = &cfg.model;
        let (b, t) = (cfg.batch, cfg.seq_len);
        let opts = CompileOptions::default();
        let (step_p, step_r) = compile(
            model::build(
                m,
                b,
                t,
                Head::Train {
                    accum: cfg.grad_accum,
                },
            ),
            opts,
        )
        .map_err(|e| e.to_string())?;
        let (eval_p, eval_r) =
            compile(model::build(m, b, t, Head::Eval), opts).map_err(|e| e.to_string())?;
        let (upd_p, _) = compile(
            model::build_update(m, cfg.weight_decay, cfg.grad_clip),
            opts,
        )
        .map_err(|e| e.to_string())?;
        let words = if cfg.hbvm_mib > 0 {
            cfg.hbvm_mib << 18
        } else {
            let p = m.num_params();
            (4 * p
                + step_r.transient_words_peak.max(eval_r.transient_words_peak) as usize
                + 4 * b * t)
                * 11
                / 10
                + (1 << 20)
        };
        let threads = if cfg.threads > 0 {
            cfg.threads
        } else {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        };
        let mut machine = Machine::new(words, threads);
        let step_lp = machine.load(step_p).map_err(|e| e.to_string())?;
        let eval_lp = machine.load(eval_p).map_err(|e| e.to_string())?;
        let update_lp = machine.load(upd_p).map_err(|e| e.to_string())?;
        let mut tr = Trainer {
            rng: Rng::new(cfg.seed),
            cfg,
            machine,
            step_lp,
            update_lp,
            eval_lp,
            data,
            step: 0,
            lr_override: None,
            paused: false,
            stop: false,
            sink,
            parent_sha: None,
            last_stats: HashMap::new(),
            last_log: Instant::now(),
            log_tokens: 0,
            tokens_seen: 0,
            last_loss: f32::NAN,
            last_val: None,
        };
        tr.init_params()?;
        if let Some(dir) = tr.cfg.resume_from.clone() {
            tr.resume(&dir)?;
        }
        let report = json!({
            "type": "event", "level": "info", "step": 0, "ts": now(),
            "msg": format!(
                "compiled: step {} → {} instrs ({} fused, {} dead), transient peak {:.1} MiB vs naive {:.1} MiB; params {}; threads {}; engines {:?}",
                step_r.instrs_in, step_r.instrs_out, step_r.fused_residual, step_r.dead_removed,
                step_r.transient_words_peak as f64 * 4.0 / 1048576.0, step_r.transient_words_naive as f64 * 4.0 / 1048576.0,
                tr.cfg.model.num_params(), tr.machine.threads(),
                forge_kernels::available().iter().map(|v| v.name()).collect::<Vec<_>>()
            ),
        });
        (tr.sink)(report);
        Ok(tr)
    }

    /// SCP initialisation (normal 0.02; wo/w2 scaled by 1/sqrt(2L); norms = 1),
    /// or the weights of `init_from`.
    fn init_params(&mut self) -> Result<(), String> {
        let ps = model::params(&self.cfg.model);
        if let Some(dir) = self.cfg.init_from.clone() {
            let path = Path::new(&dir).join("model.safetensors");
            let (tensors, _) = ckpt::load_safetensors(&path).map_err(|e| e.to_string())?;
            for p in &ps {
                let t = tensors.get(&p.name).ok_or(format!(
                    "{}: missing {}",
                    path.display(),
                    p.name
                ))?;
                if t.shape != p.shape {
                    return Err(format!(
                        "{}: {} has shape {:?}, model wants {:?}",
                        path.display(),
                        p.name,
                        t.shape,
                        p.shape
                    ));
                }
                self.machine
                    .persistent_mut(&p.name)
                    .ok_or("param not bound")?
                    .copy_from_slice(&t.data);
            }
            self.parent_sha = Some(ckpt::sha256_file(&path).map_err(|e| e.to_string())?);
            (self.sink)(event(
                "info",
                format!(
                    "initialised from {} (sha256 {})",
                    path.display(),
                    self.parent_sha.as_ref().unwrap()
                ),
                0,
            ));
            return Ok(());
        }
        let mut rng = Rng::new(self.cfg.seed ^ 0x5EED_1417);
        let scaled = 0.02 / (2.0 * self.cfg.model.n_layers as f32).sqrt();
        for p in &ps {
            let buf = self
                .machine
                .persistent_mut(&p.name)
                .ok_or("param not bound")?;
            if p.shape.len() == 1 {
                buf.fill(1.0);
            } else {
                let std = if p.name.ends_with("wo.weight") || p.name.ends_with("w2.weight") {
                    scaled
                } else {
                    0.02
                };
                buf.iter_mut().for_each(|x| *x = rng.normal() * std);
            }
        }
        Ok(())
    }

    /// Restore weights, optimiser moments, step, RNG and lineage from a checkpoint of this run.
    fn resume(&mut self, dir: &str) -> Result<(), String> {
        let d = Path::new(dir);
        let state: Value = serde_json::from_str(
            &std::fs::read_to_string(d.join("state.json")).map_err(|e| format!("{dir}: {e}"))?,
        )
        .map_err(|e| e.to_string())?;
        let saved: TrainConfig =
            serde_json::from_value(state["config"].clone()).map_err(|e| e.to_string())?;
        if saved.model != self.cfg.model {
            return Err(format!("{dir}: model config differs from this run"));
        }
        let (weights, _) =
            ckpt::load_safetensors(&d.join("model.safetensors")).map_err(|e| e.to_string())?;
        let (opt, _) =
            ckpt::load_safetensors(&d.join("optim.safetensors")).map_err(|e| e.to_string())?;
        for p in model::params(&self.cfg.model) {
            let pairs = [
                (p.name.clone(), weights.get(&p.name)),
                (model::adam_m(&p.name), opt.get(&p.name)),
                (model::adam_v(&p.name), opt.get(&format!("v.{}", p.name))),
            ];
            for (buf, t) in pairs {
                let t = t.ok_or(format!("{dir}: missing tensor for {buf}"))?;
                self.machine
                    .persistent_mut(&buf)
                    .ok_or("not bound")?
                    .copy_from_slice(&t.data);
            }
        }
        self.step = state["step"].as_u64().unwrap_or(0) as usize;
        self.tokens_seen = state["tokens"].as_u64().unwrap_or(0);
        let rs: Vec<u64> =
            serde_json::from_value(state["rng_state"].clone()).map_err(|e| e.to_string())?;
        self.rng = Rng::from_state([rs[0], rs[1], rs[2], rs[3]]);
        self.parent_sha = state["parent_sha256"].as_str().map(str::to_string);
        (self.sink)(event(
            "info",
            format!("resumed from {dir} at step {}", self.step),
            self.step,
        ));
        Ok(())
    }

    fn load_batch(&mut self, split: Split) {
        let n = self.cfg.batch * self.cfg.seq_len;
        let (x, y) = self
            .data
            .batch(&mut self.rng, split, self.cfg.batch, self.cfg.seq_len);
        let tk = self
            .machine
            .persistent_mut(&model::tokens_name(n))
            .expect("tokens bound");
        tk.iter_mut().zip(&x).for_each(|(d, s)| *d = *s as f32);
        let tg = self
            .machine
            .persistent_mut(&model::targets_name(n))
            .expect("targets bound");
        tg.iter_mut().zip(&y).for_each(|(d, s)| *d = *s as f32);
    }

    fn reset_loss_regs(&mut self) {
        for r in [regs::LOSS, regs::CORRECT, regs::LOSS_COUNT] {
            self.machine.regs[r as usize] = 0.0;
        }
    }

    pub fn autotune(&mut self) {
        let reports = self.machine.autotune(&mut self.step_lp);
        let mut votes: HashMap<GemmVariant, usize> = HashMap::new();
        for r in &reports {
            *votes.entry(r.chosen).or_default() += 1;
            (self.sink)(json!({
                "type": "autotune", "op": "MATMUL", "shape": [r.shape.2, r.shape.3, r.shape.4],
                "trans": [r.shape.0, r.shape.1],
                "results": r.results.iter().map(|(v, g)| json!({"variant": v.name(), "gflops": g})).collect::<Vec<_>>(),
                "chosen": r.chosen.name(), "step": self.step, "ts": now(),
            }));
        }
        if let Some((&best, _)) = votes.iter().max_by_key(|(_, n)| **n) {
            self.eval_lp.set_all_variants(best);
            self.machine.set_default_variant(best);
        }
    }

    fn kernel_label(&self) -> String {
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        for v in self.step_lp.variants() {
            *counts.entry(v.name()).or_default() += 1;
        }
        let mut c: Vec<_> = counts.into_iter().collect();
        c.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        c.iter()
            .map(|(n, k)| {
                if c.len() > 1 {
                    format!("{n}×{k}")
                } else {
                    n.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// One optimiser step over `grad_accum` micro-batches. Returns the mean loss.
    pub fn train_step(&mut self) -> Result<f32, String> {
        let lr = self.lr_override.map_or(self.cfg.lr_at(self.step), |peak| {
            peak * self.cfg.lr_at(self.step) / self.cfg.lr.max(1e-12)
        });
        self.reset_loss_regs();
        for _ in 0..self.cfg.grad_accum {
            self.load_batch(Split::Train);
            self.machine
                .run(&mut self.step_lp)
                .map_err(|e| e.to_string())?;
        }
        let loss =
            self.machine.regs[regs::LOSS as usize] / self.machine.regs[regs::LOSS_COUNT as usize];
        self.step += 1;
        self.machine.regs[regs::LR as usize] = lr;
        self.machine.regs[regs::STEP as usize] = self.step as f32;
        self.machine
            .run(&mut self.update_lp)
            .map_err(|e| e.to_string())?;
        self.tokens_seen += (self.cfg.batch * self.cfg.seq_len * self.cfg.grad_accum) as u64;
        self.last_loss = loss;
        Ok(loss)
    }

    /// Mean validation loss and next-token accuracy over `eval_batches` batches.
    pub fn evaluate(&mut self) -> Result<(f32, f32), String> {
        let saved = self.rng.clone();
        let mut rng = Rng::new(self.cfg.seed ^ 0xE7A1);
        std::mem::swap(&mut self.rng, &mut rng);
        self.reset_loss_regs();
        let mut result = Ok(());
        for _ in 0..self.cfg.eval_batches.max(1) {
            self.load_batch(Split::Val);
            if let Err(e) = self.machine.run(&mut self.eval_lp) {
                result = Err(e.to_string());
                break;
            }
        }
        self.rng = saved;
        result?;
        let cnt = self.machine.regs[regs::LOSS_COUNT as usize];
        Ok((
            self.machine.regs[regs::LOSS as usize] / cnt,
            self.machine.regs[regs::CORRECT as usize] / cnt,
        ))
    }

    fn telemetry(&mut self, step_ms: f64) {
        let mut ops = serde_json::Map::new();
        let mut flops = 0.0;
        for (op, s) in self.machine.op_stats() {
            let prev = self.last_stats.get(&op).copied().unwrap_or_default();
            let d = OpStats {
                calls: s.calls - prev.calls,
                ns: s.ns - prev.ns,
                flops: s.flops - prev.flops,
                bytes: s.bytes - prev.bytes,
            };
            flops += d.flops;
            ops.insert(
                op.name().into(),
                json!({"calls": d.calls, "ns": d.ns, "flops": d.flops, "bytes": d.bytes}),
            );
            self.last_stats.insert(op, s);
        }
        let secs = self.last_log.elapsed().as_secs_f64().max(1e-9);
        self.last_log = Instant::now();
        let toks = (self.tokens_seen - self.log_tokens) as f64;
        self.log_tokens = self.tokens_seen;
        let h = self.machine.hbvm_stats();
        let gn = self.machine.regs[regs::SUMSQ as usize].max(0.0).sqrt();
        let kernels = self.kernel_label();
        let v = json!({
            "type": "telemetry", "run": self.cfg.run, "step": self.step, "tokens": self.tokens_seen,
            "loss": self.last_loss, "val_loss": self.last_val, "lr": self.machine.regs[regs::LR as usize],
            "grad_norm": gn, "step_ms": step_ms, "tokens_per_s": toks / secs, "tflops": flops / secs / 1e12,
            "threads": self.machine.threads(), "paused": self.paused, "ops": ops,
            "kernels": {"MATMUL": kernels},
            "hbvm": {"capacity": h.capacity * 4, "used": h.used * 4, "peak": h.peak * 4, "live": h.live,
                     "allocs": h.allocs, "frees": h.frees, "fragmentation": h.fragmentation},
            "comm": null, "ts": now(),
        });
        (self.sink)(v);
    }

    pub fn checkpoint(&mut self) -> Result<PathBuf, String> {
        let dir = Path::new(&self.cfg.out_dir)
            .join(&self.cfg.run)
            .join(format!("step-{:06}", self.step));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let ps = model::params(&self.cfg.model);
        let collect = |m: &Machine, f: &dyn Fn(&str) -> String| -> BTreeMap<String, Tensor> {
            ps.iter()
                .map(|p| {
                    (
                        p.name.clone(),
                        Tensor {
                            shape: p.shape.clone(),
                            data: m.persistent(&f(&p.name)).expect("bound").to_vec(),
                        },
                    )
                })
                .collect()
        };
        let mut meta = BTreeMap::new();
        meta.insert("format".to_string(), "pt".to_string());
        meta.insert("step".to_string(), self.step.to_string());
        let model_sha = ckpt::save_safetensors(
            &dir.join("model.safetensors"),
            &collect(&self.machine, &|n| n.to_string()),
            &meta,
        )
        .map_err(|e| e.to_string())?;
        let mut opt = collect(&self.machine, &|n| model::adam_m(n));
        opt.extend(
            collect(&self.machine, &|n| model::adam_v(n))
                .into_iter()
                .map(|(k, v)| (format!("v.{k}"), v)),
        );
        ckpt::save_safetensors(&dir.join("optim.safetensors"), &opt, &meta)
            .map_err(|e| e.to_string())?;
        let state = json!({
            "step": self.step, "config": self.cfg, "rng_state": self.rng.state(), "model_sha256": model_sha,
            "parent_sha256": self.parent_sha, "loss": self.last_loss, "val_loss": self.last_val, "tokens": self.tokens_seen,
        });
        std::fs::write(
            dir.join("state.json"),
            serde_json::to_string_pretty(&state).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        let latest = Path::new(&self.cfg.out_dir)
            .join(&self.cfg.run)
            .join("latest.json");
        std::fs::write(
            &latest,
            json!({"dir": dir, "step": self.step, "model_sha256": model_sha}).to_string(),
        )
        .map_err(|e| e.to_string())?;
        (self.sink)(
            json!({"type": "checkpoint", "path": dir, "sha256": model_sha, "step": self.step, "ts": now()}),
        );
        Ok(dir)
    }

    fn handle(&mut self, line: &str) {
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                return (self.sink)(event(
                    "warn",
                    format!("unparseable command: {e}"),
                    self.step,
                ))
            }
        };
        let cmd = v["cmd"].as_str().unwrap_or("");
        let step = self.step;
        match cmd {
            "pause" => self.paused = true,
            "resume" => self.paused = false,
            "stop" => self.stop = true,
            "set_lr" => match v["value"].as_f64() {
                Some(x) if x > 0.0 && x < 1.0 => self.lr_override = Some(x as f32),
                _ => (self.sink)(event("warn", "set_lr needs 0 < value < 1", step)),
            },
            "set_threads" => match v["value"].as_u64() {
                Some(n) if (1..=1024).contains(&n) => self.machine.set_threads(n as usize),
                _ => (self.sink)(event("warn", "set_threads needs 1..=1024", step)),
            },
            "swap_kernel" => match v["variant"].as_str().and_then(GemmVariant::from_name) {
                Some(var) if forge_kernels::available().contains(&var) => {
                    self.step_lp.set_all_variants(var);
                    self.eval_lp.set_all_variants(var);
                }
                _ => (self.sink)(event(
                    "warn",
                    "swap_kernel: unknown or unavailable variant",
                    step,
                )),
            },
            "autotune" => self.autotune(),
            "checkpoint" => {
                if let Err(e) = self.checkpoint() {
                    (self.sink)(event("error", format!("checkpoint failed: {e}"), step));
                }
            }
            other => return (self.sink)(event("warn", format!("unknown command {other:?}"), step)),
        }
        (self.sink)(event("info", format!("applied {cmd}"), step));
    }

    fn drain(&mut self, control: Option<&Receiver<String>>) {
        let Some(rx) = control else { return };
        loop {
            match rx.try_recv() {
                Ok(line) => self.handle(&line),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return,
            }
        }
    }

    /// Run to `max_steps` (or until stopped). Returns the final train loss.
    pub fn run(&mut self, control: Option<Receiver<String>>) -> Result<f32, String> {
        if self.cfg.autotune {
            self.autotune();
        }
        self.machine.reset_stats();
        self.last_log = Instant::now();
        self.log_tokens = self.tokens_seen;
        while self.step < self.cfg.max_steps && !self.stop {
            self.drain(control.as_ref());
            if self.paused {
                std::thread::sleep(std::time::Duration::from_millis(200));
                continue;
            }
            let t0 = Instant::now();
            let loss = self.train_step()?;
            let step_ms = t0.elapsed().as_secs_f64() * 1e3;
            if !loss.is_finite() {
                (self.sink)(event(
                    "error",
                    format!("non-finite loss {loss} at step {}", self.step),
                    self.step,
                ));
            }
            if self.cfg.eval_every > 0 && self.step.is_multiple_of(self.cfg.eval_every) {
                let (vl, acc) = self.evaluate()?;
                self.last_val = Some(vl);
                (self.sink)(
                    json!({"type": "eval", "step": self.step, "val_loss": vl, "val_acc": acc, "ts": now()}),
                );
            }
            if self.step.is_multiple_of(self.cfg.log_every.max(1)) || self.step == 1 {
                self.telemetry(step_ms);
            }
            if self.cfg.ckpt_every > 0 && self.step.is_multiple_of(self.cfg.ckpt_every) {
                self.checkpoint()?;
            }
        }
        let (vl, acc) = self.evaluate()?;
        self.last_val = Some(vl);
        (self.sink)(
            json!({"type": "eval", "step": self.step, "val_loss": vl, "val_acc": acc, "ts": now()}),
        );
        self.checkpoint()?;
        (self.sink)(
            json!({"type": "done", "step": self.step, "loss": self.last_loss, "val_loss": vl, "ts": now()}),
        );
        Ok(self.last_loss)
    }

    pub fn param(&self, name: &str) -> Option<&[f32]> {
        self.machine.persistent(name)
    }
}

/// Spawn a stdin reader thread that forwards command lines.
pub fn stdin_control() -> Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut line = String::new();
        while stdin.read_line(&mut line).map(|n| n > 0).unwrap_or(false) {
            if tx.send(line.trim().to_string()).is_err() {
                break;
            }
            line.clear();
        }
    });
    rx
}
