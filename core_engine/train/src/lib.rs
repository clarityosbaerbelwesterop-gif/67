//! forge-train: SCP-1-compatible transformer training on the forge ISA.
//! Contract: docs/DESIGN.md §8–9.

pub mod ckpt;
pub mod config;
pub mod data;
pub mod diloco;
pub mod merge;
pub mod model;
pub mod rng;
pub mod trainer;

use ckpt::Tensor;
use config::ModelConfig;
use forge_isa::{compile, CompileOptions, Machine};
use std::collections::BTreeMap;

/// Autoregressive sampling with a fixed-length causal window (positions past
/// the prompt are padding and cannot influence earlier positions).
pub struct Sampler {
    machine: Machine,
    lp: forge_isa::Loaded,
    cfg: ModelConfig,
    ctx: usize,
}

impl Sampler {
    pub fn new(
        cfg: &ModelConfig,
        weights: &BTreeMap<String, Tensor>,
        ctx: usize,
        threads: usize,
    ) -> Result<Sampler, String> {
        let ctx = ctx.min(cfg.max_seq_len);
        let (prog, rep) = compile(
            model::build(cfg, 1, ctx, model::Head::Logits),
            CompileOptions::default(),
        )
        .map_err(|e| e.to_string())?;
        let words = cfg.num_params()
            + rep.transient_words_peak as usize
            + ctx * (cfg.vocab_size + 2)
            + (1 << 20);
        let mut machine = Machine::new(words, threads);
        let lp = machine.load(prog).map_err(|e| e.to_string())?;
        for p in model::params(cfg) {
            let t = weights
                .get(&p.name)
                .ok_or(format!("weights lack {}", p.name))?;
            if t.shape != p.shape {
                return Err(format!("{}: shape {:?} != {:?}", p.name, t.shape, p.shape));
            }
            machine
                .persistent_mut(&p.name)
                .ok_or("unbound")?
                .copy_from_slice(&t.data);
        }
        Ok(Sampler {
            machine,
            lp,
            cfg: cfg.clone(),
            ctx,
        })
    }

    /// Logits for the position after `ids` (the last `ctx` tokens are used).
    pub fn next_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let window = &ids[ids.len().saturating_sub(self.ctx)..];
        let tk = self
            .machine
            .persistent_mut(&model::tokens_name(self.ctx))
            .ok_or("tokens unbound")?;
        tk.fill(0.0);
        tk.iter_mut().zip(window).for_each(|(d, s)| *d = *s as f32);
        self.machine.run(&mut self.lp).map_err(|e| e.to_string())?;
        let v = self.cfg.vocab_size;
        let pos = window.len().max(1) - 1;
        Ok(self
            .machine
            .persistent(&model::logits_name(self.ctx))
            .ok_or("logits unbound")?[pos * v..(pos + 1) * v]
            .to_vec())
    }

    /// Greedy when `temperature == 0`, otherwise temperature + top-k sampling.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        seed: u64,
        stop: Option<u32>,
    ) -> Result<Vec<u32>, String> {
        let mut rng = rng::Rng::new(seed);
        let mut ids = prompt.to_vec();
        for _ in 0..max_new {
            let logits = self.next_logits(&ids)?;
            let next = if temperature <= 0.0 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            } else {
                let mut idx: Vec<usize> = (0..logits.len()).collect();
                idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
                idx.truncate(top_k.max(1));
                let mx = logits[idx[0]];
                let w: Vec<f64> = idx
                    .iter()
                    .map(|&i| (((logits[i] - mx) / temperature) as f64).exp())
                    .collect();
                let mut r = rng.next_f64() * w.iter().sum::<f64>();
                let mut pick = idx[idx.len() - 1];
                for (k, &i) in idx.iter().enumerate() {
                    r -= w[k];
                    if r <= 0.0 {
                        pick = i;
                        break;
                    }
                }
                pick as u32
            };
            ids.push(next);
            if Some(next) == stop {
                break;
            }
        }
        Ok(ids[prompt.len()..].to_vec())
    }
}

/// Mean loss and next-token accuracy of `weights` on `batches` windows of a split.
pub fn evaluate_weights(
    cfg: &ModelConfig,
    weights: &BTreeMap<String, Tensor>,
    data_spec: &str,
    split: data::Split,
    (batch, seq, batches): (usize, usize, usize),
    seed: u64,
    threads: usize,
) -> Result<(f32, f32), String> {
    let data = data::open(data_spec, cfg.vocab_size)?;
    let (prog, rep) = compile(
        model::build(cfg, batch, seq, model::Head::Eval),
        CompileOptions::default(),
    )
    .map_err(|e| e.to_string())?;
    let words = cfg.num_params() + rep.transient_words_peak as usize + 2 * batch * seq + (1 << 20);
    let mut m = Machine::new(words, threads);
    let mut lp = m.load(prog).map_err(|e| e.to_string())?;
    for p in model::params(cfg) {
        let t = weights
            .get(&p.name)
            .ok_or(format!("weights lack {}", p.name))?;
        m.persistent_mut(&p.name)
            .ok_or("unbound")?
            .copy_from_slice(&t.data);
    }
    let mut rng = rng::Rng::new(seed);
    let n = batch * seq;
    for _ in 0..batches.max(1) {
        let (x, y) = data.batch(&mut rng, split, batch, seq);
        m.persistent_mut(&model::tokens_name(n))
            .ok_or("unbound")?
            .iter_mut()
            .zip(&x)
            .for_each(|(d, s)| *d = *s as f32);
        m.persistent_mut(&model::targets_name(n))
            .ok_or("unbound")?
            .iter_mut()
            .zip(&y)
            .for_each(|(d, s)| *d = *s as f32);
        m.run(&mut lp).map_err(|e| e.to_string())?;
    }
    let cnt = m.regs[forge_isa::regs::LOSS_COUNT as usize];
    Ok((
        m.regs[forge_isa::regs::LOSS as usize] / cnt,
        m.regs[forge_isa::regs::CORRECT as usize] / cnt,
    ))
}

#[cfg(test)]
mod tests;
