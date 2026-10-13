//! Model and run configuration (SCP `ModelConfig` / `TrainConfig` equivalents).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// SwiGLU width; `None` derives SCP's 2/3·4·dim rounded up to 64.
    #[serde(default)]
    pub ffn_hidden: Option<usize>,
    pub max_seq_len: usize,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_eps")]
    pub norm_eps: f32,
}

fn default_theta() -> f32 {
    10_000.0
}
fn default_eps() -> f32 {
    1e-5
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn ffn(&self) -> usize {
        self.ffn_hidden.unwrap_or_else(|| {
            let hidden = 2 * (4 * self.dim) / 3;
            64 * hidden.div_ceil(64)
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.dim.is_multiple_of(self.n_heads) {
            return Err(format!(
                "dim {} not divisible by n_heads {}",
                self.dim, self.n_heads
            ));
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!(
                "n_heads {} not divisible by n_kv_heads {}",
                self.n_heads, self.n_kv_heads
            ));
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err("head_dim must be even for RoPE".into());
        }
        if self.vocab_size == 0 || self.n_layers == 0 || self.max_seq_len == 0 {
            return Err("vocab_size, n_layers and max_seq_len must be positive".into());
        }
        Ok(())
    }

    /// Exact parameter count with tied embeddings (matches SCP `estimate_num_params`).
    pub fn num_params(&self) -> usize {
        let (d, hd, f) = (self.dim, self.head_dim(), self.ffn());
        let attn = d * self.n_heads * hd * 2 + d * self.n_kv_heads * hd * 2;
        let block = attn + 3 * d * f + 2 * d;
        self.vocab_size * d + self.n_layers * block + d
    }

    /// Training FLOPs per token as executed by this engine: 6·N for the dense
    /// layers (forward + backward, tied head included) plus 12·L·d·T for the
    /// attention products (full T×T products; the causal half is not skipped).
    pub fn flops_per_token(&self, seq: usize) -> f64 {
        6.0 * self.num_params() as f64 + 12.0 * (self.n_layers * self.dim * seq) as f64
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    pub run: String,
    pub model: ModelConfig,
    /// SCP store meta file (`train.meta.json`); `synthetic:<period>` for a toy stream.
    pub data: String,
    pub batch: usize,
    pub seq_len: usize,
    #[serde(default = "one")]
    pub grad_accum: usize,
    pub max_steps: usize,
    pub lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    #[serde(default = "default_wd")]
    pub weight_decay: f32,
    #[serde(default = "one_f")]
    pub grad_clip: f32,
    #[serde(default)]
    pub seed: u64,
    /// Continue from this checkpoint directory (children of a shared base).
    #[serde(default)]
    pub init_from: Option<String>,
    /// Resume an interrupted run from its own checkpoint: weights, optimiser
    /// state (AdamW moments, Muon momentum), step, data RNG and lineage are
    /// restored. The checkpoint must come from the same `optimizer`.
    #[serde(default)]
    pub resume_from: Option<String>,
    pub out_dir: String,
    #[serde(default = "ten")]
    pub log_every: usize,
    #[serde(default)]
    pub eval_every: usize,
    #[serde(default = "ten")]
    pub eval_batches: usize,
    #[serde(default)]
    pub ckpt_every: usize,
    #[serde(default)]
    pub threads: usize,
    /// HBVM size in MiB (0 = derived from the model).
    #[serde(default)]
    pub hbvm_mib: usize,
    /// Run the JIT autotuner before the first step.
    #[serde(default = "yes")]
    pub autotune: bool,
    /// Learning-rate schedule after the linear warmup: "cosine" (SCP, decay to
    /// min_lr over the whole run), "wsd" (stable at lr, cosine-shaped decay to
    /// min_lr over the last `decay_steps`) or "constant" (lr). The defaults of
    /// the three schedule fields are omitted from state.json.
    #[serde(default = "cosine", skip_serializing_if = "is_cosine")]
    pub lr_schedule: String,
    /// Length of the "wsd" decay; 0 = round(0.2 · max_steps).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub decay_steps: usize,
    /// With `init_from`: also load its optimiser state (`optim.safetensors`), e.g.
    /// to continue a previous generation's TRUNK without re-warming. Steps and
    /// schedule start at 0; bias correction counts the moments' own steps.
    /// The checkpoint must come from the same `optimizer` (no mixing).
    #[serde(default, skip_serializing_if = "is_false")]
    pub init_optim: bool,
    /// "adamw" (every parameter on AdamW) or "muon": Muon on the 2-D hidden
    /// matrices of the blocks (attention wq/wk/wv/wo, SwiGLU w1/w3/w2), AdamW
    /// with `lr` and `weight_decay` on the rest (token embedding = tied head,
    /// norms). The defaults of the five optimiser fields are omitted from
    /// state.json, so AdamW runs write the same checkpoints as before.
    #[serde(default = "adamw", skip_serializing_if = "is_adamw")]
    pub optimizer: String,
    /// Muon peak learning rate. It follows the shape of `lr`'s schedule (and
    /// of a `set_lr` override): lr_muon(step) = lr(step) · muon_lr / lr.
    #[serde(default = "muon_lr", skip_serializing_if = "is_muon_lr")]
    pub muon_lr: f32,
    /// Muon momentum β (Nesterov).
    #[serde(default = "muon_momentum", skip_serializing_if = "is_muon_momentum")]
    pub muon_momentum: f32,
    /// Newton–Schulz iterations of the Muon orthogonalisation.
    #[serde(default = "muon_ns_steps", skip_serializing_if = "is_muon_ns_steps")]
    pub muon_ns_steps: usize,
    /// Decoupled weight decay of the Muon matrices, W -= lr_muon · wd · W (0 = off).
    #[serde(default, skip_serializing_if = "is_zero_f")]
    pub muon_weight_decay: f32,
}

fn one() -> usize {
    1
}
fn one_f() -> f32 {
    1.0
}
fn ten() -> usize {
    10
}
fn yes() -> bool {
    true
}
fn default_wd() -> f32 {
    0.1
}
fn cosine() -> String {
    "cosine".into()
}
fn is_cosine(s: &str) -> bool {
    s == "cosine"
}
fn is_zero(n: &usize) -> bool {
    *n == 0
}
fn is_false(b: &bool) -> bool {
    !*b
}
fn adamw() -> String {
    "adamw".into()
}
fn is_adamw(s: &str) -> bool {
    s == "adamw"
}
fn muon_lr() -> f32 {
    0.02
}
fn is_muon_lr(x: &f32) -> bool {
    *x == muon_lr()
}
fn muon_momentum() -> f32 {
    0.95
}
fn is_muon_momentum(x: &f32) -> bool {
    *x == muon_momentum()
}
fn muon_ns_steps() -> usize {
    5
}
fn is_muon_ns_steps(n: &usize) -> bool {
    *n == muon_ns_steps()
}
fn is_zero_f(x: &f32) -> bool {
    *x == 0.0
}

impl TrainConfig {
    /// Learning rate of update `step` (counts from 0). Every schedule starts
    /// with SCP's linear warmup. "cosine" is SCP `cosine_lr`: cosine decay to
    /// min_lr at max_steps. "constant" stays at lr. "wsd" stays at lr until
    /// `trunk_step` and then decays over the last D = `wsd_decay_steps` updates:
    /// update i of the decay (i = 1..=D) uses min_lr + (lr - min_lr)(1 + cos(πi/D))/2,
    /// so every decayed update is below lr and the last one runs at min_lr.
    pub fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.lr * (step + 1) as f32 / self.warmup_steps.max(1) as f32;
        }
        match self.lr_schedule.as_str() {
            "constant" => return self.lr,
            "wsd" => {
                let d = self.wsd_decay_steps();
                if step >= self.max_steps {
                    return self.min_lr;
                }
                if step + d < self.max_steps {
                    return self.lr;
                }
                // i = step + 1 - (max_steps - D), in 1..=D (D >= 1 here).
                let progress = (step + 1 + d - self.max_steps) as f32 / d as f32;
                return self.min_lr
                    + 0.5
                        * (self.lr - self.min_lr)
                        * (1.0 + (std::f32::consts::PI * progress).cos());
            }
            _ => {}
        }
        if step >= self.max_steps {
            return self.min_lr;
        }
        let progress =
            (step - self.warmup_steps) as f32 / (self.max_steps - self.warmup_steps).max(1) as f32;
        self.min_lr
            + 0.5 * (self.lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }

    /// D of the "wsd" schedule: `decay_steps`, or round(0.2 · max_steps) when 0.
    pub fn wsd_decay_steps(&self) -> usize {
        if self.decay_steps > 0 {
            self.decay_steps
        } else {
            (0.2 * self.max_steps as f64).round() as usize
        }
    }

    /// "wsd" only: the step count after the last stable update (max_steps - D),
    /// where the trainer saves the run's TRUNK.
    pub fn trunk_step(&self) -> Option<usize> {
        (self.lr_schedule == "wsd")
            .then(|| self.max_steps.saturating_sub(self.wsd_decay_steps()))
    }

    pub fn validate(&self) -> Result<(), String> {
        self.model.validate()?;
        if self.seq_len > self.model.max_seq_len {
            return Err("seq_len exceeds max_seq_len".into());
        }
        if self.batch == 0 || self.seq_len < 2 || self.grad_accum == 0 || self.max_steps == 0 {
            return Err("batch, seq_len (>= 2), grad_accum and max_steps must be positive".into());
        }
        if !self
            .run
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err("run name must be [A-Za-z0-9_-]".into());
        }
        if !["cosine", "wsd", "constant"].contains(&self.lr_schedule.as_str()) {
            return Err(format!(
                "lr_schedule must be cosine|wsd|constant, not {:?}",
                self.lr_schedule
            ));
        }
        if self.init_optim && self.init_from.is_none() {
            return Err("init_optim needs init_from".into());
        }
        match self.optimizer.as_str() {
            "adamw" => {
                // Catches e.g. a misspelt "optimizer" key, which serde ignores.
                if !is_muon_lr(&self.muon_lr)
                    || !is_muon_momentum(&self.muon_momentum)
                    || !is_muon_ns_steps(&self.muon_ns_steps)
                    || !is_zero_f(&self.muon_weight_decay)
                {
                    return Err("muon_* settings need \"optimizer\": \"muon\"".into());
                }
            }
            "muon" => {
                if !(self.lr > 0.0 && self.muon_lr > 0.0 && self.muon_lr.is_finite()) {
                    return Err("optimizer muon needs lr > 0 and muon_lr > 0 (lr_muon = lr(step) · muon_lr / lr)".into());
                }
                if !(0.0..1.0).contains(&self.muon_momentum) {
                    return Err("muon_momentum must be in [0, 1)".into());
                }
                if !(1..=100).contains(&self.muon_ns_steps) {
                    return Err("muon_ns_steps must be in 1..=100".into());
                }
                if !(self.muon_weight_decay >= 0.0 && self.muon_weight_decay.is_finite()) {
                    return Err("muon_weight_decay must be >= 0".into());
                }
            }
            other => {
                return Err(format!("optimizer must be adamw|muon, not {other:?}"));
            }
        }
        Ok(())
    }

    pub fn muon(&self) -> bool {
        self.optimizer == "muon"
    }

    /// Muon learning rate that goes with the AdamW learning rate `lr` of a step.
    pub fn muon_lr_for(&self, lr: f32) -> f32 {
        lr * (self.muon_lr / self.lr)
    }
}
