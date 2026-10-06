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
    /// Resume an interrupted run from its own checkpoint: weights, AdamW
    /// moments, step, data RNG and lineage are restored.
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

impl TrainConfig {
    /// SCP `cosine_lr`: linear warmup, cosine decay to min_lr. `step` counts from 0.
    pub fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.lr * (step + 1) as f32 / self.warmup_steps.max(1) as f32;
        }
        if step >= self.max_steps {
            return self.min_lr;
        }
        let progress =
            (step - self.warmup_steps) as f32 / (self.max_steps - self.warmup_steps).max(1) as f32;
        self.min_lr
            + 0.5 * (self.lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
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
        Ok(())
    }
}
