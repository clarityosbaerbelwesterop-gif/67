//! KV-cache decoding. The ISA programs process fixed windows, so sampling
//! with them recomputes the whole window for every new token (O(T²) work per
//! sequence). The decoder keeps each layer's rotated keys and values and runs
//! only the new positions: a prompt is one block of GEMMs, every generated
//! token one row. Same equations, parameter layout and kernels as
//! `model::build` (SCP-1); `tests.rs` checks parity with the ISA forward pass.

use crate::ckpt::Tensor;
use crate::config::ModelConfig;
use crate::{model, pick, rng};
use forge_kernels::{gemm, GemmVariant, Pool, Trans};
use rayon::prelude::*;
use std::collections::BTreeMap;

struct Layer {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ffn_norm: Vec<f32>,
    w1: Vec<f32>,
    w3: Vec<f32>,
    w2: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

pub struct Decoder {
    cfg: ModelConfig,
    emb: Vec<f32>,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    pos: usize,
    pool: Pool,
    variant: GemmVariant,
}

/// Dot product with 8 independent partial sums (vectorises; f32 throughout).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let d = w.len();
    for (o, x) in out.chunks_mut(d).zip(x.chunks(d)) {
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for j in 0..d {
            o[j] = x[j] * inv * w[j];
        }
    }
}

impl Decoder {
    pub fn new(
        cfg: &ModelConfig,
        weights: &BTreeMap<String, Tensor>,
        threads: usize,
    ) -> Result<Decoder, String> {
        cfg.validate()?;
        let take = |name: String, shape: &[usize]| -> Result<Vec<f32>, String> {
            let t = weights.get(&name).ok_or(format!("weights lack {name}"))?;
            if t.shape != shape {
                return Err(format!("{name}: shape {:?} != {shape:?}", t.shape));
            }
            Ok(t.data.clone())
        };
        let shapes: BTreeMap<String, Vec<usize>> = model::params(cfg)
            .into_iter()
            .map(|p| (p.name, p.shape))
            .collect();
        let get = |n: String| {
            let s = shapes[&n].clone();
            take(n, &s)
        };
        let (kvd, seq) = (cfg.n_kv_heads * cfg.head_dim(), cfg.max_seq_len);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let b = format!("blocks.{i}");
            layers.push(Layer {
                attn_norm: get(format!("{b}.attn_norm.weight"))?,
                wq: get(format!("{b}.attn.wq.weight"))?,
                wk: get(format!("{b}.attn.wk.weight"))?,
                wv: get(format!("{b}.attn.wv.weight"))?,
                wo: get(format!("{b}.attn.wo.weight"))?,
                ffn_norm: get(format!("{b}.ffn_norm.weight"))?,
                w1: get(format!("{b}.ffn.w1.weight"))?,
                w3: get(format!("{b}.ffn.w3.weight"))?,
                w2: get(format!("{b}.ffn.w2.weight"))?,
                k: vec![0.0; seq * kvd],
                v: vec![0.0; seq * kvd],
            });
        }
        // Same arithmetic as the ISA's ROPE table (scp_model.precompute_rope, f32).
        let (hd, half) = (cfg.head_dim(), cfg.head_dim() / 2);
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / cfg.rope_theta.powf((2 * i) as f32 / hd as f32))
            .collect();
        let (mut cos, mut sin) = (vec![0.0; seq * half], vec![0.0; seq * half]);
        for t in 0..seq {
            for i in 0..half {
                let f = t as f32 * inv[i];
                cos[t * half + i] = f.cos();
                sin[t * half + i] = f.sin();
            }
        }
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            threads
        };
        Ok(Decoder {
            cfg: cfg.clone(),
            emb: get("tok_emb.weight".into())?,
            norm: get("norm.weight".into())?,
            layers,
            cos,
            sin,
            pos: 0,
            pool: Pool::new(threads),
            variant: forge_kernels::best_available(),
        })
    }

    /// Positions held in the cache.
    pub fn len(&self) -> usize {
        self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// `y[t] = x[t] · Wᵀ` for `rows` rows (W stored `[out, inp]` as in torch).
    /// One row (decoding) is a matrix-vector product: memory-bound, so it
    /// streams W once without packing; blocks of rows (prompts) use the GEMM.
    fn linear(&self, x: &[f32], w: &[f32], rows: usize, inp: usize, out: usize, y: &mut [f32]) {
        if rows == 1 {
            self.pool.install(|| {
                y[..out].par_chunks_mut(64).enumerate().for_each(|(c, ys)| {
                    for (i, yo) in ys.iter_mut().enumerate() {
                        let r = &w[(c * 64 + i) * inp..(c * 64 + i + 1) * inp];
                        *yo = dot(r, &x[..inp]);
                    }
                })
            });
            return;
        }
        let v = self.variant;
        gemm(
            &self.pool,
            v,
            Trans::N,
            Trans::T,
            rows,
            out,
            inp,
            1.0,
            x,
            inp,
            w,
            inp,
            0.0,
            y,
            out,
        );
    }

    /// Append `tokens` at the next positions; returns the logits after the last one.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (d, h, kvh, hd, f) = (c.dim, c.n_heads, c.n_kv_heads, c.head_dim(), c.ffn());
        let (t, p0, half, group) = (tokens.len(), self.pos, hd / 2, h / kvh);
        if t == 0 {
            return Err("feed needs at least one token".into());
        }
        if p0 + t > c.max_seq_len {
            return Err(format!("context full: {} + {t} > {}", p0, c.max_seq_len));
        }
        let mut x = Vec::with_capacity(t * d);
        for &tok in tokens {
            let r = tok as usize;
            if r >= c.vocab_size {
                return Err(format!("token {tok} outside vocabulary {}", c.vocab_size));
            }
            x.extend_from_slice(&self.emb[r * d..(r + 1) * d]);
        }
        let alpha = 1.0 / (hd as f32).sqrt();
        let (mut hbuf, mut q, mut kn, mut vn) = (
            vec![0.0; t * d],
            vec![0.0; t * h * hd],
            vec![0.0; t * kvh * hd],
            vec![0.0; t * kvh * hd],
        );
        let (mut att, mut o) = (vec![0.0; t * h * hd], vec![0.0; t * d]);
        let (mut a, mut b) = (vec![0.0; t * f], vec![0.0; t * f]);
        let (cos, sin) = (&self.cos, &self.sin);
        let rope = |x: &mut [f32], heads: usize| {
            for (r, row) in x.chunks_mut(heads * hd).enumerate() {
                let p = p0 + r;
                let (cs, sn) = (
                    &cos[p * half..(p + 1) * half],
                    &sin[p * half..(p + 1) * half],
                );
                for hh in 0..heads {
                    let v = &mut row[hh * hd..(hh + 1) * hd];
                    for i in 0..half {
                        let (x0, x1) = (v[2 * i], v[2 * i + 1]);
                        v[2 * i] = x0 * cs[i] - x1 * sn[i];
                        v[2 * i + 1] = x0 * sn[i] + x1 * cs[i];
                    }
                }
            }
        };
        for li in 0..c.n_layers {
            let l = &self.layers[li];
            rmsnorm(&x, &l.attn_norm, c.norm_eps, &mut hbuf);
            self.linear(&hbuf, &l.wq, t, d, h * hd, &mut q);
            self.linear(&hbuf, &l.wk, t, d, kvh * hd, &mut kn);
            self.linear(&hbuf, &l.wv, t, d, kvh * hd, &mut vn);
            rope(&mut q, h);
            rope(&mut kn, kvh);
            let kvd = kvh * hd;
            let l = &mut self.layers[li];
            l.k[p0 * kvd..(p0 + t) * kvd].copy_from_slice(&kn);
            l.v[p0 * kvd..(p0 + t) * kvd].copy_from_slice(&vn);
            let (kc, vc) = (&l.k, &l.v);
            self.pool.install(|| {
                att.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
                    let (r, hh) = (i / h, i % h);
                    let g = hh / group;
                    let qv = &q[(r * h + hh) * hd..(r * h + hh + 1) * hd];
                    let n = p0 + r + 1;
                    let mut s: Vec<f32> = (0..n)
                        .map(|j| {
                            let kv = &kc[j * kvd + g * hd..j * kvd + (g + 1) * hd];
                            alpha * qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>()
                        })
                        .collect();
                    let mx = s.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
                    let mut sum = 0.0f32;
                    for v in &mut s {
                        *v = (*v - mx).exp();
                        sum += *v;
                    }
                    let inv = 1.0 / sum;
                    out.fill(0.0);
                    for (j, w) in s.iter().enumerate() {
                        let vv = &vc[j * kvd + g * hd..j * kvd + (g + 1) * hd];
                        let w = w * inv;
                        out.iter_mut().zip(vv).for_each(|(o, v)| *o += w * v);
                    }
                })
            });
            let l = &self.layers[li];
            self.linear(&att, &l.wo, t, h * hd, d, &mut o);
            x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
            rmsnorm(&x, &l.ffn_norm, c.norm_eps, &mut hbuf);
            self.linear(&hbuf, &l.w1, t, d, f, &mut a);
            self.linear(&hbuf, &l.w3, t, d, f, &mut b);
            a.iter_mut()
                .zip(&b)
                .for_each(|(a, b)| *a = *a * (1.0 / (1.0 + (-*a).exp())) * b);
            self.linear(&a, &l.w2, t, f, d, &mut o);
            x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
        }
        self.pos += t;
        let mut last = vec![0.0; d];
        rmsnorm(&x[(t - 1) * d..], &self.norm, c.norm_eps, &mut last);
        let mut logits = vec![0.0; c.vocab_size];
        self.linear(&last, &self.emb, 1, d, c.vocab_size, &mut logits);
        Ok(logits)
    }

    /// Like `Sampler::generate`. The prompt is cut from the left so that
    /// prompt + `max_new` fits the context; should generation still reach
    /// the end of the context, the newest half is re-encoded and decoding
    /// continues.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        seed: u64,
        stop: Option<u32>,
    ) -> Result<Vec<u32>, String> {
        let ctx = self.cfg.max_seq_len;
        let mut rng = rng::Rng::new(seed);
        let keep = ctx.saturating_sub(max_new).max(1);
        let mut ids: Vec<u32> = prompt[prompt.len().saturating_sub(keep)..].to_vec();
        if ids.is_empty() {
            ids.push(0);
        }
        let start = ids.len();
        self.reset();
        let mut logits = self.feed(&ids)?;
        for _ in 0..max_new {
            let next = pick(&logits, temperature, top_k, &mut rng);
            ids.push(next);
            if Some(next) == stop {
                break;
            }
            if self.pos == ctx {
                self.reset();
                let tail = ids[ids.len() - ctx / 2..].to_vec();
                logits = self.feed(&tail)?;
            } else {
                logits = self.feed(&[next])?;
            }
        }
        Ok(ids[start..].to_vec())
    }
}
