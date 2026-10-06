//! KV-cache decoding. The ISA programs process fixed windows, so sampling
//! with them recomputes the whole window for every new token (O(T²) work per
//! sequence). The decoder keeps each layer's rotated keys and values and runs
//! only the new positions: a prompt is one block of GEMMs, every generated
//! token one row. Same equations, parameter layout and kernels as
//! `model::build` (SCP-1); `tests.rs` checks parity with the ISA forward pass.
//!
//! [`Decoder::generate_batch`] decodes many sequences at once: every step feeds
//! one token of each active sequence as one block of rows, so each projection
//! and the tied head stream their weights once per step for all sequences
//! instead of once per sequence. Each row's arithmetic is exactly that of the
//! one-token path (same partial sums in the same order), so a sequence decodes
//! to the same bits whatever else is in the batch: [`Decoder::generate_batch`]
//! returns exactly what [`Decoder::generate`] returns for every prompt.

use crate::ckpt::Tensor;
use crate::config::ModelConfig;
use crate::{model, pick, rng};
use forge_kernels::{gemm, GemmVariant, Pool, Trans};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;

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
}

/// One sequence's rotated keys and values (`[max_seq_len × kv_dim]` per layer)
/// and the number of positions they hold.
#[derive(Default)]
pub(crate) struct Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pos: usize,
}

impl Cache {
    fn new(cfg: &ModelConfig) -> Cache {
        let n = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim();
        Cache {
            k: (0..cfg.n_layers).map(|_| vec![0.0; n]).collect(),
            v: (0..cfg.n_layers).map(|_| vec![0.0; n]).collect(),
            pos: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.pos
    }
}

pub struct Decoder {
    cfg: ModelConfig,
    emb: Vec<f32>,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    cache: Cache,
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

/// `out[r] = dot(w, xs[r])` for `R` rows at once: `w` is loaded once per chunk
/// of 8 for all rows. Every row runs exactly the operations of [`dot`] in the
/// same order (8 lane-wise partial sums, then the same reductions), so the
/// results are bit-identical to `dot`; the rows only interleave. Portable
/// build; [`dot_rows_avx2`] is the same with explicit AVX registers.
#[inline(always)]
fn dot_rows<const R: usize>(w: &[f32], xs: [&[f32]; R], out: &mut [f32]) {
    let n = w.len();
    let body = n - n % 8;
    let xs = xs.map(|x| &x[..n]);
    let mut acc = [[0f32; 8]; R];
    for (i, wc) in w[..body].chunks_exact(8).enumerate() {
        for (a, x) in acc.iter_mut().zip(&xs) {
            let xc = &x[i * 8..i * 8 + 8];
            for k in 0..8 {
                a[k] += wc[k] * xc[k];
            }
        }
    }
    for ((o, a), x) in out.iter_mut().zip(&acc).zip(&xs) {
        let tail: f32 = w[body..].iter().zip(&x[body..]).map(|(x, y)| x * y).sum();
        *o = a.iter().sum::<f32>() + tail;
    }
}

/// [`dot_rows`] with explicit AVX 8-wide registers: one register of partial
/// sums per row, a multiply then an add per chunk (never fused) — the same
/// IEEE operations in the same order, hence the same bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_rows_avx2<const R: usize>(w: &[f32], xs: [&[f32]; R], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = w.len();
    let body = n - n % 8;
    let xs = xs.map(|x| &x[..n]);
    let mut acc = [_mm256_setzero_ps(); R];
    let mut i = 0;
    while i < body {
        // In bounds: i + 8 <= body <= n, and every row of `xs` holds n values.
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        for (a, x) in acc.iter_mut().zip(&xs) {
            *a = _mm256_add_ps(*a, _mm256_mul_ps(wv, _mm256_loadu_ps(x.as_ptr().add(i))));
        }
        i += 8;
    }
    for ((o, a), x) in out.iter_mut().zip(&acc).zip(&xs) {
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), *a);
        let tail: f32 = w[body..].iter().zip(&x[body..]).map(|(x, y)| x * y).sum();
        *o = lanes.iter().sum::<f32>() + tail;
    }
}

/// `yt[o·rows + r] = dot(w[o], x[r])` for the weight rows of `w` (each `inp`
/// long) and all `rows` input rows of `x`, with row kernel `$k` on register
/// blocks of 8/4/2/1 input rows.
macro_rules! dot_block_with {
    ($k:ident, $w:expr, $x:expr, $rows:expr, $inp:expr, $yt:expr) => {{
        let (x, rows, inp): (&[f32], usize, usize) = ($x, $rows, $inp);
        let row = |r: usize| &x[r * inp..(r + 1) * inp];
        for (wr, ys) in $w.chunks_exact(inp).zip($yt.chunks_exact_mut(rows)) {
            let mut r = 0;
            while r + 8 <= rows {
                $k::<8>(wr, std::array::from_fn(|i| row(r + i)), &mut ys[r..r + 8]);
                r += 8;
            }
            if r + 4 <= rows {
                $k::<4>(wr, std::array::from_fn(|i| row(r + i)), &mut ys[r..r + 4]);
                r += 4;
            }
            if r + 2 <= rows {
                $k::<2>(wr, [row(r), row(r + 1)], &mut ys[r..r + 2]);
                r += 2;
            }
            if r < rows {
                $k::<1>(wr, [row(r)], &mut ys[r..r + 1]);
            }
        }
    }};
}

fn dot_block(w: &[f32], x: &[f32], rows: usize, inp: usize, yt: &mut [f32]) {
    dot_block_with!(dot_rows, w, x, rows, inp, yt)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_block_avx2(w: &[f32], x: &[f32], rows: usize, inp: usize, yt: &mut [f32]) {
    dot_block_with!(dot_rows_avx2, w, x, rows, inp, yt)
}

fn dot_block_dispatch(w: &[f32], x: &[f32], rows: usize, inp: usize, yt: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was just detected.
        return unsafe { dot_block_avx2(w, x, rows, inp, yt) };
    }
    dot_block(w, x, rows, inp, yt)
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

/// Rotates every head of one row (`heads × hd` values) to position `cs`/`sn`.
fn rope_row(row: &mut [f32], hd: usize, cs: &[f32], sn: &[f32]) {
    for v in row.chunks_exact_mut(hd) {
        for i in 0..hd / 2 {
            let (x0, x1) = (v[2 * i], v[2 * i + 1]);
            v[2 * i] = x0 * cs[i] - x1 * sn[i];
            v[2 * i + 1] = x0 * sn[i] + x1 * cs[i];
        }
    }
}

/// Softmax attention of one query head `q` over the first `n` cached
/// positions of KV group `g` (rows of `kvd` values); writes into `out`.
#[allow(clippy::too_many_arguments)]
fn attend(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    n: usize,
    kvd: usize,
    g: usize,
    alpha: f32,
    out: &mut [f32],
) {
    let hd = q.len();
    let mut s: Vec<f32> = (0..n)
        .map(|j| {
            let kv = &kc[j * kvd + g * hd..j * kvd + (g + 1) * hd];
            alpha * q.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>()
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
}

/// `a ← silu(a) · b`.
fn swiglu(a: &mut [f32], b: &[f32]) {
    a.iter_mut()
        .zip(b)
        .for_each(|(a, b)| *a = *a * (1.0 / (1.0 + (-*a).exp())) * b);
}

fn add(x: &mut [f32], o: &[f32]) {
    x.iter_mut().zip(o).for_each(|(x, o)| *x += o);
}

/// One sequence of [`Decoder::generate_batch`] while it decodes.
struct Slot {
    index: usize,
    rng: rng::Rng,
    ids: Vec<u32>,
    start: usize,
    cache: Cache,
    logits: Vec<f32>,
    made: usize,
    joined: bool,
    t0: Instant,
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
        let seq = cfg.max_seq_len;
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
            cache: Cache::new(cfg),
            pool: Pool::new(threads),
            variant: forge_kernels::best_available(),
        })
    }

    /// Positions held in the cache.
    pub fn len(&self) -> usize {
        self.cache.pos
    }

    pub fn is_empty(&self) -> bool {
        self.cache.pos == 0
    }

    pub fn reset(&mut self) {
        self.cache.pos = 0;
    }

    /// An empty cache for one more sequence decoded with these weights.
    pub(crate) fn new_cache(&self) -> Cache {
        Cache::new(&self.cfg)
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

    /// `y[r] = x[r] · Wᵀ` for the `rows` decoding rows of a batched step and
    /// every `(W, y)` of `outs` (all with input width `inp`; W stored
    /// `[out, inp]`): a GEMM with M = `rows` that reads every weight row
    /// once for all rows (weight-stationary). The column blocks of all
    /// matrices share one parallel region. Each element is computed exactly
    /// as [`dot`] computes it in the one-row path.
    fn linear_rows(&self, x: &[f32], rows: usize, inp: usize, outs: &mut [(&[f32], &mut [f32])]) {
        if rows == 0 {
            return;
        }
        let total: usize = outs.iter().map(|(w, _)| w.len() / inp).sum();
        // Enough column blocks to keep every thread busy, small enough to
        // keep a block's weights cache-resident while all rows use them.
        let cols = (total / (4 * self.pool.threads())).clamp(8, 64);
        let mut yts: Vec<Vec<f32>> = outs
            .iter()
            .map(|(w, _)| vec![0f32; w.len() / inp * rows])
            .collect();
        let mut blocks: Vec<(&[f32], &mut [f32])> = Vec::new();
        for ((w, _), yt) in outs.iter().zip(&mut yts) {
            blocks.extend(w.chunks(cols * inp).zip(yt.chunks_mut(cols * rows)));
        }
        self.pool.install(|| {
            blocks
                .into_par_iter()
                .for_each(|(wb, ys)| dot_block_dispatch(wb, x, rows, inp, ys))
        });
        for ((w, y), yt) in outs.iter_mut().zip(&yts) {
            let out = w.len() / inp;
            for (r, yr) in y[..rows * out].chunks_exact_mut(out).enumerate() {
                for (o, v) in yr.iter_mut().enumerate() {
                    *v = yt[o * rows + r];
                }
            }
        }
    }

    fn rope_at(&self, row: &mut [f32], p: usize) {
        let half = self.cfg.head_dim() / 2;
        rope_row(
            row,
            self.cfg.head_dim(),
            &self.cos[p * half..(p + 1) * half],
            &self.sin[p * half..(p + 1) * half],
        );
    }

    /// Append `tokens` at the next positions; returns the logits after the last one.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let mut cache = std::mem::take(&mut self.cache);
        let r = self.feed_cache(&mut cache, tokens);
        self.cache = cache;
        r
    }

    /// [`Decoder::feed`] on the given sequence's cache.
    pub(crate) fn feed_cache(&self, cache: &mut Cache, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (d, h, kvh, hd, f) = (c.dim, c.n_heads, c.n_kv_heads, c.head_dim(), c.ffn());
        let (t, p0, group) = (tokens.len(), cache.pos, h / kvh);
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
        let kvd = kvh * hd;
        for li in 0..c.n_layers {
            let l = &self.layers[li];
            rmsnorm(&x, &l.attn_norm, c.norm_eps, &mut hbuf);
            self.linear(&hbuf, &l.wq, t, d, h * hd, &mut q);
            self.linear(&hbuf, &l.wk, t, d, kvd, &mut kn);
            self.linear(&hbuf, &l.wv, t, d, kvd, &mut vn);
            for (r, row) in q.chunks_mut(h * hd).enumerate() {
                self.rope_at(row, p0 + r);
            }
            for (r, row) in kn.chunks_mut(kvd).enumerate() {
                self.rope_at(row, p0 + r);
            }
            cache.k[li][p0 * kvd..(p0 + t) * kvd].copy_from_slice(&kn);
            cache.v[li][p0 * kvd..(p0 + t) * kvd].copy_from_slice(&vn);
            let (kc, vc) = (&cache.k[li], &cache.v[li]);
            self.pool.install(|| {
                att.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
                    let (r, hh) = (i / h, i % h);
                    let qv = &q[(r * h + hh) * hd..(r * h + hh + 1) * hd];
                    attend(qv, kc, vc, p0 + r + 1, kvd, hh / group, alpha, out);
                })
            });
            self.linear(&att, &l.wo, t, h * hd, d, &mut o);
            add(&mut x, &o);
            rmsnorm(&x, &l.ffn_norm, c.norm_eps, &mut hbuf);
            self.linear(&hbuf, &l.w1, t, d, f, &mut a);
            self.linear(&hbuf, &l.w3, t, d, f, &mut b);
            swiglu(&mut a, &b);
            self.linear(&a, &l.w2, t, f, d, &mut o);
            add(&mut x, &o);
        }
        cache.pos += t;
        let mut last = vec![0.0; d];
        rmsnorm(&x[(t - 1) * d..], &self.norm, c.norm_eps, &mut last);
        let mut logits = vec![0.0; c.vocab_size];
        self.linear(&last, &self.emb, 1, d, c.vocab_size, &mut logits);
        Ok(logits)
    }

    /// One decoding step of several sequences: row `r` appends `tokens[r]` to
    /// `caches[r]` at that sequence's own position. All rows go through the
    /// projections and the tied head together ([`Decoder::linear_rows`]);
    /// RoPE and attention are per row. Returns `[rows × vocab]` logits, row
    /// `r` bit-identical to `feed(&[tokens[r]])` on sequence `r` alone.
    pub(crate) fn step(
        &self,
        caches: &mut [&mut Cache],
        tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (d, h, kvh, hd, f) = (c.dim, c.n_heads, c.n_kv_heads, c.head_dim(), c.ffn());
        let (m, group, kvd) = (tokens.len(), h / kvh, kvh * hd);
        if m != caches.len() {
            return Err(format!("{m} tokens for {} sequences", caches.len()));
        }
        if m == 0 {
            return Ok(Vec::new());
        }
        let mut x = Vec::with_capacity(m * d);
        for (&tok, cache) in tokens.iter().zip(caches.iter()) {
            if cache.pos + 1 > c.max_seq_len {
                return Err(format!(
                    "context full: {} + 1 > {}",
                    cache.pos, c.max_seq_len
                ));
            }
            let r = tok as usize;
            if r >= c.vocab_size {
                return Err(format!("token {tok} outside vocabulary {}", c.vocab_size));
            }
            x.extend_from_slice(&self.emb[r * d..(r + 1) * d]);
        }
        let pos: Vec<usize> = caches.iter().map(|c| c.pos).collect();
        let alpha = 1.0 / (hd as f32).sqrt();
        let (mut hbuf, mut q, mut kn, mut vn) = (
            vec![0.0; m * d],
            vec![0.0; m * h * hd],
            vec![0.0; m * kvd],
            vec![0.0; m * kvd],
        );
        let (mut att, mut o) = (vec![0.0; m * h * hd], vec![0.0; m * d]);
        let (mut a, mut b) = (vec![0.0; m * f], vec![0.0; m * f]);
        for li in 0..c.n_layers {
            let l = &self.layers[li];
            rmsnorm(&x, &l.attn_norm, c.norm_eps, &mut hbuf);
            self.linear_rows(
                &hbuf,
                m,
                d,
                &mut [(&l.wq[..], &mut q[..]), (&l.wk, &mut kn), (&l.wv, &mut vn)],
            );
            for r in 0..m {
                let p = pos[r];
                self.rope_at(&mut q[r * h * hd..(r + 1) * h * hd], p);
                self.rope_at(&mut kn[r * kvd..(r + 1) * kvd], p);
                caches[r].k[li][p * kvd..(p + 1) * kvd]
                    .copy_from_slice(&kn[r * kvd..(r + 1) * kvd]);
                caches[r].v[li][p * kvd..(p + 1) * kvd]
                    .copy_from_slice(&vn[r * kvd..(r + 1) * kvd]);
            }
            let views: Vec<(&[f32], &[f32])> = caches
                .iter()
                .map(|c| (&c.k[li][..], &c.v[li][..]))
                .collect();
            self.pool.install(|| {
                att.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
                    let (r, hh) = (i / h, i % h);
                    let qv = &q[(r * h + hh) * hd..(r * h + hh + 1) * hd];
                    let (kc, vc) = views[r];
                    attend(qv, kc, vc, pos[r] + 1, kvd, hh / group, alpha, out);
                })
            });
            self.linear_rows(&att, m, h * hd, &mut [(&l.wo[..], &mut o[..])]);
            add(&mut x, &o);
            rmsnorm(&x, &l.ffn_norm, c.norm_eps, &mut hbuf);
            self.linear_rows(&hbuf, m, d, &mut [(&l.w1[..], &mut a[..]), (&l.w3, &mut b)]);
            swiglu(&mut a, &b);
            self.linear_rows(&a, m, f, &mut [(&l.w2[..], &mut o[..])]);
            add(&mut x, &o);
        }
        for cache in caches.iter_mut() {
            cache.pos += 1;
        }
        rmsnorm(&x, &self.norm, c.norm_eps, &mut hbuf);
        let mut logits = vec![0.0; m * c.vocab_size];
        self.linear_rows(&hbuf, m, d, &mut [(&self.emb[..], &mut logits[..])]);
        Ok(logits)
    }

    /// The prompt as decoding starts from it: cut from the left so that
    /// prompt + `max_new` fits the context, never empty.
    fn cut_prompt(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let keep = self.cfg.max_seq_len.saturating_sub(max_new).max(1);
        let mut ids: Vec<u32> = prompt[prompt.len().saturating_sub(keep)..].to_vec();
        if ids.is_empty() {
            ids.push(0);
        }
        ids
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
        let mut ids = self.cut_prompt(prompt, max_new);
        let start = ids.len();
        self.reset();
        let mut logits = self.feed(&ids)?;
        for _ in 0..max_new {
            let next = pick(&logits, temperature, top_k, &mut rng);
            ids.push(next);
            if Some(next) == stop {
                break;
            }
            if self.cache.pos == ctx {
                self.reset();
                let tail = ids[ids.len() - ctx / 2..].to_vec();
                logits = self.feed(&tail)?;
            } else {
                logits = self.feed(&[next])?;
            }
        }
        Ok(ids[start..].to_vec())
    }

    /// [`Decoder::generate`] for every prompt (`prompts[i]` sampled with
    /// `seeds[i]`), decoding up to `batch` sequences at once; returns the
    /// generated ids in prompt order, equal to one `generate` call per prompt.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_batch(
        &self,
        prompts: &[Vec<u32>],
        seeds: &[u64],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        stop: Option<u32>,
        batch: usize,
    ) -> Result<Vec<Vec<u32>>, String> {
        let mut out = vec![Vec::new(); prompts.len()];
        self.generate_batch_with(
            prompts,
            seeds,
            max_new,
            temperature,
            top_k,
            stop,
            batch,
            |i, ids, _| out[i] = ids,
        )?;
        Ok(out)
    }

    /// Batched decoding with a callback: `done(i, ids, seconds)` is called as
    /// soon as prompt `i` finishes (not necessarily in prompt order), with
    /// the seconds since it entered the batch.
    ///
    /// Each prompt is prefilled with the block path of [`Decoder::feed`],
    /// then every step samples one token per active sequence (`pick` with its
    /// own RNG) and feeds all of them as one block of rows
    /// ([`Decoder::step`]). A sequence leaves the batch at its stop id or
    /// after `max_new` tokens and the next prompt takes its place. Prompt cut,
    /// context overflow (re-encode the newest half) and sampling follow
    /// [`Decoder::generate`] exactly. Should a prompt fail (e.g. a token
    /// outside the vocabulary), the prompts before it are still completed and
    /// reported, then its error is returned — as a loop of `generate` would.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_batch_with(
        &self,
        prompts: &[Vec<u32>],
        seeds: &[u64],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        stop: Option<u32>,
        batch: usize,
        mut done: impl FnMut(usize, Vec<u32>, f64),
    ) -> Result<(), String> {
        if seeds.len() != prompts.len() {
            return Err(format!(
                "{} seeds for {} prompts",
                seeds.len(),
                prompts.len()
            ));
        }
        let (ctx, batch) = (self.cfg.max_seq_len, batch.max(1));
        let mut spare: Vec<Cache> = Vec::new();
        let mut active: Vec<Slot> = Vec::with_capacity(batch);
        let (mut queue, mut failed) = (0..prompts.len(), None);
        loop {
            // Refill: prompts enter in order, each prefilled as one block.
            while failed.is_none() && active.len() < batch {
                let Some(i) = queue.next() else { break };
                let t0 = Instant::now();
                let ids = self.cut_prompt(&prompts[i], max_new);
                let mut cache = spare.pop().unwrap_or_else(|| self.new_cache());
                cache.pos = 0;
                match self.feed_cache(&mut cache, &ids) {
                    Err(e) => failed = Some(e),
                    Ok(_) if max_new == 0 => {
                        done(i, Vec::new(), t0.elapsed().as_secs_f64());
                        spare.push(cache);
                    }
                    Ok(logits) => active.push(Slot {
                        index: i,
                        rng: rng::Rng::new(seeds[i]),
                        start: ids.len(),
                        ids,
                        cache,
                        logits,
                        made: 0,
                        joined: false,
                        t0,
                    }),
                }
            }
            if active.is_empty() {
                break;
            }
            // Sample one token per sequence; finished ones leave, the rest
            // either re-encode (context full) or join this step's block.
            let mut finished = Vec::new();
            for (si, s) in active.iter_mut().enumerate() {
                let next = pick(&s.logits, temperature, top_k, &mut s.rng);
                s.ids.push(next);
                s.made += 1;
                s.joined = false;
                if Some(next) == stop || s.made == max_new {
                    finished.push(si);
                } else if s.cache.pos == ctx {
                    s.cache.pos = 0;
                    let tail = s.ids[s.ids.len() - ctx / 2..].to_vec();
                    s.logits = self.feed_cache(&mut s.cache, &tail)?;
                } else {
                    s.joined = true;
                }
            }
            let tokens: Vec<u32> = active
                .iter()
                .filter(|s| s.joined)
                .map(|s| *s.ids.last().unwrap())
                .collect();
            if !tokens.is_empty() {
                let mut caches: Vec<&mut Cache> = active
                    .iter_mut()
                    .filter(|s| s.joined)
                    .map(|s| &mut s.cache)
                    .collect();
                let logits = self.step(&mut caches, &tokens)?;
                let v = self.cfg.vocab_size;
                for (s, l) in active
                    .iter_mut()
                    .filter(|s| s.joined)
                    .zip(logits.chunks_exact(v))
                {
                    s.logits.copy_from_slice(l);
                }
            }
            for &si in finished.iter().rev() {
                let s = active.swap_remove(si);
                done(
                    s.index,
                    s.ids[s.start..].to_vec(),
                    s.t0.elapsed().as_secs_f64(),
                );
                spare.push(s.cache);
            }
        }
        failed.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::{dot, dot_block, dot_block_dispatch};
    use crate::rng::Rng;

    /// The multi-row kernel is bit-identical to `dot`, for every length
    /// (body + tail), every row count (8/4/2/1 blocks) and either build.
    #[test]
    fn dot_rows_is_bit_identical_to_dot() {
        let mut rng = Rng::new(5);
        for inp in [1usize, 3, 7, 8, 9, 15, 16, 17, 31, 64, 100, 129] {
            for rows in 1..=19 {
                let outs = 5;
                let w: Vec<f32> = (0..outs * inp).map(|_| rng.normal()).collect();
                let x: Vec<f32> = (0..rows * inp).map(|_| 3.0 * rng.normal()).collect();
                let (mut y1, mut y2) = (vec![0f32; outs * rows], vec![0f32; outs * rows]);
                dot_block(&w, &x, rows, inp, &mut y1);
                dot_block_dispatch(&w, &x, rows, inp, &mut y2);
                for o in 0..outs {
                    for r in 0..rows {
                        let want = dot(&w[o * inp..(o + 1) * inp], &x[r * inp..(r + 1) * inp]);
                        assert_eq!(
                            y1[o * rows + r].to_bits(),
                            want.to_bits(),
                            "inp {inp} rows {rows}"
                        );
                        assert_eq!(
                            y2[o * rows + r].to_bits(),
                            want.to_bits(),
                            "inp {inp} rows {rows}"
                        );
                    }
                }
            }
        }
    }
}
