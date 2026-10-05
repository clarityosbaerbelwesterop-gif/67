//! SCP-1 transformer emitted as forge ISA programs.
//!
//! Equations follow `swarm-compute-protocol-/model/scp_model/model.py`
//! exactly (pre-norm RMSNorm, interleaved RoPE, GQA, SwiGLU, tied head).
//! Linear weights are stored like `torch.nn.Linear` ([out, in], y = x·Wᵀ) under
//! SCP parameter names, so checkpoints map 1:1 onto the PyTorch model.
//! The backward pass is derived by hand per layer and emitted as instructions;
//! weight gradients accumulate (beta = 1) so micro-batches need no extra adds.

use crate::config::ModelConfig;
use forge_isa::{regs, Builder, Instr, Program, Stride, V};

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub shape: Vec<usize>,
}

impl Param {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// SCP applies weight decay to every parameter with ndim >= 2 (incl. the embedding).
    pub fn decays(&self) -> bool {
        self.shape.len() >= 2
    }
}

/// Parameters in SCP `named_parameters()` order (tied head deduplicated).
pub fn params(c: &ModelConfig) -> Vec<Param> {
    let (d, hd, f) = (c.dim, c.head_dim(), c.ffn());
    let p = |name: String, shape: Vec<usize>| Param { name, shape };
    let mut v = vec![p("tok_emb.weight".into(), vec![c.vocab_size, d])];
    for i in 0..c.n_layers {
        let b = format!("blocks.{i}");
        v.push(p(format!("{b}.attn_norm.weight"), vec![d]));
        v.push(p(format!("{b}.attn.wq.weight"), vec![c.n_heads * hd, d]));
        v.push(p(format!("{b}.attn.wk.weight"), vec![c.n_kv_heads * hd, d]));
        v.push(p(format!("{b}.attn.wv.weight"), vec![c.n_kv_heads * hd, d]));
        v.push(p(format!("{b}.attn.wo.weight"), vec![d, c.n_heads * hd]));
        v.push(p(format!("{b}.ffn_norm.weight"), vec![d]));
        v.push(p(format!("{b}.ffn.w1.weight"), vec![f, d]));
        v.push(p(format!("{b}.ffn.w3.weight"), vec![f, d]));
        v.push(p(format!("{b}.ffn.w2.weight"), vec![d, f]));
    }
    v.push(p("norm.weight".into(), vec![d]));
    v
}

pub fn grad_name(p: &str) -> String {
    format!("grad.{p}")
}
pub fn adam_m(p: &str) -> String {
    format!("adam_m.{p}")
}
pub fn adam_v(p: &str) -> String {
    format!("adam_v.{p}")
}
pub fn tokens_name(n: usize) -> String {
    format!("tokens.{n}")
}
pub fn targets_name(n: usize) -> String {
    format!("targets.{n}")
}
pub fn logits_name(n: usize) -> String {
    format!("logits.{n}")
}

/// What the forward program ends with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    /// Cross-entropy, then the full backward pass into persistent gradients.
    Train { accum: usize },
    /// Cross-entropy only (loss registers), no gradients.
    Eval,
    /// Persistent logits for sampling.
    Logits,
}

struct Ctx {
    b: Builder,
    batch: usize,
    seq: usize,
}

impl Ctx {
    fn n(&self) -> usize {
        self.batch * self.seq
    }
    fn p(&mut self, name: &str, len: usize) -> V {
        self.b.persistent(name, len)
    }
    fn t(&mut self, name: String, len: usize) -> V {
        self.b.transient(name, len)
    }
    /// y[n×out] = x[n×in]·Wᵀ (W stored [out, in]).
    #[allow(clippy::too_many_arguments)]
    fn linear(&mut self, y: V, x: V, w: V, n: usize, out: usize, inp: usize, accumulate: bool) {
        self.b.matmul(y, x, w, false, true, n, out, inp, accumulate);
    }
    /// dx[n×in] (+)= dy[n×out]·W[out×in]
    #[allow(clippy::too_many_arguments)]
    fn linear_dx(
        &mut self,
        dx: V,
        dy: V,
        w: V,
        n: usize,
        out: usize,
        inp: usize,
        accumulate: bool,
    ) {
        self.b
            .matmul(dx, dy, w, false, false, n, inp, out, accumulate);
    }
    /// dW[out×in] += dyᵀ[out×n]·x[n×in]
    fn linear_dw(&mut self, dw: V, dy: V, x: V, n: usize, out: usize, inp: usize) {
        self.b.matmul(dw, dy, x, true, false, out, inp, n, true);
    }
}

struct LayerActs {
    x: V,
    xn_a: V,
    rstd_a: V,
    q: V,
    k: V,
    v: V,
    p: V,
    att: V,
    xmid: V,
    xn_f: V,
    rstd_f: V,
    h1: V,
    h3: V,
    g: V,
}

struct LayerParams {
    attn_norm: String,
    wq: String,
    wk: String,
    wv: String,
    wo: String,
    ffn_norm: String,
    w1: String,
    w3: String,
    w2: String,
}

fn layer_params(i: usize) -> LayerParams {
    let b = format!("blocks.{i}");
    LayerParams {
        attn_norm: format!("{b}.attn_norm.weight"),
        wq: format!("{b}.attn.wq.weight"),
        wk: format!("{b}.attn.wk.weight"),
        wv: format!("{b}.attn.wv.weight"),
        wo: format!("{b}.attn.wo.weight"),
        ffn_norm: format!("{b}.ffn_norm.weight"),
        w1: format!("{b}.ffn.w1.weight"),
        w3: format!("{b}.ffn.w3.weight"),
        w2: format!("{b}.ffn.w2.weight"),
    }
}

/// Strides for one attention product: q/out rows interleaved per head
/// ([b, t, heads, hd]); scores and stacked per-head tensors are [b, heads, t, *].
struct Att {
    t: usize,
    h: usize,
    kv: usize,
    hd: usize,
}

impl Att {
    fn rows(&self, heads: usize, div: usize) -> Stride {
        Stride {
            outer: (self.t * heads * self.hd) as u32,
            inner: self.hd as u32,
            div: div as u32,
        }
    }
    fn scores(&self) -> Stride {
        Stride {
            outer: (self.h * self.t * self.t) as u32,
            inner: (self.t * self.t) as u32,
            div: 1,
        }
    }
    fn stacked(&self) -> Stride {
        Stride {
            outer: (self.h * self.t * self.hd) as u32,
            inner: (self.t * self.hd) as u32,
            div: 1,
        }
    }
    fn rep(&self) -> usize {
        self.h / self.kv
    }
}

/// Build the program for one micro-batch of `batch × seq` tokens.
pub fn build(c: &ModelConfig, batch: usize, seq: usize, head: Head) -> Program {
    assert!(seq <= c.max_seq_len);
    let mut x = Ctx {
        b: Builder::new(),
        batch,
        seq,
    };
    let (n, d, hd, f, vsz) = (x.n(), c.dim, c.head_dim(), c.ffn(), c.vocab_size);
    let (h, kv) = (c.n_heads, c.n_kv_heads);
    let at = Att { t: seq, h, kv, hd };
    let alpha = 1.0 / (hd as f32).sqrt();
    let train = matches!(head, Head::Train { .. });

    let emb = x.p("tok_emb.weight", vsz * d);
    let tokens = x.p(&tokens_name(n), n);
    let x0 = x.t("x.0".into(), n * d);
    x.b.emit(Instr::Embed {
        out: x0,
        table: emb,
        tokens,
        n: n as u32,
        dim: d as u32,
    });

    let mut acts: Vec<LayerActs> = Vec::new();
    let mut cur = x0;
    for i in 0..c.n_layers {
        let lp = layer_params(i);
        let w_an = x.p(&lp.attn_norm, d);
        let wq = x.p(&lp.wq, h * hd * d);
        let wk = x.p(&lp.wk, kv * hd * d);
        let wv = x.p(&lp.wv, kv * hd * d);
        let wo = x.p(&lp.wo, d * h * hd);
        let w_fn = x.p(&lp.ffn_norm, d);
        let w1 = x.p(&lp.w1, f * d);
        let w3 = x.p(&lp.w3, f * d);
        let w2 = x.p(&lp.w2, d * f);

        let xn_a = x.t(format!("xn_a.{i}"), n * d);
        let rstd_a = x.t(format!("rstd_a.{i}"), n);
        x.b.emit(Instr::RmsNorm {
            out: xn_a,
            rstd: rstd_a,
            x: cur,
            w: w_an,
            rows: n as u32,
            dim: d as u32,
            eps: c.norm_eps,
        });
        let q = x.t(format!("q.{i}"), n * h * hd);
        let k = x.t(format!("k.{i}"), n * kv * hd);
        let v = x.t(format!("v.{i}"), n * kv * hd);
        x.linear(q, xn_a, wq, n, h * hd, d, false);
        x.linear(k, xn_a, wk, n, kv * hd, d, false);
        x.linear(v, xn_a, wv, n, kv * hd, d, false);
        let rope = |x_: V, heads: usize| Instr::Rope {
            x: x_,
            rows: n as u32,
            heads: heads as u32,
            hd: hd as u32,
            seq: seq as u32,
            theta: c.rope_theta,
        };
        x.b.emit(rope(q, h));
        x.b.emit(rope(k, kv));
        let p = x.t(format!("p.{i}"), batch * h * seq * seq);
        x.b.matmul_batched(
            p,
            q,
            k,
            (false, true),
            (seq, seq, hd),
            (h * hd, kv * hd, seq),
            (batch, h),
            (at.rows(h, 1), at.rows(kv, at.rep()), at.scores()),
            alpha,
            false,
        );
        x.b.emit(Instr::SoftmaxCausal {
            x: p,
            rows: (batch * h * seq) as u32,
            cols: seq as u32,
            scale: 1.0,
        });
        let ost = x.t(format!("ost.{i}"), batch * h * seq * hd);
        x.b.matmul_batched(
            ost,
            p,
            v,
            (false, false),
            (seq, hd, seq),
            (seq, kv * hd, hd),
            (batch, h),
            (at.scores(), at.rows(kv, at.rep()), at.stacked()),
            1.0,
            false,
        );
        let att = x.t(format!("att.{i}"), n * h * hd);
        x.b.emit(Instr::HeadsToRows {
            dst: att,
            src: ost,
            b: batch as u32,
            t: seq as u32,
            heads: h as u32,
            hd: hd as u32,
            group: 1,
        });
        let o = x.t(format!("o.{i}"), n * d);
        x.linear(o, att, wo, n, d, h * hd, false);
        let xmid = x.t(format!("xmid.{i}"), n * d);
        x.b.emit(Instr::Add {
            dst: xmid,
            a: cur,
            b: o,
        });

        let xn_f = x.t(format!("xn_f.{i}"), n * d);
        let rstd_f = x.t(format!("rstd_f.{i}"), n);
        x.b.emit(Instr::RmsNorm {
            out: xn_f,
            rstd: rstd_f,
            x: xmid,
            w: w_fn,
            rows: n as u32,
            dim: d as u32,
            eps: c.norm_eps,
        });
        let h1 = x.t(format!("h1.{i}"), n * f);
        let h3 = x.t(format!("h3.{i}"), n * f);
        x.linear(h1, xn_f, w1, n, f, d, false);
        x.linear(h3, xn_f, w3, n, f, d, false);
        let g = x.t(format!("g.{i}"), n * f);
        x.b.emit(Instr::SiluMul {
            g,
            a: h1,
            b: h3,
            n: (n * f) as u32,
        });
        let o2 = x.t(format!("o2.{i}"), n * d);
        x.linear(o2, g, w2, n, d, f, false);
        let next = x.t(format!("x.{}", i + 1), n * d);
        x.b.emit(Instr::Add {
            dst: next,
            a: xmid,
            b: o2,
        });
        acts.push(LayerActs {
            x: cur,
            xn_a,
            rstd_a,
            q,
            k,
            v,
            p,
            att,
            xmid,
            xn_f,
            rstd_f,
            h1,
            h3,
            g,
        });
        cur = next;
    }
    let w_norm = x.p("norm.weight", d);
    let xf = x.t("xf".into(), n * d);
    let rstd_fin = x.t("rstd_fin".into(), n);
    x.b.emit(Instr::RmsNorm {
        out: xf,
        rstd: rstd_fin,
        x: cur,
        w: w_norm,
        rows: n as u32,
        dim: d as u32,
        eps: c.norm_eps,
    });
    let logits = if head == Head::Logits {
        x.p(&logits_name(n), n * vsz)
    } else {
        x.t("logits".into(), n * vsz)
    };
    x.linear(logits, xf, emb, n, vsz, d, false);
    if head == Head::Logits {
        return x.b.finish();
    }
    let targets = x.p(&targets_name(n), n);
    let accum = if let Head::Train { accum } = head {
        accum
    } else {
        1
    };
    let grad_scale = if train { 1.0 / (n * accum) as f32 } else { 0.0 };
    x.b.emit(Instr::Xent {
        logits,
        targets,
        rows: n as u32,
        vocab: vsz as u32,
        grad_scale,
    });
    if !train {
        return x.b.finish();
    }

    // ---------------- backward ----------------
    let g_emb = x.p(&grad_name("tok_emb.weight"), vsz * d);
    let dxf = x.t("dxf".into(), n * d);
    x.linear_dx(dxf, logits, emb, n, vsz, d, false);
    x.linear_dw(g_emb, logits, xf, n, vsz, d);
    let g_norm = x.p(&grad_name("norm.weight"), d);
    let mut dy = x.t(format!("dx.{}", c.n_layers), n * d);
    x.b.emit(Instr::RmsNormBwd {
        dx: dy,
        dw: g_norm,
        dy: dxf,
        x: cur,
        w: w_norm,
        rstd: rstd_fin,
        rows: n as u32,
        dim: d as u32,
        accumulate: false,
    });

    for i in (0..c.n_layers).rev() {
        let lp = layer_params(i);
        let a = &acts[i];
        let (w_an, wq, wk, wv, wo) = (
            x.p(&lp.attn_norm, d),
            x.p(&lp.wq, h * hd * d),
            x.p(&lp.wk, kv * hd * d),
            x.p(&lp.wv, kv * hd * d),
            x.p(&lp.wo, d * h * hd),
        );
        let (w_fn, w1, w3, w2) = (
            x.p(&lp.ffn_norm, d),
            x.p(&lp.w1, f * d),
            x.p(&lp.w3, f * d),
            x.p(&lp.w2, d * f),
        );
        let gp = |x: &mut Ctx, name: &str, len: usize| x.p(&grad_name(name), len);
        let (g_an, g_wq, g_wk, g_wv, g_wo) = (
            gp(&mut x, &lp.attn_norm, d),
            gp(&mut x, &lp.wq, h * hd * d),
            gp(&mut x, &lp.wk, kv * hd * d),
            gp(&mut x, &lp.wv, kv * hd * d),
            gp(&mut x, &lp.wo, d * h * hd),
        );
        let (g_fn, g_w1, g_w3, g_w2) = (
            gp(&mut x, &lp.ffn_norm, d),
            gp(&mut x, &lp.w1, f * d),
            gp(&mut x, &lp.w3, f * d),
            gp(&mut x, &lp.w2, d * f),
        );

        // FFN branch: x_{i+1} = xmid + W2·(silu(W1 xn_f) ∘ W3 xn_f)
        let dg = x.t(format!("dg.{i}"), n * f);
        x.linear_dx(dg, dy, w2, n, d, f, false);
        x.linear_dw(g_w2, dy, a.g, n, d, f);
        let dh1 = x.t(format!("dh1.{i}"), n * f);
        let dh3 = x.t(format!("dh3.{i}"), n * f);
        x.b.emit(Instr::SiluMulBwd {
            da: dh1,
            db: dh3,
            dg,
            a: a.h1,
            b: a.h3,
            n: (n * f) as u32,
        });
        let dxn_f = x.t(format!("dxn_f.{i}"), n * d);
        x.linear_dx(dxn_f, dh1, w1, n, f, d, false);
        x.linear_dx(dxn_f, dh3, w3, n, f, d, true);
        x.linear_dw(g_w1, dh1, a.xn_f, n, f, d);
        x.linear_dw(g_w3, dh3, a.xn_f, n, f, d);
        let dxmid = x.t(format!("dxmid.{i}"), n * d);
        x.b.emit(Instr::Copy {
            dst: dxmid,
            src: dy,
        });
        x.b.emit(Instr::RmsNormBwd {
            dx: dxmid,
            dw: g_fn,
            dy: dxn_f,
            x: a.xmid,
            w: w_fn,
            rstd: a.rstd_f,
            rows: n as u32,
            dim: d as u32,
            accumulate: true,
        });

        // Attention branch: xmid = x + Wo·att
        let datt = x.t(format!("datt.{i}"), n * h * hd);
        x.linear_dx(datt, dxmid, wo, n, d, h * hd, false);
        x.linear_dw(g_wo, dxmid, a.att, n, d, h * hd);
        let dp = x.t(format!("dp.{i}"), batch * h * seq * seq);
        // dP = dO·Vᵀ
        x.b.matmul_batched(
            dp,
            datt,
            a.v,
            (false, true),
            (seq, seq, hd),
            (h * hd, kv * hd, seq),
            (batch, h),
            (at.rows(h, 1), at.rows(kv, at.rep()), at.scores()),
            1.0,
            false,
        );
        // dV_h = Pᵀ·dO (per query head, summed per KV group below)
        let dvst = x.t(format!("dvst.{i}"), batch * h * seq * hd);
        x.b.matmul_batched(
            dvst,
            a.p,
            datt,
            (true, false),
            (seq, hd, seq),
            (seq, h * hd, hd),
            (batch, h),
            (at.scores(), at.rows(h, 1), at.stacked()),
            1.0,
            false,
        );
        x.b.emit(Instr::SoftmaxBwd {
            dp,
            p: a.p,
            rows: (batch * h * seq) as u32,
            cols: seq as u32,
            scale: alpha,
        });
        // dQ = dS·K, dK_h = dSᵀ·Q
        let dqst = x.t(format!("dqst.{i}"), batch * h * seq * hd);
        x.b.matmul_batched(
            dqst,
            dp,
            a.k,
            (false, false),
            (seq, hd, seq),
            (seq, kv * hd, hd),
            (batch, h),
            (at.scores(), at.rows(kv, at.rep()), at.stacked()),
            1.0,
            false,
        );
        let dkst = x.t(format!("dkst.{i}"), batch * h * seq * hd);
        x.b.matmul_batched(
            dkst,
            dp,
            a.q,
            (true, false),
            (seq, hd, seq),
            (seq, h * hd, hd),
            (batch, h),
            (at.scores(), at.rows(h, 1), at.stacked()),
            1.0,
            false,
        );
        let dq = x.t(format!("dq.{i}"), n * h * hd);
        let dk = x.t(format!("dk.{i}"), n * kv * hd);
        let dv = x.t(format!("dv.{i}"), n * kv * hd);
        let h2r = |dst: V, src: V, group: usize| Instr::HeadsToRows {
            dst,
            src,
            b: batch as u32,
            t: seq as u32,
            heads: h as u32,
            hd: hd as u32,
            group: group as u32,
        };
        x.b.emit(h2r(dq, dqst, 1));
        x.b.emit(h2r(dk, dkst, at.rep()));
        x.b.emit(h2r(dv, dvst, at.rep()));
        let rope_bwd = |x_: V, heads: usize| Instr::RopeBwd {
            x: x_,
            rows: n as u32,
            heads: heads as u32,
            hd: hd as u32,
            seq: seq as u32,
            theta: c.rope_theta,
        };
        x.b.emit(rope_bwd(dq, h));
        x.b.emit(rope_bwd(dk, kv));
        let dxn_a = x.t(format!("dxn_a.{i}"), n * d);
        x.linear_dx(dxn_a, dq, wq, n, h * hd, d, false);
        x.linear_dx(dxn_a, dk, wk, n, kv * hd, d, true);
        x.linear_dx(dxn_a, dv, wv, n, kv * hd, d, true);
        x.linear_dw(g_wq, dq, a.xn_a, n, h * hd, d);
        x.linear_dw(g_wk, dk, a.xn_a, n, kv * hd, d);
        x.linear_dw(g_wv, dv, a.xn_a, n, kv * hd, d);
        let dxi = x.t(format!("dx.{i}"), n * d);
        x.b.emit(Instr::Copy {
            dst: dxi,
            src: dxmid,
        });
        x.b.emit(Instr::RmsNormBwd {
            dx: dxi,
            dw: g_an,
            dy: dxn_a,
            x: a.x,
            w: w_an,
            rstd: a.rstd_a,
            rows: n as u32,
            dim: d as u32,
            accumulate: true,
        });
        dy = dxi;
    }
    x.b.emit(Instr::EmbedBwd {
        dtable: g_emb,
        dout: dy,
        tokens,
        n: n as u32,
        dim: d as u32,
    });
    x.b.finish()
}

/// Optimiser step: global-norm clipping, AdamW per parameter, zero gradients.
pub fn build_update(c: &ModelConfig, weight_decay: f32, grad_clip: f32) -> Program {
    let mut b = Builder::new();
    let ps = params(c);
    let grads: Vec<V> = ps
        .iter()
        .map(|p| b.persistent(grad_name(&p.name), p.len()))
        .collect();
    b.emit(Instr::SumSq {
        bufs: grads.clone(),
        dst: regs::SUMSQ,
    });
    b.emit(Instr::ClipCoef {
        dst: regs::CLIP_COEF,
        src: regs::SUMSQ,
        max_norm: grad_clip,
    });
    for (p, &g) in ps.iter().zip(&grads) {
        let pv = b.persistent(p.name.clone(), p.len());
        let m = b.persistent(adam_m(&p.name), p.len());
        let v = b.persistent(adam_v(&p.name), p.len());
        b.emit(Instr::AdamW {
            p: pv,
            g,
            m,
            v,
            lr: regs::LR,
            step: regs::STEP,
            clip: regs::CLIP_COEF,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            wd: if p.decays() { weight_decay } else { 0.0 },
        });
    }
    for g in grads {
        b.emit(Instr::Zero { dst: g });
    }
    b.finish()
}
