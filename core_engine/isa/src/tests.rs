use super::*;
use crate::machine::Machine;

/// Deterministic pseudo-random values in [-1, 1).
fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

struct H {
    m: Machine,
}

impl H {
    fn new() -> H {
        H {
            m: Machine::new(1 << 22, 2),
        }
    }
    fn set(&mut self, b: &mut Builder, name: &str, data: &[f32]) -> V {
        let v = b.persistent(name, data.len());
        if self.m.persistent(name).is_none() {
            let mut p = Builder::new();
            p.persistent(name, data.len());
            let lp = self.m.load(p.finish()).unwrap();
            self.m.unload(lp).unwrap();
        }
        self.m.persistent_mut(name).unwrap().copy_from_slice(data);
        v
    }
    fn run(&mut self, b: Builder) {
        let mut lp = self.m.load(b.finish()).unwrap();
        self.m.run(&mut lp).unwrap();
        self.m.unload(lp).unwrap();
    }
    fn get(&self, name: &str) -> Vec<f32> {
        self.m.persistent(name).unwrap().to_vec()
    }
}

/// Central-difference check of d(Σ out·r)/d(input) against an analytic gradient.
fn check_grad(analytic: &[f32], f: &mut dyn FnMut(&[f32]) -> f64, x: &[f32], eps: f32, tol: f64) {
    let mut worst = 0.0f64;
    let norm = analytic
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        .max(1e-6);
    let mut xp = x.to_vec();
    for i in 0..x.len() {
        xp[i] = x[i] + eps;
        let up = f(&xp);
        xp[i] = x[i] - eps;
        let dn = f(&xp);
        xp[i] = x[i];
        let num = (up - dn) / (2.0 * eps as f64);
        worst = worst.max((num - analytic[i] as f64).abs() / norm);
    }
    assert!(
        worst < tol,
        "gradient mismatch: worst relative error {worst:.3e} (tol {tol:.1e})"
    );
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

#[test]
fn bytecode_roundtrip_covers_every_opcode() {
    let mut b = Builder::new();
    let x = b.persistent("x", 64);
    let y = b.transient("y", 64);
    let z = b.persistent("z", 64);
    let t = b.persistent("tok", 4);
    let s = Stride {
        outer: 16,
        inner: 8,
        div: 1,
    };
    let ins = vec![
        Instr::Zero { dst: y },
        Instr::Copy { dst: y, src: x },
        Instr::Add { dst: z, a: x, b: y },
        Instr::AddInplace { dst: z, src: x },
        Instr::Mul { dst: z, a: x, b: y },
        Instr::Scale {
            dst: z,
            value: 0.5,
            reg: Some(regs::LR),
        },
        Instr::MatMul {
            c: z,
            a: x,
            b: y,
            ta: true,
            tb: false,
            m: 8,
            n: 8,
            k: 8,
            lda: 8,
            ldb: 8,
            ldc: 8,
            alpha: 1.0,
            beta: 0.0,
            slot: 0,
        },
        Instr::MatMulBatched {
            c: z,
            a: x,
            b: y,
            ta: false,
            tb: true,
            m: 2,
            n: 2,
            k: 4,
            lda: 4,
            ldb: 4,
            ldc: 2,
            alpha: 0.5,
            beta: 1.0,
            outer: 2,
            inner: 2,
            sa: s,
            sb: Stride {
                outer: 16,
                inner: 8,
                div: 2,
            },
            sc: Stride {
                outer: 8,
                inner: 4,
                div: 1,
            },
            slot: 1,
        },
        Instr::HeadsToRows {
            dst: y,
            src: x,
            b: 1,
            t: 4,
            heads: 4,
            hd: 4,
            group: 1,
        },
        Instr::Embed {
            out: z,
            table: x,
            tokens: t,
            n: 4,
            dim: 8,
        },
        Instr::EmbedBwd {
            dtable: z,
            dout: x,
            tokens: t,
            n: 4,
            dim: 8,
        },
        Instr::RmsNorm {
            out: z,
            rstd: y,
            x,
            w: t,
            rows: 1,
            dim: 4,
            eps: 1e-5,
        },
        Instr::RmsNormBwd {
            dx: z,
            dw: y,
            dy: x,
            x,
            w: t,
            rstd: t,
            rows: 1,
            dim: 4,
            accumulate: true,
        },
        Instr::Rope {
            x: z,
            rows: 4,
            heads: 2,
            hd: 8,
            seq: 4,
            theta: 1e4,
        },
        Instr::RopeBwd {
            x: z,
            rows: 4,
            heads: 2,
            hd: 8,
            seq: 4,
            theta: 1e4,
        },
        Instr::SoftmaxCausal {
            x: z,
            rows: 8,
            cols: 8,
            scale: 1.0,
        },
        Instr::SoftmaxBwd {
            dp: z,
            p: x,
            rows: 8,
            cols: 8,
            scale: 0.25,
        },
        Instr::SiluMul {
            g: z,
            a: x,
            b: y,
            n: 64,
        },
        Instr::SiluMulBwd {
            da: z,
            db: y,
            dg: x,
            a: t,
            b: t,
            n: 4,
        },
        Instr::Xent {
            logits: z,
            targets: t,
            rows: 4,
            vocab: 16,
            grad_scale: 0.25,
        },
        Instr::SumSq {
            bufs: vec![x, z],
            dst: regs::SUMSQ,
        },
        Instr::ClipCoef {
            dst: regs::CLIP_COEF,
            src: regs::SUMSQ,
            max_norm: 1.0,
        },
        Instr::AdamW {
            p: z,
            g: x,
            m: y,
            v: y,
            lr: 0,
            step: 1,
            clip: 2,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            wd: 0.1,
        },
        Instr::Quant { x: z, format: 1 },
        Instr::SetReg { dst: 3, value: 2.5 },
        Instr::Alloc { v: y },
        Instr::Free { v: y },
    ];
    let covered: std::collections::HashSet<Opcode> = ins.iter().map(|i| i.opcode()).collect();
    assert_eq!(
        covered.len(),
        Opcode::ALL.len(),
        "test must cover every opcode"
    );
    for i in ins {
        b.emit(i);
    }
    let p = b.finish();
    let bytes = p.encode();
    assert_eq!(&bytes[..4], MAGIC);
    let q = Program::decode(&bytes).unwrap();
    assert_eq!(p, q);
    // Every truncation is rejected, never misparsed.
    for cut in [0, 3, 10, bytes.len() / 2, bytes.len() - 1] {
        assert!(Program::decode(&bytes[..cut]).is_err());
    }
    let mut bad = bytes.clone();
    bad[0] = b'X';
    assert!(Program::decode(&bad).is_err());
    assert!(disassemble(&q).contains("MATMUL_BATCHED"));
}

#[test]
fn verifier_rejects_malformed_programs() {
    let mut b = Builder::new();
    let a = b.persistent("a", 16);
    let c = b.persistent("c", 15);
    b.matmul(c, a, a, false, false, 4, 4, 4, false);
    assert!(b.finish().verify().unwrap_err().0.contains("C needs 16"));

    let mut b = Builder::new();
    let a = b.persistent("a", 16);
    b.matmul(a, a, a, false, false, 4, 4, 4, false);
    assert!(b.finish().verify().unwrap_err().0.contains("aliases"));

    let mut b = Builder::new();
    let a = b.persistent("a", 64);
    let c = b.persistent("c", 64);
    // Interleaved (side-by-side) batched outputs would race: rejected.
    b.matmul_batched(
        c,
        a,
        a,
        (false, false),
        (4, 4, 4),
        (4, 4, 8),
        (1, 2),
        (
            Stride {
                outer: 0,
                inner: 16,
                div: 1,
            },
            Stride {
                outer: 0,
                inner: 16,
                div: 1,
            },
            Stride {
                outer: 0,
                inner: 4,
                div: 1,
            },
        ),
        1.0,
        false,
    );
    assert!(b.finish().verify().unwrap_err().0.contains("overlap"));
}

#[test]
fn matmul_and_batched_match_reference() {
    let mut h = H::new();
    let (bt, heads, kv, t, hd) = (2usize, 4usize, 2usize, 3usize, 4usize);
    let q = rnd(bt * t * heads * hd, 1);
    let k = rnd(bt * t * kv * hd, 2);
    let mut b = Builder::new();
    let qv = h.set(&mut b, "q", &q);
    let kvv = h.set(&mut b, "k", &k);
    let s = h.set(&mut b, "s", &vec![0.0; bt * heads * t * t]);
    b.matmul_batched(
        s,
        qv,
        kvv,
        (false, true),
        (t, t, hd),
        (heads * hd, kv * hd, t),
        (bt, heads),
        (
            Stride {
                outer: (t * heads * hd) as u32,
                inner: hd as u32,
                div: 1,
            },
            Stride {
                outer: (t * kv * hd) as u32,
                inner: hd as u32,
                div: (heads / kv) as u32,
            },
            Stride {
                outer: (heads * t * t) as u32,
                inner: (t * t) as u32,
                div: 1,
            },
        ),
        0.5,
        false,
    );
    h.run(b);
    let got = h.get("s");
    for bb in 0..bt {
        for hh in 0..heads {
            for i in 0..t {
                for j in 0..t {
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        acc += q[(bb * t + i) * heads * hd + hh * hd + d]
                            * k[(bb * t + j) * kv * hd + (hh / 2) * hd + d];
                    }
                    let g = got[((bb * heads + hh) * t + i) * t + j];
                    assert!((g - 0.5 * acc).abs() < 1e-5, "S[{bb},{hh},{i},{j}]");
                }
            }
        }
    }
}

#[test]
fn rmsnorm_backward_matches_numeric() {
    let (rows, dim) = (3usize, 8usize);
    let x = rnd(rows * dim, 3);
    let w: Vec<f32> = rnd(dim, 4).iter().map(|v| 1.0 + 0.5 * v).collect();
    let r = rnd(rows * dim, 5);
    let fwd = |x: &[f32], w: &[f32]| -> f64 {
        let mut h = H::new();
        let mut b = Builder::new();
        let (xv, wv) = (h.set(&mut b, "x", x), h.set(&mut b, "w", w));
        let o = h.set(&mut b, "o", &vec![0.0; rows * dim]);
        let rs = h.set(&mut b, "r", &vec![0.0; rows]);
        b.emit(Instr::RmsNorm {
            out: o,
            rstd: rs,
            x: xv,
            w: wv,
            rows: rows as u32,
            dim: dim as u32,
            eps: 1e-5,
        });
        h.run(b);
        dot(&h.get("o"), &r)
    };
    let mut h = H::new();
    let mut b = Builder::new();
    let (xv, wv) = (h.set(&mut b, "x", &x), h.set(&mut b, "w", &w));
    let o = h.set(&mut b, "o", &vec![0.0; rows * dim]);
    let rs = h.set(&mut b, "r", &vec![0.0; rows]);
    let dy = h.set(&mut b, "dy", &r);
    let dx = h.set(&mut b, "dx", &vec![0.0; rows * dim]);
    let dw = h.set(&mut b, "dw", &vec![0.0; dim]);
    b.emit(Instr::RmsNorm {
        out: o,
        rstd: rs,
        x: xv,
        w: wv,
        rows: rows as u32,
        dim: dim as u32,
        eps: 1e-5,
    });
    b.emit(Instr::RmsNormBwd {
        dx,
        dw,
        dy,
        x: xv,
        w: wv,
        rstd: rs,
        rows: rows as u32,
        dim: dim as u32,
        accumulate: false,
    });
    h.run(b);
    check_grad(&h.get("dx"), &mut |xp| fwd(xp, &w), &x, 1e-3, 5e-3);
    check_grad(&h.get("dw"), &mut |wp| fwd(&x, wp), &w, 1e-3, 5e-3);
}

#[test]
fn rope_is_a_rotation_and_backward_inverts_it() {
    let (rows, heads, hd, seq) = (6usize, 2usize, 8usize, 3usize);
    let x = rnd(rows * heads * hd, 6);
    let mut h = H::new();
    let mut b = Builder::new();
    let xv = h.set(&mut b, "x", &x);
    b.emit(Instr::Rope {
        x: xv,
        rows: rows as u32,
        heads: heads as u32,
        hd: hd as u32,
        seq: seq as u32,
        theta: 1e4,
    });
    h.run(b);
    let y = h.get("x");
    // Norm per pair is preserved; position 0 is the identity.
    for r in 0..rows {
        for i in 0..heads * hd / 2 {
            let o = r * heads * hd + 2 * i;
            let n0 = x[o].hypot(x[o + 1]);
            let n1 = y[o].hypot(y[o + 1]);
            assert!((n0 - n1).abs() < 1e-5);
            if r % seq == 0 {
                assert!((x[o] - y[o]).abs() < 1e-6);
            }
        }
    }
    let mut b = Builder::new();
    let xv = h.set(&mut b, "x", &y);
    b.emit(Instr::RopeBwd {
        x: xv,
        rows: rows as u32,
        heads: heads as u32,
        hd: hd as u32,
        seq: seq as u32,
        theta: 1e4,
    });
    h.run(b);
    let back = h.get("x");
    assert!(back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-5));
}

#[test]
fn softmax_backward_matches_numeric() {
    let (rows, cols, scale) = (6usize, 3usize, 0.7f32);
    let x = rnd(rows * cols, 7);
    let r = rnd(rows * cols, 8);
    let fwd = |x: &[f32]| -> f64 {
        let mut h = H::new();
        let mut b = Builder::new();
        let xv = h.set(&mut b, "x", x);
        b.emit(Instr::SoftmaxCausal {
            x: xv,
            rows: rows as u32,
            cols: cols as u32,
            scale,
        });
        h.run(b);
        dot(&h.get("x"), &r)
    };
    let mut h = H::new();
    let mut b = Builder::new();
    let xv = h.set(&mut b, "x", &x);
    let dp = h.set(&mut b, "dp", &r);
    b.emit(Instr::SoftmaxCausal {
        x: xv,
        rows: rows as u32,
        cols: cols as u32,
        scale,
    });
    b.emit(Instr::SoftmaxBwd {
        dp,
        p: xv,
        rows: rows as u32,
        cols: cols as u32,
        scale,
    });
    h.run(b);
    let p = h.get("x");
    for r_ in 0..rows {
        let q = r_ % cols;
        let row = &p[r_ * cols..(r_ + 1) * cols];
        assert!((row[..=q].iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(row[q + 1..].iter().all(|&v| v == 0.0), "causal mask");
    }
    check_grad(&h.get("dp"), &mut |xp| fwd(xp), &x, 1e-3, 5e-3);
}

#[test]
fn silu_mul_backward_matches_numeric() {
    let n = 17usize;
    let (a, bb, r) = (rnd(n, 9), rnd(n, 10), rnd(n, 11));
    let fwd = |a: &[f32], bb: &[f32]| -> f64 {
        let mut h = H::new();
        let mut b = Builder::new();
        let (av, bv) = (h.set(&mut b, "a", a), h.set(&mut b, "b", bb));
        let g = h.set(&mut b, "g", &vec![0.0; n]);
        b.emit(Instr::SiluMul {
            g,
            a: av,
            b: bv,
            n: n as u32,
        });
        h.run(b);
        dot(&h.get("g"), &r)
    };
    let mut h = H::new();
    let mut b = Builder::new();
    let (av, bv) = (h.set(&mut b, "a", &a), h.set(&mut b, "b", &bb));
    let dg = h.set(&mut b, "dg", &r);
    let da = h.set(&mut b, "da", &vec![0.0; n]);
    let db = h.set(&mut b, "db", &vec![0.0; n]);
    b.emit(Instr::SiluMulBwd {
        da,
        db,
        dg,
        a: av,
        b: bv,
        n: n as u32,
    });
    h.run(b);
    check_grad(&h.get("da"), &mut |ap| fwd(ap, &bb), &a, 1e-3, 5e-3);
    check_grad(&h.get("db"), &mut |bp| fwd(&a, bp), &bb, 1e-3, 5e-3);
}

#[test]
fn xent_loss_and_gradient_are_exact() {
    let (rows, vocab) = (4usize, 7usize);
    let x: Vec<f32> = rnd(rows * vocab, 12).iter().map(|v| v * 3.0).collect();
    let tg = vec![0.0f32, 6.0, 3.0, 3.0];
    let loss = |x: &[f32]| -> f64 {
        (0..rows)
            .map(|r| {
                let row = &x[r * vocab..(r + 1) * vocab];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = mx + row.iter().map(|v| (*v as f64 - mx).exp()).sum::<f64>().ln();
                lse - row[tg[r] as usize] as f64
            })
            .sum::<f64>()
            / rows as f64
    };
    let mut h = H::new();
    let mut b = Builder::new();
    let xv = h.set(&mut b, "x", &x);
    let t = h.set(&mut b, "t", &tg);
    b.emit(Instr::Xent {
        logits: xv,
        targets: t,
        rows: rows as u32,
        vocab: vocab as u32,
        grad_scale: 1.0 / rows as f32,
    });
    h.run(b);
    let mean = h.m.regs[regs::LOSS as usize] / h.m.regs[regs::LOSS_COUNT as usize];
    assert!((mean as f64 - loss(&x)).abs() < 1e-5);
    check_grad(&h.get("x"), &mut |xp| loss(xp), &x, 1e-3, 2e-3);

    let mut b = Builder::new();
    let xv = h.set(&mut b, "x", &x);
    let t = h.set(&mut b, "t", &[0.0, 7.0, 1.0, 1.0]);
    b.emit(Instr::Xent {
        logits: xv,
        targets: t,
        rows: rows as u32,
        vocab: vocab as u32,
        grad_scale: 1.0,
    });
    let mut lp = h.m.load(b.finish()).unwrap();
    assert!(matches!(
        h.m.run(&mut lp),
        Err(MachineError::BadToken { index: 1, .. })
    ));
}

#[test]
fn embedding_and_heads_to_rows_backward() {
    let (vocab, dim) = (5usize, 3usize);
    let table = rnd(vocab * dim, 13);
    let toks = vec![4.0f32, 0.0, 4.0, 2.0];
    let mut h = H::new();
    let mut b = Builder::new();
    let tv = h.set(&mut b, "tab", &table);
    let tk = h.set(&mut b, "tok", &toks);
    let o = h.set(&mut b, "o", &vec![0.0; 4 * dim]);
    let dt = h.set(&mut b, "dt", &vec![0.0; vocab * dim]);
    b.emit(Instr::Embed {
        out: o,
        table: tv,
        tokens: tk,
        n: 4,
        dim: dim as u32,
    });
    b.emit(Instr::EmbedBwd {
        dtable: dt,
        dout: o,
        tokens: tk,
        n: 4,
        dim: dim as u32,
    });
    h.run(b);
    let (o, dt) = (h.get("o"), h.get("dt"));
    assert_eq!(&o[..dim], &table[4 * dim..5 * dim]);
    for j in 0..dim {
        assert!(
            (dt[4 * dim + j] - 2.0 * table[4 * dim + j]).abs() < 1e-6,
            "token 4 used twice"
        );
        assert_eq!(dt[dim + j], 0.0);
    }
    // HEADS_TO_ROWS with group 2: [b=1, h=4, t=2, d=2] -> [1, 2, 2, 2] summing pairs.
    let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut b = Builder::new();
    let s = h.set(&mut b, "src", &src);
    let d = h.set(&mut b, "dst", &[0.0; 8]);
    b.emit(Instr::HeadsToRows {
        dst: d,
        src: s,
        b: 1,
        t: 2,
        heads: 4,
        hd: 2,
        group: 2,
    });
    h.run(b);
    // row t=0: heads (0+1) and (2+3) at t=0: [0,1]+[4,5]=[4,6], [8,9]+[12,13]=[20,22]
    assert_eq!(
        h.get("dst"),
        vec![4.0, 6.0, 20.0, 22.0, 8.0, 10.0, 24.0, 26.0]
    );
}

#[test]
fn adamw_matches_torch_reference_values() {
    // One step of torch.optim.AdamW(lr=0.1, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
    // on p=[1.0, -2.0], g=[0.5, 0.25] gives p = [0.89, -2.0 * (1 - 0.01) - 0.1].
    let mut h = H::new();
    let mut b = Builder::new();
    let p = h.set(&mut b, "p", &[1.0, -2.0]);
    let g = h.set(&mut b, "g", &[0.5, 0.25]);
    let m = h.set(&mut b, "m", &[0.0, 0.0]);
    let v = h.set(&mut b, "v", &[0.0, 0.0]);
    b.set_reg(regs::LR, 0.1);
    b.set_reg(regs::STEP, 1.0);
    b.set_reg(regs::CLIP_COEF, 1.0);
    b.emit(Instr::AdamW {
        p,
        g,
        m,
        v,
        lr: regs::LR,
        step: regs::STEP,
        clip: regs::CLIP_COEF,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        wd: 0.1,
    });
    h.run(b);
    let got = h.get("p");
    assert!((got[0] - (1.0 * 0.99 - 0.1)).abs() < 1e-6, "{got:?}");
    assert!((got[1] - (-2.0 * 0.99 - 0.1)).abs() < 1e-6, "{got:?}");
}

#[test]
fn compiler_fuses_places_and_preserves_semantics() {
    let build = || {
        let mut b = Builder::new();
        let x = b.persistent("x", 64);
        let w = b.persistent("w", 64);
        let r = b.persistent("r", 64);
        let y = b.persistent("y", 64);
        let t = b.transient("t", 64);
        let dead = b.transient("dead", 64);
        let tmp = b.transient("tmp", 64);
        b.emit(Instr::Copy { dst: dead, src: x });
        b.matmul(t, x, w, false, true, 8, 8, 8, false);
        b.emit(Instr::Add {
            dst: tmp,
            a: r,
            b: t,
        });
        b.emit(Instr::Copy { dst: y, src: tmp });
        // A second transient whose lifetime starts after `tmp` ends: placement reuses memory.
        let u = b.transient("u", 64);
        b.emit(Instr::Copy { dst: u, src: x });
        b.emit(Instr::AddInplace { dst: y, src: u });
        b.finish()
    };
    let init = |h: &mut H| {
        let mut b = Builder::new();
        h.set(&mut b, "x", &rnd(64, 20));
        h.set(&mut b, "w", &rnd(64, 21));
        h.set(&mut b, "r", &rnd(64, 22));
        h.set(&mut b, "y", &[0.0; 64]);
    };
    let (prog, rep) = compile(build(), CompileOptions::default()).unwrap();
    assert_eq!(rep.fused_residual, 1);
    assert_eq!(rep.dead_removed, 1);
    assert!(rep.transient_words_peak < rep.transient_words_naive);
    assert!(prog
        .instrs
        .iter()
        .any(|i| matches!(i, Instr::MatMul { beta, .. } if *beta == 1.0)));
    let mut a = H::new();
    init(&mut a);
    let mut lp = a.m.load(prog).unwrap();
    a.m.run(&mut lp).unwrap();
    let mut b = H::new();
    init(&mut b);
    let mut lp2 = b.m.load(build()).unwrap();
    b.m.run(&mut lp2).unwrap();
    let (ya, yb) = (a.get("y"), b.get("y"));
    assert!(ya.iter().zip(&yb).all(|(p, q)| (p - q).abs() < 1e-6));
    assert_eq!(a.m.hbvm_stats().live, 4, "placement freed every transient");
}
