//! forge-isa — the tensor instruction set of 67 Forge.
//!
//! One instruction is one tensor primitive over virtual buffers (`V`). A
//! [`Program`] is a typed instruction list with buffer declarations; it
//! serialises to bytecode (`b"F67I"`), is verified before execution, is
//! optimised by the compiler (fusion, dead-buffer elimination, liveness-based
//! HBVM placement) and is executed by the [`Machine`] interpreter, which
//! dispatches each instruction to the kernel engines and accounts FLOPs, bytes
//! and nanoseconds per opcode.
//!
//! Contract: docs/DESIGN.md §5.

mod builder;
mod compiler;
mod machine;

pub use builder::Builder;
pub use compiler::{compile, CompileOptions, CompileReport};
pub use machine::{Loaded, Machine, MachineError, OpStats, TuneReport};

use std::fmt;

/// A virtual buffer of f32 words, resolved to an HBVM handle at run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct V(pub u32);

/// Scalar register index (16 registers of f32).
pub type Reg = u8;
pub const NUM_REGS: usize = 16;

/// Register conventions used by the training programs.
pub mod regs {
    use super::Reg;
    pub const LR: Reg = 0;
    pub const STEP: Reg = 1;
    pub const CLIP_COEF: Reg = 2;
    pub const LOSS: Reg = 3;
    pub const SUMSQ: Reg = 4;
    pub const CORRECT: Reg = 5;
    pub const LOSS_COUNT: Reg = 6;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Opcode {
    Zero = 0x01,
    Copy = 0x02,
    Add = 0x03,
    AddInplace = 0x04,
    Mul = 0x05,
    Scale = 0x06,
    MatMul = 0x10,
    MatMulBatched = 0x11,
    HeadsToRows = 0x12,
    Embed = 0x20,
    EmbedBwd = 0x21,
    RmsNorm = 0x22,
    RmsNormBwd = 0x23,
    Rope = 0x24,
    RopeBwd = 0x25,
    SoftmaxCausal = 0x26,
    SoftmaxBwd = 0x27,
    SiluMul = 0x28,
    SiluMulBwd = 0x29,
    Xent = 0x2A,
    SumSq = 0x30,
    ClipCoef = 0x31,
    AdamW = 0x32,
    Quant = 0x33,
    SetReg = 0x40,
    Alloc = 0x50,
    Free = 0x51,
}

impl Opcode {
    pub const ALL: [Opcode; 27] = [
        Opcode::Zero,
        Opcode::Copy,
        Opcode::Add,
        Opcode::AddInplace,
        Opcode::Mul,
        Opcode::Scale,
        Opcode::MatMul,
        Opcode::MatMulBatched,
        Opcode::HeadsToRows,
        Opcode::Embed,
        Opcode::EmbedBwd,
        Opcode::RmsNorm,
        Opcode::RmsNormBwd,
        Opcode::Rope,
        Opcode::RopeBwd,
        Opcode::SoftmaxCausal,
        Opcode::SoftmaxBwd,
        Opcode::SiluMul,
        Opcode::SiluMulBwd,
        Opcode::Xent,
        Opcode::SumSq,
        Opcode::ClipCoef,
        Opcode::AdamW,
        Opcode::Quant,
        Opcode::SetReg,
        Opcode::Alloc,
        Opcode::Free,
    ];

    pub fn from_u8(b: u8) -> Option<Opcode> {
        Opcode::ALL.into_iter().find(|o| *o as u8 == b)
    }

    pub fn name(self) -> &'static str {
        match self {
            Opcode::Zero => "ZERO",
            Opcode::Copy => "COPY",
            Opcode::Add => "ADD",
            Opcode::AddInplace => "ADD_INPLACE",
            Opcode::Mul => "MUL",
            Opcode::Scale => "SCALE",
            Opcode::MatMul => "MATMUL",
            Opcode::MatMulBatched => "MATMUL_BATCHED",
            Opcode::HeadsToRows => "HEADS_TO_ROWS",
            Opcode::Embed => "EMBED",
            Opcode::EmbedBwd => "EMBED_BWD",
            Opcode::RmsNorm => "RMSNORM",
            Opcode::RmsNormBwd => "RMSNORM_BWD",
            Opcode::Rope => "ROPE",
            Opcode::RopeBwd => "ROPE_BWD",
            Opcode::SoftmaxCausal => "SOFTMAX_CAUSAL",
            Opcode::SoftmaxBwd => "SOFTMAX_BWD",
            Opcode::SiluMul => "SILU_MUL",
            Opcode::SiluMulBwd => "SILU_MUL_BWD",
            Opcode::Xent => "XENT",
            Opcode::SumSq => "SUMSQ",
            Opcode::ClipCoef => "CLIP_COEF",
            Opcode::AdamW => "ADAMW",
            Opcode::Quant => "QUANT",
            Opcode::SetReg => "SET_REG",
            Opcode::Alloc => "ALLOC",
            Opcode::Free => "FREE",
        }
    }
}

/// Operand layout of a strided, two-level batched matrix: element (o, i) of
/// the batch starts at `o*outer + (i/div)*inner`. `div > 1` broadcasts one
/// matrix to `div` consecutive inner indices (grouped-query attention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stride {
    pub outer: u32,
    pub inner: u32,
    pub div: u32,
}

/// Typed instruction. Operand order is documented per variant: outputs first.
#[derive(Clone, Debug, PartialEq)]
pub enum Instr {
    /// dst = 0
    Zero { dst: V },
    /// dst = src
    Copy { dst: V, src: V },
    /// dst = a + b
    Add { dst: V, a: V, b: V },
    /// dst += src
    AddInplace { dst: V, src: V },
    /// dst = a * b
    Mul { dst: V, a: V, b: V },
    /// dst *= value (or regs[reg] when reg is Some)
    Scale {
        dst: V,
        value: f32,
        reg: Option<Reg>,
    },
    /// C[m×n] = alpha·op(A)·op(B) + beta·C (row-major). `slot` selects the engine.
    MatMul {
        c: V,
        a: V,
        b: V,
        ta: bool,
        tb: bool,
        m: u32,
        n: u32,
        k: u32,
        lda: u32,
        ldb: u32,
        ldc: u32,
        alpha: f32,
        beta: f32,
        slot: u16,
    },
    /// Two-level batched MatMul over `outer × inner` matrices.
    MatMulBatched {
        c: V,
        a: V,
        b: V,
        ta: bool,
        tb: bool,
        m: u32,
        n: u32,
        k: u32,
        lda: u32,
        ldb: u32,
        ldc: u32,
        alpha: f32,
        beta: f32,
        outer: u32,
        inner: u32,
        sa: Stride,
        sb: Stride,
        sc: Stride,
        slot: u16,
    },
    /// Head-major [b, heads, t, hd] → row-major [b, t, heads/group, hd], summing each
    /// group of `group` consecutive heads (group = 1 is a plain permutation).
    HeadsToRows {
        dst: V,
        src: V,
        b: u32,
        t: u32,
        heads: u32,
        hd: u32,
        group: u32,
    },
    /// out[i, :] = table[tokens[i], :]; tokens hold exact integer ids as f32.
    Embed {
        out: V,
        table: V,
        tokens: V,
        n: u32,
        dim: u32,
    },
    /// dtable[tokens[i], :] += dout[i, :]
    EmbedBwd {
        dtable: V,
        dout: V,
        tokens: V,
        n: u32,
        dim: u32,
    },
    /// out = x * rsqrt(mean(x²) + eps) * w; rstd[row] saved for the backward pass.
    RmsNorm {
        out: V,
        rstd: V,
        x: V,
        w: V,
        rows: u32,
        dim: u32,
        eps: f32,
    },
    /// dx (= or +=) and dw += for RMSNorm.
    RmsNormBwd {
        dx: V,
        dw: V,
        dy: V,
        x: V,
        w: V,
        rstd: V,
        rows: u32,
        dim: u32,
        accumulate: bool,
    },
    /// In-place interleaved rotary embedding on rows of `heads*hd`; position = row % seq.
    Rope {
        x: V,
        rows: u32,
        heads: u32,
        hd: u32,
        seq: u32,
        theta: f32,
    },
    /// In-place inverse rotation of the gradient.
    RopeBwd {
        x: V,
        rows: u32,
        heads: u32,
        hd: u32,
        seq: u32,
        theta: f32,
    },
    /// In-place causal softmax over rows of length `cols` after scaling; query = row % cols.
    SoftmaxCausal {
        x: V,
        rows: u32,
        cols: u32,
        scale: f32,
    },
    /// In-place: dp ← p ∘ (dp − rowsum(dp ∘ p)) · scale.
    SoftmaxBwd {
        dp: V,
        p: V,
        rows: u32,
        cols: u32,
        scale: f32,
    },
    /// g = silu(a) * b
    SiluMul { g: V, a: V, b: V, n: u32 },
    /// da = dg·b·silu'(a), db = dg·silu(a)
    SiluMulBwd {
        da: V,
        db: V,
        dg: V,
        a: V,
        b: V,
        n: u32,
    },
    /// Fused softmax + cross-entropy: logits are replaced by grad·`grad_scale`;
    /// regs[LOSS] += Σ loss, regs[CORRECT] += Σ argmax==target, regs[LOSS_COUNT] += rows.
    Xent {
        logits: V,
        targets: V,
        rows: u32,
        vocab: u32,
        grad_scale: f32,
    },
    /// regs[dst] = Σ_buffers Σ x²
    SumSq { bufs: Vec<V>, dst: Reg },
    /// regs[dst] = min(1, max_norm / (sqrt(regs[src]) + 1e-6))
    ClipCoef { dst: Reg, src: Reg, max_norm: f32 },
    /// Decoupled-weight-decay Adam with bias correction; grad scaled by regs[clip].
    AdamW {
        p: V,
        g: V,
        m: V,
        v: V,
        lr: Reg,
        step: Reg,
        clip: Reg,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    },
    /// In-place fake quantisation to a forge-formats format code (see `machine::quant`).
    Quant { x: V, format: u8 },
    /// regs[dst] = value
    SetReg { dst: Reg, value: f32 },
    /// Placement: bind a transient buffer to HBVM memory.
    Alloc { v: V },
    /// Placement: release a transient buffer.
    Free { v: V },
}

impl Instr {
    pub fn opcode(&self) -> Opcode {
        match self {
            Instr::Zero { .. } => Opcode::Zero,
            Instr::Copy { .. } => Opcode::Copy,
            Instr::Add { .. } => Opcode::Add,
            Instr::AddInplace { .. } => Opcode::AddInplace,
            Instr::Mul { .. } => Opcode::Mul,
            Instr::Scale { .. } => Opcode::Scale,
            Instr::MatMul { .. } => Opcode::MatMul,
            Instr::MatMulBatched { .. } => Opcode::MatMulBatched,
            Instr::HeadsToRows { .. } => Opcode::HeadsToRows,
            Instr::Embed { .. } => Opcode::Embed,
            Instr::EmbedBwd { .. } => Opcode::EmbedBwd,
            Instr::RmsNorm { .. } => Opcode::RmsNorm,
            Instr::RmsNormBwd { .. } => Opcode::RmsNormBwd,
            Instr::Rope { .. } => Opcode::Rope,
            Instr::RopeBwd { .. } => Opcode::RopeBwd,
            Instr::SoftmaxCausal { .. } => Opcode::SoftmaxCausal,
            Instr::SoftmaxBwd { .. } => Opcode::SoftmaxBwd,
            Instr::SiluMul { .. } => Opcode::SiluMul,
            Instr::SiluMulBwd { .. } => Opcode::SiluMulBwd,
            Instr::Xent { .. } => Opcode::Xent,
            Instr::SumSq { .. } => Opcode::SumSq,
            Instr::ClipCoef { .. } => Opcode::ClipCoef,
            Instr::AdamW { .. } => Opcode::AdamW,
            Instr::Quant { .. } => Opcode::Quant,
            Instr::SetReg { .. } => Opcode::SetReg,
            Instr::Alloc { .. } => Opcode::Alloc,
            Instr::Free { .. } => Opcode::Free,
        }
    }

    /// (written, read) buffers.
    pub fn io(&self) -> (Vec<V>, Vec<V>) {
        use Instr::*;
        match self {
            Zero { dst } => (vec![*dst], vec![]),
            Copy { dst, src } => (vec![*dst], vec![*src]),
            Add { dst, a, b } | Mul { dst, a, b } => (vec![*dst], vec![*a, *b]),
            AddInplace { dst, src } => (vec![*dst], vec![*dst, *src]),
            Scale { dst, .. } => (vec![*dst], vec![*dst]),
            MatMul { c, a, b, beta, .. } | MatMulBatched { c, a, b, beta, .. } => {
                let mut r = vec![*a, *b];
                if *beta != 0.0 {
                    r.push(*c);
                }
                (vec![*c], r)
            }
            HeadsToRows { dst, src, .. } => (vec![*dst], vec![*src]),
            Embed {
                out, table, tokens, ..
            } => (vec![*out], vec![*table, *tokens]),
            EmbedBwd {
                dtable,
                dout,
                tokens,
                ..
            } => (vec![*dtable], vec![*dtable, *dout, *tokens]),
            RmsNorm {
                out, rstd, x, w, ..
            } => (vec![*out, *rstd], vec![*x, *w]),
            RmsNormBwd {
                dx,
                dw,
                dy,
                x,
                w,
                rstd,
                accumulate,
                ..
            } => {
                let mut r = vec![*dy, *x, *w, *rstd, *dw];
                if *accumulate {
                    r.push(*dx);
                }
                (vec![*dx, *dw], r)
            }
            Rope { x, .. } | RopeBwd { x, .. } | SoftmaxCausal { x, .. } | Quant { x, .. } => {
                (vec![*x], vec![*x])
            }
            SoftmaxBwd { dp, p, .. } => (vec![*dp], vec![*dp, *p]),
            SiluMul { g, a, b, .. } => (vec![*g], vec![*a, *b]),
            SiluMulBwd {
                da, db, dg, a, b, ..
            } => (vec![*da, *db], vec![*dg, *a, *b]),
            Xent {
                logits, targets, ..
            } => (vec![*logits], vec![*logits, *targets]),
            SumSq { bufs, .. } => (vec![], bufs.clone()),
            AdamW { p, g, m, v, .. } => (vec![*p, *m, *v], vec![*p, *g, *m, *v]),
            ClipCoef { .. } | SetReg { .. } => (vec![], vec![]),
            Alloc { v } | Free { v } => (vec![*v], vec![]),
        }
    }

    /// Mutable access to every buffer operand (used by compiler renaming).
    pub fn buffers_mut(&mut self) -> Vec<&mut V> {
        use Instr::*;
        match self {
            Zero { dst } => vec![dst],
            Copy { dst, src } | AddInplace { dst, src } => vec![dst, src],
            Add { dst, a, b } | Mul { dst, a, b } => vec![dst, a, b],
            Scale { dst, .. } => vec![dst],
            MatMul { c, a, b, .. } | MatMulBatched { c, a, b, .. } => vec![c, a, b],
            HeadsToRows { dst, src, .. } => vec![dst, src],
            Embed {
                out, table, tokens, ..
            } => vec![out, table, tokens],
            EmbedBwd {
                dtable,
                dout,
                tokens,
                ..
            } => vec![dtable, dout, tokens],
            RmsNorm {
                out, rstd, x, w, ..
            } => vec![out, rstd, x, w],
            RmsNormBwd {
                dx,
                dw,
                dy,
                x,
                w,
                rstd,
                ..
            } => vec![dx, dw, dy, x, w, rstd],
            Rope { x, .. } | RopeBwd { x, .. } | SoftmaxCausal { x, .. } | Quant { x, .. } => {
                vec![x]
            }
            SoftmaxBwd { dp, p, .. } => vec![dp, p],
            SiluMul { g, a, b, .. } => vec![g, a, b],
            SiluMulBwd {
                da, db, dg, a, b, ..
            } => vec![da, db, dg, a, b],
            Xent {
                logits, targets, ..
            } => vec![logits, targets],
            SumSq { bufs, .. } => bufs.iter_mut().collect(),
            AdamW { p, g, m, v, .. } => vec![p, g, m, v],
            ClipCoef { .. } | SetReg { .. } => vec![],
            Alloc { v } | Free { v } => vec![v],
        }
    }

    /// Floating-point operations performed (multiply-add = 2).
    pub fn flops(&self) -> f64 {
        use Instr::*;
        match self {
            MatMul { m, n, k, .. } => 2.0 * *m as f64 * *n as f64 * *k as f64,
            MatMulBatched {
                m,
                n,
                k,
                outer,
                inner,
                ..
            } => 2.0 * *m as f64 * *n as f64 * *k as f64 * *outer as f64 * *inner as f64,
            RmsNorm { rows, dim, .. } => 4.0 * *rows as f64 * *dim as f64,
            RmsNormBwd { rows, dim, .. } => 8.0 * *rows as f64 * *dim as f64,
            Rope {
                rows, heads, hd, ..
            }
            | RopeBwd {
                rows, heads, hd, ..
            } => 3.0 * (*rows * *heads * *hd) as f64,
            SoftmaxCausal { rows, cols, .. } => 2.5 * *rows as f64 * *cols as f64,
            SoftmaxBwd { rows, cols, .. } => 2.0 * *rows as f64 * *cols as f64,
            SiluMul { n, .. } => 6.0 * *n as f64,
            SiluMulBwd { n, .. } => 12.0 * *n as f64,
            Xent { rows, vocab, .. } => 4.0 * *rows as f64 * *vocab as f64,
            _ => 0.0,
        }
    }
}

/// Role of a declared buffer; decides lifetime and placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    /// Lives for the whole run (parameters, gradients, optimiser state, inputs).
    Persistent = 0,
    /// Lives between its first and last use within one program execution.
    Transient = 1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BufDecl {
    pub name: String,
    pub len: u32,
    pub kind: Kind,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Program {
    pub bufs: Vec<BufDecl>,
    pub instrs: Vec<Instr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsaError(pub String);

impl fmt::Display for IsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for IsaError {}

fn err<T>(msg: impl Into<String>) -> Result<T, IsaError> {
    Err(IsaError(msg.into()))
}

impl Program {
    pub fn len_of(&self, v: V) -> usize {
        self.bufs[v.0 as usize].len as usize
    }

    /// Structural verification: operand validity, buffer sizes, aliasing and
    /// register bounds. Every program is verified before it runs.
    pub fn verify(&self) -> Result<(), IsaError> {
        let nb = self.bufs.len();
        for (pc, ins) in self.instrs.iter().enumerate() {
            let ctx = |m: String| IsaError(format!("pc {pc} {}: {m}", ins.opcode().name()));
            let mut ins = ins.clone();
            for v in ins.buffers_mut() {
                if v.0 as usize >= nb {
                    return Err(ctx(format!("buffer {} undeclared", v.0)));
                }
            }
            let len = |v: V| self.bufs[v.0 as usize].len as u64;
            let need = |v: V, n: u64, what: &str| -> Result<(), IsaError> {
                if len(v) < n {
                    return Err(ctx(format!(
                        "{what} needs {n} words, buffer {} has {}",
                        v.0,
                        len(v)
                    )));
                }
                Ok(())
            };
            let reg = |r: Reg| -> Result<(), IsaError> {
                if r as usize >= NUM_REGS {
                    return Err(ctx(format!("register r{r} out of range")));
                }
                Ok(())
            };
            match &ins {
                Instr::Copy { dst, src } | Instr::AddInplace { dst, src } => {
                    if len(*dst) != len(*src) {
                        return Err(ctx("length mismatch".into()));
                    }
                }
                Instr::Add { dst, a, b } | Instr::Mul { dst, a, b } => {
                    if len(*a) != len(*dst) || len(*b) != len(*dst) {
                        return Err(ctx("length mismatch".into()));
                    }
                }
                Instr::Scale { reg: Some(r), .. } => reg(*r)?,
                Instr::MatMul {
                    c,
                    a,
                    b,
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    ..
                } => {
                    let (ar, ac) = if *ta { (*k, *m) } else { (*m, *k) };
                    let (br, bc) = if *tb { (*n, *k) } else { (*k, *n) };
                    if *lda < ac || *ldb < bc || *ldc < *n {
                        return Err(ctx("leading dimension smaller than row width".into()));
                    }
                    if *m == 0 || *n == 0 || *k == 0 {
                        return Err(ctx("empty matmul".into()));
                    }
                    need(*a, (ar as u64 - 1) * *lda as u64 + ac as u64, "A")?;
                    need(*b, (br as u64 - 1) * *ldb as u64 + bc as u64, "B")?;
                    need(*c, (*m as u64 - 1) * *ldc as u64 + *n as u64, "C")?;
                    if c == a || c == b {
                        return Err(ctx("output aliases an input".into()));
                    }
                }
                Instr::MatMulBatched {
                    c,
                    a,
                    b,
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    outer,
                    inner,
                    sa,
                    sb,
                    sc,
                    ..
                } => {
                    let (ar, ac) = if *ta { (*k, *m) } else { (*m, *k) };
                    let (br, bc) = if *tb { (*n, *k) } else { (*k, *n) };
                    if *lda < ac || *ldb < bc || *ldc < *n {
                        return Err(ctx("leading dimension smaller than row width".into()));
                    }
                    let last = |s: &Stride| {
                        (*outer as u64 - 1) * s.outer as u64
                            + ((*inner as u64 - 1) / s.div.max(1) as u64) * s.inner as u64
                    };
                    if sa.div == 0 || sb.div == 0 || sc.div == 0 || *outer == 0 || *inner == 0 {
                        return Err(ctx("zero batch or divisor".into()));
                    }
                    need(
                        *a,
                        last(sa) + (ar as u64 - 1) * *lda as u64 + ac as u64,
                        "A",
                    )?;
                    need(
                        *b,
                        last(sb) + (br as u64 - 1) * *ldb as u64 + bc as u64,
                        "B",
                    )?;
                    need(
                        *c,
                        last(sc) + (*m as u64 - 1) * *ldc as u64 + *n as u64,
                        "C",
                    )?;
                    if sc.div != 1 {
                        return Err(ctx("broadcast output is a write race".into()));
                    }
                    // Output matrices must occupy disjoint, non-interleaved ranges:
                    // each matrix spans `span` words and consecutive matrices follow
                    // without overlap (stacked layout).
                    let span = (*m as u64 - 1) * *ldc as u64 + *n as u64;
                    let stacked = *inner == 1 || sc.inner as u64 >= span;
                    let outer_ok = *outer == 1
                        || sc.outer as u64 >= (*inner as u64 - 1) * sc.inner as u64 + span;
                    if !(stacked && outer_ok) {
                        return Err(ctx(
                            "batched outputs overlap; use a stacked layout and HEADS_TO_ROWS"
                                .into(),
                        ));
                    }
                    if c == a || c == b {
                        return Err(ctx("output aliases an input".into()));
                    }
                }
                Instr::HeadsToRows {
                    dst,
                    src,
                    b,
                    t,
                    heads,
                    hd,
                    group,
                } => {
                    if *group == 0 || heads % group != 0 {
                        return Err(ctx("heads must be a multiple of group".into()));
                    }
                    let n = *b as u64 * *t as u64 * *heads as u64 * *hd as u64;
                    need(*src, n, "src")?;
                    need(*dst, n / *group as u64, "dst")?;
                    if dst == src {
                        return Err(ctx("output aliases input".into()));
                    }
                }
                Instr::Embed {
                    out,
                    table,
                    tokens,
                    n,
                    dim,
                } => {
                    need(*out, *n as u64 * *dim as u64, "out")?;
                    need(*tokens, *n as u64, "tokens")?;
                    if len(*table) % *dim as u64 != 0 {
                        return Err(ctx("table is not a multiple of dim".into()));
                    }
                }
                Instr::EmbedBwd {
                    dtable,
                    dout,
                    tokens,
                    n,
                    dim,
                } => {
                    need(*dout, *n as u64 * *dim as u64, "dout")?;
                    need(*tokens, *n as u64, "tokens")?;
                    if len(*dtable) % *dim as u64 != 0 {
                        return Err(ctx("table is not a multiple of dim".into()));
                    }
                }
                Instr::RmsNorm {
                    out,
                    rstd,
                    x,
                    w,
                    rows,
                    dim,
                    ..
                } => {
                    let n = *rows as u64 * *dim as u64;
                    need(*out, n, "out")?;
                    need(*x, n, "x")?;
                    need(*w, *dim as u64, "w")?;
                    need(*rstd, *rows as u64, "rstd")?;
                }
                Instr::RmsNormBwd {
                    dx,
                    dw,
                    dy,
                    x,
                    w,
                    rstd,
                    rows,
                    dim,
                    ..
                } => {
                    let n = *rows as u64 * *dim as u64;
                    for (v, w_) in [(dx, "dx"), (dy, "dy"), (x, "x")] {
                        need(*v, n, w_)?;
                    }
                    need(*w, *dim as u64, "w")?;
                    need(*dw, *dim as u64, "dw")?;
                    need(*rstd, *rows as u64, "rstd")?;
                }
                Instr::Rope {
                    x,
                    rows,
                    heads,
                    hd,
                    seq,
                    ..
                }
                | Instr::RopeBwd {
                    x,
                    rows,
                    heads,
                    hd,
                    seq,
                    ..
                } => {
                    if hd % 2 != 0 || *seq == 0 {
                        return Err(ctx("head_dim must be even and seq > 0".into()));
                    }
                    need(*x, *rows as u64 * *heads as u64 * *hd as u64, "x")?;
                }
                Instr::SoftmaxCausal { x, rows, cols, .. } => {
                    need(*x, *rows as u64 * *cols as u64, "x")?
                }
                Instr::SoftmaxBwd {
                    dp, p, rows, cols, ..
                } => {
                    need(*dp, *rows as u64 * *cols as u64, "dp")?;
                    need(*p, *rows as u64 * *cols as u64, "p")?;
                }
                Instr::SiluMul { g, a, b, n } => {
                    for v in [g, a, b] {
                        need(*v, *n as u64, "operand")?;
                    }
                }
                Instr::SiluMulBwd {
                    da,
                    db,
                    dg,
                    a,
                    b,
                    n,
                } => {
                    for v in [da, db, dg, a, b] {
                        need(*v, *n as u64, "operand")?;
                    }
                }
                Instr::Xent {
                    logits,
                    targets,
                    rows,
                    vocab,
                    ..
                } => {
                    need(*logits, *rows as u64 * *vocab as u64, "logits")?;
                    need(*targets, *rows as u64, "targets")?;
                }
                Instr::SumSq { dst, .. } => reg(*dst)?,
                Instr::ClipCoef { dst, src, .. } => {
                    reg(*dst)?;
                    reg(*src)?;
                }
                Instr::AdamW {
                    p,
                    g,
                    m,
                    v,
                    lr,
                    step,
                    clip,
                    ..
                } => {
                    for r in [lr, step, clip] {
                        reg(*r)?;
                    }
                    for b in [g, m, v] {
                        if len(*b) != len(*p) {
                            return Err(ctx("optimiser state length mismatch".into()));
                        }
                    }
                }
                Instr::SetReg { dst, .. } => reg(*dst)?,
                _ => {}
            }
            let (outs, ins_) = ins.io();
            if let Instr::Add { .. } | Instr::Mul { .. } | Instr::SiluMul { .. } = &ins {
                if ins_.iter().any(|v| outs.contains(v)) {
                    return Err(ctx("output aliases an input".into()));
                }
            }
            if let Instr::SiluMulBwd { da, db, .. } = &ins {
                if da == db || ins_.contains(da) || ins_.contains(db) {
                    return Err(ctx("gradient outputs alias".into()));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bytecode
// ---------------------------------------------------------------------------

pub const MAGIC: &[u8; 4] = b"F67I";
pub const VERSION: u16 = 1;

struct W(Vec<u8>);
impl W {
    fn u8(&mut self, x: u8) {
        self.0.push(x);
    }
    fn u16(&mut self, x: u16) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    fn u32(&mut self, x: u32) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    fn f32(&mut self, x: f32) {
        self.u32(x.to_bits());
    }
}

struct R<'a>(&'a [u8], usize);
impl R<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], IsaError> {
        if self.1 + n > self.0.len() {
            return err("truncated bytecode");
        }
        let s = &self.0[self.1..self.1 + n];
        self.1 += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, IsaError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, IsaError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, IsaError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, IsaError> {
        Ok(f32::from_bits(self.u32()?))
    }
}

/// Attribute words per opcode, in encoding order (operands are encoded separately).
fn attrs(ins: &Instr, w: &mut W) {
    use Instr::*;
    let stride = |w: &mut W, s: &Stride| {
        w.u32(s.outer);
        w.u32(s.inner);
        w.u32(s.div);
    };
    match ins {
        Scale { value, reg, .. } => {
            w.f32(*value);
            w.u8(reg.map_or(0xFF, |r| r));
        }
        MatMul {
            ta,
            tb,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            alpha,
            beta,
            slot,
            ..
        } => {
            w.u8(*ta as u8);
            w.u8(*tb as u8);
            for x in [m, n, k, lda, ldb, ldc] {
                w.u32(*x);
            }
            w.f32(*alpha);
            w.f32(*beta);
            w.u16(*slot);
        }
        MatMulBatched {
            ta,
            tb,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            alpha,
            beta,
            outer,
            inner,
            sa,
            sb,
            sc,
            slot,
            ..
        } => {
            w.u8(*ta as u8);
            w.u8(*tb as u8);
            for x in [m, n, k, lda, ldb, ldc] {
                w.u32(*x);
            }
            w.f32(*alpha);
            w.f32(*beta);
            w.u32(*outer);
            w.u32(*inner);
            stride(w, sa);
            stride(w, sb);
            stride(w, sc);
            w.u16(*slot);
        }
        HeadsToRows {
            b,
            t,
            heads,
            hd,
            group,
            ..
        } => {
            for x in [b, t, heads, hd, group] {
                w.u32(*x);
            }
        }
        Embed { n, dim, .. } | EmbedBwd { n, dim, .. } => {
            w.u32(*n);
            w.u32(*dim);
        }
        RmsNorm { rows, dim, eps, .. } => {
            w.u32(*rows);
            w.u32(*dim);
            w.f32(*eps);
        }
        RmsNormBwd {
            rows,
            dim,
            accumulate,
            ..
        } => {
            w.u32(*rows);
            w.u32(*dim);
            w.u8(*accumulate as u8);
        }
        Rope {
            rows,
            heads,
            hd,
            seq,
            theta,
            ..
        }
        | RopeBwd {
            rows,
            heads,
            hd,
            seq,
            theta,
            ..
        } => {
            for x in [rows, heads, hd, seq] {
                w.u32(*x);
            }
            w.f32(*theta);
        }
        SoftmaxCausal {
            rows, cols, scale, ..
        }
        | SoftmaxBwd {
            rows, cols, scale, ..
        } => {
            w.u32(*rows);
            w.u32(*cols);
            w.f32(*scale);
        }
        SiluMul { n, .. } | SiluMulBwd { n, .. } => w.u32(*n),
        Xent {
            rows,
            vocab,
            grad_scale,
            ..
        } => {
            w.u32(*rows);
            w.u32(*vocab);
            w.f32(*grad_scale);
        }
        SumSq { dst, .. } => w.u8(*dst),
        ClipCoef { dst, src, max_norm } => {
            w.u8(*dst);
            w.u8(*src);
            w.f32(*max_norm);
        }
        AdamW {
            lr,
            step,
            clip,
            beta1,
            beta2,
            eps,
            wd,
            ..
        } => {
            w.u8(*lr);
            w.u8(*step);
            w.u8(*clip);
            for x in [beta1, beta2, eps, wd] {
                w.f32(*x);
            }
        }
        Quant { format, .. } => w.u8(*format),
        SetReg { dst, value } => {
            w.u8(*dst);
            w.f32(*value);
        }
        Zero { .. }
        | Copy { .. }
        | Add { .. }
        | AddInplace { .. }
        | Mul { .. }
        | Alloc { .. }
        | Free { .. } => {}
    }
}

impl Program {
    /// Serialise: header, buffer table, then
    /// `u8 opcode, u8 n_operands, u16 attr_len, u32 operands[n], attr bytes` per instruction.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.0.extend_from_slice(MAGIC);
        w.u16(VERSION);
        w.u16(0);
        w.u32(self.bufs.len() as u32);
        w.u32(self.instrs.len() as u32);
        for b in &self.bufs {
            w.u32(b.len);
            w.u8(b.kind as u8);
            w.u16(b.name.len() as u16);
            w.0.extend_from_slice(b.name.as_bytes());
        }
        for ins in &self.instrs {
            let mut ins2 = ins.clone();
            let ops: Vec<u32> = ins2.buffers_mut().into_iter().map(|v| v.0).collect();
            let mut a = W(Vec::new());
            attrs(ins, &mut a);
            w.u8(ins.opcode() as u8);
            w.u8(ops.len() as u8);
            w.u16(a.0.len() as u16);
            for o in ops {
                w.u32(o);
            }
            w.0.extend_from_slice(&a.0);
        }
        w.0
    }

    pub fn decode(bytes: &[u8]) -> Result<Program, IsaError> {
        let mut r = R(bytes, 0);
        if r.take(4)? != MAGIC {
            return err("bad magic");
        }
        let version = r.u16()?;
        if version != VERSION {
            return err(format!("unsupported bytecode version {version}"));
        }
        let _flags = r.u16()?;
        let nbufs = r.u32()? as usize;
        let ninstr = r.u32()? as usize;
        let mut bufs = Vec::with_capacity(nbufs.min(1 << 20));
        for _ in 0..nbufs {
            let len = r.u32()?;
            let kind = match r.u8()? {
                0 => Kind::Persistent,
                1 => Kind::Transient,
                k => return err(format!("bad buffer kind {k}")),
            };
            let nlen = r.u16()? as usize;
            let name = String::from_utf8(r.take(nlen)?.to_vec())
                .map_err(|_| IsaError("bad buffer name".into()))?;
            bufs.push(BufDecl { name, len, kind });
        }
        let mut instrs = Vec::with_capacity(ninstr.min(1 << 20));
        for _ in 0..ninstr {
            let opb = r.u8()?;
            let op = Opcode::from_u8(opb)
                .ok_or_else(|| IsaError(format!("unknown opcode 0x{opb:02x}")))?;
            let nops = r.u8()? as usize;
            let alen = r.u16()? as usize;
            let mut ops = Vec::with_capacity(nops);
            for _ in 0..nops {
                ops.push(V(r.u32()?));
            }
            let mut a = R(r.take(alen)?, 0);
            let ins = decode_instr(op, &ops, &mut a)?;
            if a.1 != alen {
                return err(format!(
                    "{}: {} trailing attribute bytes",
                    op.name(),
                    alen - a.1
                ));
            }
            instrs.push(ins);
        }
        if r.1 != bytes.len() {
            return err("trailing bytes after program");
        }
        Ok(Program { bufs, instrs })
    }
}

fn decode_instr(op: Opcode, o: &[V], a: &mut R<'_>) -> Result<Instr, IsaError> {
    use Instr::*;
    let want = |n: usize| -> Result<(), IsaError> {
        if o.len() != n {
            return err(format!(
                "{} expects {n} operands, got {}",
                op.name(),
                o.len()
            ));
        }
        Ok(())
    };
    let stride = |a: &mut R<'_>| -> Result<Stride, IsaError> {
        Ok(Stride {
            outer: a.u32()?,
            inner: a.u32()?,
            div: a.u32()?,
        })
    };
    Ok(match op {
        Opcode::Zero => {
            want(1)?;
            Zero { dst: o[0] }
        }
        Opcode::Copy => {
            want(2)?;
            Copy {
                dst: o[0],
                src: o[1],
            }
        }
        Opcode::Add => {
            want(3)?;
            Add {
                dst: o[0],
                a: o[1],
                b: o[2],
            }
        }
        Opcode::AddInplace => {
            want(2)?;
            AddInplace {
                dst: o[0],
                src: o[1],
            }
        }
        Opcode::Mul => {
            want(3)?;
            Mul {
                dst: o[0],
                a: o[1],
                b: o[2],
            }
        }
        Opcode::Scale => {
            want(1)?;
            let value = a.f32()?;
            let r = a.u8()?;
            Scale {
                dst: o[0],
                value,
                reg: (r != 0xFF).then_some(r),
            }
        }
        Opcode::MatMul => {
            want(3)?;
            let ta = a.u8()? != 0;
            let tb = a.u8()? != 0;
            let (m, n, k, lda, ldb, ldc) =
                (a.u32()?, a.u32()?, a.u32()?, a.u32()?, a.u32()?, a.u32()?);
            let (alpha, beta, slot) = (a.f32()?, a.f32()?, a.u16()?);
            MatMul {
                c: o[0],
                a: o[1],
                b: o[2],
                ta,
                tb,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                alpha,
                beta,
                slot,
            }
        }
        Opcode::MatMulBatched => {
            want(3)?;
            let ta = a.u8()? != 0;
            let tb = a.u8()? != 0;
            let (m, n, k, lda, ldb, ldc) =
                (a.u32()?, a.u32()?, a.u32()?, a.u32()?, a.u32()?, a.u32()?);
            let (alpha, beta) = (a.f32()?, a.f32()?);
            let (outer, inner) = (a.u32()?, a.u32()?);
            let (sa, sb, sc) = (stride(a)?, stride(a)?, stride(a)?);
            let slot = a.u16()?;
            MatMulBatched {
                c: o[0],
                a: o[1],
                b: o[2],
                ta,
                tb,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                alpha,
                beta,
                outer,
                inner,
                sa,
                sb,
                sc,
                slot,
            }
        }
        Opcode::HeadsToRows => {
            want(2)?;
            HeadsToRows {
                dst: o[0],
                src: o[1],
                b: a.u32()?,
                t: a.u32()?,
                heads: a.u32()?,
                hd: a.u32()?,
                group: a.u32()?,
            }
        }
        Opcode::Embed => {
            want(3)?;
            Embed {
                out: o[0],
                table: o[1],
                tokens: o[2],
                n: a.u32()?,
                dim: a.u32()?,
            }
        }
        Opcode::EmbedBwd => {
            want(3)?;
            EmbedBwd {
                dtable: o[0],
                dout: o[1],
                tokens: o[2],
                n: a.u32()?,
                dim: a.u32()?,
            }
        }
        Opcode::RmsNorm => {
            want(4)?;
            RmsNorm {
                out: o[0],
                rstd: o[1],
                x: o[2],
                w: o[3],
                rows: a.u32()?,
                dim: a.u32()?,
                eps: a.f32()?,
            }
        }
        Opcode::RmsNormBwd => {
            want(6)?;
            RmsNormBwd {
                dx: o[0],
                dw: o[1],
                dy: o[2],
                x: o[3],
                w: o[4],
                rstd: o[5],
                rows: a.u32()?,
                dim: a.u32()?,
                accumulate: a.u8()? != 0,
            }
        }
        Opcode::Rope | Opcode::RopeBwd => {
            want(1)?;
            let (rows, heads, hd, seq, theta) = (a.u32()?, a.u32()?, a.u32()?, a.u32()?, a.f32()?);
            if op == Opcode::Rope {
                Rope {
                    x: o[0],
                    rows,
                    heads,
                    hd,
                    seq,
                    theta,
                }
            } else {
                RopeBwd {
                    x: o[0],
                    rows,
                    heads,
                    hd,
                    seq,
                    theta,
                }
            }
        }
        Opcode::SoftmaxCausal => {
            want(1)?;
            SoftmaxCausal {
                x: o[0],
                rows: a.u32()?,
                cols: a.u32()?,
                scale: a.f32()?,
            }
        }
        Opcode::SoftmaxBwd => {
            want(2)?;
            SoftmaxBwd {
                dp: o[0],
                p: o[1],
                rows: a.u32()?,
                cols: a.u32()?,
                scale: a.f32()?,
            }
        }
        Opcode::SiluMul => {
            want(3)?;
            SiluMul {
                g: o[0],
                a: o[1],
                b: o[2],
                n: a.u32()?,
            }
        }
        Opcode::SiluMulBwd => {
            want(5)?;
            SiluMulBwd {
                da: o[0],
                db: o[1],
                dg: o[2],
                a: o[3],
                b: o[4],
                n: a.u32()?,
            }
        }
        Opcode::Xent => {
            want(2)?;
            Xent {
                logits: o[0],
                targets: o[1],
                rows: a.u32()?,
                vocab: a.u32()?,
                grad_scale: a.f32()?,
            }
        }
        Opcode::SumSq => SumSq {
            bufs: o.to_vec(),
            dst: a.u8()?,
        },
        Opcode::ClipCoef => {
            want(0)?;
            ClipCoef {
                dst: a.u8()?,
                src: a.u8()?,
                max_norm: a.f32()?,
            }
        }
        Opcode::AdamW => {
            want(4)?;
            AdamW {
                p: o[0],
                g: o[1],
                m: o[2],
                v: o[3],
                lr: a.u8()?,
                step: a.u8()?,
                clip: a.u8()?,
                beta1: a.f32()?,
                beta2: a.f32()?,
                eps: a.f32()?,
                wd: a.f32()?,
            }
        }
        Opcode::Quant => {
            want(1)?;
            Quant {
                x: o[0],
                format: a.u8()?,
            }
        }
        Opcode::SetReg => {
            want(0)?;
            SetReg {
                dst: a.u8()?,
                value: a.f32()?,
            }
        }
        Opcode::Alloc => {
            want(1)?;
            Alloc { v: o[0] }
        }
        Opcode::Free => {
            want(1)?;
            Free { v: o[0] }
        }
    })
}

/// Human-readable listing, one instruction per line (`forge disasm`).
pub fn disassemble(p: &Program) -> String {
    let name = |v: &V| {
        p.bufs
            .get(v.0 as usize)
            .map_or("?".to_string(), |b| format!("%{}", b.name))
    };
    let mut out = String::new();
    for (pc, ins) in p.instrs.iter().enumerate() {
        let mut i = ins.clone();
        let ops: Vec<String> = i.buffers_mut().into_iter().map(|v| name(v)).collect();
        let detail = match ins {
            Instr::MatMul {
                ta,
                tb,
                m,
                n,
                k,
                alpha,
                beta,
                slot,
                ..
            } => {
                format!(
                    " [{}{} {m}x{n}x{k} a={alpha} b={beta} slot={slot}]",
                    if *ta { "T" } else { "N" },
                    if *tb { "T" } else { "N" }
                )
            }
            Instr::MatMulBatched {
                m,
                n,
                k,
                outer,
                inner,
                slot,
                ..
            } => format!(" [{outer}x{inner} × {m}x{n}x{k} slot={slot}]"),
            _ => String::new(),
        };
        out.push_str(&format!(
            "{pc:5}  {:<15} {}{}\n",
            ins.opcode().name(),
            ops.join(", "),
            detail
        ));
    }
    out
}

#[cfg(test)]
mod tests;
