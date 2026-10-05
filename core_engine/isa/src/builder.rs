//! Program builder: declares buffers and emits typed instructions.

use crate::{BufDecl, Instr, Kind, Program, Reg, Stride, V};

#[derive(Default)]
pub struct Builder {
    p: Program,
    slots: u16,
}

impl Builder {
    pub fn new() -> Builder {
        Builder::default()
    }

    pub fn persistent(&mut self, name: impl Into<String>, len: usize) -> V {
        self.decl(name.into(), len, Kind::Persistent)
    }

    pub fn transient(&mut self, name: impl Into<String>, len: usize) -> V {
        self.decl(name.into(), len, Kind::Transient)
    }

    fn decl(&mut self, name: String, len: usize, kind: Kind) -> V {
        assert!(
            len > 0 && len <= u32::MAX as usize,
            "buffer {name}: bad length {len}"
        );
        self.p.bufs.push(BufDecl {
            name,
            len: len as u32,
            kind,
        });
        V((self.p.bufs.len() - 1) as u32)
    }

    pub fn emit(&mut self, ins: Instr) {
        self.p.instrs.push(ins);
    }

    fn slot(&mut self) -> u16 {
        self.slots += 1;
        self.slots - 1
    }

    /// C[m×n] = op(A)·op(B) (+ C when accumulate). Row-major, dense leading dims.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &mut self,
        c: V,
        a: V,
        b: V,
        ta: bool,
        tb: bool,
        m: usize,
        n: usize,
        k: usize,
        accumulate: bool,
    ) {
        let lda = if ta { m } else { k };
        let ldb = if tb { k } else { n };
        let slot = self.slot();
        self.emit(Instr::MatMul {
            c,
            a,
            b,
            ta,
            tb,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: lda as u32,
            ldb: ldb as u32,
            ldc: n as u32,
            alpha: 1.0,
            beta: if accumulate { 1.0 } else { 0.0 },
            slot,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_batched(
        &mut self,
        c: V,
        a: V,
        b: V,
        (ta, tb): (bool, bool),
        (m, n, k): (usize, usize, usize),
        (lda, ldb, ldc): (usize, usize, usize),
        (outer, inner): (usize, usize),
        (sa, sb, sc): (Stride, Stride, Stride),
        alpha: f32,
        accumulate: bool,
    ) {
        let slot = self.slot();
        self.emit(Instr::MatMulBatched {
            c,
            a,
            b,
            ta,
            tb,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: lda as u32,
            ldb: ldb as u32,
            ldc: ldc as u32,
            alpha,
            beta: if accumulate { 1.0 } else { 0.0 },
            outer: outer as u32,
            inner: inner as u32,
            sa,
            sb,
            sc,
            slot,
        });
    }

    pub fn set_reg(&mut self, dst: Reg, value: f32) {
        self.emit(Instr::SetReg { dst, value });
    }

    pub fn finish(self) -> Program {
        self.p
    }
}
