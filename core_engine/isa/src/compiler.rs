//! Graph compiler: verification, fusion, dead-code elimination and
//! liveness-based HBVM placement.
//!
//! Passes (in order):
//! 1. `fuse_residual`: `MATMUL t = A·B` immediately followed by `ADD y = t + r`
//!    (either operand order), with `t` transient and used nowhere else,
//!    becomes `COPY y ← r; MATMUL y += A·B` — the addition moves into the GEMM
//!    epilogue and the intermediate `t` disappears.
//! 2. `eliminate_dead`: drops instructions whose only effects are writes to
//!    transient buffers that are never read afterwards.
//! 3. `place`: inserts `ALLOC` before the first use and `FREE` after the last
//!    use of every transient buffer, so peak memory follows liveness rather
//!    than the sum of all activations.

use crate::{Instr, IsaError, Kind, Program, V};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct CompileOptions {
    pub fuse: bool,
    pub eliminate_dead: bool,
    pub place: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            fuse: true,
            eliminate_dead: true,
            place: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompileReport {
    pub instrs_in: usize,
    pub instrs_out: usize,
    pub fused_residual: usize,
    pub dead_removed: usize,
    /// Words if every transient buffer were alive at once.
    pub transient_words_naive: u64,
    /// Peak words of simultaneously live transient buffers after placement.
    pub transient_words_peak: u64,
    pub persistent_words: u64,
}

fn uses(p: &Program) -> HashMap<V, usize> {
    let mut n = HashMap::new();
    for ins in &p.instrs {
        let mut i = ins.clone();
        for v in i.buffers_mut() {
            *n.entry(*v).or_insert(0) += 1;
        }
    }
    n
}

fn fuse_residual(p: &mut Program) -> usize {
    let counts = uses(p);
    let mut out = Vec::with_capacity(p.instrs.len());
    let mut fused = 0;
    let mut i = 0;
    while i < p.instrs.len() {
        if let (
            Some(Instr::MatMul {
                c: t, a, b, beta, ..
            }),
            Some(Instr::Add { dst: y, a: x, b: z }),
        ) = (p.instrs.get(i), p.instrs.get(i + 1))
        {
            let r = if x == t {
                Some(*z)
            } else if z == t {
                Some(*x)
            } else {
                None
            };
            let transient = p.bufs[t.0 as usize].kind == Kind::Transient;
            if let Some(r) = r {
                if *beta == 0.0
                    && transient
                    && counts.get(t) == Some(&2)
                    && r != *t
                    && *y != *a
                    && *y != *b
                    && *y != r
                {
                    let mut mm = p.instrs[i].clone();
                    if let Instr::MatMul { c, beta, .. } = &mut mm {
                        *c = *y;
                        *beta = 1.0;
                    }
                    out.push(Instr::Copy { dst: *y, src: r });
                    out.push(mm);
                    fused += 1;
                    i += 2;
                    continue;
                }
            }
        }
        out.push(p.instrs[i].clone());
        i += 1;
    }
    p.instrs = out;
    fused
}

fn has_register_effect(ins: &Instr) -> bool {
    matches!(
        ins,
        Instr::Xent { .. } | Instr::SumSq { .. } | Instr::ClipCoef { .. } | Instr::SetReg { .. }
    )
}

fn eliminate_dead(p: &mut Program) -> usize {
    let mut removed = 0;
    loop {
        let mut read_later: Vec<bool> = vec![false; p.bufs.len()];
        let mut keep = vec![true; p.instrs.len()];
        for (pc, ins) in p.instrs.iter().enumerate().rev() {
            let (outs, ins_) = ins.io();
            let all_dead_transient = !outs.is_empty()
                && outs.iter().all(|v| {
                    p.bufs[v.0 as usize].kind == Kind::Transient && !read_later[v.0 as usize]
                });
            if all_dead_transient && !has_register_effect(ins) {
                keep[pc] = false;
                continue;
            }
            for v in ins_ {
                read_later[v.0 as usize] = true;
            }
        }
        let before = p.instrs.len();
        let mut k = keep.into_iter();
        p.instrs.retain(|_| k.next().unwrap());
        let now = before - p.instrs.len();
        removed += now;
        if now == 0 {
            return removed;
        }
    }
}

fn place(p: &mut Program) -> (u64, u64) {
    let n = p.bufs.len();
    let mut first = vec![usize::MAX; n];
    let mut last = vec![0usize; n];
    for (pc, ins) in p.instrs.iter().enumerate() {
        let mut i = ins.clone();
        for v in i.buffers_mut() {
            let k = v.0 as usize;
            first[k] = first[k].min(pc);
            last[k] = last[k].max(pc);
        }
    }
    let mut allocs: Vec<Vec<V>> = vec![Vec::new(); p.instrs.len()];
    let mut frees: Vec<Vec<V>> = vec![Vec::new(); p.instrs.len()];
    let mut naive = 0u64;
    for k in 0..n {
        if p.bufs[k].kind == Kind::Transient && first[k] != usize::MAX {
            allocs[first[k]].push(V(k as u32));
            frees[last[k]].push(V(k as u32));
            naive += p.bufs[k].len as u64;
        }
    }
    let mut out = Vec::with_capacity(p.instrs.len() + 2 * n);
    let (mut live, mut peak) = (0u64, 0u64);
    for (pc, ins) in p.instrs.drain(..).enumerate() {
        for v in &allocs[pc] {
            out.push(Instr::Alloc { v: *v });
            live += p.bufs[v.0 as usize].len as u64;
        }
        peak = peak.max(live);
        out.push(ins);
        for v in &frees[pc] {
            out.push(Instr::Free { v: *v });
            live -= p.bufs[v.0 as usize].len as u64;
        }
    }
    p.instrs = out;
    (naive, peak)
}

/// Optimise and place a program. The input must not contain ALLOC/FREE.
pub fn compile(mut p: Program, opt: CompileOptions) -> Result<(Program, CompileReport), IsaError> {
    p.verify()?;
    if p.instrs
        .iter()
        .any(|i| matches!(i, Instr::Alloc { .. } | Instr::Free { .. }))
    {
        return Err(IsaError(
            "compile: input already contains placement instructions".into(),
        ));
    }
    let mut r = CompileReport {
        instrs_in: p.instrs.len(),
        persistent_words: p
            .bufs
            .iter()
            .filter(|b| b.kind == Kind::Persistent)
            .map(|b| b.len as u64)
            .sum(),
        ..CompileReport::default()
    };
    if opt.fuse {
        r.fused_residual = fuse_residual(&mut p);
    }
    if opt.eliminate_dead {
        r.dead_removed = eliminate_dead(&mut p);
    }
    if opt.place {
        let (naive, peak) = place(&mut p);
        r.transient_words_naive = naive;
        r.transient_words_peak = peak;
    }
    r.instrs_out = p.instrs.len();
    p.verify()?;
    Ok((p, r))
}
