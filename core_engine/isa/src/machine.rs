//! The interpreter: binds virtual buffers to HBVM memory, executes verified
//! programs instruction by instruction on the kernel engines, and accounts
//! calls, nanoseconds, FLOPs and bytes per opcode.

use crate::{Instr, IsaError, Kind, Opcode, Program, Stride, NUM_REGS, V};
use forge_hbvm::{Buf, Hbvm, HbvmError, HbvmStats};
use forge_kernels::{GemmVariant, Pool, Trans};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpStats {
    pub calls: u64,
    pub ns: u64,
    pub flops: f64,
    pub bytes: f64,
}

#[derive(Debug)]
pub enum MachineError {
    Hbvm(HbvmError),
    Isa(IsaError),
    Unbound {
        pc: usize,
        buffer: String,
    },
    BadToken {
        pc: usize,
        index: usize,
        value: f32,
        limit: usize,
    },
    Unsupported(&'static str),
    Persistent {
        name: String,
        have: usize,
        want: usize,
    },
}

impl fmt::Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MachineError::Hbvm(e) => write!(f, "{e}"),
            MachineError::Isa(e) => write!(f, "{e}"),
            MachineError::Unbound { pc, buffer } => {
                write!(f, "pc {pc}: buffer %{buffer} is not bound (missing ALLOC?)")
            }
            MachineError::BadToken {
                pc,
                index,
                value,
                limit,
            } => {
                write!(
                    f,
                    "pc {pc}: token[{index}] = {value} is not an integer id below {limit}"
                )
            }
            MachineError::Unsupported(what) => write!(f, "unsupported: {what}"),
            MachineError::Persistent { name, have, want } => {
                write!(
                    f,
                    "persistent buffer %{name} exists with {have} words, program declares {want}"
                )
            }
        }
    }
}

impl std::error::Error for MachineError {}

impl From<HbvmError> for MachineError {
    fn from(e: HbvmError) -> Self {
        MachineError::Hbvm(e)
    }
}

impl From<IsaError> for MachineError {
    fn from(e: IsaError) -> Self {
        MachineError::Isa(e)
    }
}

/// A verified program bound to a machine.
pub struct Loaded {
    pub program: Program,
    vmap: Vec<Option<Buf>>,
    /// Engine per MATMUL slot.
    variants: Vec<GemmVariant>,
    placed: bool,
}

impl Loaded {
    pub fn variants(&self) -> &[GemmVariant] {
        &self.variants
    }
    pub fn set_all_variants(&mut self, v: GemmVariant) {
        self.variants.iter_mut().for_each(|x| *x = v);
    }
}

/// Result of a JIT autotune over one GEMM shape.
#[derive(Clone, Debug)]
pub struct TuneReport {
    pub shape: (bool, bool, u32, u32, u32),
    pub results: Vec<(GemmVariant, f64)>,
    pub chosen: GemmVariant,
}

type RopeTable = Arc<(Vec<f32>, Vec<f32>)>;

pub struct Machine {
    hbvm: Hbvm,
    persistent: HashMap<String, Buf>,
    pub regs: [f32; NUM_REGS],
    pool: Pool,
    default_variant: GemmVariant,
    stats: Vec<OpStats>,
    rope: HashMap<(u32, u32, u32), RopeTable>,
}

impl Machine {
    /// `capacity_words`: HBVM size in f32 words.
    pub fn new(capacity_words: usize, threads: usize) -> Machine {
        Machine {
            hbvm: Hbvm::new(capacity_words),
            persistent: HashMap::new(),
            regs: [0.0; NUM_REGS],
            pool: Pool::new(threads),
            default_variant: forge_kernels::best_available(),
            stats: vec![OpStats::default(); 256],
            rope: HashMap::new(),
        }
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Core scaling: rebuild the worker pool with `n` threads.
    pub fn set_threads(&mut self, n: usize) {
        self.pool = Pool::new(n.max(1));
    }

    pub fn default_variant(&self) -> GemmVariant {
        self.default_variant
    }

    pub fn set_default_variant(&mut self, v: GemmVariant) {
        self.default_variant = v;
    }

    pub fn hbvm_stats(&self) -> HbvmStats {
        self.hbvm.stats()
    }

    pub fn op_stats(&self) -> Vec<(Opcode, OpStats)> {
        Opcode::ALL
            .into_iter()
            .map(|o| (o, self.stats[o as usize]))
            .filter(|(_, s)| s.calls > 0)
            .collect()
    }

    pub fn reset_stats(&mut self) {
        self.stats.iter_mut().for_each(|s| *s = OpStats::default());
    }

    /// Verify and bind a program. Persistent buffers are shared by name across
    /// programs; transient buffers are bound by ALLOC/FREE, or for the lifetime
    /// of the program when it carries no placement instructions.
    pub fn load(&mut self, program: Program) -> Result<Loaded, MachineError> {
        program.verify()?;
        let placed = program
            .instrs
            .iter()
            .any(|i| matches!(i, Instr::Alloc { .. }));
        let mut vmap = vec![None; program.bufs.len()];
        for (k, d) in program.bufs.iter().enumerate() {
            match d.kind {
                Kind::Persistent => {
                    let buf = match self.persistent.get(&d.name) {
                        Some(&b) => {
                            let have = self.hbvm.len(b);
                            if have != d.len as usize {
                                return Err(MachineError::Persistent {
                                    name: d.name.clone(),
                                    have,
                                    want: d.len as usize,
                                });
                            }
                            b
                        }
                        None => {
                            let b = self.hbvm.alloc(d.len as usize)?;
                            self.persistent.insert(d.name.clone(), b);
                            b
                        }
                    };
                    vmap[k] = Some(buf);
                }
                Kind::Transient if !placed => vmap[k] = Some(self.hbvm.alloc(d.len as usize)?),
                Kind::Transient => {}
            }
        }
        let nslots = program
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::MatMul { slot, .. } | Instr::MatMulBatched { slot, .. } => {
                    Some(*slot as usize + 1)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        Ok(Loaded {
            program,
            vmap,
            variants: vec![self.default_variant; nslots],
            placed,
        })
    }

    /// Release the transient memory a program holds outside of placement.
    pub fn unload(&mut self, lp: Loaded) -> Result<(), MachineError> {
        for (k, d) in lp.program.bufs.iter().enumerate() {
            if d.kind == Kind::Transient {
                if let Some(b) = lp.vmap[k] {
                    self.hbvm.free(b)?;
                }
            }
        }
        Ok(())
    }

    pub fn persistent(&self, name: &str) -> Option<&[f32]> {
        self.persistent.get(name).map(|&b| self.hbvm.slice(b))
    }

    pub fn persistent_mut(&mut self, name: &str) -> Option<&mut [f32]> {
        let b = *self.persistent.get(name)?;
        Some(self.hbvm.slice_mut(b))
    }

    pub fn persistent_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.persistent.keys().cloned().collect();
        v.sort();
        v
    }

    /// Read a buffer of a loaded program (transient buffers only while bound).
    pub fn read(&self, lp: &Loaded, v: V) -> Option<&[f32]> {
        lp.vmap[v.0 as usize].map(|b| self.hbvm.slice(b))
    }

    pub fn run(&mut self, lp: &mut Loaded) -> Result<(), MachineError> {
        for pc in 0..lp.program.instrs.len() {
            let ins = &lp.program.instrs[pc];
            match ins {
                Instr::Alloc { v } => {
                    let len = lp.program.len_of(*v);
                    lp.vmap[v.0 as usize] = Some(self.hbvm.alloc(len)?);
                    continue;
                }
                Instr::Free { v } => {
                    if let Some(b) = lp.vmap[v.0 as usize].take() {
                        self.hbvm.free(b)?;
                    }
                    continue;
                }
                _ => {}
            }
            let t0 = Instant::now();
            let bytes = exec(pc, ins, &lp.program, &lp.vmap, &lp.variants, self)?;
            let s = &mut self.stats[ins.opcode() as usize];
            s.calls += 1;
            s.ns += t0.elapsed().as_nanos() as u64;
            s.flops += ins.flops();
            s.bytes += bytes;
        }
        debug_assert!(
            !lp.placed
                || lp
                    .program
                    .bufs
                    .iter()
                    .enumerate()
                    .all(|(k, d)| d.kind == Kind::Persistent || lp.vmap[k].is_none())
        );
        Ok(())
    }

    /// JIT specialisation: measure every engine on every distinct GEMM shape of
    /// the program and bind the fastest engine to each MATMUL slot.
    pub fn autotune(&mut self, lp: &mut Loaded) -> Vec<TuneReport> {
        let mut by_shape: HashMap<(bool, bool, u32, u32, u32), Vec<u16>> = HashMap::new();
        for ins in &lp.program.instrs {
            match ins {
                Instr::MatMul {
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    slot,
                    ..
                }
                | Instr::MatMulBatched {
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    slot,
                    ..
                } => {
                    by_shape
                        .entry((*ta, *tb, *m, *n, *k))
                        .or_default()
                        .push(*slot);
                }
                _ => {}
            }
        }
        let mut shapes: Vec<_> = by_shape.into_iter().collect();
        shapes.sort_by_key(|(s, _)| *s);
        let mut reports = Vec::new();
        for (shape, slots) in shapes {
            let (ta, tb, m, n, k) = shape;
            let tr = |t: bool| if t { Trans::T } else { Trans::N };
            let results = forge_kernels::autotune(
                &self.pool,
                tr(ta),
                tr(tb),
                m as usize,
                n as usize,
                k as usize,
            );
            let chosen = results.first().map_or(self.default_variant, |r| r.variant);
            for s in slots {
                lp.variants[s as usize] = chosen;
            }
            reports.push(TuneReport {
                shape,
                results: results.iter().map(|r| (r.variant, r.gflops)).collect(),
                chosen,
            });
        }
        reports
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

struct Views<'a> {
    outs: Vec<&'a mut [f32]>,
    ins: Vec<&'a [f32]>,
}

/// Resolve operands to slices. Outputs must be pairwise distinct and distinct
/// from inputs (the verifier guarantees this for well-formed programs; it is
/// re-checked here because the views are created from raw pointers).
fn views<'a>(
    pc: usize,
    p: &Program,
    vmap: &[Option<Buf>],
    hb: &'a mut Hbvm,
    outs: &[V],
    ins: &[V],
) -> Result<Views<'a>, MachineError> {
    let bind = |v: &V| {
        vmap[v.0 as usize].ok_or_else(|| MachineError::Unbound {
            pc,
            buffer: p.bufs[v.0 as usize].name.clone(),
        })
    };
    let ob: Vec<Buf> = outs.iter().map(bind).collect::<Result<_, _>>()?;
    let ib: Vec<Buf> = ins.iter().map(bind).collect::<Result<_, _>>()?;
    for (i, a) in ob.iter().enumerate() {
        if ob[i + 1..].contains(a) || ib.contains(a) {
            return Err(MachineError::Isa(IsaError(format!(
                "pc {pc}: aliased operands at run time"
            ))));
        }
    }
    let raw: Vec<(*mut f32, usize)> = ob
        .iter()
        .chain(ib.iter())
        .map(|&b| hb.raw_parts(b))
        .collect();
    // SAFETY: each pointer covers one live HBVM allocation; allocations never
    // overlap, outputs are pairwise distinct and disjoint from inputs (checked
    // above), and the slices do not outlive the `&mut Hbvm` borrow.
    let outs = raw[..ob.len()]
        .iter()
        .map(|&(p, n)| unsafe { std::slice::from_raw_parts_mut(p, n) })
        .collect();
    let ins = raw[ob.len()..]
        .iter()
        .map(|&(p, n)| unsafe { std::slice::from_raw_parts(p as *const f32, n) })
        .collect();
    Ok(Views { outs, ins })
}

const CHUNK: usize = 1 << 14;

fn tr(t: bool) -> Trans {
    if t {
        Trans::T
    } else {
        Trans::N
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn token(pc: usize, index: usize, value: f32, limit: usize) -> Result<usize, MachineError> {
    if value >= 0.0 && value.fract() == 0.0 && (value as usize) < limit {
        Ok(value as usize)
    } else {
        Err(MachineError::BadToken {
            pc,
            index,
            value,
            limit,
        })
    }
}

fn rope_table(m: &mut Machine, seq: u32, hd: u32, theta: f32) -> RopeTable {
    m.rope
        .entry((seq, hd, theta.to_bits()))
        .or_insert_with(|| {
            // Same arithmetic as scp_model.precompute_rope, in f32.
            let half = (hd / 2) as usize;
            let inv: Vec<f32> = (0..half)
                .map(|i| 1.0 / theta.powf((2 * i) as f32 / hd as f32))
                .collect();
            let mut cos = vec![0.0f32; seq as usize * half];
            let mut sin = vec![0.0f32; seq as usize * half];
            for t in 0..seq as usize {
                for i in 0..half {
                    let f = t as f32 * inv[i];
                    cos[t * half + i] = f.cos();
                    sin[t * half + i] = f.sin();
                }
            }
            Arc::new((cos, sin))
        })
        .clone()
}

/// Execute one instruction; returns the bytes it touched (for telemetry).
fn exec(
    pc: usize,
    ins: &Instr,
    p: &Program,
    vmap: &[Option<Buf>],
    variants: &[GemmVariant],
    m: &mut Machine,
) -> Result<f64, MachineError> {
    use Instr::*;
    // Precompute tables that need `&mut Machine` before borrowing memory.
    let rope = match ins {
        Rope { seq, hd, theta, .. } | RopeBwd { seq, hd, theta, .. } => {
            Some(rope_table(m, *seq, *hd, *theta))
        }
        _ => None,
    };
    let (outs, reads) = ins.io();
    let outs_only: Vec<V> = outs.clone();
    // In-place operands appear in both lists; views() takes them as outputs only.
    let ins_only: Vec<V> = reads
        .iter()
        .copied()
        .filter(|v| !outs.contains(v))
        .collect();
    let bytes = 4.0
        * outs_only
            .iter()
            .chain(ins_only.iter())
            .map(|v| p.len_of(*v) as f64)
            .sum::<f64>();
    let pool = &m.pool;
    let regs = &mut m.regs;
    let mut vw = views(pc, p, vmap, &mut m.hbvm, &outs_only, &ins_only)?;
    let idx = |v: &V| {
        ins_only
            .iter()
            .position(|x| x == v)
            .expect("operand listed")
    };
    match ins {
        Zero { .. } => pool.install(|| vw.outs[0].par_chunks_mut(CHUNK).for_each(|c| c.fill(0.0))),
        Copy { src, .. } => {
            let s = vw.ins[idx(src)];
            pool.install(|| {
                vw.outs[0]
                    .par_chunks_mut(CHUNK)
                    .zip(s.par_chunks(CHUNK))
                    .for_each(|(d, s)| d.copy_from_slice(s))
            })
        }
        Add { a, b, .. } | Mul { a, b, .. } => {
            let (x, y) = (vw.ins[idx(a)], vw.ins[idx(b)]);
            let mul = matches!(ins, Mul { .. });
            pool.install(|| {
                vw.outs[0]
                    .par_chunks_mut(CHUNK)
                    .zip(x.par_chunks(CHUNK))
                    .zip(y.par_chunks(CHUNK))
                    .for_each(|((d, x), y)| {
                        for i in 0..d.len() {
                            d[i] = if mul { x[i] * y[i] } else { x[i] + y[i] };
                        }
                    })
            })
        }
        AddInplace { src, .. } => {
            let s = vw.ins[idx(src)];
            pool.install(|| {
                vw.outs[0]
                    .par_chunks_mut(CHUNK)
                    .zip(s.par_chunks(CHUNK))
                    .for_each(|(d, s)| d.iter_mut().zip(s).for_each(|(d, s)| *d += s))
            })
        }
        Scale { value, reg, .. } => {
            let f = reg.map_or(*value, |r| regs[r as usize]);
            pool.install(|| {
                vw.outs[0]
                    .par_chunks_mut(CHUNK)
                    .for_each(|c| c.iter_mut().for_each(|x| *x *= f))
            })
        }
        MatMul {
            a,
            b,
            ta,
            tb,
            m: mm,
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
            let v = variants[*slot as usize];
            let (av, bv) = (vw.ins[idx(a)], vw.ins[idx(b)]);
            forge_kernels::gemm(
                pool,
                v,
                tr(*ta),
                tr(*tb),
                *mm as usize,
                *n as usize,
                *k as usize,
                *alpha,
                av,
                *lda as usize,
                bv,
                *ldb as usize,
                *beta,
                vw.outs[0],
                *ldc as usize,
            );
        }
        MatMulBatched {
            a,
            b,
            ta,
            tb,
            m: mm,
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
            let v = variants[*slot as usize];
            let (av, bv) = (vw.ins[idx(a)], vw.ins[idx(b)]);
            let (mm, n, k) = (*mm as usize, *n as usize, *k as usize);
            let (lda, ldb, ldc) = (*lda as usize, *ldb as usize, *ldc as usize);
            let (ar, ac) = if *ta { (k, mm) } else { (mm, k) };
            let (br, bc) = if *tb { (n, k) } else { (k, n) };
            let (span_a, span_b, span_c) =
                ((ar - 1) * lda + ac, (br - 1) * ldb + bc, (mm - 1) * ldc + n);
            let off = |s: &Stride, o: usize, i: usize| {
                o * s.outer as usize + (i / s.div as usize) * s.inner as usize
            };
            let inner = *inner as usize;
            let c_base = vw.outs[0].as_mut_ptr() as usize;
            pool.install(|| {
                (0..*outer as usize * inner).into_par_iter().for_each(|ix| {
                    let (o, i) = (ix / inner, ix % inner);
                    let (oa, ob, oc) = (off(sa, o, i), off(sb, o, i), off(sc, o, i));
                    // SAFETY: the verifier guarantees stacked, non-overlapping
                    // output matrices, so each task writes a disjoint range.
                    let cm = unsafe {
                        std::slice::from_raw_parts_mut((c_base as *mut f32).add(oc), span_c)
                    };
                    forge_kernels::gemm(
                        pool,
                        v,
                        tr(*ta),
                        tr(*tb),
                        mm,
                        n,
                        k,
                        *alpha,
                        &av[oa..oa + span_a],
                        lda,
                        &bv[ob..ob + span_b],
                        ldb,
                        *beta,
                        cm,
                        ldc,
                    );
                })
            });
        }
        HeadsToRows {
            src,
            b,
            t,
            heads,
            hd,
            group,
            ..
        } => {
            let s = vw.ins[idx(src)];
            let (t, heads, hd, group) =
                (*t as usize, *heads as usize, *hd as usize, *group as usize);
            let out_heads = heads / group;
            let _ = b;
            pool.install(|| {
                vw.outs[0]
                    .par_chunks_mut(out_heads * hd)
                    .enumerate()
                    .for_each(|(row, d)| {
                        let (bb, tt) = (row / t, row % t);
                        for hh in 0..out_heads {
                            let dst = &mut d[hh * hd..(hh + 1) * hd];
                            for g in 0..group {
                                let h = hh * group + g;
                                let so = ((bb * heads + h) * t + tt) * hd;
                                let srow = &s[so..so + hd];
                                if g == 0 {
                                    dst.copy_from_slice(srow);
                                } else {
                                    dst.iter_mut().zip(srow).for_each(|(d, x)| *d += x);
                                }
                            }
                        }
                    })
            });
        }
        Embed {
            table,
            tokens,
            n,
            dim,
            ..
        } => {
            let (tab, tok) = (vw.ins[idx(table)], vw.ins[idx(tokens)]);
            let dim = *dim as usize;
            let vocab = tab.len() / dim;
            let ids: Vec<usize> = (0..*n as usize)
                .map(|i| token(pc, i, tok[i], vocab))
                .collect::<Result<_, _>>()?;
            pool.install(|| {
                vw.outs[0][..ids.len() * dim]
                    .par_chunks_mut(dim)
                    .zip(ids.par_iter())
                    .for_each(|(row, &id)| row.copy_from_slice(&tab[id * dim..(id + 1) * dim]))
            });
        }
        EmbedBwd {
            dout,
            tokens,
            n,
            dim,
            ..
        } => {
            let (dy, tok) = (vw.ins[idx(dout)], vw.ins[idx(tokens)]);
            let dim = *dim as usize;
            let vocab = vw.outs[0].len() / dim;
            let ids: Vec<usize> = (0..*n as usize)
                .map(|i| token(pc, i, tok[i], vocab))
                .collect::<Result<_, _>>()?;
            let base = vw.outs[0].as_mut_ptr() as usize;
            let cols = 64usize;
            pool.install(|| {
                (0..dim.div_ceil(cols)).into_par_iter().for_each(|cb| {
                    let (c0, c1) = (cb * cols, ((cb + 1) * cols).min(dim));
                    for (i, &id) in ids.iter().enumerate() {
                        // SAFETY: tasks own disjoint column ranges [c0, c1) of every row.
                        let row = unsafe {
                            std::slice::from_raw_parts_mut(
                                (base as *mut f32).add(id * dim + c0),
                                c1 - c0,
                            )
                        };
                        row.iter_mut()
                            .zip(&dy[i * dim + c0..i * dim + c1])
                            .for_each(|(d, x)| *d += x);
                    }
                })
            });
        }
        RmsNorm {
            x,
            w,
            rows,
            dim,
            eps,
            ..
        } => {
            let (xv, wv) = (vw.ins[idx(x)], vw.ins[idx(w)]);
            let (rows, dim) = (*rows as usize, *dim as usize);
            let (out, rstd) = vw.outs.split_at_mut(1);
            pool.install(|| {
                out[0][..rows * dim]
                    .par_chunks_mut(dim)
                    .zip(rstd[0][..rows].par_iter_mut())
                    .zip(xv.par_chunks(dim))
                    .for_each(|((o, r), x)| {
                        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
                        let inv = 1.0 / (ss + eps).sqrt();
                        *r = inv;
                        for j in 0..dim {
                            o[j] = x[j] * inv * wv[j];
                        }
                    })
            });
        }
        RmsNormBwd {
            dy,
            x,
            w,
            rstd,
            rows,
            dim,
            accumulate,
            ..
        } => {
            let (dyv, xv, wv, rv) = (
                vw.ins[idx(dy)],
                vw.ins[idx(x)],
                vw.ins[idx(w)],
                vw.ins[idx(rstd)],
            );
            let (rows, dim, acc) = (*rows as usize, *dim as usize, *accumulate);
            let (dx, dw) = vw.outs.split_at_mut(1);
            pool.install(|| {
                dx[0][..rows * dim]
                    .par_chunks_mut(dim)
                    .enumerate()
                    .for_each(|(r, d)| {
                        let (x, g) = (&xv[r * dim..(r + 1) * dim], &dyv[r * dim..(r + 1) * dim]);
                        let inv = rv[r];
                        let dot: f32 = (0..dim).map(|j| g[j] * wv[j] * x[j]).sum();
                        let c = inv * inv * dot / dim as f32;
                        for j in 0..dim {
                            let v = inv * (g[j] * wv[j] - x[j] * c);
                            d[j] = if acc { d[j] + v } else { v };
                        }
                    });
                let part = (0..rows)
                    .into_par_iter()
                    .fold(
                        || vec![0.0f32; dim],
                        |mut a, r| {
                            let inv = rv[r];
                            for j in 0..dim {
                                a[j] += dyv[r * dim + j] * xv[r * dim + j] * inv;
                            }
                            a
                        },
                    )
                    .reduce(
                        || vec![0.0f32; dim],
                        |mut a, b| {
                            a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                            a
                        },
                    );
                dw[0][..dim].iter_mut().zip(part).for_each(|(d, p)| *d += p);
            });
        }
        Rope {
            rows,
            heads,
            hd,
            seq,
            ..
        }
        | RopeBwd {
            rows,
            heads,
            hd,
            seq,
            ..
        } => {
            let tab = rope.expect("rope table");
            let (cos, sin) = (&tab.0, &tab.1);
            let (heads, hd, seq) = (*heads as usize, *hd as usize, *seq as usize);
            let half = hd / 2;
            let inverse = matches!(ins, RopeBwd { .. });
            pool.install(|| {
                vw.outs[0][..*rows as usize * heads * hd]
                    .par_chunks_mut(heads * hd)
                    .enumerate()
                    .for_each(|(r, row)| {
                        let t = r % seq;
                        let (c, s) = (
                            &cos[t * half..(t + 1) * half],
                            &sin[t * half..(t + 1) * half],
                        );
                        for h in 0..heads {
                            let x = &mut row[h * hd..(h + 1) * hd];
                            for i in 0..half {
                                let (a, b) = (x[2 * i], x[2 * i + 1]);
                                let sn = if inverse { -s[i] } else { s[i] };
                                x[2 * i] = a * c[i] - b * sn;
                                x[2 * i + 1] = a * sn + b * c[i];
                            }
                        }
                    })
            });
        }
        SoftmaxCausal {
            rows, cols, scale, ..
        } => {
            let cols = *cols as usize;
            pool.install(|| {
                vw.outs[0][..*rows as usize * cols]
                    .par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(r, x)| {
                        let q = r % cols;
                        let mut mx = f32::NEG_INFINITY;
                        for v in &x[..=q] {
                            mx = mx.max(v * scale);
                        }
                        let mut sum = 0.0f32;
                        for v in &mut x[..=q] {
                            *v = (*v * scale - mx).exp();
                            sum += *v;
                        }
                        let inv = 1.0 / sum;
                        x[..=q].iter_mut().for_each(|v| *v *= inv);
                        x[q + 1..].fill(0.0);
                    })
            });
        }
        SoftmaxBwd {
            p: pp,
            rows,
            cols,
            scale,
            ..
        } => {
            let pv = vw.ins[idx(pp)];
            let cols = *cols as usize;
            pool.install(|| {
                vw.outs[0][..*rows as usize * cols]
                    .par_chunks_mut(cols)
                    .zip(pv.par_chunks(cols))
                    .enumerate()
                    .for_each(|(r, (d, p))| {
                        let q = r % cols;
                        let dot: f32 = (0..=q).map(|j| d[j] * p[j]).sum();
                        for j in 0..=q {
                            d[j] = p[j] * (d[j] - dot) * scale;
                        }
                        d[q + 1..].fill(0.0);
                    })
            });
        }
        SiluMul { a, b, n, .. } => {
            let (av, bv) = (vw.ins[idx(a)], vw.ins[idx(b)]);
            let n = *n as usize;
            pool.install(|| {
                vw.outs[0][..n]
                    .par_chunks_mut(CHUNK)
                    .zip(av[..n].par_chunks(CHUNK))
                    .zip(bv[..n].par_chunks(CHUNK))
                    .for_each(|((g, a), b)| {
                        for i in 0..g.len() {
                            g[i] = a[i] * sigmoid(a[i]) * b[i];
                        }
                    })
            });
        }
        SiluMulBwd { dg, a, b, n, .. } => {
            let (gv, av, bv) = (vw.ins[idx(dg)], vw.ins[idx(a)], vw.ins[idx(b)]);
            let n = *n as usize;
            let (da, db) = vw.outs.split_at_mut(1);
            pool.install(|| {
                da[0][..n]
                    .par_chunks_mut(CHUNK)
                    .zip(db[0][..n].par_chunks_mut(CHUNK))
                    .zip(gv[..n].par_chunks(CHUNK))
                    .zip(av[..n].par_chunks(CHUNK))
                    .zip(bv[..n].par_chunks(CHUNK))
                    .for_each(|((((da, db), g), a), b)| {
                        for i in 0..da.len() {
                            let s = sigmoid(a[i]);
                            da[i] = g[i] * b[i] * s * (1.0 + a[i] * (1.0 - s));
                            db[i] = g[i] * a[i] * s;
                        }
                    })
            });
        }
        Xent {
            targets,
            rows,
            vocab,
            grad_scale,
            ..
        } => {
            let tv = vw.ins[idx(targets)];
            let (rows, vocab) = (*rows as usize, *vocab as usize);
            let ids: Vec<usize> = (0..rows)
                .map(|i| token(pc, i, tv[i], vocab))
                .collect::<Result<_, _>>()?;
            let gs = *grad_scale;
            let (loss, correct) = pool.install(|| {
                vw.outs[0][..rows * vocab]
                    .par_chunks_mut(vocab)
                    .zip(ids.par_iter())
                    .map(|(x, &t)| {
                        let (mut mx, mut arg) = (f32::NEG_INFINITY, 0usize);
                        for (j, &v) in x.iter().enumerate() {
                            if v > mx {
                                mx = v;
                                arg = j;
                            }
                        }
                        let target_logit = x[t];
                        let mut sum = 0.0f32;
                        for v in x.iter_mut() {
                            *v = (*v - mx).exp();
                            sum += *v;
                        }
                        let inv = 1.0 / sum;
                        for v in x.iter_mut() {
                            *v *= inv * gs;
                        }
                        x[t] -= gs;
                        ((sum.ln() + mx - target_logit) as f64, (arg == t) as u64)
                    })
                    .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
            });
            regs[crate::regs::LOSS as usize] += loss as f32;
            regs[crate::regs::CORRECT as usize] += correct as f32;
            regs[crate::regs::LOSS_COUNT as usize] += rows as f32;
        }
        SumSq { dst, .. } => {
            let total: f64 = pool.install(|| {
                vw.ins
                    .iter()
                    .map(|b| {
                        b.par_chunks(CHUNK)
                            .map(|c| c.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>())
                            .sum::<f64>()
                    })
                    .sum()
            });
            regs[*dst as usize] = total as f32;
        }
        ClipCoef { dst, src, max_norm } => {
            let norm = regs[*src as usize].max(0.0).sqrt();
            regs[*dst as usize] = (max_norm / (norm + 1e-6)).min(1.0);
        }
        AdamW {
            g,
            lr,
            step,
            clip,
            beta1,
            beta2,
            eps,
            wd,
            ..
        } => {
            let gv = vw.ins[idx(g)];
            let (lr, t, c) = (
                regs[*lr as usize],
                regs[*step as usize],
                regs[*clip as usize],
            );
            let (b1, b2, eps, wd) = (*beta1, *beta2, *eps, *wd);
            // torch.optim.AdamW: p *= 1 - lr*wd; p -= lr/bc1 * m / (sqrt(v)/sqrt(bc2) + eps)
            let bc1 = 1.0 - b1.powf(t);
            let bc2s = (1.0 - b2.powf(t)).sqrt();
            let step_size = lr / bc1;
            let decay = 1.0 - lr * wd;
            let (pp, rest) = vw.outs.split_at_mut(1);
            let (mm, vv) = rest.split_at_mut(1);
            pool.install(|| {
                pp[0]
                    .par_chunks_mut(CHUNK)
                    .zip(mm[0].par_chunks_mut(CHUNK))
                    .zip(vv[0].par_chunks_mut(CHUNK))
                    .zip(gv.par_chunks(CHUNK))
                    .for_each(|(((p, m), v), g)| {
                        for i in 0..p.len() {
                            let gi = g[i] * c;
                            m[i] = b1 * m[i] + (1.0 - b1) * gi;
                            v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                            p[i] = p[i] * decay - step_size * m[i] / (v[i].sqrt() / bc2s + eps);
                        }
                    })
            });
        }
        Quant { format, .. } => quant(vw.outs[0], *format)?,
        SetReg { dst, value } => regs[*dst as usize] = *value,
        Alloc { .. } | Free { .. } => unreachable!("placement handled by run()"),
    }
    Ok(bytes)
}

/// Fake quantisation for low-precision simulation. Codes: 1 = BF16 (round to
/// nearest even); further formats are provided by forge-formats.
fn quant(x: &mut [f32], format: u8) -> Result<(), MachineError> {
    match format {
        1 => {
            x.par_chunks_mut(CHUNK).for_each(|c| {
                for v in c.iter_mut() {
                    let u = v.to_bits();
                    if v.is_nan() {
                        continue;
                    }
                    let r = u.wrapping_add(0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000;
                    *v = f32::from_bits(r);
                }
            });
            Ok(())
        }
        _ => Err(MachineError::Unsupported("QUANT format code")),
    }
}
