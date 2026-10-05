//! Generic blocked GEMM driver shared by every engine.
//!
//! Loop nest (Goto/BLIS), for `C[m×n] = alpha·op(A)·op(B) + beta·C`:
//!
//! ```text
//! for each column slab   jc (≤ NC columns)
//!   for each depth block pc (≤ KC, a multiple of KU)       beta' = beta for the first block, else 1
//!     phase 1: pack B[pc, jc] into NR-wide panels          (shared slab, parallel over panels)
//!     phase 2: parallel over output units (row range × column range):
//!       for each row chunk (≤ MC rows) of the unit
//!         pack A[chunk, pc] into MR-tall panels           (thread-local buffer)
//!         for each NB-wide block of the unit's columns     (L2 block of the slab)
//!           engine macro-kernel: for each MR panel, for each NR panel: micro-kernel
//! ```
//!
//! Every output element is owned by exactly one unit, and its summation order
//! depends only on the shape and the engine (never on the thread count or the
//! partition), so results are bitwise reproducible across pool sizes.

use std::mem::size_of;
use std::ops::Range;

use rayon::prelude::*;

use crate::buffer::{with_buf, SendPtr, PACK_A, PACK_B};
use crate::Trans;

/// Strided view of a stored row-major operand together with its transpose
/// flag. Logical element `(r, c)` of `op(X)` lives at
/// `ptr[r*ld + c]` (N) or `ptr[c*ld + r]` (T).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatRef {
    pub(crate) ptr: *const f32,
    pub(crate) ld: usize,
    pub(crate) trans: Trans,
}

impl MatRef {
    /// Pointer to logical element `(r, c)` of `op(X)`.
    ///
    /// # Safety
    /// The element must lie inside the validated extent of the operand.
    #[inline(always)]
    pub(crate) unsafe fn ptr_at(&self, r: usize, c: usize) -> *const f32 {
        match self.trans {
            Trans::N => self.ptr.add(r * self.ld + c),
            Trans::T => self.ptr.add(c * self.ld + r),
        }
    }

    /// Logical element `(r, c)` of `op(X)`.
    ///
    /// # Safety
    /// As for [`MatRef::ptr_at`].
    #[inline(always)]
    pub(crate) unsafe fn at(&self, r: usize, c: usize) -> f32 {
        *self.ptr_at(r, c)
    }
}

/// A validated GEMM problem over raw pointers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Problem {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) alpha: f32,
    pub(crate) beta: f32,
    pub(crate) a: MatRef,
    pub(crate) b: MatRef,
    pub(crate) c: *mut f32,
    pub(crate) ldc: usize,
}

// SAFETY: a `Problem` is built from borrowed slices that outlive the call and is
// only shared between the tasks of that call, which read A and B and write
// pairwise disjoint regions of C.
unsafe impl Send for Problem {}
// SAFETY: as above.
unsafe impl Sync for Problem {}

impl Problem {
    pub(crate) fn flops(&self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64
    }

    /// True when A and B are referenced (BLAS semantics: they are not when
    /// `k == 0` or `alpha == 0`).
    pub(crate) fn reads_ab(&self) -> bool {
        self.m > 0 && self.n > 0 && self.k > 0 && self.alpha != 0.0
    }

    /// The `i`-th problem of a strided batch.
    ///
    /// # Safety
    /// The batch extents must have been validated.
    pub(crate) unsafe fn item(
        &self,
        i: usize,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
    ) -> Problem {
        let mut p = *self;
        // A and B are only offset when they are referenced; an unreferenced
        // operand may be an empty slice.
        if self.reads_ab() {
            p.a.ptr = self.a.ptr.add(i * stride_a);
            p.b.ptr = self.b.ptr.add(i * stride_b);
        }
        p.c = self.c.add(i * stride_c);
        p
    }
}

/// Number of elements spanned by a stored `rows × cols` row-major matrix with
/// row stride `ld` (0 for an empty matrix). Panics on an invalid stride.
pub(crate) fn extent(rows: usize, cols: usize, ld: usize, what: &str) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }
    assert!(
        rows == 1 || ld >= cols,
        "gemm: {what} row stride {ld} is smaller than its row length {cols}"
    );
    (rows - 1)
        .checked_mul(ld)
        .and_then(|x| x.checked_add(cols))
        .unwrap_or_else(|| panic!("gemm: {what} extent overflows usize"))
}

/// Stored `(rows, cols)` of an operand whose logical shape is `rows × cols`.
pub(crate) fn stored_dims(t: Trans, rows: usize, cols: usize) -> (usize, usize) {
    match t {
        Trans::N => (rows, cols),
        Trans::T => (cols, rows),
    }
}

/// Element extents `(A, B, C)` of one problem; A and B are 0 when unreferenced.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extents(
    ta: Trans,
    tb: Trans,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    lda: usize,
    ldb: usize,
    ldc: usize,
) -> (usize, usize, usize) {
    let ext_c = extent(m, n, ldc, "C");
    if m == 0 || n == 0 || k == 0 || alpha == 0.0 {
        return (0, 0, ext_c);
    }
    let (ar, ac) = stored_dims(ta, m, k);
    let (br, bc) = stored_dims(tb, k, n);
    (extent(ar, ac, lda, "A"), extent(br, bc, ldb, "B"), ext_c)
}

/// `C = beta·C` over the logical `m × n` region (the result of a GEMM whose
/// product term vanishes: `k == 0` or `alpha == 0`). `beta == 0` stores zeros
/// without reading C.
///
/// # Safety
/// `p.c` must be valid for the validated C extent.
pub(crate) unsafe fn scale_c(p: &Problem) {
    if p.beta == 1.0 {
        return;
    }
    for i in 0..p.m {
        let row = std::slice::from_raw_parts_mut(p.c.add(i * p.ldc), p.n);
        if p.beta == 0.0 {
            row.fill(0.0);
        } else {
            row.iter_mut().for_each(|x| *x *= p.beta);
        }
    }
}

/// A packed-panel GEMM engine.
///
/// Packed A is a sequence of panels of `MR` rows; packed B a sequence of panels
/// of `NR` columns; both cover a depth block of `kc` padded to `kc_pad` (a
/// multiple of `KU`). Panel strides are `a_panel_len(kc_pad)` and
/// `b_panel_len(kc_pad)` elements; the internal layout of a panel is private to
/// the engine. Padding rows, columns and depth entries are zero.
pub(crate) trait Engine {
    /// Packed element type (`f32`, or bf16 bits as `u16`).
    type P: Copy + Send + Sync + 'static;
    /// Rows per A panel / micro-tile.
    const MR: usize;
    /// Columns per B panel / micro-tile.
    const NR: usize;
    /// Depth granularity: `kc_pad` is a multiple of this.
    const KU: usize;
    /// Maximum depth block (a multiple of `KU`).
    const KC: usize;
    /// Maximum rows per packed A chunk (L2 block).
    const MC: usize;
    /// Columns per L2 block of the packed B slab (a multiple of `NR`).
    const NB: usize;
    /// Maximum slab width (a multiple of `NR`).
    const NC: usize;
    /// Minimum useful work per parallel task, in flops.
    const TASK_FLOPS: f64;

    fn a_panel_len(kc_pad: usize) -> usize {
        (kc_pad * Self::MR).next_multiple_of(16)
    }
    fn b_panel_len(kc_pad: usize) -> usize {
        (kc_pad * Self::NR).next_multiple_of(16)
    }

    /// Packs rows `i0..i0+mr` (`mr <= MR`) and depth `p0..p0+kc` of `op(A)`
    /// into one panel at `dst`.
    ///
    /// # Safety
    /// The source region must be inside the validated extent of A and `dst`
    /// must hold `a_panel_len(kc_pad)` elements.
    unsafe fn pack_a(
        a: &MatRef,
        i0: usize,
        mr: usize,
        p0: usize,
        kc: usize,
        kc_pad: usize,
        dst: *mut Self::P,
    );

    /// Packs depth `p0..p0+kc` and columns `j0..j0+nr` (`nr <= NR`) of `op(B)`
    /// into one panel at `dst`.
    ///
    /// # Safety
    /// The source region must be inside the validated extent of B and `dst`
    /// must hold `b_panel_len(kc_pad)` elements.
    unsafe fn pack_b(
        b: &MatRef,
        p0: usize,
        kc: usize,
        kc_pad: usize,
        j0: usize,
        nr: usize,
        dst: *mut Self::P,
    );

    /// `C[mc×nc] = alpha·Ap·Bp + beta·C` for packed panels covering `mc` rows
    /// and `nc` columns (`beta == 0` must not read C).
    ///
    /// # Safety
    /// `ap`/`bp` must hold the packed panels; `c` must be valid for an
    /// `mc × nc` block with row stride `ldc` that no other thread touches.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kernel_block(
        mc: usize,
        nc: usize,
        kc_pad: usize,
        ap: *const Self::P,
        bp: *const Self::P,
        c: *mut f32,
        ldc: usize,
        alpha: f32,
        beta: f32,
    );

    /// Per-task setup (e.g. AMX tile configuration), run on the executing
    /// thread before any `kernel_block` of the task.
    ///
    /// # Safety
    /// Must be paired with [`Engine::end`] on the same thread.
    unsafe fn begin() {}

    /// Per-task teardown.
    ///
    /// # Safety
    /// See [`Engine::begin`].
    unsafe fn end() {}
}

/// Splits `0..total` into blocks of at most `max` elements, all of equal size
/// `step` (a multiple of `unit`) except possibly a shorter last one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Blocks {
    total: usize,
    step: usize,
}

impl Blocks {
    pub(crate) fn new(total: usize, max: usize, unit: usize) -> Blocks {
        debug_assert!(max >= unit && max.is_multiple_of(unit));
        let count = total.div_ceil(max).max(1);
        let step = total.div_ceil(count).next_multiple_of(unit).max(unit);
        Blocks { total, step }
    }

    pub(crate) fn step(&self) -> usize {
        self.step
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = Range<usize>> {
        let (total, step) = (self.total, self.step);
        (0..total)
            .step_by(step)
            .map(move |s| s..(s + step).min(total))
    }
}

/// Splits `0..len` into `parts` contiguous ranges whose boundaries are
/// multiples of `align` (except the end) and whose sizes differ by at most
/// `align`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Split {
    len: usize,
    parts: usize,
    align: usize,
}

impl Split {
    pub(crate) fn new(len: usize, parts: usize, align: usize) -> Split {
        let units = len.div_ceil(align).max(1);
        Split {
            len,
            parts: parts.clamp(1, units),
            align,
        }
    }

    pub(crate) fn parts(&self) -> usize {
        self.parts
    }

    pub(crate) fn range(&self, i: usize) -> Range<usize> {
        let units = self.len.div_ceil(self.align);
        let lo = i * units / self.parts;
        let hi = (i + 1) * units / self.parts;
        (lo * self.align).min(self.len)..(hi * self.align).min(self.len)
    }
}

/// Packing work below which the B slab is packed by the driving thread alone.
const PAR_PACK_MIN_ELEMS: usize = 1 << 15;

/// Units per thread when the work allows it (dynamic balance on a noisy machine).
const OVERSUB: usize = 2;

/// Computes `p` with engine `E` using up to `threads` workers of the current
/// rayon pool. Must be called inside the pool (`Pool::install`) and only for
/// problems that reference A and B.
pub(crate) fn run<E: Engine>(p: &Problem, threads: usize) {
    debug_assert!(p.reads_ab());
    let tasks = (p.flops() / E::TASK_FLOPS) as usize;
    let threads = if tasks < 2 {
        1
    } else {
        threads.clamp(1, tasks)
    };
    drive::<E>(p, threads);
}

fn drive<E: Engine>(p: &Problem, threads: usize) {
    let kb = Blocks::new(p.k, E::KC, E::KU);
    let nb = Blocks::new(p.n, E::NC, E::NR);
    let slab_elems = E::b_panel_len(kb.step()) * nb.step().div_ceil(E::NR);
    with_buf(&PACK_B, slab_elems * size_of::<E::P>(), |raw| {
        let slab = SendPtr(raw.cast::<E::P>());
        for cols in nb.iter() {
            for (ki, ks) in kb.iter().enumerate() {
                let kc = ks.len();
                let kc_pad = kc.next_multiple_of(E::KU);
                let beta = if ki == 0 { p.beta } else { 1.0 };
                pack_slab::<E>(p, ks.start, kc, kc_pad, cols.clone(), slab, threads);
                compute::<E>(p, ks.start, kc, kc_pad, cols.clone(), slab, beta, threads);
            }
        }
    });
}

/// Phase 1: packs `op(B)[p0..p0+kc, cols]` into consecutive NR panels.
fn pack_slab<E: Engine>(
    p: &Problem,
    p0: usize,
    kc: usize,
    kc_pad: usize,
    cols: Range<usize>,
    slab: SendPtr<E::P>,
    threads: usize,
) {
    let panels = cols.len().div_ceil(E::NR);
    let stride = E::b_panel_len(kc_pad);
    let pack = |jp: usize| {
        let j0 = cols.start + jp * E::NR;
        let nr = E::NR.min(cols.end - j0);
        // SAFETY: the panel lies inside the validated B extent and the slab
        // holds `panels * stride` elements; panels are written disjointly.
        unsafe { E::pack_b(&p.b, p0, kc, kc_pad, j0, nr, slab.get().add(jp * stride)) }
    };
    if threads <= 1 || panels < 2 || panels * stride < PAR_PACK_MIN_ELEMS {
        (0..panels).for_each(pack);
    } else {
        let per_task = panels.div_ceil(threads * 2).max(1);
        (0..panels)
            .into_par_iter()
            .with_min_len(per_task)
            .for_each(pack);
    }
}

/// Chooses the unit partition of one `m × nc × kc` block.
fn plan<E: Engine>(m: usize, nc: usize, kc: usize, threads: usize) -> (Split, Split) {
    if threads <= 1 {
        return (Split::new(m, 1, E::MR), Split::new(nc, 1, E::NR));
    }
    let block_flops = 2.0 * m as f64 * nc as f64 * kc as f64;
    let by_work = ((block_flops / E::TASK_FLOPS) as usize).max(1);
    let target = if by_work >= threads {
        threads * (by_work / threads).min(OVERSUB)
    } else {
        by_work
    };
    let pm = target.min(m.div_ceil(E::MR));
    let pn = target.div_ceil(pm).min(nc.div_ceil(E::NR));
    (Split::new(m, pm, E::MR), Split::new(nc, pn, E::NR))
}

/// Phase 2: all output units of one depth block.
#[allow(clippy::too_many_arguments)]
fn compute<E: Engine>(
    p: &Problem,
    p0: usize,
    kc: usize,
    kc_pad: usize,
    cols: Range<usize>,
    slab: SendPtr<E::P>,
    beta: f32,
    threads: usize,
) {
    let (rs, cs) = plan::<E>(p.m, cols.len(), kc, threads);
    let stride = E::b_panel_len(kc_pad);
    let run_unit = |u: usize| {
        let rows = rs.range(u % rs.parts());
        let c = cs.range(u / rs.parts());
        let unit_cols = cols.start + c.start..cols.start + c.end;
        // SAFETY: units cover pairwise disjoint regions of C; the slab holds the
        // panels of `cols`, and `c.start` is a multiple of NR.
        unsafe {
            let bp = slab.get().add(c.start / E::NR * stride).cast_const();
            unit::<E>(p, rows, unit_cols, bp, p0, kc, kc_pad, beta);
        }
    };
    let units = rs.parts() * cs.parts();
    if units == 1 {
        run_unit(0);
    } else {
        (0..units).into_par_iter().for_each(run_unit);
    }
}

/// One output unit: rows × columns of C for one depth block, run sequentially on
/// the calling thread.
///
/// # Safety
/// `bp` must point at the packed panel of column `cols.start`; the unit's C
/// region must not be accessed concurrently.
#[allow(clippy::too_many_arguments)]
unsafe fn unit<E: Engine>(
    p: &Problem,
    rows: Range<usize>,
    cols: Range<usize>,
    bp: *const E::P,
    p0: usize,
    kc: usize,
    kc_pad: usize,
    beta: f32,
) {
    if rows.is_empty() || cols.is_empty() {
        return;
    }
    let chunks = rows.len().div_ceil(E::MC);
    let mc = rows.len().div_ceil(chunks).next_multiple_of(E::MR);
    let a_stride = E::a_panel_len(kc_pad);
    let b_stride = E::b_panel_len(kc_pad);
    let a_elems = a_stride * mc.div_ceil(E::MR);
    with_buf(&PACK_A, a_elems * size_of::<E::P>(), |raw| {
        let ap = raw.cast::<E::P>();
        E::begin();
        let mut r = rows.start;
        while r < rows.end {
            let mrows = mc.min(rows.end - r);
            for ip in 0..mrows.div_ceil(E::MR) {
                let i0 = r + ip * E::MR;
                let mr = E::MR.min(r + mrows - i0);
                E::pack_a(&p.a, i0, mr, p0, kc, kc_pad, ap.add(ip * a_stride));
            }
            let mut j = cols.start;
            while j < cols.end {
                let ncols = E::NB.min(cols.end - j);
                let bpj = bp.add((j - cols.start) / E::NR * b_stride);
                let cp = p.c.add(r * p.ldc + j);
                E::kernel_block(mrows, ncols, kc_pad, ap, bpj, cp, p.ldc, p.alpha, beta);
                j += ncols;
            }
            r += mrows;
        }
        E::end();
    });
}

/// Strided batch with engine `E` inside the current rayon pool.
///
/// Small items run in parallel across the batch (each item sequential on one
/// worker); large items run one after the other, each parallel inside. When the
/// C items may overlap, items run in order to keep sequential semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_batched<E: Engine>(
    p: &Problem,
    batch: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
    c_disjoint: bool,
    threads: usize,
) {
    debug_assert!(p.reads_ab());
    // SAFETY (all `item` calls): the batch extents were validated by the caller.
    let item = |i: usize| unsafe { p.item(i, stride_a, stride_b, stride_c) };
    let item_tasks = (p.flops() / E::TASK_FLOPS) as usize;
    let across = threads > 1
        && batch > 1
        && c_disjoint
        && (item_tasks < 2 * threads || (batch >= threads && batch.is_multiple_of(threads)));
    if across {
        (0..batch)
            .into_par_iter()
            .for_each(|i| drive::<E>(&item(i), 1));
    } else {
        for i in 0..batch {
            run::<E>(&item(i), threads);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_cover_exactly_with_aligned_steps() {
        for total in [1usize, 2, 31, 32, 33, 100, 511, 512, 513, 1000, 4096, 5000] {
            for (max, unit) in [
                (32usize, 32usize),
                (64, 32),
                (256, 1),
                (384, 1),
                (512, 32),
                (16, 8),
            ] {
                let b = Blocks::new(total, max, unit);
                assert!(
                    b.step() <= max && b.step().is_multiple_of(unit),
                    "{total} {max} {unit}"
                );
                let ranges: Vec<_> = b.iter().collect();
                assert_eq!(ranges.first().unwrap().start, 0);
                assert_eq!(ranges.last().unwrap().end, total);
                for w in ranges.windows(2) {
                    assert_eq!(w[0].end, w[1].start);
                }
                assert!(ranges.iter().all(|r| !r.is_empty()));
                assert_eq!(ranges.len(), total.div_ceil(max));
            }
        }
    }

    #[test]
    fn split_is_aligned_balanced_and_complete() {
        for len in [1usize, 5, 14, 15, 64, 100, 1000, 4096] {
            for parts in 1..=12 {
                for align in [1usize, 4, 6, 14, 32] {
                    let s = Split::new(len, parts, align);
                    let mut next = 0;
                    for i in 0..s.parts() {
                        let r = s.range(i);
                        assert_eq!(r.start, next);
                        assert!(!r.is_empty(), "{len} {parts} {align}");
                        assert_eq!(r.start % align, 0);
                        next = r.end;
                    }
                    assert_eq!(next, len);
                }
            }
        }
    }

    #[test]
    fn extent_rules() {
        assert_eq!(extent(0, 5, 0, "X"), 0);
        assert_eq!(extent(5, 0, 0, "X"), 0);
        assert_eq!(extent(1, 5, 0, "X"), 5);
        assert_eq!(extent(3, 5, 7, "X"), 19);
        assert!(std::panic::catch_unwind(|| extent(2, 5, 4, "X")).is_err());
        assert!(std::panic::catch_unwind(|| extent(usize::MAX, 2, usize::MAX, "X")).is_err());
    }
}
