//! Portable engine: plain Rust micro-kernel over packed panels (the compiler
//! may auto-vectorise it for the baseline target, e.g. SSE2 or NEON), plus the
//! generic f32 packers shared with the SIMD engines for reference and edges.

use crate::driver::{Engine, MatRef};

/// Packs an `MR`-row panel of `op(A)`: element `(i, p)` goes to
/// `dst[p*MR + i]`; rows `mr..MR` are zero.
///
/// # Safety
/// See [`Engine::pack_a`].
pub(crate) unsafe fn pack_a_ref<const MR: usize>(
    a: &MatRef,
    i0: usize,
    mr: usize,
    p0: usize,
    kc: usize,
    dst: *mut f32,
) {
    debug_assert!(mr <= MR);
    for p in 0..kc {
        let d = dst.add(p * MR);
        for i in 0..mr {
            *d.add(i) = a.at(i0 + i, p0 + p);
        }
        for i in mr..MR {
            *d.add(i) = 0.0;
        }
    }
}

/// Packs an `NR`-column panel of `op(B)`: element `(p, j)` goes to
/// `dst[p*NR + j]`; columns `nr..NR` are zero.
///
/// # Safety
/// See [`Engine::pack_b`].
pub(crate) unsafe fn pack_b_ref<const NR: usize>(
    b: &MatRef,
    p0: usize,
    kc: usize,
    j0: usize,
    nr: usize,
    dst: *mut f32,
) {
    debug_assert!(nr <= NR);
    for p in 0..kc {
        let d = dst.add(p * NR);
        for j in 0..nr {
            *d.add(j) = b.at(p0 + p, j0 + j);
        }
        for j in nr..NR {
            *d.add(j) = 0.0;
        }
    }
}

/// Applies `C = alpha·acc + beta·C` to the `mr × nr` corner of an `MR × NR`
/// accumulator tile stored row-major with row stride `NR`. `beta == 0` does not
/// read C.
///
/// # Safety
/// `c` must be valid for an `mr × nr` block with row stride `ldc`.
#[inline]
pub(crate) unsafe fn store_tile<const NR: usize>(
    acc: &[f32],
    mr: usize,
    nr: usize,
    c: *mut f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) {
    for i in 0..mr {
        let src = &acc[i * NR..i * NR + nr];
        let dst = std::slice::from_raw_parts_mut(c.add(i * ldc), nr);
        if beta == 0.0 {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = alpha * s;
            }
        } else {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = alpha * s + beta * *d;
            }
        }
    }
}

pub(crate) struct Scalar;

const MR: usize = 4;
const NR: usize = 8;

/// `MR × NR` micro-kernel: plain multiply-add (no fused FMA on the baseline
/// target), accumulators kept in a fixed array the optimiser maps to registers.
///
/// # Safety
/// `a`/`b` must hold `kc` packed steps; `c` as for [`store_tile`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn micro(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
    mr: usize,
    nr: usize,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for p in 0..kc {
        let av = *a.add(p * MR).cast::<[f32; MR]>();
        let bv = *b.add(p * NR).cast::<[f32; NR]>();
        for i in 0..MR {
            for j in 0..NR {
                acc[i][j] += av[i] * bv[j];
            }
        }
    }
    store_tile::<NR>(acc.as_flattened(), mr, nr, c, ldc, alpha, beta);
}

impl Engine for Scalar {
    type P = f32;
    const MR: usize = MR;
    const NR: usize = NR;
    const KU: usize = 1;
    const KC: usize = 256;
    const MC: usize = 64;
    const NB: usize = 256;
    const NC: usize = 4096;
    const TASK_FLOPS: f64 = 4.0e5;

    unsafe fn pack_a(
        a: &MatRef,
        i0: usize,
        mr: usize,
        p0: usize,
        kc: usize,
        _kc_pad: usize,
        dst: *mut f32,
    ) {
        pack_a_ref::<MR>(a, i0, mr, p0, kc, dst);
    }

    unsafe fn pack_b(
        b: &MatRef,
        p0: usize,
        kc: usize,
        _kc_pad: usize,
        j0: usize,
        nr: usize,
        dst: *mut f32,
    ) {
        pack_b_ref::<NR>(b, p0, kc, j0, nr, dst);
    }

    unsafe fn kernel_block(
        mc: usize,
        nc: usize,
        kc: usize,
        ap: *const f32,
        bp: *const f32,
        c: *mut f32,
        ldc: usize,
        alpha: f32,
        beta: f32,
    ) {
        let (sa, sb) = (Self::a_panel_len(kc), Self::b_panel_len(kc));
        for ir in (0..mc).step_by(MR) {
            let a = ap.add(ir / MR * sa);
            for jr in (0..nc).step_by(NR) {
                let b = bp.add(jr / NR * sb);
                let (mr, nr) = (MR.min(mc - ir), NR.min(nc - jr));
                micro(kc, a, b, c.add(ir * ldc + jr), ldc, alpha, beta, mr, nr);
            }
        }
    }
}
