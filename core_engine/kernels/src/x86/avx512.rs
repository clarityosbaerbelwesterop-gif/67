//! AVX-512F engine: 14×32 f32 micro-kernel (28 zmm accumulators, two B
//! vectors, broadcasts from memory), masked epilogue for ragged edges and
//! 16×16-transpose packing.

use std::arch::x86_64::*;

use super::{mask16, transpose16x16};
use crate::driver::{Engine, MatRef};
use crate::Trans;

pub(crate) fn supported() -> bool {
    is_x86_feature_detected!("avx512f")
}

const MR: usize = 14;
const NR: usize = 32;
const MR_MASK: u16 = (1 << MR) - 1;

pub(crate) struct Avx512;

impl Engine for Avx512 {
    type P = f32;
    const MR: usize = MR;
    const NR: usize = NR;
    const KU: usize = 1;
    const KC: usize = 384;
    const MC: usize = 168;
    const NB: usize = 512;
    const NC: usize = 4096;
    const TASK_FLOPS: f64 = 4.0e6;

    unsafe fn pack_a(
        a: &MatRef,
        i0: usize,
        mr: usize,
        p0: usize,
        kc: usize,
        _kc_pad: usize,
        dst: *mut f32,
    ) {
        pack_a(a, i0, mr, p0, kc, dst);
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
        pack_b(b, p0, kc, j0, nr, dst);
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
        kernel_block(mc, nc, kc, ap, bp, c, ldc, alpha, beta);
    }
}

/// Packs a 14-row panel: element `(i, p)` to `dst[p*14 + i]`, rows `mr..14`
/// zero.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn pack_a(a: &MatRef, i0: usize, mr: usize, p0: usize, kc: usize, dst: *mut f32) {
    match a.trans {
        Trans::T => {
            let m = mask16(mr);
            for p in 0..kc {
                let v = _mm512_maskz_loadu_ps(m, a.ptr_at(i0, p0 + p));
                _mm512_mask_storeu_ps(dst.add(p * MR), MR_MASK, v);
            }
        }
        Trans::N => {
            let mut p = 0;
            while p < kc {
                let w = 16.min(kc - p);
                let m = mask16(w);
                let mut r = [_mm512_setzero_si512(); 16];
                for (i, row) in r.iter_mut().enumerate().take(mr) {
                    *row = _mm512_castps_si512(_mm512_maskz_loadu_ps(m, a.ptr_at(i0 + i, p0 + p)));
                }
                transpose16x16(&mut r);
                for (q, col) in r.iter().enumerate().take(w) {
                    _mm512_mask_storeu_ps(
                        dst.add((p + q) * MR),
                        MR_MASK,
                        _mm512_castsi512_ps(*col),
                    );
                }
                p += w;
            }
        }
    }
}

/// Packs a 32-column panel: element `(p, j)` to `dst[p*32 + j]`, columns
/// `nr..32` zero.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn pack_b(b: &MatRef, p0: usize, kc: usize, j0: usize, nr: usize, dst: *mut f32) {
    match b.trans {
        Trans::N => {
            let (m0, m1) = (mask16(nr.min(16)), mask16(nr.saturating_sub(16)));
            for p in 0..kc {
                let src = b.ptr_at(p0 + p, j0);
                let d = dst.add(p * NR);
                _mm512_storeu_ps(d, _mm512_maskz_loadu_ps(m0, src));
                _mm512_storeu_ps(d.add(16), _mm512_maskz_loadu_ps(m1, src.wrapping_add(16)));
            }
        }
        Trans::T => {
            let mut p = 0;
            while p < kc {
                let w = 16.min(kc - p);
                let m = mask16(w);
                for h in 0..2 {
                    let cols = nr.saturating_sub(16 * h).min(16);
                    let mut r = [_mm512_setzero_si512(); 16];
                    for (jj, row) in r.iter_mut().enumerate().take(cols) {
                        *row = _mm512_castps_si512(_mm512_maskz_loadu_ps(
                            m,
                            b.ptr_at(p0 + p, j0 + 16 * h + jj),
                        ));
                    }
                    transpose16x16(&mut r);
                    for (q, col) in r.iter().enumerate().take(w) {
                        _mm512_storeu_ps(dst.add((p + q) * NR + 16 * h), _mm512_castsi512_ps(*col));
                    }
                }
                p += w;
            }
        }
    }
}

/// `M × 32` micro-kernel (`M <= 14` rows of a 14-row panel), writing the
/// `M × nr` corner of the tile with `C = alpha·acc + beta·C` through column
/// masks (`beta == 0` never reads C).
#[inline]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro<const M: usize>(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
    nr: usize,
) {
    let mut c0 = [_mm512_setzero_ps(); M];
    let mut c1 = [_mm512_setzero_ps(); M];
    let (mut pa, mut pb) = (a, b);
    for _ in 0..kc {
        let b0 = _mm512_loadu_ps(pb);
        let b1 = _mm512_loadu_ps(pb.add(16));
        for i in 0..M {
            let ai = _mm512_set1_ps(*pa.add(i));
            c0[i] = _mm512_fmadd_ps(ai, b0, c0[i]);
            c1[i] = _mm512_fmadd_ps(ai, b1, c1[i]);
        }
        pa = pa.add(MR);
        pb = pb.add(NR);
    }
    let (m0, m1) = (mask16(nr.min(16)), mask16(nr.saturating_sub(16)));
    let va = _mm512_set1_ps(alpha);
    let vb = _mm512_set1_ps(beta);
    for i in 0..M {
        let cp = c.add(i * ldc);
        let cp1 = cp.wrapping_add(16);
        let (mut r0, mut r1) = (c0[i], c1[i]);
        if alpha != 1.0 {
            r0 = _mm512_mul_ps(r0, va);
            r1 = _mm512_mul_ps(r1, va);
        }
        if beta == 1.0 {
            r0 = _mm512_add_ps(_mm512_maskz_loadu_ps(m0, cp), r0);
            r1 = _mm512_add_ps(_mm512_maskz_loadu_ps(m1, cp1), r1);
        } else if beta != 0.0 {
            r0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m0, cp), vb, r0);
            r1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m1, cp1), vb, r1);
        }
        _mm512_mask_storeu_ps(cp, m0, r0);
        _mm512_mask_storeu_ps(cp1, m1, r1);
    }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
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
    let (sa, sb) = (Avx512::a_panel_len(kc), Avx512::b_panel_len(kc));
    for ir in (0..mc).step_by(MR) {
        let a = ap.add(ir / MR * sa);
        for jr in (0..nc).step_by(NR) {
            let b = bp.add(jr / NR * sb);
            let nr = NR.min(nc - jr);
            let cp = c.add(ir * ldc + jr);
            match mc - ir {
                1 => micro::<1>(kc, a, b, cp, ldc, alpha, beta, nr),
                2 => micro::<2>(kc, a, b, cp, ldc, alpha, beta, nr),
                3 => micro::<3>(kc, a, b, cp, ldc, alpha, beta, nr),
                4 => micro::<4>(kc, a, b, cp, ldc, alpha, beta, nr),
                5 => micro::<5>(kc, a, b, cp, ldc, alpha, beta, nr),
                6 => micro::<6>(kc, a, b, cp, ldc, alpha, beta, nr),
                7 => micro::<7>(kc, a, b, cp, ldc, alpha, beta, nr),
                8 => micro::<8>(kc, a, b, cp, ldc, alpha, beta, nr),
                9 => micro::<9>(kc, a, b, cp, ldc, alpha, beta, nr),
                10 => micro::<10>(kc, a, b, cp, ldc, alpha, beta, nr),
                11 => micro::<11>(kc, a, b, cp, ldc, alpha, beta, nr),
                12 => micro::<12>(kc, a, b, cp, ldc, alpha, beta, nr),
                13 => micro::<13>(kc, a, b, cp, ldc, alpha, beta, nr),
                _ => micro::<MR>(kc, a, b, cp, ldc, alpha, beta, nr),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{pack_a_ref, pack_b_ref};
    use crate::tests::Rng;

    #[test]
    fn packers_match_reference() {
        if !supported() {
            return;
        }
        let mut rng = Rng::new(12);
        for t in [Trans::N, Trans::T] {
            for &(rows, cols) in &[(1usize, 1usize), (14, 16), (15, 17), (29, 40), (70, 33)] {
                let ld = cols + rng.below(3);
                let data: Vec<f32> = (0..(rows - 1) * ld + cols).map(|_| rng.sym()).collect();
                let (lr, lc) = match t {
                    Trans::N => (rows, cols),
                    Trans::T => (cols, rows),
                };
                let x = MatRef {
                    ptr: data.as_ptr(),
                    ld,
                    trans: t,
                };
                for _ in 0..20 {
                    let i0 = rng.below(lr);
                    let mr = 1 + rng.below(MR.min(lr - i0));
                    let p0 = rng.below(lc);
                    let kc = 1 + rng.below(lc - p0);
                    let mut want = vec![f32::NAN; kc * MR];
                    let mut got = vec![f32::NAN; kc * MR];
                    // SAFETY: regions are in bounds; AVX-512F is available.
                    unsafe {
                        pack_a_ref::<MR>(&x, i0, mr, p0, kc, want.as_mut_ptr());
                        pack_a(&x, i0, mr, p0, kc, got.as_mut_ptr());
                    }
                    assert_eq!(want, got, "A {t:?} i0={i0} mr={mr} p0={p0} kc={kc}");
                    let p0 = rng.below(lr);
                    let kc = 1 + rng.below(lr - p0);
                    let j0 = rng.below(lc);
                    let nr = 1 + rng.below(NR.min(lc - j0));
                    let mut want = vec![f32::NAN; kc * NR];
                    let mut got = vec![f32::NAN; kc * NR];
                    // SAFETY: as above.
                    unsafe {
                        pack_b_ref::<NR>(&x, p0, kc, j0, nr, want.as_mut_ptr());
                        pack_b(&x, p0, kc, j0, nr, got.as_mut_ptr());
                    }
                    assert_eq!(want, got, "B {t:?} p0={p0} kc={kc} j0={j0} nr={nr}");
                }
            }
        }
    }
}
