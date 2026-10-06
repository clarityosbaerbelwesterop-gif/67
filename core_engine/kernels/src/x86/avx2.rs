//! AVX2 + FMA engine: 6×16 f32 micro-kernel (12 accumulators, two B vectors,
//! one broadcast), vectorised packing for both transpose cases.

use std::arch::x86_64::*;

use super::{mask8, transpose8x8};
use crate::driver::{Engine, MatRef};
use crate::scalar::store_tile;
use crate::Trans;

pub(crate) fn supported() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

const MR: usize = 6;
const NR: usize = 16;

pub(crate) struct Avx2;

impl Engine for Avx2 {
    type P = f32;
    const MR: usize = MR;
    const NR: usize = NR;
    const KU: usize = 1;
    const KC: usize = 256;
    const MC: usize = 96;
    const NB: usize = 512;
    const NC: usize = 4096;
    const TASK_FLOPS: f64 = 2.0e6;

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

/// Loads the first `w` (`<= 8`) floats at `src` (zero elsewhere) without
/// touching memory past them.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load8(src: *const f32, w: usize) -> __m256 {
    if w == 8 {
        _mm256_loadu_ps(src)
    } else {
        _mm256_maskload_ps(src, mask8(w))
    }
}

/// Stores lanes 0..6 of `v` at `dst`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store6(dst: *mut f32, v: __m256) {
    _mm_storeu_ps(dst, _mm256_castps256_ps128(v));
    _mm_storel_epi64(
        dst.add(4).cast(),
        _mm_castps_si128(_mm256_extractf128_ps::<1>(v)),
    );
}

/// Packs a 6-row panel: element `(i, p)` to `dst[p*6 + i]`, rows `mr..6` zero.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn pack_a(a: &MatRef, i0: usize, mr: usize, p0: usize, kc: usize, dst: *mut f32) {
    match a.trans {
        Trans::T => {
            // Stored k×m: the panel's rows are contiguous for each depth index.
            let m = mask8(mr);
            for p in 0..kc {
                let v = _mm256_maskload_ps(a.ptr_at(i0, p0 + p), m);
                store6(dst.add(p * MR), v);
            }
        }
        Trans::N => {
            // Stored m×k: transpose 6(+2 zero)×8 blocks.
            let mut p = 0;
            while p < kc {
                let w = 8.min(kc - p);
                let mut r = [_mm256_setzero_ps(); 8];
                for (i, row) in r.iter_mut().enumerate().take(mr) {
                    *row = load8(a.ptr_at(i0 + i, p0 + p), w);
                }
                transpose8x8(&mut r);
                for (q, col) in r.iter().enumerate().take(w) {
                    store6(dst.add((p + q) * MR), *col);
                }
                p += w;
            }
        }
    }
}

/// Packs a 16-column panel: element `(p, j)` to `dst[p*16 + j]`, columns
/// `nr..16` zero.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn pack_b(b: &MatRef, p0: usize, kc: usize, j0: usize, nr: usize, dst: *mut f32) {
    match b.trans {
        Trans::N => {
            // Stored k×n: one contiguous row segment per depth index.
            let (w0, w1) = (nr.min(8), nr.saturating_sub(8));
            for p in 0..kc {
                let src = b.ptr_at(p0 + p, j0);
                let d = dst.add(p * NR);
                _mm256_storeu_ps(d, load8(src, w0));
                let hi = if w1 == 0 {
                    _mm256_setzero_ps()
                } else {
                    load8(src.add(8), w1)
                };
                _mm256_storeu_ps(d.add(8), hi);
            }
        }
        Trans::T => {
            // Stored n×k: transpose 8×8 blocks of each half panel.
            let mut p = 0;
            while p < kc {
                let w = 8.min(kc - p);
                for h in 0..2 {
                    let cols = nr.saturating_sub(8 * h).min(8);
                    let mut r = [_mm256_setzero_ps(); 8];
                    for (jj, row) in r.iter_mut().enumerate().take(cols) {
                        *row = load8(b.ptr_at(p0 + p, j0 + 8 * h + jj), w);
                    }
                    transpose8x8(&mut r);
                    for (q, col) in r.iter().enumerate().take(w) {
                        _mm256_storeu_ps(dst.add((p + q) * NR + 8 * h), *col);
                    }
                }
                p += w;
            }
        }
    }
}

/// `M × 16` micro-kernel (`M <= 6` rows of a 6-row panel), writing the
/// `M × nr` corner of the tile with `C = alpha·acc + beta·C`.
#[inline]
#[target_feature(enable = "avx2,fma")]
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
    // Pull the C tile towards L1 while the FMA loop runs.
    for i in 0..M {
        let row = c.wrapping_add(i * ldc).cast::<i8>().cast_const();
        _mm_prefetch::<_MM_HINT_T0>(row);
        _mm_prefetch::<_MM_HINT_T0>(row.wrapping_add(nr * 4 - 1));
    }
    let mut c0 = [_mm256_setzero_ps(); M];
    let mut c1 = [_mm256_setzero_ps(); M];
    let (mut pa, mut pb) = (a, b);
    for _ in 0..kc {
        let b0 = _mm256_loadu_ps(pb);
        let b1 = _mm256_loadu_ps(pb.add(8));
        for i in 0..M {
            let ai = _mm256_set1_ps(*pa.add(i));
            c0[i] = _mm256_fmadd_ps(ai, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(ai, b1, c1[i]);
        }
        pa = pa.add(MR);
        pb = pb.add(NR);
    }
    if nr == NR {
        let va = _mm256_set1_ps(alpha);
        let vb = _mm256_set1_ps(beta);
        for i in 0..M {
            let cp = c.add(i * ldc);
            let (mut r0, mut r1) = (c0[i], c1[i]);
            if alpha != 1.0 {
                r0 = _mm256_mul_ps(r0, va);
                r1 = _mm256_mul_ps(r1, va);
            }
            if beta == 1.0 {
                r0 = _mm256_add_ps(_mm256_loadu_ps(cp), r0);
                r1 = _mm256_add_ps(_mm256_loadu_ps(cp.add(8)), r1);
            } else if beta != 0.0 {
                r0 = _mm256_fmadd_ps(_mm256_loadu_ps(cp), vb, r0);
                r1 = _mm256_fmadd_ps(_mm256_loadu_ps(cp.add(8)), vb, r1);
            }
            _mm256_storeu_ps(cp, r0);
            _mm256_storeu_ps(cp.add(8), r1);
        }
    } else {
        let mut tile = [0.0f32; MR * NR];
        for i in 0..M {
            _mm256_storeu_ps(tile.as_mut_ptr().add(i * NR), c0[i]);
            _mm256_storeu_ps(tile.as_mut_ptr().add(i * NR + 8), c1[i]);
        }
        store_tile::<NR>(&tile, M, nr, c, ldc, alpha, beta);
    }
}

#[target_feature(enable = "avx2,fma")]
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
    let (sa, sb) = (Avx2::a_panel_len(kc), Avx2::b_panel_len(kc));
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

    /// Every packer path against the scalar reference packers.
    #[test]
    fn packers_match_reference() {
        if !supported() {
            return;
        }
        let mut rng = Rng::new(11);
        for t in [Trans::N, Trans::T] {
            for &(rows, cols) in &[(1usize, 1usize), (6, 8), (7, 9), (13, 33), (40, 17)] {
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
                    // A panel: logical rows i0..i0+mr, depth p0..p0+kc.
                    let i0 = rng.below(lr);
                    let mr = 1 + rng.below(MR.min(lr - i0));
                    let p0 = rng.below(lc);
                    let kc = 1 + rng.below(lc - p0);
                    let mut want = vec![f32::NAN; kc * MR];
                    let mut got = vec![f32::NAN; kc * MR];
                    // SAFETY: regions are in bounds; AVX2 is available.
                    unsafe {
                        pack_a_ref::<MR>(&x, i0, mr, p0, kc, want.as_mut_ptr());
                        pack_a(&x, i0, mr, p0, kc, got.as_mut_ptr());
                    }
                    assert_eq!(want, got, "A {t:?} i0={i0} mr={mr} p0={p0} kc={kc}");
                    // B panel: depth p0..p0+kc (logical rows), columns j0..j0+nr.
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
