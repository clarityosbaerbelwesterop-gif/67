//! x86-64 engines. Every function using an instruction-set extension is a
//! `#[target_feature]` function reached only after runtime detection.

use std::arch::x86_64::*;

pub(crate) mod amx;
pub(crate) mod avx2;
pub(crate) mod avx512;
#[cfg(test)]
mod probe;

/// In-register transpose of an 8×8 f32 block: on return `r[j]` holds column
/// `j` of the input rows.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose8x8(r: &mut [__m256; 8]) {
    let t0 = _mm256_unpacklo_ps(r[0], r[1]);
    let t1 = _mm256_unpackhi_ps(r[0], r[1]);
    let t2 = _mm256_unpacklo_ps(r[2], r[3]);
    let t3 = _mm256_unpackhi_ps(r[2], r[3]);
    let t4 = _mm256_unpacklo_ps(r[4], r[5]);
    let t5 = _mm256_unpackhi_ps(r[4], r[5]);
    let t6 = _mm256_unpacklo_ps(r[6], r[7]);
    let t7 = _mm256_unpackhi_ps(r[6], r[7]);
    // u: columns (c | c+4) of rows 0-3 (u0..u3) and rows 4-7 (u4..u7).
    let u0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    let u1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
    let u2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    let u3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    let u4 = _mm256_shuffle_ps::<0x44>(t4, t6);
    let u5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
    let u6 = _mm256_shuffle_ps::<0x44>(t5, t7);
    let u7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
    r[0] = _mm256_permute2f128_ps::<0x20>(u0, u4);
    r[1] = _mm256_permute2f128_ps::<0x20>(u1, u5);
    r[2] = _mm256_permute2f128_ps::<0x20>(u2, u6);
    r[3] = _mm256_permute2f128_ps::<0x20>(u3, u7);
    r[4] = _mm256_permute2f128_ps::<0x31>(u0, u4);
    r[5] = _mm256_permute2f128_ps::<0x31>(u1, u5);
    r[6] = _mm256_permute2f128_ps::<0x31>(u2, u6);
    r[7] = _mm256_permute2f128_ps::<0x31>(u3, u7);
}

/// In-register transpose of a 16×16 block of 32-bit elements: on return `r[j]`
/// holds column `j` of the input rows.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) fn transpose16x16(r: &mut [__m512i; 16]) {
    // Stage 1: t[2i] lane L = r2i[4L], r2i+1[4L], r2i[4L+1], r2i+1[4L+1];
    //          t[2i+1] the same for elements 4L+2, 4L+3.
    let mut t = [_mm512_setzero_si512(); 16];
    for i in 0..8 {
        t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
    }
    // Stage 2: u[4i+c] lane L = column 4L+c, rows 4i..4i+4.
    let mut u = [_mm512_setzero_si512(); 16];
    for i in 0..4 {
        u[4 * i] = _mm512_unpacklo_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 1] = _mm512_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 2] = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
        u[4 * i + 3] = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
    }
    // Stage 3: v[c]    = col c rows 0-3, col 8+c rows 0-3, col c rows 4-7, col 8+c rows 4-7
    //          v[4+c]  = the same for columns 4+c / 12+c; v[8+c], v[12+c] for rows 8-15.
    let mut v = [_mm512_setzero_si512(); 16];
    for c in 0..4 {
        v[c] = _mm512_shuffle_i32x4::<0x88>(u[c], u[4 + c]);
        v[4 + c] = _mm512_shuffle_i32x4::<0xDD>(u[c], u[4 + c]);
        v[8 + c] = _mm512_shuffle_i32x4::<0x88>(u[8 + c], u[12 + c]);
        v[12 + c] = _mm512_shuffle_i32x4::<0xDD>(u[8 + c], u[12 + c]);
    }
    // Stage 4: whole columns.
    for c in 0..4 {
        r[c] = _mm512_shuffle_i32x4::<0x88>(v[c], v[8 + c]);
        r[8 + c] = _mm512_shuffle_i32x4::<0xDD>(v[c], v[8 + c]);
        r[4 + c] = _mm512_shuffle_i32x4::<0x88>(v[4 + c], v[12 + c]);
        r[12 + c] = _mm512_shuffle_i32x4::<0xDD>(v[4 + c], v[12 + c]);
    }
}

/// Mask selecting the first `w` (`<= 16`) lanes.
#[inline(always)]
pub(crate) fn mask16(w: usize) -> u16 {
    debug_assert!(w <= 16);
    ((1u32 << w) - 1) as u16
}

/// AVX2 lane mask (all bits set in the first `w` (`<= 8`) lanes) for
/// `_mm256_maskload_ps` / `_mm256_maskstore_ps`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn mask8(w: usize) -> __m256i {
    debug_assert!(w <= 8);
    _mm256_cmpgt_epi32(
        _mm256_set1_epi32(w as i32),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose8x8_matches_scalar() {
        if !avx2::supported() {
            return;
        }
        let src: Vec<f32> = (0..64).map(|x| x as f32).collect();
        // SAFETY: AVX2 is available (checked above).
        let out = unsafe {
            let mut r = [_mm256_setzero_ps(); 8];
            for (i, row) in r.iter_mut().enumerate() {
                *row = _mm256_loadu_ps(src.as_ptr().add(i * 8));
            }
            transpose8x8(&mut r);
            let mut out = [0.0f32; 64];
            for (j, col) in r.iter().enumerate() {
                _mm256_storeu_ps(out.as_mut_ptr().add(j * 8), *col);
            }
            out
        };
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(out[j * 8 + i], src[i * 8 + j]);
            }
        }
    }

    #[test]
    fn transpose16x16_matches_scalar() {
        if !avx512::supported() {
            return;
        }
        let src: Vec<i32> = (0..256).collect();
        // SAFETY: AVX-512F is available (checked above).
        let out = unsafe {
            let mut r = [_mm512_setzero_si512(); 16];
            for (i, row) in r.iter_mut().enumerate() {
                *row = _mm512_loadu_si512(src.as_ptr().add(i * 16).cast());
            }
            transpose16x16(&mut r);
            let mut out = [0i32; 256];
            for (j, col) in r.iter().enumerate() {
                _mm512_storeu_si512(out.as_mut_ptr().add(j * 16).cast(), *col);
            }
            out
        };
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(out[j * 16 + i], src[i * 16 + j]);
            }
        }
    }

    #[test]
    fn masks() {
        assert_eq!(mask16(0), 0);
        assert_eq!(mask16(5), 0b11111);
        assert_eq!(mask16(16), 0xFFFF);
        if avx2::supported() {
            // SAFETY: AVX2 is available.
            let lanes: [i32; 8] = unsafe { std::mem::transmute(mask8(3)) };
            assert_eq!(lanes, [-1, -1, -1, 0, 0, 0, 0, 0]);
        }
    }
}
