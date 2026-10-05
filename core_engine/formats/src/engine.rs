//! Shared bit-level rounding engine for every "minifloat" in this crate.
//!
//! A target format is described by its mantissa width `M`, exponent bias and
//! the largest finite magnitude code. Its magnitude codes are laid out
//! IEEE-style: `code = exponent_field << M | mantissa`, exponent field 0 holds
//! the subnormals (quantum `2^(1 - bias - M)`). The same engine also covers
//! the MX INT8 element (bias 0, M = 7: every in-range value is "subnormal"
//! with quantum `2^-6`), because a fixed-point grid is a subnormal-only float.
//!
//! The rounding is done purely on integers:
//!
//! 1. decompose the f32 into a 24-bit significand `m` (normalised so that bit
//!    23 is set, also for f32 subnormals) and an unbiased exponent `e`;
//! 2. compute how many low bits `d` of `m` fall below the target's kept
//!    mantissa (more when the result is subnormal in the target);
//! 3. round `m >> d` with the requested [`Rounding`];
//! 4. assemble `((max(e, emin) + bias - 1) << M) + q`. Because `q` still
//!    carries the implicit bit, a mantissa carry naturally bumps the exponent
//!    field, and a subnormal that rounds up to `2^M` naturally becomes the
//!    smallest normal.
//!
//! The returned code uses an unbounded exponent, so a value that rounds above
//! the largest finite value yields a code greater than `max_code`; callers
//! decide whether that saturates, becomes Inf, or becomes NaN.

use crate::Rounding;

/// Parameters of a sign-magnitude binary floating-point format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MiniFloat {
    /// Explicit mantissa bits.
    pub m_bits: u32,
    /// Exponent bias.
    pub bias: i32,
    /// Largest finite magnitude code (sign bit excluded).
    pub max_code: u32,
}

impl MiniFloat {
    /// Unbiased exponent of the smallest normal number.
    #[inline(always)]
    pub(crate) const fn emin(self) -> i32 {
        1 - self.bias
    }
}

/// Drop the low `d` bits of `m` (1 <= d <= 63) with rounding mode `r`.
///
/// `Stochastic(bits)` aligns the 32 random bits directly below the kept part
/// (the most significant random bit has weight one half of the kept LSB),
/// adds them, then truncates. An exactly representable input is therefore
/// never changed, and the round-up probability equals the dropped fraction
/// (truncated to a multiple of 2^-32 when more than 32 bits are dropped).
#[inline(always)]
pub(crate) fn round_shift(m: u64, d: u32, r: Rounding) -> u64 {
    debug_assert!((1..=63).contains(&d), "shift {d} out of range");
    debug_assert!(m < 1 << 40);
    let q = m >> d;
    match r {
        Rounding::TowardZero => q,
        Rounding::NearestEven => {
            let half = 1u64 << (d - 1);
            let rem = m & ((1u64 << d) - 1);
            // rem > half, or rem == half with odd q.
            q + ((rem + (q & 1) > half) as u64)
        }
        Rounding::Stochastic(bits) => {
            let add = if d >= 32 {
                u64::from(bits) << (d - 32)
            } else {
                u64::from(bits) >> (32 - d)
            };
            (m + add) >> d
        }
    }
}

/// Round the finite, non-negative f32 whose bit pattern is `abs`, divided by
/// `2^exp_shift`, onto the grid of `f`. Returns the magnitude code with an
/// unbounded exponent (it can exceed `f.max_code`).
#[inline(always)]
pub(crate) fn round_magnitude(abs: u32, exp_shift: i32, f: MiniFloat, r: Rounding) -> u32 {
    debug_assert!(abs < 0x7f80_0000, "non-finite input reached the rounding core");
    let ef = (abs >> 23) as i32;
    let m0 = (abs & 0x007f_ffff) | (u32::from(ef != 0) << 23);
    // Normalise f32 subnormals so that bit 23 is the leading one. Normal
    // numbers have exactly 8 leading zeros (k = 0); zero gets k = 24, m = 0.
    let k = m0.leading_zeros().saturating_sub(8);
    let m = m0 << k;
    let e = ef.max(1) - 127 - k as i32 - exp_shift;
    let emin = f.emin();
    let d = (23 - f.m_bits as i32 + (emin - e).max(0)).min(63) as u32;
    let field = (e.max(emin) + f.bias - 1) as u32;
    let q = round_shift(u64::from(m), d, r) as u32;
    (field << f.m_bits) + q
}

/// Decode a finite magnitude code of `f` (exponent field width implied by the
/// code) to f32 bits. Valid for formats whose every value is an f32 normal or
/// zero (true for FP16, FP8, FP4 and INT8-as-minifloat).
pub(crate) const fn decode_magnitude_bits(code: u32, f: MiniFloat) -> u32 {
    let ef = code >> f.m_bits;
    let mant = code & ((1 << f.m_bits) - 1);
    if ef == 0 {
        if mant == 0 {
            return 0;
        }
        // value = mant * 2^(emin - M); leading one at bit p of mant.
        let p = 31 - mant.leading_zeros();
        let exp = f.emin() - f.m_bits as i32 + p as i32;
        (((exp + 127) as u32) << 23) | ((mant << (23 - p)) & 0x007f_ffff)
    } else {
        let exp = ef as i32 - f.bias;
        (((exp + 127) as u32) << 23) | (mant << (23 - f.m_bits))
    }
}

/// `2^s` as an f32 for `s` in `[-149, 127]` (exact, subnormal below -126).
#[inline(always)]
pub(crate) const fn pow2(s: i32) -> f32 {
    debug_assert!(s >= -149 && s <= 127);
    if s >= -126 {
        f32::from_bits(((s + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (s + 149))
    }
}

/// `floor(log2(v))` for the positive finite f32 with bit pattern `abs`
/// (`abs` in `1..0x7f80_0000`).
#[inline(always)]
pub(crate) fn floor_log2_bits(abs: u32) -> i32 {
    debug_assert!(abs != 0 && abs < 0x7f80_0000);
    let ef = (abs >> 23) as i32;
    if ef != 0 {
        ef - 127
    } else {
        // Subnormal: value = abs * 2^-149, top bit p.
        (31 - abs.leading_zeros() as i32) - 149
    }
}

/// Counter-based random bits for slice-level stochastic rounding: a
/// SplitMix64 finaliser over `(seed, index)`. Element `index` of a slice
/// rounded with `Rounding::Stochastic(seed)` uses these bits.
#[inline(always)]
pub fn stochastic_bits(seed: u32, index: u64) -> u32 {
    let mut z = index
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(u64::from(seed).wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 32) as u32
}

/// Run `f(i, element_rounding)` for `i in 0..n`, with the rounding-mode match
/// hoisted out of the loop. For `Stochastic(seed)`, element `i` gets
/// `Stochastic(stochastic_bits(seed, base + i))`.
#[inline(always)]
pub(crate) fn for_each_rounding(n: usize, base: u64, r: Rounding, mut f: impl FnMut(usize, Rounding)) {
    match r {
        Rounding::NearestEven => (0..n).for_each(|i| f(i, Rounding::NearestEven)),
        Rounding::TowardZero => (0..n).for_each(|i| f(i, Rounding::TowardZero)),
        Rounding::Stochastic(seed) => (0..n)
            .for_each(|i| f(i, Rounding::Stochastic(stochastic_bits(seed, base + i as u64)))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const E4M3: MiniFloat = MiniFloat { m_bits: 3, bias: 7, max_code: 0x7e };

    #[test]
    fn round_shift_modes() {
        // 0b1011 >> 2: q = 2, rem = 3 > half 2 -> 3
        assert_eq!(round_shift(0b1011, 2, Rounding::NearestEven), 3);
        assert_eq!(round_shift(0b1011, 2, Rounding::TowardZero), 2);
        // ties: 0b1010 (q=2 even, rem=half) -> 2 ; 0b1110 (q=3 odd) -> 4
        assert_eq!(round_shift(0b1010, 2, Rounding::NearestEven), 2);
        assert_eq!(round_shift(0b1110, 2, Rounding::NearestEven), 4);
        // stochastic extremes
        assert_eq!(round_shift(0b1001, 2, Rounding::Stochastic(0)), 2);
        assert_eq!(round_shift(0b1001, 2, Rounding::Stochastic(u32::MAX)), 3);
        assert_eq!(round_shift(0b1000, 2, Rounding::Stochastic(u32::MAX)), 2);
        // d >= 32: the dropped fraction is resolved to 2^-32 only
        assert_eq!(round_shift(1, 40, Rounding::Stochastic(u32::MAX)), 0);
        assert_eq!(round_shift(1 << 8, 40, Rounding::Stochastic(u32::MAX)), 1);
        assert_eq!(round_shift(1 << 8, 40, Rounding::Stochastic(u32::MAX - 1)), 0);
        assert_eq!(round_shift(3 << 38, 40, Rounding::Stochastic(1 << 30)), 1);
        assert_eq!(round_shift(3 << 38, 40, Rounding::Stochastic((1 << 30) - 1)), 0);
        assert_eq!(round_shift(0, 63, Rounding::Stochastic(u32::MAX)), 0);
        assert_eq!(round_shift((1 << 24) - 1, 63, Rounding::NearestEven), 0);
    }

    #[test]
    fn magnitude_basics() {
        let c = |x: f32| round_magnitude(x.to_bits(), 0, E4M3, Rounding::NearestEven);
        assert_eq!(c(0.0), 0);
        assert_eq!(c(1.0), 0x38);
        assert_eq!(c(448.0), 0x7e);
        assert_eq!(c(2f32.powi(-9)), 0x01);
        assert_eq!(c(2f32.powi(-6)), 0x08);
        assert_eq!(c(1e30), ((99 + 7 - 1) << 3) + 13); // 1e30 = 1.578 * 2^99 -> q = 13/8, unbounded exponent
        assert_eq!(c(f32::from_bits(1)), 0);
        // shift: 3 * 2^-100 scaled by 2^-100 -> 3
        let x = 3.0 * 2f32.powi(-100);
        assert_eq!(round_magnitude(x.to_bits(), -100, E4M3, Rounding::NearestEven), 0x44);
    }

    #[test]
    fn decode_magnitude() {
        assert_eq!(f32::from_bits(decode_magnitude_bits(0x7e, E4M3)), 448.0);
        assert_eq!(f32::from_bits(decode_magnitude_bits(0x01, E4M3)), 2f32.powi(-9));
        assert_eq!(f32::from_bits(decode_magnitude_bits(0x07, E4M3)), 7.0 * 2f32.powi(-9));
        assert_eq!(f32::from_bits(decode_magnitude_bits(0x00, E4M3)), 0.0);
    }

    #[test]
    fn pow2_and_log2() {
        for s in -149..=127 {
            assert_eq!(pow2(s) as f64, 2f64.powi(s));
        }
        for s in -149..=127 {
            let b = pow2(s).to_bits();
            assert_eq!(floor_log2_bits(b), s);
            if s > -149 {
                assert_eq!(floor_log2_bits(b + 1), s, "s={s}");
            }
            if s < 127 {
                assert_eq!(floor_log2_bits(pow2(s + 1).to_bits() - 1), s);
            }
        }
    }

    #[test]
    fn stochastic_bits_spread() {
        // Not a statistical proof, just a sanity check against gross bias.
        let n = 1 << 16;
        let mut ones = [0u32; 32];
        for i in 0..n {
            let b = stochastic_bits(12345, i);
            for (k, o) in ones.iter_mut().enumerate() {
                *o += (b >> k) & 1;
            }
        }
        for o in ones {
            let p = o as f64 / n as f64;
            assert!((p - 0.5).abs() < 0.02, "bit frequency {p}");
        }
        assert_ne!(stochastic_bits(1, 0), stochastic_bits(2, 0));
        assert_ne!(stochastic_bits(1, 0), stochastic_bits(1, 1));
    }
}
