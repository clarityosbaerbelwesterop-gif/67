//! IEEE 754 binary16 (s.5.10, bias 15), including subnormals, ±Inf and NaN.

use crate::engine::{
    map_rounding, map_rounding_inplace, multiversion, round_magnitude_with, MiniFloat,
};
use crate::Rounding;

/// Largest finite value, 65504.
pub const MAX_FINITE: u16 = 0x7bff;
/// Canonical quiet NaN.
pub const NAN: u16 = 0x7e00;
/// +Infinity.
pub const INFINITY: u16 = 0x7c00;

pub(crate) const FP16: MiniFloat = MiniFloat {
    m_bits: 10,
    bias: 15,
    max_code: MAX_FINITE as u32,
};

/// f32 → binary16, round to nearest even, branch-free integer code.
/// Overflow (|x| >= 65520) gives ±Inf; NaN stays NaN (quieted, sign and
/// upper payload kept); tiny values round into the subnormal range.
#[inline(always)]
pub fn from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let a = b & 0x7fff_ffff;

    // Normal range (|x| >= 2^-14): rebias and round off 13 bits. The carry
    // propagates into the exponent; anything >= 65520 lands at/after 0x7c00.
    let norm = (a
        .wrapping_sub(112 << 23)
        .wrapping_add(0x0fff + ((a >> 13) & 1))
        >> 13)
        .min(0x7c00);

    // Subnormal range: significand with implicit bit, shifted right by
    // 13 + (113 - exp). Shifts >= 25 always round to zero; clamp to 31.
    let e = (a >> 23) as i32;
    let m = (a & 0x007f_ffff) | 0x0080_0000;
    let d = (126 - e).clamp(1, 31) as u32;
    let q = m >> d;
    let rem = m & ((1u32 << d) - 1);
    let half = 1u32 << (d - 1);
    let sub = q + u32::from(rem + (q & 1) > half);

    let nan = 0x7e00 | ((a >> 13) & 0x03ff);
    let mag = if a > 0x7f80_0000 {
        nan
    } else if a >= 0x3880_0000 {
        norm
    } else {
        sub
    };
    (sign | mag) as u16
}

/// binary16 → f32 (exact). NaN payloads are preserved (shifted left by 13).
#[inline(always)]
pub fn to_f32(h: u16) -> f32 {
    let h = u32::from(h);
    let sign = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x03ff;
    let normal = ((e + 112) << 23) | (m << 13);
    let special = 0x7f80_0000 | (m << 13);
    // Subnormal m * 2^-24: leading one at bit p -> exponent p - 24.
    let p = 31 - (m | 1).leading_zeros();
    let sub = if m == 0 {
        0
    } else {
        ((p + 103) << 23) | ((m << (23 - p)) & 0x007f_ffff)
    };
    let mag = if e == 0x1f {
        special
    } else if e != 0 {
        normal
    } else {
        sub
    };
    f32::from_bits(sign | mag)
}

/// f32 → binary16 with an explicit rounding mode.
///
/// `TowardZero` never overflows a finite input (gives ±65504). NaN → quiet
/// NaN, ±Inf → ±Inf in every mode.
#[inline(always)]
pub fn encode(x: f32, r: Rounding) -> u16 {
    if r == Rounding::NearestEven {
        return from_f32(x);
    }
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let a = b & 0x7fff_ffff;
    // Every f32 subnormal is far below the binary16 range: no normalisation.
    let c = round_magnitude_with::<false>(a, 0, FP16, r);
    let finite_tz = a < 0x7f80_0000 && r == Rounding::TowardZero;
    let ovf = if finite_tz {
        FP16.max_code
    } else {
        u32::from(INFINITY)
    };
    let mag = if c > FP16.max_code { ovf } else { c };
    let mag = if a > 0x7f80_0000 {
        0x7e00 | ((a >> 13) & 0x03ff)
    } else {
        mag
    };
    (sign | mag) as u16
}

/// Alias of [`to_f32`], named like the other formats' decoders.
#[inline(always)]
pub fn decode(h: u16) -> f32 {
    to_f32(h)
}

multiversion! {
    /// Encode a slice. For `Stochastic(seed)`, element `i` uses
    /// [`crate::stochastic_bits`]`(seed, i)`.
    ///
    /// # Panics
    /// If `x.len() != out.len()`.
    pub fn encode_slice(x: &[f32], out: &mut [u16], r: Rounding) {
        assert_eq!(x.len(), out.len(), "fp16::encode_slice: length mismatch");
        map_rounding(x, out, 0, r, encode);
    }
}

multiversion! {
    /// Decode a slice.
    ///
    /// # Panics
    /// If `h.len() != out.len()`.
    pub fn decode_slice(h: &[u16], out: &mut [f32]) {
        assert_eq!(h.len(), out.len(), "fp16::decode_slice: length mismatch");
        for (o, &v) in out.iter_mut().zip(h) {
            *o = to_f32(v);
        }
    }
}

multiversion! {
    /// Round every element to binary16 in place (see [`crate::fake_quant`]).
    pub fn fake_quant_slice(x: &mut [f32], r: Rounding) {
        map_rounding_inplace(x, 0, r, |v, ri| to_f32(encode(v, ri)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::round_magnitude;

    fn generic_rne(x: f32) -> u16 {
        let b = x.to_bits();
        let a = b & 0x7fff_ffff;
        if a >= 0x7f80_0000 {
            return from_f32(x);
        }
        let c = round_magnitude(a, 0, FP16, Rounding::NearestEven).min(0x7c00);
        (((b >> 16) & 0x8000) | c) as u16
    }

    #[test]
    fn known_values() {
        assert_eq!(from_f32(1.0), 0x3c00);
        assert_eq!(from_f32(-2.0), 0xc000);
        assert_eq!(from_f32(65504.0), 0x7bff);
        assert_eq!(from_f32(65519.996), 0x7bff);
        assert_eq!(from_f32(65520.0), 0x7c00); // tie with odd max -> Inf
        assert_eq!(from_f32(1e10), 0x7c00);
        assert_eq!(from_f32(-1e10), 0xfc00);
        assert_eq!(encode(1e10, Rounding::TowardZero), 0x7bff);
        assert_eq!(encode(-1e10, Rounding::TowardZero), 0xfbff);
        assert_eq!(from_f32(2f32.powi(-14)), 0x0400);
        assert_eq!(from_f32(2f32.powi(-24)), 0x0001);
        assert_eq!(from_f32(2f32.powi(-25)), 0x0000); // tie to even (0)
        assert_eq!(
            from_f32(f32::from_bits(2f32.powi(-25).to_bits() + 1)),
            0x0001
        );
        assert_eq!(from_f32(3.0 * 2f32.powi(-25)), 0x0002); // 1.5 ulp tie -> 2
        assert_eq!(from_f32(-0.0), 0x8000);
        assert_eq!(from_f32(f32::from_bits(1)), 0);
        assert_eq!(from_f32(f32::INFINITY), 0x7c00);
        assert_eq!(to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        assert_eq!(to_f32(0x7bff), 65504.0);
        assert_eq!(to_f32(0x7c00), f32::INFINITY);
        assert_eq!(to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(to_f32(0x7e00).is_nan());
        assert_eq!(to_f32(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(to_f32(0x3555), f32::from_bits(0x3eaa_a000));
    }

    #[test]
    fn decode_all_codes_roundtrip() {
        for h in 0..=u16::MAX {
            let x = to_f32(h);
            let e = (h >> 10) & 0x1f;
            let m = h & 0x3ff;
            if e == 0x1f && m != 0 {
                assert!(x.is_nan());
                assert_eq!(from_f32(x), h | 0x0200, "NaN payload kept, quieted");
                continue;
            }
            // value from the definition, in f64
            let mag = if e == 0 {
                m as f64 * 2f64.powi(-24)
            } else if e == 0x1f {
                f64::INFINITY
            } else {
                (1.0 + m as f64 / 1024.0) * 2f64.powi(e as i32 - 15)
            };
            let v = if h & 0x8000 != 0 { -mag } else { mag };
            assert_eq!(x as f64, v, "{h:#x}");
            assert_eq!(from_f32(x), h, "{h:#x}");
            for r in [
                Rounding::TowardZero,
                Rounding::Stochastic(u32::MAX),
                Rounding::Stochastic(0),
            ] {
                assert_eq!(encode(x, r), h, "{h:#x} {r:?}");
            }
        }
    }

    #[test]
    fn fast_rne_matches_generic_engine() {
        // Every 7th f32 bit pattern: ~613M would be slow in debug; stride keeps it fast.
        let mut b = 0u32;
        while b < 0x7f80_0000 {
            let x = f32::from_bits(b);
            assert_eq!(from_f32(x), generic_rne(x), "{b:#x}");
            assert_eq!(from_f32(-x), generic_rne(-x), "{b:#x}");
            b += 4093;
        }
    }

    #[test]
    fn slices() {
        let x: Vec<f32> = (0..3000).map(|i| (i as f32 - 1500.0) * 37.77).collect();
        let mut h = vec![0u16; x.len()];
        let mut y = vec![0f32; x.len()];
        for r in [
            Rounding::NearestEven,
            Rounding::TowardZero,
            Rounding::Stochastic(77),
        ] {
            encode_slice(&x, &mut h, r);
            decode_slice(&h, &mut y);
            let mut z = x.clone();
            fake_quant_slice(&mut z, r);
            for i in 0..x.len() {
                let ri = match r {
                    Rounding::Stochastic(s) => {
                        Rounding::Stochastic(crate::stochastic_bits(s, i as u64))
                    }
                    o => o,
                };
                assert_eq!(h[i], encode(x[i], ri));
                assert_eq!(y[i].to_bits(), to_f32(h[i]).to_bits());
                assert_eq!(z[i].to_bits(), y[i].to_bits());
            }
        }
    }
}
