//! bfloat16: the upper 16 bits of an IEEE binary32 (s.8.7, bias 127).
//!
//! BF16 shares f32's exponent range, so every conversion is a pure operation
//! on the f32 bit pattern: the low 16 bits are rounded away.

use crate::Rounding;

/// Largest finite BF16 value, `0x7F7F` (≈ 3.3895e38).
pub const MAX_FINITE: u16 = 0x7f7f;
/// Canonical quiet NaN.
pub const NAN: u16 = 0x7fc0;
/// +Infinity.
pub const INFINITY: u16 = 0x7f80;

/// f32 → BF16, round to nearest even. Overflow rounds to ±Inf; NaN stays
/// NaN (quieted, sign and upper payload kept).
#[inline(always)]
pub fn from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let rne = b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16;
    let nan = (b >> 16) | 0x0040;
    let is_nan = (b & 0x7fff_ffff) > 0x7f80_0000;
    (if is_nan { nan } else { rne }) as u16
}

/// BF16 → f32 (exact; the bit pattern is widened unchanged).
#[inline(always)]
pub fn to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// f32 → BF16 with an explicit rounding mode.
///
/// `TowardZero` truncates (never overflows to Inf from a finite input).
/// `Stochastic(bits)` adds `bits >> 16` to the 16 discarded bits, then
/// truncates; it may overflow to ±Inf just like `NearestEven`.
#[inline(always)]
pub fn encode(x: f32, r: Rounding) -> u16 {
    let b = x.to_bits();
    let abs = b & 0x7fff_ffff;
    match r {
        Rounding::NearestEven => from_f32(x),
        Rounding::TowardZero => {
            let nan = (b >> 16) | 0x0040;
            (if abs > 0x7f80_0000 { nan } else { b >> 16 }) as u16
        }
        Rounding::Stochastic(bits) => {
            // Inf and NaN must not absorb the random addend.
            let special = if abs > 0x7f80_0000 { (b >> 16) | 0x0040 } else { b >> 16 };
            let sr = b.wrapping_add(bits >> 16) >> 16;
            (if abs >= 0x7f80_0000 { special } else { sr }) as u16
        }
    }
}

/// Alias of [`to_f32`], named like the other formats' decoders.
#[inline(always)]
pub fn decode(h: u16) -> f32 {
    to_f32(h)
}

/// Encode a slice. For `Stochastic(seed)`, element `i` uses
/// [`crate::stochastic_bits`]`(seed, i)`.
///
/// # Panics
/// If `x.len() != out.len()`.
pub fn encode_slice(x: &[f32], out: &mut [u16], r: Rounding) {
    assert_eq!(x.len(), out.len(), "bf16::encode_slice: length mismatch");
    match r {
        Rounding::NearestEven => {
            for (o, &v) in out.iter_mut().zip(x) {
                *o = from_f32(v);
            }
        }
        _ => crate::engine::for_each_rounding(x.len(), 0, r, |i, ri| out[i] = encode(x[i], ri)),
    }
}

/// Decode a slice.
///
/// # Panics
/// If `h.len() != out.len()`.
pub fn decode_slice(h: &[u16], out: &mut [f32]) {
    assert_eq!(h.len(), out.len(), "bf16::decode_slice: length mismatch");
    for (o, &v) in out.iter_mut().zip(h) {
        *o = to_f32(v);
    }
}

/// Round every element to the nearest BF16 value in place
/// (see [`crate::fake_quant`]).
pub fn fake_quant_slice(x: &mut [f32], r: Rounding) {
    match r {
        Rounding::NearestEven => {
            for v in x.iter_mut() {
                *v = to_f32(from_f32(*v));
            }
        }
        _ => crate::engine::for_each_rounding(x.len(), 0, r, |i, ri| x[i] = to_f32(encode(x[i], ri))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{round_magnitude, MiniFloat};

    const BF16: MiniFloat = MiniFloat { m_bits: 7, bias: 127, max_code: MAX_FINITE as u32 };

    /// Reference via the generic engine (independent code path).
    fn generic(x: f32, r: Rounding) -> u16 {
        let b = x.to_bits();
        let sign = (b >> 16) & 0x8000;
        let abs = b & 0x7fff_ffff;
        if abs > 0x7f80_0000 {
            return ((b >> 16) | 0x40) as u16;
        }
        if abs == 0x7f80_0000 {
            return (sign | 0x7f80) as u16;
        }
        let c = round_magnitude(abs, 0, BF16, r);
        let c = if c > BF16.max_code {
            if r == Rounding::TowardZero {
                BF16.max_code
            } else {
                0x7f80
            }
        } else {
            c
        };
        (sign | c) as u16
    }

    #[test]
    fn known_values() {
        assert_eq!(from_f32(1.0), 0x3f80);
        assert_eq!(from_f32(-2.0), 0xc000);
        assert_eq!(from_f32(0.0), 0x0000);
        assert_eq!(from_f32(-0.0), 0x8000);
        assert_eq!(from_f32(f32::INFINITY), 0x7f80);
        assert_eq!(from_f32(f32::NEG_INFINITY), 0xff80);
        assert_eq!(from_f32(f32::MAX), 0x7f80); // rounds up past max
        assert_eq!(encode(f32::MAX, Rounding::TowardZero), MAX_FINITE);
        assert_eq!(from_f32(f32::from_bits(0x7f7f_7fff)), MAX_FINITE);
        assert_eq!(from_f32(f32::from_bits(0x7f7f_8000)), 0x7f80); // tie, odd -> up
        assert_eq!(to_f32(0x3f80), 1.0);
        assert_eq!(to_f32(0x0001), f32::from_bits(0x0001_0000)); // subnormal
        // ties
        assert_eq!(from_f32(f32::from_bits(0x3f80_8000)), 0x3f80); // even stays
        assert_eq!(from_f32(f32::from_bits(0x3f81_8000)), 0x3f82); // odd rounds up
        assert_eq!(from_f32(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(from_f32(f32::from_bits(0x3f80_7fff)), 0x3f80);
    }

    #[test]
    fn nan_is_quiet_and_signed() {
        for b in [0x7f80_0001u32, 0x7fc0_0000, 0xff80_0001, 0xffff_ffff, 0x7fbf_ffff] {
            for r in [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(u32::MAX)] {
                let h = encode(f32::from_bits(b), r);
                assert!(to_f32(h).is_nan(), "{b:#x} {r:?}");
                assert_eq!(h & 0x0040, 0x0040, "quiet bit");
                assert_eq!(u32::from(h >> 15), b >> 31, "sign");
            }
        }
    }

    #[test]
    fn infinities_survive_every_mode() {
        for r in [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(u32::MAX)] {
            assert_eq!(encode(f32::INFINITY, r), 0x7f80);
            assert_eq!(encode(f32::NEG_INFINITY, r), 0xff80);
        }
    }

    #[test]
    fn fast_paths_match_generic_engine() {
        let mut s = 0x1234_5678_9abc_def0u64;
        for i in 0..2_000_000u32 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let b = if i < 1 << 16 { i << 16 | (s as u32 & 0xffff) } else { s as u32 };
            let x = f32::from_bits(b);
            assert_eq!(encode(x, Rounding::NearestEven), generic(x, Rounding::NearestEven), "{b:#x}");
            assert_eq!(encode(x, Rounding::TowardZero), generic(x, Rounding::TowardZero), "{b:#x}");
            // For f32 normals the generic engine uses the same 16 random bits.
            let ab = b & 0x7fff_ffff;
            if ab >= 0x0080_0000 && ab < 0x7f80_0000 {
                let rb = (s >> 32) as u32;
                assert_eq!(
                    encode(x, Rounding::Stochastic(rb)),
                    generic(x, Rounding::Stochastic(rb)),
                    "{b:#x}"
                );
            }
        }
    }

    #[test]
    fn slices() {
        let x: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.123_456).collect();
        let mut h = vec![0u16; x.len()];
        encode_slice(&x, &mut h, Rounding::NearestEven);
        let mut y = vec![0f32; x.len()];
        decode_slice(&h, &mut y);
        for i in 0..x.len() {
            assert_eq!(h[i], from_f32(x[i]));
            assert_eq!(y[i], to_f32(h[i]));
        }
        let mut z = x.clone();
        fake_quant_slice(&mut z, Rounding::NearestEven);
        assert_eq!(z, y);
        encode_slice(&x, &mut h, Rounding::TowardZero);
        for i in 0..x.len() {
            assert_eq!(h[i], encode(x[i], Rounding::TowardZero));
        }
        encode_slice(&x, &mut h, Rounding::Stochastic(9));
        let mut z = x.clone();
        fake_quant_slice(&mut z, Rounding::Stochastic(9));
        for i in 0..x.len() {
            assert_eq!(h[i], encode(x[i], Rounding::Stochastic(crate::stochastic_bits(9, i as u64))));
            assert_eq!(z[i], to_f32(h[i]));
        }
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn slice_length_mismatch_panics() {
        encode_slice(&[1.0, 2.0], &mut [0u16; 3], Rounding::NearestEven);
    }
}
