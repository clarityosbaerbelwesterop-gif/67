//! FP4 E2M1 as defined by OCP MX v1.0: s.2.1, bias 1, one subnormal (0.5),
//! no Inf, no NaN. Magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
//!
//! Codes live in the low nibble of a `u8` (bit 3 = sign); the high nibble is
//! ignored on decode and zero on encode. Conversion always saturates.
//! Packed helpers store two codes per byte, low nibble first (the MX layout).

use crate::engine::{decode_magnitude_bits, for_each_rounding, round_magnitude, MiniFloat};
use crate::Rounding;

pub(crate) const E2M1: MiniFloat = MiniFloat { m_bits: 1, bias: 1, max_code: 7 };

/// Code of +6.0, the largest magnitude.
pub const MAX_CODE: u8 = 0x7;
/// Largest finite value.
pub const MAX_FINITE: f32 = 6.0;

const fn build_lut() -> [u32; 16] {
    let mut lut = [0u32; 16];
    let mut c = 0u32;
    while c < 16 {
        lut[c as usize] = ((c & 0x8) << 28) | decode_magnitude_bits(c & 0x7, E2M1);
        c += 1;
    }
    lut
}

pub(crate) static LUT: [u32; 16] = build_lut();

/// Decode the low nibble of `code`.
#[inline(always)]
pub fn decode(code: u8) -> f32 {
    f32::from_bits(LUT[usize::from(code & 0xf)])
}

/// Encode to E2M1 (low nibble), saturating: finite overflow and ±Inf clamp
/// to ±6. FP4 has no NaN; a NaN input encodes to +6 (`0x7`), so it stays
/// conspicuous rather than silently becoming zero. `-0.0` → `0x8`.
#[inline(always)]
pub fn encode(x: f32, r: Rounding) -> u8 {
    let b = x.to_bits();
    let abs = b & 0x7fff_ffff;
    let sign = (b >> 28) & 0x8;
    if abs > 0x7f80_0000 {
        return MAX_CODE;
    }
    // Inf saturates; skip the engine for it (it expects finite input).
    let mag = if abs == 0x7f80_0000 { E2M1.max_code } else { round_magnitude(abs, 0, E2M1, r).min(E2M1.max_code) };
    (sign | mag) as u8
}

/// Encode a slice, one code per output byte (low nibble). For
/// `Stochastic(seed)`, element `i` uses [`crate::stochastic_bits`]`(seed, i)`.
///
/// # Panics
/// If `x.len() != out.len()`.
pub fn encode_slice(x: &[f32], out: &mut [u8], r: Rounding) {
    assert_eq!(x.len(), out.len(), "fp4::encode_slice: length mismatch");
    for_each_rounding(x.len(), 0, r, |i, ri| out[i] = encode(x[i], ri));
}

/// Decode a slice of one-code-per-byte values (high nibbles ignored).
///
/// # Panics
/// If `codes.len() != out.len()`.
pub fn decode_slice(codes: &[u8], out: &mut [f32]) {
    assert_eq!(codes.len(), out.len(), "fp4::decode_slice: length mismatch");
    for (o, &c) in out.iter_mut().zip(codes) {
        *o = decode(c);
    }
}

/// Number of bytes needed to pack `n` FP4 codes.
pub const fn packed_len(n: usize) -> usize {
    n.div_ceil(2)
}

/// Encode a slice into packed nibbles: element `i` goes to byte `i / 2`,
/// low nibble for even `i`. An unused final high nibble is zero.
///
/// # Panics
/// If `out.len() != packed_len(x.len())`.
pub fn encode_packed(x: &[f32], out: &mut [u8], r: Rounding) {
    assert_eq!(out.len(), packed_len(x.len()), "fp4::encode_packed: length mismatch");
    out.fill(0);
    for_each_rounding(x.len(), 0, r, |i, ri| out[i / 2] |= encode(x[i], ri) << (4 * (i & 1)));
}

/// Decode `out.len()` packed FP4 values.
///
/// # Panics
/// If `packed.len() != packed_len(out.len())`.
pub fn decode_packed(packed: &[u8], out: &mut [f32]) {
    assert_eq!(packed.len(), packed_len(out.len()), "fp4::decode_packed: length mismatch");
    let mut pairs = out.chunks_exact_mut(2);
    for (o, &p) in (&mut pairs).zip(packed) {
        o[0] = decode(p);
        o[1] = decode(p >> 4);
    }
    if let [last] = pairs.into_remainder() {
        *last = decode(packed[packed.len() - 1]);
    }
}

/// Fake-quantize in place (saturating); NaN passes through unchanged
/// (see [`crate::fake_quant`]).
pub fn fake_quant_slice(x: &mut [f32], r: Rounding) {
    for_each_rounding(x.len(), 0, r, |i, ri| {
        let v = x[i];
        if !v.is_nan() {
            x[i] = decode(encode(v, ri));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table() {
        let want = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (c, &w) in want.iter().enumerate() {
            assert_eq!(decode(c as u8), w);
            assert_eq!(decode(c as u8 | 0x8), -w);
            assert_eq!(decode(c as u8 | 0xf0), w, "high nibble ignored");
        }
        assert_eq!(decode(0x8).to_bits(), 0x8000_0000);
    }

    #[test]
    fn packing() {
        let x = [0.5f32, -6.0, 3.0, 1.0, -0.5];
        let mut p = [0xffu8; 3];
        encode_packed(&x, &mut p, Rounding::NearestEven);
        assert_eq!(p, [0xf1, 0x25, 0x09]);
        let mut y = [0f32; 5];
        decode_packed(&p, &mut y);
        assert_eq!(y, x);
        let mut y4 = [0f32; 4];
        decode_packed(&p[..2], &mut y4);
        assert_eq!(y4, x[..4]);
        let mut e: [u8; 0] = [];
        encode_packed(&[], &mut e, Rounding::NearestEven);
        decode_packed(&[], &mut []);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn packed_length_checked() {
        encode_packed(&[1.0, 2.0, 3.0], &mut [0u8; 1], Rounding::NearestEven);
    }
}
