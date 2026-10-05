//! OCP OFP8 rev 1.0: E4M3 and E5M2.
//!
//! * **E4M3** — s.4.3, bias 7, subnormals, no infinities. The only NaN
//!   encodings are `S.1111.111` (0x7F / 0xFF). Max finite ±448 (`S.1111.110`).
//! * **E5M2** — s.5.2, bias 15, IEEE-style: subnormals, ±Inf
//!   (`S.11111.00`), NaN (`S.11111.{01,10,11}`). Max finite ±57344.
//!
//! Decoding is a 256-entry table built at compile time from the bit-level
//! definition; encoding uses the crate's integer rounding engine.

use crate::engine::{
    decode_magnitude_bits, map_rounding, map_rounding_inplace, multiversion, round_magnitude_with,
    MiniFloat,
};
use crate::Rounding;

/// The two OFP8 encodings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// 4 exponent bits, 3 mantissa bits, bias 7, no Inf.
    E4M3,
    /// 5 exponent bits, 2 mantissa bits, bias 15, IEEE specials.
    E5M2,
}

#[derive(Clone, Copy)]
struct Spec {
    mf: MiniFloat,
    /// Magnitude code produced for NaN inputs.
    nan: u32,
    /// Magnitude code for non-saturating overflow (NaN for E4M3, Inf for E5M2).
    overflow: u32,
}

const E4M3: Spec = Spec {
    mf: MiniFloat {
        m_bits: 3,
        bias: 7,
        max_code: 0x7e,
    },
    nan: 0x7f,
    overflow: 0x7f,
};
const E5M2: Spec = Spec {
    mf: MiniFloat {
        m_bits: 2,
        bias: 15,
        max_code: 0x7b,
    },
    nan: 0x7e,
    overflow: 0x7c,
};

impl Kind {
    #[inline(always)]
    const fn spec(self) -> Spec {
        match self {
            Kind::E4M3 => E4M3,
            Kind::E5M2 => E5M2,
        }
    }

    /// Exponent bits.
    pub const fn exponent_bits(self) -> u32 {
        match self {
            Kind::E4M3 => 4,
            Kind::E5M2 => 5,
        }
    }

    /// Explicit mantissa bits.
    pub const fn mantissa_bits(self) -> u32 {
        self.spec().mf.m_bits
    }

    /// Exponent bias.
    pub const fn bias(self) -> i32 {
        self.spec().mf.bias
    }

    /// Unbiased exponent of the largest finite value (8 for E4M3, 15 for E5M2);
    /// the `emax` used by MX scale selection.
    pub const fn emax(self) -> i32 {
        match self {
            Kind::E4M3 => 8,
            Kind::E5M2 => 15,
        }
    }

    /// Largest finite magnitude code (0x7E for E4M3, 0x7B for E5M2).
    pub const fn max_code(self) -> u8 {
        self.spec().mf.max_code as u8
    }

    /// Canonical quiet NaN code (0x7F for E4M3, 0x7E for E5M2).
    pub const fn nan_code(self) -> u8 {
        self.spec().nan as u8
    }

    /// Whether `code` is a NaN encoding of this kind.
    #[inline(always)]
    pub const fn is_nan(self, code: u8) -> bool {
        match self {
            Kind::E4M3 => code & 0x7f == 0x7f,
            Kind::E5M2 => code & 0x7f > 0x7c,
        }
    }

    /// Whether `code` is ±Inf (only E5M2 has infinities).
    #[inline(always)]
    pub const fn is_inf(self, code: u8) -> bool {
        match self {
            Kind::E4M3 => false,
            Kind::E5M2 => code & 0x7f == 0x7c,
        }
    }
}

const fn build_lut(kind: Kind) -> [u32; 256] {
    let s = kind.spec();
    let mut lut = [0u32; 256];
    let mut c = 0u32;
    while c < 256 {
        let sign = (c & 0x80) << 24;
        let mag = c & 0x7f;
        let bits = if kind.is_nan(c as u8) {
            0x7fc0_0000
        } else if kind.is_inf(c as u8) {
            0x7f80_0000
        } else {
            decode_magnitude_bits(mag, s.mf)
        };
        lut[c as usize] = sign | bits;
        c += 1;
    }
    lut
}

static LUT_E4M3: [u32; 256] = build_lut(Kind::E4M3);
static LUT_E5M2: [u32; 256] = build_lut(Kind::E5M2);

#[inline(always)]
fn lut(kind: Kind) -> &'static [u32; 256] {
    match kind {
        Kind::E4M3 => &LUT_E4M3,
        Kind::E5M2 => &LUT_E5M2,
    }
}

/// Decode one FP8 code. NaN codes decode to a quiet NaN carrying the code's
/// sign; E5M2 infinities to ±Inf.
#[inline(always)]
pub fn decode(kind: Kind, code: u8) -> f32 {
    f32::from_bits(lut(kind)[usize::from(code)])
}

/// Largest finite value: 448 (E4M3) or 57344 (E5M2).
pub fn max_finite(kind: Kind) -> f32 {
    decode(kind, kind.max_code())
}

/// Encode one f32.
///
/// * NaN → NaN (`S.1111.111` for E4M3, `S.11111.10` for E5M2), sign kept.
/// * Finite overflow (the value rounds above max): `saturate=true` clamps to
///   ±max (448 / 57344); `false` maps to NaN (E4M3) / ±Inf (E5M2).
///   With `Rounding::TowardZero` a finite input never overflows (IEEE 754
///   §7.4): it gives ±max in both modes.
/// * ±Inf: `saturate=true` → ±max; `false` → NaN (E4M3) / ±Inf (E5M2).
/// * Underflow keeps the sign (−tiny → −0).
#[inline(always)]
pub fn encode(kind: Kind, x: f32, r: Rounding, saturate: bool) -> u8 {
    let s = kind.spec();
    let b = x.to_bits();
    let sign = (b >> 24) & 0x80;
    let abs = b & 0x7fff_ffff;
    // Branch-free: the engine yields an out-of-range code for Inf/NaN, and the
    // specials are patched in with selects. Without a shift every f32
    // subnormal is an FP8 subnormal, so the non-normalising path is exact.
    let c = round_magnitude_with::<false>(abs, 0, s.mf, r);
    let clamp = saturate || (abs < 0x7f80_0000 && r == Rounding::TowardZero);
    let ovf = if clamp { s.mf.max_code } else { s.overflow };
    let mag = if c > s.mf.max_code { ovf } else { c };
    let mag = if abs > 0x7f80_0000 { s.nan } else { mag };
    (sign | mag) as u8
}

multiversion! {
    /// Encode a slice. For `Stochastic(seed)`, element `i` uses
    /// [`crate::stochastic_bits`]`(seed, i)`.
    ///
    /// # Panics
    /// If `x.len() != out.len()`.
    pub fn encode_slice(kind: Kind, x: &[f32], out: &mut [u8], r: Rounding, saturate: bool) {
        assert_eq!(x.len(), out.len(), "fp8::encode_slice: length mismatch");
        // Monomorphise on kind and saturate so the inner loop has constant specs.
        match (kind, saturate) {
            (Kind::E4M3, true) => map_rounding(x, out, 0, r, |v, ri| encode(Kind::E4M3, v, ri, true)),
            (Kind::E4M3, false) => map_rounding(x, out, 0, r, |v, ri| encode(Kind::E4M3, v, ri, false)),
            (Kind::E5M2, true) => map_rounding(x, out, 0, r, |v, ri| encode(Kind::E5M2, v, ri, true)),
            (Kind::E5M2, false) => map_rounding(x, out, 0, r, |v, ri| encode(Kind::E5M2, v, ri, false)),
        }
    }
}

multiversion! {
    /// Decode a slice (table lookup).
    ///
    /// # Panics
    /// If `codes.len() != out.len()`.
    pub fn decode_slice(kind: Kind, codes: &[u8], out: &mut [f32]) {
        assert_eq!(codes.len(), out.len(), "fp8::decode_slice: length mismatch");
        let t = lut(kind);
        for (o, &c) in out.iter_mut().zip(codes) {
            *o = f32::from_bits(t[usize::from(c)]);
        }
    }
}

multiversion! {
    /// Fake-quantize in place with saturation; NaN stays NaN
    /// (see [`crate::fake_quant`]).
    pub fn fake_quant_slice(kind: Kind, x: &mut [f32], r: Rounding) {
        match kind {
            Kind::E4M3 => map_rounding_inplace(x, 0, r, |v, ri| decode(Kind::E4M3, encode(Kind::E4M3, v, ri, true))),
            Kind::E5M2 => map_rounding_inplace(x, 0, r, |v, ri| decode(Kind::E5M2, encode(Kind::E5M2, v, ri, true))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spot_values() {
        use Kind::*;
        assert_eq!(decode(E4M3, 0x38), 1.0);
        assert_eq!(decode(E4M3, 0x7e), 448.0);
        assert_eq!(decode(E4M3, 0xfe), -448.0);
        assert_eq!(decode(E4M3, 0x01), 2f32.powi(-9));
        assert_eq!(decode(E4M3, 0x08), 2f32.powi(-6));
        assert!(decode(E4M3, 0x7f).is_nan());
        assert!(decode(E4M3, 0xff).is_nan());
        assert_eq!(decode(E4M3, 0x80).to_bits(), 0x8000_0000);
        assert_eq!(decode(E5M2, 0x3c), 1.0);
        assert_eq!(decode(E5M2, 0x7b), 57344.0);
        assert_eq!(decode(E5M2, 0x7c), f32::INFINITY);
        assert_eq!(decode(E5M2, 0xfc), f32::NEG_INFINITY);
        assert!(decode(E5M2, 0x7d).is_nan());
        assert_eq!(decode(E5M2, 0x01), 2f32.powi(-16));
        assert_eq!(decode(E5M2, 0x04), 2f32.powi(-14));
        assert_eq!(max_finite(E4M3), 448.0);
        assert_eq!(max_finite(E5M2), 57344.0);
        assert_eq!(encode(E4M3, 1.0, Rounding::NearestEven, false), 0x38);
        assert_eq!(encode(E5M2, -1.0, Rounding::NearestEven, false), 0xbc);
    }

    #[test]
    fn kind_metadata() {
        assert_eq!(Kind::E4M3.exponent_bits() + Kind::E4M3.mantissa_bits(), 7);
        assert_eq!(Kind::E5M2.exponent_bits() + Kind::E5M2.mantissa_bits(), 7);
        assert_eq!(Kind::E4M3.bias(), 7);
        assert_eq!(Kind::E5M2.bias(), 15);
        for k in [Kind::E4M3, Kind::E5M2] {
            assert_eq!(max_finite(k), 2f32.powi(k.emax()) * 1.75);
            assert!(decode(k, k.nan_code()).is_nan());
            assert!(k.is_nan(k.nan_code() | 0x80));
        }
    }
}
