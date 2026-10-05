//! forge-formats: bit-exact low-precision number formats. Contract: docs/DESIGN.md §2.
//!
//! Normative references: OCP 8-bit Floating Point Specification (OFP8)
//! rev 1.0, OCP Microscaling Formats (MX) v1.0, IEEE 754-2019.
//!
//! | format          | module  | layout            | specials                     | max finite |
//! |-----------------|---------|-------------------|------------------------------|-----------:|
//! | BF16            | [`bf16`]| s.8.7, bias 127   | IEEE Inf/NaN                 | ~3.39e38   |
//! | FP16 (binary16) | [`fp16`]| s.5.10, bias 15   | IEEE Inf/NaN, subnormals     | 65504      |
//! | FP8 E4M3        | [`fp8`] | s.4.3, bias 7     | NaN = S.1111.111, no Inf     | 448        |
//! | FP8 E5M2        | [`fp8`] | s.5.2, bias 15    | IEEE Inf/NaN                 | 57344      |
//! | FP4 E2M1        | [`fp4`] | s.2.1, bias 1     | none (saturating)            | 6          |
//! | MX blocks       | [`mx`]  | 32 elems + E8M0   | NaN scale 0xFF               |            |
//!
//! Everything is implemented with integer bit manipulation (no float
//! arithmetic on the rounding path), so results are identical on every
//! platform and independent of the FPU rounding mode.
//!
//! # Rounding
//!
//! Every encoder takes a [`Rounding`]. For scalar encoders,
//! `Rounding::Stochastic(bits)` uses `bits` directly: the 32 random bits are
//! added just below the kept mantissa LSB, then the value is truncated. For
//! slice encoders, [`mx::quantize`] and [`fake_quant`] the `u32` is a *seed*:
//! element `i` uses `stochastic_bits(seed, i)`, a counter-based hash, so
//! results are reproducible and independent of chunking/threading.
//!
//! Overflow follows IEEE 754 §7.4: rounding to nearest (or stochastically)
//! past the largest finite value overflows (to Inf, NaN, or `±max` when
//! saturating, depending on the format); rounding toward zero never turns a
//! finite input into Inf/NaN, it yields `±max`.

#![deny(missing_docs)]

mod engine;

pub mod bf16;
pub mod fp16;
pub mod fp4;
pub mod fp8;
pub mod mx;

pub use crate::engine::{active_isa, set_isa_limit, stochastic_bits, Isa};

/// Rounding mode for every encoder in this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// Round to nearest, ties to even (IEEE 754 default).
    NearestEven,
    /// Truncate the magnitude.
    TowardZero,
    /// Stochastic rounding. Carries 32 random bits; see the crate docs for how
    /// scalar and slice encoders use them.
    Stochastic(u32),
}

/// A storage format, used for fake quantization and cost accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// IEEE binary32 (identity).
    F32,
    /// bfloat16.
    Bf16,
    /// IEEE binary16.
    Fp16,
    /// OCP OFP8.
    Fp8(fp8::Kind),
    /// OCP MX FP4 E2M1 element format (no shared scale).
    Fp4,
    /// OCP MX block format (32 elements share one E8M0 scale).
    Mx(mx::Elem),
}

impl Format {
    /// Every format, in a stable order.
    pub const ALL: [Format; 10] = [
        Format::F32,
        Format::Bf16,
        Format::Fp16,
        Format::Fp8(fp8::Kind::E4M3),
        Format::Fp8(fp8::Kind::E5M2),
        Format::Fp4,
        Format::Mx(mx::Elem::Fp8E4M3),
        Format::Mx(mx::Elem::Fp8E5M2),
        Format::Mx(mx::Elem::Fp4E2M1),
        Format::Mx(mx::Elem::Int8),
    ];

    /// Stable lowercase name, e.g. `"fp8-e4m3"`, `"mx-fp4-e2m1"`.
    pub fn name(self) -> &'static str {
        match self {
            Format::F32 => "f32",
            Format::Bf16 => "bf16",
            Format::Fp16 => "fp16",
            Format::Fp8(fp8::Kind::E4M3) => "fp8-e4m3",
            Format::Fp8(fp8::Kind::E5M2) => "fp8-e5m2",
            Format::Fp4 => "fp4-e2m1",
            Format::Mx(mx::Elem::Fp8E4M3) => "mx-fp8-e4m3",
            Format::Mx(mx::Elem::Fp8E5M2) => "mx-fp8-e5m2",
            Format::Mx(mx::Elem::Fp4E2M1) => "mx-fp4-e2m1",
            Format::Mx(mx::Elem::Int8) => "mx-int8",
        }
    }

    /// Inverse of [`Format::name`].
    pub fn from_name(s: &str) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.name() == s)
    }

    /// Compact one-byte tag (index into [`Format::ALL`]), e.g. for bytecode.
    pub fn code(self) -> u8 {
        Format::ALL.iter().position(|&f| f == self).expect("listed") as u8
    }

    /// Inverse of [`Format::code`].
    pub fn from_code(c: u8) -> Option<Format> {
        Format::ALL.get(usize::from(c)).copied()
    }
}

/// Fake-quantize in place (quantize then dequantize): for low-precision
/// simulation and QAT.
///
/// * `Bf16` / `Fp16`: IEEE semantics, overflow to ±Inf, NaN stays NaN.
/// * `Fp8(_)`: saturating conversion (finite overflow and ±Inf clamp to
///   ±max), NaN stays NaN.
/// * `Fp4`: saturating; NaN stays NaN (FP4 cannot encode it, so the
///   simulation passes it through).
/// * `Mx(_)`: blocks of [`mx::BLOCK`] consecutive elements starting at
///   index 0; identical to `mx::dequantize(&mx::quantize(x, e, r))`.
///
/// For `Rounding::Stochastic(seed)`, element `i` uses
/// `stochastic_bits(seed, i)`.
pub fn fake_quant(x: &mut [f32], f: Format, r: Rounding) {
    match f {
        Format::F32 => {}
        Format::Bf16 => bf16::fake_quant_slice(x, r),
        Format::Fp16 => fp16::fake_quant_slice(x, r),
        Format::Fp8(kind) => fp8::fake_quant_slice(kind, x, r),
        Format::Fp4 => fp4::fake_quant_slice(x, r),
        Format::Mx(elem) => mx::fake_quant_slice(x, elem, r),
    }
}

/// Storage cost in bits per element, including MX scale amortisation
/// (one 8-bit scale per [`mx::BLOCK`] elements).
pub fn bits_per_element(f: Format) -> f32 {
    match f {
        Format::F32 => 32.0,
        Format::Bf16 | Format::Fp16 => 16.0,
        Format::Fp8(_) => 8.0,
        Format::Fp4 => 4.0,
        Format::Mx(e) => e.bits() as f32 + 8.0 / mx::BLOCK as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_per_element_table() {
        assert_eq!(bits_per_element(Format::F32), 32.0);
        assert_eq!(bits_per_element(Format::Bf16), 16.0);
        assert_eq!(bits_per_element(Format::Fp16), 16.0);
        assert_eq!(bits_per_element(Format::Fp8(fp8::Kind::E4M3)), 8.0);
        assert_eq!(bits_per_element(Format::Fp8(fp8::Kind::E5M2)), 8.0);
        assert_eq!(bits_per_element(Format::Fp4), 4.0);
        assert_eq!(bits_per_element(Format::Mx(mx::Elem::Fp8E4M3)), 8.25);
        assert_eq!(bits_per_element(Format::Mx(mx::Elem::Fp8E5M2)), 8.25);
        assert_eq!(bits_per_element(Format::Mx(mx::Elem::Int8)), 8.25);
        assert_eq!(bits_per_element(Format::Mx(mx::Elem::Fp4E2M1)), 4.25);
    }

    #[test]
    fn format_names_and_codes_roundtrip() {
        for (i, f) in Format::ALL.into_iter().enumerate() {
            assert_eq!(Format::from_name(f.name()), Some(f));
            assert_eq!(f.code() as usize, i);
            assert_eq!(Format::from_code(f.code()), Some(f));
        }
        assert_eq!(Format::from_name("fp7"), None);
        assert_eq!(Format::from_code(10), None);
    }

    #[test]
    fn fake_quant_f32_is_identity() {
        let mut v = vec![1.1f32, f32::NAN, -0.0, f32::INFINITY, 1e-45];
        let before: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
        fake_quant(&mut v, Format::F32, Rounding::NearestEven);
        let after: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
        assert_eq!(before, after);
    }
}
