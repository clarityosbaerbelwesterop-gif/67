//! OCP Microscaling (MX) v1.0 block formats.
//!
//! A tensor is split into blocks of [`BLOCK`] = 32 consecutive elements (the
//! last block may be shorter). Each block stores one shared E8M0 scale
//! (`2^(code - 127)`, code 0xFF = NaN) and 32 narrow elements.
//!
//! Scale selection (MX v1.0 §6.3): `X = 2^(floor(log2(amax)) - emax_elem)`,
//! with `emax` = 8 (E4M3), 15 (E5M2), 2 (E2M1), 0 (INT8). The exponent is
//! clamped to the E8M0 range [-127, 127]. An all-zero block gets the smallest
//! scale (code 0, `2^-127`). A block containing NaN or ±Inf gets the NaN
//! scale and dequantizes to all-NaN.
//!
//! Elements are `round(x / X)` in the element format with saturation (values
//! of `x / X` in `[2^emax, 2^(emax+1))` can exceed the element maximum).
//! INT8 elements are two's complement integers with an implicit `2^-6`
//! scale, so the effective quantum is `2^(floor(log2 amax) - 6)`; they are
//! clamped symmetrically to [-127, 127] (-128 decodes to -2 if present).
//!
//! Division by `X` is done exactly on the exponent inside the integer
//! rounding engine (no f32 multiply, so no double rounding), and
//! dequantization multiplies by `X`, which is exact for every element value
//! and every scale.

use crate::engine::{
    fast_path_ok, floor_log2_bits, map_rounding, map_rounding_inplace, multiversion, pow2, round_magnitude_with, MiniFloat,
};
use crate::{fp4, fp8, Rounding};

/// Elements per shared scale.
pub const BLOCK: usize = 32;
/// The E8M0 NaN scale code.
pub const SCALE_NAN: u8 = 0xff;

/// MX element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    /// MXFP8 E4M3.
    Fp8E4M3,
    /// MXFP8 E5M2.
    Fp8E5M2,
    /// MXFP4 E2M1 (two elements per byte, low nibble first).
    Fp4E2M1,
    /// MXINT8: two's complement, implicit scale 2^-6.
    Int8,
}

const INT8: MiniFloat = MiniFloat { m_bits: 7, bias: 0, max_code: 127 };
const E4M3: MiniFloat = MiniFloat { m_bits: 3, bias: 7, max_code: 0x7e };
const E5M2: MiniFloat = MiniFloat { m_bits: 2, bias: 15, max_code: 0x7b };

impl Elem {
    /// All element types.
    pub const ALL: [Elem; 4] = [Elem::Fp8E4M3, Elem::Fp8E5M2, Elem::Fp4E2M1, Elem::Int8];

    /// Storage bits per element (excluding the shared scale).
    pub const fn bits(self) -> u32 {
        match self {
            Elem::Fp4E2M1 => 4,
            _ => 8,
        }
    }

    /// Exponent of the largest power of two representable by the element
    /// (8, 15, 2, 0); the scale is `2^(floor(log2 amax) - emax)`.
    pub const fn emax(self) -> i32 {
        match self {
            Elem::Fp8E4M3 => 8,
            Elem::Fp8E5M2 => 15,
            Elem::Fp4E2M1 => 2,
            Elem::Int8 => 0,
        }
    }

    /// Largest finite element value (448, 57344, 6, 127/64).
    pub fn max_finite(self) -> f32 {
        match self {
            Elem::Fp8E4M3 => fp8::max_finite(fp8::Kind::E4M3),
            Elem::Fp8E5M2 => fp8::max_finite(fp8::Kind::E5M2),
            Elem::Fp4E2M1 => fp4::MAX_FINITE,
            Elem::Int8 => 127.0 / 64.0,
        }
    }

    /// Bytes of element data for `n` elements.
    pub const fn data_len(self, n: usize) -> usize {
        match self {
            Elem::Fp4E2M1 => n.div_ceil(2),
            _ => n,
        }
    }

    #[inline(always)]
    const fn minifloat(self) -> MiniFloat {
        match self {
            Elem::Fp8E4M3 => E4M3,
            Elem::Fp8E5M2 => E5M2,
            Elem::Fp4E2M1 => fp4::E2M1,
            Elem::Int8 => INT8,
        }
    }
}

/// A quantized MX tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MxTensor {
    /// Element type.
    pub elem: Elem,
    /// Number of logical elements.
    pub len: usize,
    /// One E8M0 scale per block: `len.div_ceil(BLOCK)` bytes.
    pub scales: Vec<u8>,
    /// Element codes: `elem.data_len(len)` bytes. FP4: 2 per byte, low nibble first.
    pub data: Vec<u8>,
}

impl MxTensor {
    /// Total storage in bytes (scales + data).
    pub fn bytes(&self) -> usize {
        self.scales.len() + self.data.len()
    }

    /// Number of blocks.
    pub fn blocks(&self) -> usize {
        self.scales.len()
    }

    /// Raw element code of element `i` (FP4: the nibble).
    ///
    /// # Panics
    /// If `i >= self.len`.
    pub fn code(&self, i: usize) -> u8 {
        assert!(i < self.len, "index {i} out of range for MX tensor of length {}", self.len);
        match self.elem {
            Elem::Fp4E2M1 => (self.data[i / 2] >> (4 * (i & 1))) & 0xf,
            _ => self.data[i],
        }
    }
}

/// E8M0 scale code → value (`2^(c-127)`, 0xFF → NaN).
#[inline(always)]
pub fn e8m0_to_f32(c: u8) -> f32 {
    if c == SCALE_NAN {
        f32::NAN
    } else {
        pow2(i32::from(c) - 127)
    }
}

/// Shared scale code for a block whose largest magnitude has f32 bits
/// `amax_bits` (sign cleared).
#[inline(always)]
fn scale_code_from_amax_bits(amax_bits: u32, elem: Elem) -> u8 {
    if amax_bits >= 0x7f80_0000 {
        SCALE_NAN
    } else if amax_bits == 0 {
        0
    } else {
        ((floor_log2_bits(amax_bits) - elem.emax()).clamp(-127, 127) + 127) as u8
    }
}

#[inline(always)]
fn amax_bits(block: &[f32]) -> u32 {
    // Max over |x| bit patterns: integer order equals magnitude order for
    // non-negative floats, NaN patterns sort above Inf. Vectorises well.
    block.iter().fold(0u32, |m, &v| m.max(v.to_bits() & 0x7fff_ffff))
}

/// Shared E8M0 scale code that [`quantize`] chooses for one block.
pub fn scale_code(block: &[f32], elem: Elem) -> u8 {
    scale_code_from_amax_bits(amax_bits(block), elem)
}

/// Encode one finite element `x / 2^shift` (saturating). Returns the code as
/// stored (FP4 in the low nibble). Branch-free.
#[inline(always)]
fn encode_elem<const NORMALIZE: bool>(elem: Elem, x: f32, shift: i32, r: Rounding) -> u8 {
    let b = x.to_bits();
    let mf = elem.minifloat();
    let mag = round_magnitude_with::<NORMALIZE>(b, shift, mf, r).min(mf.max_code);
    match elem {
        Elem::Fp8E4M3 | Elem::Fp8E5M2 => (((b >> 24) & 0x80) | mag) as u8,
        Elem::Fp4E2M1 => (((b >> 28) & 0x8) | mag) as u8,
        Elem::Int8 => {
            // two's complement negate when the sign bit is set: (m ^ -s) + s
            let neg = (b >> 31) as i32;
            ((mag as i32 ^ -neg) + neg) as u8
        }
    }
}

/// Map every element of one finite block through `f(v, rounding)` choosing
/// the cheaper non-normalising engine path whenever it is exact for `shift`.
macro_rules! with_engine_path {
    ($elem:expr, $shift:expr, |$norm:ident| $body:expr) => {
        if fast_path_ok($shift, $elem.minifloat()) {
            const $norm: bool = false;
            $body
        } else {
            const $norm: bool = true;
            $body
        }
    };
}

const fn build_int8_lut() -> [u32; 256] {
    let mut lut = [0u32; 256];
    let mut c = 0usize;
    while c < 256 {
        // value = (c as i8) * 2^-6 ; exact in f32.
        let v = c as u8 as i8 as i32;
        let mag = v.unsigned_abs();
        let bits = if mag == 0 {
            0
        } else {
            let p = 31 - mag.leading_zeros();
            ((p + 127 - 6) << 23) | ((mag << (23 - p)) & 0x007f_ffff)
        };
        lut[c] = if v < 0 { bits | 0x8000_0000 } else { bits };
        c += 1;
    }
    lut
}

static LUT_INT8: [u32; 256] = build_int8_lut();

/// Decode one element code (unscaled).
#[inline(always)]
pub fn decode_elem(elem: Elem, code: u8) -> f32 {
    match elem {
        Elem::Fp8E4M3 => fp8::decode(fp8::Kind::E4M3, code),
        Elem::Fp8E5M2 => fp8::decode(fp8::Kind::E5M2, code),
        Elem::Fp4E2M1 => fp4::decode(code),
        Elem::Int8 => f32::from_bits(LUT_INT8[usize::from(code)]),
    }
}

/// Quantize `x` to MX blocks. For `Stochastic(seed)`, element `i` (global
/// index) uses [`crate::stochastic_bits`]`(seed, i)`.
pub fn quantize(x: &[f32], elem: Elem, r: Rounding) -> MxTensor {
    let mut t = MxTensor {
        elem,
        len: x.len(),
        scales: vec![0u8; x.len().div_ceil(BLOCK)],
        data: vec![0u8; elem.data_len(x.len())],
    };
    quantize_into(x, elem, r, &mut t.scales, &mut t.data);
    t
}

multiversion! {
    /// Quantize into caller-provided buffers: `scales.len() ==
    /// x.len().div_ceil(BLOCK)`, `data.len() == elem.data_len(x.len())`.
    /// Elements of a NaN-scale block are stored as 0.
    ///
    /// # Panics
    /// If the buffer lengths are wrong.
    pub fn quantize_into(x: &[f32], elem: Elem, r: Rounding, scales: &mut [u8], data: &mut [u8]) {
        assert_eq!(scales.len(), x.len().div_ceil(BLOCK), "mx::quantize_into: scales length");
        assert_eq!(data.len(), elem.data_len(x.len()), "mx::quantize_into: data length");
        // Monomorphise the per-block loop on the element type.
        macro_rules! go {
            ($e:expr) => {{
                let block_bytes = $e.data_len(BLOCK);
                let mut tmp = [0u8; BLOCK];
                let blocks = x.chunks(BLOCK).zip(scales.iter_mut().zip(data.chunks_mut(block_bytes)));
                for (bi, (block, (sc_out, d))) in blocks.enumerate() {
                    let sc = scale_code(block, $e);
                    *sc_out = sc;
                    let codes = &mut tmp[..block.len()];
                    if sc == SCALE_NAN {
                        codes.fill(0);
                    } else {
                        let shift = i32::from(sc) - 127;
                        let base = (bi * BLOCK) as u64;
                        with_engine_path!($e, shift, |NORM| map_rounding(block, codes, base, r, |v, ri| {
                            encode_elem::<NORM>($e, v, shift, ri)
                        }));
                    }
                    if $e == Elem::Fp4E2M1 {
                        fp4::pack_nibbles(codes, d);
                    } else {
                        d.copy_from_slice(codes);
                    }
                }
            }};
        }
        match elem {
            Elem::Fp8E4M3 => go!(Elem::Fp8E4M3),
            Elem::Fp8E5M2 => go!(Elem::Fp8E5M2),
            Elem::Fp4E2M1 => go!(Elem::Fp4E2M1),
            Elem::Int8 => go!(Elem::Int8),
        }
    }
}

/// Dequantize to a new vector of `t.len` values.
///
/// # Panics
/// If `t.scales` / `t.data` lengths do not match `t.len`.
pub fn dequantize(t: &MxTensor) -> Vec<f32> {
    let mut out = vec![0f32; t.len];
    dequantize_into(t, &mut out);
    out
}

/// Dequantize into `out` (`out.len() == t.len`).
///
/// # Panics
/// If `out.len() != t.len` or the tensor's buffers are malformed.
pub fn dequantize_into(t: &MxTensor, out: &mut [f32]) {
    assert_eq!(out.len(), t.len, "mx::dequantize_into: output length mismatch");
    assert_eq!(t.scales.len(), t.len.div_ceil(BLOCK), "mx: scales length does not match len");
    assert_eq!(t.data.len(), t.elem.data_len(t.len), "mx: data length does not match len");
    dequantize_raw(t.elem, &t.scales, &t.data, out);
}

multiversion! {
    /// Dequantize raw MX buffers (`scales`, `data` laid out as in
    /// [`MxTensor`]) for `out.len()` elements.
    ///
    /// # Panics
    /// If the buffer lengths do not match `out.len()`.
    pub fn dequantize_raw(elem: Elem, scales: &[u8], data: &[u8], out: &mut [f32]) {
        assert_eq!(scales.len(), out.len().div_ceil(BLOCK), "mx: scales length does not match len");
        assert_eq!(data.len(), elem.data_len(out.len()), "mx: data length does not match len");
        macro_rules! go {
            ($e:expr) => {{
                let blocks = out.chunks_mut(BLOCK).zip(scales.iter().zip(data.chunks($e.data_len(BLOCK))));
                for (o, (&sc, d)) in blocks {
                    if sc == SCALE_NAN {
                        o.fill(f32::NAN);
                        continue;
                    }
                    let s = e8m0_to_f32(sc);
                    if $e == Elem::Fp4E2M1 {
                        let mut pairs = o.chunks_exact_mut(2);
                        for (v, &byte) in (&mut pairs).zip(d) {
                            v[0] = fp4::decode(byte) * s;
                            v[1] = fp4::decode(byte >> 4) * s;
                        }
                        if let [last] = pairs.into_remainder() {
                            *last = fp4::decode(d[d.len() - 1]) * s;
                        }
                    } else {
                        for (v, &c) in o.iter_mut().zip(d) {
                            *v = decode_elem($e, c) * s;
                        }
                    }
                }
            }};
        }
        match elem {
            Elem::Fp8E4M3 => go!(Elem::Fp8E4M3),
            Elem::Fp8E5M2 => go!(Elem::Fp8E5M2),
            Elem::Fp4E2M1 => go!(Elem::Fp4E2M1),
            Elem::Int8 => go!(Elem::Int8),
        }
    }
}

multiversion! {
    /// In-place quantize→dequantize, bit-identical to
    /// `dequantize(&quantize(x, elem, r))` but without allocating.
    pub fn fake_quant_slice(x: &mut [f32], elem: Elem, r: Rounding) {
        macro_rules! go {
            ($e:expr) => {
                for (bi, block) in x.chunks_mut(BLOCK).enumerate() {
                    let sc = scale_code(block, $e);
                    if sc == SCALE_NAN {
                        block.fill(f32::NAN);
                        continue;
                    }
                    let shift = i32::from(sc) - 127;
                    let s = pow2(shift);
                    let base = (bi * BLOCK) as u64;
                    with_engine_path!($e, shift, |NORM| map_rounding_inplace(block, base, r, |v, ri| {
                        decode_elem($e, encode_elem::<NORM>($e, v, shift, ri)) * s
                    }));
                }
            };
        }
        match elem {
            Elem::Fp8E4M3 => go!(Elem::Fp8E4M3),
            Elem::Fp8E5M2 => go!(Elem::Fp8E5M2),
            Elem::Fp4E2M1 => go!(Elem::Fp4E2M1),
            Elem::Int8 => go!(Elem::Int8),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_lut() {
        for c in 0..=255u8 {
            assert_eq!(decode_elem(Elem::Int8, c), (c as i8) as f32 / 64.0);
        }
    }

    #[test]
    fn e8m0() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
        assert_eq!(e8m0_to_f32(254), 2f32.powi(127));
        assert!(e8m0_to_f32(255).is_nan());
    }

    #[test]
    fn scale_selection() {
        let mut b = [0f32; 32];
        b[3] = -5.0; // floor(log2 5) = 2
        assert_eq!(scale_code(&b, Elem::Fp8E4M3), (2 - 8 + 127) as u8);
        assert_eq!(scale_code(&b, Elem::Fp8E5M2), (2 - 15 + 127) as u8);
        assert_eq!(scale_code(&b, Elem::Fp4E2M1), 127);
        assert_eq!(scale_code(&b, Elem::Int8), 129);
        assert_eq!(scale_code(&[0.0; 32], Elem::Int8), 0);
        assert_eq!(scale_code(&[-0.0; 5], Elem::Fp4E2M1), 0);
        assert_eq!(scale_code(&[1.0, f32::NAN], Elem::Int8), SCALE_NAN);
        assert_eq!(scale_code(&[1.0, f32::NEG_INFINITY], Elem::Int8), SCALE_NAN);
        // clamping: tiny amax
        assert_eq!(scale_code(&[f32::from_bits(1)], Elem::Fp8E4M3), 0);
        assert_eq!(scale_code(&[f32::MAX], Elem::Int8), 254);
        assert_eq!(scale_code(&[f32::MAX], Elem::Fp4E2M1), 252);
    }

    #[test]
    fn simple_roundtrip() {
        let x: Vec<f32> = (0..70).map(|i| i as f32 * 0.25 - 8.0).collect();
        for e in Elem::ALL {
            let t = quantize(&x, e, Rounding::NearestEven);
            assert_eq!(t.blocks(), 3);
            assert_eq!(t.data.len(), e.data_len(70));
            assert_eq!(t.bytes(), 3 + e.data_len(70));
            let y = dequantize(&t);
            for (i, (&a, &b)) in x.iter().zip(&y).enumerate() {
                assert_eq!(decode_elem(e, t.code(i)) * e8m0_to_f32(t.scales[i / 32]), b);
                // multiples of 1/4 below 16 sit on the INT8 grid (quantum 1/8)
                if e == Elem::Int8 {
                    assert_eq!(a, b, "{e:?} {i}");
                }
            }
        }
    }
}
