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
//! 1. decompose the f32 into a 24-bit significand `m` (bit 23 = implicit one)
//!    and an unbiased exponent `e`;
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
//!
//! Everything is 32-bit lane arithmetic with selects instead of branches, so
//! the slice kernels (compiled per ISA level by [`multiversion!`]) vectorise.

use crate::Rounding;
use std::sync::atomic::{AtomicU8, Ordering};

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

/// Drop the low `d` bits of `m` (`m < 2^24`, `1 <= d <= 63`) with rounding
/// mode `r`.
///
/// The dropped part is first aligned to a 32-bit fraction `frac` (weight of
/// its MSB = one half of the kept LSB). Bits below 2^-32 are truncated; that
/// only happens when `d > 32`, where `m < 2^24` makes `frac < 2^24`, far from
/// any nearest-even decision.
///
/// `Stochastic(bits)` adds the 32 random bits to that fraction and keeps the
/// carry, i.e. it adds `bits` directly below the kept LSB and truncates. An
/// exactly representable input is never changed, and the round-up
/// probability equals the dropped fraction (truncated to a multiple of
/// 2^-32).
#[inline(always)]
pub(crate) fn round_shift(m: u32, d: u32, r: Rounding) -> u32 {
    debug_assert!((1..=63).contains(&d), "shift {d} out of range");
    debug_assert!(m < 1 << 24);
    // m < 2^24, so shifting by 24..=31 already yields 0.
    let q = m >> d.min(31);
    let frac = if d <= 32 {
        m.wrapping_shl(32 - d) // shift amount 0..=31; d == 32 keeps m
    } else {
        m >> (d - 32).min(31)
    };
    let up = match r {
        Rounding::TowardZero => false,
        // frac > 1/2, or frac == 1/2 with odd q.
        Rounding::NearestEven => (frac | (q & 1)) > 0x8000_0000,
        // carry out of frac + bits
        Rounding::Stochastic(bits) => frac > !bits,
    };
    q + u32::from(up)
}

/// Round the f32 magnitude whose bit pattern is `abs` (sign ignored),
/// divided by `2^exp_shift`, onto the grid of `f`. Returns the magnitude code
/// with an unbounded exponent (it can exceed `f.max_code`).
///
/// `NORMALIZE = false` skips normalising f32 subnormal inputs. That is exact
/// whenever every f32 subnormal lands in the *target's* subnormal range,
/// i.e. [`fast_path_ok`] (true for FP16/FP8/FP4 without a shift and for MX
/// blocks whose scale is not near the bottom of the E8M0 range).
///
/// Non-finite `abs` yields some code above `max_code` without panicking, so
/// callers may compute it unconditionally and select.
#[inline(always)]
pub(crate) fn round_magnitude_with<const NORMALIZE: bool>(
    abs: u32,
    exp_shift: i32,
    f: MiniFloat,
    r: Rounding,
) -> u32 {
    let abs = abs & 0x7fff_ffff;
    let ef = (abs >> 23) as i32;
    let m0 = (abs & 0x007f_ffff) | (u32::from(ef != 0) << 23);
    let (m, k) = if NORMALIZE {
        // Normalise f32 subnormals so that bit 23 is the leading one. Normal
        // numbers have exactly 8 leading zeros (k = 0); zero gets k = 24, m = 0.
        let k = m0.leading_zeros().saturating_sub(8);
        (m0 << k, k as i32)
    } else {
        (m0, 0)
    };
    let e = ef.max(1) - 127 - k - exp_shift;
    let emin = f.emin();
    let d = (23 - f.m_bits as i32 + (emin - e).max(0)).min(63) as u32;
    let field = (e.max(emin) + f.bias - 1) as u32;
    (field << f.m_bits) + round_shift(m, d, r)
}

/// General rounding (any input, any shift). See [`round_magnitude_with`].
#[cfg(test)]
#[inline(always)]
pub(crate) fn round_magnitude(abs: u32, exp_shift: i32, f: MiniFloat, r: Rounding) -> u32 {
    round_magnitude_with::<true>(abs, exp_shift, f, r)
}

/// Whether `round_magnitude_with::<false>` is exact for every input with
/// this shift: every f32 subnormal must be subnormal in the target too.
#[inline(always)]
pub(crate) const fn fast_path_ok(exp_shift: i32, f: MiniFloat) -> bool {
    -126 - exp_shift < f.emin()
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
    debug_assert!(-149 <= s && s <= 127);
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

#[inline(always)]
const fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^ (h >> 16)
}

/// Per-(seed, high index word) key of [`stochastic_bits`].
#[inline(always)]
const fn sr_key(seed: u32, hi: u32) -> u32 {
    fmix32(seed ^ fmix32(hi.wrapping_mul(0x27D4_EB2F) ^ 0x1656_67B1))
}

#[inline(always)]
const fn sr_bits(key: u32, lo: u32) -> u32 {
    fmix32(lo.wrapping_mul(0x9E37_79B9) ^ key)
}

/// Counter-based random bits for slice-level stochastic rounding: element
/// `index` of a slice rounded with `Rounding::Stochastic(seed)` uses
/// `stochastic_bits(seed, index)` as its 32 random bits.
///
/// A MurmurHash3 finaliser over a Weyl sequence keyed by the seed, in 32-bit
/// arithmetic so slice loops vectorise. Deterministic: results are
/// reproducible and independent of how a tensor is chunked.
#[inline(always)]
pub const fn stochastic_bits(seed: u32, index: u64) -> u32 {
    sr_bits(sr_key(seed, (index >> 32) as u32), index as u32)
}

/// Run `f(i, element_rounding)` for `i in 0..n`, with the rounding-mode match
/// hoisted out of the loop. For `Stochastic(seed)`, element `i` gets
/// `Stochastic(stochastic_bits(seed, base + i))`.
#[cfg(test)]
pub(crate) fn for_each_rounding(
    n: usize,
    base: u64,
    r: Rounding,
    mut f: impl FnMut(usize, Rounding),
) {
    let g = RoundGen::new(r, base, n);
    (0..n).for_each(|i| f(i, g.at(i)));
}

/// Per-element rounding source for a contiguous index range: hoists the
/// mode match and the seed key out of element loops.
#[derive(Clone, Copy)]
pub(crate) enum RoundGen {
    Fixed(Rounding),
    Keyed { key: u32, lo: u32 },
    Wide { seed: u32, base: u64 },
}

impl RoundGen {
    /// Rounding source for elements with global indices `base..base + n`.
    #[inline(always)]
    pub(crate) fn new(r: Rounding, base: u64, n: usize) -> RoundGen {
        match r {
            Rounding::Stochastic(seed) => {
                let last = base.wrapping_add(n as u64).wrapping_sub(1);
                if n == 0 || base >> 32 == last >> 32 {
                    RoundGen::Keyed {
                        key: sr_key(seed, (base >> 32) as u32),
                        lo: base as u32,
                    }
                } else {
                    RoundGen::Wide { seed, base }
                }
            }
            fixed => RoundGen::Fixed(fixed),
        }
    }

    /// Rounding for local element `i`.
    #[inline(always)]
    pub(crate) fn at(self, i: usize) -> Rounding {
        match self {
            RoundGen::Fixed(r) => r,
            RoundGen::Keyed { key, lo } => {
                Rounding::Stochastic(sr_bits(key, lo.wrapping_add(i as u32)))
            }
            RoundGen::Wide { seed, base } => {
                Rounding::Stochastic(stochastic_bits(seed, base.wrapping_add(i as u64)))
            }
        }
    }
}

/// `out[i] = f(x[i], rounding_i)` over zipped slices (no bounds checks in the
/// loop), with the rounding-mode dispatch hoisted out of the loop.
/// Element `i` has global index `base + i` for stochastic rounding.
#[inline(always)]
pub(crate) fn map_rounding<T>(
    x: &[f32],
    out: &mut [T],
    base: u64,
    r: Rounding,
    f: impl Fn(f32, Rounding) -> T,
) {
    debug_assert_eq!(x.len(), out.len());
    match RoundGen::new(r, base, x.len()) {
        RoundGen::Fixed(Rounding::NearestEven) => {
            out.iter_mut()
                .zip(x)
                .for_each(|(o, &v)| *o = f(v, Rounding::NearestEven));
        }
        RoundGen::Fixed(Rounding::TowardZero) => {
            out.iter_mut()
                .zip(x)
                .for_each(|(o, &v)| *o = f(v, Rounding::TowardZero));
        }
        RoundGen::Keyed { key, lo } => {
            out.iter_mut().zip(x).enumerate().for_each(|(i, (o, &v))| {
                *o = f(
                    v,
                    Rounding::Stochastic(sr_bits(key, lo.wrapping_add(i as u32))),
                );
            })
        }
        g => out
            .iter_mut()
            .zip(x)
            .enumerate()
            .for_each(|(i, (o, &v))| *o = f(v, g.at(i))),
    }
}

/// In-place variant of [`map_rounding`].
#[inline(always)]
pub(crate) fn map_rounding_inplace(
    x: &mut [f32],
    base: u64,
    r: Rounding,
    f: impl Fn(f32, Rounding) -> f32,
) {
    match RoundGen::new(r, base, x.len()) {
        RoundGen::Fixed(Rounding::NearestEven) => {
            x.iter_mut().for_each(|v| *v = f(*v, Rounding::NearestEven))
        }
        RoundGen::Fixed(Rounding::TowardZero) => {
            x.iter_mut().for_each(|v| *v = f(*v, Rounding::TowardZero))
        }
        RoundGen::Keyed { key, lo } => x.iter_mut().enumerate().for_each(|(i, v)| {
            *v = f(
                *v,
                Rounding::Stochastic(sr_bits(key, lo.wrapping_add(i as u32))),
            );
        }),
        g => x
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = f(*v, g.at(i))),
    }
}

/// Define a slice kernel compiled three times (baseline, x86-64-v3 AVX2,
/// AVX-512) and dispatched at run time on the CPU's features. The body is
/// ordinary safe code; only the call into a feature-gated copy is `unsafe`,
/// justified by the runtime feature check right before it.
macro_rules! multiversion {
    ($(#[$meta:meta])* $vis:vis fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $body:block) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx2,fma,bmi1,bmi2,lzcnt")]
                fn v4($($arg: $ty),*) $body
                #[target_feature(enable = "avx2,fma,bmi1,bmi2,lzcnt")]
                fn v3($($arg: $ty),*) $body
                if $crate::engine::has_avx512() {
                    // SAFETY: the CPU supports every feature enabled on `v4`.
                    return unsafe { v4($($arg),*) };
                }
                if $crate::engine::has_avx2() {
                    // SAFETY: the CPU supports every feature enabled on `v3`.
                    return unsafe { v3($($arg),*) };
                }
            }
            #[inline(always)]
            fn base($($arg: $ty),*) $body
            base($($arg),*)
        }
    };
}
pub(crate) use multiversion;

/// Instruction-set level of the slice kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Isa {
    /// Portable code for the compilation target's baseline.
    Baseline = 0,
    /// x86-64-v3: AVX2, FMA, BMI1/2, LZCNT.
    Avx2 = 1,
    /// AVX-512 F/BW/VL/DQ/CD (on top of AVX2).
    Avx512 = 2,
}

static ISA_LIMIT: AtomicU8 = AtomicU8::new(Isa::Avx512 as u8);

/// Cap the instruction set the slice kernels may use (default: the best the
/// CPU supports). Every level produces bit-identical results; this exists
/// for testing and benchmarking. Process-global.
pub fn set_isa_limit(limit: Isa) {
    ISA_LIMIT.store(limit as u8, Ordering::Relaxed);
}

/// The instruction set the slice kernels currently dispatch to.
pub fn active_isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            return Isa::Avx512;
        }
        if has_avx2() {
            return Isa::Avx2;
        }
    }
    Isa::Baseline
}

/// AVX-512 (F/BW/VL/DQ/CD) plus everything [`has_avx2`] needs.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn has_avx512() -> bool {
    ISA_LIMIT.load(Ordering::Relaxed) >= Isa::Avx512 as u8
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && has_avx2()
}

/// x86-64-v3 subset used by the AVX2 kernels.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn has_avx2() -> bool {
    ISA_LIMIT.load(Ordering::Relaxed) >= Isa::Avx2 as u8
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("lzcnt")
}

#[cfg(test)]
mod tests {
    use super::*;

    const E4M3: MiniFloat = MiniFloat {
        m_bits: 3,
        bias: 7,
        max_code: 0x7e,
    };
    const FMTS: [MiniFloat; 5] = [
        E4M3,
        MiniFloat {
            m_bits: 2,
            bias: 15,
            max_code: 0x7b,
        },
        MiniFloat {
            m_bits: 1,
            bias: 1,
            max_code: 7,
        },
        MiniFloat {
            m_bits: 7,
            bias: 0,
            max_code: 127,
        },
        MiniFloat {
            m_bits: 10,
            bias: 15,
            max_code: 0x7bff,
        },
    ];

    /// Straightforward 64-bit reference of `round_shift`.
    fn round_shift_ref(m: u64, d: u32, r: Rounding) -> u64 {
        let q = m >> d;
        match r {
            Rounding::TowardZero => q,
            Rounding::NearestEven => {
                let half = 1u64 << (d - 1);
                let rem = m & ((1u64 << d) - 1);
                q + u64::from(rem > half || (rem == half && q & 1 == 1))
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

    #[test]
    fn round_shift_matches_64bit_reference() {
        let mut s = 0x0123_4567_89ab_cdefu64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for d in 1..=63u32 {
            let t = d.min(24);
            for _ in 0..20_000 {
                let v = next();
                let m = match v % 4 {
                    0 => (v >> 8) as u32 & 0x00ff_ffff,
                    // exact tie below the kept LSB (for d <= 24)
                    1 => (((v >> 8) as u32 & 0x00ff_ffff) & !((1u32 << t) - 1)) | (1u32 << (t - 1)),
                    2 => 0x00ff_ffff,
                    _ => (v >> 40) as u32 & 0xff,
                };
                let bits = (v >> 3) as u32;
                for r in [
                    Rounding::NearestEven,
                    Rounding::TowardZero,
                    Rounding::Stochastic(bits),
                    Rounding::Stochastic(u32::MAX),
                    Rounding::Stochastic(0),
                ] {
                    assert_eq!(
                        u64::from(round_shift(m, d, r)),
                        round_shift_ref(u64::from(m), d, r),
                        "m={m:#x} d={d} {r:?}"
                    );
                }
            }
        }
    }

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
        // d > 32: the dropped fraction is resolved to 2^-32 only
        assert_eq!(round_shift(1, 40, Rounding::Stochastic(u32::MAX)), 0);
        assert_eq!(round_shift(1 << 8, 40, Rounding::Stochastic(u32::MAX)), 1);
        assert_eq!(
            round_shift(1 << 8, 40, Rounding::Stochastic(u32::MAX - 1)),
            0
        );
        // fraction 0.75: rounds up iff bits >= 2^30
        assert_eq!(round_shift(3 << 22, 24, Rounding::Stochastic(1 << 30)), 1);
        assert_eq!(
            round_shift(3 << 22, 24, Rounding::Stochastic((1 << 30) - 1)),
            0
        );
        assert_eq!(round_shift(0, 63, Rounding::Stochastic(u32::MAX)), 0);
        assert_eq!(round_shift((1 << 24) - 1, 63, Rounding::NearestEven), 0);
        assert_eq!(round_shift((1 << 24) - 1, 32, Rounding::NearestEven), 0);
        assert_eq!(round_shift(1 << 23, 24, Rounding::NearestEven), 0); // exact half, even
        assert_eq!(round_shift((1 << 23) + 1, 24, Rounding::NearestEven), 1);
    }

    #[test]
    fn fast_path_agrees_when_allowed() {
        let mut b = 0u32;
        let mut compared = 0u64;
        while b < 0x7f80_0000 {
            for f in FMTS {
                for shift in [-127, -120, -100, -10, 0, 10, 100, 127] {
                    if fast_path_ok(shift, f) {
                        for r in [
                            Rounding::NearestEven,
                            Rounding::TowardZero,
                            Rounding::Stochastic(b.rotate_left(7)),
                        ] {
                            assert_eq!(
                                round_magnitude_with::<false>(b, shift, f, r),
                                round_magnitude(b, shift, f, r),
                                "{b:#x} {shift} {f:?}"
                            );
                            compared += 1;
                        }
                    }
                }
            }
            b += if b < 0x0100_0000 { 997 } else { 99_991 };
        }
        assert!(compared > 100_000);
        assert!(fast_path_ok(0, E4M3));
        assert!(!fast_path_ok(-127, E4M3));
        assert!(!fast_path_ok(
            0,
            MiniFloat {
                m_bits: 7,
                bias: 127,
                max_code: 0x7f7f
            }
        ));
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
        assert_eq!(
            round_magnitude(x.to_bits(), -100, E4M3, Rounding::NearestEven),
            0x44
        );
        // f32 subnormal scaled into the normal range needs normalisation
        let tiny = f32::from_bits(0x0000_0300); // 3 * 2^-141
        assert_eq!(
            round_magnitude(tiny.to_bits(), -141, E4M3, Rounding::NearestEven),
            0x44
        );
        // non-finite inputs give an out-of-range code instead of panicking
        for f in FMTS {
            assert!(round_magnitude(0x7f80_0000, 0, f, Rounding::NearestEven) > f.max_code);
            assert!(
                round_magnitude_with::<false>(0x7fff_ffff, 0, f, Rounding::Stochastic(u32::MAX))
                    > f.max_code
            );
        }
    }

    #[test]
    fn decode_magnitude() {
        assert_eq!(f32::from_bits(decode_magnitude_bits(0x7e, E4M3)), 448.0);
        assert_eq!(
            f32::from_bits(decode_magnitude_bits(0x01, E4M3)),
            2f32.powi(-9)
        );
        assert_eq!(
            f32::from_bits(decode_magnitude_bits(0x07, E4M3)),
            7.0 * 2f32.powi(-9)
        );
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
        // Not a statistical proof, just a sanity check against gross bias,
        // for several seeds and for both index words.
        let n = 1u64 << 16;
        for (seed, base) in [
            (12345u32, 0u64),
            (0, 0),
            (u32::MAX, 1 << 32),
            (7, (1 << 32) - 100),
        ] {
            let mut ones = [0u32; 32];
            for i in 0..n {
                let b = stochastic_bits(seed, base + i);
                for (k, o) in ones.iter_mut().enumerate() {
                    *o += (b >> k) & 1;
                }
            }
            for o in ones {
                let p = o as f64 / n as f64;
                assert!((p - 0.5).abs() < 0.02, "bit frequency {p}");
            }
        }
        assert_ne!(stochastic_bits(1, 0), stochastic_bits(2, 0));
        assert_ne!(stochastic_bits(1, 0), stochastic_bits(1, 1));
        assert_ne!(stochastic_bits(1, 0), stochastic_bits(1, 1 << 32));
    }

    #[test]
    fn for_each_rounding_uses_global_indices() {
        for base in [0u64, 5, (1 << 32) - 3, 1 << 33] {
            let mut got = vec![];
            for_each_rounding(7, base, Rounding::Stochastic(99), |i, r| got.push((i, r)));
            for (i, r) in got {
                assert_eq!(
                    r,
                    Rounding::Stochastic(stochastic_bits(99, base + i as u64)),
                    "base {base} i {i}"
                );
            }
        }
        let mut n = 0;
        for_each_rounding(3, 0, Rounding::TowardZero, |_, r| {
            assert_eq!(r, Rounding::TowardZero);
            n += 1;
        });
        assert_eq!(n, 3);
    }
}
