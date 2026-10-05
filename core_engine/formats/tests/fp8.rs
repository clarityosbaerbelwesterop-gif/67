//! Exhaustive FP8 tests (OCP OFP8 rev 1.0) against an independent reference.

mod common;

use common::{assert_unbiased, next_down, next_up, Grid, RefFmt, RefOut, Rng, E4M3, E5M2};
use forge_formats::fp8::{self, Kind};
use forge_formats::Rounding;

const KINDS: [(Kind, RefFmt); 2] = [(Kind::E4M3, E4M3), (Kind::E5M2, E5M2)];
const DET: [Rounding; 2] = [Rounding::NearestEven, Rounding::TowardZero];

fn is_nan_code(kind: Kind, c: u8) -> bool {
    match kind {
        Kind::E4M3 => c & 0x7f == 0x7f,
        Kind::E5M2 => (c & 0x7f) >= 0x7d,
    }
}

/// Full reference encode (sign included) for finite x.
fn ref_encode(kind: Kind, f: RefFmt, g: &Grid, x: f32, r: Rounding, sat: bool) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let mag = match g.round(x.abs() as f64, r) {
        RefOut::Code(c) => c as u8,
        RefOut::Overflow if sat => f.max_code as u8,
        RefOut::Overflow => match kind {
            Kind::E4M3 => 0x7f,
            Kind::E5M2 => 0x7c,
        },
    };
    sign | mag
}

#[test]
fn decode_table_all_256_codes() {
    for (kind, f) in KINDS {
        let mut nan_codes = vec![];
        for c in 0..=255u8 {
            let v = fp8::decode(kind, c);
            let mag = u32::from(c & 0x7f);
            let neg = c & 0x80 != 0;
            assert_eq!(v.is_sign_negative(), neg, "{kind:?} {c:#04x} sign");
            if is_nan_code(kind, c) {
                assert!(v.is_nan(), "{kind:?} {c:#04x}");
                nan_codes.push(c);
            } else if kind == Kind::E5M2 && mag == 0x7c {
                assert!(v.is_infinite(), "{kind:?} {c:#04x}");
            } else {
                let want = if neg { -f.value(mag) } else { f.value(mag) };
                assert_eq!(v as f64, want, "{kind:?} {c:#04x}");
            }
        }
        match kind {
            Kind::E4M3 => assert_eq!(nan_codes, vec![0x7f, 0xff]),
            Kind::E5M2 => assert_eq!(nan_codes, vec![0x7d, 0x7e, 0x7f, 0xfd, 0xfe, 0xff]),
        }
    }
    // Literal spec values (OFP8 rev 1.0, table 2).
    let e4 = |c| fp8::decode(Kind::E4M3, c);
    let e5 = |c| fp8::decode(Kind::E5M2, c);
    assert_eq!(e4(0x7e), 448.0);
    assert_eq!(e4(0x08), 2f32.powi(-6)); // min normal
    assert_eq!(e4(0x07), 0.875 * 2f32.powi(-6)); // max subnormal
    assert_eq!(e4(0x01), 2f32.powi(-9)); // min subnormal
    assert_eq!(e5(0x7b), 57344.0);
    assert_eq!(e5(0x04), 2f32.powi(-14));
    assert_eq!(e5(0x03), 0.75 * 2f32.powi(-14));
    assert_eq!(e5(0x01), 2f32.powi(-16));
    assert_eq!(fp8::max_finite(Kind::E4M3), 448.0);
    assert_eq!(fp8::max_finite(Kind::E5M2), 57344.0);
}

#[test]
fn encode_decode_identity_for_every_non_nan_code() {
    for (kind, _) in KINDS {
        for c in 0..=255u8 {
            if is_nan_code(kind, c) {
                continue;
            }
            let v = fp8::decode(kind, c);
            let is_inf = v.is_infinite();
            let rmodes = [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(0), Rounding::Stochastic(u32::MAX), Rounding::Stochastic(0x8000_0000)];
            for r in rmodes {
                assert_eq!(fp8::encode(kind, v, r, false), c, "{kind:?} {c:#04x} {r:?}");
                if !is_inf {
                    assert_eq!(fp8::encode(kind, v, r, true), c, "{kind:?} {c:#04x} {r:?} sat");
                }
            }
        }
    }
}

#[test]
fn nan_decoded_codes_reencode_to_nan() {
    for (kind, _) in KINDS {
        for c in 0..=255u8 {
            if is_nan_code(kind, c) {
                for sat in [false, true] {
                    let e = fp8::encode(kind, fp8::decode(kind, c), Rounding::NearestEven, sat);
                    assert!(is_nan_code(kind, e));
                    assert_eq!(e & 0x80, c & 0x80, "NaN keeps sign");
                }
            }
        }
    }
}

#[test]
fn signed_zero() {
    for (kind, _) in KINDS {
        for sat in [false, true] {
            for r in [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(u32::MAX)] {
                assert_eq!(fp8::encode(kind, 0.0, r, sat), 0x00);
                assert_eq!(fp8::encode(kind, -0.0, r, sat), 0x80);
            }
        }
        assert_eq!(fp8::decode(kind, 0x00).to_bits(), 0x0000_0000);
        assert_eq!(fp8::decode(kind, 0x80).to_bits(), 0x8000_0000);
        // underflow keeps the sign
        assert_eq!(fp8::encode(kind, -1e-30, Rounding::NearestEven, false), 0x80);
        assert_eq!(fp8::encode(kind, 1e-30, Rounding::NearestEven, false), 0x00);
        assert_eq!(fp8::encode(kind, -f32::from_bits(1), Rounding::TowardZero, true), 0x80);
    }
}

#[test]
fn ties_to_even_exactly_on_midpoints() {
    for (kind, f) in KINDS {
        let g = f.grid();
        for c in 0..=f.max_code {
            let lo = g.vals[c as usize];
            let hi = if c == f.max_code { g.beyond } else { g.vals[c as usize + 1] };
            let mid = ((lo + hi) / 2.0) as f32;
            assert_eq!(mid as f64, (lo + hi) / 2.0, "midpoint exact in f32");
            for neg in [false, true] {
                let s = if neg { -1.0f32 } else { 1.0 };
                let sb = if neg { 0x80u8 } else { 0 };
                let enc = |x: f32, sat| fp8::encode(kind, s * x, Rounding::NearestEven, sat);
                let even = if c % 2 == 0 { c } else { c + 1 };
                if c < f.max_code {
                    assert_eq!(enc(mid, false), sb | even as u8, "{kind:?} mid of {c:#x}");
                    assert_eq!(enc(next_down(mid), false), sb | c as u8);
                    assert_eq!(enc(next_up(mid), false), sb | (c + 1) as u8);
                    // TowardZero always picks the lower neighbour inside the interval
                    assert_eq!(fp8::encode(kind, s * mid, Rounding::TowardZero, false), sb | c as u8);
                    assert_eq!(fp8::encode(kind, s * next_down(hi as f32), Rounding::TowardZero, false), sb | c as u8);
                } else {
                    // Midpoint between max and the first overflow value.
                    let below = enc(next_down(mid), false);
                    assert_eq!(below, sb | c as u8, "{kind:?} just below top midpoint");
                    let at = enc(mid, false);
                    let overflow = match kind {
                        Kind::E4M3 => 0x7f, // max 0x7e is even -> tie stays at max
                        Kind::E5M2 => 0x7c, // max 0x7b is odd -> tie overflows
                    };
                    let want = if kind == Kind::E4M3 { c as u8 } else { overflow };
                    assert_eq!(at, sb | want, "{kind:?} top midpoint");
                    assert_eq!(enc(next_up(mid), false), sb | overflow);
                    assert_eq!(enc(mid, true), sb | c as u8);
                    assert_eq!(enc(next_up(mid), true), sb | c as u8);
                }
            }
        }
    }
}

#[test]
fn dense_sweep_matches_reference_and_is_monotonic() {
    // Every 61st positive f32 bit pattern (~35M) for each kind, plus negatives
    // via symmetry check on a subset.
    for (kind, f) in KINDS {
        let g = f.grid();
        for r in DET {
            for sat in [false, true] {
                let mut prev = f32::NEG_INFINITY;
                let mut b = 0u32;
                let mut checked = 0u64;
                while b < 0x7f80_0000 {
                    let x = f32::from_bits(b);
                    let c = fp8::encode(kind, x, r, sat);
                    // monotone in x (NaN from overflow only at the top, treat as +inf)
                    let y = fp8::decode(kind, c);
                    let y = if y.is_nan() { f32::INFINITY } else { y };
                    assert!(y >= prev, "{kind:?} {r:?} not monotonic at {x:e}");
                    prev = y;
                    // reference check on a sub-sampled set (the reference is slow-ish)
                    if b % 7 == 0 {
                        assert_eq!(c, ref_encode(kind, f, &g, x, r, sat), "{kind:?} {r:?} sat={sat} x={x:e}");
                        assert_eq!(fp8::encode(kind, -x, r, sat), c | 0x80, "sign symmetry {x:e}");
                        checked += 1;
                    }
                    b += 61;
                }
                assert!(checked > 1_000_000);
            }
        }
    }
}

#[test]
fn overflow_saturate_vs_non_saturate() {
    let big = [500.0f32, 1e6, 3.0e38, f32::MAX, 65536.0, 61440.0, 1e5];
    for x in big {
        for s in [1.0f32, -1.0] {
            let sb = if s < 0.0 { 0x80 } else { 0 };
            let v = s * x;
            // E4M3
            if x > 464.0 {
                assert_eq!(fp8::encode(Kind::E4M3, v, Rounding::NearestEven, false), sb | 0x7f, "{v}");
                assert_eq!(fp8::encode(Kind::E4M3, v, Rounding::NearestEven, true), sb | 0x7e, "{v}");
                assert_eq!(fp8::encode(Kind::E4M3, v, Rounding::TowardZero, false), sb | 0x7e, "{v}");
                assert_eq!(fp8::encode(Kind::E4M3, v, Rounding::Stochastic(0), false), sb | 0x7f, "{v}");
                assert_eq!(fp8::encode(Kind::E4M3, v, Rounding::Stochastic(0), true), sb | 0x7e, "{v}");
            }
            // E5M2: overflow from 61440 (tie with odd max) upwards
            if x >= 61440.0 {
                assert_eq!(fp8::encode(Kind::E5M2, v, Rounding::NearestEven, false), sb | 0x7c, "{v}");
                assert_eq!(fp8::decode(Kind::E5M2, sb | 0x7c), s * f32::INFINITY);
                assert_eq!(fp8::encode(Kind::E5M2, v, Rounding::NearestEven, true), sb | 0x7b, "{v}");
                assert_eq!(fp8::encode(Kind::E5M2, v, Rounding::TowardZero, false), sb | 0x7b, "{v}");
            }
        }
    }
    // Just inside the range: no overflow.
    assert_eq!(fp8::encode(Kind::E4M3, 464.0, Rounding::NearestEven, false), 0x7e);
    assert_eq!(fp8::encode(Kind::E4M3, 449.0, Rounding::NearestEven, false), 0x7e);
    assert_eq!(fp8::encode(Kind::E5M2, 61439.0, Rounding::NearestEven, false), 0x7b);
    // Stochastic rounding between max and the overflow point: either max or overflow.
    for bits in [0u32, 1 << 31, u32::MAX] {
        let c = fp8::encode(Kind::E4M3, 470.0, Rounding::Stochastic(bits), false);
        assert!(c == 0x7e || c == 0x7f);
        let c = fp8::encode(Kind::E4M3, 470.0, Rounding::Stochastic(bits), true);
        assert_eq!(c, 0x7e);
    }
    assert_eq!(fp8::encode(Kind::E4M3, 470.0, Rounding::Stochastic(u32::MAX), false), 0x7f);
}

#[test]
fn nan_and_inf_inputs() {
    let mut rng = Rng::new(7);
    for _ in 0..10_000 {
        let payload = (rng.next_u32() & 0x007f_ffff).max(1);
        let sign = rng.next_u32() & 0x8000_0000;
        let x = f32::from_bits(sign | 0x7f80_0000 | payload);
        assert!(x.is_nan());
        for r in [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(rng.next_u32())] {
            for sat in [false, true] {
                let c4 = fp8::encode(Kind::E4M3, x, r, sat);
                assert_eq!(c4 & 0x7f, 0x7f);
                assert_eq!(u32::from(c4 & 0x80) << 24, sign);
                let c5 = fp8::encode(Kind::E5M2, x, r, sat);
                assert_eq!(c5 & 0x7f, 0x7e);
                assert_eq!(u32::from(c5 & 0x80) << 24, sign);
            }
        }
    }
    for r in [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(u32::MAX)] {
        assert_eq!(fp8::encode(Kind::E4M3, f32::INFINITY, r, false), 0x7f);
        assert_eq!(fp8::encode(Kind::E4M3, f32::NEG_INFINITY, r, false), 0xff);
        assert_eq!(fp8::encode(Kind::E4M3, f32::INFINITY, r, true), 0x7e);
        assert_eq!(fp8::encode(Kind::E4M3, f32::NEG_INFINITY, r, true), 0xfe);
        assert_eq!(fp8::encode(Kind::E5M2, f32::INFINITY, r, false), 0x7c);
        assert_eq!(fp8::encode(Kind::E5M2, f32::NEG_INFINITY, r, false), 0xfc);
        assert_eq!(fp8::encode(Kind::E5M2, f32::INFINITY, r, true), 0x7b);
        assert_eq!(fp8::encode(Kind::E5M2, f32::NEG_INFINITY, r, true), 0xfb);
    }
}

#[test]
fn stochastic_rounding_is_unbiased() {
    let mut rng = Rng::new(0xfeed);
    let n = 100_000;
    for (kind, f) in KINDS {
        let max = f.value(f.max_code) as f32;
        let xs = [
            1.3f32,
            -2.71,
            0.1,
            f.value(1) as f32 * 0.37, // below the smallest subnormal
            f.value(3) as f32 * 1.21, // subnormal range
            max * 0.97,               // top binade
            -max * 0.61,
        ];
        for x in xs {
            let draws: Vec<f64> = (0..n)
                .map(|_| fp8::decode(kind, fp8::encode(kind, x, Rounding::Stochastic(rng.next_u32()), true)) as f64)
                .collect();
            // every draw is one of the two neighbours
            let g = f.grid();
            let (lo, hi) = g.bracket((x as f64).abs());
            for &d in &draws {
                assert!(d.abs() == lo || d.abs() == hi, "{kind:?} {x}: draw {d} not in {{{lo}, {hi}}}");
            }
            assert_unbiased(&format!("{kind:?}"), x as f64, &draws);
        }
    }
}

#[test]
fn stochastic_round_up_probability_is_exact_for_short_shifts() {
    // For E4M3 normal values 20 bits are dropped, so with bits uniform the
    // round-up probability is exactly rem / 2^20: count over a full lattice.
    let x = 1.0f32 + 0.3 * 0.125; // between 1.0 and 1.125
    let rem = (x.to_bits() & 0xfffff) as u64;
    let mut ups = 0u64;
    let steps = 1u64 << 16;
    for k in 0..steps {
        let bits = (k << 16) as u32; // uniform lattice over the top 16 bits
        if fp8::encode(Kind::E4M3, x, Rounding::Stochastic(bits), true) == 0x39 {
            ups += 1;
        }
    }
    // The top 16 random bits decide when the low 4 rem bits are ignored:
    assert_eq!(ups, rem >> 4);
}

#[test]
fn slices_match_scalar() {
    let mut rng = Rng::new(3);
    let x: Vec<f32> = (0..10_007).map(|_| (rng.normal() * 100.0) as f32).collect();
    for (kind, _) in KINDS {
        for sat in [false, true] {
            for r in [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(42)] {
                let mut codes = vec![0u8; x.len()];
                fp8::encode_slice(kind, &x, &mut codes, r, sat);
                let mut y = vec![0f32; x.len()];
                fp8::decode_slice(kind, &codes, &mut y);
                for i in 0..x.len() {
                    let ri = match r {
                        Rounding::Stochastic(s) => Rounding::Stochastic(forge_formats::stochastic_bits(s, i as u64)),
                        o => o,
                    };
                    assert_eq!(codes[i], fp8::encode(kind, x[i], ri, sat));
                    assert_eq!(y[i].to_bits(), fp8::decode(kind, codes[i]).to_bits());
                }
            }
        }
    }
}
