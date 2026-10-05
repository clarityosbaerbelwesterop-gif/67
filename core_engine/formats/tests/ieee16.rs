//! BF16 and FP16 against an independent grid-search reference over millions
//! of sampled f32 bit patterns, every midpoint, and every code.

mod common;

use common::{assert_unbiased, next_down, next_up, Grid, RefFmt, RefOut, Rng, BF16, FP16};
use forge_formats::{bf16, fp16, Rounding};

struct Fmt {
    name: &'static str,
    f: RefFmt,
    encode: fn(f32, Rounding) -> u16,
    decode: fn(u16) -> f32,
    inf: u16,
}

const FMTS: [Fmt; 2] = [
    Fmt { name: "bf16", f: BF16, encode: bf16::encode, decode: bf16::to_f32, inf: 0x7f80 },
    Fmt { name: "fp16", f: FP16, encode: fp16::encode, decode: fp16::to_f32, inf: 0x7c00 },
];

fn reference(fm: &Fmt, g: &Grid, x: f32, r: Rounding) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    if x.is_infinite() {
        return sign | fm.inf;
    }
    sign | match g.round(x.abs() as f64, r) {
        RefOut::Code(c) => c as u16,
        RefOut::Overflow => fm.inf,
    }
}

fn check(fm: &Fmt, g: &Grid, b: u32) {
    let x = f32::from_bits(b);
    for r in [Rounding::NearestEven, Rounding::TowardZero] {
        let got = (fm.encode)(x, r);
        if x.is_nan() {
            assert!((fm.decode)(got).is_nan(), "{} NaN {b:#x}", fm.name);
            assert_eq!(got >> 15, (b >> 31) as u16, "{} NaN sign", fm.name);
            continue;
        }
        assert_eq!(got, reference(fm, g, x, r), "{} {r:?} x={x:e} ({b:#010x})", fm.name);
    }
}

#[test]
fn millions_of_sampled_bit_patterns() {
    let mut rng = Rng::new(0x5eed_1234);
    for fm in &FMTS {
        let g = fm.f.grid();
        // 3M uniformly random bit patterns (every exponent, NaN, Inf, subnormals)
        for _ in 0..3_000_000 {
            check(fm, &g, rng.next_u32());
        }
        // 2M patterns with the exponent inside / around the format's range
        let (lo_e, hi_e) = if fm.name == "fp16" { (95u32, 145u32) } else { (0, 255) };
        for _ in 0..2_000_000 {
            let e = lo_e + rng.next_u32() % (hi_e - lo_e);
            check(fm, &g, (rng.next_u32() & 0x807f_ffff) | (e << 23));
        }
        // dense stride over all positive patterns
        let mut b = 0u32;
        while b <= 0x7fff_ffff - 9973 {
            check(fm, &g, b);
            b += 9973;
        }
    }
}

#[test]
fn every_code_roundtrips() {
    for fm in &FMTS {
        for c in 0..=u16::MAX {
            let x = (fm.decode)(c);
            if x.is_nan() {
                continue;
            }
            let mag = u32::from(c & 0x7fff);
            if u16::try_from(mag).unwrap() < fm.inf {
                let v = fm.f.value(mag);
                assert_eq!(x.abs() as f64, v, "{} {c:#06x}", fm.name);
            }
            for r in [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(u32::MAX), Rounding::Stochastic(0)] {
                assert_eq!((fm.encode)(x, r), c, "{} {c:#06x} {r:?}", fm.name);
            }
        }
    }
}

#[test]
fn ties_to_even_on_every_midpoint() {
    for fm in &FMTS {
        let g = fm.f.grid();
        let max = fm.f.max_code;
        for c in 0..=max {
            let lo = g.vals[c as usize];
            let hi = if c == max { g.beyond } else { g.vals[c as usize + 1] };
            let mid64 = (lo + hi) / 2.0;
            let mid = mid64 as f32;
            if mid as f64 != mid64 {
                // bf16 midpoints between f32 subnormals are always exact; anything else is a bug
                panic!("{} midpoint not exact in f32", fm.name);
            }
            let even = if c % 2 == 0 { c } else { c + 1 };
            let want_mid = if even > max { fm.inf } else { even as u16 };
            for (s, sb) in [(1.0f32, 0u16), (-1.0, 0x8000)] {
                let e = |x: f32| (fm.encode)(s * x, Rounding::NearestEven);
                assert_eq!(e(mid), sb | want_mid, "{} mid above {c:#x}", fm.name);
                assert_eq!(e(next_down(mid)), sb | c as u16);
                let up = if c == max { fm.inf } else { c as u16 + 1 };
                assert_eq!(e(next_up(mid)), sb | up);
                // the fast RNE entry points agree
                let fast = if fm.name == "bf16" { bf16::from_f32(s * mid) } else { fp16::from_f32(s * mid) };
                assert_eq!(fast, sb | want_mid);
            }
        }
    }
}

#[test]
fn specials() {
    for fm in &FMTS {
        for r in [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(u32::MAX)] {
            assert_eq!((fm.encode)(f32::INFINITY, r), fm.inf);
            assert_eq!((fm.encode)(f32::NEG_INFINITY, r), 0x8000 | fm.inf);
            assert_eq!((fm.encode)(0.0, r), 0);
            assert_eq!((fm.encode)(-0.0, r), 0x8000);
            assert!((fm.decode)((fm.encode)(f32::NAN, r)).is_nan());
        }
        assert_eq!((fm.decode)(fm.inf), f32::INFINITY);
        assert_eq!((fm.decode)(0x8000 | fm.inf), f32::NEG_INFINITY);
        assert_eq!((fm.decode)(0x8000).to_bits(), 0x8000_0000);
        // TowardZero never overflows; NearestEven does at f32::MAX for fp16 & bf16
        assert_eq!((fm.encode)(f32::MAX, Rounding::TowardZero), fm.f.max_code as u16);
        assert_eq!((fm.encode)(f32::MAX, Rounding::NearestEven), fm.inf);
        assert_eq!((fm.encode)(-f32::MAX, Rounding::NearestEven), 0x8000 | fm.inf);
    }
    assert_eq!(bf16::NAN, 0x7fc0);
    assert!(bf16::to_f32(bf16::NAN).is_nan());
    assert!(fp16::to_f32(fp16::NAN).is_nan());
    assert_eq!(bf16::to_f32(bf16::MAX_FINITE), f32::from_bits(0x7f7f_0000));
    assert_eq!(fp16::to_f32(fp16::MAX_FINITE), 65504.0);
    assert_eq!(bf16::decode(0x3f80), 1.0);
    assert_eq!(fp16::decode(0x3c00), 1.0);
}

#[test]
fn monotonic_dense_sweep() {
    for fm in &FMTS {
        let mut prev = 0f32;
        let mut b = 0u32;
        while b < 0x7f80_0000 {
            let y = (fm.decode)((fm.encode)(f32::from_bits(b), Rounding::NearestEven));
            assert!(y >= prev, "{} at {b:#x}", fm.name);
            prev = y;
            b += 37;
        }
    }
}

#[test]
fn stochastic_rounding_is_unbiased() {
    let mut rng = Rng::new(31337);
    for fm in &FMTS {
        let xs: Vec<f32> = if fm.name == "fp16" {
            vec![1.000_3, -3.141_592_7, 1e-6, 6e-8, 65400.7, 0.1]
        } else {
            vec![1.001_3, -3.141_592_7, 1e-6, 1e-39, 3.0e38, 0.1]
        };
        for x in xs {
            let draws: Vec<f64> = (0..100_000)
                .map(|_| (fm.decode)((fm.encode)(x, Rounding::Stochastic(rng.next_u32()))) as f64)
                .collect();
            assert_unbiased(fm.name, x as f64, &draws);
        }
    }
}
