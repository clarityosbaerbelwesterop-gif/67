//! `fake_quant` / `bits_per_element` contract tests across every format.

mod common;

use common::{assert_unbiased, Rng};
use forge_formats::fp8::Kind;
use forge_formats::{
    bf16, bits_per_element, fake_quant, fp16, fp4, fp8, mx, stochastic_bits, Format, Rounding,
};

fn sample(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| match i % 11 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => -f32::INFINITY,
            3 => -0.0,
            4 => 1e30,
            5 => f32::from_bits(rng.next_u32()),
            _ => (rng.normal() * 10f64.powi((i % 7) as i32 - 3)) as f32,
        })
        .collect()
}

fn elem_rounding(r: Rounding, i: usize) -> Rounding {
    match r {
        Rounding::Stochastic(s) => Rounding::Stochastic(stochastic_bits(s, i as u64)),
        o => o,
    }
}

#[test]
fn fake_quant_matches_per_format_codecs() {
    let x = sample(5000, 1);
    for r in [
        Rounding::NearestEven,
        Rounding::TowardZero,
        Rounding::Stochastic(3),
    ] {
        for f in Format::ALL {
            let mut y = x.clone();
            fake_quant(&mut y, f, r);
            for i in 0..x.len() {
                let ri = elem_rounding(r, i);
                let v = x[i];
                let want = match f {
                    Format::F32 => v,
                    Format::Bf16 => bf16::to_f32(bf16::encode(v, ri)),
                    Format::Fp16 => fp16::to_f32(fp16::encode(v, ri)),
                    Format::Fp8(k) => fp8::decode(k, fp8::encode(k, v, ri, true)),
                    Format::Fp4 => {
                        if v.is_nan() {
                            v
                        } else {
                            fp4::decode(fp4::encode(v, ri))
                        }
                    }
                    Format::Mx(_) => continue, // covered by tests/mx.rs
                };
                assert!(
                    want.to_bits() == y[i].to_bits() || (want.is_nan() && y[i].is_nan()),
                    "{f:?} {r:?} x={v:e}: got {} want {want}",
                    y[i]
                );
            }
        }
    }
}

#[test]
fn fake_quant_semantics_for_specials() {
    for f in Format::ALL {
        let mut v = vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 1e30, -0.0];
        fake_quant(&mut v, f, Rounding::NearestEven);
        assert!(v[0].is_nan(), "{f:?}: NaN stays NaN");
        match f {
            Format::F32 | Format::Bf16 | Format::Fp16 => {
                assert_eq!(v[1], f32::INFINITY, "{f:?}");
                assert_eq!(v[2], f32::NEG_INFINITY, "{f:?}");
            }
            Format::Fp8(k) => {
                assert_eq!(v[1], fp8::max_finite(k), "{f:?} saturates");
                assert_eq!(v[2], -fp8::max_finite(k));
                assert_eq!(v[3], fp8::max_finite(k));
            }
            Format::Fp4 => {
                assert_eq!(v[1], 6.0);
                assert_eq!(v[2], -6.0);
                assert_eq!(v[3], 6.0);
            }
            Format::Mx(_) => assert!(v.iter().all(|x| x.is_nan()), "block with NaN/Inf is NaN"),
        }
        if !matches!(f, Format::Mx(_)) {
            assert_eq!(v[4].to_bits(), 0x8000_0000, "{f:?}: -0 stays -0");
        }
    }
}

#[test]
fn fake_quant_is_idempotent_for_deterministic_rounding() {
    let x: Vec<f32> = sample(4096, 2)
        .into_iter()
        .map(|v| if v.is_finite() { v } else { 0.5 })
        .collect();
    for f in Format::ALL {
        for r in [Rounding::NearestEven, Rounding::TowardZero] {
            let mut once = x.clone();
            fake_quant(&mut once, f, r);
            let mut twice = once.clone();
            fake_quant(&mut twice, f, r);
            let a: Vec<u32> = once.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = twice.iter().map(|v| v.to_bits()).collect();
            assert!(a == b, "{f:?} {r:?}");
        }
    }
}

#[test]
fn fake_quant_stochastic_is_unbiased_for_every_format() {
    // 100k copies of one value: the per-index hash supplies independent bits.
    // MX is excluded: a constant block is its own amax, which MX saturates by
    // design (e.g. 1.7 -> 6.8 * 2^-2 -> 6 * 2^-2); tests/mx.rs covers MX with an
    // anchored amax.
    for f in Format::ALL
        .into_iter()
        .filter(|f| !matches!(f, Format::Mx(_)))
    {
        for x in [0.3f32, -1.7, 2.9] {
            let mut v = vec![x; 100_000];
            fake_quant(&mut v, f, Rounding::Stochastic(0xc0ffee));
            let draws: Vec<f64> = v.iter().map(|&d| d as f64).collect();
            assert_unbiased(f.name(), x as f64, &draws);
        }
    }
}

#[test]
fn bits_per_element_contract() {
    let want = [32.0, 16.0, 16.0, 8.0, 8.0, 4.0, 8.25, 8.25, 4.25, 8.25];
    for (f, w) in Format::ALL.into_iter().zip(want) {
        assert_eq!(bits_per_element(f), w, "{f:?}");
    }
    // MX amortisation matches the real storage of a full tensor.
    let x = vec![1.0f32; 32 * 64];
    for e in mx::Elem::ALL {
        let t = mx::quantize(&x, e, Rounding::NearestEven);
        assert_eq!(
            (t.bytes() * 8) as f32 / x.len() as f32,
            bits_per_element(Format::Mx(e))
        );
    }
    assert_eq!(Kind::E4M3, Kind::E4M3);
}
