//! MX block format tests (OCP MX v1.0) against an f64 reference.

mod common;

use common::{assert_unbiased, Grid, RefOut, Rng, E2M1, E4M3, E5M2};
use forge_formats::mx::{self, Elem, MxTensor, BLOCK};
use forge_formats::{fake_quant, Format, Rounding};

fn emax(e: Elem) -> i32 {
    match e {
        Elem::Fp8E4M3 => 8,
        Elem::Fp8E5M2 => 15,
        Elem::Fp4E2M1 => 2,
        Elem::Int8 => 0,
    }
}

fn grid(e: Elem) -> Grid {
    match e {
        Elem::Fp8E4M3 => E4M3.grid(),
        Elem::Fp8E5M2 => E5M2.grid(),
        Elem::Fp4E2M1 => E2M1.grid(),
        Elem::Int8 => Grid::new((0..=127).map(|c| c as f64 / 64.0).collect()),
    }
}

/// floor(log2(a)) for a > 0, by search (no bit tricks).
fn floor_log2(a: f64) -> i32 {
    let mut k = -160;
    while 2f64.powi(k + 1) <= a {
        k += 1;
    }
    k
}

/// Reference scale exponent; None for a NaN scale.
fn ref_scale_exp(block: &[f32], e: Elem) -> Option<i32> {
    if block.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let amax = block.iter().fold(0f64, |m, &v| m.max((v as f64).abs()));
    if amax == 0.0 {
        return Some(-127);
    }
    Some((floor_log2(amax) - emax(e)).clamp(-127, 127))
}

/// Reference dequantized values for deterministic rounding.
fn ref_fake_quant(x: &[f32], e: Elem, r: Rounding) -> Vec<f64> {
    let g = grid(e);
    let mut out = Vec::with_capacity(x.len());
    for block in x.chunks(BLOCK) {
        match ref_scale_exp(block, e) {
            None => out.extend(block.iter().map(|_| f64::NAN)),
            Some(s) => {
                let sc = 2f64.powi(s);
                for &v in block {
                    let y = (v as f64).abs() / sc;
                    let mag = match g.round(y, r) {
                        RefOut::Code(c) => g.vals[c as usize],
                        RefOut::Overflow => *g.vals.last().unwrap(),
                    };
                    out.push(if v < 0.0 { -mag * sc } else { mag * sc });
                }
            }
        }
    }
    out
}

/// Worst-case |x - deq(x)| / amax(block) when the scale is not clamped.
/// Nearest: max(half an ulp of the top element binade, saturation loss
/// `1 - max_elem / 2^(emax+1)`), each relative to amax >= 2^emax * X.
/// Toward zero: a full ulp instead of half.
fn bound(e: Elem, r: Rounding) -> f64 {
    let ne = match e {
        Elem::Fp8E4M3 | Elem::Fp8E5M2 => 1.0 / 8.0, // E4M3: max(16/256, 64/512); E5M2: max(4096/32768, 8192/65536)
        Elem::Fp4E2M1 => 1.0 / 4.0,                 // max(1/4, 2/8)
        Elem::Int8 => 1.0 / 128.0,                  // max(2^-7, 1 - 127/128)
    };
    match (r, e) {
        (Rounding::NearestEven, _) => ne,
        (_, Elem::Fp8E4M3) => 1.0 / 8.0, // full ulp 32/256 equals the saturation loss
        (_, Elem::Fp8E5M2) => 1.0 / 4.0,
        (_, Elem::Fp4E2M1) => 1.0 / 2.0,
        (_, Elem::Int8) => 1.0 / 64.0,
    }
}

fn datasets() -> Vec<(&'static str, Vec<f32>)> {
    let mut rng = Rng::new(0xabcdef);
    let n = 32 * 512 + 17;
    let mut sets = vec![];
    sets.push(("normal", (0..n).map(|_| rng.normal() as f32).collect()));
    sets.push(("uniform", (0..n).map(|_| (rng.uniform() * 2.0 - 1.0) as f32 * 1e3).collect()));
    sets.push(("lognormal", (0..n).map(|_| ((rng.normal() * 4.0).exp() * if rng.uniform() < 0.5 { -1.0 } else { 1.0 }) as f32).collect()));
    sets.push((
        "outliers",
        (0..n).map(|i| if i % 97 == 0 { (rng.normal() * 1e4) as f32 } else { (rng.normal() * 1e-2) as f32 }).collect(),
    ));
    sets.push(("tiny", (0..n).map(|_| (rng.normal() * 1e-30) as f32).collect()));
    sets.push(("f32-subnormal", (0..n).map(|_| (rng.normal() * 1e-39) as f32).collect()));
    sets.push(("huge", (0..n).map(|_| (rng.normal() * 1e37) as f32).collect()));
    // amax exactly a power of two, and just below one (scale boundaries)
    sets.push((
        "pow2-edges",
        (0..n)
            .map(|i| {
                let k = (i / 32 % 40) as i32 - 20;
                match i % 32 {
                    0 if (i / 32) % 2 == 0 => 2f32.powi(k),
                    0 => f32::from_bits(2f32.powi(k).to_bits() - 1),
                    _ => (rng.uniform() * 2.0 - 1.0) as f32 * 2f32.powi(k),
                }
            })
            .collect(),
    ));
    // values that sit exactly on scaled grid midpoints
    sets.push((
        "grid-mids",
        (0..n).map(|i| if i % 32 == 0 { 7.9 } else { (((i * 7919) % 255) as f32 - 127.0) / 16.0 + 1.0 / 64.0 }).collect(),
    ));
    sets
}

#[test]
fn matches_reference_and_error_bounds_hold() {
    for (name, x) in datasets() {
        for e in Elem::ALL {
            for r in [Rounding::NearestEven, Rounding::TowardZero] {
                let t = mx::quantize(&x, e, r);
                let y = mx::dequantize(&t);
                let want = ref_fake_quant(&x, e, r);
                assert_eq!(t.scales.len(), x.len().div_ceil(32));
                for (bi, block) in x.chunks(BLOCK).enumerate() {
                    let s = ref_scale_exp(block, e).unwrap();
                    assert_eq!(i32::from(t.scales[bi]) - 127, s, "{name} {e:?} block {bi} scale");
                    let amax = block.iter().fold(0f64, |m, &v| m.max((v as f64).abs()));
                    let clamped = amax > 0.0 && floor_log2(amax) - emax(e) < -127;
                    for (j, &v) in block.iter().enumerate() {
                        let i = bi * BLOCK + j;
                        assert_eq!(y[i] as f64, want[i], "{name} {e:?} {r:?} i={i} x={v:e}");
                        let err = (v as f64 - y[i] as f64).abs();
                        if clamped {
                            assert!(err <= amax, "{name} {e:?} clamped block err {err} amax {amax}");
                        } else {
                            assert!(err <= bound(e, r) * amax, "{name} {e:?} {r:?} i={i} x={v:e} y={} err/amax={}", y[i], err / amax);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn stochastic_error_bounds_hold() {
    for (name, x) in datasets() {
        for e in Elem::ALL {
            let y = mx::dequantize(&mx::quantize(&x, e, Rounding::Stochastic(77)));
            for (block, yb) in x.chunks(BLOCK).zip(y.chunks(BLOCK)) {
                let amax = block.iter().fold(0f64, |m, &v| m.max((v as f64).abs()));
                if amax == 0.0 || floor_log2(amax) - emax(e) < -127 {
                    continue;
                }
                for (&v, &w) in block.iter().zip(yb) {
                    let err = (v as f64 - w as f64).abs();
                    assert!(err <= bound(e, Rounding::Stochastic(0)) * amax, "{name} {e:?} x={v:e} y={w:e}");
                }
            }
        }
    }
}

#[test]
fn error_bounds_are_reached_not_loose() {
    // Saturation case: amax just below 2^(emax+1) * X shows the worst case.
    for e in Elem::ALL {
        let mut x = vec![0f32; 32];
        x[0] = f32::from_bits(2f32.to_bits() - 1); // 1.99999988 -> floor(log2) = 0
        let y = mx::dequantize(&mx::quantize(&x, e, Rounding::NearestEven));
        let rel = ((x[0] - y[0]) / x[0]) as f64;
        assert!(rel <= bound(e, Rounding::NearestEven) && rel > bound(e, Rounding::NearestEven) * 0.95, "{e:?} rel {rel}");
    }
}

#[test]
fn zero_nan_inf_blocks() {
    for e in Elem::ALL {
        let mut x = vec![0f32; 96];
        x[33] = f32::NAN;
        x[70] = f32::NEG_INFINITY;
        x[71] = 1.0;
        let t = mx::quantize(&x, e, Rounding::NearestEven);
        assert_eq!(t.scales, vec![0u8, mx::SCALE_NAN, mx::SCALE_NAN]);
        assert_eq!(mx::e8m0_to_f32(t.scales[0]), 2f32.powi(-127), "all-zero block gets the smallest scale");
        assert!(t.data[..e.data_len(32)].iter().all(|&b| b == 0));
        let y = mx::dequantize(&t);
        assert!(y[..32].iter().all(|&v| v == 0.0));
        assert!(y[32..].iter().all(|v| v.is_nan()));
        let mut z = x.clone();
        fake_quant(&mut z, Format::Mx(e), Rounding::NearestEven);
        assert!(z[..32].iter().all(|&v| v == 0.0));
        assert!(z[32..].iter().all(|v| v.is_nan()));
    }
}

#[test]
fn partial_blocks_and_lengths() {
    let mut rng = Rng::new(17);
    for n in [0usize, 1, 2, 31, 32, 33, 63, 64, 65, 100] {
        let x: Vec<f32> = (0..n).map(|_| rng.normal() as f32).collect();
        for e in Elem::ALL {
            let t = mx::quantize(&x, e, Rounding::NearestEven);
            assert_eq!(t.len, n);
            assert_eq!(t.scales.len(), n.div_ceil(32));
            assert_eq!(t.data.len(), e.data_len(n));
            assert_eq!(t.bytes(), t.scales.len() + t.data.len());
            let y = mx::dequantize(&t);
            let want = ref_fake_quant(&x, e, Rounding::NearestEven);
            for i in 0..n {
                assert_eq!(y[i] as f64, want[i]);
                assert_eq!(mx::decode_elem(e, t.code(i)) * mx::e8m0_to_f32(t.scales[i / 32]), y[i]);
            }
            if e == Elem::Fp4E2M1 && n % 2 == 1 {
                assert_eq!(t.data[n / 2] >> 4, 0);
            }
        }
    }
}

#[test]
fn int8_is_symmetric_twos_complement() {
    let x: Vec<f32> = (0..32).map(|i| -1.999 + i as f32 * 0.05).collect();
    let t = mx::quantize(&x, Elem::Int8, Rounding::NearestEven);
    assert_eq!(mx::e8m0_to_f32(t.scales[0]), 1.0); // amax 1.999 -> floor(log2) = 0
    assert!(t.data.iter().all(|&c| c != 0x80), "-128 is never produced");
    assert_eq!(t.data[0] as i8, -127); // -1.999 * 64 = -127.9 saturates to -127
    for (i, &v) in x.iter().enumerate() {
        let q = t.data[i] as i8;
        assert_eq!(q as f32 / 64.0, mx::dequantize(&t)[i]);
        assert!(((v * 64.0).round() as i32).clamp(-127, 127) == i32::from(q) || (v * 64.0).fract().abs() == 0.5);
    }
    // a hand-built tensor with -128 decodes to -2 * scale
    let h = MxTensor { elem: Elem::Int8, len: 1, scales: vec![128], data: vec![0x80] };
    assert_eq!(mx::dequantize(&h), vec![-4.0]);
}

#[test]
fn fake_quant_is_bit_identical_to_quantize_dequantize() {
    for (_, x) in datasets() {
        for e in Elem::ALL {
            for r in [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(0xdead_beef)] {
                let y = mx::dequantize(&mx::quantize(&x, e, r));
                let mut z = x.clone();
                fake_quant(&mut z, Format::Mx(e), r);
                let yb: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
                let zb: Vec<u32> = z.iter().map(|v| v.to_bits()).collect();
                assert!(yb == zb, "{e:?} {r:?}");
            }
        }
    }
}

#[test]
fn stochastic_mx_is_unbiased() {
    // 100k blocks, each anchored by amax = 1.0 in slot 0 with the probe in slot 1.
    let blocks = 100_000;
    for e in Elem::ALL {
        for probe in [0.3f32, -0.71, 0.013, 0.96] {
            let mut x = vec![0f32; blocks * 32];
            for b in 0..blocks {
                x[b * 32] = 1.0;
                x[b * 32 + 1] = probe;
            }
            let t = mx::quantize(&x, e, Rounding::Stochastic(0x1234_5678));
            let y = mx::dequantize(&t);
            let draws: Vec<f64> = (0..blocks).map(|b| y[b * 32 + 1] as f64).collect();
            assert_unbiased(&format!("{e:?} probe {probe}"), probe as f64, &draws);
            // and every draw is one of the two neighbours on the scaled grid
            let g = grid(e);
            let sc = 2f64.powi(-emax(e));
            let (lo, hi) = g.bracket((probe as f64).abs() / sc);
            assert!(draws.iter().all(|d| d.abs() == lo * sc || d.abs() == hi * sc), "{e:?}");
        }
    }
}

#[test]
fn stochastic_mx_depends_only_on_seed_and_index() {
    let mut rng = Rng::new(8);
    let x: Vec<f32> = (0..1000).map(|_| rng.normal() as f32).collect();
    for e in Elem::ALL {
        let a = mx::quantize(&x, e, Rounding::Stochastic(5));
        let b = mx::quantize(&x, e, Rounding::Stochastic(5));
        let c = mx::quantize(&x, e, Rounding::Stochastic(6));
        assert_eq!(a, b);
        assert_ne!(a, c);
        // The first 64 elements quantize identically on their own (global index = local index).
        let p = mx::quantize(&x[..64], e, Rounding::Stochastic(5));
        assert_eq!(p.scales[..], a.scales[..2]);
        assert_eq!(p.data[..], a.data[..e.data_len(64)]);
    }
}

#[test]
#[should_panic(expected = "scales length")]
fn malformed_tensor_panics() {
    let t = MxTensor { elem: Elem::Fp8E4M3, len: 33, scales: vec![127], data: vec![0; 33] };
    mx::dequantize(&t);
}

#[test]
fn elem_metadata() {
    assert_eq!(Elem::Fp8E4M3.max_finite(), 448.0);
    assert_eq!(Elem::Fp8E5M2.max_finite(), 57344.0);
    assert_eq!(Elem::Fp4E2M1.max_finite(), 6.0);
    assert_eq!(Elem::Int8.max_finite(), 127.0 / 64.0);
    for e in Elem::ALL {
        assert_eq!(e.emax(), emax(e));
        assert_eq!(e.max_finite() as f64, *grid(e).vals.last().unwrap());
    }
    assert_eq!(BLOCK, 32);
}
