//! Every runtime-dispatched slice kernel (baseline / AVX2 / AVX-512) must be
//! bit-identical to the scalar per-element definition. Lives in its own test
//! binary because the ISA limit is process-global.

mod common;

use common::Rng;
use forge_formats::fp8::Kind;
use forge_formats::{bf16, fake_quant, fp16, fp4, fp8, mx, set_isa_limit, stochastic_bits, Format, Isa, Rounding};

fn input(n: usize) -> Vec<f32> {
    let mut rng = Rng::new(0x15a);
    (0..n)
        .map(|i| match i % 13 {
            0 => f32::NAN,
            1 => f32::NEG_INFINITY,
            2 => f32::INFINITY,
            3 => f32::from_bits(rng.next_u32() & 0x807f_ffff), // f32 subnormals
            4 => -0.0,
            5 => f32::from_bits(rng.next_u32()),
            6 => (rng.normal() * 1e5) as f32,
            7 => (rng.normal() * 1e-5) as f32,
            _ => (rng.normal() * 3.0) as f32,
        })
        .collect()
}

/// Inputs without non-finite values, at very different block magnitudes, so
/// MX scales cover the whole E8M0 range including the clamped bottom.
fn mx_input(n: usize) -> Vec<f32> {
    let mut rng = Rng::new(0x3ee);
    (0..n)
        .map(|i| {
            let block = i / 32;
            let mag = match block % 6 {
                0 => 1e-42,
                1 => 1e-38,
                2 => 1e-30,
                3 => 1.0,
                4 => 1e30,
                _ => 1e38,
            };
            let v = rng.normal() * mag;
            if i % 29 == 0 {
                f32::from_bits(rng.next_u32() & 0x807f_ffff)
            } else {
                v as f32
            }
        })
        .collect()
}

fn elem_r(r: Rounding, i: usize) -> Rounding {
    match r {
        Rounding::Stochastic(s) => Rounding::Stochastic(stochastic_bits(s, i as u64)),
        o => o,
    }
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

const MODES: [Rounding; 3] = [Rounding::NearestEven, Rounding::TowardZero, Rounding::Stochastic(0xabc)];

#[test]
fn all_isa_levels_match_scalar_definitions() {
    let n = 4099; // not a multiple of any vector width
    let x = input(n);
    let xm = mx_input(n);
    // Reference MX results computed at the baseline level.
    set_isa_limit(Isa::Baseline);
    let mx_ref: Vec<Vec<mx::MxTensor>> =
        mx::Elem::ALL.iter().map(|&e| MODES.iter().map(|&r| mx::quantize(&xm, e, r)).collect()).collect();

    for isa in [Isa::Baseline, Isa::Avx2, Isa::Avx512] {
        set_isa_limit(isa);
        let active = forge_formats::active_isa();
        assert!(active <= isa);

        for r in MODES {
            // bf16 / fp16
            let mut h = vec![0u16; n];
            bf16::encode_slice(&x, &mut h, r);
            for i in 0..n {
                assert_eq!(h[i], bf16::encode(x[i], elem_r(r, i)), "{active:?} bf16 {r:?} {i}");
            }
            let mut y = vec![0f32; n];
            bf16::decode_slice(&h, &mut y);
            assert!(same_bits(&y, &h.iter().map(|&c| bf16::to_f32(c)).collect::<Vec<_>>()));
            fp16::encode_slice(&x, &mut h, r);
            for i in 0..n {
                assert_eq!(h[i], fp16::encode(x[i], elem_r(r, i)), "{active:?} fp16 {r:?} {i}");
            }
            fp16::decode_slice(&h, &mut y);
            assert!(same_bits(&y, &h.iter().map(|&c| fp16::to_f32(c)).collect::<Vec<_>>()));

            // fp8
            let mut b = vec![0u8; n];
            for kind in [Kind::E4M3, Kind::E5M2] {
                for sat in [false, true] {
                    fp8::encode_slice(kind, &x, &mut b, r, sat);
                    for i in 0..n {
                        assert_eq!(b[i], fp8::encode(kind, x[i], elem_r(r, i), sat), "{active:?} {kind:?} {r:?} {i}");
                    }
                    fp8::decode_slice(kind, &b, &mut y);
                    assert!(same_bits(&y, &b.iter().map(|&c| fp8::decode(kind, c)).collect::<Vec<_>>()));
                }
            }

            // fp4
            fp4::encode_slice(&x, &mut b, r);
            for i in 0..n {
                assert_eq!(b[i], fp4::encode(x[i], elem_r(r, i)), "{active:?} fp4 {r:?} {i}");
            }
            let mut p = vec![0u8; fp4::packed_len(n)];
            fp4::encode_packed(&x, &mut p, r);
            for i in 0..n {
                assert_eq!((p[i / 2] >> (4 * (i % 2))) & 0xf, b[i]);
            }
            fp4::decode_packed(&p, &mut y);
            assert!(same_bits(&y, &b.iter().map(|&c| fp4::decode(c)).collect::<Vec<_>>()));

            // fake_quant for scalar formats
            for f in [Format::Bf16, Format::Fp16, Format::Fp8(Kind::E4M3), Format::Fp8(Kind::E5M2), Format::Fp4] {
                let mut z = x.clone();
                fake_quant(&mut z, f, r);
                for i in 0..n {
                    let ri = elem_r(r, i);
                    let want = match f {
                        Format::Bf16 => bf16::to_f32(bf16::encode(x[i], ri)),
                        Format::Fp16 => fp16::to_f32(fp16::encode(x[i], ri)),
                        Format::Fp8(k) => fp8::decode(k, fp8::encode(k, x[i], ri, true)),
                        _ if x[i].is_nan() => x[i],
                        _ => fp4::decode(fp4::encode(x[i], ri)),
                    };
                    assert!(want.to_bits() == z[i].to_bits() || (want.is_nan() && z[i].is_nan()), "{active:?} {f:?} {i}");
                }
            }
        }

        // MX: identical tensors and dequantized values at every level.
        for (ei, &e) in mx::Elem::ALL.iter().enumerate() {
            for (ri, &r) in MODES.iter().enumerate() {
                let t = mx::quantize(&xm, e, r);
                assert_eq!(t, mx_ref[ei][ri], "{active:?} {e:?} {r:?}");
                let y = mx::dequantize(&t);
                let mut z = xm.clone();
                fake_quant(&mut z, Format::Mx(e), r);
                assert!(same_bits(&y, &z), "{active:?} {e:?} {r:?} fake_quant");
                for i in 0..n {
                    let s = mx::e8m0_to_f32(t.scales[i / 32]);
                    assert_eq!(y[i].to_bits(), (mx::decode_elem(e, t.code(i)) * s).to_bits());
                }
            }
            // MX with NaN/Inf blocks too
            let t = mx::quantize(&x, e, Rounding::NearestEven);
            set_isa_limit(Isa::Baseline);
            let t0 = mx::quantize(&x, e, Rounding::NearestEven);
            let y0 = mx::dequantize(&t0);
            set_isa_limit(isa);
            assert_eq!(t, t0);
            let y = mx::dequantize(&t);
            assert!(y.iter().zip(&y0).all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())));
        }
    }
    set_isa_limit(Isa::Avx512);
}
