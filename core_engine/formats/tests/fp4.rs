//! Exhaustive FP4 E2M1 tests (OCP MX v1.0) against an independent reference.

mod common;

use common::{assert_unbiased, next_down, next_up, RefOut, Rng, E2M1};
use forge_formats::{fp4, Rounding};

const TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[test]
fn decode_all_16_codes() {
    for c in 0..16u8 {
        let v = fp4::decode(c);
        assert_eq!(v.to_bits(), TABLE[c as usize].to_bits(), "code {c:#x}");
        assert_eq!(
            v as f64 * if c & 8 != 0 { -1.0 } else { 1.0 },
            E2M1.value(u32::from(c & 7))
        );
        for hi in 1..16u8 {
            assert_eq!(
                fp4::decode(c | hi << 4).to_bits(),
                v.to_bits(),
                "high nibble ignored"
            );
        }
    }
    assert_eq!(fp4::MAX_FINITE, 6.0);
    assert_eq!(fp4::decode(fp4::MAX_CODE), 6.0);
}

#[test]
fn encode_decode_identity_and_signed_zero() {
    for c in 0..16u8 {
        for r in [
            Rounding::NearestEven,
            Rounding::TowardZero,
            Rounding::Stochastic(0),
            Rounding::Stochastic(u32::MAX),
        ] {
            assert_eq!(fp4::encode(fp4::decode(c), r), c, "{c:#x} {r:?}");
        }
    }
    assert_eq!(fp4::encode(0.0, Rounding::NearestEven), 0x0);
    assert_eq!(fp4::encode(-0.0, Rounding::NearestEven), 0x8);
    assert_eq!(fp4::encode(-0.2, Rounding::NearestEven), 0x8); // underflow keeps sign
    assert_eq!(fp4::encode(0.25, Rounding::NearestEven), 0x0); // tie -> even (0)
    assert_eq!(fp4::encode(0.75, Rounding::NearestEven), 0x2); // tie 0.5|1 -> 1 (even code 2)
}

#[test]
fn midpoint_ties_go_to_even_codes() {
    let g = E2M1.grid();
    let mids = [
        (0.25f32, 0u8),
        (0.75, 2),
        (1.25, 2),
        (1.75, 4),
        (2.5, 4),
        (3.5, 6),
        (5.0, 6),
    ];
    for (m, want) in mids {
        assert_eq!(fp4::encode(m, Rounding::NearestEven), want, "mid {m}");
        assert_eq!(fp4::encode(-m, Rounding::NearestEven), want | 8, "mid -{m}");
        assert_eq!(g.nearest_even(m as f64), RefOut::Code(u32::from(want)));
        let (lo, _) = g.bracket(m as f64);
        let lo_code = g.vals.iter().position(|&v| v == lo).unwrap() as u8;
        assert_eq!(fp4::encode(next_down(m), Rounding::NearestEven), lo_code);
        assert_eq!(fp4::encode(next_up(m), Rounding::NearestEven), lo_code + 1);
        assert_eq!(fp4::encode(m, Rounding::TowardZero), lo_code);
    }
}

#[test]
fn saturation_nan_inf() {
    for r in [
        Rounding::NearestEven,
        Rounding::TowardZero,
        Rounding::Stochastic(u32::MAX),
        Rounding::Stochastic(0),
    ] {
        for x in [6.5f32, 7.0, 7.5, 8.0, 100.0, 1e30, f32::MAX, f32::INFINITY] {
            assert_eq!(fp4::encode(x, r), 0x7, "{x} {r:?}");
            assert_eq!(fp4::encode(-x, r), 0xf, "-{x} {r:?}");
        }
        assert_eq!(fp4::encode(f32::NAN, r), 0x7);
        assert_eq!(fp4::encode(-f32::NAN, r), 0x7);
        assert_eq!(fp4::encode(f32::from_bits(0x7f80_0001), r), 0x7);
    }
}

#[test]
fn dense_sweep_matches_reference_and_is_monotonic() {
    let g = E2M1.grid();
    for r in [Rounding::NearestEven, Rounding::TowardZero] {
        let mut prev = f32::NEG_INFINITY;
        let mut b = 0u32;
        while b < 0x7f80_0000 {
            let x = f32::from_bits(b);
            let c = fp4::encode(x, r);
            assert_eq!(c & 0xf0, 0, "only the low nibble is used");
            let y = fp4::decode(c);
            assert!(y >= prev, "not monotonic at {x:e}");
            prev = y;
            let want = match g.round(x as f64, r) {
                RefOut::Code(k) => k as u8,
                RefOut::Overflow => 7,
            };
            assert_eq!(c, want, "{r:?} {x:e}");
            assert_eq!(fp4::encode(-x, r), want | 8);
            b += 211;
        }
    }
}

#[test]
fn stochastic_rounding_is_unbiased() {
    let mut rng = Rng::new(99);
    let g = E2M1.grid();
    for x in [0.1f32, 0.3, 0.6, 1.2, -1.9, 2.2, 3.7, -4.9, 5.99] {
        let draws: Vec<f64> = (0..100_000)
            .map(|_| fp4::decode(fp4::encode(x, Rounding::Stochastic(rng.next_u32()))) as f64)
            .collect();
        let (lo, hi) = g.bracket((x as f64).abs());
        assert!(draws.iter().all(|d| d.abs() == lo || d.abs() == hi));
        assert_unbiased("fp4", x as f64, &draws);
    }
}

#[test]
fn slices_and_packing_match_scalar() {
    let mut rng = Rng::new(5);
    for n in [0usize, 1, 2, 3, 31, 32, 33, 1001] {
        let x: Vec<f32> = (0..n).map(|_| (rng.normal() * 3.0) as f32).collect();
        for r in [
            Rounding::NearestEven,
            Rounding::TowardZero,
            Rounding::Stochastic(1234),
        ] {
            let mut codes = vec![0u8; n];
            fp4::encode_slice(&x, &mut codes, r);
            let mut packed = vec![0u8; fp4::packed_len(n)];
            fp4::encode_packed(&x, &mut packed, r);
            let mut y = vec![0f32; n];
            fp4::decode_slice(&codes, &mut y);
            let mut z = vec![0f32; n];
            fp4::decode_packed(&packed, &mut z);
            for i in 0..n {
                let ri = match r {
                    Rounding::Stochastic(s) => {
                        Rounding::Stochastic(forge_formats::stochastic_bits(s, i as u64))
                    }
                    o => o,
                };
                assert_eq!(codes[i], fp4::encode(x[i], ri));
                assert_eq!((packed[i / 2] >> (4 * (i % 2))) & 0xf, codes[i]);
                assert_eq!(y[i].to_bits(), fp4::decode(codes[i]).to_bits());
                assert_eq!(z[i].to_bits(), y[i].to_bits());
            }
            if n % 2 == 1 {
                assert_eq!(packed[n / 2] >> 4, 0, "unused high nibble is zero");
            }
        }
    }
}
