//! Cross-check against golden vectors produced by an independent
//! implementation: numpy (float16) and ml_dtypes (bfloat16, float8_e4m3fn,
//! float8_e5m2, float4_e2m1fn). Regenerate with
//! `python3 tests/data/gen_golden_mldtypes.py > tests/data/golden_mldtypes.txt`.
//!
//! Columns (hex): f32 bits, bf16, fp16, E4M3 (RNE, non-saturating),
//! E5M2 (RNE, non-saturating), E4M3 (saturating), E5M2 (saturating),
//! FP4 E2M1 (saturating). ml_dtypes' non-saturating E4M3 overflow is a NaN
//! code; NaN codes are compared by NaN-ness.

use forge_formats::fp8::Kind;
use forge_formats::{bf16, fp16, fp4, fp8, Rounding};

const GOLDEN: &str = include_str!("data/golden_mldtypes.txt");

#[test]
fn matches_ml_dtypes_and_numpy() {
    let r = Rounding::NearestEven;
    let mut rows = 0;
    for line in GOLDEN
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let c: Vec<u32> = line
            .split_whitespace()
            .map(|t| u32::from_str_radix(t, 16).unwrap())
            .collect();
        assert_eq!(c.len(), 8, "bad golden line {line}");
        let x = f32::from_bits(c[0]);
        assert!(!x.is_nan());
        assert_eq!(u32::from(bf16::from_f32(x)), c[1], "bf16 {x:e}");
        assert_eq!(u32::from(fp16::from_f32(x)), c[2], "fp16 {x:e}");
        let e4 = fp8::encode(Kind::E4M3, x, r, false);
        if Kind::E4M3.is_nan(c[3] as u8) {
            assert!(Kind::E4M3.is_nan(e4), "e4m3 {x:e}");
        } else {
            assert_eq!(u32::from(e4), c[3], "e4m3 {x:e}");
        }
        let e5 = fp8::encode(Kind::E5M2, x, r, false);
        if Kind::E5M2.is_nan(c[4] as u8) {
            assert!(Kind::E5M2.is_nan(e5), "e5m2 {x:e}");
        } else {
            assert_eq!(u32::from(e5), c[4], "e5m2 {x:e}");
        }
        assert_eq!(
            u32::from(fp8::encode(Kind::E4M3, x, r, true)),
            c[5],
            "e4m3 sat {x:e}"
        );
        assert_eq!(
            u32::from(fp8::encode(Kind::E5M2, x, r, true)),
            c[6],
            "e5m2 sat {x:e}"
        );
        assert_eq!(u32::from(fp4::encode(x, r)), c[7], "fp4 {x:e}");
        rows += 1;
    }
    assert!(rows > 5000, "golden file truncated ({rows} rows)");
}
