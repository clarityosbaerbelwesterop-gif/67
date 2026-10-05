//! Correctness tests of every variant against an f64 reference.
//!
//! Error model: for an f32 engine every element must satisfy
//! `|c - ref| <= 1.5·(k+4)·2^-24 · mag`, where `mag = |alpha|·Σ|a·b| + |beta·c0|`
//! (a rigorous bound for f32 accumulation in any order). The AmxBf16 variant is
//! checked against a reference computed on bf16-rounded inputs with the same
//! bound (products of bf16 values are exact in f32), and against the unrounded
//! reference with the additional bf16 input-rounding error `2^-8 · mag`.
//!
//! Operand padding (between the logical row length and the leading dimension,
//! and after the last row) is filled with NaN so that any read outside the
//! logical region poisons the result; padding of C must stay bit-identical.

use crate::*;

const U: f64 = 1.0 / (1u64 << 24) as f64;
/// Bit pattern stored in every non-logical element of C.
const C_SENTINEL: u32 = 0x7FC0_BEEF;

pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        Rng(seed ^ 0xD1B5_4A32_D192_ED03)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Uniform in [-1, 1), a multiple of 2^-23.
    pub(crate) fn sym(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u32 << 23) as f32) - 1.0
    }
    pub(crate) fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fill {
    /// Uniform [-1, 1).
    Random,
    /// Small integers in [-4, 4] (exact in bf16; exact sums for modest k).
    Ints,
    Nan,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    ta: Trans,
    tb: Trans,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    pad_a: usize,
    pad_b: usize,
    pad_c: usize,
}

impl Case {
    fn new(ta: Trans, tb: Trans, m: usize, n: usize, k: usize) -> Case {
        Case {
            ta,
            tb,
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
            pad_a: 0,
            pad_b: 0,
            pad_c: 0,
        }
    }
    fn ab(mut self, alpha: f32, beta: f32) -> Case {
        self.alpha = alpha;
        self.beta = beta;
        self
    }
    fn pads(mut self, pa: usize, pb: usize, pc: usize) -> Case {
        self.pad_a = pa;
        self.pad_b = pb;
        self.pad_c = pc;
        self
    }
}

fn stored(t: Trans, rows: usize, cols: usize) -> (usize, usize) {
    match t {
        Trans::N => (rows, cols),
        Trans::T => (cols, rows),
    }
}

/// Builds a padded stored matrix; returns (data, ld). Padding is NaN.
fn matrix(
    rows: usize,
    cols: usize,
    pad: usize,
    fill: Fill,
    rng: &mut Rng,
    tail: usize,
) -> (Vec<f32>, usize) {
    let ld = cols + pad;
    let len = if rows == 0 || cols == 0 {
        0
    } else {
        (rows - 1) * ld + cols
    } + tail;
    let mut v = vec![f32::NAN; len];
    for r in 0..rows {
        for c in 0..cols {
            v[r * ld + c] = match fill {
                Fill::Random => rng.sym(),
                Fill::Ints => (rng.below(9) as f32) - 4.0,
                Fill::Nan => f32::NAN,
            };
        }
    }
    (v, ld)
}

struct Data {
    a: Vec<f32>,
    lda: usize,
    b: Vec<f32>,
    ldb: usize,
    c0: Vec<f32>,
    ldc: usize,
}

fn make(case: &Case, ab_fill: Fill, c_fill: Fill, rng: &mut Rng) -> Data {
    let (ar, ac) = stored(case.ta, case.m, case.k);
    let (br, bc) = stored(case.tb, case.k, case.n);
    let tail = rng.below(3);
    let (a, lda) = matrix(ar, ac, case.pad_a, ab_fill, rng, tail);
    let (b, ldb) = matrix(br, bc, case.pad_b, ab_fill, rng, tail);
    let ldc = case.n + case.pad_c;
    let len = if case.m == 0 || case.n == 0 {
        0
    } else {
        (case.m - 1) * ldc + case.n
    } + tail;
    let mut c0 = vec![f32::from_bits(C_SENTINEL); len];
    for i in 0..case.m {
        for j in 0..case.n {
            c0[i * ldc + j] = match c_fill {
                Fill::Random => rng.sym(),
                Fill::Ints => (rng.below(9) as f32) - 4.0,
                Fill::Nan => f32::NAN,
            };
        }
    }
    Data {
        a,
        lda,
        b,
        ldb,
        c0,
        ldc,
    }
}

fn elem(x: &[f32], ld: usize, t: Trans, r: usize, c: usize) -> f32 {
    match t {
        Trans::N => x[r * ld + c],
        Trans::T => x[c * ld + r],
    }
}

/// f64 reference and per-element magnitude `|alpha|·Σ|a·b| + |beta·c0|`.
fn reference(case: &Case, d: &Data, bf16: bool) -> (Vec<f64>, Vec<f64>) {
    let r = |x: f32| {
        if bf16 {
            f64::from(bf16_round(x))
        } else {
            f64::from(x)
        }
    };
    let (m, n, k) = (case.m, case.n, case.k);
    let (alpha, beta) = (f64::from(case.alpha), f64::from(case.beta));
    let mut out = vec![0.0; m * n];
    let mut mag = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let (mut s, mut g) = (0.0f64, 0.0f64);
            if case.alpha != 0.0 {
                for p in 0..k {
                    let t =
                        r(elem(&d.a, d.lda, case.ta, i, p)) * r(elem(&d.b, d.ldb, case.tb, p, j));
                    s += t;
                    g += t.abs();
                }
            }
            let cb = if case.beta == 0.0 {
                0.0
            } else {
                beta * f64::from(d.c0[i * d.ldc + j])
            };
            out[i * n + j] = alpha * s + cb;
            mag[i * n + j] = alpha.abs() * g + cb.abs();
        }
    }
    (out, mag)
}

fn uses_bf16(v: GemmVariant) -> bool {
    effective_variant(v) == GemmVariant::AmxBf16
}

/// Checks `got` (the full C buffer after the call) against the reference.
fn verify(v: GemmVariant, case: &Case, d: &Data, got: &[f32]) -> Result<(), String> {
    let (m, n, k) = (case.m, case.n, case.k);
    let bf16 = uses_bf16(v);
    let (rf, mag) = reference(case, d, bf16);
    let f = 1.5 * (k as f64 + 4.0) * U;
    let ctx = || format!("{v:?} {case:?}");
    for i in 0..m {
        for j in 0..n {
            let x = f64::from(got[i * d.ldc + j]);
            let (want, g) = (rf[i * n + j], mag[i * n + j]);
            let tol = f * g + 1e-35;
            let ok = (x - want).abs() <= tol;
            if !ok {
                return Err(format!(
                    "{}: C[{i},{j}] = {x}, reference {want}, tol {tol:e}",
                    ctx()
                ));
            }
        }
    }
    if bf16 {
        let (rf, mag) = reference(case, d, false);
        let f = f + 1.0 / 256.0 + U;
        for i in 0..m {
            for j in 0..n {
                let x = f64::from(got[i * d.ldc + j]);
                let tol = f * mag[i * n + j] + 1e-35;
                let ok = (x - rf[i * n + j]).abs() <= tol;
                if !ok {
                    return Err(format!(
                        "{}: C[{i},{j}] = {x} vs unrounded {}, tol {tol:e}",
                        ctx(),
                        rf[i * n + j]
                    ));
                }
            }
        }
    } else if m > 0 && n > 0 {
        // Design contract: within 1e-4 relative (vs f64), scaled by sqrt(k).
        let max_ref = rf.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
        let max_err = (0..m * n)
            .map(|e| (f64::from(got[e / n * d.ldc + e % n]) - rf[e]).abs())
            .fold(0.0f64, f64::max);
        if max_ref > 0.0 && max_err > 1e-4 * (k.max(1) as f64).sqrt() * max_ref {
            return Err(format!(
                "{}: relative error {:e} too large",
                ctx(),
                max_err / max_ref
            ));
        }
    }
    // Everything outside the logical region is untouched.
    for (e, (&g, &o)) in got.iter().zip(&d.c0).enumerate() {
        let (i, j) = (e / d.ldc.max(1), e % d.ldc.max(1));
        let logical = i < m && j < n;
        if !logical && g.to_bits() != o.to_bits() {
            return Err(format!("{}: C padding element {e} modified ({g})", ctx()));
        }
    }
    Ok(())
}

fn run_case(pool: &Pool, v: GemmVariant, case: &Case, d: &Data) -> Vec<f32> {
    let mut got = d.c0.clone();
    gemm(
        pool, v, case.ta, case.tb, case.m, case.n, case.k, case.alpha, &d.a, d.lda, &d.b, d.ldb,
        case.beta, &mut got, d.ldc,
    );
    got
}

fn check_case(pool: &Pool, v: GemmVariant, case: &Case, ab: Fill, cf: Fill, rng: &mut Rng) {
    let d = make(case, ab, cf, rng);
    let got = run_case(pool, v, case, &d);
    if let Err(e) = verify(v, case, &d, &got) {
        panic!("{e}");
    }
}

const TRANS: [(Trans, Trans); 4] = [
    (Trans::N, Trans::N),
    (Trans::N, Trans::T),
    (Trans::T, Trans::N),
    (Trans::T, Trans::T),
];
const ALPHA_BETA: [(f32, f32); 8] = [
    (1.0, 0.0),
    (1.0, 1.0),
    (-1.5, 0.0),
    (0.5, 1.0),
    (1.0, -0.75),
    (2.0, 0.5),
    (-1.0, 1.0),
    (0.25, -2.0),
];

/// Every variant (unavailable ones exercise the fallback path).
fn variants() -> [GemmVariant; 4] {
    GemmVariant::ALL
}

#[test]
fn small_shapes_exhaustive() {
    let pool = Pool::new(2);
    let dims = [1usize, 2, 3, 5, 7, 8, 13, 14, 15, 16, 17, 31, 32, 33];
    for v in variants() {
        let mut rng = Rng::new(1 + v as u64);
        let mut it = 0usize;
        for &m in &dims {
            for &n in &dims {
                for &k in &dims {
                    for (ta, tb) in TRANS {
                        let (alpha, beta) = ALPHA_BETA[it % ALPHA_BETA.len()];
                        let pads = (it % 3, (it / 3) % 2 * 3, (it / 6) % 3);
                        it += 1;
                        let case = Case::new(ta, tb, m, n, k)
                            .ab(alpha, beta)
                            .pads(pads.0, pads.1, pads.2);
                        let cf = if beta == 0.0 { Fill::Nan } else { Fill::Random };
                        check_case(&pool, v, &case, Fill::Random, cf, &mut rng);
                    }
                }
            }
        }
    }
}

#[test]
fn sampled_shapes_from_design_list() {
    let pool = Pool::new(4);
    let dims = [
        1usize, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 511,
    ];
    for v in variants() {
        let mut rng = Rng::new(100 + v as u64);
        for it in 0..48 {
            let (m, n, k) = (rng.pick(&dims), rng.pick(&dims), rng.pick(&dims));
            for (ta, tb) in TRANS {
                let (alpha, beta) = ALPHA_BETA[(it + rng.below(8)) % ALPHA_BETA.len()];
                let case = Case::new(ta, tb, m, n, k).ab(alpha, beta).pads(
                    rng.below(2) * rng.below(9),
                    rng.below(2) * rng.below(9),
                    rng.below(2) * rng.below(9),
                );
                let cf = if beta == 0.0 { Fill::Nan } else { Fill::Random };
                check_case(&pool, v, &case, Fill::Random, cf, &mut rng);
            }
        }
    }
}

#[test]
fn large_shapes_cross_all_blocking_boundaries() {
    let pool = Pool::new(4);
    // Shapes spanning several depth blocks, slabs and row units.
    let shapes = [
        (511usize, 511usize, 511usize),
        (300, 257, 1100),
        (64, 4100, 70),
        (1000, 33, 600),
        (129, 1030, 513),
    ];
    for v in variants() {
        let mut rng = Rng::new(200 + v as u64);
        for (idx, &(m, n, k)) in shapes.iter().enumerate() {
            let (ta, tb) = TRANS[idx % 4];
            let (alpha, beta) = ALPHA_BETA[idx % ALPHA_BETA.len()];
            let case = Case::new(ta, tb, m, n, k)
                .ab(alpha, beta)
                .pads(idx % 2, 3 * (idx % 3), 5);
            check_case(&pool, v, &case, Fill::Random, Fill::Random, &mut rng);
            let case = case.ab(alpha, 0.0);
            check_case(&pool, v, &case, Fill::Random, Fill::Nan, &mut rng);
        }
    }
}

#[test]
fn beta_zero_overwrites_nan_and_inf() {
    let pool = Pool::new(3);
    for v in variants() {
        let mut rng = Rng::new(300 + v as u64);
        for (ta, tb) in TRANS {
            for &(m, n, k) in &[
                (1, 1, 1),
                (7, 9, 5),
                (33, 65, 17),
                (64, 64, 64),
                (100, 37, 300),
            ] {
                for alpha in [1.0f32, -2.0, 0.0] {
                    let case = Case::new(ta, tb, m, n, k).ab(alpha, 0.0).pads(1, 0, 2);
                    let mut d = make(&case, Fill::Random, Fill::Nan, &mut rng);
                    // Mix in infinities: beta == 0 must not read them either.
                    for (i, x) in d.c0.iter_mut().enumerate() {
                        if i % 5 == 0 && x.to_bits() != C_SENTINEL {
                            *x = if i % 2 == 0 {
                                f32::INFINITY
                            } else {
                                f32::NEG_INFINITY
                            };
                        }
                    }
                    let got = run_case(&pool, v, &case, &d);
                    verify(v, &case, &d, &got).unwrap();
                    for i in 0..m {
                        for j in 0..n {
                            assert!(got[i * d.ldc + j].is_finite(), "{v:?} {case:?} C[{i},{j}]");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn alpha_beta_combinations() {
    let pool = Pool::new(2);
    let alphas = [1.0f32, -1.0, 0.5, 3.0, 0.0];
    let betas = [0.0f32, 1.0, -1.0, 0.25, 2.0];
    for v in variants() {
        let mut rng = Rng::new(400 + v as u64);
        for (t, &(m, n, k)) in [(45usize, 70usize, 33usize), (128, 96, 200), (17, 300, 64)]
            .iter()
            .enumerate()
        {
            for &alpha in &alphas {
                for &beta in &betas {
                    let (ta, tb) = TRANS[t % 4];
                    let case = Case::new(ta, tb, m, n, k).ab(alpha, beta).pads(2, 1, 3);
                    let cf = if beta == 0.0 { Fill::Nan } else { Fill::Random };
                    check_case(&pool, v, &case, Fill::Random, cf, &mut rng);
                }
            }
        }
    }
}

#[test]
fn exact_integer_products_are_exact() {
    // Small integers are exact in bf16 and every partial sum is an exact f32,
    // so every variant must reproduce the reference bit for bit.
    let pool = Pool::new(4);
    for v in variants() {
        let mut rng = Rng::new(500 + v as u64);
        for (ta, tb) in TRANS {
            for &(m, n, k) in &[(5, 3, 2), (33, 47, 64), (64, 64, 300), (97, 31, 1000)] {
                for (alpha, beta) in [(1.0f32, 0.0f32), (1.0, 1.0), (2.0, -1.0), (-0.5, 0.5)] {
                    let case = Case::new(ta, tb, m, n, k).ab(alpha, beta).pads(1, 2, 3);
                    let d = make(&case, Fill::Ints, Fill::Ints, &mut rng);
                    let got = run_case(&pool, v, &case, &d);
                    let (rf, _) = reference(&case, &d, false);
                    for i in 0..m {
                        for j in 0..n {
                            assert_eq!(
                                f64::from(got[i * d.ldc + j]),
                                rf[i * n + j],
                                "{v:?} {case:?} C[{i},{j}]"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn nan_and_inf_inputs_propagate() {
    let pool = Pool::new(2);
    for v in variants() {
        let mut rng = Rng::new(600 + v as u64);
        for (ta, tb) in TRANS {
            let (m, n, k) = (40, 50, 70);
            let case = Case::new(ta, tb, m, n, k);
            let mut d = make(&case, Fill::Random, Fill::Random, &mut rng);
            // NaN at op(A)[3, 10]: row 3 of C becomes NaN.
            let (ia, pa) = (3usize, 10usize);
            let pos = match ta {
                Trans::N => ia * d.lda + pa,
                Trans::T => pa * d.lda + ia,
            };
            d.a[pos] = f32::NAN;
            // +Inf at op(B)[20, 7] with a positive op(A)[5, 20]: C[5, 7] is +Inf.
            let pb = match tb {
                Trans::N => 20 * d.ldb + 7,
                Trans::T => 7 * d.ldb + 20,
            };
            d.b[pb] = f32::INFINITY;
            let pa5 = match ta {
                Trans::N => 5 * d.lda + 20,
                Trans::T => 20 * d.lda + 5,
            };
            d.a[pa5] = 0.75;
            let got = run_case(&pool, v, &case, &d);
            for j in 0..n {
                assert!(got[ia * d.ldc + j].is_nan(), "{v:?} {case:?} C[{ia},{j}]");
            }
            assert_eq!(got[5 * d.ldc + 7], f32::INFINITY, "{v:?} {case:?}");
            assert!(got[6 * d.ldc + 8].is_finite(), "{v:?} {case:?}");
        }
    }
}

#[test]
fn degenerate_dimensions() {
    let pool = Pool::new(2);
    for v in variants() {
        // m == 0 or n == 0: nothing is touched, empty slices are fine.
        gemm(
            &pool,
            v,
            Trans::N,
            Trans::N,
            0,
            5,
            3,
            1.0,
            &[],
            3,
            &[1.0; 15],
            5,
            0.0,
            &mut [],
            5,
        );
        let mut c = [7.0f32; 4];
        gemm(
            &pool,
            v,
            Trans::N,
            Trans::N,
            2,
            0,
            3,
            1.0,
            &[1.0; 6],
            3,
            &[],
            0,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, [7.0; 4]);
        // k == 0: C = beta·C, A and B unreferenced (may be empty).
        let mut c = [1.0f32, 2.0, f32::NAN, 4.0];
        gemm(
            &pool,
            v,
            Trans::N,
            Trans::T,
            2,
            2,
            0,
            1.0,
            &[],
            0,
            &[],
            0,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, [0.0; 4]);
        let mut c = [1.0f32, 2.0, 3.0, 4.0];
        gemm(
            &pool,
            v,
            Trans::T,
            Trans::N,
            2,
            2,
            0,
            1.0,
            &[],
            0,
            &[],
            0,
            -2.0,
            &mut c,
            2,
        );
        assert_eq!(c, [-2.0, -4.0, -6.0, -8.0]);
        // alpha == 0: A and B unreferenced even when k > 0.
        let mut c = [1.0f32, 2.0, 3.0, 4.0, 99.0];
        gemm(
            &pool,
            v,
            Trans::N,
            Trans::N,
            2,
            2,
            5,
            0.0,
            &[],
            5,
            &[],
            2,
            0.5,
            &mut c,
            2,
        );
        assert_eq!(c, [0.5, 1.0, 1.5, 2.0, 99.0]);
        let mut c = [f32::NAN; 4];
        gemm(
            &pool,
            v,
            Trans::N,
            Trans::N,
            2,
            2,
            5,
            0.0,
            &[],
            5,
            &[],
            2,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, [0.0; 4]);
        // Single row / column: the leading dimension of a one-row matrix is free.
        let mut c = [0.0f32; 3];
        gemm(
            &pool,
            v,
            Trans::N,
            Trans::N,
            1,
            3,
            2,
            1.0,
            &[1.0, 2.0],
            0,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3,
            0.0,
            &mut c,
            0,
        );
        assert_eq!(c, [9.0, 12.0, 15.0]);
    }
}

#[test]
fn argument_validation_panics() {
    let pool = Pool::new(1);
    let ok = |f: &dyn Fn()| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_ok();
    let a = vec![1.0f32; 12];
    let b = vec![1.0f32; 12];
    // Baseline call is fine.
    assert!(ok(&|| {
        let mut c = vec![0.0f32; 9];
        gemm(
            &pool,
            GemmVariant::Scalar,
            Trans::N,
            Trans::N,
            3,
            3,
            4,
            1.0,
            &a,
            4,
            &b,
            3,
            0.0,
            &mut c,
            3,
        );
    }));
    // A too short.
    assert!(!ok(&|| {
        let mut c = vec![0.0f32; 9];
        gemm(
            &pool,
            GemmVariant::Scalar,
            Trans::N,
            Trans::N,
            3,
            3,
            4,
            1.0,
            &a[..11],
            4,
            &b,
            3,
            0.0,
            &mut c,
            3,
        );
    }));
    // B too short (transposed layout needs (n-1)*ldb + k).
    assert!(!ok(&|| {
        let mut c = vec![0.0f32; 9];
        gemm(
            &pool,
            GemmVariant::Scalar,
            Trans::N,
            Trans::T,
            3,
            3,
            4,
            1.0,
            &a,
            4,
            &b[..11],
            4,
            0.0,
            &mut c,
            3,
        );
    }));
    // C too short.
    assert!(!ok(&|| {
        let mut c = vec![0.0f32; 8];
        gemm(
            &pool,
            GemmVariant::Scalar,
            Trans::N,
            Trans::N,
            3,
            3,
            4,
            1.0,
            &a,
            4,
            &b,
            3,
            0.0,
            &mut c,
            3,
        );
    }));
    // Leading dimension smaller than the row length.
    assert!(!ok(&|| {
        let mut c = vec![0.0f32; 9];
        gemm(
            &pool,
            GemmVariant::Scalar,
            Trans::N,
            Trans::N,
            3,
            3,
            4,
            1.0,
            &a,
            3,
            &b,
            3,
            0.0,
            &mut c,
            3,
        );
    }));
    assert!(!ok(&|| {
        let mut c = vec![0.0f32; 9];
        gemm(
            &pool,
            GemmVariant::Scalar,
            Trans::N,
            Trans::N,
            3,
            3,
            4,
            1.0,
            &a,
            4,
            &b,
            3,
            0.0,
            &mut c,
            2,
        );
    }));
    // Batched: the last item must fit.
    assert!(!ok(&|| {
        let mut c = vec![0.0f32; 9];
        gemm_batched(
            &pool,
            GemmVariant::Scalar,
            Trans::N,
            Trans::N,
            2,
            1,
            3,
            2,
            1.0,
            &a,
            2,
            2,
            &b,
            3,
            6,
            0.0,
            &mut c,
            3,
            7,
        );
    }));
}

#[test]
fn results_independent_of_thread_count() {
    let pools: Vec<Pool> = [1, 2, 3, 4].into_iter().map(Pool::new).collect();
    for v in variants() {
        let mut rng = Rng::new(700 + v as u64);
        for (idx, &(m, n, k)) in [
            (300usize, 200usize, 700usize),
            (37, 1000, 129),
            (1024, 64, 256),
        ]
        .iter()
        .enumerate()
        {
            let (ta, tb) = TRANS[idx + 1];
            let case = Case::new(ta, tb, m, n, k).ab(1.0, 0.5).pads(0, 1, 0);
            let d = make(&case, Fill::Random, Fill::Random, &mut rng);
            let first = run_case(&pools[0], v, &case, &d);
            verify(v, &case, &d, &first).unwrap();
            for pool in &pools[1..] {
                let got = run_case(pool, v, &case, &d);
                assert!(
                    got.iter()
                        .zip(&first)
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{v:?} {case:?} differs with {} threads",
                    pool.threads()
                );
            }
        }
    }
}

/// Reference for one batch item at the given offsets.
#[allow(clippy::too_many_arguments)]
fn check_batch(
    v: GemmVariant,
    case: &Case,
    batch: usize,
    a: &[f32],
    lda: usize,
    sa: usize,
    b: &[f32],
    ldb: usize,
    sb: usize,
    c0: &[f32],
    got: &[f32],
    ldc: usize,
    sc: usize,
) {
    let ext_c = (case.m - 1) * ldc + case.n;
    for i in 0..batch {
        let d = Data {
            a: a[i * sa..].to_vec(),
            lda,
            b: b[i * sb..].to_vec(),
            ldb,
            c0: c0[i * sc..i * sc + ext_c].to_vec(),
            ldc,
        };
        let g = &got[i * sc..i * sc + ext_c];
        if let Err(e) = verify(v, case, &d, g) {
            panic!("batch item {i}: {e}");
        }
    }
    // Gaps between items are untouched.
    for (e, (&g, &o)) in got.iter().zip(c0).enumerate() {
        let item = e / sc.max(1);
        let off = e - item * sc;
        let inside = item < batch && off < ext_c && off % ldc < case.n;
        if !inside {
            assert_eq!(g.to_bits(), o.to_bits(), "{v:?} gap element {e} modified");
        }
    }
}

#[test]
fn batched_strided_items() {
    let pool = Pool::new(4);
    for v in variants() {
        let mut rng = Rng::new(800 + v as u64);
        // (batch, m, n, k): small items (parallel across the batch) and large
        // items (parallel inside each item); batch not a multiple of threads.
        for &(batch, m, n, k) in &[
            (5usize, 7usize, 9usize, 11usize),
            (32, 64, 64, 32),
            (3, 260, 300, 200),
            (8, 256, 256, 64),
            (1, 33, 17, 5),
        ] {
            for (ta, tb) in TRANS {
                let (alpha, beta) = ALPHA_BETA[rng.below(ALPHA_BETA.len())];
                let case = Case::new(ta, tb, m, n, k).ab(alpha, beta).pads(1, 2, 3);
                let (ar, ac) = stored(ta, m, k);
                let (br, bc) = stored(tb, k, n);
                let (lda, ldb, ldc) = (ac + 1, bc + 2, n + 3);
                let (sa, sb, sc) = (
                    (ar - 1) * lda + ac + 5,
                    (br - 1) * ldb + bc + 1,
                    (m - 1) * ldc + n + 7,
                );
                let a: Vec<f32> = (0..batch * sa).map(|_| rng.sym()).collect();
                let b: Vec<f32> = (0..batch * sb).map(|_| rng.sym()).collect();
                let c0: Vec<f32> = (0..batch * sc)
                    .map(|_| if beta == 0.0 { f32::NAN } else { rng.sym() })
                    .collect();
                let mut got = c0.clone();
                gemm_batched(
                    &pool, v, ta, tb, batch, m, n, k, alpha, &a, lda, sa, &b, ldb, sb, beta,
                    &mut got, ldc, sc,
                );
                check_batch(
                    v, &case, batch, &a, lda, sa, &b, ldb, sb, &c0, &got, ldc, sc,
                );
            }
        }
    }
}

#[test]
fn batched_broadcast_operand_and_overlapping_outputs() {
    let pool = Pool::new(3);
    for v in variants() {
        let mut rng = Rng::new(900 + v as u64);
        let (batch, m, n, k) = (6usize, 20usize, 24usize, 40usize);
        // Shared weight (stride_b = 0) applied to every item.
        let a: Vec<f32> = (0..batch * m * k).map(|_| rng.sym()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.sym()).collect();
        let c0 = vec![f32::NAN; batch * m * n];
        let mut got = c0.clone();
        gemm_batched(
            &pool,
            v,
            Trans::N,
            Trans::N,
            batch,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            m * k,
            &b,
            n,
            0,
            0.0,
            &mut got,
            n,
            m * n,
        );
        let case = Case::new(Trans::N, Trans::N, m, n, k);
        check_batch(v, &case, batch, &a, k, m * k, &b, n, 0, &c0, &got, n, m * n);

        // stride_c = 0 with beta = 1: items accumulate in order into one C,
        // exactly like a loop of gemm calls.
        let c_init: Vec<f32> = (0..m * n).map(|_| rng.sym()).collect();
        let mut acc = c_init.clone();
        gemm_batched(
            &pool,
            v,
            Trans::N,
            Trans::N,
            batch,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            m * k,
            &b,
            n,
            0,
            1.0,
            &mut acc,
            n,
            0,
        );
        let mut seq = c_init.clone();
        for i in 0..batch {
            gemm(
                &pool,
                v,
                Trans::N,
                Trans::N,
                m,
                n,
                k,
                1.0,
                &a[i * m * k..],
                k,
                &b,
                n,
                1.0,
                &mut seq,
                n,
            );
        }
        assert!(
            acc.iter()
                .zip(&seq)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{v:?} overlapping C"
        );

        // batch == 0 touches nothing; alpha == 0 scales each item.
        let mut c = vec![3.0f32; 8];
        gemm_batched(
            &pool,
            v,
            Trans::N,
            Trans::N,
            0,
            2,
            2,
            2,
            1.0,
            &[],
            2,
            4,
            &[],
            2,
            4,
            0.0,
            &mut c,
            2,
            4,
        );
        assert_eq!(c, vec![3.0; 8]);
        gemm_batched(
            &pool,
            v,
            Trans::N,
            Trans::N,
            2,
            2,
            2,
            2,
            0.0,
            &[],
            2,
            4,
            &[],
            2,
            4,
            0.5,
            &mut c,
            2,
            4,
        );
        assert_eq!(c, vec![1.5; 8]);
    }
}

#[test]
fn bf16_round_matches_bit_reference() {
    // Ties to even, carries into the exponent, overflow to infinity, NaN quieting.
    let cases: [(u32, u16); 10] = [
        (0x3F80_0000, 0x3F80), // 1.0
        (0x3F80_8000, 0x3F80), // tie, even stays
        (0x3F81_8000, 0x3F82), // tie, odd rounds up
        (0x3F80_8001, 0x3F81), // above tie
        (0x3FFF_FFFF, 0x4000), // carry into exponent
        (0x7F7F_FFFF, 0x7F80), // f32::MAX rounds to +inf
        (0xFF80_0000, 0xFF80), // -inf
        (0x7F80_0001, 0x7FC0), // signalling NaN is quieted
        (0xFFC0_0001, 0xFFC0), // negative quiet NaN
        (0x0000_8000, 0x0000), // tiny denormal tie rounds to even zero
    ];
    for (bits, want) in cases {
        assert_eq!(bf16_bits(f32::from_bits(bits)), want, "{bits:#x}");
    }
    assert!(bf16_round(f32::NAN).is_nan());
    let mut rng = Rng::new(42);
    for _ in 0..100_000 {
        let x = f32::from_bits(rng.next_u64() as u32);
        let r = bf16_round(x);
        if x.is_nan() {
            assert!(r.is_nan());
            continue;
        }
        // Nearest: no other bf16 value is closer.
        let h = bf16_bits(x);
        let up = f32::from_bits(u32::from(h.wrapping_add(1)) << 16);
        let dn = f32::from_bits(u32::from(h.wrapping_sub(1)) << 16);
        let e = (f64::from(r) - f64::from(x)).abs();
        for nb in [up, dn] {
            if nb.is_finite() && r.is_finite() {
                assert!(e <= (f64::from(nb) - f64::from(x)).abs(), "{x:e}");
            }
        }
    }
}

#[test]
fn variant_names_round_trip() {
    for v in GemmVariant::ALL {
        assert_eq!(GemmVariant::from_name(v.name()), Some(v));
        assert_eq!(v.to_string(), v.name());
    }
    assert_eq!(GemmVariant::from_name("AMX"), None);
    assert_eq!(best_available(), *available().last().unwrap());
    assert!(available().contains(&GemmVariant::Scalar));
    for v in available() {
        assert!(v.is_available());
        assert_eq!(effective_variant(v), v);
    }
}

#[test]
fn pool_reports_threads_and_installs() {
    let pool = Pool::new(3);
    assert_eq!(pool.threads(), 3);
    assert_eq!(pool.install(rayon::current_num_threads), 3);
    assert_eq!(Pool::new(0).threads(), 1);
    assert!(format!("{pool:?}").contains('3'));
}

#[test]
fn autotune_measures_every_available_variant() {
    let pool = Pool::new(2);
    let res = autotune(&pool, Trans::T, Trans::N, 96, 80, 64);
    let mut seen: Vec<_> = res.iter().map(|r| r.variant).collect();
    seen.sort_by_key(|v| GemmVariant::ALL.iter().position(|x| x == v));
    assert_eq!(seen, available());
    assert!(res.windows(2).all(|w| w[0].gflops >= w[1].gflops));
    assert!(res.iter().all(|r| r.gflops.is_finite() && r.gflops > 0.0));
    // Degenerate shapes do not panic.
    let res = autotune(&pool, Trans::N, Trans::N, 0, 4, 4);
    assert_eq!(res.len(), available().len());
}
