//! Autotuning: time every available variant on a shape.

use std::time::Instant;

use crate::driver::stored_dims;
use crate::{available, gemm, Pool, Trans, TuneResult};

/// Target wall time spent measuring one variant (after the warm-up run).
const BUDGET_SECS: f64 = 0.3;
/// A warm-up run slower than this is used as the only sample.
const SINGLE_SAMPLE_SECS: f64 = 1.0;
const MAX_REPS: usize = 51;

/// Wall-clock seconds of `f`: one warm-up run, then the median of as many runs
/// as fit the time budget (at least one, at most `MAX_REPS`).
pub(crate) fn median_secs(mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    f();
    let warm = t.elapsed().as_secs_f64();
    if warm >= SINGLE_SAMPLE_SECS {
        return warm;
    }
    let reps = ((BUDGET_SECS / warm.max(1e-9)).round() as usize).clamp(1, MAX_REPS);
    let mut samples: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        0.5 * (samples[mid - 1] + samples[mid])
    }
}

/// Deterministic test data in [-1, 1).
pub(crate) fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) * (2.0 / (1u64 << 24) as f32) - 1.0
        })
        .collect()
}

/// Measure every available variant on this shape; sorted fastest first.
///
/// Each variant gets a warm-up run and then the median of several timed runs
/// of `C = op(A)·op(B)` (alpha 1, beta 0) on dense operands in the pool. Ties
/// keep the order of [`available`].
pub fn autotune(
    pool: &Pool,
    ta: Trans,
    tb: Trans,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<TuneResult> {
    let (ar, ac) = stored_dims(ta, m, k);
    let (br, bc) = stored_dims(tb, k, n);
    let a = pseudo_random(ar * ac, 1);
    let b = pseudo_random(br * bc, 2);
    let mut c = vec![0.0f32; m * n];
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let mut results: Vec<TuneResult> = available()
        .into_iter()
        .map(|variant| {
            let secs = median_secs(|| {
                gemm(
                    pool, variant, ta, tb, m, n, k, 1.0, &a, ac, &b, bc, 0.0, &mut c, n,
                )
            });
            let gflops = if flops > 0.0 && secs > 0.0 {
                flops / secs * 1e-9
            } else {
                0.0
            };
            TuneResult { variant, gflops }
        })
        .collect();
    results.sort_by(|x, y| y.gflops.total_cmp(&x.gflops));
    results
}
