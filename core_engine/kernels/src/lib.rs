//! forge-kernels: GEMM engines and the worker pool. Contract: docs/DESIGN.md §3.
//! This file starts with the scalar reference engine; faster engines plug in
//! behind the same `gemm` entry point.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trans {
    N,
    T,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmVariant {
    Scalar,
    Avx2Fma,
    Avx512F32,
    AmxBf16,
}

impl GemmVariant {
    pub fn name(self) -> &'static str {
        match self {
            GemmVariant::Scalar => "scalar",
            GemmVariant::Avx2Fma => "avx2-fma",
            GemmVariant::Avx512F32 => "avx512-f32",
            GemmVariant::AmxBf16 => "amx-bf16",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        [Self::Scalar, Self::Avx2Fma, Self::Avx512F32, Self::AmxBf16]
            .into_iter()
            .find(|v| v.name() == s)
    }
}

pub fn available() -> Vec<GemmVariant> {
    vec![GemmVariant::Scalar]
}

pub fn best_available() -> GemmVariant {
    *available().last().unwrap()
}

pub struct Pool {
    pool: rayon::ThreadPool,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build()
            .expect("thread pool");
        Pool { pool }
    }
    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.pool.install(f)
    }
}

#[inline]
fn at(x: &[f32], ld: usize, t: Trans, r: usize, c: usize) -> f32 {
    match t {
        Trans::N => x[r * ld + c],
        Trans::T => x[c * ld + r],
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gemm(
    _pool: &Pool,
    _v: GemmVariant,
    ta: Trans,
    tb: Trans,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += at(a, lda, ta, i, p) * at(b, ldb, tb, p, j);
            }
            let dst = &mut c[i * ldc + j];
            *dst = if beta == 0.0 { alpha * acc } else { alpha * acc + beta * *dst };
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_batched(
    pool: &Pool,
    v: GemmVariant,
    ta: Trans,
    tb: Trans,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    stride_a: usize,
    b: &[f32],
    ldb: usize,
    stride_b: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
    stride_c: usize,
) {
    for i in 0..batch {
        let (a, b) = (&a[i * stride_a..], &b[i * stride_b..]);
        let c = &mut c[i * stride_c..];
        gemm(pool, v, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

pub struct TuneResult {
    pub variant: GemmVariant,
    pub gflops: f64,
}

pub fn autotune(_pool: &Pool, _ta: Trans, _tb: Trans, _m: usize, _n: usize, _k: usize) -> Vec<TuneResult> {
    vec![TuneResult { variant: GemmVariant::Scalar, gflops: 0.0 }]
}
