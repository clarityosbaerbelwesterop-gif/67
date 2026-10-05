//! forge-kernels: GEMM engines and the worker pool. Contract: docs/DESIGN.md §3.
//!
//! All engines share one blocked driver (see `driver.rs`): B is packed once per
//! depth block into a shared slab of column panels, output units are computed
//! in parallel on the [`Pool`], each packing its own rows of A into a
//! thread-local buffer and running the engine's register-blocked micro-kernel.
//!
//! | variant      | micro-tile | arithmetic                                      |
//! |--------------|------------|-------------------------------------------------|
//! | `Scalar`     | 4×8        | portable Rust, f32 multiply-add                  |
//! | `Avx2Fma`    | 6×16       | AVX2 + FMA3, f32                                |
//! | `Avx512F32`  | 14×32      | AVX-512F, f32 (masked edges)                    |
//! | `AmxBf16`    | 32×32      | AMX TMUL: inputs rounded to bf16 (RNE), f32 accumulation |
//!
//! Variants are detected at run time; requesting one that is unavailable runs
//! the best available variant ranked below it ([`effective_variant`]). Results
//! of a variant are bitwise identical for every pool size.

use std::fmt;

mod buffer;
mod detect;
mod driver;
mod scalar;
#[cfg(test)]
mod tests;
mod tune;
#[cfg(target_arch = "x86_64")]
mod x86;

pub use tune::autotune;

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
    /// Every variant, in ascending order of preference.
    pub const ALL: [GemmVariant; 4] = [
        GemmVariant::Scalar,
        GemmVariant::Avx2Fma,
        GemmVariant::Avx512F32,
        GemmVariant::AmxBf16,
    ];

    pub fn name(self) -> &'static str {
        match self {
            GemmVariant::Scalar => "scalar",
            GemmVariant::Avx2Fma => "avx2-fma",
            GemmVariant::Avx512F32 => "avx512-f32",
            GemmVariant::AmxBf16 => "amx-bf16",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.name() == s)
    }

    /// Whether this variant can run on this machine (see [`available`]).
    pub fn is_available(self) -> bool {
        detect::available().contains(&self)
    }
}

impl fmt::Display for GemmVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Variants usable on this machine, in ascending order of preference
/// (`Scalar` first, the best last). Detection runs once per process:
/// `Avx2Fma` needs AVX2 and FMA, `Avx512F32` AVX-512F, and `AmxBf16` needs CPUID
/// AMX-TILE + AMX-BF16 (plus AVX-512F/BW for packing), a granted XTILEDATA
/// permission (`arch_prctl(ARCH_REQ_XCOMP_PERM)`) and a passing tile self-test.
pub fn available() -> Vec<GemmVariant> {
    detect::available().to_vec()
}

/// The preferred available variant: AmxBf16 > Avx512F32 > Avx2Fma > Scalar.
pub fn best_available() -> GemmVariant {
    *detect::available()
        .last()
        .expect("Scalar is always available")
}

/// The variant [`gemm`] actually runs when `v` is requested: `v` if available,
/// otherwise the best available variant ranked below it (never one above).
pub fn effective_variant(v: GemmVariant) -> GemmVariant {
    detect::resolve(v)
}

/// Rounds `x` to the nearest bf16 value (ties to even, NaN stays a quiet NaN)
/// and widens it back to f32: exactly the input rounding the `AmxBf16`
/// variant applies to A and B before its f32 accumulation.
pub fn bf16_round(x: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(x)) << 16)
}

/// bf16 bit pattern of `x` (round to nearest even, NaN quieted).
pub(crate) fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    if u & 0x7FFF_FFFF > 0x7F80_0000 {
        return ((u >> 16) | 0x0040) as u16;
    }
    (u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

/// A worker pool. GEMM calls run inside it; a new pool with a different thread
/// count changes how many cores subsequent calls use.
pub struct Pool {
    pool: rayon::ThreadPool,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("forge-gemm-{i}"))
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

impl fmt::Debug for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pool")
            .field("threads", &self.threads())
            .finish()
    }
}

/// Runs `$body` with the type alias `$E` bound to the engine of variant `$v`.
macro_rules! with_engine {
    ($v:expr, $E:ident => $body:expr) => {
        match $v {
            #[cfg(target_arch = "x86_64")]
            GemmVariant::Avx2Fma => {
                type $E = crate::x86::avx2::Avx2;
                $body
            }
            #[cfg(target_arch = "x86_64")]
            GemmVariant::Avx512F32 => {
                type $E = crate::x86::avx512::Avx512;
                $body
            }
            #[cfg(target_arch = "x86_64")]
            GemmVariant::AmxBf16 => {
                type $E = crate::x86::amx::Amx;
                $body
            }
            #[allow(unreachable_patterns)]
            _ => {
                type $E = crate::scalar::Scalar;
                $body
            }
        }
    };
}

fn check_len(what: &str, len: usize, need: usize) {
    assert!(
        len >= need,
        "gemm: {what} has {len} elements but the arguments require {need}"
    );
}

#[allow(clippy::too_many_arguments)]
fn problem(
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
) -> driver::Problem {
    driver::Problem {
        m,
        n,
        k,
        alpha,
        beta,
        a: driver::MatRef {
            ptr: a.as_ptr(),
            ld: lda,
            trans: ta,
        },
        b: driver::MatRef {
            ptr: b.as_ptr(),
            ld: ldb,
            trans: tb,
        },
        c: c.as_mut_ptr(),
        ldc,
    }
}

/// Row-major. C[m×n] = alpha·op(A)·op(B) + beta·C, op(A) is m×k, op(B) is k×n.
/// lda/ldb/ldc are row strides of the stored (untransposed) matrices.
/// beta == 0 must ignore (overwrite) existing C, including NaN.
///
/// Elements of C outside the logical `m × n` region (row padding up to `ldc`)
/// are never touched. As in BLAS, A and B are not referenced (and may be empty)
/// when `k == 0` or `alpha == 0`; then `C = beta·C`.
///
/// # Panics
/// When a slice is shorter than its operand's extent, or a row stride is
/// smaller than the stored row length of a matrix with more than one row.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    pool: &Pool,
    v: GemmVariant,
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
    let (ea, eb, ec) = driver::extents(ta, tb, m, n, k, alpha, lda, ldb, ldc);
    check_len("A", a.len(), ea);
    check_len("B", b.len(), eb);
    check_len("C", c.len(), ec);
    if m == 0 || n == 0 {
        return;
    }
    let p = problem(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    if !p.reads_ab() {
        // SAFETY: C was validated above.
        unsafe { driver::scale_c(&p) };
        return;
    }
    let v = detect::resolve(v);
    pool.install(|| {
        let threads = rayon::current_num_threads();
        with_engine!(v, E => driver::run::<E>(&p, threads))
    });
}

/// Elements spanned by `batch` items of extent `ext` spaced `stride` apart.
fn batch_extent(what: &str, batch: usize, stride: usize, ext: usize) -> usize {
    if batch == 0 || ext == 0 {
        return 0;
    }
    (batch - 1)
        .checked_mul(stride)
        .and_then(|x| x.checked_add(ext))
        .unwrap_or_else(|| panic!("gemm_batched: {what} batch extent overflows usize"))
}

/// Strided batch of independent GEMMs (attention heads): item `i` uses
/// `a[i*stride_a..]`, `b[i*stride_b..]` and `c[i*stride_c..]` with the
/// semantics of [`gemm`].
///
/// Small items are computed in parallel across the batch, large ones one after
/// the other with parallelism inside each item. Strides of 0 for A or B
/// broadcast one operand to every item. If the C items can overlap
/// (`stride_c` smaller than one item's extent), items are applied in order,
/// exactly like a loop of [`gemm`] calls.
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
    let (ea, eb, ec) = driver::extents(ta, tb, m, n, k, alpha, lda, ldb, ldc);
    check_len("A", a.len(), batch_extent("A", batch, stride_a, ea));
    check_len("B", b.len(), batch_extent("B", batch, stride_b, eb));
    check_len("C", c.len(), batch_extent("C", batch, stride_c, ec));
    if batch == 0 || m == 0 || n == 0 {
        return;
    }
    let p = problem(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    if !p.reads_ab() {
        for i in 0..batch {
            // SAFETY: the batch extent of C was validated above.
            unsafe { driver::scale_c(&p.item(i, stride_a, stride_b, stride_c)) };
        }
        return;
    }
    let c_disjoint = batch == 1 || stride_c >= ec;
    let v = detect::resolve(v);
    pool.install(|| {
        let threads = rayon::current_num_threads();
        with_engine!(v, E => driver::run_batched::<E>(
            &p, batch, stride_a, stride_b, stride_c, c_disjoint, threads
        ))
    });
}

/// One autotune measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TuneResult {
    pub variant: GemmVariant,
    pub gflops: f64,
}
