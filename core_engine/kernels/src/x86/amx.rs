//! Intel AMX engine: the TMUL unit is a 16×16 systolic array per core that
//! multiplies bf16 tiles into f32 accumulators (`TDPBF16PS`).
//!
//! Micro-tile: 32×32 of C held in four accumulator tiles (tmm0–tmm3), fed by
//! two A tiles (tmm4, tmm5: 16 rows × 32 bf16) and two B tiles (tmm6, tmm7:
//! 16 pair-rows × 16 columns in the VNNI pair-interleaved layout). Inputs are
//! rounded to bf16 (round-to-nearest-even) while packing; accumulation, alpha
//! and beta are f32.
//!
//! Enabling AMX needs three things: CPUID (AMX-TILE, AMX-BF16), the OS
//! enabling the tile state in XCR0 (bits 17, 18), and a per-process
//! `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` permission. A
//! self-test on a known product must pass before the engine is offered.

use std::arch::asm;
use std::arch::x86_64::__cpuid_count;
use std::sync::OnceLock;

use crate::driver::{Engine, MatRef};

const MR: usize = 32;
const NR: usize = 32;
/// bf16 elements per tile row (64 bytes): the depth step of one TDPBF16PS.
const KT: usize = 32;
/// Elements of one 16-row tile.
const TILE: usize = 16 * KT;
const ROW_BYTES: usize = 64;

#[repr(C, align(64))]
struct TileConfig {
    palette: u8,
    start_row: u8,
    reserved: [u8; 14],
    colsb: [u16; 16],
    rows: [u8; 16],
}

static CONFIG: TileConfig = TileConfig {
    palette: 1,
    start_row: 0,
    reserved: [0; 14],
    // tmm0–tmm7: 16 rows of 64 bytes each.
    colsb: [64, 64, 64, 64, 64, 64, 64, 64, 0, 0, 0, 0, 0, 0, 0, 0],
    rows: [16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0],
};

/// Round-to-nearest-even f32 → bf16 bits (NaN stays a quiet NaN).
#[inline(always)]
pub(crate) fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    (u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

fn xcr0() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: XGETBV with ECX = 0 is valid whenever OSXSAVE is set, which is
    // checked by the caller through CPUID leaf 1 ECX bit 27.
    unsafe { asm!("xgetbv", in("ecx") 0u32, out("eax") lo, out("edx") hi, options(nomem, nostack)) };
    ((hi as u64) << 32) | lo as u64
}

fn request_permission() -> bool {
    const ARCH_REQ_XCOMP_PERM: libc::c_long = 0x1023;
    const XFEATURE_XTILEDATA: libc::c_long = 18;
    // SAFETY: plain syscall with integer arguments; failure is reported via the return value.
    unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0 }
}

fn detect() -> bool {
    // SAFETY: CPUID is available on every x86-64 processor.
    #[allow(unused_unsafe)]
    let (l1, l7) = unsafe { (__cpuid_count(1, 0), __cpuid_count(7, 0)) };
    let osxsave = l1.ecx & (1 << 27) != 0;
    let amx_bf16 = l7.edx & (1 << 22) != 0;
    let amx_tile = l7.edx & (1 << 24) != 0;
    if !(osxsave && amx_bf16 && amx_tile) {
        return false;
    }
    let tile_state = (1 << 17) | (1 << 18);
    if xcr0() & tile_state != tile_state || !request_permission() {
        return false;
    }
    self_test()
}

/// 32×32×64 product of small integers (exact in bf16 and f32) against a scalar reference.
fn self_test() -> bool {
    let (m, n, k) = (32usize, 32usize, 64usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i * 3) % 7) as f32 - 3.0).collect();
    let am = MatRef { ptr: a.as_ptr(), ld: k, trans: crate::Trans::N };
    let bm = MatRef { ptr: b.as_ptr(), ld: n, trans: crate::Trans::N };
    let mut ap = vec![0u16; Amx::a_panel_len(k)];
    let mut bp = vec![0u16; Amx::b_panel_len(k)];
    let mut c = vec![f32::NAN; m * n];
    // SAFETY: buffers are sized by the engine's own panel formulas; the
    // configuration is set and released on this thread.
    unsafe {
        Amx::pack_a(&am, 0, m, 0, k, k, ap.as_mut_ptr());
        Amx::pack_b(&bm, 0, k, k, 0, n, bp.as_mut_ptr());
        Amx::begin();
        Amx::kernel_block(m, n, k, ap.as_ptr(), bp.as_ptr(), c.as_mut_ptr(), n, 1.0, 0.0);
        Amx::end();
    }
    (0..m).all(|i| {
        (0..n).all(|j| {
            let r: f32 = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
            c[i * n + j] == r
        })
    })
}

pub(crate) fn supported() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(detect)
}

pub(crate) struct Amx;

impl Engine for Amx {
    type P = u16;
    const MR: usize = MR;
    const NR: usize = NR;
    const KU: usize = KT;
    const KC: usize = 512;
    const MC: usize = 256;
    const NB: usize = 256;
    const NC: usize = 4096;
    const TASK_FLOPS: f64 = 8.0e6;

    /// Panel: per 32-deep block, 32 rows × 32 bf16 (tile 0 = rows 0..16, tile 1 = rows 16..32).
    unsafe fn pack_a(a: &MatRef, i0: usize, mr: usize, p0: usize, kc: usize, kc_pad: usize, dst: *mut u16) {
        for kb in 0..kc_pad / KT {
            let block = dst.add(kb * MR * KT);
            for i in 0..MR {
                let row = block.add(i * KT);
                for kk in 0..KT {
                    let p = kb * KT + kk;
                    *row.add(kk) = if i < mr && p < kc { bf16(a.at(i0 + i, p0 + p)) } else { 0 };
                }
            }
        }
    }

    /// Panel: per 32-deep block, two VNNI tiles (columns 0..16 and 16..32), each
    /// 16 pair-rows of [b(2r, j), b(2r+1, j)] for 16 columns j.
    unsafe fn pack_b(b: &MatRef, p0: usize, kc: usize, kc_pad: usize, j0: usize, nr: usize, dst: *mut u16) {
        for kb in 0..kc_pad / KT {
            for t in 0..2 {
                let tile = dst.add(kb * 2 * TILE + t * TILE);
                for r in 0..KT / 2 {
                    let row = tile.add(r * KT);
                    for jj in 0..16 {
                        let j = t * 16 + jj;
                        for e in 0..2 {
                            let p = kb * KT + 2 * r + e;
                            *row.add(2 * jj + e) = if j < nr && p < kc { bf16(b.at(p0 + p, j0 + j)) } else { 0 };
                        }
                    }
                }
            }
        }
    }

    unsafe fn kernel_block(
        mc: usize,
        nc: usize,
        kc_pad: usize,
        ap: *const u16,
        bp: *const u16,
        c: *mut f32,
        ldc: usize,
        alpha: f32,
        beta: f32,
    ) {
        block(micro, mc, nc, kc_pad, ap, bp, c, ldc, alpha, beta);
    }

    unsafe fn begin() {
        asm!("ldtilecfg [{0}]", in(reg) &CONFIG as *const TileConfig, options(nostack, readonly));
    }

    unsafe fn end() {
        asm!("tilerelease", options(nostack, nomem));
    }
}

type Micro = unsafe fn(*const u16, *const u16, usize, *mut f32);

/// Walks the packed panels in 32×32 micro-tiles and applies the epilogue
/// `C = alpha·acc + beta·C` (C is not read when beta is 0).
#[allow(clippy::too_many_arguments)]
unsafe fn block(
    micro: Micro,
    mc: usize,
    nc: usize,
    kc_pad: usize,
    ap: *const u16,
    bp: *const u16,
    c: *mut f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) {
    let (sa, sb) = (Amx::a_panel_len(kc_pad), Amx::b_panel_len(kc_pad));
    let mut acc = [0f32; MR * NR];
    for ir in (0..mc).step_by(MR) {
        let a = ap.add(ir / MR * sa);
        for jr in (0..nc).step_by(NR) {
            let b = bp.add(jr / NR * sb);
            micro(a, b, kc_pad / KT, acc.as_mut_ptr());
            let (mr, nr) = (MR.min(mc - ir), NR.min(nc - jr));
            for i in 0..mr {
                let crow = c.add((ir + i) * ldc + jr);
                let arow = &acc[i * NR..i * NR + nr];
                if beta == 0.0 {
                    for (j, v) in arow.iter().enumerate() {
                        *crow.add(j) = alpha * v;
                    }
                } else {
                    for (j, v) in arow.iter().enumerate() {
                        *crow.add(j) = alpha * v + beta * *crow.add(j);
                    }
                }
            }
        }
    }
}

/// One 32×32 output tile over `blocks` depth blocks of 32, written row-major to `out` (stride NR).
///
/// # Safety
/// Tiles must be configured on this thread (`Amx::begin`); `a`/`b` point at
/// packed panels of `blocks` depth blocks; `out` holds MR×NR f32.
unsafe fn micro(a: *const u16, b: *const u16, blocks: usize, out: *mut f32) {
    let stride = ROW_BYTES;
    asm!("tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3", options(nostack, nomem));
    for kb in 0..blocks {
        let (a0, a1) = (a.add(kb * MR * KT), a.add(kb * MR * KT + TILE));
        let (b0, b1) = (b.add(kb * 2 * TILE), b.add(kb * 2 * TILE + TILE));
        asm!(
            "tileloadd tmm4, [{a0} + {s}*1]",
            "tileloadd tmm5, [{a1} + {s}*1]",
            "tileloadd tmm6, [{b0} + {s}*1]",
            "tileloadd tmm7, [{b1} + {s}*1]",
            "tdpbf16ps tmm0, tmm4, tmm6",
            "tdpbf16ps tmm1, tmm4, tmm7",
            "tdpbf16ps tmm2, tmm5, tmm6",
            "tdpbf16ps tmm3, tmm5, tmm7",
            a0 = in(reg) a0, a1 = in(reg) a1, b0 = in(reg) b0, b1 = in(reg) b1, s = in(reg) stride,
            options(nostack, readonly),
        );
    }
    // C tiles: tmm0 rows 0..16 cols 0..16, tmm1 cols 16..32, tmm2/tmm3 rows 16..32.
    let cs = NR * 4;
    asm!(
        "tilestored [{c0} + {s}*1], tmm0",
        "tilestored [{c1} + {s}*1], tmm1",
        "tilestored [{c2} + {s}*1], tmm2",
        "tilestored [{c3} + {s}*1], tmm3",
        c0 = in(reg) out, c1 = in(reg) out.add(16), c2 = in(reg) out.add(16 * NR), c3 = in(reg) out.add(16 * NR + 16),
        s = in(reg) cs,
        options(nostack),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::Problem;
    use crate::tests::Rng;
    use crate::Trans;

    /// TDPBF16PS on memory, as specified in the Intel SDM: tile rows are
    /// loaded with a 64-byte stride; for every row m, dword k and column n,
    /// C[m][n] += A[m].bf16[2k]·B[k].bf16[2n] + A[m].bf16[2k+1]·B[k].bf16[2n+1].
    /// Runs the exact panel layout the hardware path consumes, so the packing
    /// and tile addressing are verified on hosts without AMX.
    unsafe fn emulated(a: *const u16, b: *const u16, blocks: usize, out: *mut f32) {
        let f = |x: u16| f32::from_bits((x as u32) << 16);
        let mut acc = [[0f32; NR]; MR];
        for kb in 0..blocks {
            for (t, u) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let at = a.add(kb * MR * KT + t * TILE);
                let bt = b.add(kb * 2 * TILE + u * TILE);
                for m in 0..16 {
                    for k in 0..KT / 2 {
                        for n in 0..16 {
                            let c = &mut acc[t * 16 + m][u * 16 + n];
                            *c += f(*at.add(m * KT + 2 * k)) * f(*bt.add(k * KT + 2 * n));
                            *c += f(*at.add(m * KT + 2 * k + 1)) * f(*bt.add(k * KT + 2 * n + 1));
                        }
                    }
                }
            }
        }
        for (i, row) in acc.iter().enumerate() {
            std::ptr::copy_nonoverlapping(row.as_ptr(), out.add(i * NR), NR);
        }
    }

    /// The AMX engine with the tile unit replaced by the emulator.
    struct Emulated;

    impl Engine for Emulated {
        type P = u16;
        const MR: usize = MR;
        const NR: usize = NR;
        const KU: usize = KT;
        const KC: usize = 64;
        const MC: usize = 64;
        const NB: usize = 64;
        const NC: usize = 96;
        const TASK_FLOPS: f64 = 1.0e4;

        unsafe fn pack_a(a: &MatRef, i0: usize, mr: usize, p0: usize, kc: usize, kc_pad: usize, dst: *mut u16) {
            Amx::pack_a(a, i0, mr, p0, kc, kc_pad, dst)
        }
        unsafe fn pack_b(b: &MatRef, p0: usize, kc: usize, kc_pad: usize, j0: usize, nr: usize, dst: *mut u16) {
            Amx::pack_b(b, p0, kc, kc_pad, j0, nr, dst)
        }
        #[allow(clippy::too_many_arguments)]
        unsafe fn kernel_block(
            mc: usize,
            nc: usize,
            kc_pad: usize,
            ap: *const u16,
            bp: *const u16,
            c: *mut f32,
            ldc: usize,
            alpha: f32,
            beta: f32,
        ) {
            block(emulated, mc, nc, kc_pad, ap, bp, c, ldc, alpha, beta)
        }
    }

    fn round(x: f32) -> f64 {
        f32::from_bits((bf16(x) as u32) << 16) as f64
    }

    #[test]
    fn bf16_rounding_is_nearest_even() {
        assert_eq!(bf16(1.0), 0x3F80);
        // Halfway between 1.0 and the next bf16 (1 + 2^-7): ties to even (1.0).
        assert_eq!(bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        // Halfway with an odd lower neighbour rounds up.
        assert_eq!(bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(bf16(-2.5), 0xC020);
        assert_eq!(bf16(f32::INFINITY), 0x7F80);
        assert_eq!(bf16(f32::MAX), 0x7F80, "overflow rounds to infinity");
        assert!(f32::from_bits((bf16(f32::NAN) as u32) << 16).is_nan());
        assert!(f32::from_bits((bf16(f32::from_bits(0x7F80_0001)) as u32) << 16).is_nan());
    }

    /// Full driver (k/n/m blocking, threads, edge tiles, both transposes,
    /// beta) through the AMX panel layout against an f64 reference on
    /// bf16-rounded inputs.
    #[test]
    fn driver_through_tile_layout_matches_reference() {
        let mut rng = Rng::new(31);
        let shapes = [(1, 1, 1), (32, 32, 32), (33, 31, 65), (70, 100, 130), (5, 97, 3), (64, 64, 200)];
        for &(m, n, k) in &shapes {
            for (ta, tb) in [(Trans::N, Trans::N), (Trans::T, Trans::N), (Trans::N, Trans::T), (Trans::T, Trans::T)] {
                for &(alpha, beta) in &[(1.0f32, 0.0f32), (0.5, 1.0), (-1.25, 0.75)] {
                    let a: Vec<f32> = (0..m * k).map(|_| rng.sym()).collect();
                    let b: Vec<f32> = (0..k * n).map(|_| rng.sym()).collect();
                    let c0: Vec<f32> = (0..m * n).map(|_| rng.sym()).collect();
                    let (lda, ldb) = (if ta == Trans::N { k } else { m }, if tb == Trans::N { n } else { k });
                    let at = |i: usize, p: usize| if ta == Trans::N { a[i * k + p] } else { a[p * m + i] };
                    let bt = |p: usize, j: usize| if tb == Trans::N { b[p * n + j] } else { b[j * k + p] };
                    let mut c = c0.clone();
                    let p = Problem {
                        m,
                        n,
                        k,
                        alpha,
                        beta,
                        a: MatRef { ptr: a.as_ptr(), ld: lda, trans: ta },
                        b: MatRef { ptr: b.as_ptr(), ld: ldb, trans: tb },
                        c: c.as_mut_ptr(),
                        ldc: n,
                    };
                    crate::driver::run::<Emulated>(&p, 3);
                    for i in 0..m {
                        for j in 0..n {
                            let dot: f64 = (0..k).map(|q| round(at(i, q)) * round(bt(q, j))).sum();
                            let want = alpha as f64 * dot + beta as f64 * c0[i * n + j] as f64;
                            let tol = 1e-5 * (1.0 + (k as f64).sqrt());
                            let got = c[i * n + j] as f64;
                            assert!(
                                (got - want).abs() <= tol,
                                "{m}x{n}x{k} {ta:?}{tb:?} a={alpha} b={beta} [{i},{j}]: {got} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// On hosts with AMX the hardware path must agree with the emulator bit for bit
    /// on a product whose partial sums are exact.
    #[test]
    fn hardware_matches_emulator_when_available() {
        if !supported() {
            return;
        }
        assert!(self_test());
    }
}
