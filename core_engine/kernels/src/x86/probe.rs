//! Component micro-benchmarks (ignored tests; run with
//! `cargo test --release -p forge-kernels probe -- --ignored --nocapture`).

use std::time::Instant;

use crate::driver::{Engine, MatRef};
use crate::tests::Rng;
use crate::Trans;

fn best_of<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// Macro-kernel throughput on packed operands of one `mc × nc × kc` block.
fn kernel_gflops<E: Engine<P = f32>>(mc: usize, nc: usize, kc: usize) -> f64 {
    let mut rng = Rng::new(1);
    let sa = E::a_panel_len(kc);
    let sb = E::b_panel_len(kc);
    let ap: Vec<f32> = (0..sa * mc.div_ceil(E::MR)).map(|_| rng.sym()).collect();
    let bp: Vec<f32> = (0..sb * nc.div_ceil(E::NR)).map(|_| rng.sym()).collect();
    let mut c = vec![0.0f32; mc * nc];
    let secs = best_of(20, || unsafe {
        E::kernel_block(
            mc,
            nc,
            kc,
            ap.as_ptr(),
            bp.as_ptr(),
            c.as_mut_ptr(),
            nc,
            1.0,
            0.0,
        )
    });
    2.0 * (mc * nc * kc) as f64 / secs * 1e-9
}

/// Packing throughput in elements per nanosecond.
fn pack_rate<E: Engine<P = f32>>(t: Trans, a_side: bool, rows: usize, kc: usize) -> f64 {
    let mut rng = Rng::new(2);
    let data: Vec<f32> = (0..rows * kc).map(|_| rng.sym()).collect();
    let (ld, panel, width) = if a_side {
        (E::MR, E::a_panel_len(kc), E::MR)
    } else {
        (E::NR, E::b_panel_len(kc), E::NR)
    };
    let _ = ld;
    let x = match t {
        // op(X) is rows × kc (A side) or kc × rows (B side).
        Trans::N if a_side => MatRef {
            ptr: data.as_ptr(),
            ld: kc,
            trans: t,
        },
        Trans::T if a_side => MatRef {
            ptr: data.as_ptr(),
            ld: rows,
            trans: t,
        },
        Trans::N => MatRef {
            ptr: data.as_ptr(),
            ld: rows,
            trans: t,
        },
        Trans::T => MatRef {
            ptr: data.as_ptr(),
            ld: kc,
            trans: t,
        },
    };
    let panels = rows / width;
    let mut dst = vec![0.0f32; panel * panels];
    let secs = best_of(20, || unsafe {
        for i in 0..panels {
            if a_side {
                E::pack_a(
                    &x,
                    i * width,
                    width,
                    0,
                    kc,
                    kc,
                    dst.as_mut_ptr().add(i * panel),
                );
            } else {
                E::pack_b(
                    &x,
                    0,
                    kc,
                    kc,
                    i * width,
                    width,
                    dst.as_mut_ptr().add(i * panel),
                );
            }
        }
    });
    (panels * width * kc) as f64 / secs * 1e-9
}

fn report<E: Engine<P = f32>>(name: &str) {
    for kc in [128, 192, 256, 384, 512] {
        let mc = E::MC.next_multiple_of(E::MR);
        println!(
            "{name}: kernel_block mc={mc} nc={} kc={kc}: {:.1} GFLOPS",
            E::NB,
            kernel_gflops::<E>(mc, E::NB, kc)
        );
    }
    for t in [Trans::N, Trans::T] {
        let rows_a = E::MR * 24;
        let rows_b = E::NR * 16;
        println!(
            "{name}: pack_a {t:?} {:.2} elem/ns   pack_b {t:?} {:.2} elem/ns",
            pack_rate::<E>(t, true, rows_a, 256),
            pack_rate::<E>(t, false, rows_b, 256)
        );
    }
}

#[test]
#[ignore]
fn probe_avx512_components() {
    if super::avx512::supported() {
        report::<super::avx512::Avx512>("avx512");
    }
}

#[test]
#[ignore]
fn probe_avx2_components() {
    if super::avx2::supported() {
        report::<super::avx2::Avx2>("avx2");
    }
}
