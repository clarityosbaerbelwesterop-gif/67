//! Encode/decode throughput for every forge-formats codec.
//!
//! Run: `cargo run --release -p forge-formats --example bench [n_elems] [reps] [isa]`
//! where `isa` is `baseline`, `avx2`, `avx512` or `all` (default: the best the
//! CPU supports).
//!
//! Prints the best-of-`reps` single-threaded throughput in million elements
//! per second. Numbers depend on the machine and its load; they are not
//! projections.

use forge_formats::fp8::Kind;
use forge_formats::{
    active_isa, bf16, fake_quant, fp16, fp4, fp8, mx, set_isa_limit, Format, Isa, Rounding,
};
use std::hint::black_box;
use std::time::Instant;

fn best_of(reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn report(name: &str, n: usize, secs: f64) {
    println!(
        "  {name:<34} {:>9.1} Melem/s  {:>8.3} ms",
        n as f64 / secs / 1e6,
        secs * 1e3
    );
}

fn run(x: &[f32], reps: usize) {
    let n = x.len();
    let mut h = vec![0u16; n];
    let mut b = vec![0u8; n];
    let mut y = vec![0f32; n];
    let rne = Rounding::NearestEven;
    let rtz = Rounding::TowardZero;
    let sr = Rounding::Stochastic(0x5eed);

    report(
        "bf16 encode (RNE)",
        n,
        best_of(reps, || bf16::encode_slice(black_box(x), &mut h, rne)),
    );
    report(
        "bf16 encode (stochastic)",
        n,
        best_of(reps, || bf16::encode_slice(black_box(x), &mut h, sr)),
    );
    report(
        "bf16 decode",
        n,
        best_of(reps, || bf16::decode_slice(black_box(&h), &mut y)),
    );
    report(
        "fp16 encode (RNE)",
        n,
        best_of(reps, || fp16::encode_slice(black_box(x), &mut h, rne)),
    );
    report(
        "fp16 encode (toward zero)",
        n,
        best_of(reps, || fp16::encode_slice(black_box(x), &mut h, rtz)),
    );
    report(
        "fp16 encode (stochastic)",
        n,
        best_of(reps, || fp16::encode_slice(black_box(x), &mut h, sr)),
    );
    report(
        "fp16 decode",
        n,
        best_of(reps, || fp16::decode_slice(black_box(&h), &mut y)),
    );
    for kind in [Kind::E4M3, Kind::E5M2] {
        report(
            &format!("fp8 {kind:?} encode (RNE, sat)"),
            n,
            best_of(reps, || {
                fp8::encode_slice(kind, black_box(x), &mut b, rne, true)
            }),
        );
        report(
            &format!("fp8 {kind:?} encode (stochastic)"),
            n,
            best_of(reps, || {
                fp8::encode_slice(kind, black_box(x), &mut b, sr, true)
            }),
        );
        report(
            &format!("fp8 {kind:?} decode"),
            n,
            best_of(reps, || fp8::decode_slice(kind, black_box(&b), &mut y)),
        );
    }
    let mut p = vec![0u8; fp4::packed_len(n)];
    report(
        "fp4 encode packed (RNE)",
        n,
        best_of(reps, || fp4::encode_packed(black_box(x), &mut p, rne)),
    );
    report(
        "fp4 decode packed",
        n,
        best_of(reps, || fp4::decode_packed(black_box(&p), &mut y)),
    );
    for e in mx::Elem::ALL {
        let mut scales = vec![0u8; n.div_ceil(mx::BLOCK)];
        let mut data = vec![0u8; e.data_len(n)];
        report(
            &format!("mx {e:?} quantize (RNE)"),
            n,
            best_of(reps, || {
                mx::quantize_into(black_box(x), e, rne, &mut scales, &mut data)
            }),
        );
        report(
            &format!("mx {e:?} dequantize"),
            n,
            best_of(reps, || {
                mx::dequantize_raw(e, black_box(&scales), &data, &mut y)
            }),
        );
    }
    for f in Format::ALL.into_iter().skip(1) {
        report(
            &format!("fake_quant {} (RNE)", f.name()),
            n,
            best_of(reps, || {
                y.copy_from_slice(x);
                fake_quant(black_box(&mut y), f, rne)
            }),
        );
    }
    black_box((&h, &b, &y, &p));
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(1 << 22, |s| s.parse().expect("n_elems"));
    let reps: usize = args.next().map_or(7, |s| s.parse().expect("reps"));
    let isas: Vec<Isa> = match args.next().as_deref() {
        None => vec![Isa::Avx512],
        Some("baseline") => vec![Isa::Baseline],
        Some("avx2") => vec![Isa::Avx2],
        Some("avx512") => vec![Isa::Avx512],
        Some("all") => vec![Isa::Baseline, Isa::Avx2, Isa::Avx512],
        Some(o) => panic!("unknown isa {o:?} (baseline | avx2 | avx512 | all)"),
    };

    // Deterministic normal-ish data (sum of uniforms), spread over magnitudes.
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let x: Vec<f32> = (0..n)
        .map(|i| {
            let mut acc = 0f32;
            for _ in 0..4 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                acc += (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            }
            acc * (1 << (i % 8)) as f32
        })
        .collect();
    println!("forge-formats bench: n = {n} elements, best of {reps}, single thread");
    for isa in isas {
        set_isa_limit(isa);
        println!("\nISA limit {isa:?} -> dispatching {:?}", active_isa());
        run(&x, reps);
    }
}
