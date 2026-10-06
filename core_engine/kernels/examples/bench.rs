//! GEMM throughput per variant on transformer-shaped problems.
//!
//! ```text
//! cargo run --release -p forge-kernels --example bench
//! cargo run --release -p forge-kernels --example bench -- --threads 1,4 --variants avx512-f32,amx-bf16 --shapes 2048
//! ```
//!
//! Each cell is `2·m·n·k·batch / t` in GFLOPS, where `t` is the median wall time
//! of several runs after one warm-up run (C = op(A)·op(B), alpha 1, beta 0).

use std::time::Instant;

use forge_kernels::{available, gemm, gemm_batched, GemmVariant, Pool, Trans};

struct Shape {
    label: &'static str,
    ta: Trans,
    tb: Trans,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
}

const SHAPES: [Shape; 6] = [
    Shape { label: "4096x512x512   NN", ta: Trans::N, tb: Trans::N, m: 4096, n: 512, k: 512, batch: 1 },
    Shape { label: "4096x1408x512  NN", ta: Trans::N, tb: Trans::N, m: 4096, n: 1408, k: 512, batch: 1 },
    Shape { label: "4096x512x1408  NN", ta: Trans::N, tb: Trans::N, m: 4096, n: 512, k: 1408, batch: 1 },
    Shape { label: "512x512x4096   TN", ta: Trans::T, tb: Trans::N, m: 512, n: 512, k: 4096, batch: 1 },
    Shape { label: "256x256x64 x32 NT", ta: Trans::N, tb: Trans::T, m: 256, n: 256, k: 64, batch: 32 },
    Shape { label: "2048x2048x2048 NN", ta: Trans::N, tb: Trans::N, m: 2048, n: 2048, k: 2048, batch: 1 },
];

fn data(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn stored(t: Trans, rows: usize, cols: usize) -> (usize, usize) {
    match t {
        Trans::N => (rows, cols),
        Trans::T => (cols, rows),
    }
}

/// Median (or best) seconds over runs fitting `budget` (at least `min_reps`)
/// after a warm-up run.
fn time_secs(mut f: impl FnMut(), budget: f64, min_reps: usize, best: bool) -> f64 {
    let t = Instant::now();
    f();
    let warm = t.elapsed().as_secs_f64();
    let reps = ((budget / warm.max(1e-9)) as usize).clamp(min_reps, 101);
    let mut s: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .collect();
    s.sort_by(f64::total_cmp);
    if best {
        s[0]
    } else {
        s[s.len() / 2]
    }
}

fn gflops(pool: &Pool, v: GemmVariant, s: &Shape, budget: f64, min_reps: usize, best: bool) -> f64 {
    let (ar, ac) = stored(s.ta, s.m, s.k);
    let (br, bc) = stored(s.tb, s.k, s.n);
    let (sa, sb, sc) = (ar * ac, br * bc, s.m * s.n);
    let a = data(sa * s.batch, 1);
    let b = data(sb * s.batch, 2);
    let mut c = vec![0.0f32; sc * s.batch];
    let secs = time_secs(
        || {
            if s.batch == 1 {
                gemm(pool, v, s.ta, s.tb, s.m, s.n, s.k, 1.0, &a, ac, &b, bc, 0.0, &mut c, s.n);
            } else {
                gemm_batched(
                    pool, v, s.ta, s.tb, s.batch, s.m, s.n, s.k, 1.0, &a, ac, sa, &b, bc, sb, 0.0, &mut c, s.n, sc,
                );
            }
        },
        budget,
        min_reps,
        best,
    );
    2.0 * (s.m * s.n * s.k * s.batch) as f64 / secs * 1e-9
}

fn parse_list<T>(arg: Option<String>, f: impl Fn(&str) -> Option<T>) -> Option<Vec<T>> {
    arg.map(|s| s.split(',').map(|x| f(x.trim()).unwrap_or_else(|| panic!("bad value {x:?}"))).collect())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut threads, mut variants, mut shapes, mut budget, mut best) = (None, None, None, 0.5, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => threads = parse_list(args.next(), |x| x.parse::<usize>().ok()),
            "--variants" => variants = parse_list(args.next(), GemmVariant::from_name),
            "--shapes" => shapes = args.next(),
            "--budget" => budget = args.next().and_then(|x| x.parse().ok()).expect("--budget <seconds>"),
            "--best" => best = true,
            "-h" | "--help" => {
                println!("bench [--threads 1,4] [--variants scalar,avx2-fma,...] [--shapes substr,...] [--budget secs] [--best]");
                return;
            }
            other => panic!("unknown argument {other:?}"),
        }
    }
    let all = std::thread::available_parallelism().map_or(1, |n| n.get());
    let threads = threads.unwrap_or_else(|| if all > 1 { vec![1, all] } else { vec![1] });
    let variants = variants.unwrap_or_else(available);
    let shapes: Vec<&Shape> = SHAPES
        .iter()
        .filter(|s| shapes.as_ref().is_none_or(|f: &String| f.split(',').any(|x| s.label.contains(x))))
        .collect();
    println!("available: {:?}", available().iter().map(|v| v.name()).collect::<Vec<_>>());
    for &t in &threads {
        let pool = Pool::new(t);
        let stat = if best { "best" } else { "median" };
        println!("\nthreads = {t}   (GFLOPS, {stat} of runs after warm-up)");
        print!("{:<18}", "shape (m,n,k)");
        for v in &variants {
            print!("{:>12}", v.name());
        }
        println!();
        for s in &shapes {
            print!("{:<18}", s.label);
            for &v in &variants {
                if !v.is_available() {
                    print!("{:>12}", "n/a");
                    continue;
                }
                // Long single runs (scalar on large shapes) get fewer repetitions.
                let g = gflops(&pool, v, s, budget, if v == GemmVariant::Scalar { 1 } else { 3 }, best);
                print!("{g:>12.1}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            println!();
        }
    }
}
