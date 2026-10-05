//! Independent reference implementation shared by the integration tests.
//!
//! Values are computed from the format definitions in f64 (exact for every
//! format here), and rounding is done by searching the sorted value grid —
//! no bit tricks, so it shares no code or technique with the crate.
#![allow(dead_code)]

use forge_formats::Rounding;

/// xorshift64* test RNG.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Approximately standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// A sign-magnitude binary float, described by its fields.
#[derive(Clone, Copy, Debug)]
pub struct RefFmt {
    pub e_bits: u32,
    pub m_bits: u32,
    pub bias: i32,
    /// Largest finite magnitude code.
    pub max_code: u32,
}

pub const BF16: RefFmt = RefFmt { e_bits: 8, m_bits: 7, bias: 127, max_code: 0x7f7f };
pub const FP16: RefFmt = RefFmt { e_bits: 5, m_bits: 10, bias: 15, max_code: 0x7bff };
pub const E4M3: RefFmt = RefFmt { e_bits: 4, m_bits: 3, bias: 7, max_code: 0x7e };
pub const E5M2: RefFmt = RefFmt { e_bits: 5, m_bits: 2, bias: 15, max_code: 0x7b };
pub const E2M1: RefFmt = RefFmt { e_bits: 2, m_bits: 1, bias: 1, max_code: 0x7 };

impl RefFmt {
    /// Value of a finite magnitude code, from the definition.
    pub fn value(&self, code: u32) -> f64 {
        let e = code >> self.m_bits;
        let m = code & ((1 << self.m_bits) - 1);
        let frac = m as f64 / (1u64 << self.m_bits) as f64;
        if e == 0 {
            frac * 2f64.powi(1 - self.bias)
        } else {
            (1.0 + frac) * 2f64.powi(e as i32 - self.bias)
        }
    }

    pub fn grid(&self) -> Grid {
        Grid::new((0..=self.max_code).map(|c| self.value(c)).collect())
    }
}

/// Sorted grid of non-negative representable magnitudes; index = magnitude code.
pub struct Grid {
    pub vals: Vec<f64>,
    /// The value one step past the maximum (first "overflow" point).
    pub beyond: f64,
}

/// Result of reference rounding of a magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefOut {
    Code(u32),
    Overflow,
}

impl Grid {
    pub fn new(vals: Vec<f64>) -> Grid {
        for w in vals.windows(2) {
            assert!(w[0] < w[1], "grid must be strictly increasing");
        }
        let n = vals.len();
        let beyond = 2.0 * vals[n - 1] - vals[n - 2];
        Grid { vals, beyond }
    }

    pub fn max_code(&self) -> u32 {
        self.vals.len() as u32 - 1
    }

    /// Largest code whose value is <= a (a >= 0).
    fn floor_code(&self, a: f64) -> usize {
        self.vals.partition_point(|&v| v <= a) - 1
    }

    /// Round-to-nearest-even of magnitude `a` with unbounded top (overflow
    /// reported when the IEEE rounding with unbounded exponent passes max).
    pub fn nearest_even(&self, a: f64) -> RefOut {
        assert!(a >= 0.0 && a.is_finite());
        let max = self.max_code() as usize;
        let lo = self.floor_code(a);
        let (lo_v, hi_v) = (self.vals[lo], if lo == max { self.beyond } else { self.vals[lo + 1] });
        if a == lo_v {
            return RefOut::Code(lo as u32);
        }
        if lo == max && a >= self.beyond {
            return RefOut::Overflow;
        }
        let mid = (lo_v + hi_v) / 2.0;
        let up = a > mid || (a == mid && lo % 2 == 1);
        if !up {
            RefOut::Code(lo as u32)
        } else if lo == max {
            RefOut::Overflow
        } else {
            RefOut::Code(lo as u32 + 1)
        }
    }

    /// Truncation: never overflows (clamps to max).
    pub fn toward_zero(&self, a: f64) -> u32 {
        self.floor_code(a) as u32
    }

    pub fn round(&self, a: f64, r: Rounding) -> RefOut {
        match r {
            Rounding::NearestEven => self.nearest_even(a),
            Rounding::TowardZero => RefOut::Code(self.toward_zero(a)),
            Rounding::Stochastic(_) => panic!("no deterministic reference for stochastic rounding"),
        }
    }

    /// Neighbouring grid values around `a`: (lo, hi) with lo <= a < hi
    /// (hi = beyond for the last interval).
    pub fn bracket(&self, a: f64) -> (f64, f64) {
        let lo = self.floor_code(a);
        let hi = if lo as u32 == self.max_code() { self.beyond } else { self.vals[lo + 1] };
        (self.vals[lo], hi)
    }
}

/// f32 neighbours.
pub fn next_up(x: f32) -> f32 {
    assert!(x >= 0.0 && x.is_finite());
    f32::from_bits(x.to_bits() + 1)
}
pub fn next_down(x: f32) -> f32 {
    assert!(x > 0.0 && x.is_finite());
    f32::from_bits(x.to_bits() - 1)
}

/// Mean and standard error of a sample.
pub fn mean_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// Assert the stochastic-rounding sample mean is within 4 standard errors of
/// `x` (and exactly `x` if every draw was identical).
pub fn assert_unbiased(what: &str, x: f64, draws: &[f64]) {
    let (mean, se) = mean_se(draws);
    if se == 0.0 {
        assert_eq!(mean, x, "{what}: constant draws must equal the input");
    } else {
        let z = (mean - x) / se;
        assert!(z.abs() <= 4.0, "{what}: x={x} mean={mean} se={se} z={z}");
    }
}
