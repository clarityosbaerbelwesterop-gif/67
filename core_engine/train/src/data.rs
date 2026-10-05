//! Batch sources: SCP token stores and a synthetic stream.

use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Val,
}

pub trait Batches: Send {
    fn vocab_size(&self) -> usize;
    fn len(&self, split: Split) -> usize;
    /// (inputs, targets), each `b*t` token ids. Windows start uniformly in
    /// [0, N-t-1]; x = w[0..t], y = w[1..t+1] (SCP `data_pipeline` semantics).
    fn batch(&self, rng: &mut Rng, split: Split, b: usize, t: usize) -> (Vec<u32>, Vec<u32>);
}

/// Sample windows from a token slice.
pub fn sample_windows(
    tokens: &dyn Fn(usize) -> u32,
    n: usize,
    rng: &mut Rng,
    b: usize,
    t: usize,
) -> (Vec<u32>, Vec<u32>) {
    assert!(n > t + 1, "split has {n} tokens, need more than {}", t + 1);
    let mut x = Vec::with_capacity(b * t);
    let mut y = Vec::with_capacity(b * t);
    for _ in 0..b {
        let start = rng.below((n - t - 1) as u64) as usize;
        for j in 0..t {
            x.push(tokens(start + j));
            y.push(tokens(start + j + 1));
        }
    }
    (x, y)
}

/// Deterministic toy stream: token i = (i mod period) mod vocab. A correct
/// trainer drives its loss towards zero; used by tests and smoke runs.
pub struct Synthetic {
    pub vocab: usize,
    pub period: usize,
    pub n: usize,
}

impl Batches for Synthetic {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn len(&self, _: Split) -> usize {
        self.n
    }
    fn batch(&self, rng: &mut Rng, _: Split, b: usize, t: usize) -> (Vec<u32>, Vec<u32>) {
        let (p, v) = (self.period, self.vocab);
        sample_windows(&|i| ((i % p) % v) as u32, self.n, rng, b, t)
    }
}

/// SCP CorpusStore split files (`train.meta.json` + headerless LE uint16/uint32 bins).
/// Format contract: docs/DESIGN.md §7 and SCP `data_pipeline.build_token_file`.
pub struct Store {
    vocab: usize,
    wide: bool,
    train: Vec<u8>,
    val: Vec<u8>,
}

impl Store {
    pub fn open(meta: &str) -> Result<Store, String> {
        let path = std::path::Path::new(meta);
        let text = std::fs::read_to_string(path).map_err(|e| format!("{meta}: {e}"))?;
        let m: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{meta}: {e}"))?;
        let dir = path.parent().unwrap_or(std::path::Path::new("."));
        let wide = match m["dtype"].as_str().unwrap_or("uint16") {
            "uint16" => false,
            "uint32" => true,
            other => return Err(format!("{meta}: unsupported dtype {other}")),
        };
        let vocab = m["vocab_size"]
            .as_u64()
            .ok_or(format!("{meta}: vocab_size missing"))? as usize;
        let read = |key: &str| -> Result<Vec<u8>, String> {
            let name = m[key].as_str().ok_or(format!("{meta}: {key} missing"))?;
            let p = dir.join(
                std::path::Path::new(name)
                    .file_name()
                    .ok_or(format!("{meta}: bad {key}"))?,
            );
            std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
        };
        let (train, val) = (read("bin")?, read("val_bin")?);
        let width = if wide { 4 } else { 2 };
        for (k, b, want) in [
            ("tokens", &train, &m["tokens"]),
            ("val_tokens", &val, &m["val_tokens"]),
        ] {
            if b.len() % width != 0 || want.as_u64().is_some_and(|w| w as usize != b.len() / width)
            {
                return Err(format!("{meta}: {k} does not match the bin size"));
            }
        }
        Ok(Store {
            vocab,
            wide,
            train,
            val,
        })
    }

    fn token(&self, split: Split, i: usize) -> u32 {
        let b = match split {
            Split::Train => &self.train,
            Split::Val => &self.val,
        };
        if self.wide {
            u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())
        } else {
            u16::from_le_bytes(b[2 * i..2 * i + 2].try_into().unwrap()) as u32
        }
    }
}

impl Batches for Store {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn len(&self, split: Split) -> usize {
        let w = if self.wide { 4 } else { 2 };
        match split {
            Split::Train => self.train.len() / w,
            Split::Val => self.val.len() / w,
        }
    }
    fn batch(&self, rng: &mut Rng, split: Split, b: usize, t: usize) -> (Vec<u32>, Vec<u32>) {
        let n = self.len(split);
        sample_windows(&|i| self.token(split, i), n, rng, b, t)
    }
}

pub fn open(spec: &str, vocab: usize) -> Result<Box<dyn Batches>, String> {
    if let Some(p) = spec.strip_prefix("synthetic:") {
        let period: usize = p.parse().map_err(|_| format!("bad synthetic period {p}"))?;
        return Ok(Box::new(Synthetic {
            vocab,
            period,
            n: 1 << 20,
        }));
    }
    Ok(Box::new(Store::open(spec)?))
}
