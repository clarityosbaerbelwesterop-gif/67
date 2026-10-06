//! forge-data: the SCP byte-level BPE tokenizer (tokenizer.json v2) in Rust,
//! so prompts and generations are text end to end without Python.
//! Contract: docs/DESIGN.md §7. The token store and RNG live in forge-train.
//!
//! Encoding equals `scp_model.bpe.BPETokenizer.encode`: the text is split
//! with SCP's GPT pattern under Python-`re` Unicode semantics, every chunk
//! starts as UTF-8 bytes, and the lowest-ranked adjacent pair is merged
//! (all non-overlapping occurrences, left to right) until no pair has a rank.
//! Python's `\w` is `[\p{L}\p{N}_]` and its `\s` is the fixed set of
//! `str.isspace()`; both are spelled out because Rust's Unicode `\w`/`\s`
//! differ on marks, connector punctuation and U+001C..U+001F.

use fancy_regex::Regex;
use std::collections::HashMap;
use std::path::Path;

/// `scp_model.bpe.GPT_SPLIT_PATTERN`, verbatim.
pub const GPT_SPLIT_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+";

/// Python `str.isspace()` characters.
const PY_SPACE: &str = r"\t\n\x0B\x0C\r\x1C-\x1F \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}";

/// The GPT pattern with Python's character classes made explicit.
fn python_semantics_pattern() -> String {
    let w = r"\p{L}\p{N}_";
    let s = PY_SPACE;
    format!(
        r"'(?:[sdmt]|ll|ve|re)| ?[[{w}]--\p{{Nd}}]+| ?\p{{Nd}}+| ?[^{s}{w}]+|[{s}]+(?![^{s}])|[{s}]+"
    )
}

pub struct Tokenizer {
    merges: HashMap<(u32, u32), u32>,
    vocab: Vec<Vec<u8>>,
    specials: Vec<String>,
    split: Option<Regex>,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Tokenizer, String> {
        let p = path.as_ref();
        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        Tokenizer::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Tokenizer, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if v["version"].as_u64() != Some(2) {
            return Err("tokenizer.json: only version 2 is supported".into());
        }
        let specials: Vec<String> = v["specials"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let mut merges: Vec<(u32, u32, u32)> = Vec::new();
        for m in v["merges"]
            .as_array()
            .ok_or("tokenizer.json: merges missing")?
        {
            let t: Vec<u64> = m
                .as_array()
                .ok_or("merge entry")?
                .iter()
                .filter_map(|x| x.as_u64())
                .collect();
            if t.len() != 3 {
                return Err(format!("bad merge entry {m}"));
            }
            merges.push((t[0] as u32, t[1] as u32, t[2] as u32));
        }
        merges.sort_by_key(|m| m.2);
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut map = HashMap::with_capacity(merges.len());
        for (k, &(a, b, idx)) in merges.iter().enumerate() {
            if idx as usize != 256 + k || a >= idx || b >= idx {
                return Err(format!(
                    "merge {k} = ({a}, {b}) -> {idx} is not in SCP order"
                ));
            }
            let mut bytes = vocab[a as usize].clone();
            bytes.extend_from_slice(&vocab[b as usize]);
            vocab.push(bytes);
            map.insert((a, b), idx);
        }
        let split = match v["pattern"].as_str() {
            None => None,
            Some(GPT_SPLIT_PATTERN) => Some(python_semantics_pattern()),
            // Other patterns compile as written (Rust Unicode classes).
            Some(other) => Some(other.to_string()),
        }
        .map(|p| Regex::new(&p).map_err(|e| format!("pattern: {e}")))
        .transpose()?;
        Ok(Tokenizer {
            merges: map,
            vocab,
            specials,
            split,
        })
    }

    /// Bytes tokens + merges + specials.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + self.specials.len()
    }

    fn special(&self, name: &str) -> Option<u32> {
        self.specials
            .iter()
            .position(|s| s == name)
            .map(|i| (self.vocab.len() + i) as u32)
    }

    pub fn bos(&self) -> Option<u32> {
        self.special("<bos>")
    }
    pub fn eos(&self) -> Option<u32> {
        self.special("<eos>")
    }
    pub fn pad(&self) -> Option<u32> {
        self.special("<pad>")
    }

    fn encode_chunk(&self, piece: &str, out: &mut Vec<u32>) {
        let mut ids: Vec<u32> = piece.bytes().map(u32::from).collect();
        while ids.len() >= 2 {
            let best = ids
                .windows(2)
                .filter_map(|w| self.merges.get(&(w[0], w[1])).map(|&r| (r, (w[0], w[1]))))
                .min();
            let Some((new, pair)) = best else { break };
            let mut merged = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
                    merged.push(new);
                    i += 2;
                } else {
                    merged.push(ids[i]);
                    i += 1;
                }
            }
            ids = merged;
        }
        out.extend(ids);
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        match &self.split {
            None => {
                if !text.is_empty() {
                    self.encode_chunk(text, &mut out)
                }
            }
            Some(re) => {
                let mut pos = 0;
                while pos < text.len() {
                    match re.find_from_pos(text, pos) {
                        Ok(Some(m)) if m.end() > m.start() => {
                            self.encode_chunk(m.as_str(), &mut out);
                            pos = m.end();
                        }
                        // findall semantics: unmatched characters are skipped.
                        _ => pos += text[pos..].chars().next().map_or(1, char::len_utf8),
                    }
                }
            }
        }
        out
    }

    /// Specials and unknown ids are skipped; invalid UTF-8 becomes U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter_map(|&i| self.vocab.get(i as usize))
            .flatten()
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> Tokenizer {
        // merges: "a"+"b" -> 256, 256+"c" -> 257, " "+"a" -> 258
        Tokenizer::from_json(&format!(
            r#"{{"version":2,"pattern":{},"specials":["<bos>","<eos>","<pad>"],"merges":[[97,98,256],[256,99,257],[32,97,258]]}}"#,
            serde_json::to_string(GPT_SPLIT_PATTERN).unwrap()
        ))
        .unwrap()
    }

    #[test]
    fn merges_apply_in_rank_order_and_decode_roundtrips() {
        let t = toy();
        assert_eq!(t.vocab_size(), 262);
        assert_eq!(
            (t.bos(), t.eos(), t.pad()),
            (Some(259), Some(260), Some(261))
        );
        assert_eq!(t.encode("abc"), vec![257]);
        assert_eq!(t.encode("ababc"), vec![256, 257]);
        // " ab" is one chunk: " a" (rank 2) loses to "ab" (rank 0).
        assert_eq!(t.encode(" abc"), vec![32, 257]);
        for s in [
            "abc ab  x\n\n  y",
            "héllo wörld 123",
            "日本語 😀\t\r\n",
            "",
            "a'sd 're",
        ] {
            assert_eq!(t.decode(&t.encode(s)), s);
        }
        assert_eq!(t.decode(&[257, 260, 9999]), "abc");
    }

    #[test]
    fn split_follows_python_re_semantics() {
        let re = Regex::new(&python_semantics_pattern()).unwrap();
        let split = |s: &str| -> Vec<String> {
            re.find_iter(s)
                .map(|m| m.unwrap().as_str().to_string())
                .collect()
        };
        // Values produced by Python: re.findall(GPT_SPLIT_PATTERN, s)
        assert_eq!(
            split("Hello world, it's 2024!"),
            ["Hello", " world", ",", " it", "'s", " 2024", "!"]
        );
        assert_eq!(
            split("x  = 1\n\n\ty"),
            ["x", " ", " =", " 1", "\n\n", "\t", "y"]
        );
        assert_eq!(split("foo_bar9 baz"), ["foo_bar", "9", " baz"]);
        // U+001C is whitespace for Python, not for Rust's \s.
        assert_eq!(split("a\u{1c}b"), ["a", "\u{1c}", "b"]);
        // A combining mark is not \w for Python: it is a symbol run.
        assert_eq!(split("e\u{301}x"), ["e", "\u{301}", "x"]);
        // '½' (No) is \w for Python but not \d.
        assert_eq!(split("½3"), ["½", "3"]);
    }

    #[test]
    fn rejects_out_of_order_merges() {
        let bad = r#"{"version":2,"pattern":null,"specials":[],"merges":[[97,98,300]]}"#;
        assert!(Tokenizer::from_json(bad).is_err());
        let v1 = r#"{"version":1,"merges":[]}"#;
        assert!(Tokenizer::from_json(v1).is_err());
    }
}
