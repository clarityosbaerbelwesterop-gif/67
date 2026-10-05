//! Checkpoints: safetensors (f32, SCP parameter names) + SHA-256 lineage.
//!
//! safetensors layout: u64 LE header size, JSON header
//! `{name: {dtype: "F32", shape: [...], data_offsets: [begin, end]}, "__metadata__": {...}}`,
//! then the raw little-endian tensor bytes. No pickle anywhere.

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

/// SHA-256 (FIPS 180-4), std only.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    Ok(hex(&sha256(&fs::read(path)?)))
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Write tensors in name order. Returns the file's SHA-256.
pub fn save_safetensors(
    path: &Path,
    tensors: &BTreeMap<String, Tensor>,
    metadata: &BTreeMap<String, String>,
) -> io::Result<String> {
    let mut header = Map::new();
    let mut offset = 0usize;
    for (name, t) in tensors {
        assert_eq!(
            t.shape.iter().product::<usize>(),
            t.data.len(),
            "{name}: shape/data mismatch"
        );
        let end = offset + 4 * t.data.len();
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": t.shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    header.insert("__metadata__".into(), json!(metadata));
    let mut hbytes = serde_json::to_vec(&Value::Object(header)).map_err(io::Error::other)?;
    while hbytes.len() % 8 != 0 {
        hbytes.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hbytes.len() + offset);
    out.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&hbytes);
    for t in tensors.values() {
        for v in &t.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    let digest = hex(&sha256(&out));
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, &out)?;
    fs::rename(&tmp, path)?;
    Ok(digest)
}

pub fn load_safetensors(
    path: &Path,
) -> io::Result<(BTreeMap<String, Tensor>, BTreeMap<String, String>)> {
    let bytes = fs::read(path)?;
    let bad = |m: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {m}", path.display()),
        )
    };
    if bytes.len() < 8 {
        return Err(bad("truncated"));
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let body = 8 + hlen;
    if body > bytes.len() {
        return Err(bad("header length out of range"));
    }
    let header: Map<String, Value> =
        serde_json::from_slice(&bytes[8..body]).map_err(|e| bad(&e.to_string()))?;
    let mut tensors = BTreeMap::new();
    let mut meta = BTreeMap::new();
    for (name, v) in header {
        if name == "__metadata__" {
            if let Some(m) = v.as_object() {
                for (k, v) in m {
                    meta.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
                }
            }
            continue;
        }
        if v["dtype"] != "F32" {
            return Err(bad(&format!("{name}: only F32 is supported")));
        }
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .ok_or_else(|| bad("shape"))?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = v["data_offsets"].as_array().ok_or_else(|| bad("offsets"))?;
        let (b, e) = (
            off[0].as_u64().unwrap_or(0) as usize + body,
            off[1].as_u64().unwrap_or(0) as usize + body,
        );
        if e > bytes.len() || b > e || (e - b) != 4 * shape.iter().product::<usize>() {
            return Err(bad(&format!("{name}: data range")));
        }
        let data = bytes[b..e]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        tensors.insert(name, Tensor { shape, data });
    }
    Ok((tensors, meta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let long = vec![b'a'; 1_000_000];
        assert_eq!(
            hex(&sha256(&long)),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn safetensors_roundtrip() {
        let dir = std::env::temp_dir().join(format!("forge-st-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.safetensors");
        let mut t = BTreeMap::new();
        t.insert(
            "a.weight".into(),
            Tensor {
                shape: vec![2, 3],
                data: vec![1.0, -2.0, 3.5, 0.0, f32::MIN_POSITIVE, 7.0],
            },
        );
        t.insert(
            "b".into(),
            Tensor {
                shape: vec![1],
                data: vec![42.0],
            },
        );
        let mut meta = BTreeMap::new();
        meta.insert("step".into(), "12".into());
        let digest = save_safetensors(&p, &t, &meta).unwrap();
        assert_eq!(digest, sha256_file(&p).unwrap());
        let (back, m) = load_safetensors(&p).unwrap();
        assert_eq!(m["step"], "12");
        assert_eq!(back["a.weight"].shape, vec![2, 3]);
        assert_eq!(back["a.weight"].data, t["a.weight"].data);
        let mut raw = fs::read(&p).unwrap();
        raw.truncate(raw.len() - 1);
        fs::write(&p, raw).unwrap();
        assert!(load_safetensors(&p).is_err());
        fs::remove_dir_all(dir).ok();
    }
}
