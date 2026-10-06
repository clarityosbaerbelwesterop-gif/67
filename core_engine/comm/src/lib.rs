//! forge-comm: ring all-reduce over TCP. Contract: docs/DESIGN.md §6.
//!
//! Every rank keeps two connections: one to its right neighbour (send) and
//! one from its left neighbour (receive). `all_reduce_sum` is the
//! bandwidth-optimal ring (Patarasuk & Yuan, 2009): a reduce-scatter of
//! `world − 1` steps leaves each rank owning the full sum of one chunk, and an
//! all-gather of `world − 1` steps circulates the owned chunks. Each rank sends
//! 2·(world − 1)/world of the buffer, independent of the number of ranks.
//!
//! Sending and receiving overlap (a scoped sender thread per step), so ring
//! steps never deadlock on full socket buffers. Frames are
//! `b"F67C" u32 seq u32 len u32 crc32 payload` (little endian); a wrong magic,
//! sequence number or CRC is an error, never silently accepted data.
//!
//! Compression applies to the wire only. Partial sums are accumulated in f32;
//! before the all-gather the owner of a chunk replaces it by its
//! encode→decode image, so every rank ends with bit-identical values.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

const MAGIC: [u8; 4] = *b"F67C";
const HELLO: u32 = 0x4F4C_4C45;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    None,
    /// bf16 round-to-nearest-even: 2 bytes per value.
    Bf16,
    /// Per block of `block` values one f32 scale (max|x| / 127) and i8 codes:
    /// (4 + block) bytes per block.
    Int8Block {
        block: usize,
    },
}

impl Compression {
    pub fn parse(s: &str) -> Option<Compression> {
        match s {
            "none" => Some(Compression::None),
            "bf16" => Some(Compression::Bf16),
            "int8" => Some(Compression::Int8Block { block: 256 }),
            _ => s
                .strip_prefix("int8:")
                .and_then(|b| b.parse().ok())
                .filter(|&b: &usize| b > 0)
                .map(|block| Compression::Int8Block { block }),
        }
    }

    fn encode(self, x: &[f32], out: &mut Vec<u8>) {
        out.clear();
        match self {
            Compression::None => x
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            Compression::Bf16 => x
                .iter()
                .for_each(|&v| out.extend_from_slice(&bf16(v).to_le_bytes())),
            Compression::Int8Block { block } => {
                for b in x.chunks(block) {
                    let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let scale = if amax > 0.0 && amax.is_finite() {
                        amax / 127.0
                    } else {
                        0.0
                    };
                    out.extend_from_slice(&scale.to_le_bytes());
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    out.extend(
                        b.iter()
                            .map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8 as u8),
                    );
                }
            }
        }
    }

    /// Decode `bytes` (exactly the encoding of `out.len()` values) and apply `f(out[i], decoded)`.
    fn decode(self, bytes: &[u8], out: &mut [f32], f: impl Fn(&mut f32, f32)) -> io::Result<()> {
        if bytes.len() != self.wire_len(out.len()) {
            return Err(invalid(format!(
                "payload of {} bytes for {} values",
                bytes.len(),
                out.len()
            )));
        }
        match self {
            Compression::None => {
                for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    f(o, f32::from_le_bytes(c.try_into().unwrap()));
                }
            }
            Compression::Bf16 => {
                for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    f(
                        o,
                        f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16),
                    );
                }
            }
            Compression::Int8Block { block } => {
                let mut p = 0;
                for o in out.chunks_mut(block) {
                    let scale = f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
                    p += 4;
                    let len = o.len();
                    for (v, &q) in o.iter_mut().zip(&bytes[p..p + len]) {
                        f(v, q as i8 as f32 * scale);
                    }
                    p += len;
                }
            }
        }
        Ok(())
    }

    pub fn wire_len(self, n: usize) -> usize {
        match self {
            Compression::None => 4 * n,
            Compression::Bf16 => 2 * n,
            Compression::Int8Block { block } => n + 4 * n.div_ceil(block),
        }
    }

    /// The values a receiver reconstructs from `x`.
    pub fn round_trip(self, x: &mut [f32]) {
        if self == Compression::None {
            return;
        }
        let mut buf = Vec::new();
        self.encode(x, &mut buf);
        self.decode(&buf, x, |o, v| *o = v).expect("own encoding");
    }
}

/// f32 → bf16 bits, round to nearest even; NaN stays NaN.
pub fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    (u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

/// CRC-32 (IEEE 802.3, reflected, as zlib's `crc32`).
pub fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *e = c;
        }
        t
    });
    !data.iter().fold(!0u32, |c, &b| {
        t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8)
    })
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CommStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub ops: u64,
    pub last_ms: f64,
}

struct Link {
    stream: TcpStream,
    seq: u32,
}

impl Link {
    fn send(&mut self, payload: &[u8]) -> io::Result<u64> {
        let mut head = [0u8; 16];
        head[..4].copy_from_slice(&MAGIC);
        head[4..8].copy_from_slice(&self.seq.to_le_bytes());
        head[8..12].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        head[12..16].copy_from_slice(&crc32(payload).to_le_bytes());
        self.stream.write_all(&head)?;
        self.stream.write_all(payload)?;
        self.seq = self.seq.wrapping_add(1);
        Ok(16 + payload.len() as u64)
    }

    fn recv(&mut self, payload: &mut Vec<u8>) -> io::Result<u64> {
        let mut head = [0u8; 16];
        self.stream.read_exact(&mut head)?;
        if head[..4] != MAGIC {
            return Err(invalid("bad frame magic"));
        }
        let seq = u32::from_le_bytes(head[4..8].try_into().unwrap());
        if seq != self.seq {
            return Err(invalid(format!("frame {seq}, expected {}", self.seq)));
        }
        let len = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(head[12..16].try_into().unwrap());
        payload.resize(len, 0);
        self.stream.read_exact(payload)?;
        if crc32(payload) != crc {
            return Err(invalid(format!("CRC mismatch in frame {seq}")));
        }
        self.seq = self.seq.wrapping_add(1);
        Ok(16 + len as u64)
    }
}

pub struct Ring {
    pub rank: usize,
    pub world: usize,
    right: Option<Link>,
    left: Option<Link>,
    stats: CommStats,
    tx: Vec<u8>,
    rx: Vec<u8>,
}

impl Ring {
    /// Listen on `listen`, connect to `peers[(rank + 1) % world]`, accept from
    /// the left neighbour. Connecting retries until `timeout`, so ranks may
    /// start in any order. A hello frame checks that the left neighbour is the
    /// expected rank of the same world size.
    pub fn connect(
        rank: usize,
        world: usize,
        listen: SocketAddr,
        peers: &[SocketAddr],
        timeout: Duration,
    ) -> io::Result<Ring> {
        let listener = TcpListener::bind(listen)?;
        Ring::with_listener(rank, world, listener, peers, timeout)
    }

    fn with_listener(
        rank: usize,
        world: usize,
        listener: TcpListener,
        peers: &[SocketAddr],
        timeout: Duration,
    ) -> io::Result<Ring> {
        if world == 0 || rank >= world || peers.len() != world {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "rank {rank} of world {world} with {} peer addresses",
                    peers.len()
                ),
            ));
        }
        let mut ring = Ring {
            rank,
            world,
            right: None,
            left: None,
            stats: CommStats::default(),
            tx: Vec::new(),
            rx: Vec::new(),
        };
        if world == 1 {
            return Ok(ring);
        }
        let deadline = Instant::now() + timeout;
        let target = peers[(rank + 1) % world];
        let right = loop {
            match TcpStream::connect_timeout(&target, Duration::from_millis(500)) {
                Ok(s) => break s,
                Err(e) if Instant::now() >= deadline => {
                    return Err(io::Error::new(
                        e.kind(),
                        format!("connect to {target}: {e}"),
                    ))
                }
                Err(_) => std::thread::sleep(Duration::from_millis(50)),
            }
        };
        right.set_nodelay(true)?;
        let mut right = Link {
            stream: right,
            seq: 0,
        };
        let mut hello = Vec::with_capacity(12);
        for v in [HELLO, rank as u32, world as u32] {
            hello.extend_from_slice(&v.to_le_bytes());
        }
        right.send(&hello)?;

        listener.set_nonblocking(true)?;
        let left = loop {
            match listener.accept() {
                Ok((s, _)) => break s,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "no connection from the left neighbour",
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(e),
            }
        };
        left.set_nonblocking(false)?;
        left.set_nodelay(true)?;
        left.set_read_timeout(Some(timeout.max(Duration::from_secs(1))))?;
        let mut left = Link {
            stream: left,
            seq: 0,
        };
        let mut buf = Vec::new();
        left.recv(&mut buf)?;
        let want = [HELLO, ((rank + world - 1) % world) as u32, world as u32];
        let got: Vec<u32> = buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if got != want {
            return Err(invalid(format!("handshake {got:?}, expected {want:?}")));
        }
        left.stream.set_read_timeout(None)?;
        ring.right = Some(right);
        ring.left = Some(left);
        Ok(ring)
    }

    /// Bound on how long one receive may block (a dead peer turns into an error).
    pub fn set_io_timeout(&mut self, t: Option<Duration>) -> io::Result<()> {
        if let Some(l) = &self.left {
            l.stream.set_read_timeout(t)?;
        }
        if let Some(r) = &self.right {
            r.stream.set_write_timeout(t)?;
        }
        Ok(())
    }

    pub fn stats(&self) -> CommStats {
        self.stats
    }

    /// Chunk `i` of a buffer of length `n` split into `world` near-equal parts.
    fn chunk(&self, n: usize, i: usize) -> std::ops::Range<usize> {
        let (q, r) = (n / self.world, n % self.world);
        let start = i * q + i.min(r);
        start..start + q + usize::from(i < r)
    }

    /// Send `self.tx` to the right while receiving one frame from the left into `self.rx`.
    fn exchange(&mut self) -> io::Result<()> {
        let (right, left) = (self.right.as_mut().unwrap(), self.left.as_mut().unwrap());
        let (tx, rx) = (&self.tx, &mut self.rx);
        let (sent, got) = std::thread::scope(|s| {
            let h = s.spawn(move || right.send(tx));
            let got = left.recv(rx);
            (h.join().expect("sender thread"), got)
        });
        self.stats.bytes_sent += sent?;
        self.stats.bytes_received += got?;
        Ok(())
    }

    /// In-place sum over all ranks (reduce-scatter + all-gather).
    pub fn all_reduce_sum(&mut self, buf: &mut [f32], c: Compression) -> io::Result<()> {
        let t0 = Instant::now();
        let (w, r, n) = (self.world, self.rank, buf.len());
        if w > 1 {
            for s in 0..w - 1 {
                c.encode(&buf[self.chunk(n, (r + w - s) % w)], &mut self.tx);
                self.exchange()?;
                c.decode(
                    &self.rx,
                    &mut buf[self.chunk(n, (r + 2 * w - s - 1) % w)],
                    |o, v| *o += v,
                )?;
            }
            // All-gather forwards the received bytes unchanged, so every rank
            // (the owner included) decodes the very same encoding.
            let own = self.chunk(n, (r + 1) % w);
            c.encode(&buf[own.clone()], &mut self.tx);
            c.decode(&self.tx, &mut buf[own], |o, v| *o = v)?;
            for s in 0..w - 1 {
                self.exchange()?;
                c.decode(
                    &self.rx,
                    &mut buf[self.chunk(n, (r + w - s) % w)],
                    |o, v| *o = v,
                )?;
                std::mem::swap(&mut self.tx, &mut self.rx);
            }
        }
        self.stats.ops += 1;
        self.stats.last_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// Copy `root`'s buffer to every rank, forwarded along the ring (exact f32).
    pub fn broadcast(&mut self, buf: &mut [f32], root: usize) -> io::Result<()> {
        let t0 = Instant::now();
        if self.world > 1 {
            let c = Compression::None;
            if self.rank != root {
                let left = self.left.as_mut().unwrap();
                self.stats.bytes_received += left.recv(&mut self.rx)?;
                c.decode(&self.rx, buf, |o, v| *o = v)?;
            }
            if (self.rank + 1) % self.world != root {
                c.encode(buf, &mut self.tx);
                self.stats.bytes_sent += self.right.as_mut().unwrap().send(&self.tx)?;
            }
        }
        self.stats.ops += 1;
        self.stats.last_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    pub fn barrier(&mut self) -> io::Result<()> {
        let mut one = [0f32; 1];
        self.all_reduce_sum(&mut one, Compression::None)
    }
}

/// Test helper: `world` rings on 127.0.0.1 ephemeral ports, connected in threads.
pub fn local_rings(world: usize) -> io::Result<Vec<Ring>> {
    let listeners = (0..world)
        .map(|_| TcpListener::bind("127.0.0.1:0"))
        .collect::<io::Result<Vec<_>>>()?;
    let peers = listeners
        .iter()
        .map(|l| l.local_addr())
        .collect::<io::Result<Vec<_>>>()?;
    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, l)| {
            let peers = peers.clone();
            std::thread::spawn(move || {
                Ring::with_listener(rank, world, l, &peers, Duration::from_secs(10))
            })
        })
        .collect();
    handles
        .into_iter()
        .map(|h| h.join().expect("ring thread"))
        .collect()
}

#[cfg(test)]
mod tests;
