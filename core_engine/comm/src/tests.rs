use super::*;

fn run<T: Send + 'static>(
    rings: Vec<Ring>,
    f: impl Fn(&mut Ring) -> T + Send + Sync + Clone + 'static,
) -> Vec<(Ring, T)> {
    let hs: Vec<_> = rings
        .into_iter()
        .map(|mut r| {
            let f = f.clone();
            std::thread::spawn(move || {
                let t = f(&mut r);
                (r, t)
            })
        })
        .collect();
    hs.into_iter().map(|h| h.join().unwrap()).collect()
}

/// Deterministic per-rank data: small integers, so f32 sums are exact in any order.
fn data(rank: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + rank * 13) % 23) as f32 - 11.0)
        .collect()
}

#[test]
fn crc32_matches_ieee_check_value() {
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(crc32(b""), 0);
}

#[test]
fn bf16_is_round_to_nearest_even() {
    assert_eq!(bf16(1.0), 0x3F80);
    assert_eq!(bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
    assert_eq!(bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
    assert!(f32::from_bits((bf16(f32::NAN) as u32) << 16).is_nan());
}

#[test]
fn compression_parse_and_bounds() {
    assert_eq!(Compression::parse("none"), Some(Compression::None));
    assert_eq!(Compression::parse("bf16"), Some(Compression::Bf16));
    assert_eq!(
        Compression::parse("int8"),
        Some(Compression::Int8Block { block: 256 })
    );
    assert_eq!(
        Compression::parse("int8:64"),
        Some(Compression::Int8Block { block: 64 })
    );
    assert_eq!(Compression::parse("int8:0"), None);
    assert_eq!(Compression::parse("fp4"), None);
    let x: Vec<f32> = (0..1000)
        .map(|i| ((i as f32) * 0.37).sin() * (1.0 + i as f32 / 100.0))
        .collect();
    for c in [Compression::Bf16, Compression::Int8Block { block: 64 }] {
        let mut y = x.clone();
        c.round_trip(&mut y);
        let mut wire = Vec::new();
        c.encode(&x, &mut wire);
        assert_eq!(wire.len(), c.wire_len(x.len()));
        for (blk, (xs, ys)) in x.chunks(64).zip(y.chunks(64)).enumerate() {
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            for (a, b) in xs.iter().zip(ys) {
                let bound = match c {
                    Compression::Bf16 => a.abs() * 2f32.powi(-8),
                    _ => amax / 127.0 * 0.5 * 1.0001,
                };
                assert!((a - b).abs() <= bound, "{c:?} block {blk}: {a} vs {b}");
            }
        }
    }
    assert_eq!(
        Compression::Int8Block { block: 64 }.wire_len(130),
        130 + 3 * 4
    );
}

#[test]
fn all_reduce_sum_is_exact_for_every_world_and_length() {
    for world in 1..=5 {
        for n in [0usize, 1, 3, 4, 5, 17, 1000] {
            let rings = local_rings(world).unwrap();
            let out = run(rings, move |r| {
                let mut v = data(r.rank, n);
                r.all_reduce_sum(&mut v, Compression::None).unwrap();
                v
            });
            let want: Vec<f32> = (0..n)
                .map(|i| (0..world).map(|k| data(k, n)[i]).sum())
                .collect();
            for (r, v) in &out {
                assert_eq!(v, &want, "world {world} n {n} rank {}", r.rank);
                assert_eq!(r.stats().ops, 1);
            }
        }
    }
}

#[test]
fn compressed_all_reduce_is_bit_identical_across_ranks_and_close() {
    for c in [Compression::Bf16, Compression::Int8Block { block: 32 }] {
        let world = 4;
        let n = 1001;
        let out = run(local_rings(world).unwrap(), move |r| {
            let mut v: Vec<f32> = (0..n)
                .map(|i| ((i * 31 + r.rank * 7) as f32 * 0.01).sin())
                .collect();
            r.all_reduce_sum(&mut v, c).unwrap();
            v
        });
        let want: Vec<f32> = (0..n)
            .map(|i| {
                (0..world)
                    .map(|k| ((i * 31 + k * 7) as f32 * 0.01).sin())
                    .sum()
            })
            .collect();
        for (_, v) in &out {
            assert_eq!(v, &out[0].1, "{c:?}: ranks disagree");
        }
        let err = out[0]
            .1
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let tol = match c {
            Compression::Bf16 => 0.05,
            _ => 0.12,
        };
        assert!(err < tol, "{c:?}: max error {err}");
    }
}

#[test]
fn bytes_on_the_wire_follow_the_ring_bound() {
    let (world, n) = (4, 4000);
    for c in [Compression::None, Compression::Bf16] {
        let out = run(local_rings(world).unwrap(), move |r| {
            let mut v = data(r.rank, n);
            r.all_reduce_sum(&mut v, c).unwrap();
        });
        for (r, _) in &out {
            let chunk = c.wire_len(n / world) as u64 + 16;
            assert_eq!(r.stats().bytes_sent, 2 * (world as u64 - 1) * chunk);
            assert_eq!(r.stats().bytes_received, r.stats().bytes_sent);
        }
    }
}

#[test]
fn broadcast_from_every_root_and_barrier() {
    let world = 3;
    for root in 0..world {
        let out = run(local_rings(world).unwrap(), move |r| {
            let mut v = if r.rank == root {
                data(99, 10)
            } else {
                vec![0.0; 10]
            };
            r.broadcast(&mut v, root).unwrap();
            r.barrier().unwrap();
            v
        });
        for (_, v) in &out {
            assert_eq!(v, &data(99, 10));
        }
    }
}

#[test]
fn many_rounds_keep_sequence_numbers_in_step() {
    let out = run(local_rings(3).unwrap(), |r| {
        for k in 0..50 {
            let mut v = vec![k as f32; 7];
            r.all_reduce_sum(&mut v, Compression::None).unwrap();
            assert_eq!(v, vec![3.0 * k as f32; 7]);
        }
    });
    assert_eq!(out.len(), 3);
}

fn pair() -> (Link, Link) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let a = TcpStream::connect(l.local_addr().unwrap()).unwrap();
    let (b, _) = l.accept().unwrap();
    (Link { stream: a, seq: 0 }, Link { stream: b, seq: 0 })
}

#[test]
fn corrupted_or_out_of_order_frames_are_rejected() {
    let (mut a, mut b) = pair();
    let mut buf = Vec::new();
    a.send(b"hello").unwrap();
    b.recv(&mut buf).unwrap();
    assert_eq!(buf, b"hello");
    // Bad CRC.
    let mut frame = Vec::new();
    frame.extend_from_slice(&MAGIC);
    frame.extend_from_slice(&1u32.to_le_bytes());
    frame.extend_from_slice(&3u32.to_le_bytes());
    frame.extend_from_slice(&(crc32(b"abc") ^ 1).to_le_bytes());
    frame.extend_from_slice(b"abc");
    a.stream.write_all(&frame).unwrap();
    let e = b.recv(&mut buf).unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    assert!(e.to_string().contains("CRC"), "{e}");
    // Wrong sequence number.
    let (mut a, mut b) = pair();
    a.seq = 5;
    a.send(b"x").unwrap();
    assert!(b
        .recv(&mut buf)
        .unwrap_err()
        .to_string()
        .contains("expected 0"));
    // Wrong magic.
    let (mut a, mut b) = pair();
    a.stream.write_all(&[0u8; 16]).unwrap();
    assert!(b.recv(&mut buf).unwrap_err().to_string().contains("magic"));
}

#[test]
fn handshake_rejects_a_miswired_ring() {
    let ls: Vec<_> = (0..2)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let peers: Vec<_> = ls.iter().map(|l| l.local_addr().unwrap()).collect();
    let mut it = ls.into_iter();
    let (l0, l1) = (it.next().unwrap(), it.next().unwrap());
    let p0 = peers.clone();
    // Rank 1 believes the world has 3 ranks: rank 0 must refuse its hello.
    let bad = std::thread::spawn(move || {
        let three = [p0[0], p0[1], p0[0]];
        Ring::with_listener(1, 3, l1, &three, Duration::from_secs(3))
    });
    let r0 = Ring::with_listener(0, 2, l0, &peers, Duration::from_secs(3));
    assert!(r0.is_err());
    let _ = bad.join();
}

#[test]
fn invalid_arguments_and_unreachable_peers_fail_cleanly() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    assert!(Ring::connect(2, 2, addr, &[addr, addr], Duration::from_millis(10)).is_err());
    let solo = Ring::connect(0, 1, addr, &[addr], Duration::from_millis(10)).unwrap();
    assert_eq!(solo.world, 1);
    // Nobody listens at the right neighbour's address.
    let dead = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let e = Ring::connect(0, 2, addr, &[addr, dead], Duration::from_millis(300));
    assert!(e.is_err());
}
