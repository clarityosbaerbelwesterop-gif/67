use crate::config::{ModelConfig, TrainConfig};
use crate::model::{self, Head};
use crate::rng::Rng;
use crate::trainer::Trainer;
use crate::{ckpt, merge, Sampler};
use forge_isa::{compile, regs, CompileOptions, Machine};
use forge_kernels::GemmVariant;
use serde_json::Value;
use std::sync::{Arc, Mutex};

fn tiny(vocab: usize) -> ModelConfig {
    ModelConfig {
        vocab_size: vocab,
        dim: 16,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        ffn_hidden: Some(24),
        max_seq_len: 16,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
    }
}

#[test]
fn param_count_matches_scp_formula() {
    // SCP preset "tiny": dim 128, 4 layers, 4/2 heads, vocab 256, ffn derived = 384.
    let c = ModelConfig {
        vocab_size: 256,
        dim: 128,
        n_layers: 4,
        n_heads: 4,
        n_kv_heads: 2,
        ffn_hidden: None,
        max_seq_len: 256,
        rope_theta: 1e4,
        norm_eps: 1e-5,
    };
    assert_eq!(c.ffn(), 384);
    let manual: usize = model::params(&c).iter().map(|p| p.len()).sum();
    assert_eq!(manual, c.num_params());
}

/// End-to-end check of the hand-derived backward pass of the whole transformer
/// (embedding, RMSNorm, RoPE, GQA attention, SwiGLU, tied head, cross-entropy)
/// against central differences of the forward loss.
#[test]
fn full_model_gradients_match_finite_differences() {
    let c = tiny(11);
    let (b, t) = (2, 5);
    let mut m = Machine::new(1 << 22, 2);
    m.set_default_variant(GemmVariant::Scalar);
    let opts = CompileOptions::default();
    let (step, _) = compile(model::build(&c, b, t, Head::Train { accum: 1 }), opts).unwrap();
    let (eval, _) = compile(model::build(&c, b, t, Head::Eval), opts).unwrap();
    let mut step = m.load(step).unwrap();
    let mut eval = m.load(eval).unwrap();
    let mut rng = Rng::new(7);
    for p in model::params(&c) {
        let buf = m.persistent_mut(&p.name).unwrap();
        for x in buf.iter_mut() {
            *x = if p.shape.len() == 1 {
                1.0 + 0.2 * rng.normal()
            } else {
                0.25 * rng.normal()
            };
        }
    }
    let n = b * t;
    let toks: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 11) as f32).collect();
    let tgts: Vec<f32> = (0..n).map(|i| ((i * 5 + 1) % 11) as f32).collect();
    m.persistent_mut(&model::tokens_name(n))
        .unwrap()
        .copy_from_slice(&toks);
    m.persistent_mut(&model::targets_name(n))
        .unwrap()
        .copy_from_slice(&tgts);
    let loss = |m: &mut Machine, eval: &mut forge_isa::Loaded| -> f64 {
        for r in [regs::LOSS, regs::LOSS_COUNT, regs::CORRECT] {
            m.regs[r as usize] = 0.0;
        }
        m.run(eval).unwrap();
        (m.regs[regs::LOSS as usize] / m.regs[regs::LOSS_COUNT as usize]) as f64
    };
    m.run(&mut step).unwrap();
    let eps = 3e-3f32;
    let mut checked = 0;
    for p in model::params(&c) {
        let grad = m.persistent(&model::grad_name(&p.name)).unwrap().to_vec();
        for k in 0..4 {
            let i = (k * 7919 + p.len() / 3) % p.len();
            let orig = m.persistent(&p.name).unwrap()[i];
            m.persistent_mut(&p.name).unwrap()[i] = orig + eps;
            let up = loss(&mut m, &mut eval);
            m.persistent_mut(&p.name).unwrap()[i] = orig - eps;
            let dn = loss(&mut m, &mut eval);
            m.persistent_mut(&p.name).unwrap()[i] = orig;
            let num = (up - dn) / (2.0 * eps as f64);
            let ana = grad[i] as f64;
            assert!(
                (num - ana).abs() <= 2e-3 + 0.05 * ana.abs(),
                "{}[{i}]: analytic {ana:.6} vs numeric {num:.6}",
                p.name
            );
            checked += 1;
        }
    }
    assert!(checked >= 4 * (2 + 9 * c.n_layers));
}

fn collect() -> (crate::trainer::Sink, Arc<Mutex<Vec<Value>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let l2 = log.clone();
    (Box::new(move |v| l2.lock().unwrap().push(v)), log)
}

fn cfg(run: &str, out: &str, steps: usize) -> TrainConfig {
    TrainConfig {
        run: run.into(),
        model: tiny(8),
        data: "synthetic:5".into(),
        batch: 4,
        seq_len: 12,
        grad_accum: 2,
        max_steps: steps,
        lr: 1e-2,
        min_lr: 1e-3,
        warmup_steps: 5,
        weight_decay: 0.0,
        grad_clip: 1.0,
        seed: 3,
        init_from: None,
        resume_from: None,
        out_dir: out.into(),
        log_every: 10,
        eval_every: 0,
        eval_batches: 2,
        ckpt_every: 0,
        threads: 2,
        hbvm_mib: 0,
        autotune: false,
    }
}

#[test]
fn trainer_learns_a_periodic_stream_and_samples_it_back() {
    let out = std::env::temp_dir().join(format!("forge-train-{}", std::process::id()));
    let (sink, log) = collect();
    let mut tr = Trainer::new(cfg("toy", out.to_str().unwrap(), 120), sink).unwrap();
    let first = tr.train_step().unwrap();
    let last = tr.run(None).unwrap();
    assert!(first > 1.5, "initial loss {first} should be near ln(8)");
    assert!(last < 0.1, "loss did not converge: {first} -> {last}");
    let log = log.lock().unwrap();
    assert!(log
        .iter()
        .any(|v| v["type"] == "telemetry" && v["tflops"].as_f64().unwrap() > 0.0));
    let done = log
        .iter()
        .find(|v| v["type"] == "done")
        .expect("done event");
    assert!(done["val_loss"].as_f64().unwrap() < 0.1);
    let ck = log
        .iter()
        .rev()
        .find(|v| v["type"] == "checkpoint")
        .expect("checkpoint event");
    let dir = std::path::PathBuf::from(ck["path"].as_str().unwrap());
    let (w, _) = ckpt::load_safetensors(&dir.join("model.safetensors")).unwrap();
    assert_eq!(
        ckpt::sha256_file(&dir.join("model.safetensors")).unwrap(),
        ck["sha256"].as_str().unwrap()
    );

    // The learned model continues the pattern 0,1,2,3,4,0,1,... greedily.
    let mut s = Sampler::new(&tiny(8), &w, 12, 2).unwrap();
    let gen = s.generate(&[2, 3, 4, 0], 6, 0.0, 1, 0, None).unwrap();
    assert_eq!(gen, vec![1, 2, 3, 4, 0, 1]);

    // A child initialised from this checkpoint starts from identical weights and records lineage.
    let (sink2, log2) = collect();
    let mut child_cfg = cfg("child", out.to_str().unwrap(), 1);
    child_cfg.init_from = Some(dir.to_str().unwrap().into());
    let child = Trainer::new(child_cfg, sink2).unwrap();
    assert_eq!(
        child.param("tok_emb.weight").unwrap(),
        &w["tok_emb.weight"].data[..]
    );
    assert!(log2.lock().unwrap().iter().any(|v| v["msg"]
        .as_str()
        .is_some_and(|m| m.contains("initialised from"))));

    // Merging two children that equal the base reproduces the base.
    let merged = merge::merge(
        &w,
        &[
            ckpt::load_safetensors(&dir.join("model.safetensors"))
                .unwrap()
                .0,
            ckpt::load_safetensors(&dir.join("model.safetensors"))
                .unwrap()
                .0,
        ],
        merge::Method::Ties { density: 0.5 },
        1.0,
    )
    .unwrap();
    assert_eq!(merged["norm.weight"].data, w["norm.weight"].data);
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn control_commands_are_applied_or_rejected() {
    let out = std::env::temp_dir().join(format!("forge-ctl-{}", std::process::id()));
    let (sink, log) = collect();
    let mut tr = Trainer::new(cfg("ctl", out.to_str().unwrap(), 3), sink).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    for c in [
        r#"{"cmd":"set_threads","value":1}"#,
        r#"{"cmd":"set_lr","value":5.0}"#,
        r#"{"cmd":"swap_kernel","variant":"scalar"}"#,
        r#"{"cmd":"bogus"}"#,
        "not json",
    ] {
        tx.send(c.to_string()).unwrap();
    }
    drop(tx);
    tr.run(Some(rx)).unwrap();
    assert_eq!(tr.machine.threads(), 1);
    let log = log.lock().unwrap();
    let warns: Vec<&str> = log
        .iter()
        .filter(|v| v["level"] == "warn")
        .filter_map(|v| v["msg"].as_str())
        .collect();
    assert_eq!(warns.len(), 3, "{warns:?}");
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn diloco_with_one_worker_equals_plain_training() {
    use crate::diloco::{local, DiLoCo};
    let out = std::env::temp_dir().join(format!("forge-dl1-{}", std::process::id()));
    let (s1, _) = collect();
    let (s2, _) = collect();
    let mut plain = Trainer::new(cfg("plain", out.to_str().unwrap(), 20), s1).unwrap();
    let mut dl = Trainer::new(cfg("dl", out.to_str().unwrap(), 20), s2).unwrap();
    for _ in 0..8 {
        plain.train_step().unwrap();
    }
    let mut g = local::group(1);
    // outer lr 1, momentum 0: θ ← θ − (θ − θ_local) = θ_local, i.e. plain training.
    let mut d = DiLoCo::new(&dl, 4, 1.0, 0.0);
    d.round(&mut dl, &mut g[0]).unwrap();
    d.round(&mut dl, &mut g[0]).unwrap();
    for p in model::params(&dl.cfg.model) {
        let (a, b) = (plain.param(&p.name).unwrap(), dl.param(&p.name).unwrap());
        assert!(
            a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5),
            "{} diverged",
            p.name
        );
    }
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn diloco_workers_stay_in_sync_and_learn() {
    use crate::diloco::{local, Collective, DiLoCo};
    let out = std::env::temp_dir().join(format!("forge-dl2-{}", std::process::id()));
    let world = 3;
    let group = local::group(world);
    let handles: Vec<_> = group
        .into_iter()
        .map(|mut coll| {
            let out = out.clone();
            std::thread::spawn(move || {
                let (sink, _) = collect();
                let mut c = cfg(&format!("w{}", coll.rank()), out.to_str().unwrap(), 1000);
                c.seed = 100 + coll.rank() as u64; // different data windows and init per worker
                c.threads = 1;
                let mut tr = Trainer::new(c, sink).unwrap();
                let mut d = DiLoCo::new(&tr, 10, 0.7, 0.9);
                d.synchronise(&mut tr, &mut coll).unwrap();
                let first = tr.evaluate().unwrap().0;
                let mut last = 0.0;
                for _ in 0..12 {
                    last = d.round(&mut tr, &mut coll).unwrap();
                }
                (
                    first,
                    last,
                    tr.evaluate().unwrap().0,
                    d.global().to_vec(),
                    d.bytes_communicated,
                )
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let g0 = &results[0].3;
    for (first, _, val, g, bytes) in &results {
        assert_eq!(g, g0, "workers must hold identical global parameters");
        assert!(*val < 0.3 * first, "DiLoCo did not learn: {first} -> {val}");
        assert_eq!(
            *bytes,
            12 * 4 * g0.len() as u64,
            "one exchange per round of 10 steps"
        );
    }
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn resume_continues_exactly_where_the_run_stopped() {
    let out = std::env::temp_dir().join(format!("forge-resume-{}", std::process::id()));
    let o = out.to_str().unwrap();
    // Reference: 12 uninterrupted steps.
    let (s1, _) = collect();
    let mut a = Trainer::new(cfg("ref", o, 12), s1).unwrap();
    for _ in 0..12 {
        a.train_step().unwrap();
    }
    // Interrupted: 7 steps, checkpoint, new process state, resume, 5 more.
    let (s2, _) = collect();
    let mut b = Trainer::new(cfg("cut", o, 12), s2).unwrap();
    for _ in 0..7 {
        b.train_step().unwrap();
    }
    let dir = b.checkpoint().unwrap();
    let (s3, _) = collect();
    let mut rc = cfg("cut", o, 12);
    rc.resume_from = Some(dir.to_str().unwrap().into());
    let mut c = Trainer::new(rc, s3).unwrap();
    assert_eq!(c.step, 7);
    for _ in 0..5 {
        c.train_step().unwrap();
    }
    for p in model::params(&a.cfg.model) {
        assert_eq!(
            a.param(&p.name).unwrap(),
            c.param(&p.name).unwrap(),
            "{} differs after resume",
            p.name
        );
    }
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn diloco_over_the_tcp_ring_with_bf16_pseudo_gradients() {
    use crate::diloco::{Collective, DiLoCo, RingCollective};
    let out = std::env::temp_dir().join(format!("forge-dl3-{}", std::process::id()));
    let rings = forge_comm::local_rings(2).unwrap();
    let handles: Vec<_> = rings
        .into_iter()
        .map(|ring| {
            let out = out.clone();
            std::thread::spawn(move || {
                let mut coll = RingCollective {
                    ring,
                    compression: forge_comm::Compression::Bf16,
                };
                let (sink, _) = collect();
                let mut c = cfg(&format!("t{}", coll.rank()), out.to_str().unwrap(), 1000);
                c.seed = 7 + coll.rank() as u64;
                c.threads = 1;
                let mut tr = Trainer::new(c, sink).unwrap();
                let mut d = DiLoCo::new(&tr, 10, 0.7, 0.9);
                d.synchronise(&mut tr, &mut coll).unwrap();
                let synced = d.global().to_vec();
                let first = tr.evaluate().unwrap().0;
                for _ in 0..10 {
                    d.round(&mut tr, &mut coll).unwrap();
                }
                let stats = coll.ring.stats();
                (
                    synced,
                    first,
                    tr.evaluate().unwrap().0,
                    d.global().to_vec(),
                    stats,
                )
            })
        })
        .collect();
    let r: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(
        r[0].0, r[1].0,
        "broadcast must start both workers from rank 0's θ bit for bit"
    );
    assert_eq!(
        r[0].3, r[1].3,
        "workers must hold identical global parameters"
    );
    for (_, first, val, _, stats) in &r {
        assert!(
            *val < 0.3 * first,
            "DiLoCo over TCP did not learn: {first} -> {val}"
        );
        // 10 rounds of bf16 pseudo-gradients + 1 exact f32 broadcast on the wire.
        assert!(stats.bytes_sent > 0 && stats.ops == 11, "{stats:?}");
    }
    std::fs::remove_dir_all(out).ok();
}

fn weights_of(tr: &Trainer) -> std::collections::BTreeMap<String, ckpt::Tensor> {
    model::params(&tr.cfg.model)
        .into_iter()
        .map(|p| {
            let data = tr.param(&p.name).unwrap().to_vec();
            (
                p.name,
                ckpt::Tensor {
                    shape: p.shape,
                    data,
                },
            )
        })
        .collect()
}

#[test]
fn kv_cache_decoder_matches_the_isa_forward_pass() {
    use crate::decode::Decoder;
    let out = std::env::temp_dir().join(format!("forge-kv-{}", std::process::id()));
    let mut c = cfg("kv", out.to_str().unwrap(), 1);
    c.model = ModelConfig {
        vocab_size: 37,
        dim: 24,
        n_layers: 3,
        n_heads: 6,
        n_kv_heads: 2,
        ffn_hidden: Some(40),
        max_seq_len: 24,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
    };
    let (sink, _) = collect();
    let tr = Trainer::new(c.clone(), sink).unwrap();
    let w = weights_of(&tr);
    let ids: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 37).collect();
    let mut s = Sampler::new(&c.model, &w, 24, 2).unwrap();
    let mut d = Decoder::new(&c.model, &w, 2).unwrap();
    let close = |a: &[f32], b: &[f32], at: usize| {
        let scale = a.iter().fold(1f32, |m, v| m.max(v.abs()));
        let err = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(
            err <= 1e-4 * scale,
            "position {at}: max |Δlogit| {err} (scale {scale})"
        );
    };
    // A prompt block of 8 positions, then one position at a time.
    let got = d.feed(&ids[..8]).unwrap();
    close(&s.next_logits(&ids[..8]).unwrap(), &got, 8);
    for i in 9..=ids.len() {
        let got = d.feed(&ids[i - 1..i]).unwrap();
        close(&s.next_logits(&ids[..i]).unwrap(), &got, i);
    }
    assert_eq!(d.len(), 20);
    assert!(
        d.feed(&[1, 2, 3, 4, 5]).is_err(),
        "context overflow must be an error"
    );
    assert!(
        d.feed(&[99]).is_err(),
        "token outside the vocabulary must be an error"
    );
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn kv_cache_generation_equals_window_sampling() {
    use crate::decode::Decoder;
    let out = std::env::temp_dir().join(format!("forge-kvg-{}", std::process::id()));
    let (sink, _) = collect();
    let mut tr = Trainer::new(cfg("kvg", out.to_str().unwrap(), 60), sink).unwrap();
    tr.run(None).unwrap();
    let w = weights_of(&tr);
    let m = tiny(8);
    let mut s = Sampler::new(&m, &w, m.max_seq_len, 2).unwrap();
    let mut d = Decoder::new(&m, &w, 2).unwrap();
    for prompt in [vec![2u32, 3, 4, 0], vec![0], vec![4, 0, 1, 2, 3, 4, 0, 1]] {
        let n = m.max_seq_len - prompt.len();
        let want = s.generate(&prompt, n, 0.0, 1, 0, None).unwrap();
        assert_eq!(d.generate(&prompt, n, 0.0, 1, 0, None).unwrap(), want);
        // Sampled decoding draws from the same RNG stream.
        let want = s.generate(&prompt, n, 0.8, 4, 9, None).unwrap();
        assert_eq!(d.generate(&prompt, n, 0.8, 4, 9, None).unwrap(), want);
    }
    assert_eq!(
        d.generate(&[2, 3], 6, 0.0, 1, 0, None).unwrap(),
        vec![4, 0, 1, 2, 3, 4]
    );
    // Longer than the context: the cache is rebuilt and decoding continues.
    let long = d.generate(&[0, 1, 2], 40, 0.0, 1, 0, None).unwrap();
    assert_eq!(long.len(), 40);
    assert!(long.windows(2).all(|p| p[1] == (p[0] + 1) % 5), "{long:?}");
    std::fs::remove_dir_all(out).ok();
}

/// A briefly trained model for the batched-decoding tests. Its widths (12,
/// 20) are not multiples of 8, so every projection has a tail after the
/// 8-wide partial sums; tokens 5..13 never occur in its training stream.
fn batch_model(
    tag: &str,
    steps: usize,
) -> (
    ModelConfig,
    std::collections::BTreeMap<String, ckpt::Tensor>,
) {
    let out = std::env::temp_dir().join(format!("forge-{tag}-{}", std::process::id()));
    let mut c = cfg(tag, out.to_str().unwrap(), steps);
    c.model = ModelConfig {
        vocab_size: 13,
        dim: 12,
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        ffn_hidden: Some(20),
        max_seq_len: 16,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
    };
    let (sink, _) = collect();
    let mut tr = Trainer::new(c.clone(), sink).unwrap();
    for _ in 0..steps {
        tr.train_step().unwrap();
    }
    let w = weights_of(&tr);
    std::fs::remove_dir_all(out).ok();
    (c.model, w)
}

/// Prompts of lengths 0 to 20 (context 16): empty, unseen tokens, and one
/// longer than the context that is cut from the left.
fn batch_prompts() -> Vec<Vec<u32>> {
    vec![
        vec![],
        vec![3],
        vec![2, 3, 4, 0],
        vec![4, 0, 1, 2, 3, 4, 0],
        vec![9, 12, 7],
        (0..20).map(|i| (i * 3 % 5) as u32).collect(),
        vec![1, 2],
        vec![0; 11],
        vec![11, 3, 4, 6, 8],
    ]
}

/// One `Decoder::generate` per prompt: the reference for batched decoding.
fn one_by_one(
    d: &mut crate::decode::Decoder,
    prompts: &[Vec<u32>],
    seeds: &[u64],
    (max_new, temperature, top_k, stop): (usize, f32, usize, Option<u32>),
) -> Vec<Vec<u32>> {
    prompts
        .iter()
        .zip(seeds)
        .map(|(p, &s)| d.generate(p, max_new, temperature, top_k, s, stop).unwrap())
        .collect()
}

#[test]
fn batched_greedy_decoding_equals_sequential() {
    use crate::decode::Decoder;
    let (m, w) = batch_model("bgreedy", 25);
    let mut d = Decoder::new(&m, &w, 2).unwrap();
    let prompts = batch_prompts();
    let seeds = vec![0u64; prompts.len()];
    for stop in [None, Some(2)] {
        let want = one_by_one(&mut d, &prompts, &seeds, (9, 0.0, 1, stop));
        if stop.is_some() {
            // The stop id ends some sequences early; others run to max_new.
            assert!(want.iter().any(|g| g.len() < 9), "{want:?}");
            assert!(want.iter().any(|g| g.len() == 9), "{want:?}");
        }
        for batch in [1, 2, 4, 16] {
            let got = d
                .generate_batch(&prompts, &seeds, 9, 0.0, 1, stop, batch)
                .unwrap();
            assert_eq!(got, want, "batch {batch}, stop {stop:?}");
        }
    }
    // max_new 0 yields empty completions, as `generate` does.
    let empty = d
        .generate_batch(&prompts, &seeds, 0, 0.0, 1, None, 4)
        .unwrap();
    assert!(empty.iter().all(|g| g.is_empty()));
    // A bad prompt: the ones before it are completed and reported, then its
    // error is returned (a loop of `generate` behaves the same way).
    let bad = vec![vec![1, 2], vec![3], vec![1, 99], vec![2]];
    let mut seen = Vec::new();
    let err = d
        .generate_batch_with(&bad, &[0; 4], 9, 0.0, 1, None, 4, |i, ids, _| {
            seen.push((i, ids))
        })
        .unwrap_err();
    assert!(err.contains("outside vocabulary"), "{err}");
    seen.sort();
    let want = one_by_one(&mut d, &bad[..2], &[0, 0], (9, 0.0, 1, None));
    assert_eq!(seen, vec![(0, want[0].clone()), (1, want[1].clone())]);
}

#[test]
fn batched_sampling_equals_sequential_for_the_same_seeds() {
    use crate::decode::Decoder;
    let (m, w) = batch_model("bsample", 25);
    let mut d = Decoder::new(&m, &w, 2).unwrap();
    let prompts = batch_prompts();
    let seeds: Vec<u64> = (0..prompts.len() as u64).map(|i| 100 + i).collect();
    for (temperature, top_k, stop) in [(0.9, 6, Some(4)), (1.3, 13, None)] {
        let want = one_by_one(&mut d, &prompts, &seeds, (10, temperature, top_k, stop));
        if stop.is_some() {
            assert!(want.iter().any(|g| g.len() < 10), "{want:?}");
            assert!(want.iter().any(|g| g.len() == 10), "{want:?}");
        }
        for batch in [3, 16] {
            let got = d
                .generate_batch(&prompts, &seeds, 10, temperature, top_k, stop, batch)
                .unwrap();
            assert_eq!(got, want, "batch {batch}, T {temperature}");
        }
    }
    // Different seeds draw different completions (the seeds are really used).
    let other: Vec<u64> = seeds.iter().map(|s| s + 1000).collect();
    let got = d
        .generate_batch(&prompts, &other, 10, 1.3, 13, None, 16)
        .unwrap();
    assert_ne!(
        got,
        one_by_one(&mut d, &prompts, &seeds, (10, 1.3, 13, None))
    );
}

#[test]
fn batched_step_logits_match_single_sequence_logits() {
    use crate::decode::Decoder;
    let (m, w) = batch_model("bstep", 25);
    let mut d = Decoder::new(&m, &w, 2).unwrap();
    // Five sequences at different positions (1, 4, 7, 9, 15 cached tokens).
    let prompts: Vec<Vec<u32>> = vec![
        vec![3],
        vec![2, 3, 4, 0],
        vec![4, 0, 1, 2, 3, 4, 0],
        vec![9, 12, 7, 1, 1, 2, 3, 4, 5],
        (0..15).map(|i| (i % 5) as u32).collect(),
    ];
    let mut caches: Vec<_> = prompts
        .iter()
        .map(|p| {
            let mut c = d.new_cache();
            d.feed_cache(&mut c, p).unwrap();
            c
        })
        .collect();
    let v = m.vocab_size;
    let mut fed: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    for tokens in [[1u32, 0, 1, 6, 0], [2, 1, 2, 7, 1]] {
        let open: Vec<usize> = (0..prompts.len())
            .filter(|&i| caches[i].len() < m.max_seq_len)
            .collect();
        let toks: Vec<u32> = open.iter().map(|&i| tokens[i]).collect();
        let mut refs: Vec<_> = caches
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| open.contains(i))
            .map(|(_, c)| c)
            .collect();
        let got = d.step(&mut refs, &toks).unwrap();
        assert_eq!(got.len(), open.len() * v);
        for (row, &i) in open.iter().enumerate() {
            // The same sequence alone: prompt block, the tokens of earlier
            // steps one at a time, then this step's token.
            d.reset();
            d.feed(&prompts[i]).unwrap();
            for &t in &fed[i] {
                d.feed(&[t]).unwrap();
            }
            let want = d.feed(&[tokens[i]]).unwrap();
            fed[i].push(tokens[i]);
            let row = &got[row * v..(row + 1) * v];
            let scale = want.iter().fold(1f32, |a, x| a.max(x.abs()));
            let err = want
                .iter()
                .zip(row)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            assert!(
                err <= 1e-4 * scale,
                "sequence {i}: max |Δlogit| {err} (scale {scale})"
            );
            // Stronger: the batched row is bit-identical to the one-row path.
            assert!(
                want.iter()
                    .zip(row)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "sequence {i}: batched logits differ in bits from the single-sequence logits"
            );
        }
    }
    // The sequence that filled the context was left out of the second step.
    assert_eq!(caches[4].len(), m.max_seq_len);
    assert!(
        d.step(&mut [&mut caches[4]], &[0]).is_err(),
        "context overflow must be an error"
    );
}

#[test]
fn batched_context_overflow_equals_sequential() {
    use crate::decode::Decoder;
    let (m, w) = batch_model("boverflow", 25);
    let mut d = Decoder::new(&m, &w, 2).unwrap();
    let prompts = batch_prompts();
    let seeds: Vec<u64> = (0..prompts.len() as u64).map(|i| 7 + 3 * i).collect();
    // 30 new tokens > context 16: the prompt keeps its last token, the cache
    // fills after 16 positions and the newest half is re-encoded (twice per
    // sequence). With a stop id, sequences end and new ones enter at
    // different steps, so overflows happen at different steps in the batch.
    let max_new = 30;
    for (temperature, top_k, stop) in [(0.0, 1, None), (1.0, 8, Some(4)), (1.2, 13, None)] {
        let want = one_by_one(
            &mut d,
            &prompts,
            &seeds,
            (max_new, temperature, top_k, stop),
        );
        assert!(
            want.iter().any(|g| g.len() > m.max_seq_len),
            "no sequence overflowed the context: {want:?}"
        );
        if stop.is_some() {
            assert!(want.iter().any(|g| g.len() < max_new), "{want:?}");
        }
        for batch in [1, 3, 16] {
            let got = d
                .generate_batch(&prompts, &seeds, max_new, temperature, top_k, stop, batch)
                .unwrap();
            assert_eq!(got, want, "batch {batch}, T {temperature}, stop {stop:?}");
        }
    }
}
