use crate::config::{ModelConfig, TrainConfig};
use crate::model::{self, Head};
use crate::rng::Rng;
use crate::trainer::{optim_slots, Trainer};
use crate::{ckpt, merge, Sampler};
use forge_isa::{compile, regs, Builder, CompileOptions, Instr, Machine};
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
        lr_schedule: "cosine".into(),
        decay_steps: 0,
        init_optim: false,
        optimizer: "adamw".into(),
        muon_lr: None,
        muon_momentum: None,
        muon_ns_steps: None,
        muon_weight_decay: None,
    }
}

#[test]
fn lr_schedules_cosine_unchanged_wsd_and_constant_shapes() {
    let mut c = cfg("lr", "/tmp", 100);
    c.lr = 1.0;
    c.min_lr = 0.1;
    c.warmup_steps = 10;
    // "cosine" is SCP's formula, bit for bit.
    for step in [0usize, 5, 9, 10, 37, 99, 100, 150] {
        let want = if step < 10 {
            (step + 1) as f32 / 10.0
        } else if step >= 100 {
            0.1
        } else {
            let p = (step - 10) as f32 / 90.0;
            0.1 + 0.5 * 0.9 * (1.0 + (std::f32::consts::PI * p).cos())
        };
        assert_eq!(c.lr_at(step).to_bits(), want.to_bits(), "cosine step {step}");
    }
    c.lr_schedule = "constant".into();
    assert_eq!(c.lr_at(3), 0.4);
    assert_eq!(c.lr_at(99), 1.0);
    c.lr_schedule = "wsd".into();
    assert_eq!(c.wsd_decay_steps(), 20);
    assert_eq!(c.trunk_step(), Some(80));
    assert_eq!(c.lr_at(79), 1.0, "last stable update");
    assert!(c.lr_at(80) < 1.0 && c.lr_at(80) > 0.99, "first decayed update");
    assert!((c.lr_at(89) - 0.55).abs() < 1e-6, "half-way through the decay");
    assert!((c.lr_at(99) - 0.1).abs() < 1e-6, "last update at min_lr");
    c.decay_steps = 50;
    assert_eq!(c.trunk_step(), Some(50));
    c.lr_schedule = "linear".into();
    assert!(c.validate().is_err());
}

#[test]
fn wsd_writes_the_trunk_once_and_init_optim_continues_it() {
    let out = std::env::temp_dir().join(format!("forge-trunk-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let mut c = cfg("gen1", out.to_str().unwrap(), 40);
    c.lr_schedule = "wsd".into();
    c.decay_steps = 10;
    let (sink, log) = collect();
    let mut tr = Trainer::new(c, sink).unwrap();
    tr.run(None).unwrap();
    let trunk = out.join("gen1/TRUNK");
    let st: Value =
        serde_json::from_str(&std::fs::read_to_string(trunk.join("state.json")).unwrap()).unwrap();
    assert_eq!(st["step"], 30);
    assert_eq!(st["trunk"], true);
    assert!(trunk.join("optim.safetensors").exists());
    assert_eq!(
        log.lock().unwrap().iter().filter(|v| v["type"] == "trunk").count(),
        1
    );
    assert!(!out.join("gen1/TRUNK.tmp").exists());

    // The next generation starts from the trunk's weights and moments.
    let mut c2 = cfg("gen2", out.to_str().unwrap(), 4);
    c2.init_from = Some(trunk.to_str().unwrap().into());
    c2.init_optim = true;
    c2.warmup_steps = 0;
    c2.lr_schedule = "wsd".into();
    let (sink2, log2) = collect();
    let mut tr2 = Trainer::new(c2, sink2).unwrap();
    let (opt, _) = ckpt::load_safetensors(&trunk.join("optim.safetensors")).unwrap();
    let m = &opt["tok_emb.weight"].data;
    assert_eq!(
        tr2.param(&model::adam_m("tok_emb.weight")).unwrap(),
        &m[..],
        "first moments loaded"
    );
    assert!(log2.lock().unwrap().iter().any(|v| v["msg"]
        .as_str()
        .is_some_and(|m| m.contains("30 steps behind them"))));
    let first = tr2.train_step().unwrap();
    assert!(first.is_finite());
    let dir = tr2.checkpoint().unwrap();
    let st2: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!(st2["optim_offset"], 30);
    assert_eq!(st2["parent_sha256"], st["model_sha256"]);

    // Without moments in the init checkpoint, init_optim is an error.
    std::fs::remove_file(trunk.join("optim.safetensors")).unwrap();
    let mut c3 = cfg("gen3", out.to_str().unwrap(), 4);
    c3.init_from = Some(trunk.to_str().unwrap().into());
    c3.init_optim = true;
    let (sink3, _) = collect();
    assert!(Trainer::new(c3, sink3).is_err());
    let _ = std::fs::remove_dir_all(&out);
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

// ---------------------------------------------------------------- Muon

/// One MUON instruction on its own machine with p = 0, m = 0, momentum 0,
/// lr 1, scale 1, clip 1 and no decay: p ends as −O, the Newton–Schulz output of g.
fn muon_orthogonalise(g: &[f32], rows: usize, cols: usize, threads: usize) -> Vec<f32> {
    let n = rows * cols;
    let mut b = Builder::new();
    let (p, gv, m) = (
        b.persistent("p", n),
        b.persistent("g", n),
        b.persistent("m", n),
    );
    b.emit(Instr::Muon {
        p,
        g: gv,
        m,
        rows: rows as u32,
        cols: cols as u32,
        lr: regs::MUON_LR,
        clip: regs::CLIP_COEF,
        momentum: 0.0,
        ns_steps: 5,
        scale: 1.0,
        wd: 0.0,
    });
    let mut mc = Machine::new(1 << 20, threads);
    let mut lp = mc.load(b.finish()).unwrap();
    mc.persistent_mut("g").unwrap().copy_from_slice(g);
    mc.regs[regs::MUON_LR as usize] = 1.0;
    mc.regs[regs::CLIP_COEF as usize] = 1.0;
    mc.run(&mut lp).unwrap();
    assert_eq!(mc.persistent("m").unwrap(), g, "momentum 0 leaves m = g");
    mc.persistent("p").unwrap().iter().map(|x| -x).collect()
}

/// Row-major f64 product of an m×k and a k×n matrix.
fn mm64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for t in 0..k {
            let x = a[i * k + t];
            for j in 0..n {
                c[i * n + j] += x * b[t * n + j];
            }
        }
    }
    c
}

fn tr64(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut t = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = a[i * c + j];
        }
    }
    t
}

/// modded-nanogpt `zeropower_via_newtonschulz5` written out in f64:
/// transpose when rows > cols, normalise, iterate, transpose back.
fn newton_schulz_f64(g: &[f32], rows: usize, cols: usize, steps: usize) -> Vec<f64> {
    let (a, b, c) = (3.4445, -4.7750, 2.0315);
    let g: Vec<f64> = g.iter().map(|&v| v as f64).collect();
    let tall = rows > cols;
    let (s, l) = if tall { (cols, rows) } else { (rows, cols) };
    let mut x = if tall { tr64(&g, rows, cols) } else { g };
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-7;
    x.iter_mut().for_each(|v| *v /= norm);
    for _ in 0..steps {
        let am = mm64(&x, &tr64(&x, s, l), s, l, s);
        let aa = mm64(&am, &am, s, s, s);
        let bm: Vec<f64> = am.iter().zip(&aa).map(|(p, q)| b * p + c * q).collect();
        let bx = mm64(&bm, &x, s, s, l);
        x = x.iter().zip(&bx).map(|(p, q)| a * p + q).collect();
    }
    if tall {
        tr64(&x, s, l)
    } else {
        x
    }
}

/// Eigenvalues of a symmetric n×n matrix (cyclic Jacobi, f64).
fn sym_eigenvalues(mut a: Vec<f64>, n: usize) -> Vec<f64> {
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..n {
                    let (kp, kq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * kp - s * kq;
                    a[k * n + q] = s * kp + c * kq;
                }
                for k in 0..n {
                    let (pk, qk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * pk - s * qk;
                    a[q * n + k] = s * pk + c * qk;
                }
            }
        }
    }
    (0..n).map(|i| a[i * n + i]).collect()
}

/// Singular values of a rows×cols matrix: sqrt of the eigenvalues of the
/// min(rows, cols)² Gram matrix (X·Xᵀ when rows <= cols, Xᵀ·X otherwise).
fn singular_values(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let (gram, s) = if rows <= cols {
        (mm64(&x, &tr64(&x, rows, cols), rows, cols, rows), rows)
    } else {
        (mm64(&tr64(&x, rows, cols), &x, cols, rows, cols), cols)
    };
    let tr: f64 = (0..s).map(|i| gram[i * s + i]).sum();
    let ev = sym_eigenvalues(gram, s);
    assert!(
        (ev.iter().sum::<f64>() - tr).abs() <= 1e-9 * tr.max(1.0),
        "Jacobi lost the trace"
    );
    ev.into_iter().map(|e| e.max(0.0).sqrt()).collect()
}

#[test]
fn muon_newton_schulz_orthogonalises() {
    // Wide (X·Xᵀ ≈ I), tall (Xᵀ·X ≈ I: the transposed GEMM path) and square.
    for (rows, cols, seed) in [(24usize, 64usize, 1u64), (64, 24, 2), (40, 40, 3)] {
        let mut rng = Rng::new(seed);
        let g: Vec<f32> = (0..rows * cols).map(|_| rng.normal() * 3.0).collect();
        let o = muon_orthogonalise(&g, rows, cols, 2);
        // Same algorithm in plain f64 loops (modded-nanogpt, with explicit transposes).
        let want = newton_schulz_f64(&g, rows, cols, 5);
        let err = o
            .iter()
            .zip(&want)
            .fold(0f64, |m, (a, b)| m.max((*a as f64 - b).abs()));
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        assert!(
            err <= 1e-4 * scale,
            "{rows}x{cols}: max |O - O_f64| {err:e} (max |O| {scale})"
        );
        // Deterministic for any pool size.
        assert_eq!(
            muon_orthogonalise(&g, rows, cols, 1),
            o,
            "{rows}x{cols}: threads 1 vs 2"
        );
        if rows == cols {
            continue; // a square Gaussian has singular values near 0 that 5 steps cannot lift
        }
        // The input is ill-conditioned; the output's singular values sit in the band
        // the quintic (3.4445, -4.7750, 2.0315) maps them to after 5 steps:
        // [0.68, 1.155] for normalised inputs >= 0.005 (f32 slack: [0.65, 1.2]).
        let sin = singular_values(&g, rows, cols);
        let (imin, imax) = sin
            .iter()
            .fold((f64::MAX, 0f64), |(a, b), &s| (a.min(s), b.max(s)));
        assert!(
            imax / imin > 2.5,
            "{rows}x{cols}: input condition {}",
            imax / imin
        );
        let sv = singular_values(&o, rows, cols);
        assert_eq!(sv.len(), rows.min(cols));
        for s in &sv {
            assert!(
                (0.65..=1.2).contains(s),
                "{rows}x{cols}: singular value {s} (all: {sv:?})"
            );
        }
    }
}

#[test]
fn muon_takes_exactly_the_hidden_matrices() {
    let base_s = ModelConfig {
        vocab_size: 8192,
        dim: 384,
        n_layers: 6,
        n_heads: 6,
        n_kv_heads: 2,
        ffn_hidden: Some(1024),
        max_seq_len: 512,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
    };
    let opts = model::MuonOpts {
        momentum: 0.95,
        ns_steps: 5,
        weight_decay: 0.0,
    };
    for c in [tiny(8), base_s] {
        let ps = model::params(&c);
        let want: Vec<String> = (0..c.n_layers)
            .flat_map(|i| {
                [
                    "attn.wq", "attn.wk", "attn.wv", "attn.wo", "ffn.w1", "ffn.w3", "ffn.w2",
                ]
                .map(|w| format!("blocks.{i}.{w}.weight"))
            })
            .collect();
        let muon: Vec<String> = ps
            .iter()
            .filter(|p| p.muon())
            .map(|p| p.name.clone())
            .collect();
        assert_eq!(muon, want);
        let rest: Vec<String> = ps
            .iter()
            .filter(|p| !p.muon())
            .map(|p| p.name.clone())
            .collect();
        let mut want_rest = vec!["tok_emb.weight".to_string()];
        for i in 0..c.n_layers {
            want_rest.push(format!("blocks.{i}.attn_norm.weight"));
            want_rest.push(format!("blocks.{i}.ffn_norm.weight"));
        }
        want_rest.push("norm.weight".into());
        assert_eq!(
            rest, want_rest,
            "embedding (= tied head) and norms stay on AdamW"
        );

        // The update program: MUON on exactly those, AdamW on the rest.
        let prog = model::build_update(&c, 0.1, 1.0, Some(opts));
        let name = |v: &forge_isa::V| prog.bufs[v.0 as usize].name.clone();
        let (mut on_muon, mut on_adam) = (Vec::new(), Vec::new());
        for ins in &prog.instrs {
            match ins {
                Instr::Muon {
                    p,
                    m,
                    rows,
                    cols,
                    lr,
                    clip,
                    momentum,
                    ns_steps,
                    scale,
                    wd,
                    ..
                } => {
                    let shape = &ps.iter().find(|q| q.name == name(p)).unwrap().shape;
                    assert_eq!([*rows as usize, *cols as usize], [shape[0], shape[1]]);
                    let want_scale = (shape[0] as f32 / shape[1] as f32).max(1.0).sqrt();
                    assert_eq!(*scale, want_scale, "{}: max(1, rows/cols)^0.5", name(p));
                    assert_eq!(name(m), model::muon_m(&name(p)));
                    assert_eq!((*lr, *clip), (regs::MUON_LR, regs::CLIP_COEF));
                    assert_eq!((*momentum, *ns_steps, *wd), (0.95, 5, 0.0));
                    on_muon.push(name(p));
                }
                Instr::AdamW { p, .. } => on_adam.push(name(p)),
                _ => {}
            }
        }
        assert_eq!(on_muon, want);
        assert_eq!(on_adam, want_rest);
        if c.dim == 384 {
            // SwiGLU w1/w3 are [1024, 384]: scale sqrt(1024/384); wk [128, 384], w2 [384, 1024]: 1.
            let s = |w: &str| {
                prog.instrs
                    .iter()
                    .find_map(|i| match i {
                        Instr::Muon { p, scale, .. }
                            if name(p) == format!("blocks.0.{w}.weight") =>
                        {
                            Some(*scale)
                        }
                        _ => None,
                    })
                    .unwrap()
            };
            assert_eq!(s("ffn.w1"), (1024f32 / 384.0).sqrt());
            assert_eq!(s("ffn.w3"), (1024f32 / 384.0).sqrt());
            assert_eq!((s("attn.wk"), s("ffn.w2"), s("attn.wq")), (1.0, 1.0, 1.0));
        }
        // Without Muon every parameter is on AdamW and no Muon state exists.
        let plain = model::build_update(&c, 0.1, 1.0, None);
        assert!(!plain.instrs.iter().any(|i| matches!(i, Instr::Muon { .. })));
        assert_eq!(
            plain
                .instrs
                .iter()
                .filter(|i| matches!(i, Instr::AdamW { .. }))
                .count(),
            ps.len()
        );
        assert!(!plain.bufs.iter().any(|b| b.name.starts_with("muon_m.")));
    }
}

#[test]
fn adamw_stays_the_default_and_muon_settings_validate() {
    let c: TrainConfig = serde_json::from_value(serde_json::json!({
        "run": "d", "model": tiny(8), "data": "synthetic:5", "batch": 2, "seq_len": 8,
        "max_steps": 30, "lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 3, "out_dir": "/tmp"
    }))
    .unwrap();
    assert_eq!(c.optimizer, "adamw");
    assert!(!c.muon());
    assert_eq!(
        (
            c.muon_lr,
            c.muon_momentum,
            c.muon_ns_steps,
            c.muon_weight_decay
        ),
        (None, None, None, None)
    );
    assert_eq!(
        (
            c.muon_lr(),
            c.muon_momentum(),
            c.muon_ns_steps(),
            c.muon_weight_decay()
        ),
        (0.02, 0.95, 5, 0.0)
    );
    c.validate().unwrap();
    // An AdamW state.json carries none of the new keys, and its optimiser
    // tensors keep their names: m under the parameter, v under "v.<name>".
    let v = serde_json::to_value(&c).unwrap();
    for k in [
        "optimizer",
        "muon_lr",
        "muon_momentum",
        "muon_ns_steps",
        "muon_weight_decay",
    ] {
        assert!(v.get(k).is_none(), "{k} must not appear for AdamW");
    }
    let keys: Vec<String> = optim_slots(&c).into_iter().map(|s| s.1).collect();
    let mut want = Vec::new();
    for p in model::params(&c.model) {
        want.push(p.name.clone());
        want.push(format!("v.{}", p.name));
    }
    assert_eq!(keys, want);

    let mut m = c.clone();
    m.optimizer = "muon".into();
    m.validate().unwrap();
    let v = serde_json::to_value(&m).unwrap();
    assert_eq!(v["optimizer"], "muon");
    assert!(v.get("muon_lr").is_none(), "defaults stay implicit");
    m.muon_lr = Some(0.01);
    let back: TrainConfig = serde_json::from_value(serde_json::to_value(&m).unwrap()).unwrap();
    assert_eq!(back, m);
    // lr_muon follows the shape of the AdamW schedule: lr(step) · muon_lr / lr.
    for step in [0, 2, 3, 17, 29, 30] {
        let (a, b) = (m.lr_at(step), m.muon_lr_for(m.lr_at(step)));
        assert!((b / a - 10.0).abs() < 1e-5, "step {step}: {a} vs {b}");
    }

    let bad = |f: &dyn Fn(&mut TrainConfig), base: &TrainConfig| {
        let mut x = base.clone();
        f(&mut x);
        x.validate().unwrap_err()
    };
    assert!(
        bad(&|x| x.muon_lr = Some(0.01), &c).contains("optimizer"),
        "muon_* without optimizer muon"
    );
    assert!(bad(&|x| x.optimizer = "sgd".into(), &c).contains("adamw|muon"));
    bad(&|x| x.muon_momentum = Some(1.0), &m);
    bad(&|x| x.muon_ns_steps = Some(0), &m);
    bad(&|x| x.muon_lr = Some(0.0), &m);
    bad(&|x| x.lr = 0.0, &m);
    bad(&|x| x.muon_weight_decay = Some(-0.1), &m);
}

/// A run meant as Muon must never quietly train AdamW: the British spelling of
/// the key selects Muon, and with optimizer "adamw" every explicit muon_* key
/// is refused, also one equal to its default (review: {"optimiser": "muon",
/// "muon_lr": 0.02} used to train AdamW).
#[test]
fn optimiser_alias_and_explicit_muon_keys_never_fall_back_to_adamw() {
    let base = serde_json::json!({
        "run": "d", "model": tiny(8), "data": "synthetic:5", "batch": 2, "seq_len": 8,
        "max_steps": 30, "lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 3, "out_dir": "/tmp"
    });
    let parse = |extra: serde_json::Value| -> Result<TrainConfig, String> {
        let mut v = base.clone();
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        let c: TrainConfig = serde_json::from_value(v).map_err(|e| e.to_string())?;
        c.validate()?;
        Ok(c)
    };

    // "optimiser" is the same key as "optimizer".
    for extra in [
        serde_json::json!({"optimiser": "muon"}),
        serde_json::json!({"optimiser": "muon", "muon_lr": 0.02}),
        serde_json::json!({"optimiser": "muon", "muon_lr": 0.05}),
    ] {
        let c = parse(extra.clone()).unwrap();
        assert!(c.muon(), "{extra} must select Muon");
        // state.json uses the canonical spelling.
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["optimizer"], "muon");
        assert!(v.get("optimiser").is_none());
    }
    assert!(!parse(serde_json::json!({"optimiser": "adamw"}))
        .unwrap()
        .muon());
    // Both spellings at once is ambiguous: serde refuses the duplicate.
    let e = parse(serde_json::json!({"optimizer": "adamw", "optimiser": "muon"})).unwrap_err();
    assert!(e.contains("duplicate"), "{e}");

    // Under AdamW (explicit or default), any muon_* key is refused, also at its
    // default value; the message names the keys.
    for (k, x) in [
        ("muon_lr", serde_json::json!(0.02)),
        ("muon_lr", serde_json::json!(0.05)),
        ("muon_momentum", serde_json::json!(0.95)),
        ("muon_ns_steps", serde_json::json!(5)),
        ("muon_weight_decay", serde_json::json!(0.0)),
    ] {
        for opt in [None, Some("adamw")] {
            let mut extra = serde_json::json!({ k: x });
            if let Some(o) = opt {
                extra["optimizer"] = serde_json::json!(o);
            }
            let e = parse(extra.clone()).unwrap_err();
            assert!(
                e.contains(k) && e.contains("\"optimizer\": \"muon\""),
                "{extra}: {e}"
            );
        }
    }
    // A misspelt key that serde ignores, with a muon_* setting next to it.
    let e = parse(serde_json::json!({"optimzer": "muon", "muon_lr": 0.02})).unwrap_err();
    assert!(e.contains("muon_lr"), "{e}");

    // Explicit defaults under Muon are accepted and round-trip exactly.
    let c = parse(serde_json::json!({
        "optimizer": "muon", "muon_lr": 0.02, "muon_momentum": 0.95,
        "muon_ns_steps": 5, "muon_weight_decay": 0.0
    }))
    .unwrap();
    assert_eq!(c.muon_lr, Some(0.02));
    let back: TrainConfig = serde_json::from_value(serde_json::to_value(&c).unwrap()).unwrap();
    assert_eq!(back, c);
}

fn muon_cfg(run: &str, out: &str, steps: usize) -> TrainConfig {
    let mut c = cfg(run, out, steps);
    c.optimizer = "muon".into();
    c.muon_lr = Some(0.02);
    c
}

fn new_err(c: TrainConfig) -> String {
    let (sink, _) = collect();
    match Trainer::new(c, sink) {
        Ok(_) => panic!("Trainer::new should have failed"),
        Err(e) => e,
    }
}

#[test]
fn muon_resume_equals_uninterrupted_run() {
    let out = std::env::temp_dir().join(format!("forge-muon-resume-{}", std::process::id()));
    let o = out.to_str().unwrap();
    // One thread: RMSNORM_BWD sums the norm-gain gradients with a rayon
    // fold/reduce whose split order varies between runs, so with more threads
    // even two uninterrupted runs can differ in the last bit of the norm
    // moments (AdamW too). MUON itself is pool-size independent (see
    // muon_newton_schulz_orthogonalises); this test compares every buffer.
    let mcfg = |run: &str| {
        let mut c = muon_cfg(run, o, 12);
        c.threads = 1;
        c
    };
    let (s1, _) = collect();
    let mut a = Trainer::new(mcfg("ref"), s1).unwrap();
    for _ in 0..12 {
        a.train_step().unwrap();
    }
    let (s2, _) = collect();
    let mut b = Trainer::new(mcfg("cut"), s2).unwrap();
    for _ in 0..7 {
        b.train_step().unwrap();
    }
    let dir = b.checkpoint().unwrap();
    // optim.safetensors: Muon momentum for the hidden matrices, AdamW m/v for the rest.
    let (opt, _) = ckpt::load_safetensors(&dir.join("optim.safetensors")).unwrap();
    assert_eq!(opt.len(), optim_slots(&b.cfg).len());
    for p in model::params(&b.cfg.model) {
        let has = |k: String| opt.contains_key(&k);
        let mk = format!("muon.{}", p.name);
        if p.muon() {
            assert!(has(mk.clone()) && !has(p.name.clone()) && !has(format!("v.{}", p.name)));
            assert!(opt[&mk].data.iter().any(|x| *x != 0.0), "{mk} is zero");
        } else {
            assert!(has(p.name.clone()) && has(format!("v.{}", p.name)) && !has(mk));
        }
    }
    let st: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!(st["config"]["optimizer"], "muon");

    let (s3, _) = collect();
    let mut rc = mcfg("cut");
    rc.resume_from = Some(dir.to_str().unwrap().into());
    let mut c = Trainer::new(rc, s3).unwrap();
    assert_eq!(c.step, 7);
    for _ in 0..5 {
        c.train_step().unwrap();
    }
    for (buf, _, _) in optim_slots(&a.cfg) {
        assert_eq!(
            a.param(&buf).unwrap(),
            c.param(&buf).unwrap(),
            "{buf} differs after resume"
        );
    }
    for p in model::params(&a.cfg.model) {
        assert_eq!(
            a.param(&p.name).unwrap(),
            c.param(&p.name).unwrap(),
            "{} differs after resume",
            p.name
        );
    }
    // Muon really drives the hidden matrices: AdamW from the same start ends elsewhere.
    let (s4, _) = collect();
    let mut wc = cfg("adamw", o, 12);
    wc.threads = 1;
    let mut w = Trainer::new(wc, s4).unwrap();
    for _ in 0..12 {
        w.train_step().unwrap();
    }
    assert_ne!(
        w.param("blocks.0.attn.wq.weight"),
        a.param("blocks.0.attn.wq.weight")
    );
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn init_optim_and_resume_refuse_mixing_optimizers() {
    let out = std::env::temp_dir().join(format!("forge-muon-mix-{}", std::process::id()));
    let o = out.to_str().unwrap();
    let (s1, _) = collect();
    let mut a = Trainer::new(cfg("adam", o, 3), s1).unwrap();
    let (s2, _) = collect();
    let mut m = Trainer::new(muon_cfg("muon", o, 3), s2).unwrap();
    for _ in 0..3 {
        a.train_step().unwrap();
        m.train_step().unwrap();
    }
    let adam_dir = a.checkpoint().unwrap().to_str().unwrap().to_string();
    let muon_dir = m.checkpoint().unwrap().to_str().unwrap().to_string();

    let mut c = muon_cfg("x", o, 3);
    c.init_from = Some(adam_dir.clone());
    c.init_optim = true;
    let e = new_err(c);
    assert!(
        e.contains("adamw optimiser state") && e.contains("mixing"),
        "{e}"
    );
    let mut c = cfg("y", o, 3);
    c.init_from = Some(muon_dir.clone());
    c.init_optim = true;
    let e = new_err(c);
    assert!(
        e.contains("muon optimiser state") && e.contains("mixing"),
        "{e}"
    );
    let mut c = muon_cfg("adam", o, 6);
    c.resume_from = Some(adam_dir.clone());
    assert!(new_err(c).contains("optimizer"));
    let mut c = cfg("muon", o, 6);
    c.resume_from = Some(muon_dir.clone());
    assert!(new_err(c).contains("optimizer"));

    // Weights alone switch optimiser freely.
    let mut c = muon_cfg("z", o, 3);
    c.init_from = Some(adam_dir);
    let (s3, _) = collect();
    assert!(Trainer::new(c, s3).is_ok());
    // Muon → Muon with init_optim continues the momentum and the AdamW moments.
    let mut c = muon_cfg("w", o, 3);
    c.init_from = Some(muon_dir);
    c.init_optim = true;
    let (s4, log) = collect();
    let t = Trainer::new(c, s4).unwrap();
    for (buf, _, _) in optim_slots(&t.cfg) {
        assert_eq!(t.param(&buf).unwrap(), m.param(&buf).unwrap(), "{buf}");
    }
    assert!(log
        .lock()
        .unwrap()
        .iter()
        .any(|v| v["msg"]
            .as_str()
            .is_some_and(|s| s.contains("AdamW moments and Muon momentum")
                && s.contains("3 steps behind them"))));
    std::fs::remove_dir_all(out).ok();
}

#[test]
fn muon_trainer_learns_a_periodic_stream() {
    let out = std::env::temp_dir().join(format!("forge-muon-toy-{}", std::process::id()));
    let (sink, log) = collect();
    let mut tr = Trainer::new(muon_cfg("muon-toy", out.to_str().unwrap(), 120), sink).unwrap();
    let first = tr.train_step().unwrap();
    let last = tr.run(None).unwrap();
    assert!(first > 1.5, "initial loss {first} should be near ln(8)");
    assert!(last < 0.1, "Muon did not converge: {first} -> {last}");
    let log = log.lock().unwrap();
    let tele: Vec<&Value> = log.iter().filter(|v| v["type"] == "telemetry").collect();
    // lr_muon = lr(step) · muon_lr / lr on every line: 2× the AdamW lr here.
    for t in &tele {
        let (ml, l) = (t["muon_lr"].as_f64().unwrap(), t["lr"].as_f64().unwrap());
        assert!(
            (ml / l - 2.0).abs() < 1e-5,
            "step {}: muon_lr {ml}, lr {l}",
            t["step"]
        );
    }
    let lastt = tele.last().unwrap();
    assert!(lastt["lr"].as_f64().unwrap() < 1.1e-3, "end of the cosine");
    assert!(lastt["ops"]["MUON"]["flops"].as_f64().unwrap() > 0.0);
    assert!(log.iter().any(|v| v["msg"]
        .as_str()
        .is_some_and(|s| s.starts_with("optimizer muon: 14 matrices"))));
    std::fs::remove_dir_all(out).ok();
}
