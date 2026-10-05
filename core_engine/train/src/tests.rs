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
