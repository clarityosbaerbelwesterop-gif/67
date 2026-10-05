//! `forge` — command line of the 67 Forge code accelerator.
//!
//!   forge bench    [--threads N] [--config run.json] [--steps K]
//!   forge train    --config run.json
//!   forge merge    --base DIR --child DIR --child DIR [--method ties|linear] [--density D] [--lambda L] --out DIR
//!   forge eval     --ckpt DIR --data META [--split val|train] [--batch B] [--seq T] [--batches K]
//!   forge generate --ckpt DIR --prompt-ids 1,2,3 [--max-new N] [--temperature T] [--top-k K] [--seed S]
//!   forge disasm   --config run.json [--limit N]

use forge_isa::{compile, disassemble, CompileOptions};
use forge_kernels::{GemmVariant, Pool, Trans};
use forge_train::config::{ModelConfig, TrainConfig};
use forge_train::{ckpt, data::Split, merge, model, trainer};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Peak dense BF16 throughput of one NVIDIA B300: NVIDIA lists 2.25 PF (HGX B300)
/// to 2.5 PF (GB300 NVL72). The upper figure is used so our ratio is never overstated.
const B300_DENSE_BF16_TFLOPS: f64 = 2500.0;

struct Args(Vec<String>);

impl Args {
    fn get(&self, k: &str) -> Option<String> {
        self.0
            .iter()
            .position(|a| a == k)
            .and_then(|i| self.0.get(i + 1))
            .cloned()
    }
    fn all(&self, k: &str) -> Vec<String> {
        self.0
            .windows(2)
            .filter(|w| w[0] == k)
            .map(|w| w[1].clone())
            .collect()
    }
    fn need(&self, k: &str) -> Result<String, String> {
        self.get(k).ok_or(format!("missing {k}"))
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T, String> {
        self.get(k).map_or(Ok(default), |v| {
            v.parse().map_err(|_| format!("bad value for {k}: {v}"))
        })
    }
}

fn load_config(path: &str) -> Result<TrainConfig, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
}

fn ckpt_model(dir: &str) -> Result<(ModelConfig, BTreeMap<String, ckpt::Tensor>, String), String> {
    let d = Path::new(dir);
    let state: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(d.join("state.json"))
            .map_err(|e| format!("{dir}/state.json: {e}"))?,
    )
    .map_err(|e| e.to_string())?;
    let cfg: ModelConfig = serde_json::from_value(state["config"]["model"].clone())
        .map_err(|e| format!("{dir}: {e}"))?;
    let path = d.join("model.safetensors");
    let (w, _) = ckpt::load_safetensors(&path).map_err(|e| e.to_string())?;
    let sha = ckpt::sha256_file(&path).map_err(|e| e.to_string())?;
    Ok((cfg, w, sha))
}

fn threads(a: &Args) -> Result<usize, String> {
    a.num(
        "--threads",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    )
}

fn bench(a: &Args) -> Result<(), String> {
    let th = threads(a)?;
    let pool = Pool::new(th);
    let shapes = [
        (Trans::N, Trans::T, 4096, 512, 512),
        (Trans::N, Trans::T, 4096, 1408, 512),
        (Trans::N, Trans::N, 4096, 512, 1408),
        (Trans::T, Trans::N, 512, 512, 4096),
        (Trans::N, Trans::N, 2048, 2048, 2048),
    ];
    let mut best_gemm = 0.0f64;
    for (ta, tb, m, n, k) in shapes {
        let r = forge_kernels::autotune(&pool, ta, tb, m, n, k);
        best_gemm = best_gemm.max(r.first().map_or(0.0, |x| x.gflops));
        println!(
            "{}",
            json!({"type": "gemm", "shape": [m, n, k], "trans": [format!("{ta:?}"), format!("{tb:?}")], "threads": th,
                   "results": r.iter().map(|x| json!({"variant": x.variant.name(), "gflops": x.gflops})).collect::<Vec<_>>()})
        );
    }
    let mut train_tflops = None;
    if let Some(path) = a.get("--config") {
        let mut cfg = load_config(&path)?;
        let steps: usize = a.num("--steps", 5)?;
        cfg.max_steps = steps + 1;
        cfg.threads = th;
        cfg.out_dir = std::env::temp_dir()
            .join("forge-bench")
            .to_string_lossy()
            .into();
        let mut tr = trainer::Trainer::new(cfg.clone(), Box::new(|_| {}))?;
        tr.autotune();
        tr.train_step()?; // warm-up
        tr.machine.reset_stats();
        let t0 = Instant::now();
        for _ in 0..steps {
            tr.train_step()?;
        }
        let secs = t0.elapsed().as_secs_f64();
        let flops: f64 = tr.machine.op_stats().iter().map(|(_, s)| s.flops).sum();
        let toks = (cfg.batch * cfg.seq_len * cfg.grad_accum * steps) as f64;
        train_tflops = Some(flops / secs / 1e12);
        println!(
            "{}",
            json!({"type": "train_bench", "run": cfg.run, "params": cfg.model.num_params(), "steps": steps,
                   "tokens_per_s": toks / secs, "tflops": flops / secs / 1e12, "step_ms": secs * 1e3 / steps as f64,
                   "ops": tr.machine.op_stats().iter().map(|(o, s)| json!({"op": o.name(), "share": s.ns as f64 / (secs * 1e9), "gflops": s.flops / (s.ns.max(1) as f64)})).collect::<Vec<_>>()})
        );
    }
    let tf = train_tflops.unwrap_or(best_gemm / 1e3);
    println!(
        "{}",
        json!({"type": "verdict", "measured_tflops": tf, "basis": if train_tflops.is_some() { "end-to-end training" } else { "best GEMM" },
               "b300_dense_bf16_tflops": B300_DENSE_BF16_TFLOPS,
               "fraction_of_one_b300": tf / B300_DENSE_BF16_TFLOPS,
               "fraction_of_100_b300": tf / (100.0 * B300_DENSE_BF16_TFLOPS),
               "engines": forge_kernels::available().iter().map(|v| v.name()).collect::<Vec<_>>()})
    );
    Ok(())
}

fn merge_cmd(a: &Args) -> Result<(), String> {
    let (bcfg, base, base_sha) = ckpt_model(&a.need("--base")?)?;
    let children: Vec<String> = a.all("--child");
    if children.len() < 2 {
        return Err("merge needs at least two --child".into());
    }
    let mut cw = Vec::new();
    let mut lineage = Vec::new();
    for c in &children {
        let (ccfg, w, sha) = ckpt_model(c)?;
        if ccfg != bcfg {
            return Err(format!("{c}: model config differs from the base"));
        }
        let parent: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(Path::new(c).join("state.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if parent["parent_sha256"].as_str() != Some(base_sha.as_str()) {
            return Err(format!(
                "{c} was not trained from this base (parent {} != {base_sha})",
                parent["parent_sha256"]
            ));
        }
        cw.push(w);
        lineage.push(json!({"dir": c, "sha256": sha}));
    }
    let method = match a.get("--method").as_deref().unwrap_or("ties") {
        "ties" => merge::Method::Ties {
            density: a.num("--density", 0.5)?,
        },
        "linear" => merge::Method::Linear,
        m => return Err(format!("unknown method {m}")),
    };
    let lambda: f32 = a.num("--lambda", 1.0)?;
    let merged = merge::merge(&base, &cw, method, lambda)?;
    let out = PathBuf::from(a.need("--out")?);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut meta = BTreeMap::new();
    meta.insert("format".into(), "pt".into());
    let sha = ckpt::save_safetensors(&out.join("model.safetensors"), &merged, &meta)
        .map_err(|e| e.to_string())?;
    let state = json!({"step": 0, "config": {"model": bcfg}, "model_sha256": sha, "parent_sha256": base_sha,
                       "merge": {"method": format!("{method:?}"), "lambda": lambda, "children": lineage}});
    std::fs::write(
        out.join("state.json"),
        serde_json::to_string_pretty(&state).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"type": "merge", "out": out, "sha256": sha, "method": format!("{method:?}"), "lambda": lambda})
    );
    Ok(())
}

fn eval_cmd(a: &Args) -> Result<(), String> {
    let dir = a.need("--ckpt")?;
    let (cfg, w, sha) = ckpt_model(&dir)?;
    let split = if a.get("--split").as_deref() == Some("train") {
        Split::Train
    } else {
        Split::Val
    };
    let shape = (
        a.num("--batch", 8)?,
        a.num("--seq", cfg.max_seq_len.min(256))?,
        a.num("--batches", 20)?,
    );
    let (loss, acc) = forge_train::evaluate_weights(
        &cfg,
        &w,
        &a.need("--data")?,
        split,
        shape,
        a.num("--seed", 1234)?,
        threads(a)?,
    )?;
    println!(
        "{}",
        json!({"type": "eval", "ckpt": dir, "sha256": sha, "data": a.get("--data"), "split": format!("{split:?}"),
               "tokens": shape.0 * shape.1 * shape.2, "loss": loss, "perplexity": loss.exp(), "next_token_acc": acc})
    );
    Ok(())
}

fn generate_cmd(a: &Args) -> Result<(), String> {
    let (cfg, w, _) = ckpt_model(&a.need("--ckpt")?)?;
    let parse = |s: &str| -> Result<Vec<u32>, String> {
        s.split(',')
            .map(|x| x.trim().parse().map_err(|_| format!("bad id {x}")))
            .collect()
    };
    let mut s = forge_train::Sampler::new(&cfg, &w, a.num("--ctx", cfg.max_seq_len)?, threads(a)?)?;
    let (max_new, temp, top_k, seed) = (
        a.num("--max-new", 32)?,
        a.num("--temperature", 0.0)?,
        a.num("--top-k", 40)?,
        a.num("--seed", 0)?,
    );
    let stop: Option<u32> = a
        .get("--stop-id")
        .map(|v| v.parse().map_err(|_| "bad --stop-id".to_string()))
        .transpose()?;
    // Batch mode: one JSON array of prompt ids per line; weights are loaded once.
    if let Some(file) = a.get("--prompts-file") {
        let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
        for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let ids: Vec<u32> =
                serde_json::from_str(line).map_err(|e| format!("{file}:{}: {e}", i + 1))?;
            let t0 = Instant::now();
            let out = s.generate(&ids, max_new, temp, top_k, seed + i as u64, stop)?;
            println!(
                "{}",
                json!({"type": "generate", "index": i, "ids": out, "seconds": t0.elapsed().as_secs_f64()})
            );
        }
        return Ok(());
    }
    let ids = parse(&a.need("--prompt-ids")?)?;
    let out = s.generate(&ids, max_new, temp, top_k, seed, stop)?;
    println!(
        "{}",
        json!({"type": "generate", "prompt_ids": ids, "ids": out})
    );
    Ok(())
}

fn logits_cmd(a: &Args) -> Result<(), String> {
    let (cfg, w, _) = ckpt_model(&a.need("--ckpt")?)?;
    let ids: Vec<u32> = a
        .need("--prompt-ids")?
        .split(',')
        .map(|s| s.trim().parse().map_err(|_| format!("bad id {s}")))
        .collect::<Result<_, _>>()?;
    let mut s = forge_train::Sampler::new(&cfg, &w, a.num("--ctx", cfg.max_seq_len)?, threads(a)?)?;
    println!(
        "{}",
        json!({"type": "logits", "prompt_ids": ids, "logits": s.next_logits(&ids)?})
    );
    Ok(())
}

fn disasm(a: &Args) -> Result<(), String> {
    let cfg = load_config(&a.need("--config")?)?;
    let (p, rep) = compile(
        model::build(
            &cfg.model,
            cfg.batch,
            cfg.seq_len,
            model::Head::Train {
                accum: cfg.grad_accum,
            },
        ),
        CompileOptions::default(),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"type": "compile", "report": format!("{rep:?}"), "bytecode_bytes": p.encode().len()})
    );
    let limit: usize = a.num("--limit", 80)?;
    for line in disassemble(&p).lines().take(limit) {
        println!("{line}");
    }
    Ok(())
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = argv
        .split_first()
        .map(|(c, r)| (c.clone(), r.to_vec()))
        .unwrap_or_default();
    let a = Args(rest);
    let result = match cmd.as_str() {
        "bench" => bench(&a),
        "train" => a.need("--config").and_then(|p| load_config(&p)).and_then(|cfg| {
            let mut tr = trainer::Trainer::new(cfg, trainer::stdout_sink())?;
            tr.run(Some(trainer::stdin_control())).map(|_| ())
        }),
        "merge" => merge_cmd(&a),
        "eval" => eval_cmd(&a),
        "generate" => generate_cmd(&a),
        "logits" => logits_cmd(&a),
        "disasm" => disasm(&a),
        "engines" => {
            println!("{}", json!({"available": forge_kernels::available().iter().map(|v: &GemmVariant| v.name()).collect::<Vec<_>>()}));
            Ok(())
        }
        _ => Err("usage: forge bench|train|merge|eval|generate|disasm|engines [options] (see core_engine/cli/src/main.rs)".into()),
    };
    if let Err(e) = result {
        eprintln!("forge: {e}");
        std::process::exit(1);
    }
}
