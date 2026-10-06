//! `forge` — command line of the 67 Forge code accelerator.
//!
//!   forge bench    [--threads N] [--config run.json] [--steps K]
//!   forge train    --config run.json
//!   forge diloco   --config run.json --rank R --world W --peers h0:p,h1:p,.. [--listen ADDR]
//!                  [--inner-steps H] [--outer-lr 0.7] [--momentum 0.9] [--compression none|bf16|int8[:B]]
//!   forge merge    --base DIR --child DIR --child DIR [--method ties|linear] [--density D] [--lambda L] --out DIR
//!   forge eval     --ckpt DIR --data META [--split val|train] [--batch B] [--seq T] [--batches K]
//!   forge generate --ckpt DIR --prompt-ids 1,2,3 | --prompts-file F [--max-new N] [--temperature T] [--top-k K]
//!                  [--seed S] [--stop-id ID] [--decoder kv|window]
//!   forge generate --ckpt DIR --prompt "text" [--tokenizer tokenizer.json] [...]   (text in, text out)
//!   forge tokenize [--tokenizer tokenizer.json] (--text S | --jsonl F | --decode 1,2,3)
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

/// One DiLoCo worker on a forge-comm TCP ring: `max_steps` local AdamW steps
/// in rounds of `--inner-steps`, exchanging only pseudo-gradients. Every rank
/// samples its own data windows (seed + rank); rank 0 writes the checkpoint.
fn diloco_cmd(a: &Args) -> Result<(), String> {
    use forge_train::diloco::{Collective, DiLoCo, RingCollective};
    let mut cfg = load_config(&a.need("--config")?)?;
    let rank: usize = a.num("--rank", 0)?;
    let world: usize = a.num("--world", 1)?;
    let peers = a
        .need("--peers")?
        .split(',')
        .map(|p| p.trim().parse().map_err(|e| format!("peer {p}: {e}")))
        .collect::<Result<Vec<std::net::SocketAddr>, String>>()?;
    let listen: std::net::SocketAddr = match a.get("--listen") {
        Some(l) => l.parse().map_err(|e| format!("--listen {l}: {e}"))?,
        None => {
            let p = peers.get(rank).ok_or("--rank outside --peers")?;
            let any: std::net::IpAddr = if p.is_ipv4() {
                [0u8; 4].into()
            } else {
                [0u16; 8].into()
            };
            (any, p.port()).into()
        }
    };
    let inner: usize = a.num("--inner-steps", 50)?;
    let outer_lr: f32 = a.num("--outer-lr", 0.7)?;
    let momentum: f32 = a.num("--momentum", 0.9)?;
    let comp_name = a.get("--compression").unwrap_or_else(|| "none".into());
    let compression =
        forge_comm::Compression::parse(&comp_name).ok_or("--compression none|bf16|int8[:block]")?;
    let timeout = std::time::Duration::from_secs(a.num("--timeout-s", 120u64)?);
    if inner == 0 {
        return Err("--inner-steps must be positive".into());
    }
    cfg.seed += rank as u64;
    if rank != 0 {
        cfg.run = format!("{}-w{rank}", cfg.run);
    }
    let rounds = cfg.max_steps.div_ceil(inner);
    let ring = forge_comm::Ring::connect(rank, world, listen, &peers, timeout)
        .map_err(|e| format!("ring: {e}"))?;
    let mut coll = RingCollective { ring, compression };
    let mut tr = trainer::Trainer::new(cfg.clone(), trainer::stdout_sink())?;
    if cfg.autotune {
        tr.autotune();
    }
    let mut d = DiLoCo::new(&tr, inner, outer_lr, momentum);
    d.synchronise(&mut tr, &mut coll)?;
    let wall = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    for round in 1..=rounds {
        let t0 = Instant::now();
        let loss = d.round(&mut tr, &mut coll)?;
        let st = coll.ring.stats();
        println!(
            "{}",
            json!({"type": "diloco", "run": cfg.run, "rank": rank, "world": coll.world(), "round": round,
                   "rounds": rounds, "step": tr.step, "inner_steps": inner, "loss": loss, "round_ms": wall(t0),
                   "comm_ms": st.last_ms, "bytes_sent": st.bytes_sent, "bytes_received": st.bytes_received,
                   "compression": comp_name, "ts": unix_now()})
        );
    }
    let (val_loss, val_acc) = tr.evaluate()?;
    println!(
        "{}",
        json!({"type": "eval", "run": cfg.run, "rank": rank, "step": tr.step, "val_loss": val_loss, "val_acc": val_acc, "ts": unix_now()})
    );
    if rank == 0 {
        tr.checkpoint()?;
    }
    coll.ring
        .barrier()
        .map_err(|e| format!("final barrier: {e}"))?;
    println!(
        "{}",
        json!({"type": "done", "run": cfg.run, "rank": rank, "step": tr.step, "val_loss": val_loss, "ts": unix_now()})
    );
    Ok(())
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
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
    if shape.1 < 2 || shape.1 > cfg.max_seq_len || shape.0 == 0 || shape.2 == 0 {
        return Err(format!(
            "--seq must be in 2..={} for this model, --batch and --batches positive",
            cfg.max_seq_len
        ));
    }
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
    let (max_new, temp, top_k, seed) = (
        a.num("--max-new", 32)?,
        a.num("--temperature", 0.0)?,
        a.num("--top-k", 40)?,
        a.num("--seed", 0)?,
    );
    let mut stop: Option<u32> = a
        .get("--stop-id")
        .map(|v| v.parse().map_err(|_| "bad --stop-id".to_string()))
        .transpose()?;
    if stop.is_none() && a.get("--prompt").is_some() {
        stop = tokenizer(a)?.eos();
    }
    // Default: KV-cache decoding. `--decoder window` (or an explicit `--ctx`)
    // re-runs the ISA window program for every token, as a reference.
    type Gen = Box<dyn FnMut(&[u32], u64) -> Result<Vec<u32>, String>>;
    let window = a.get("--decoder").as_deref() == Some("window") || a.get("--ctx").is_some();
    let (mut generate, decoder): (Gen, &str) = if window {
        let mut s =
            forge_train::Sampler::new(&cfg, &w, a.num("--ctx", cfg.max_seq_len)?, threads(a)?)?;
        (
            Box::new(move |ids, seed| s.generate(ids, max_new, temp, top_k, seed, stop)),
            "window",
        )
    } else {
        let mut d = forge_train::decode::Decoder::new(&cfg, &w, threads(a)?)?;
        (
            Box::new(move |ids, seed| d.generate(ids, max_new, temp, top_k, seed, stop)),
            "kv",
        )
    };
    // Batch mode: one JSON array of prompt ids per line; weights are loaded once.
    if let Some(file) = a.get("--prompts-file") {
        let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
        for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let ids: Vec<u32> =
                serde_json::from_str(line).map_err(|e| format!("{file}:{}: {e}", i + 1))?;
            let t0 = Instant::now();
            let out = generate(&ids, seed + i as u64)?;
            println!(
                "{}",
                json!({"type": "generate", "index": i, "ids": out, "decoder": decoder, "seconds": t0.elapsed().as_secs_f64()})
            );
        }
        return Ok(());
    }
    if let Some(text) = a.get("--prompt") {
        let tok = tokenizer(a)?;
        let ids = tok.encode(&text);
        let out = generate(&ids, seed)?;
        println!(
            "{}",
            json!({"type": "generate", "prompt": text, "prompt_ids": ids, "ids": out, "text": tok.decode(&out), "decoder": decoder})
        );
        return Ok(());
    }
    let ids = parse(&a.need("--prompt-ids")?)?;
    let out = generate(&ids, seed)?;
    println!(
        "{}",
        json!({"type": "generate", "prompt_ids": ids, "ids": out, "decoder": decoder})
    );
    Ok(())
}

fn tokenizer(a: &Args) -> Result<forge_data::Tokenizer, String> {
    let path = a
        .get("--tokenizer")
        .unwrap_or_else(|| "data/stores/base/tokenizer.json".into());
    forge_data::Tokenizer::load(&path)
}

/// Encode text with the SCP BPE tokenizer, or decode ids. `--jsonl` reads one
/// JSON string per line and writes one JSON array of ids per line.
fn tokenize_cmd(a: &Args) -> Result<(), String> {
    let tok = tokenizer(a)?;
    if let Some(ids) = a.get("--decode") {
        let ids: Vec<u32> = ids
            .split(',')
            .map(|x| x.trim().parse().map_err(|_| format!("bad id {x}")))
            .collect::<Result<_, _>>()?;
        println!("{}", json!({"text": tok.decode(&ids)}));
    } else if let Some(file) = a.get("--jsonl") {
        let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
        let mut out = std::io::BufWriter::new(std::io::stdout().lock());
        for (i, line) in text.lines().enumerate() {
            let s: String =
                serde_json::from_str(line).map_err(|e| format!("{file}:{}: {e}", i + 1))?;
            use std::io::Write;
            writeln!(out, "{}", json!(tok.encode(&s))).map_err(|e| e.to_string())?;
        }
    } else {
        let s = a.need("--text")?;
        println!(
            "{}",
            json!({"ids": tok.encode(&s), "vocab_size": tok.vocab_size()})
        );
    }
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
        "train" => a.need("--config").and_then(|p| load_config(&p)).and_then(|mut cfg| {
            if let Some(dir) = a.get("--resume") {
                cfg.resume_from = Some(dir);
            }
            let mut tr = trainer::Trainer::new(cfg, trainer::stdout_sink())?;
            tr.run(Some(trainer::stdin_control())).map(|_| ())
        }),
        "diloco" => diloco_cmd(&a),
        "merge" => merge_cmd(&a),
        "eval" => eval_cmd(&a),
        "generate" => generate_cmd(&a),
        "logits" => logits_cmd(&a),
        "tokenize" => tokenize_cmd(&a),
        "disasm" => disasm(&a),
        "engines" => {
            println!("{}", json!({"available": forge_kernels::available().iter().map(|v: &GemmVariant| v.name()).collect::<Vec<_>>()}));
            Ok(())
        }
        _ => Err("usage: forge bench|train|diloco|merge|eval|generate|tokenize|disasm|engines [options] (see core_engine/cli/src/main.rs)".into()),
    };
    if let Err(e) = result {
        eprintln!("forge: {e}");
        std::process::exit(1);
    }
}
