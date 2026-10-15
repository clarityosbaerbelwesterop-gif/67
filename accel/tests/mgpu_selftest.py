"""Run: XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu nice -n 19 <venv python> tests.py [all|resume_child ...]"""
import sys, os, json, shutil, subprocess
import numpy as np, jax
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_r, "accel")) and _r != "/": _r = os.path.dirname(_r)
sys.path.insert(0, _r)  # repo root (box copy: this dir; repo: two levels up)
from accel.config import ModelConfig
from accel import model as M
from accel import mgpu as G

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
MC = ModelConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=128,
                 max_seq_len=32, rope_theta=10000.0, norm_eps=1e-5)
def hp(prec="f32", **kw): return G.HP(precision=prec, lr=3e-3, min_lr=3e-4, warmup_steps=2, max_steps=10, **kw)
def init(): return M.init_params(MC, 7, exact=False)

def run(data, fsdp, steps, prec="f32", devices=None):
    mesh = G.make_mesh(data, fsdp, devices)
    st = G.init_state(MC, mesh, init())
    st, ls = G.train(MC, hp(prec), mesh, st, steps)
    return st, ls, mesh

def t1_t4():
    print("params", MC.num_params())
    ref_st, ref, _ = run(1, 1, 5, devices=jax.devices()[:1])
    refp = G.gather(ref_st[0])
    print("1dev losses", [f"{x:.7f}" for x in ref])
    for d, f in [(8, 1), (1, 8), (2, 4)]:
        st, ls, mesh = run(d, f, 5)
        dl = max(abs(a - b) for a, b in zip(ls, ref))
        dp = max(float(np.max(np.abs(a - refp[k]))) for k, a in G.gather(st[0]).items())
        print(f"mesh {d}x{f}: losses {[f'{x:.7f}' for x in ls]} max|dloss|={dl:.2e} max|dparam| after 5 steps={dp:.2e}")
    for prec in ["bf16"]:
        _, ls1, _ = run(1, 1, 5, prec, jax.devices()[:1]); _, ls8, _ = run(2, 4, 5, prec)
        print(f"{prec}: 1dev {[f'{x:.5f}' for x in ls1]} 2x4 {[f'{x:.5f}' for x in ls8]} max|d|={max(abs(a-b) for a,b in zip(ls1,ls8)):.2e}")

def t2():
    for d, f in [(8, 1), (1, 8), (2, 4)]:
        mesh = G.make_mesh(d, f); st, _ = G.train(MC, hp(), mesh, G.init_state(MC, mesh, init()), 1)
        p, m, v, _ = st
        for name in ["tok_emb.weight", "blocks.0.ffn.w2.weight", "blocks.0.attn_norm.weight"]:
            for lab, tr in [("param", p), ("adam_m", m), ("adam_v", v)]:
                a = tr[name]; sh = a.addressable_shards
                print(f"mesh {d}x{f} {lab:6s} {name:26s} global {a.shape} spec {a.sharding.spec} "
                      f"shard {sh[0].data.shape} bytes/dev {[s.data.nbytes for s in sh]}")
        tot = lambda tr: sum(s.data.nbytes for a in tr.values() for s in a.addressable_shards if s.device == jax.devices()[0])
        full = sum(a.nbytes for a in p.values())
        print(f"mesh {d}x{f}: device0 holds params {tot(p)} B + m {tot(m)} B + v {tot(v)} B; full params {full} B")

def t3():
    N = 6
    mesh = G.make_mesh(2, 4)
    st, straight = G.train(MC, hp(), mesh, G.init_state(MC, mesh, init()), N)
    sp = G.gather(st[0]); sm = G.gather(st[1])
    st, first = G.train(MC, hp(), mesh, G.init_state(MC, mesh, init()), N // 2)
    ck = os.path.join(TMP, "ck_half"); G.save(ck, MC, hp(), *st)
    out = os.path.join(TMP, "resumed.npz")
    subprocess.run([sys.executable, __file__, "resume_child", ck, out, str(N - N // 2)], check=True)
    r = np.load(out); second = list(r["losses"])
    print("straight", [f"{x:.7f}" for x in straight])
    print("resumed ", [f"{x:.7f}" for x in first + second])
    print("max|dloss|", max(abs(a - b) for a, b in zip(straight, first + second)))
    print("max|dparam|", max(float(np.max(np.abs(r["p_" + k] - sp[k]))) for k in sp),
          "max|dadam_m|", max(float(np.max(np.abs(r["m_" + k] - sm[k]))) for k in sm), "step", int(r["step"]))

def resume_child(ck, out, n):
    mesh = G.make_mesh(4, 2)  # resume on a DIFFERENT mesh shape on purpose
    p, m, v, t, _ = G.load(ck, MC, mesh)
    (p, m, v, t), ls = G.train(MC, hp(), mesh, (p, m, v, t), int(n))
    np.savez(out, losses=np.array(ls), step=int(t), **{"p_" + k: a for k, a in G.gather(p).items()},
             **{"m_" + k: a for k, a in G.gather(m).items()})

def s2_check(path="/workspace/grow-1b/ckpt/s2_1b"):
    st = json.load(open(f"{path}/state.json")); mc = ModelConfig.from_dict(st["config"]["model"])
    f = f"{path}/model.safetensors"; h, off = G.st_header(f); h.pop("__metadata__", None)
    specs = dict(mc.param_specs()); assert set(specs) == set(h), "name mismatch"
    assert all(tuple(h[k]["shape"]) == specs[k] for k in specs)
    print("s2 header ok:", len(h), "tensors,", mc.num_params(), "params")
    mesh = G.make_mesh(2, 4); sh = G.shardings(mc, mesh)
    abs_p = {k: jax.ShapeDtypeStruct(s, np.float32, sharding=sh[k]) for k, s in specs.items()}
    out = jax.eval_shape(lambda p: M.loss(p, mc, (np.zeros((8, 512), np.int32),) * 2), abs_p)
    print("eval_shape forward on 8 fake devices ok:", out)
    per = sum(int(np.prod(sh[k].shard_shape(s))) * 4 for k, s in specs.items())
    print(f"per-device param bytes on 2x4 (fsdp=4): {per/2**30:.3f} GiB (full {sum(int(np.prod(s)) for s in specs.values())*4/2**30:.3f} GiB)")
    for name in ["blocks.0.ffn.w1.weight", "tok_emb.weight"]:
        a = G.lazy_sharded(f, name, sh[name], (h, off))
        mm = np.memmap(f, "<f4", "r", off + h[name]["data_offsets"][0], tuple(h[name]["shape"]))
        ok = all(np.array_equal(np.asarray(s.data), mm[s.index]) for s in a.addressable_shards)
        print(f"lazy shard load {name} {a.shape} spec {a.sharding.spec} shard {a.addressable_shards[0].data.shape} matches file: {ok}")

BASE_V3 = "/workspace/67-train/data/out/base-v3/train.meta.json"
MCV = ModelConfig(vocab_size=8192, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=128,
                  max_seq_len=64, rope_theta=10000.0, norm_eps=1e-5)
def hpd(): return hp(data=BASE_V3, seq=64, batch=16, accum=2, seed=67101, clean_tsv="", allow_unfiltered=True)

def t5_data():
    N = 6; mesh = G.make_mesh(2, 4)
    ids = []
    st, straight = G.train(MCV, hpd(), mesh, G.init_state(MCV, mesh, M.init_params(MCV, 7, exact=False)), N, record_ids=ids)
    sp = G.gather(st[0])
    ids1 = []
    st, first = G.train(MCV, hpd(), mesh, G.init_state(MCV, mesh, M.init_params(MCV, 7, exact=False)), N // 2, record_ids=ids1)
    ck = os.path.join(TMP, "ck_data"); G.save(ck, MCV, hpd(), *st)
    out = os.path.join(TMP, "data_res.npz")
    subprocess.run([sys.executable, __file__, "data_child", ck, out, str(N - N // 2)], check=True)
    r = np.load(out)
    allids = np.concatenate([np.stack(ids1), r["ids"]])
    print("Store:", BASE_V3, "train tokens", len(G.Store(BASE_V3, 8192)))
    print("first row step0 micro0 ids[:8]", ids[0][0, 0, :8].tolist())
    print("batch ids identical (6 steps x A2 x B16 x T64):", bool(np.array_equal(np.stack(ids), allids)),
          "distinct rows step0:", len({tuple(rw) for rw in ids[0].reshape(-1, 64)}))
    print("straight", [f"{x:.7f}" for x in straight]); print("resumed ", [f"{x:.7f}" for x in first + list(r["losses"])])
    print("max|dloss|", max(abs(a - b) for a, b in zip(straight, first + list(r["losses"]))),
          "max|dparam|", max(float(np.max(np.abs(r["p_" + k] - sp[k]))) for k in sp))

def data_child(ck, out, n):
    mesh = G.make_mesh(4, 2)
    p, m, v, t, _ = G.load(ck, MCV, mesh); ids = []
    st, ls = G.train(MCV, hpd(), mesh, (p, m, v, t), int(n), record_ids=ids)
    np.savez(out, ids=np.stack(ids), losses=np.array(ls), **{"p_" + k: a for k, a in G.gather(st[0]).items()})

def t6_attn():
    import jax.numpy as jnp
    mc = ModelConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=128,
                     max_seq_len=64, rope_theta=10000.0, norm_eps=1e-5)
    p = {k: jnp.asarray(a) for k, a in M.init_params(mc, 3, exact=False).items()}
    # make attention non-trivial: scale q/k up so softmax is peaky
    for i in range(2):
        p[f"blocks.{i}.attn.wq.weight"] = p[f"blocks.{i}.attn.wq.weight"] * 20
        p[f"blocks.{i}.attn.wk.weight"] = p[f"blocks.{i}.attn.wk.weight"] * 20
    r = np.random.default_rng(0)
    for T in (64, 50, 37):
        x = jnp.asarray(r.integers(0, 256, (2, T)), jnp.int32); y = jnp.asarray(r.integers(0, 256, (2, T)), jnp.int32)
        for prec in ("f32", "bf16"):
            ref_lg = jax.jit(lambda p, x: M.forward(p, mc, x, prec, False, "naive"))(p, x)
            ref_l, ref_g = jax.jit(jax.value_and_grad(lambda p: M.loss(p, mc, (x, y), prec, False, "fast", "naive")))(p)
            for be in ("blockwise:16", "blockwise:16:8", "blockwise:128", "xla"):
                for remat in (False, True):
                    if be == "xla" and remat: continue
                    lg = jax.jit(lambda p, x: M.forward(p, mc, x, prec, remat, be))(p, x)
                    l, g = jax.jit(jax.value_and_grad(lambda p: M.loss(p, mc, (x, y), prec, remat, "fast", be)))(p)
                    dg = max(float(jnp.max(jnp.abs(g[k] - ref_g[k]))) for k in g)
                    gmax = max(float(jnp.max(jnp.abs(ref_g[k]))) for k in g)
                    print(f"T={T:2d} {prec:4s} {be:15s} remat={remat!s:5s} max|dlogit|={float(jnp.max(jnp.abs(lg-ref_lg))):.2e} "
                          f"|dloss|={abs(float(l-ref_l)):.2e} max|dgrad|={dg:.2e} (max|grad| {gmax:.2e})")
    try:
        q = jnp.zeros((1, 16, 4, 16), jnp.bfloat16)
        jax.jit(lambda q: jax.nn.dot_product_attention(q, q, q, is_causal=True, implementation="cudnn"))(q)
        print("cudnn ran (unexpected on CPU)")
    except Exception as e:
        print("cudnn on CPU raises as expected:", type(e).__name__, str(e).splitlines()[0][:160])
    print("resolve('auto') on this box ->", __import__("accel.attn", fromlist=["x"]).resolve("auto"))

def t7_hp():
    h = G.HP()
    print("champion HP:", {k: getattr(h, k) for k in ("lr","min_lr","warmup_steps","lr_schedule","max_steps","decay_steps","weight_decay","grad_clip")},
          "B1,B2,EPS", G.B1, G.B2, G.EPS, "wsd decay steps", h.wsd_decay_steps(), "trunk at", h.max_steps - h.wsd_decay_steps())
    print("lr at", {s: float(h.lr_at(s)) for s in (0, 49, 99, 100, 4594, 4595, 4596, 5170, 5743, 5744)})

TSV = "/workspace/contam-check/dedup-v4/basev3-full/kept_t50_drop20.tsv"
def hpc(verify=False): return hpd().__class__(**{**hpd().__dict__, "clean_tsv": TSV, "verify_sha": verify, "allow_unfiltered": False})

def t8_clean():
    import time
    t0 = time.time(); cs = G.CleanStore(BASE_V3, TSV, 8192, verify_sha=True)
    print(f"CleanStore init incl. sha256: {time.time()-t0:.1f}s (sha256 alone {cs.sha_seconds:.1f}s) -> sha OK")
    t0 = time.time(); G.CleanStore(BASE_V3, TSV, 8192, verify_sha=False); print(f"init without sha: {time.time()-t0:.2f}s")
    print("kept docs", len(cs.doc), "virtual tokens (docs incl. EOS)", cs.n, "of train.bin", len(cs.base.train),
          "| sum len", int((cs.seg_len - 1).sum()))
    eos = np.asarray(cs.base.train[cs.seg_start + cs.seg_len - 1]); print("token at start+len for all kept docs: unique", np.unique(eos)[:5].tolist(), "count", len(np.unique(eos)))
    # windows: many random starts + forced segment-boundary-crossing starts
    rng = G.Rng(99); T = 512
    st = np.concatenate([cs.starts(rng, "train", 2000, T), cs.cum[1:2001] - 100])
    v = st[:, None] + np.arange(T + 1)[None, :]; ph = cs.physical(v)
    seg = np.searchsorted(cs.seg_start, ph, side="right") - 1
    inside = (seg >= 0) & (ph >= cs.seg_start[seg]) & (ph <= cs.seg_start[seg] + cs.seg_len[seg] - 1)
    x, y = cs.windows("train", st, T)
    ok_tok = np.array_equal(np.concatenate([x, y[:, -1:]], 1), np.asarray(cs.base.train[ph]).astype(np.int32))
    cross = int(np.sum(np.diff(seg, axis=1) != 0))
    print(f"windows {len(st)} x {T+1} tokens: all inside kept [start,start+len]: {bool(inside.all())}, "
          f"segment crossings {cross}, gathered ids match physical positions: {ok_tok}")
    # trainer-path resume determinism on clean stream
    N = 6; mesh = G.make_mesh(2, 4); ids = []
    st_, straight = G.train(MCV, hpc(), mesh, G.init_state(MCV, mesh, M.init_params(MCV, 7, exact=False)), N, record_ids=ids)
    sp = G.gather(st_[0]); ids1 = []
    st_, first = G.train(MCV, hpc(), mesh, G.init_state(MCV, mesh, M.init_params(MCV, 7, exact=False)), N // 2, record_ids=ids1)
    ck = os.path.join(TMP, "ck_clean"); G.save(ck, MCV, hpc(), *st_); out = os.path.join(TMP, "clean.npz")
    subprocess.run([sys.executable, __file__, "clean_child", ck, out, str(N - N // 2)], check=True)
    r = np.load(out)
    print("clean ids identical:", bool(np.array_equal(np.stack(ids), np.concatenate([np.stack(ids1), r["ids"]]))))
    print("straight", [f"{x:.7f}" for x in straight]); print("resumed ", [f"{x:.7f}" for x in first + list(r["losses"])])
    print("max|dloss|", max(abs(a - b) for a, b in zip(straight, first + list(r["losses"]))),
          "max|dparam|", max(float(np.max(np.abs(r["p_" + k] - sp[k]))) for k in sp))
    # trainer batches inside kept docs too
    allst = np.concatenate([G.ShardedData(hpc(), mesh, 8192).starts(s_).reshape(-1) for s_ in range(N)])
    print("trainer starts virtual-only, all < n-T-1:", bool((allst < cs.n - 65).all()))

def clean_child(ck, out, n):
    mesh = G.make_mesh(4, 2); p, m, v, t, _ = G.load(ck, MCV, mesh); ids = []
    st, ls = G.train(MCV, hpc(), mesh, (p, m, v, t), int(n), record_ids=ids)
    np.savez(out, ids=np.stack(ids), losses=np.array(ls), **{"p_" + k: a for k, a in G.gather(st[0]).items()})

def t9_s2lr():
    for name, ms in (("s2", 5744), ("s2", 20000)):
        h = G.HP.preset(name, max_steps=ms); d = h.wsd_decay_steps(); tr = ms - d
        steps = (0, 99, 249, 499, 500, tr - 1, tr, tr + d // 2, ms - 1, ms)
        print(f"preset {name} max_steps={ms}: decay {d} steps, trunk {tr}; lr", {s_: f"{float(h.lr_at(s_)):.6e}" for s_ in steps})

def t10_failclosed():
    import io
    d = os.path.join(TMP, "fc"); os.makedirs(d, exist_ok=True)
    tsv_ok = TSV
    # copy of TSV (5.7 MB) next to a bad / missing report.json
    shutil.copy(TSV, os.path.join(d, "kept.tsv"))
    def rep(sha):
        json.dump({"train_bin_sha256_expected": sha}, open(os.path.join(d, "report.json"), "w"))
    def cli(*args):
        root = os.path.dirname(os.path.dirname(G.__file__)) if G.__file__.endswith("accel/mgpu.py") else os.path.dirname(G.__file__)
        code = ("import sys\nsys.path.insert(0, sys.argv[1])\ntry:\n    from accel import mgpu as G\n"
                "    assert G.__file__.endswith('accel/mgpu.py')\nexcept (ImportError, AssertionError):\n    import mgpu as G\n"
                "sys.exit(G.main(sys.argv[2:]))")
        r = subprocess.run([sys.executable, "-c", code, root, "--check-data", "--data", BASE_V3, *args], capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1][:150]
    cases = [("missing tsv", ["--clean-tsv", os.path.join(d, "nope.tsv")], None),
             ("report.json missing", ["--clean-tsv", os.path.join(d, "kept.tsv")], None),
             ("report.json unreadable", ["--clean-tsv", os.path.join(d, "kept.tsv")], "garbage"),
             ("sha mismatch", ["--clean-tsv", os.path.join(d, "kept.tsv")], "0" * 64),
             ("empty clean_tsv", ["--clean-tsv", ""], None),
             ("good (real tsv, sha verified)", ["--clean-tsv", tsv_ok], None)]
    for name, args, r_ in cases:
        rp = os.path.join(d, "report.json")
        if os.path.exists(rp): os.remove(rp)
        if r_ == "garbage": open(rp, "w").write("{not json")
        elif r_: rep(r_)
        code, msg = cli(*args)
        print(f"{name:32s} exit={code}  {msg}")
    # in-process: train() with no clean_tsv must raise, not silently use raw data
    mesh = G.make_mesh(2, 4)
    try:
        G.train(MCV, hp(data=BASE_V3, seq=64, clean_tsv=""), mesh, G.init_state(MCV, mesh, M.init_params(MCV, 7, exact=False)), 1)
        print("train without clean_tsv: NO ERROR (bad)")
    except G.DataSafetyError as e:
        print("train without clean_tsv raises DataSafetyError:", e)
    print("HP() default clean_tsv:", G.HP().clean_tsv, "allow_unfiltered:", G.HP().allow_unfiltered)

def t11_run1():
    h = G.HP.preset("s2_run1"); d = h.wsd_decay_steps()
    print(f"s2_run1: max_steps {h.max_steps} warmup {h.warmup_steps} decay {d} (from step {h.max_steps-d}) peak {h.lr} min {h.min_lr}")
    print("lr", {s_: f"{float(h.lr_at(s_)):.6e}" for s_ in (0, 99, 149, 299, 300, 500, 799, 800, 850, 900, 950, 999, 1000)})
    # multi-run resume: 3 chained processes (0->4, 4->8, 8->12) vs straight 12, recording lr used per step
    base = dict(max_steps=12, warmup_steps=2, decay_steps=0)
    H = lambda: G.HP.preset("s2_run1", **base, data=BASE_V3, clean_tsv=TSV, verify_sha=False, seq=64, batch=16, accum=2, seed=67101)
    mesh = G.make_mesh(2, 4)
    lrs = []
    st, straight = G.train(MCV, H(), mesh, G.init_state(MCV, mesh, M.init_params(MCV, 7, exact=False)), 12,
                           log=lambda t, l, g: lrs.append(float(H().lr_at(t - 1))))
    sp = G.gather(st[0])
    st, l0 = G.train(MCV, H(), mesh, G.init_state(MCV, mesh, M.init_params(MCV, 7, exact=False)), 4)
    ck = os.path.join(TMP, "r1"); G.save(ck, MCV, H(), *st)
    chained = list(l0); hyp = []
    for k in range(2):
        out = os.path.join(TMP, f"r{k+2}.npz"); nxt = os.path.join(TMP, f"r{k+2}")
        subprocess.run([sys.executable, __file__, "lr_child", ck, out, "4", nxt], check=True)
        r = np.load(out); chained += list(r["losses"]); hyp += list(r["lr"]); ck = nxt
    print("lr straight steps 4..11", [f"{x:.4e}" for x in lrs[4:]]); print("lr chained (from state.json step)", [f"{x:.4e}" for x in hyp])
    print("lr identical:", lrs[4:] == hyp, "| max|dloss|", max(abs(a-b) for a, b in zip(straight, chained)),
          "| max|dparam|", max(float(np.max(np.abs(r["p_" + k] - sp[k]))) for k in sp), "| final step", int(r["step"]))
    print("straight", [f"{x:.6f}" for x in straight]); print("chained ", [f"{x:.6f}" for x in chained])

def lr_child(ck, out, n, save_to):
    base = dict(max_steps=12, warmup_steps=2, decay_steps=0)
    h = G.HP.preset("s2_run1", **base, data=BASE_V3, clean_tsv=TSV, verify_sha=False, seq=64, batch=16, accum=2, seed=67101)
    mesh = G.make_mesh(4, 2); p, m, v, t, _ = G.load(ck, MCV, mesh); lr = []
    st, ls = G.train(MCV, h, mesh, (p, m, v, t), int(n), log=lambda t_, l, g: lr.append(float(h.lr_at(t_ - 1))))
    G.save(save_to, MCV, h, *st)
    np.savez(out, losses=np.array(ls), lr=np.array(lr), step=int(st[3]), **{"p_" + k: a for k, a in G.gather(st[0]).items()})

def t12_export():
    import ml_dtypes
    mesh = G.make_mesh(2, 4); st, _ = G.train(MC, hp(), mesh, G.init_state(MC, mesh, init()), 2)
    out = G.finish_run(os.path.join(TMP, "run"), MC, hp(), *st)
    e = out["export"]; n = MC.num_params()
    print("export", os.path.basename(e["path"]), "bytes", e["bytes"], "for", n, "params (2 B/param =", 2 * n, "+ header)", "tensors", e["tensors"])
    ck = os.listdir(out["ckpt"]); sz = {f: os.path.getsize(os.path.join(out["ckpt"], f)) for f in ck}
    print("full ckpt files", sz)
    h, off = G.st_header(e["path"]); h.pop("__metadata__")
    full = G.gather(st[0]); ok = True; md = 0.0
    with open(e["path"], "rb") as f:
        for k, en in h.items():
            f.seek(off + en["data_offsets"][0]); b = f.read(en["data_offsets"][1] - en["data_offsets"][0])
            a = np.frombuffer(b, ml_dtypes.bfloat16).reshape(en["shape"])
            ok &= np.array_equal(a, full[k].astype(ml_dtypes.bfloat16)); md = max(md, float(np.max(np.abs(a.astype(np.float32) - full[k]))))
    print("bf16 file == RNE(f32 params):", bool(ok), "max|bf16-f32|", f"{md:.2e}", "no optimizer tensors:", not any(k.startswith("v.") for k in h))
    mc1 = ModelConfig.from_dict(json.load(open("/workspace/grow-1b/ckpt/s2_1b/state.json"))["config"]["model"])
    print("estimate for s2 (959,541,120 params): bf16 export", mc1.num_params() * 2, "B =", f"{mc1.num_params()*2/1e9:.3f} GB", "| full f32 model+optim", mc1.num_params() * 12, "B")

if __name__ == "__main__":
    assert jax.device_count() == 8, jax.device_count()
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    if a == "resume_child": resume_child(*sys.argv[2:]); sys.exit()
    if a == "lr_child": lr_child(*sys.argv[2:]); sys.exit()
    if a == "clean_child": clean_child(*sys.argv[2:]); sys.exit()
    if a == "data_child": data_child(*sys.argv[2:]); sys.exit()
    os.makedirs(TMP, exist_ok=True)
    try:
        for name in (["t1_t4", "t2", "t3", "s2_check", "t5_data", "t6_attn", "t7_hp", "t8_clean", "t9_s2lr", "t10_failclosed", "t11_run1", "t12_export"] if a == "all" else a.split(",")):
            print(f"===== {name}", flush=True); globals()[name](); sys.stdout.flush()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
