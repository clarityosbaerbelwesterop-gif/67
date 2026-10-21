"""python -m accel.generate --ckpt DIR --prompts-file F [--max-new 32] [--temperature 0] [--top-k 40] [--seed 0]
                           [--stop-id ID] [--batch 64] [--precision bf16|f32]

`forge generate --prompts-file` on a JAX device (cli/src/main.rs generate_cmd, train/src/decode.rs): one JSON
array of prompt ids per line; prompt i decodes with seed + i; each prompt is cut from the left so that prompt +
max_new fits the context; greedy at temperature 0, else top-k sampling at the temperature; decoding of a prompt
stops after its stop id. Prints one line per prompt in order: {"type":"generate","index":i,"ids":[...],
"decoder":"accel-kv","batch":B}. The sampling stream is JAX's (random.fold_in per prompt and step), not forge's,
so sampled ids differ from forge's while greedy ids agree up to float rounding.

Decoding: `--batch` prompts at a time with a key/value cache of max_seq_len positions. The first
min(prompt length) - 1 positions are encoded in one pass, then one position per step for every row (a row still
inside its prompt is fed its next prompt id, the others their last sampled id), so every row keeps positions
0..t like forge. Weights are bf16 on the device with f32 accumulation (--precision bf16, forge's amx-bf16
arithmetic) or f32.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time

import numpy as np

from .config import ConfigError, ModelConfig

PREFILL = (0, 64, 128, 256)  # the only prefill lengths, so a model compiles at most four decoding programs


def cut_prompt(ids: list[int], ctx: int, max_new: int) -> list[int]:
    keep = max(ctx - max_new, 1)
    out = list(ids[max(len(ids) - keep, 0):])
    return out or [0]


@functools.lru_cache(maxsize=None)
def _programs(cfg: ModelConfig, precision: str, B: int, S: int):
    import jax
    import jax.numpy as jnp
    from .model import ein, rms_norm, rope, rope_table

    H, KV, hd, L, d = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_layers, cfg.dim
    rep = H // KV
    alpha = np.float32(1.0) / np.sqrt(np.float32(hd), dtype=np.float32)
    cos_t, sin_t = (jnp.asarray(a) for a in rope_table(S, hd, cfg.rope_theta))
    cdt = jnp.bfloat16 if precision == "bf16" else jnp.float32

    def layer_p(params, i):
        pre = f"blocks.{i}."
        return {k[len(pre):]: v for k, v in params.items() if k.startswith(pre)}

    def ffn(x, p):
        xn = rms_norm(x, p["ffn_norm.weight"], cfg.norm_eps, d)
        h1 = ein("...d,fd->...f", xn, p["ffn.w1.weight"], precision)
        h3 = ein("...d,fd->...f", xn, p["ffn.w3.weight"], precision)
        g = (h1 * (np.float32(1.0) / (np.float32(1.0) + jnp.exp(-h1)))) * h3
        return x + ein("...f,df->...d", g, p["ffn.w2.weight"], precision)

    def prefill(params, ids):
        """ids [B,T] at positions 0..T-1 -> caches [(k,v) per layer] with these positions filled."""
        T = ids.shape[1]
        x = jnp.take(params["tok_emb.weight"], ids, axis=0).astype(jnp.float32)
        cos, sin = cos_t[:T], sin_t[:T]
        mask = jnp.asarray(np.tril(np.ones((T, T), bool)))
        caches = []
        for i in range(L):
            p = layer_p(params, i)
            xn = rms_norm(x, p["attn_norm.weight"], cfg.norm_eps, d)
            q = rope(ein("btd,fd->btf", xn, p["attn.wq.weight"], precision).reshape(B, T, H, hd), cos, sin)
            k = rope(ein("btd,fd->btf", xn, p["attn.wk.weight"], precision).reshape(B, T, KV, hd), cos, sin)
            v = ein("btd,fd->btf", xn, p["attn.wv.weight"], precision).reshape(B, T, KV, hd)
            s = jnp.einsum("bikrd,bjkd->bkrij", q.reshape(B, T, KV, rep, hd), k) * alpha
            s = jnp.where(mask, s, -jnp.inf)
            pr = jax.nn.softmax(s, axis=-1)
            o = jnp.einsum("bkrij,bjkd->bikrd", pr, v).reshape(B, T, H * hd)
            x = x + ein("btf,df->btd", o, p["attn.wo.weight"], precision)
            x = ffn(x, p)
            kc = jnp.zeros((B, S, KV, hd), cdt).at[:, :T].set(k.astype(cdt))
            vc = jnp.zeros((B, S, KV, hd), cdt).at[:, :T].set(v.astype(cdt))
            caches.append((kc, vc))
        return caches

    def step(params, caches, tok, t):
        """tok [B] at position t (same for every row) -> logits [B,V] for position t+1, updated caches."""
        x = jnp.take(params["tok_emb.weight"], tok, axis=0).astype(jnp.float32)
        cos = jax.lax.dynamic_slice_in_dim(cos_t, t, 1)
        sin = jax.lax.dynamic_slice_in_dim(sin_t, t, 1)
        live = (jnp.arange(S) <= t)[None, None, None, :]
        out = []
        for i in range(L):
            p = layer_p(params, i)
            kc, vc = caches[i]
            xn = rms_norm(x, p["attn_norm.weight"], cfg.norm_eps, d)
            q = rope(ein("bd,fd->bf", xn, p["attn.wq.weight"], precision).reshape(B, 1, H, hd), cos, sin)[:, 0]
            k = rope(ein("bd,fd->bf", xn, p["attn.wk.weight"], precision).reshape(B, 1, KV, hd), cos, sin)
            v = ein("bd,fd->bf", xn, p["attn.wv.weight"], precision).reshape(B, 1, KV, hd)
            kc = jax.lax.dynamic_update_slice(kc, k.astype(cdt), (0, t, 0, 0))
            vc = jax.lax.dynamic_update_slice(vc, v.astype(cdt), (0, t, 0, 0))
            s = jnp.einsum("bkrd,bskd->bkrs", q.reshape(B, KV, rep, hd), kc.astype(jnp.float32)) * alpha
            s = jnp.where(live, s, -jnp.inf)
            pr = jax.nn.softmax(s, axis=-1)
            o = jnp.einsum("bkrs,bskd->bkrd", pr, vc.astype(jnp.float32)).reshape(B, H * hd)
            x = x + ein("bf,df->bd", o, p["attn.wo.weight"], precision)
            x = ffn(x, p)
            out.append((kc, vc))
        xf = rms_norm(x, params["norm.weight"], cfg.norm_eps, d)
        return ein("bd,vd->bv", xf, params["tok_emb.weight"], precision), out

    def pick(logits, keys, t, temperature, top_k):
        greedy = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        vals, idx = jax.lax.top_k(logits, top_k)
        ks = jax.vmap(lambda k: jax.random.fold_in(k, t))(keys)
        temp = jnp.maximum(temperature, np.float32(1e-6))
        choice = jax.vmap(lambda k, v: jax.random.categorical(k, (v - v[0]) / temp))(ks, vals)
        sampled = jnp.take_along_axis(idx, choice[:, None], axis=1)[:, 0].astype(jnp.int32)
        return jnp.where(temperature > 0, sampled, greedy)

    def decode(params, prompts, plen, keys, nvalid, n_steps, max_new, stop, temperature, start, top_k):
        """prompts [B,S] (padded), plen [B]; rows >= nvalid are padding; positions < start are prefilled; only
        start and top_k are compile-time. Returns ids [B,S] (the first count[b] of row b are valid), count [B]."""
        caches = prefill(params, prompts[:, :start]) if start > 0 else [
            (jnp.zeros((B, S, KV, hd), cdt), jnp.zeros((B, S, KV, hd), cdt)) for _ in range(L)]
        out = jnp.zeros((B, S), jnp.int32)
        cnt = jnp.zeros((B,), jnp.int32)
        done = jnp.arange(B) >= nvalid
        last = jnp.zeros((B,), jnp.int32)

        def cond(c):
            t, _, _, _, done, _ = c
            return (t < start + n_steps) & ~jnp.all(done)

        def body(c):
            t, caches, out, cnt, done, last = c
            in_prompt = t < plen
            tok = jnp.where(in_prompt, prompts[:, jnp.minimum(t, S - 1)], last)
            logits, caches = step(params, caches, tok, t)
            nxt = pick(logits, keys, t, temperature, top_k)
            emit = (t >= plen - 1) & ~done
            out = jnp.where(emit[:, None] & (jnp.arange(S)[None, :] == cnt[:, None]), nxt[:, None], out)
            cnt = cnt + emit.astype(jnp.int32)
            done = done | (emit & ((nxt == stop) | (cnt >= max_new)))
            last = jnp.where(emit, nxt, last)
            return t + 1, caches, out, cnt, done, last

        _, _, out, cnt, _, _ = jax.lax.while_loop(cond, body, (jnp.int32(start), caches, out, cnt, done, last))
        return out, cnt

    return jax.jit(decode, static_argnums=(9, 10))


def generate(cfg: ModelConfig, params, prompts: list[list[int]], max_new: int, temperature: float = 0.0,
             top_k: int = 40, seed: int = 0, stop: int | None = None, batch: int = 64, precision: str = "bf16"):
    """Yields (index, generated ids) in prompt order; `params` are device arrays (see device_params)."""
    import jax
    import jax.numpy as jnp
    S = cfg.max_seq_len
    max_new = max(1, min(max_new, S - 1))
    cut = [cut_prompt(p, S, max_new) for p in prompts]
    stop_id = -1 if stop is None else int(stop)
    for b0 in range(0, len(cut), batch):
        rows = cut[b0:b0 + batch]
        n = len(rows)
        rows_p = rows + [[0]] * (batch - n)
        plen = np.array([len(r) for r in rows_p], np.int32)
        arr = np.zeros((batch, S), np.int32)
        for i, r in enumerate(rows_p):
            arr[i, :len(r)] = r
        start = max(x for x in PREFILL if x <= int(plen[:n].min()) - 1)
        n_steps = min(int(plen[:n].max()) - 1 + max_new - start, S - start)
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(jnp.arange(b0, b0 + batch))
        fn = _programs(cfg, precision, batch, S)
        out, cnt = fn(params, jnp.asarray(arr), jnp.asarray(plen), keys, jnp.int32(n), jnp.int32(n_steps),
                      jnp.int32(max_new), jnp.int32(stop_id), jnp.float32(temperature), start, int(top_k))
        out, cnt = np.asarray(out), np.asarray(cnt)
        for i in range(n):
            ids = [int(x) for x in out[i, :cnt[i]]]
            if stop is not None and stop in ids:
                ids = ids[:ids.index(stop) + 1]
            yield b0 + i, ids


def load(ckpt: str) -> tuple[ModelConfig, dict]:
    """state.json config.model and model.safetensors in F32 or BF16 (a scripts/import_export.py directory of a bf16
    export); BF16 stays BF16 on the host."""
    from pathlib import Path
    import ml_dtypes
    from . import forge_io as fio
    d = Path(ckpt)
    cfg = ModelConfig.from_dict(json.loads((d / "state.json").read_text())["config"]["model"])
    ts, _, _ = fio.R.read_safetensors(d / "model.safetensors")
    dts = {"F32": np.dtype("<f4"), "BF16": np.dtype(ml_dtypes.bfloat16)}
    w = {}
    for name, t in ts.items():
        if t["dtype"] not in dts:
            raise ValueError(f"{d}: {name}: dtype {t['dtype']} (F32 or BF16 only)")
        w[name] = np.frombuffer(t["data"], dts[t["dtype"]]).reshape(t["shape"])
    return cfg, fio.check_weights(w, cfg.param_specs(), str(d / "model.safetensors"))


def device_params(w: dict, precision: str):
    import jax.numpy as jnp
    dt = jnp.bfloat16 if precision == "bf16" else jnp.float32
    return {k: jnp.asarray(v, dt) if v.ndim >= 2 else jnp.asarray(v, jnp.float32) for k, v in w.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="accel.generate", description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-id", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--precision", choices=("bf16", "f32"), default="bf16")
    a = ap.parse_args(argv)
    try:
        prompts = [json.loads(l) for l in open(a.prompts_file) if l.strip()]
        cfg, w = load(a.ckpt)
        params = device_params(w, a.precision)
        del w
        t0 = time.time()
        for i, ids in generate(cfg, params, prompts, a.max_new, a.temperature, a.top_k, a.seed, a.stop_id,
                               a.batch, a.precision):
            print(json.dumps({"type": "generate", "index": i, "ids": ids, "decoder": "accel-kv", "batch": a.batch,
                              "seconds": round(time.time() - t0, 3)}), flush=True)
    except (ConfigError, ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"accel: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
