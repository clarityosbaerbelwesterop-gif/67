"""Switchable causal attention. q [B,T,H,hd], k/v [B,T,KV,hd] (GQA: q head h reads kv head h//rep).
backends: "naive" (reference, T x T scores, = original model.py math), "blockwise[:QB[:KB]]" (pure-JAX
flash-style: query blocks x key blocks, online softmax, O(T*block) memory), "xla" / "cudnn"
(jax.nn.dot_product_attention, is_causal=True; cudnn needs a GPU and fp16/bf16 inputs)."""
from __future__ import annotations
import jax, jax.numpy as jnp, numpy as np

f32 = np.float32

def parse(backend: str):
    parts = backend.split(":")
    name = parts[0]
    if name not in ("naive", "blockwise", "xla", "cudnn"):
        raise ValueError(f"attention backend {backend!r}")
    qb = int(parts[1]) if len(parts) > 1 else 128
    kb = int(parts[2]) if len(parts) > 2 else qb
    return name, qb, kb

def naive(q, k, v, ein, precision):
    B, T, H, hd = q.shape; KV = k.shape[2]; rep = H // KV
    alpha = f32(1.0) / np.sqrt(f32(hd), dtype=f32)
    mask = jnp.asarray(np.tril(np.ones((T, T), bool)))
    qq = q.reshape(B, T, KV, rep, hd)
    s = ein("bikrd,bjkd->bkrij", qq, k, precision) * alpha
    s = jnp.where(mask, s, -jnp.inf)
    mx = jax.lax.stop_gradient(jnp.max(s, axis=-1, keepdims=True))
    e = jnp.where(mask, jnp.exp(s - mx), f32(0.0))
    pr = e * (f32(1.0) / jnp.sum(e, axis=-1, keepdims=True))
    return ein("bkrij,bjkd->bikrd", pr, v, precision).reshape(B, T, H * hd)

def blockwise(q, k, v, ein, precision, qb: int, kb: int):
    B, T, H, hd = q.shape; KV = k.shape[2]; rep = H // KV
    alpha = f32(1.0) / np.sqrt(f32(hd), dtype=f32)
    qb, kb = min(qb, T), min(kb, T)
    nq, nk = -(-T // qb), -(-T // kb)
    q = jnp.pad(q, ((0, 0), (0, nq * qb - T), (0, 0), (0, 0))).reshape(B, nq, qb, KV, rep, hd)
    k = jnp.pad(k, ((0, 0), (0, nk * kb - T), (0, 0), (0, 0))).reshape(B, nk, kb, KV, hd)
    v = jnp.pad(v, ((0, 0), (0, nk * kb - T), (0, 0), (0, 0))).reshape(B, nk, kb, KV, hd)
    neg = f32(-1e30)  # finite: padded/fully-masked rows stay NaN-free

    def one_q(iq):
        qi = q[:, iq]                                   # [B,qb,KV,rep,hd]
        qpos = iq * qb + jnp.arange(qb)

        def body(carry, ik):
            m, l, acc = carry                           # m,l [B,KV,rep,qb]; acc [B,KV,rep,qb,hd]
            kj, vj = k[:, ik], v[:, ik]
            kpos = ik * kb + jnp.arange(kb)
            msk = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] < T)
            s = ein("bikrd,bjkd->bkrij", qi, kj, precision) * alpha
            s = jnp.where(msk, s, neg)
            mn = jnp.maximum(m, jax.lax.stop_gradient(jnp.max(s, axis=-1)))
            p = jnp.where(msk, jnp.exp(s - mn[..., None]), f32(0.0))
            c = jnp.exp(m - mn)
            acc = acc * c[..., None] + ein("bkrij,bjkd->bkrid", p, vj, precision)
            return (mn, l * c + jnp.sum(p, axis=-1), acc), None

        init = (jnp.full((B, KV, rep, qb), neg, jnp.float32), jnp.zeros((B, KV, rep, qb), jnp.float32),
                jnp.zeros((B, KV, rep, qb, hd), jnp.float32))
        # causal: key blocks after this query block contribute nothing; scan only over the needed ones
        # would need a dynamic trip count, so we scan all and mask (2x FLOPs worst case, correct).
        (m, l, acc), _ = jax.lax.scan(body, init, jnp.arange(nk))
        o = acc / jnp.maximum(l, f32(1e-30))[..., None]  # [B,KV,rep,qb,hd]
        return jnp.transpose(o, (0, 3, 1, 2, 4))         # [B,qb,KV,rep,hd]

    o = jax.lax.map(one_q, jnp.arange(nq))               # [nq,B,qb,KV,rep,hd]
    o = jnp.transpose(o, (1, 0, 2, 3, 4, 5)).reshape(B, nq * qb, H * hd)
    return o[:, :T]

def sdpa(q, k, v, impl: str, precision):
    B, T, H, hd = q.shape
    dt = jnp.bfloat16 if (precision == "bf16" or impl == "cudnn") else jnp.float32
    o = jax.nn.dot_product_attention(q.astype(dt), k.astype(dt), v.astype(dt), is_causal=True,
                                     scale=float(f32(1.0) / np.sqrt(f32(hd), dtype=f32)), implementation=impl)
    return o.astype(jnp.float32).reshape(B, T, H * hd)

def attend(q, k, v, backend: str, ein, precision: str):
    name, qb, kb = parse(backend)
    if name == "naive":
        return naive(q, k, v, ein, precision)
    if name == "blockwise":
        return blockwise(q, k, v, ein, precision, qb, kb)
    return sdpa(q, k, v, name, precision)

def resolve(backend: str) -> str:
    """'auto': cudnn on GPU, blockwise elsewhere."""
    if backend != "auto":
        return backend
    return "cudnn" if jax.default_backend() == "gpu" else "blockwise:128"
