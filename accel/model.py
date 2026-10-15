"""forge's SCP-1 transformer in pure jax.numpy (core_engine/train/src/model.rs, isa/src/machine.rs).

Parameters are a dict keyed by forge's tensor names, float32, linear weights [out, in]
(y = x @ W.T), tied output head. Everything is jit-able; the model config is a static
argument (ModelConfig is a frozen dataclass).

precision:
  "f32"   every GEMM in float32 with Precision.HIGHEST (true f32 on TPU/GPU; no TF32 or
          single-pass bf16). Matches forge's f32 engines (scalar / avx2-fma / avx512-f32).
  "bf16"  both GEMM inputs rounded to bf16 (round-to-nearest-even), products accumulated in
          f32 -- forward and backward. This is exactly the arithmetic of forge's amx-bf16
          engine and the fast path on TPU MXUs and GPU tensor cores. Parameters, optimizer
          state and all element-wise math stay f32.
"""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig
from .forge_io import Rng, cosf, powf, sinf
from .attn import attend as _attend

f32 = np.float32
PRECISIONS = ("f32", "bf16")


# ---------------------------------------------------------------------------------------------- GEMMs
@functools.lru_cache(maxsize=None)
def _bf16_einsum(eq: str):
    """einsum with bf16-rounded operands and f32 accumulation in forward AND backward (the
    gradient GEMMs round their inputs too, like every amx-bf16 GEMM in forge)."""
    ins, out = eq.split("->")
    sa, sb = ins.split(",")
    eq_ga, eq_gb = f"{out},{sb}->{sa}", f"{out},{sa}->{sb}"

    def mm(a16, b16, e):
        return jnp.einsum(e, a16, b16, preferred_element_type=jnp.float32)

    @jax.custom_vjp
    def f(a, b):
        return mm(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), eq)

    def fwd(a, b):
        a16, b16 = a.astype(jnp.bfloat16), b.astype(jnp.bfloat16)
        return mm(a16, b16, eq), (a16, b16)

    def bwd(res, g):
        a16, b16 = res
        g16 = g.astype(jnp.bfloat16)
        return mm(g16, b16, eq_ga), mm(g16, a16, eq_gb)

    f.defvjp(fwd, bwd)
    return f


def ein(eq: str, a, b, precision: str = "f32"):
    if precision == "bf16":
        return _bf16_einsum(eq)(a, b)
    return jnp.einsum(eq, a, b, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)


# ---------------------------------------------------------------------------------------------- constants
@functools.lru_cache(maxsize=64)
def rope_table(seq: int, hd: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """machine.rs rope_table: inv_i = 1/theta.powf(2i/hd), ang = f32(t)*inv_i, cos/sin in f32 (libm)."""
    half = hd // 2
    th = f32(theta)
    inv = [f32(1.0) / powf(th, f32(2 * i) / f32(hd)) for i in range(half)]
    cos = np.empty((seq, half), f32)
    sin = np.empty((seq, half), f32)
    for t in range(seq):
        for i in range(half):
            a = f32(t) * inv[i]
            cos[t, i] = cosf(a)
            sin[t, i] = sinf(a)
    return cos, sin


def rms_norm(x, w, eps: float, dim: int):
    """machine.rs RmsNorm: ss = sum(x^2)/d; inv = 1/sqrt(ss+eps); out = (x*inv)*w."""
    ss = jnp.sum(x * x, axis=-1, keepdims=True) / f32(dim)
    inv = f32(1.0) / jnp.sqrt(ss + f32(eps))
    return (x * inv) * w


def rope(x, cos, sin):
    """Interleaved pairs: (x[2i], x[2i+1]) <- (a*c - b*s, a*s + b*c). x [B,T,h,hd]; cos/sin [T,hd/2].
    The pairs are split by a reshape to [..., hd/2, 2], not by strided slices: x[..., 0::2] lowers to a
    gather over the head dim and its gradient to a scatter-add, slow paths on TPUs and GPUs. Same
    arithmetic either way."""
    xr = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    a, b = xr[..., 0], xr[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return jnp.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(x.shape)


# ---------------------------------------------------------------------------------------------- model
def _block(x, p: dict, cfg: ModelConfig, cos, sin, mask, precision: str, attn: str = "naive"):
    """One pre-norm layer (model.rs:206-351). x [B,T,d]; p holds the layer's tensors by short name."""
    B, T, d = x.shape
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    xn = rms_norm(x, p["attn_norm.weight"], cfg.norm_eps, d)
    q = rope(ein("btd,fd->btf", xn, p["attn.wq.weight"], precision).reshape(B, T, H, hd), cos, sin)
    k = rope(ein("btd,fd->btf", xn, p["attn.wk.weight"], precision).reshape(B, T, KV, hd), cos, sin)
    v = ein("btd,fd->btf", xn, p["attn.wv.weight"], precision).reshape(B, T, KV, hd)
    o = _attend(q, k, v, attn, ein, precision)
    x = x + ein("btf,df->btd", o, p["attn.wo.weight"], precision)
    xn = rms_norm(x, p["ffn_norm.weight"], cfg.norm_eps, d)
    h1 = ein("btd,fd->btf", xn, p["ffn.w1.weight"], precision)
    h3 = ein("btd,fd->btf", xn, p["ffn.w3.weight"], precision)
    g = (h1 * (f32(1.0) / (f32(1.0) + jnp.exp(-h1)))) * h3
    return x + ein("btf,df->btd", g, p["ffn.w2.weight"], precision)


def forward(params: dict, cfg: ModelConfig, ids, precision: str = "f32", remat: bool = False, attn: str = "naive"):
    """Logits [B,T,V] (f32) of token ids [B,T]; every window starts at position 0 (model.rs:206-387)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    B, T = ids.shape
    cos, sin = (jnp.asarray(a) for a in rope_table(T, cfg.head_dim, cfg.rope_theta))
    mask = jnp.asarray(np.tril(np.ones((T, T), bool)))
    emb = params["tok_emb.weight"]
    x = jnp.take(emb, ids, axis=0)
    block = _block
    if remat:
        block = jax.checkpoint(_block, static_argnums=(2, 6, 7))
    for i in range(cfg.n_layers):
        pre = f"blocks.{i}."
        p = {k[len(pre):]: v for k, v in params.items() if k.startswith(pre)}
        x = block(x, p, cfg, cos, sin, mask, precision, attn)
    xf = rms_norm(x, params["norm.weight"], cfg.norm_eps, cfg.dim)
    return ein("btd,vd->btv", xf, emb, precision)


XENT_MODES = ("fast", "forge")


def sequential_row_sum(e, unroll: int = 16):
    """sum over the last axis, strictly left to right in f32 (forge's `sum += v` loop in Xent)."""
    et = jnp.moveaxis(e, -1, 0)
    s, _ = jax.lax.scan(lambda acc, col: (acc + col, None), jnp.zeros(et.shape[1:], jnp.float32), et, unroll=unroll)
    return s


def row_losses(params: dict, cfg: ModelConfig, ids, targets, precision: str = "f32", remat: bool = False,
               xent: str = "fast", attn: str = "naive"):
    """Per-row cross entropy f32(ln(sum exp(l - max)) + max - l[target]) and next-token hit
    (argmax = first index of the max), as machine.rs Xent computes them.

    xent="forge" sums exp(l - max) over the vocabulary strictly in order, like forge; over 8192
    logits that sequential f32 sum runs ~4e-6 low, which is the whole measured gap between accel's
    and forge's eval loss (1.1e-5 on base-g3; bit-identical with "forge"). It is a V-step scan, so
    "fast" (a tree reduction) is the default for training steps; the gradient is the same either way."""
    if xent not in XENT_MODES:
        raise ValueError(f"xent must be one of {XENT_MODES}")
    lg = forward(params, cfg, ids, precision, remat, attn)
    mx = jax.lax.stop_gradient(jnp.max(lg, axis=-1))  # d/dlogits = softmax - onehot, as forge's Xent
    e = jnp.exp(lg - mx[..., None])
    lse = jnp.log(jnp.sum(e, axis=-1))
    if xent == "forge":  # forge's value, the tree sum's gradient (a + (b - a) == b exactly for b ~ a)
        lse = lse + jax.lax.stop_gradient(jnp.log(sequential_row_sum(jax.lax.stop_gradient(e))) - lse)
    tl = jnp.take_along_axis(lg, targets[..., None], axis=-1)[..., 0]
    ce = (lse + mx) - tl
    hit = jnp.argmax(lg, axis=-1) == targets
    return ce, hit


def pairwise_sum(x, width: int = 128):
    """f32 sum by levels of `width`-element partial sums. forge sums these reductions (SumSq over
    the gradients, a micro-batch's row losses) in f64; f64 is slow or absent on TPUs, and a flat f32
    reduction over millions of elements can be off by ~1e-5 relative on CPU. Levels of 128 keep
    the f32 error near 1e-7."""
    x = jnp.ravel(x).astype(jnp.float32)
    while x.size > width:
        n = -(-x.size // width) * width
        x = jnp.pad(x, (0, n - x.size)).reshape(-1, width).sum(axis=1)
    return jnp.sum(x)


def loss(params: dict, cfg: ModelConfig, batch, precision: str = "f32", remat: bool = False, xent: str = "forge",
         attn: str = "naive"):
    """Mean next-token cross entropy over every token of batch = (ids, targets), like forge's
    LOSS / LOSS_COUNT (no ignore index, no padding mask, no z-loss, no label smoothing)."""
    ids, targets = batch
    ce, _ = row_losses(params, cfg, ids, targets, precision, remat, xent, attn)
    return pairwise_sum(ce) / f32(ce.size)


# ---------------------------------------------------------------------------------------------- init
EXACT_INIT_MAX_PARAMS = 150_000_000


def init_params(cfg: ModelConfig, seed: int, exact: bool | None = None) -> dict:
    """trainer.rs init_params: Rng(seed ^ 0x5EED_1417); in params() order, 1-D tensors = 1.0 (no draws),
    matrices element by element row-major: normal() * std (f32), std = 0.02/sqrt(2L) for wo/w2, else 0.02.
    exact=True replays forge's generator bit for bit (pure Python, ~1.7 s per million weights);
    exact=False draws the same distribution with numpy (fast, not forge's numbers). None: exact up to
    EXACT_INIT_MAX_PARAMS."""
    if exact is None:
        exact = cfg.num_params() <= EXACT_INIT_MAX_PARAMS
    scaled = f32(0.02) / np.sqrt(f32(2.0) * f32(cfg.n_layers), dtype=f32)
    rng = Rng((seed ^ 0x5EED_1417) & ((1 << 64) - 1))
    nprng = np.random.default_rng(seed)
    out = {}
    for name, shape in cfg.param_specs():
        if len(shape) == 1:
            out[name] = np.ones(shape, f32)
            continue
        std = scaled if (name.endswith("wo.weight") or name.endswith("w2.weight")) else f32(0.02)
        n = int(np.prod(shape))
        z = rng.normals(n) if exact else nprng.standard_normal(n, dtype=np.float32)
        out[name] = (z * std).astype(f32).reshape(shape)
    return out
