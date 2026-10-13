"""forge's ModelConfig / TrainConfig (core_engine/train/src/config.rs) in Python.

Every float of a forge config is an f32: JSON holds whatever the user wrote, forge
parses it into f32 and state.json prints the f64 expansion of that f32
(0.0004 -> 0.00039999998989515007). The classes here keep the f32 values as
numpy.float32 and serialise them the same way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

f32 = np.float32

# Keys of forge's TrainConfig (config.rs:79-168; "optimiser" is serde's alias of "optimizer"). Anything else in
# a run config is ignored with a warning (forge itself silently ignores unknown keys: serde default).
MUON_KEYS = ("muon_lr", "muon_momentum", "muon_ns_steps", "muon_weight_decay")
TRAIN_KEYS = ("run", "model", "data", "batch", "seq_len", "grad_accum", "max_steps", "lr", "min_lr", "warmup_steps",
              "weight_decay", "grad_clip", "seed", "init_from", "resume_from", "out_dir", "log_every", "eval_every",
              "eval_batches", "ckpt_every", "threads", "hbvm_mib", "autotune", "lr_schedule", "decay_steps", "init_optim",
              "optimizer", "optimiser") + MUON_KEYS
# accel implements AdamW only; forge's "muon" optimizer is refused here until it is ported.
OPTIMIZERS = ("adamw",)
# Keys forge understands but that have no meaning on an XLA device: accepted silently.
CPU_ONLY_KEYS = ("threads", "hbvm_mib", "autotune")
MODEL_KEYS = ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "ffn_hidden", "max_seq_len", "rope_theta",
              "norm_eps")


class ConfigError(ValueError):
    pass


def _uint(d: dict, k: str, default: Any = None) -> int:
    v = d.get(k, default)
    if v is None:
        raise ConfigError(f"missing field `{k}`")
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ConfigError(f"`{k}` must be a non-negative integer, not {v!r}")
    return int(v)


def _float32(d: dict, k: str, default: Any = None) -> np.float32:
    v = d.get(k, default)
    if v is None:
        raise ConfigError(f"missing field `{k}`")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"`{k}` must be a number, not {v!r}")
    return f32(v)


@dataclass(frozen=True)
class ModelConfig:
    """Hashable (usable as a static jit argument)."""
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int | None
    max_seq_len: int
    rope_theta: float  # f32 value held as a Python float (exactly representable)
    norm_eps: float

    @staticmethod
    def from_dict(d: dict) -> "ModelConfig":
        if not isinstance(d, dict):
            raise ConfigError("model must be an object")
        fh = d.get("ffn_hidden")
        c = ModelConfig(
            vocab_size=_uint(d, "vocab_size"), dim=_uint(d, "dim"), n_layers=_uint(d, "n_layers"),
            n_heads=_uint(d, "n_heads"), n_kv_heads=_uint(d, "n_kv_heads"),
            ffn_hidden=None if fh is None else _uint(d, "ffn_hidden"), max_seq_len=_uint(d, "max_seq_len"),
            rope_theta=float(_float32(d, "rope_theta", 10000.0)), norm_eps=float(_float32(d, "norm_eps", 1e-5)))
        c.validate()
        return c

    def to_dict(self) -> dict:
        """As forge serialises it (ffn_hidden: null when derived)."""
        return {"vocab_size": self.vocab_size, "dim": self.dim, "n_layers": self.n_layers, "n_heads": self.n_heads,
                "n_kv_heads": self.n_kv_heads, "ffn_hidden": self.ffn_hidden, "max_seq_len": self.max_seq_len,
                "rope_theta": self.rope_theta, "norm_eps": self.norm_eps}

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn(self) -> int:
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        hidden = 2 * (4 * self.dim) // 3
        return 64 * (-(-hidden // 64))

    def validate(self) -> None:
        if self.n_heads == 0 or self.dim % self.n_heads:
            raise ConfigError(f"dim {self.dim} not divisible by n_heads {self.n_heads}")
        if self.n_kv_heads == 0 or self.n_heads % self.n_kv_heads:
            raise ConfigError(f"n_heads {self.n_heads} not divisible by n_kv_heads {self.n_kv_heads}")
        if self.head_dim % 2:
            raise ConfigError("head_dim must be even for RoPE")
        if self.vocab_size == 0 or self.n_layers == 0 or self.max_seq_len == 0:
            raise ConfigError("vocab_size, n_layers and max_seq_len must be positive")

    def param_specs(self) -> list[tuple[str, tuple[int, ...]]]:
        """forge's params() order (model.rs:33-51); linear weights are [out, in]."""
        d, H, KV, hd, f = self.dim, self.n_heads, self.n_kv_heads, self.head_dim, self.ffn
        out = [("tok_emb.weight", (self.vocab_size, d))]
        for i in range(self.n_layers):
            b = f"blocks.{i}"
            out += [(f"{b}.attn_norm.weight", (d,)), (f"{b}.attn.wq.weight", (H * hd, d)),
                    (f"{b}.attn.wk.weight", (KV * hd, d)), (f"{b}.attn.wv.weight", (KV * hd, d)),
                    (f"{b}.attn.wo.weight", (d, H * hd)), (f"{b}.ffn_norm.weight", (d,)),
                    (f"{b}.ffn.w1.weight", (f, d)), (f"{b}.ffn.w3.weight", (f, d)), (f"{b}.ffn.w2.weight", (d, f))]
        out.append(("norm.weight", (d,)))
        return out

    def num_params(self) -> int:
        d, hd, f = self.dim, self.head_dim, self.ffn
        attn = d * self.n_heads * hd * 2 + d * self.n_kv_heads * hd * 2
        return self.vocab_size * d + self.n_layers * (attn + 3 * d * f + 2 * d) + d

    def flops_per_token(self, seq: int) -> float:
        """forge's count (config.rs flops_per_token): 6N + 12·L·d·T."""
        return 6.0 * self.num_params() + 12.0 * self.n_layers * self.dim * seq


@dataclass
class TrainConfig:
    run: str
    model: ModelConfig
    data: str
    batch: int
    seq_len: int
    max_steps: int
    lr: np.float32
    min_lr: np.float32
    warmup_steps: int
    out_dir: str
    grad_accum: int = 1
    weight_decay: np.float32 = f32(0.1)
    grad_clip: np.float32 = f32(1.0)
    seed: int = 0
    init_from: str | None = None
    resume_from: str | None = None
    log_every: int = 10
    eval_every: int = 0
    eval_batches: int = 10
    ckpt_every: int = 0
    threads: int = 0
    hbvm_mib: int = 0
    autotune: bool = True
    lr_schedule: str = "cosine"
    decay_steps: int = 0
    init_optim: bool = False
    optimizer: str = "adamw"
    muon_set: tuple = ()  # muon_* keys the input sets (non-null); only valid with optimizer muon (not serialised)
    ignored: list = field(default_factory=list)  # unknown keys of the input (not serialised)

    @staticmethod
    def from_dict(d: dict) -> "TrainConfig":
        if not isinstance(d, dict):
            raise ConfigError("a run config must be a JSON object")
        if "model" not in d:
            raise ConfigError("missing field `model`")
        for k in ("run", "data", "out_dir"):
            if not isinstance(d.get(k), str):
                raise ConfigError(f"missing or non-string field `{k}`")
        opt_str = {}
        for k in ("init_from", "resume_from"):
            v = d.get(k)
            if v is not None and not isinstance(v, str):
                raise ConfigError(f"`{k}` must be a string or null")
            opt_str[k] = v
        sched = d.get("lr_schedule", "cosine")
        if not isinstance(sched, str):
            raise ConfigError("`lr_schedule` must be a string")
        for k in ("autotune", "init_optim"):
            if k in d and not isinstance(d[k], bool):
                raise ConfigError(f"`{k}` must be a boolean")
        if "optimizer" in d and "optimiser" in d:
            raise ConfigError("duplicate field `optimizer`")  # serde: "optimiser" is an alias of "optimizer"
        optimizer = d.get("optimizer", d.get("optimiser", "adamw"))
        if not isinstance(optimizer, str):
            raise ConfigError("`optimizer` must be a string")
        c = TrainConfig(
            run=d["run"], model=ModelConfig.from_dict(d["model"]), data=d["data"], batch=_uint(d, "batch"),
            seq_len=_uint(d, "seq_len"), max_steps=_uint(d, "max_steps"), lr=_float32(d, "lr"),
            min_lr=_float32(d, "min_lr"), warmup_steps=_uint(d, "warmup_steps"), out_dir=d["out_dir"],
            grad_accum=_uint(d, "grad_accum", 1), weight_decay=_float32(d, "weight_decay", 0.1),
            grad_clip=_float32(d, "grad_clip", 1.0), seed=_uint(d, "seed", 0), init_from=opt_str["init_from"],
            resume_from=opt_str["resume_from"], log_every=_uint(d, "log_every", 10), eval_every=_uint(d, "eval_every", 0),
            eval_batches=_uint(d, "eval_batches", 10), ckpt_every=_uint(d, "ckpt_every", 0), threads=_uint(d, "threads", 0),
            hbvm_mib=_uint(d, "hbvm_mib", 0), autotune=bool(d.get("autotune", True)), lr_schedule=sched,
            decay_steps=_uint(d, "decay_steps", 0), init_optim=bool(d.get("init_optim", False)), optimizer=optimizer,
            muon_set=tuple(k for k in MUON_KEYS if d.get(k) is not None),
            ignored=sorted(k for k in d if k not in TRAIN_KEYS))
        if c.seed >= 1 << 64:
            raise ConfigError("seed must fit in u64")
        c.validate()
        return c

    def validate(self) -> None:
        """config.rs validate()."""
        self.model.validate()
        if self.seq_len > self.model.max_seq_len:
            raise ConfigError("seq_len exceeds max_seq_len")
        if self.batch == 0 or self.seq_len < 2 or self.grad_accum == 0 or self.max_steps == 0:
            raise ConfigError("batch, seq_len (>= 2), grad_accum and max_steps must be positive")
        if not self.run or not all(c.isascii() and (c.isalnum() or c in "-_") for c in self.run):
            raise ConfigError("run name must be [A-Za-z0-9_-]")
        if self.lr_schedule not in ("cosine", "wsd", "constant"):
            raise ConfigError(f"lr_schedule must be cosine|wsd|constant, not {self.lr_schedule!r}")
        if self.init_optim and self.init_from is None:
            raise ConfigError("init_optim needs init_from")
        # forge's optimizer checks, then the one accel adds: no Muon (yet), so a Muon config must not
        # quietly train AdamW here.
        if self.optimizer not in ("adamw", "muon"):
            raise ConfigError(f"optimizer must be adamw|muon, not {self.optimizer!r}")
        if self.optimizer == "adamw" and self.muon_set:
            raise ConfigError(f"{', '.join(self.muon_set)} set, but optimizer is \"adamw\": "
                              "muon_* settings need \"optimizer\": \"muon\"")
        if self.optimizer not in OPTIMIZERS:
            raise ConfigError(f"optimizer {self.optimizer!r} is not implemented by accel (it runs "
                              f"{'|'.join(OPTIMIZERS)} only); train this config with forge")

    def to_dict(self) -> dict:
        """The config as forge writes it into state.json: every field, f32 values as their f64 expansion,
        lr_schedule omitted when "cosine", decay_steps when 0, init_optim when false. optimizer and muon_* are
        never written: validate() admits only "adamw" without muon_* keys, which forge omits as well."""
        d = {"run": self.run, "model": self.model.to_dict(), "data": self.data, "batch": self.batch,
             "seq_len": self.seq_len, "grad_accum": self.grad_accum, "max_steps": self.max_steps, "lr": float(self.lr),
             "min_lr": float(self.min_lr), "warmup_steps": self.warmup_steps, "weight_decay": float(self.weight_decay),
             "grad_clip": float(self.grad_clip), "seed": self.seed, "init_from": self.init_from,
             "resume_from": self.resume_from, "out_dir": self.out_dir, "log_every": self.log_every,
             "eval_every": self.eval_every, "eval_batches": self.eval_batches, "ckpt_every": self.ckpt_every,
             "threads": self.threads, "hbvm_mib": self.hbvm_mib, "autotune": self.autotune}
        if self.lr_schedule != "cosine":
            d["lr_schedule"] = self.lr_schedule
        if self.decay_steps != 0:
            d["decay_steps"] = self.decay_steps
        if self.init_optim:
            d["init_optim"] = True
        return d

    # ---------------------------------------------------------------- LR schedule (config.rs:210-263)
    def wsd_decay_steps(self) -> int:
        if self.decay_steps > 0:
            return self.decay_steps
        return int(math.floor(0.2 * self.max_steps + 0.5))  # f64::round: half away from zero (x >= 0)

    def trunk_step(self) -> int | None:
        if self.lr_schedule != "wsd":
            return None
        return max(self.max_steps - self.wsd_decay_steps(), 0)

    def lr_at(self, step: int) -> np.float32:
        """Learning rate of update `step` (0-based), all in f32 like forge."""
        from .forge_io import cosf  # libm cosf, the function forge's f32::cos calls
        lr, mn = f32(self.lr), f32(self.min_lr)
        if step < self.warmup_steps:
            return f32(lr * f32(step + 1) / f32(max(self.warmup_steps, 1)))
        if self.lr_schedule == "constant":
            return lr
        pi = f32(math.pi)
        if self.lr_schedule == "wsd":
            d = self.wsd_decay_steps()
            if step >= self.max_steps:
                return mn
            if step + d < self.max_steps:
                return lr
            progress = f32(step + 1 + d - self.max_steps) / f32(d)
            return f32(mn + f32(0.5) * (lr - mn) * (f32(1.0) + cosf(f32(pi * progress))))
        if step >= self.max_steps:
            return mn
        progress = f32(step - self.warmup_steps) / f32(max(self.max_steps - self.warmup_steps, 1))
        return f32(mn + f32(0.5) * (lr - mn) * (f32(1.0) + cosf(f32(pi * progress))))
