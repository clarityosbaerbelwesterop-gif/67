"""accel.generate (KV-cache decoding on JAX devices) against a no-cache reference and forge generate."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from conftest import FORGE, ROOT

from accel import forge_io as fio
from accel.config import ModelConfig
from accel import generate as G

MODEL = {"vocab_size": 300, "dim": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "ffn_hidden": 96,
         "max_seq_len": 48, "rope_theta": 10000.0, "norm_eps": 1e-5}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from accel.model import init_params
    cfg = ModelConfig.from_dict(MODEL)
    p = {k: np.asarray(v, np.float32) for k, v in init_params(cfg, 3, exact=False).items()}
    p = {k: (v * 12 if v.ndim == 2 else v) for k, v in p.items()}  # peaked logits: no near-ties between engines
    z = {k: np.zeros_like(v) for k, v in p.items()}
    d = tmp_path_factory.mktemp("gen") / "ck"
    fio.save_checkpoint(d, 0, p, z, z, {"step": 0, "config": {"model": MODEL}})
    return d, cfg, p


def reference(cfg, p, prompt, max_new, stop=None):
    import jax.numpy as jnp
    from accel.model import forward
    ids = G.cut_prompt(prompt, cfg.max_seq_len, max_new)
    out = []
    for _ in range(max_new):
        lg = forward({k: jnp.asarray(v) for k, v in p.items()}, cfg, jnp.asarray([ids], jnp.int32), "f32")
        nxt = int(np.argmax(np.asarray(lg[0, -1])))
        ids.append(nxt)
        out.append(nxt)
        if nxt == stop:
            break
    return out


PROMPTS = [[5, 9, 11], [7] * 20, [1, 2, 3, 4, 5, 6, 7, 8], [42], list(range(10, 60))]


def run_gen(cfg, p, prompts, **kw):
    params = G.device_params(p, kw.pop("precision", "f32"))
    return [ids for _, ids in G.generate(cfg, params, prompts, precision="f32", **kw)]


def test_greedy_matches_reference_and_batching(ckpt):
    _, cfg, p = ckpt
    got = run_gen(cfg, p, PROMPTS, max_new=12, batch=8)
    want = [reference(cfg, p, pr, 12) for pr in PROMPTS]
    assert got == want
    assert run_gen(cfg, p, PROMPTS, max_new=12, batch=2) == want  # batch size and padding rows change nothing


def test_stop_and_sampling(ckpt):
    _, cfg, p = ckpt
    want = reference(cfg, p, PROMPTS[0], 12)
    stop = want[3]
    got = run_gen(cfg, p, PROMPTS[:1], max_new=12, stop=stop)[0]
    assert got == want[:want.index(stop) + 1]
    a = run_gen(cfg, p, PROMPTS * 2, max_new=10, temperature=1.0, seed=7, batch=4)
    b = run_gen(cfg, p, PROMPTS * 2, max_new=10, temperature=1.0, seed=7, batch=4)
    c = run_gen(cfg, p, PROMPTS * 2, max_new=10, temperature=1.0, seed=8, batch=4)
    assert a == b and a != c and all(len(x) == 10 and all(0 <= i < 300 for i in x) for x in a)


def test_cli_matches_forge_greedy(ckpt, tmp_path):
    d, cfg, p = ckpt
    pf = tmp_path / "p.jsonl"
    pf.write_text("".join(json.dumps(x) + "\n" for x in PROMPTS))
    import sys
    out = subprocess.run([sys.executable, "-m", "accel.generate", "--ckpt", str(d), "--prompts-file", str(pf),
                          "--max-new", "10", "--precision", "f32", "--batch", "4"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    acc = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    assert [r["index"] for r in acc] == list(range(len(PROMPTS)))
    if not FORGE.exists():
        pytest.skip("no forge binary")
    fo = subprocess.run([str(FORGE), "generate", "--ckpt", str(d), "--prompts-file", str(pf), "--max-new", "10",
                         "--threads", "1"], capture_output=True, text=True, check=True).stdout
    fg = [json.loads(l)["ids"] for l in fo.splitlines() if l.startswith("{")]
    assert [r["ids"] for r in acc] == fg


def test_prefill_path_matches_reference(tmp_path):
    """Prompts longer than 64 ids start with a one-pass prefill of 64 positions."""
    from accel.model import init_params
    m = {**MODEL, "max_seq_len": 160}
    cfg = ModelConfig.from_dict(m)
    p = {k: np.asarray(v, np.float32) for k, v in init_params(cfg, 4, exact=False).items()}
    p = {k: (v * 12 if v.ndim == 2 else v) for k, v in p.items()}
    prompts = [list(range(3, 103)), list(range(50, 140)) + [7, 7], list(range(1, 80))]
    got = run_gen(cfg, p, prompts, max_new=9, batch=4)
    assert got == [reference(cfg, p, pr, 9) for pr in prompts]
