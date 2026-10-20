"""scripts/grow.py: the grown checkpoint computes the source's logits and greedy tokens (checked with forge itself),
has the target shapes and lineage, and never overwrites."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FORGE = ROOT / "target/release/forge"
spec = importlib.util.spec_from_file_location("grow", ROOT / "scripts" / "grow.py")
grow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grow)
needs_forge = pytest.mark.skipif(not FORGE.exists(), reason="needs target/release/forge")

MODEL = {"vocab_size": 256, "dim": 128, "n_layers": 3, "n_heads": 2, "n_kv_heads": 1, "ffn_hidden": 192,
         "max_seq_len": 64, "rope_theta": 10000.0, "norm_eps": 1e-5}


def tiny_ckpt(d: Path) -> Path:
    rng = np.random.default_rng(0)
    m, hd = MODEL, MODEL["dim"] // MODEL["n_heads"]
    w = {"tok_emb.weight": rng.standard_normal((m["vocab_size"], m["dim"])) * 0.5,
         "norm.weight": 1 + 0.1 * rng.standard_normal(m["dim"])}
    for i in range(m["n_layers"]):
        b = f"blocks.{i}"
        w.update({f"{b}.attn_norm.weight": 1 + 0.1 * rng.standard_normal(m["dim"]),
                  f"{b}.ffn_norm.weight": 1 + 0.1 * rng.standard_normal(m["dim"]),
                  f"{b}.attn.wq.weight": rng.standard_normal((m["n_heads"] * hd, m["dim"])) * 0.1,
                  f"{b}.attn.wk.weight": rng.standard_normal((m["n_kv_heads"] * hd, m["dim"])) * 0.1,
                  f"{b}.attn.wv.weight": rng.standard_normal((m["n_kv_heads"] * hd, m["dim"])) * 0.1,
                  f"{b}.attn.wo.weight": rng.standard_normal((m["dim"], m["n_heads"] * hd)) * 0.1,
                  f"{b}.ffn.w1.weight": rng.standard_normal((m["ffn_hidden"], m["dim"])) * 0.1,
                  f"{b}.ffn.w3.weight": rng.standard_normal((m["ffn_hidden"], m["dim"])) * 0.1,
                  f"{b}.ffn.w2.weight": rng.standard_normal((m["dim"], m["ffn_hidden"])) * 0.1})
    d.mkdir()
    shapes = {k: v.shape for k, v in w.items()}
    sha = grow.write_streaming(d / "model.safetensors", shapes,
                               ((k, w[k].astype(np.float32)) for k in sorted(w)), {"format": "pt", "step": "1"})
    (d / "state.json").write_text(json.dumps({"config": {"model": MODEL}, "model_sha256": sha, "step": 1}))
    return d


def forge(*args) -> dict:
    out = subprocess.run([str(FORGE), *args], check=True, capture_output=True, text=True).stdout
    return json.loads(next(l for l in out.splitlines() if l.startswith("{")))


@needs_forge
def test_grown_model_keeps_logits_and_tokens(tmp_path):
    src = tiny_ckpt(tmp_path / "src")
    out = tmp_path / "grown"
    assert grow.main([str(src), str(out), "--dim", "256", "--ffn-hidden", "384", "--n-layers", "5"]) == 0
    st = json.loads((out / "state.json").read_text())
    m = st["config"]["model"]
    assert (m["dim"], m["n_heads"], m["n_kv_heads"], m["ffn_hidden"], m["n_layers"]) == (256, 4, 2, 384, 5)
    assert st["parent_sha256"] == json.loads((src / "state.json").read_text())["model_sha256"]
    assert st["grown_from"]["model"] == MODEL and abs(m["norm_eps"] - 0.5e-5) < 1e-12
    ids = "3,77,150,9,201,42,18,255"
    a = np.array(forge("logits", "--ckpt", str(src), "--prompt-ids", ids)["logits"])
    b = np.array(forge("logits", "--ckpt", str(out), "--prompt-ids", ids)["logits"])
    assert np.abs(a - b).max() < 1e-4 * max(1.0, np.abs(a).max()), np.abs(a - b).max()
    g = [forge("generate", "--ckpt", str(c), "--prompt-ids", ids, "--max-new", "16")["ids"] for c in (src, out)]
    assert g[0] == g[1]
    with pytest.raises(SystemExit):  # never overwrites
        grow.main([str(src), str(out), "--dim", "256", "--ffn-hidden", "384", "--n-layers", "5"])


def test_target_config_rules():
    t = grow.target_config(MODEL, 512, 768, 6, None)
    assert (t["n_heads"], t["n_kv_heads"]) == (8, 4)  # head_dim 64 and the 2:1 grouping stay
    with pytest.raises(SystemExit):
        grow.target_config(MODEL, 96, 768, 6, None)   # not a multiple of head_dim x group
    with pytest.raises(SystemExit):
        grow.target_config(MODEL, 256, 128, 6, None)  # shrinking
    assert grow.num_params(MODEL) == 256 * 128 + 128 + 3 * (2 * 128 + 2 * 128 * 128 + 2 * 64 * 128 + 3 * 128 * 192)
