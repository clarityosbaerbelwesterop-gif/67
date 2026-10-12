"""DiLoCo outer step of cluster/fabric.py on tiny synthetic checkpoints."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fabric as F  # noqa: E402

MODEL = {"vocab_size": 8, "dim": 2, "n_layers": 1, "n_heads": 1, "n_kv_heads": 1, "ffn_hidden": 2, "max_seq_len": 8}


def test_init_aggregate_nesterov_and_quorum(tmp_path):
    src = tmp_path / "src"
    F.write_ckpt(src, {"w": np.array([1.0, 2.0], np.float32)}, MODEL, None, {})
    s = tmp_path / "state"
    assert F.main(["init", "--from", str(src), "--state", str(s)]) == 0
    rnd = json.loads((s / "round.json").read_text())
    deltas = []
    for i, d in enumerate(([0.2, 0.0], [0.0, 0.4], [9.0, 9.0])):
        out = tmp_path / f"d{i}"
        out.mkdir()
        F.save(out / "delta.safetensors", {"w": np.array(d, np.float32)}, "F16")
        rec = {"round": 0, "worker": i, "tokens": 10, "seconds": 1.0, "tokens_per_s": 10.0, "loss_last": 2.0,
               "global_sha256": rnd["global_sha256"] if i < 2 else "stale"}
        (out / "worker.json").write_text(json.dumps(rec))
        deltas.append(str(out))
    assert F.main(["aggregate", "--state", str(s), "--deltas", *deltas, "--expected", "4", "--quorum", "0.75"]) == 2, "2 of 4 < quorum"
    assert F.main(["aggregate", "--state", str(s), "--deltas", *deltas, "--expected", "3"]) == 0
    g, _ = F.load(s / "global" / "model.safetensors")
    outer = np.array([0.1, 0.2])  # mean of the two valid deltas; the stale one is ignored
    m = outer  # momentum starts at zero
    want = np.array([1.0, 2.0]) - 0.7 * (outer + 0.9 * m)
    assert np.allclose(g["w"], want, atol=1e-3)
    st = json.loads((s / "round.json").read_text())
    assert st["round"] == 1 and st["history"][0]["workers"] == 2
    parent = json.loads((s / "global" / "state.json").read_text())["parent_sha256"]
    assert parent == rnd["global_sha256"], "lineage: the new global descends from the old"
