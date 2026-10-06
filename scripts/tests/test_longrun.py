"""Long-run orchestrator: config consistency, plan arithmetic, configs."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("longrun", ROOT / "scripts" / "longrun.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)


def test_longrun_json_equals_defaults():
    assert json.loads((ROOT / "training/configs/longrun.json").read_text()) == json.loads(json.dumps(lr.DEFAULTS))


def test_every_rsi_promotion_requires_approval_with_failsafe_timeout():
    rsi = lr.DEFAULTS["rsi"]
    assert all(m["require_approval"] for m in rsi["models"].values())
    assert 0 < rsi["approval_timeout_hours"] <= 6
    assert abs(sum(lr.DEFAULTS["shares"].values()) - 1.0) < 1e-9


def dry(days: str) -> dict:
    out = subprocess.run([sys.executable, str(ROOT / "scripts/longrun.py"), "--dry-run", "--json", "--days", days, "--allow-v1"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout[out.stdout.index("{"):])


def test_dry_run_budget_scales_with_days():
    p3, p20 = dry("3"), dry("20")
    assert len(p3["cycles"]) == 3 and len(p20["cycles"]) == 20
    assert len(dry("10")["cycles"]) == 10
    tok = lambda p: sum(c["phases"]["base"]["tokens"] for c in p["cycles"])  # noqa: E731
    assert 6.0 < tok(p20) / tok(p3) < 7.4


def test_base_m_is_a_valid_40m_config():
    c = json.loads((ROOT / "training/configs/base-m.json").read_text())
    m = c["model"]
    assert m["dim"] % m["n_heads"] == 0 and m["n_heads"] % m["n_kv_heads"] == 0 and (m["dim"] // m["n_heads"]) % 2 == 0
    hd = m["dim"] // m["n_heads"]
    n = m["vocab_size"] * m["dim"] + m["n_layers"] * (m["dim"] * m["n_heads"] * hd * 2 + m["dim"] * m["n_kv_heads"] * hd * 2
                                                       + 3 * m["dim"] * m["ffn_hidden"] + 2 * m["dim"]) + m["dim"]
    assert 35e6 <= n <= 45e6
    assert c["seq_len"] <= m["max_seq_len"]
