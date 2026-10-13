"""resume_pack: the age (public-key) path writes a pack the receiver restores; no shared secret is needed."""
import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resume_pack as RP  # noqa: E402

RECIPIENT = "age1" + "q" * 58  # syntactically valid, not a real key: the fake age below never uses it


def fake_age(tmp_path: Path, monkeypatch) -> Path:
    """An `age` that records its arguments and passes the bytes through (stands in for real encryption)."""
    bin_dir, log = tmp_path / "bin", tmp_path / "age-args.log"
    bin_dir.mkdir()
    exe = bin_dir / "age"
    exe.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexec cat\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def world(tmp_path: Path, monkeypatch) -> list[Path]:
    root = tmp_path / "repo"
    (root / "runs/longrun").mkdir(parents=True)
    (root / "runs/longrun/state.json").write_text(json.dumps({"current": {"quasnir": {"dir": "runs/q/FINAL"}}}))
    ck = root / "runs/q/FINAL"
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(os.urandom(4096))
    (ck / "state.json").write_text('{"step": 1}')
    files = [ck / "model.safetensors", ck / "state.json", root / "runs/longrun/state.json"]
    monkeypatch.setattr(RP, "ROOT", root)
    monkeypatch.setattr(RP, "PACK", root / "resume/pack")
    monkeypatch.setattr(RP, "collect", lambda: sorted(files))
    monkeypatch.delenv("FORGE_PACK_KEY", raising=False)
    return files


def test_age_pack_round_trip_without_a_shared_key(tmp_path, monkeypatch):
    log = fake_age(tmp_path, monkeypatch)
    files = world(tmp_path, monkeypatch)
    before = {f: f.read_bytes() for f in files}
    assert RP.save(False, RECIPIENT) == 0
    man = json.loads((RP.PACK / "MANIFEST.json").read_text())
    assert man["cipher"].startswith("age") and RECIPIENT in man["cipher"] and man["files"] == 3
    assert log.read_text().split() == ["-r", RECIPIENT]
    for f in files:
        f.unlink()
    with pytest.raises(SystemExit, match="age-identity"):
        RP.restore()
    assert RP.restore(str(tmp_path / "forge.key")) == 0
    assert log.read_text().splitlines()[-1].split() == ["-d", "-i", str(tmp_path / "forge.key")]
    assert {f: f.read_bytes() for f in files} == before


def test_a_malformed_recipient_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="age X25519 public key"):
        RP.save(False, "age1-not-a-key")
    assert not RP.PACK.exists()


def test_the_shared_key_path_still_needs_its_key(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="FORGE_PACK_KEY"):
        RP.save(False)


def test_the_running_base_step_packs_its_own_newest_checkpoint_with_optimizer(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    monkeypatch.setattr(RP, "ROOT", root)
    (root / "runs/longrun").mkdir(parents=True)
    (root / "runs/longrun/state.json").write_text(json.dumps({
        "current": {}, "runs": {"base-g4": {"family": "base", "finished": False}},
        "cycles": {"4": {"steps": {"base": {"status": "running"}}}}}))  # no run name recorded, as in the live state
    for d in ("runs/base-g4/step-000010", "runs/base-g4/step-000020", "runs/smoke/step-000060"):
        (root / d).mkdir(parents=True)
        for f in ("model.safetensors", "optim.safetensors", "state.json"):
            (root / d / f).write_text("x")
    (root / "runs/base-g4/latest.json").write_text(json.dumps({"dir": "runs/base-g4/step-000020"}))
    got = {str(f.relative_to(root)) for f in RP.collect()}
    assert "runs/base-g4/step-000020/optim.safetensors" in got and "runs/base-g4/step-000020/model.safetensors" in got
    assert not any(g.startswith("runs/smoke/") for g in got), "never another run's newest checkpoint"
