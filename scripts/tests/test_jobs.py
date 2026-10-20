"""Pod jobs in jobs/: the volume inventory reads, never writes, and never follows symlinks."""
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("inventory", ROOT / "jobs" / "inventory.py")
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)


def test_inventory_lists_sizes_big_files_and_manifests(tmp_path, monkeypatch):
    vol = tmp_path / "vol"
    (vol / "runs/s2/export").mkdir(parents=True)
    (vol / "data/v6").mkdir(parents=True)
    (vol / "runs/s2/export/EXPORT.json").write_text(json.dumps({"export": {"sha256": "ab", "step": 7500}}))
    (vol / "runs/s2/export/w.safetensors").write_bytes(b"\0" * 2048)
    (vol / "data/v6/part.parquet").write_bytes(b"x" * 300)
    (vol / ".env").write_text("SECRET=1")              # never read: not a manifest name
    (vol / "runs/rouge-2").mkdir(parents=True)
    (vol / "runs/rouge-2/state.json").write_text(json.dumps({"private": "config"}))  # size only
    os.symlink(tmp_path, vol / "loop")                  # never followed
    monkeypatch.setattr(inv, "BIG", 1024)
    before = sorted((p.relative_to(vol), p.stat().st_mtime_ns) for p in vol.rglob("*") if p.is_file())
    out = tmp_path / "out"
    assert inv.main([str(vol), str(out)]) == 0
    r = json.loads((out / "result.json").read_text())
    assert r["files"] == 5 and not r["partial"] and r["dirs"]["runs/rouge-2"]["files"] == 1
    assert r["dirs"]["runs"]["bytes"] == sum(len(p.read_bytes()) for p in (vol / "runs").rglob("*") if p.is_file())
    assert [b["path"] for b in r["big_files"]] == ["runs/s2/export/w.safetensors"]
    assert r["manifests"] == {"runs/s2/export/EXPORT.json": {"export": {"sha256": "ab", "step": 7500}}}
    assert "SECRET" not in (out / "result.json").read_text() and "private" not in (out / "result.json").read_text()
    assert before == sorted((p.relative_to(vol), p.stat().st_mtime_ns) for p in vol.rglob("*") if p.is_file())
