"""find_via: a child trained from a rebased init merges through it (forge merge --via)."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("merge_search", Path(__file__).resolve().parents[1] / "merge_search.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def ckpt(d: Path, sha: str, parent: str | None) -> str:
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"model_sha256": sha, "parent_sha256": parent}))
    return str(d)


def test_find_via(tmp_path):
    base = ckpt(tmp_path / "base/FINAL", "B", None)
    init = ckpt(tmp_path / "rouge/rebased-init-x-lam0.5", "I", "B")
    ckpt(tmp_path / "rouge/rebased-init-y-lam1", "J", "OTHER")  # derived from another base
    rouge = ckpt(tmp_path / "rouge/FINAL", "R", "I")
    quasnir = ckpt(tmp_path / "quasnir/FINAL", "Q", "B")
    stray = ckpt(tmp_path / "stray/FINAL", "S", "J")
    assert ms.find_via(base, [rouge, quasnir]) == [init]
    assert ms.find_via(base, [quasnir]) == []
    assert ms.find_via(base, [stray]) == [], "no init of this base: forge merge will refuse the child"
