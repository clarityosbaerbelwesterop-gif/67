"""ARC-AGI-2 suite of scripts/humaneval.py: prompts and exact-match grid parsing."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("humaneval", Path(__file__).resolve().parents[1] / "humaneval.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def test_arc_prompts_and_grid_parsing(tmp_path, monkeypatch):
    f = tmp_path / "arc.json"
    f.write_text(json.dumps({"t1": {"train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                                    "test": [{"input": [[3, 4], [5, 6]], "output": [[4, 3], [6, 5]]},
                                             {"input": [[7, 8]], "output": [[8, 7]]}]}}))
    monkeypatch.setattr(H, "ARC_FILE", f)
    tasks = H.arc_tasks()
    assert [(t[0], t[1]) for t in tasks] == [("t1", 0), ("t1", 1)], "every test output is scored"
    assert tasks[0][2] == "in:\n12\nout:\n21\n\nin:\n34\n56\nout:\n" and tasks[0][3] == "43\n65"
    assert H.arc_grid("43\n65\n\nin:\n11") == "43\n65"
    assert H.arc_grid("43\n6x\n") == "43", "a non-digit line ends the grid"
    assert H.arc_grid("\n43\n65") == "43\n65", "leading blank lines are skipped"
