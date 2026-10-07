"""Task-vector rebase: hand-written safetensors, F32/BF16 arithmetic and rounding,
identities, refusals, and a real rebase checked with `forge eval`.

Outputs go under TOOLS_TEST_DIR (default: the session scratchpad .../scratchpad/wf/tools,
else a pytest tmp dir)."""

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rebase", ROOT / "scripts" / "rebase.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

FORGE = ROOT / "target" / "release" / "forge"
SCRATCHPAD = Path("/tmp/claude-0/-home-user/d103ac1f-f13d-5616-bc4d-c936a331fc71/scratchpad/wf/tools")
MODEL = {"dim": 8, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "vocab_size": 16, "max_seq_len": 32}


@pytest.fixture
def scratch(request, tmp_path) -> Path:
    base = Path(os.environ["TOOLS_TEST_DIR"]) if os.environ.get("TOOLS_TEST_DIR") else (
        SCRATCHPAD if SCRATCHPAD.parent.exists() else tmp_path)
    d = base / "rebase" / request.node.name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def encode(dtype: str, arr) -> bytes:
    arr = np.asarray(arr, dtype=np.float64)
    return R.f32_to_bf16(arr.astype(np.float32)).tobytes() if dtype == "BF16" else arr.astype(R.FLOAT[dtype]).tobytes()


def make_ckpt(d: Path, arrays: dict, *, model=MODEL, sha_override=None) -> Path:
    """A forge-like checkpoint dir; arrays = {name: (dtype, values)}."""
    d.mkdir(parents=True)
    tensors = {n: {"dtype": dt, "shape": list(np.shape(v)), "data": encode(dt, v)} for n, (dt, v) in sorted(arrays.items())}
    sha = R.write_safetensors(d / "model.safetensors", tensors, {"format": "pt", "step": "7"})
    state = {"step": 7, "config": {"run": d.name, "model": model}, "model_sha256": sha_override or sha,
             "parent_sha256": None, "loss": 1.5, "val_loss": 1.6, "rng_state": [1, 2, 3, 4], "tokens": 10}
    (d / "state.json").write_text(json.dumps(state))
    (d / "optim.safetensors").write_bytes(b"adam moments of this checkpoint")
    return d


def tensors_of(d: Path) -> dict:
    ts, _, _ = R.read_safetensors(d / "model.safetensors")
    return ts


def values(d: Path) -> dict:
    return {n: R.to_f64(t["dtype"], t["data"]).reshape(t["shape"]) for n, t in tensors_of(d).items()}


def bf16_reference(x: np.ndarray) -> np.ndarray:
    """Independent round-to-nearest-even of float64 values to bf16, by search over all finite bf16 values."""
    pos = np.arange(0x0000, 0x7F80, dtype=np.uint32)
    grid = (pos << 16).view(np.float32).astype(np.float64)
    out = []
    for v in x.ravel():
        a = abs(v)
        i = int(np.searchsorted(grid, a))
        if i >= len(grid):
            out.append(0x7F80)  # beyond the largest finite bf16 (inputs here stay far below)
        elif grid[i] == a or i == 0:
            out.append(int(pos[i]))
        else:
            lo, hi = grid[i - 1], grid[i]
            dl, dh = a - lo, hi - a
            out.append(int(pos[i - 1] if dl < dh or (dl == dh and pos[i - 1] % 2 == 0) else pos[i]))
        if v < 0 or (v == 0 and math.copysign(1, v) < 0):
            out[-1] |= 0x8000
    return np.array(out, dtype=np.uint16)


# --------------------------------------------------------------------------- safetensors

def test_safetensors_layout_matches_forge(scratch):
    rng = np.random.default_rng(0)
    arrays = {"b.weight": ("F32", rng.normal(size=(3, 5))), "a.norm": ("BF16", rng.normal(size=(7,))),
              "c.half": ("F16", rng.normal(size=(2, 2))), "d.wide": ("F64", rng.normal(size=(4,))), "e.scalar": ("F32", 3.5)}
    d = make_ckpt(scratch / "ck", arrays)
    raw = (d / "model.safetensors").read_bytes()
    hlen = int.from_bytes(raw[:8], "little")
    assert hlen % 8 == 0 and raw[8 + hlen - 1:8 + hlen] in (b" ", b"}")
    header = json.loads(raw[8:8 + hlen])
    # forge (serde_json): sorted keys, compact, __metadata__ kept, tensors back to back in name order
    assert raw[8:8 + hlen].rstrip() == json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    assert header["__metadata__"] == {"format": "pt", "step": "7"}
    names = sorted(arrays)
    offs = [header[n]["data_offsets"] for n in names]
    assert offs[0][0] == 0 and all(a[1] == b[0] for a, b in zip(offs, offs[1:])) and offs[-1][1] == len(raw) - 8 - hlen
    ts, meta, sha = R.read_safetensors(d / "model.safetensors")
    assert list(ts) == names and meta == {"format": "pt", "step": "7"} and sha == hashlib.sha256(raw).hexdigest()
    assert ts["e.scalar"]["shape"] == [] and R.to_f64("F32", ts["e.scalar"]["data"]).tolist() == [3.5]
    np.testing.assert_array_equal(R.to_f64("F32", ts["b.weight"]["data"]).reshape(3, 5), arrays["b.weight"][1].astype(np.float32))
    # rewriting what was read reproduces the file bit for bit
    assert R.write_safetensors(scratch / "again.safetensors", ts, meta) == sha


def test_safetensors_rejects_corrupt_files(scratch):
    d = make_ckpt(scratch / "ck", {"w": ("F32", np.ones((2, 2)))})
    raw = (d / "model.safetensors").read_bytes()
    hlen = int.from_bytes(raw[:8], "little")
    cases = [(raw[:-1], "data range"), (b"\x01\x02", "truncated"),
             ((10 ** 9).to_bytes(8, "little") + raw[8:], "header length out of range")]
    header = json.loads(raw[8:8 + hlen])
    header["w"]["shape"] = [3, 2]
    hb = json.dumps(header).encode()
    cases.append((len(hb).to_bytes(8, "little") + hb + raw[8 + hlen:], "data range"))
    header["w"] = {"dtype": "Q4", "shape": [2, 2], "data_offsets": [0, 16]}
    hb = json.dumps(header).encode()
    cases.append((len(hb).to_bytes(8, "little") + hb + raw[8 + hlen:], "unknown dtype"))
    header["w"] = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    header["v"] = {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}
    hb = json.dumps(header).encode()
    cases.append((len(hb).to_bytes(8, "little") + hb + raw[8 + hlen:], "overlap"))
    for blob, want in cases:
        p = scratch / "bad.safetensors"
        p.write_bytes(blob)
        with pytest.raises(ValueError, match=want):
            R.read_safetensors(p)


# --------------------------------------------------------------------------- bf16

def test_bf16_roundtrip_is_exact_for_every_bit_pattern():
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    back = np.frombuffer(R.from_f64("BF16", R.to_f64("BF16", bits.tobytes())), dtype="<u2")
    nan = ((bits & 0x7F80) == 0x7F80) & ((bits & 0x007F) != 0)
    np.testing.assert_array_equal(back[~nan], bits[~nan])
    assert (((back[nan] & 0x7FC0) == 0x7FC0) & ((back[nan] & 0x8000) == (bits[nan] & 0x8000))).all()  # quiet NaN, sign kept


def test_bf16_rounds_to_nearest_even_without_double_rounding():
    one = 1.0
    cases = [
        (one + 2 ** -8, 0x3F80),  # tie between 1.0 and 1+2^-7 -> even (1.0)
        (one + 3 * 2 ** -8, 0x3F82),  # tie between 0x3F81 and 0x3F82 -> even
        (one + 2 ** -8 + 2 ** -40, 0x3F81),  # just above the tie; float32 first would round down to the tie
        (one + 2 ** -8 - 2 ** -40, 0x3F80),
        (-(one + 2 ** -8 + 2 ** -40), 0xBF81),
        (3.4e38, 0x7F80),  # above the largest bf16 -> inf
        (float(np.finfo(np.float32).max), 0x7F80),
        (-math.inf, 0xFF80),
        (2.0 ** -133, 0x0001),  # smallest bf16 subnormal
        (0.0, 0x0000),
        (-0.0, 0x8000),
    ]
    got = np.frombuffer(R.from_f64("BF16", np.array([x for x, _ in cases], dtype=np.float64)), dtype="<u2")
    assert [hex(v) for v in got] == [hex(b) for _, b in cases]
    assert np.frombuffer(R.from_f64("BF16", np.array([math.nan])), dtype="<u2")[0] & 0x7FC0 == 0x7FC0
    rng = np.random.default_rng(1)
    y = rng.normal(size=4000) * np.exp(rng.uniform(-30, 30, size=4000))
    np.testing.assert_array_equal(np.frombuffer(R.from_f64("BF16", y), dtype="<u2"), bf16_reference(y))


# --------------------------------------------------------------------------- rebase

def three(scratch, rng):
    """new_base, old_base and child checkpoints with one F32, one BF16 and one F16 tensor each."""
    def arrays():
        return {"w.f32": ("F32", rng.normal(size=(6, 4)) * 0.05), "w.bf16": ("BF16", rng.normal(size=6)),
                "w.f16": ("F16", rng.normal(size=3) * 0.1)}
    return make_ckpt(scratch / "new", arrays()), make_ckpt(scratch / "old", arrays()), make_ckpt(scratch / "child", arrays())


def test_rebase_formula_f32_bf16_f16(scratch):
    nb, ob, ch = three(scratch, np.random.default_rng(2))
    lam = 0.7
    rec = R.rebase(nb, ob, ch, scratch / "out", lam=lam)
    n, o, c, out = values(nb), values(ob), values(ch), values(scratch / "out")
    exact = {k: n[k] + lam * (c[k] - o[k]) for k in n}
    np.testing.assert_array_equal(out["w.f32"], exact["w.f32"].astype(np.float32))  # one rounding from float64
    np.testing.assert_array_equal(out["w.f16"], exact["w.f16"].astype(np.float16))
    np.testing.assert_array_equal(np.frombuffer(tensors_of(scratch / "out")["w.bf16"]["data"], dtype="<u2"),
                                  bf16_reference(exact["w.bf16"]))
    assert rec["lam"] == lam and rec["tensors"] == 3 and rec["dtypes"] == {"BF16": 1, "F16": 1, "F32": 1}
    tv = math.sqrt(sum(float(np.square(c[k] - o[k]).sum()) for k in n))
    assert rec["task_vector_norm"] == pytest.approx(tv)


def test_identities_lam_zero_and_equal_bases(scratch):
    nb, ob, ch = three(scratch, np.random.default_rng(3))
    rec = R.rebase(nb, ob, ch, scratch / "lam0", lam=0.0)
    assert (scratch / "lam0" / "model.safetensors").read_bytes() == (nb / "model.safetensors").read_bytes()
    assert rec["out"]["model_sha256"] == rec["inputs"]["new_base"]["model_sha256"]

    # new_base == old_base -> child, bit for bit, also where float arithmetic would not give it back:
    # 1 + (1e-12 - 1) != 1e-12 even in float64.
    base = {"w.f32": ("F32", [[1.0, -3.0, 0.25], [1e-3, 7.0, -0.0]]), "w.bf16": ("BF16", [1.0, 1e-30, -2.5, 0.0])}
    kid = {"w.f32": ("F32", [[1e-12, 2.0, 0.25], [-5e-20, 7.0, 0.0]]), "w.bf16": ("BF16", [1e-12, 3.0, -2.5, -1e-38])}
    b, k = make_ckpt(scratch / "b", base), make_ckpt(scratch / "k", kid)
    R.rebase(b, b, k, scratch / "same")
    got, want = tensors_of(scratch / "same"), tensors_of(k)
    assert {n: bytes(t["data"]) for n, t in got.items()} == {n: bytes(t["data"]) for n, t in want.items()}

    # a child that left an element unchanged keeps new_base's value there exactly
    nb2 = make_ckpt(scratch / "nb2", {"w.f32": ("F32", [[0.1, 0.2, 0.3], [0.4, 0.5, -0.0]]), "w.bf16": ("BF16", [0.1, 0.2, 0.3, 0.4])})
    R.rebase(nb2, b, b, scratch / "unchanged", lam=1.37)
    got = tensors_of(scratch / "unchanged")
    assert {n: bytes(t["data"]) for n, t in got.items()} == {n: bytes(t["data"]) for n, t in tensors_of(nb2).items()}


def test_output_dir_files_state_and_record(scratch):
    nb, ob, ch = three(scratch, np.random.default_rng(4))
    (nb / "notes.txt").write_text("copied")
    (nb / "model.tmp").write_text("left by a crashed writer")
    rec = R.rebase(nb, ob, ch, scratch / "out", lam=0.5)
    out = scratch / "out"
    assert sorted(p.name for p in out.iterdir()) == ["model.safetensors", "notes.txt", "rebase.json", "state.json"]
    assert not (scratch / "out.tmp").exists()
    sha = hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    st = json.loads((out / "state.json").read_text())
    nst = json.loads((nb / "state.json").read_text())
    assert st["model_sha256"] == sha and st["parent_sha256"] == nst["model_sha256"]
    assert st["config"] == nst["config"] and st["loss"] is None and st["val_loss"] is None
    assert st["rebase"]["lam"] == 0.5 and st["rebase"]["child"] == json.loads((ch / "state.json").read_text())["model_sha256"]
    disk = json.loads((out / "rebase.json").read_text())
    assert disk == json.loads(json.dumps(rec))
    for role, d in (("new_base", nb), ("old_base", ob), ("child", ch)):
        assert rec["inputs"][role]["model_sha256"] == hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest()
    assert rec["out"]["model_sha256"] == sha and rec["copied"] == ["notes.txt"] and rec["not_copied"] == ["optim.safetensors"]
    assert rec["formula"] == "out = new_base + lam * (child - old_base)"
    assert R.read_safetensors(out / "model.safetensors")[1] == {"format": "pt", "step": "7"}  # new_base's __metadata__


def test_refusals(scratch):
    rng = np.random.default_rng(5)
    good = {"a": ("F32", rng.normal(size=(2, 3))), "b": ("BF16", rng.normal(size=4))}
    nb = make_ckpt(scratch / "nb", good)
    ob = make_ckpt(scratch / "ob", good)
    cases = {
        "tensor names differ": {"a": good["a"]},
        r"a: dtype/shape differ.*F32 \[2, 3\].*F32 \[3, 2\]": {"a": ("F32", rng.normal(size=(3, 2))), "b": good["b"]},
        "b: dtype/shape differ.*BF16.*F32": {"a": good["a"], "b": ("F32", rng.normal(size=4))},
        "non-finite": {"a": ("F32", [[math.nan, 0, 0], [0, 0, 0]]), "b": good["b"]},
        "not a float type": {"a": good["a"], "b": ("BF16", np.ones(4)), "c": ("I32", 1)},
    }
    for i, (msg, arrays) in enumerate(cases.items()):
        if "c" in arrays:  # an integer tensor in all three checkpoints
            ts = {n: {"dtype": dt, "shape": list(np.shape(v)), "data": encode(dt, v) if dt != "I32" else np.int32(v).tobytes()}
                  for n, (dt, v) in arrays.items()}
            dirs = []
            for role in ("n", "o", "c"):
                d = scratch / f"int-{role}"
                d.mkdir()
                sha = R.write_safetensors(d / "model.safetensors", ts, None)
                (d / "state.json").write_text(json.dumps({"config": {"model": MODEL}, "model_sha256": sha}))
                dirs.append(d)
            with pytest.raises(ValueError, match=msg):
                R.rebase(*dirs, scratch / f"out{i}")
            continue
        ch = make_ckpt(scratch / f"child{i}", arrays)
        with pytest.raises(ValueError, match=msg):
            R.rebase(nb, ob, ch, scratch / f"out{i}")
        assert not (scratch / f"out{i}").exists() and not (scratch / f"out{i}.tmp").exists()
    with pytest.raises(ValueError, match="non-finite"):  # overflow of the result
        R.rebase(nb, ob, make_ckpt(scratch / "big", {"a": ("F32", np.full((2, 3), 3e38)), "b": good["b"]}), scratch / "o", lam=1e10)
    with pytest.raises(ValueError, match="model configs differ"):
        R.rebase(nb, ob, make_ckpt(scratch / "cfg", good, model={**MODEL, "n_heads": 4}), scratch / "o")
    with pytest.raises(ValueError, match="state.json records"):
        R.rebase(nb, ob, make_ckpt(scratch / "sha", good, sha_override="0" * 64), scratch / "o")
    with pytest.raises(FileNotFoundError, match="no model.safetensors"):
        R.rebase(nb, ob, scratch / "missing", scratch / "o")
    with pytest.raises(ValueError, match="lam must be finite"):
        R.rebase(nb, ob, nb, scratch / "o", lam=math.inf)
    (nb / "state.json").unlink()
    with pytest.raises(ValueError, match="config.model is required"):
        R.rebase(nb, ob, ob, scratch / "o")
    with pytest.raises(FileExistsError, match="never overwrites"):
        R.rebase(ob, ob, ob, nb)
    assert not (scratch / "o").exists()


def test_tmp_dir_from_an_aborted_rebase_is_replaced_foreign_one_refused(scratch):
    nb, ob, ch = three(scratch, np.random.default_rng(6))
    tmp = scratch / "out.tmp"
    tmp.mkdir()
    (tmp / "model.safetensors").write_bytes(b"half written")
    (tmp / R.TMP_MARK).write_text("{}")
    R.rebase(nb, ob, ch, scratch / "out")
    assert not tmp.exists() and (scratch / "out" / "rebase.json").exists()
    (scratch / "out2.tmp").mkdir()
    (scratch / "out2.tmp" / "precious").write_text("not ours")
    with pytest.raises(FileExistsError, match="not left by rebase"):
        R.rebase(nb, ob, ch, scratch / "out2")
    assert (scratch / "out2.tmp" / "precious").exists() and not (scratch / "out2").exists()


def test_cli(scratch):
    nb, ob, ch = three(scratch, np.random.default_rng(7))
    cmd = [sys.executable, str(ROOT / "scripts" / "rebase.py"), "--new-base", str(nb), "--old-base", str(ob),
           "--child", str(ch), "--lam", "0.25", "--out", str(scratch / "out")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout)
    assert rec["lam"] == 0.25 and rec["out"]["model_sha256"] == json.loads((scratch / "out" / "state.json").read_text())["model_sha256"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and json.loads(r.stderr)["type"] == "error" and "exists" in r.stderr


# --------------------------------------------------------------------------- real checkpoints

REAL = {"new": ROOT / "runs/base-g2/FINAL", "old": ROOT / "runs/base-s/FINAL", "child": ROOT / "runs/quasnir-1/FINAL"}
CODE_V2 = ROOT / "data/out/code-v2/train.meta.json"


@pytest.mark.skipif(not FORGE.exists() or not CODE_V2.exists() or not all((d / "model.safetensors").exists() for d in REAL.values()),
                    reason="forge binary, data/out/code-v2 or the base-s/base-g2/quasnir-1 checkpoints missing")
def test_real_rebase_quasnir1_onto_base_g2_evaluates(scratch):
    out = scratch / "quasnir-1-on-base-g2"
    rec = R.rebase(REAL["new"], REAL["old"], REAL["child"], out)
    for role, key in (("new_base", "new"), ("old_base", "old"), ("child", "child")):
        assert rec["inputs"][role]["model_sha256"] == json.loads((REAL[key] / "state.json").read_text())["model_sha256"]
    assert rec["tensors"] > 0 and rec["dtypes"] == {"F32": rec["tensors"]} and rec["task_vector_norm"] > 0
    st = json.loads((out / "state.json").read_text())
    assert st["model_sha256"] == rec["out"]["model_sha256"] == hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    r = subprocess.run(["nice", "-n", "15", str(FORGE), "eval", "--ckpt", str(out), "--data", str(CODE_V2), "--batches", "1",
                        "--seq", "64", "--threads", "2"], capture_output=True, text=True, timeout=900, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-500:]
    ev = json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1])
    assert ev["type"] == "eval" and ev["sha256"] == rec["out"]["model_sha256"]
    assert isinstance(ev["loss"], float) and math.isfinite(ev["loss"]) and 0 < ev["loss"] < 20
