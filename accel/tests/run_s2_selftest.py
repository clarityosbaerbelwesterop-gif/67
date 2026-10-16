"""CPU test of the time-budget guard. Run from repo root:
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu python accel/tests/run_s2_selftest.py"""
import json, os, shutil, subprocess, sys, time
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, R)
from accel.run_s2 import plan_schedule, plan_total
from accel import mgpu as G
META = "/workspace/67-train/data/out/base-v3/train.meta.json"
TSV = "/workspace/contam-check/dedup-v4/basev3-full/kept_t50_drop20.tsv"
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_run_s2")

def run(out, deadline, extra=()):
    cmd = [sys.executable, "-m", "accel.run_s2", "--data", META, "--clean-tsv", TSV, "--out", out, "--deadline", str(deadline),
           "--tiny", "--no-verify-sha", "--batch", "16", "--accum", "2", "--seq", "64", "--mesh", "2x4", "--attn", "blockwise:64", *extra]
    r = subprocess.run(cmd, cwd=R, capture_output=True, text=True)
    ev = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    p = os.path.join(out, "DONE.json")
    return r.returncode, ev, (json.load(open(p)) if os.path.exists(p) else None), r.stderr[-400:]

def pick(d, ks): return {k: d.get(k) for k in ks} if d else None

def main():
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    try:
        print("plan_schedule:", {n: plan_schedule(n) for n in (1000, 1001, 999, 600, 300, 40, 3)})
        print("plan_total(20 done, 2.5 s/step, 2400 s left, reserve 600) =", plan_total(20, 2.5, 0, 2400, 600, 1000),
              "| (10000 s left) =", plan_total(20, 2.5, 0, 10000, 600, 1000))
        h = G.HP.preset("s2_run1", **plan_schedule(700))
        print("lr for total=700:", {s: f"{float(h.lr_at(s)):.4e}" for s in (0, 299, 300, 559, 560, 630, 699)})
        rc, ev, dn, err = run(os.path.join(TMP, "a"), time.time() + 3600, ["--cap", "30", "--plan-at", "8", "--reserve-s", "5"])
        pl = [e for e in ev if e.get("event") == "plan"]
        print("A rc", rc, "plan", pl[0] if pl else err, "| DONE", pick(dn, ("status", "step", "planned", "mean_tok_s")))
        steps = sorted(d for d in os.listdir(os.path.join(TMP, "a")) if d.startswith("step-"))
        print("A export bytes", dn and dn["export"]["bytes"], "| step dirs on disk", steps, "| ckpt dir exists", os.path.exists(os.path.join(TMP, "a", "ckpt")))
        rc, ev, dn, err = run(os.path.join(TMP, "b"), time.time() + 45, ["--cap", "1000", "--plan-at", "8", "--reserve-s", "5", "--fake-step-s", "0.5"])
        pl = [e for e in ev if e.get("event") == "plan"]; st = [e for e in ev if e.get("event") == "start"]
        print("B rc", rc, "setup_s", st and st[0]["setup_s"], "plan", pl[0] if pl else err, "| DONE", pick(dn, ("status", "step", "planned", "wall_s")))
        rc, ev, dn, err = run(os.path.join(TMP, "c"), time.time() + 40, ["--cap", "1000", "--plan-at", "100000", "--reserve-s", "3", "--fake-step-s", "0.5"])
        g = [e for e in ev if e.get("event") == "deadline_guard"]
        print("C rc", rc, "guard", g[0] if g else err, "| DONE", pick(dn, ("status", "step")), "export present", bool(dn and dn.get("export")))
        rc, ev, dn2, err = run(os.path.join(TMP, "c"), time.time() + 25, ["--cap", "1000", "--plan-at", "100000", "--reserve-s", "3", "--fake-step-s", "0.5"])
        st = [e for e in ev if e.get("event") == "start"]
        print("D step dirs kept (no deletion):", sorted(d for d in os.listdir(os.path.join(TMP, "c")) if d.startswith("step-")))
        print("D (resume C) rc", rc, "start", st and pick(st[0], ("src", "step", "plan")), "| DONE", pick(dn2, ("status", "step")))
        src = open(os.path.join(R, "accel", "run_s2.py")).read()
        print("static: 'rmtree' in run_s2.py:", "rmtree" in src, "| 'unlink'/'remove(' :", "unlink" in src or "remove(" in src)
        out = os.path.join(TMP, "e")
        cmd = [sys.executable, "-m", "accel.run_s2", "--data", META, "--clean-tsv", "/nonexistent.tsv", "--out", out, "--deadline", str(time.time() + 99), "--tiny", "--no-verify-sha"]
        r = subprocess.run(cmd, cwd=R, capture_output=True, text=True); dn = json.load(open(os.path.join(out, "DONE.json")))
        print("E rc", r.returncode, "DONE", dn["status"], dn["error"][:110])
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

if __name__ == "__main__":
    main()
