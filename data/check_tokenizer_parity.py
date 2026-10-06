"""Parity of the Rust tokenizer (`forge tokenize`) with scp_model.bpe on real
text (CPython stdlib, docs) and random Unicode across scripts, marks, spaces
and symbols. Needs the swarm-compute-protocol- checkout (SCP_ROOT).
Usage: python3 data/check_tokenizer_parity.py /tmp/texts.jsonl"""
import glob, json, random, subprocess, sys, importlib.util, types, time
import os
SCP = os.path.join(os.environ.get("SCP_ROOT", "/home/user/swarm-compute-protocol-"), "model", "scp_model")
pkg = types.ModuleType("scp_model"); pkg.__path__ = [SCP]; sys.modules["scp_model"] = pkg
spec = importlib.util.spec_from_file_location("scp_model.bpe", SCP + "/bpe.py"); bpe = importlib.util.module_from_spec(spec); spec.loader.exec_module(bpe)
tok = bpe.BPETokenizer.load("data/stores/base/tokenizer.json")
texts = []
for f in sorted(glob.glob("/usr/lib/python3.11/*.py"))[:120] + glob.glob("docs/*.md") + glob.glob("README.md") + glob.glob("/home/user/swarm-compute-protocol-/**/*.md", recursive=True)[:40]:
    try: texts.append(open(f, encoding="utf-8").read())
    except Exception: pass
rng = random.Random(5)
ranges = [(0x20, 0x7e), (0x00, 0x20), (0xa0, 0x2ff), (0x300, 0x36f), (0x370, 0x3ff), (0x400, 0x4ff), (0x600, 0x6ff), (0x900, 0x97f),
          (0x1680, 0x1680), (0x2000, 0x206f), (0x2150, 0x218f), (0x3000, 0x303f), (0x4e00, 0x4eff), (0xff00, 0xffef), (0x1f300, 0x1f64f), (0x1d400, 0x1d7ff)]
for _ in range(3000):
    n = rng.randint(1, 40)
    s = "".join(chr(rng.randint(*rng.choice(ranges))) for _ in range(n))
    texts.append("".join(c for c in s if not 0xd800 <= ord(c) <= 0xdfff))
open(sys.argv[1], "w").write("\n".join(json.dumps(t) for t in texts) + "\n")
t0 = time.time(); py = [tok.encode(t) for t in texts]; tpy = time.time() - t0
t0 = time.time(); out = subprocess.run(["target/release/forge", "tokenize", "--jsonl", sys.argv[1]], check=True, capture_output=True, text=True).stdout; trs = time.time() - t0
rs = [json.loads(l) for l in out.splitlines()]
bad = [i for i, (a, b) in enumerate(zip(py, rs)) if a != b]
chars = sum(len(t) for t in texts)
print(f"texts {len(texts)} chars {chars} tokens {sum(map(len, py))} mismatches {len(bad)} python {tpy:.1f}s rust {trs:.1f}s")
for i in bad[:5]:
    print(repr(texts[i][:80]))
sys.exit(1 if bad else 0)
