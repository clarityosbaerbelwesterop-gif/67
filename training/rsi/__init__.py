"""Controlled recursive self-improvement (verifier-gated expert iteration / STaR).

Modules:
  tasks   procedurally generated, verifiable task families (code, text, mixed),
          difficulty levels 0..9, disjoint train / held-out seed spaces, 13-gram
          decontamination against data/stores/evals
  verify  sandboxed execution of candidate code (approach and limits of
          scripts/humaneval.py) and exact-match checking of text answers
  store   SCP CorpusStore writer for accepted samples mixed with a replay sample
  loop    the human-controlled round loop (gates, approval, STOP, limits, audit)

CLI: python3 -m training.rsi.loop --help   (see training/rsi/README.md)
"""
