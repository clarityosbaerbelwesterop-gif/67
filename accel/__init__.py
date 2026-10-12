"""accel: forge (core_engine/) on JAX devices -- TPU, GPU, or CPU.

The same model, loss, initialisation, data sampler, AdamW, clipping and LR schedules as
`forge train`, the same run-config schema, the same stdout JSON lines and the same
checkpoint files, so checkpoints move freely between forge and accel and scripts that drive
forge (scripts/longrun.py, cluster/fabric.py) can drive accel through accel/bin/forge-accel.

  python -m accel.train  --config RUN.json [--precision f32|bf16]
  python -m accel.eval   --ckpt DIR --data META [--split val] [--batch 8] [--seq 256] [--batches 20]
  python -m accel.logits --ckpt DIR --prompt-ids 1,2,3 [--ctx N]

Measured parity with forge and how to run it on TPUs and GPUs: docs/ACCEL.md.
"""

__version__ = "1.0.0"
