"""Golden vectors from ml_dtypes/numpy (independent reference) for forge-formats tests.

Columns (hex): f32 bits, bf16, fp16, e4m3 (non-saturating RNE), e5m2 (non-saturating RNE),
e4m3 (saturating), e5m2 (saturating), fp4 e2m1 (saturating). NaN inputs are excluded.
"""
import sys
import numpy as np
import ml_dtypes as md

rng = np.random.default_rng(20261005)
n = 6000
parts = []
# uniform bit patterns
parts.append(rng.integers(0, 2**32, n // 3, dtype=np.uint64).astype(np.uint32))
# exponent window [2^-20, 2^20)
b = rng.integers(0, 2**32, n // 3, dtype=np.uint64).astype(np.uint32)
e = rng.integers(107, 147, n // 3).astype(np.uint32)
parts.append((b & np.uint32(0x807FFFFF)) | (e << np.uint32(23)))
# values right at / around grid midpoints of fp8: low mantissa bits zero, then +-1 ulp
b = rng.integers(0, 2**32, n // 3, dtype=np.uint64).astype(np.uint32)
e = rng.integers(100, 145, n // 3).astype(np.uint32)
b = (b & np.uint32(0x80780000)) | np.uint32(0x00040000) | (e << np.uint32(23))
d = rng.integers(-1, 2, n // 3).astype(np.int64)
parts.append((b.astype(np.int64) + d).astype(np.uint32))
xb = np.concatenate(parts)
x = xb.view(np.float32)
x = x[~np.isnan(x)]
xb = x.view(np.uint32)
with np.errstate(all='ignore'):
    cols = [
        xb,
        x.astype(md.bfloat16).view(np.uint16),
        x.astype(np.float16).view(np.uint16),
        x.astype(md.float8_e4m3fn).view(np.uint8),
        x.astype(md.float8_e5m2).view(np.uint8),
        np.clip(x, -448, 448).astype(md.float8_e4m3fn).view(np.uint8),
        np.clip(x, -57344, 57344).astype(md.float8_e5m2).view(np.uint8),
        np.clip(x, -6, 6).astype(md.float4_e2m1fn).view(np.uint8) & 0xF,
    ]
out = sys.stdout
out.write('# ml_dtypes %s, numpy %s; see tests/golden_mldtypes.rs for column meaning\n' % (md.__version__, np.__version__))
for i in range(len(x)):
    out.write('%08x %04x %04x %02x %02x %02x %02x %x\n' % tuple(int(c[i]) for c in cols))
