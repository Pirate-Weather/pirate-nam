"""Done-rule check for phys-rrtm-lw-taugb13: NVRTC compile, one full-size run, no NaN/Inf.

Full size: 954 x 835 x 60 (CONTRACT section 5). Inputs are random but in the index ranges the
setcoef and table layouts allow, so every table read is in bounds. colamt has the Fortran maxgas = 7
planes (plane 7 is CO). Also checks that only rows NS13+1..NS13+4 (zero-based 130..133) are written.
Usage (node-2, on a short pool lease): python3 tools/check_phys_rrtm_lw_taugb13.py
"""
import time
from types import SimpleNamespace

import cupy as cp
import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_lw_taugb13 as unit

IM, JM, NLAY = 954, 835, 60
rng = cp.random.default_rng(1234)


def u(shape, lo, hi, dtype=cp.float32):
    return (lo + (hi - lo) * rng.random(shape, dtype=cp.float32)).astype(dtype)


def ri(shape, lo, hi):
    return rng.integers(lo, hi + 1, size=shape, dtype=cp.int32)


t0 = time.time()
mod = kernels.module(unit.UNIT)
mod.get_function(unit.KERNEL)
print(f"compile ok {time.time() - t0:.1f}s", flush=True)

ctx = SimpleNamespace(tables={}, stream=None)
tab = unit.run(ctx)
chi = u((unit.NREF, unit.NGAS_REF), 0.1, 1.0)

lt = ri((JM, IM), 10, 50)
inp = {
    "laytrop": lt,
    "colamt": u((unit.NGAS_LW, NLAY, JM, IM), 1e-3, 1.0),
    "coldry": u((NLAY, JM, IM), 1e18, 1e21),
    "jp": ri((NLAY, JM, IM), 1, 11),
    "jt": ri((NLAY, JM, IM), 1, 4),
    "jt1": ri((NLAY, JM, IM), 1, 4),
    "rfrate": u((2, unit.NRATES, NLAY, JM, IM), 0.1, 2.0),
    "fac00": u((NLAY, JM, IM), 0, 1), "fac01": u((NLAY, JM, IM), 0, 1),
    "fac10": u((NLAY, JM, IM), 0, 1), "fac11": u((NLAY, JM, IM), 0, 1),
    "selffac": u((NLAY, JM, IM), 0, 1e-3), "selffrac": u((NLAY, JM, IM), 0, 1),
    "indself": ri((NLAY, JM, IM), 1, 9),
    "forfac": u((NLAY, JM, IM), 0, 1e-3), "forfrac": u((NLAY, JM, IM), 0, 1),
    "indfor": ri((NLAY, JM, IM), 1, 3),
    "minorfrac": u((NLAY, JM, IM), 0, 1),
    "indminor": ri((NLAY, JM, IM), 1, 18),
}
out = unit.alloc_outputs(NLAY, JM, IM)
t0 = time.time()
unit.taugb13(inp, chi, tab, out)
cp.cuda.Stream.null.synchronize()
print(f"run ok {time.time() - t0:.2f}s", flush=True)
for name in ("taug13", "fracs13"):
    a = out[name]
    bad = int((~cp.isfinite(a)).sum())
    print(name, a.shape, "nonfinite", bad, "min", float(a.min()), "max", float(a.max()), flush=True)
    if bad:
        raise SystemExit(f"{name}: {bad} NaN/Inf")

# scatter into full gpoint arrays for one column chunk: only rows NS13..NS13+3 may be written
CH = 64
full_t = cp.zeros((NLAY, unit.NGPTLW, JM, CH), dtype=cp.float32)
full_f = cp.zeros((NLAY, unit.NGPTLW, JM, CH), dtype=cp.float32)
loc = {"taug13": out["taug13"][:, :, :, :CH], "fracs13": out["fracs13"][:, :, :, :CH]}
unit.scatter_band13(loc, full_t, full_f)
rows = cp.zeros(unit.NGPTLW, dtype=bool)
rows[unit.NS13:unit.NS13 + unit.NG13] = True
written = cp.any(full_t != 0, axis=(0, 2, 3)) | cp.any(full_f != 0, axis=(0, 2, 3))
if not bool(cp.array_equal(written, rows)) and bool(cp.any(written & ~rows)):
    raise SystemExit("scatter wrote rows outside 131..134")
print("scatter rows written", [int(r) + 1 for r in cp.nonzero(written)[0].get()], flush=True)
print("PASS phys-rrtm-lw-taugb13: compile, full-size run, no NaN/Inf, scatter to rows 131..134")
