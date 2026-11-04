"""Done-rule check for drv-step-plan (CONTRACT.md section 6).

1. Host plan check: drv_step_plan.check_plan() must report no problems (cadences, kwargs, Fortran order).
2. Device check: the nonfinite kernel compiles through kernels.get and counts exactly on a full-size
   954 x 835 x 60 float32 array: a clean array gives 0; a planted set of NaN and Inf values gives the
   planted count. Only one array (191 MiB) is allocated.
"""
import sys
from pathlib import Path

ROOT = Path.home() / "nam-gpu"
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402

from namgpu.steps import drv_step_plan as w  # noqa: E402

problems, pending = w.check_plan()
print("plan problems:", len(problems))
for p in problems:
    print("  ", p)
print("plan pending wrappers:", len(pending))
if problems:
    sys.exit(1)

shape = (60, 835, 954)
rng = cp.random.default_rng(7)
a = rng.random(shape, dtype=cp.float32)
c0 = int(w.nonfinite_count(a).get()[0])
print("clean array nonfinite count:", c0)

flat = a.reshape(-1)
planted_nan = 1000
planted_inf = 500
idx = np.random.default_rng(3).choice(flat.size, size=planted_nan + planted_inf, replace=False)
flat[idx[:planted_nan]] = cp.float32("nan")
flat[idx[planted_nan:]] = cp.float32("inf")
c1 = int(w.nonfinite_count(a).get()[0])
print("planted", planted_nan, "NaN and", planted_inf, "Inf; counted:", c1)
ok = (c0 == 0) and (c1 == planted_nan + planted_inf)
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
