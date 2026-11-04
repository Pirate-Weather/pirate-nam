"""Done-rule check for phys-noah-nopac (CONTRACT.md section 6, parts 1 and 2).

1. compiles the kernel file (NVRTC via CuPy);
2. runs the wrapper ONCE on a full-size 954 x 835 column set and asserts no NaN/Inf in every
   array it writes (outputs, updated soil state, CMC, T1, BETA).
"""
import sys
import time

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import cupy as cp
from namgpu import kernels
from namgpu.physics.noah import phys_noah_nopac as w

kernels.module("phys_noah_nopac").compile()
print("compile ok", flush=True)

t0 = time.perf_counter()
out = w.run()
cp.cuda.Stream.null.synchronize()
print("wrapper time_s", round(time.perf_counter() - t0, 4), flush=True)
for name, arr in out.items():
    a = arr.get()
    fin = bool(np.isfinite(a).all())
    print(name, tuple(a.shape), "finite", fin, "min", float(a.min()), "max", float(a.max()), flush=True)
    assert fin, f"NaN/Inf in {name}"
print("PASS", flush=True)
