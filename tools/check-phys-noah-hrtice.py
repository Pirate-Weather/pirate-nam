"""Done-rule check for phys-noah-hrtice (CONTRACT.md section 6, parts 1 and 2).

1. compiles the kernel file (NVRTC via CuPy);
2. runs the wrapper ONCE on a full-size 954 x 835 x 4 soil column set (random STC in the
   sea-ice range) and asserts no NaN/Inf in RHSTS, AI, BI, CI.
"""
import sys, time
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import cupy as cp
from namgpu import kernels
from namgpu.physics.noah import phys_noah_hrtice as w

kernels.module("phys_noah_hrtice").compile()
print("compile ok")

t0 = time.perf_counter()
out = w.run()
cp.cuda.Stream.null.synchronize()
print("wrapper time_s", round(time.perf_counter() - t0, 4))
for name, arr in out.items():
    a = arr.get()
    print(name, tuple(a.shape), "finite", bool(np.isfinite(a).all()), "min", float(a.min()), "max", float(a.max()))
    assert np.isfinite(a).all(), f"NaN/Inf in {name}"
print("PASS")
