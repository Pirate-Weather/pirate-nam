"""Done-rule check for phys-gwd-col-wave (CONTRACT.md section 6, parts 1 and 2).

1. compiles the kernel file (NVRTC via CuPy);
2. runs the wrapper ONCE on a full-size 954 x 835 x 60 batch and asserts no NaN/Inf in any output.
"""
import sys, time
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import cupy as cp
from namgpu import kernels
from namgpu.physics.gwd import phys_gwd_col_wave as w

kernels.module("phys_gwd_col_wave").compile()
print("compile ok")

t0 = time.perf_counter()
out = w.run()
cp.cuda.Stream.null.synchronize()
print("wrapper time_s", round(time.perf_counter() - t0, 4))
for name, arr in out.items():
    a = arr.get()
    print(name, tuple(a.shape), "finite", bool(np.isfinite(a).all()),
          "min", float(a.min()), "max", float(a.max()), "nonzero", int(np.count_nonzero(a)))
    assert np.isfinite(a).all(), f"NaN/Inf in {name}"
print("PASS")
