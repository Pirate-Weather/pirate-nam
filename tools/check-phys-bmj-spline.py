import sys, time
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cupy as cp
from namgpu import kernels
from namgpu.physics.bmj import phys_bmj_spline as w
mod = kernels.module("phys_bmj_spline")
t0 = time.perf_counter()
out = w.run(None)
cp.cuda.runtime.deviceSynchronize()
dt = time.perf_counter() - t0
bad = {k: bool(cp.isfinite(v).all()) for k, v in out.items()}
print("shape ynew", out["ynew"].shape, "y2", out["y2"].shape, "finite", bad, "time %.3f s" % dt)
assert all(bad.values()), "NaN/Inf"
# knot interpolation: spline must reproduce yold at the knots when evaluated there (one batch row, small)
xo = cp.asarray([[0.,1.,2.,3.,4.,5.]], dtype=cp.float32); yo = cp.asarray([[0.,1.,0.,1.,0.,1.]], dtype=cp.float32)
y2 = cp.zeros((1,6), cp.float32); xn = cp.asarray([[0.,1.,2.,3.,4.,4.99]], cp.float32); yn = cp.zeros((1,6), cp.float32)
w.spline_batch(1, 6, 6, xo, yo, y2, xn, yn)
print("knot values", yn.get().tolist())
