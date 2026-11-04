import sys, time
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import cupy as cp
from namgpu import kernels
from namgpu.physics.myj import phys_myj_tables as w

kernels.module("phys_myj_tables").compile()
t0 = time.perf_counter()
out = w.run(None)
cp.cuda.runtime.deviceSynchronize()
dt = time.perf_counter() - t0
fin = {k: bool(cp.isfinite(v).all()) for k, v in out.items()}
print("time %.3f s" % dt)
print("finite", fin)
assert all(fin.values()), "NaN/Inf"
herff = out["myj_herff"].get()
print("herff[0]", herff[0], "herff[-1]", herff[-1], "rdxe", out["myj_rdxe"].get())
pt = out["myj_ptbl"].get()
tt = out["myj_ttbl"].get()
print("ptbl shape", pt.shape, "corners", pt[0, 0], pt[-1, -1], "ttbl shape", tt.shape, "corners", tt[0, 0], tt[-1, -1])
print("scalars rdth", out["myj_rdth"].get(), "rdq", out["myj_rdq"].get(), "rdp", out["myj_rdp"].get(), "rdthe", out["myj_rdthe"].get())
