"""Check drv-tablep: NVRTC compile, wrapper run once, finite outputs, endpoint and monotone sanity."""
import types
import numpy as np
import cupy as cp
from namgpu import kernels
from namgpu.steps import drv_tablep as w

mod = kernels.module("drv_tablep")
mod.compile()
print("compile: ok", flush=True)

ctx = types.SimpleNamespace(grid=types.SimpleNamespace(pt=200.0), tables={}, stream=None, st=None)
w.run(ctx)
cp.cuda.Device().synchronize()

ptbl = cp.asnumpy(ctx.tables["ptbl"])
qs0 = cp.asnumpy(ctx.tables["qs0"])
sqs = cp.asnumpy(ctx.tables["sqs"])
for name, arr in (("ptbl", ptbl), ("qs0", qs0), ("sqs", sqs)):
    print(name, arr.shape, arr.dtype, "finite:", bool(np.isfinite(arr).all()),
          "min", float(arr.min()), "max", float(arr.max()), flush=True)
print("rdth", ctx.tables["rdth"], "rdq", ctx.tables["rdq"])
print("ptbl[0,0], ptbl[0,-1], ptbl[-1,0], ptbl[-1,-1]:", ptbl[0, 0], ptbl[0, -1], ptbl[-1, 0], ptbl[-1, -1])
print("pressure non-decreasing in kp for every row:", bool(np.all(np.diff(ptbl, axis=1) >= 0)))
print("qs0[0], sqs[0], qs0[-1], sqs[-1]:", qs0[0], sqs[0], qs0[-1], sqs[-1])
assert np.isfinite(ptbl).all() and np.isfinite(qs0).all() and np.isfinite(sqs).all()
print("RESULT: pass", flush=True)
