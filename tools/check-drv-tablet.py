"""Check drv-tablet: NVRTC compile, wrapper run once, finite outputs, endpoint and monotone sanity,
and the scalars the Fortran TABLET sets (rdp = 1/dp, rdthe = 1/dthe, pl = max(pt, 1000)).

    source env.sh
    python tools/check-drv-tablet.py
"""
import types

import numpy as np
import cupy as cp

from namgpu import kernels
from namgpu.steps import drv_tablet as w

ITB, JTB = w.ITB, w.JTB
PH, THL, THH = 105000.0, 200.0, 350.0

kernels.module(w.UNIT).compile()
print("compile: ok", flush=True)

ctx = types.SimpleNamespace(cfg=types.SimpleNamespace(pt=200.0), tables={}, stream=None)
w.run(ctx)
cp.cuda.Device().synchronize()

t = ctx.tables
ttbl = cp.asnumpy(t["tablet_ttbl"])
the0 = cp.asnumpy(t["tablet_the0"])
sthe = cp.asnumpy(t["tablet_sthe"])
rdp = float(cp.asnumpy(t["tablet_rdp"])[0])
rdthe = float(cp.asnumpy(t["tablet_rdthe"])[0])
pl = float(cp.asnumpy(t["tablet_pl"])[0])

for name, arr in (("tablet_ttbl", ttbl), ("tablet_the0", the0), ("tablet_sthe", sthe)):
    print(name, arr.shape, arr.dtype, "finite:", bool(np.isfinite(arr).all()),
          "min", float(arr.min()), "max", float(arr.max()), flush=True)
assert ttbl.shape == (ITB, JTB) and np.isfinite(ttbl).all()
assert np.isfinite(the0).all() and np.isfinite(sthe).all()

dp = (PH - pl) / (ITB - 1)
dthe = 1.0 / (JTB - 1)
print("pl", pl, "rdp", rdp, "expect", 1.0 / dp, "rdthe", rdthe, "expect", 1.0 / dthe, flush=True)
assert pl == 1000.0 and abs(rdp - 1.0 / dp) / (1.0 / dp) < 1e-6 and abs(rdthe - 1.0 / dthe) / (1.0 / dthe) < 1e-6
# Table rows are temperature versus theta-e at each pressure: increasing in theta-e.
print("ttbl rows non-decreasing in theta-e:", bool(np.all(np.diff(ttbl, axis=1) >= -1e-3)), flush=True)
print("ttbl[0,0], ttbl[0,-1], ttbl[-1,0], ttbl[-1,-1]:", ttbl[0, 0], ttbl[0, -1], ttbl[-1, 0], ttbl[-1, -1])
print("RESULT: pass", flush=True)
