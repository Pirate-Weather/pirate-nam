"""Host-only check: EF4T seen by dyn-vtoa at each TDFI step matches SOLVER_RUN's lifecycle.

    source ~/nam-gpu/env.sh && python tools/check-dyn-vtoa-dfi-ef4t.py
Expected (Fortran): backward steps see -(filt_dt/fund_dt)*ef4t0, forward steps see
+(filt_dt/fund_dt)*ef4t0 (changedir flip), and after the filter EF4T = ef4t0 again (EF4T_SV).
"""
from types import SimpleNamespace

import numpy as np

from namgpu.dynamics import dyn_vtoa
from namgpu.steps import drv_dfi_tdfi_driver as d

cfg = SimpleNamespace(dt=25.0, filter_method=3, filt_dt=8.0,
                      raw={"nsecs_bcktdfi": 80.0, "nsecs_fwdtdfi": 40.0})
ctx = SimpleNamespace(cfg=cfg, dt=25.0, st=SimpleNamespace(scalars={}))
ef0 = dyn_vtoa.consts_ef4t(cfg.dt)
seen = []


def integ(c, direction, dt, hdiff_on, mean_on, k):
    seen.append((direction, float(dt), float(c.st.scalars["EF4T"])))


noop = lambda *a, **k: None  # noqa: E731
ops = d.Ops(dyn_init=noop, dyn_sum=noop, dyn_average=noop, phy_save=noop, phy_restore=noop,
            integrate=integ)
res = d.run(ctx, ops)
bad = []
for direction, dt, e in seen:
    want = -(8.0 / 25.0) * ef0 if direction == "Bckward" else (8.0 / 25.0) * ef0
    if not np.isclose(e, want, rtol=1e-6, atol=0):
        bad.append((direction, e, want))
after = float(ctx.st.scalars["EF4T"])
print("steps", res, "ef0", float(ef0), "after", after)
print("bad steps", bad[:5], "count", len(bad), "n seen", len(seen))
assert not bad, bad[:5]
assert np.isclose(after, ef0, rtol=1e-6, atol=0), after
print("PASS dfi ef4t lifecycle")
