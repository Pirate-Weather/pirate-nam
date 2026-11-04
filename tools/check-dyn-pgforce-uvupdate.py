"""Full-size check of dyn-pgforce-uvupdate on the 12 km grid (954 x 835 x 60), both branches.

Runs the real wrapper and NVRTC kernel on random PGF scratch and checks:
  - fresh first step (restart false): TCU and TCV zeroed over the Fortran i/j range, no NaN/Inf;
  - restart first step (restart true, from cfg.raw as text): TCU/TCV updated, no NaN/Inf;
  - later step: TCU/TCV updated, no NaN/Inf;
  - the restart flag parses config text (.true., false, bool).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp  # noqa: E402

from namgpu.driver import Ctx  # noqa: E402
from namgpu.dynamics import dyn_pgforce_uvupdate as w  # noqa: E402
from namgpu.state import State  # noqa: E402

IM, JM, LM = 954, 835, 60
rng = np.random.default_rng(1)
grid = types.SimpleNamespace(
    im=IM, jm=JM, lm=LM,
    dxv=(2.9e4 * (1 + 0.1 * rng.random(JM))).astype(np.float32),
    dyv=float(3.0e4),
    pdsg1=(1000.0 * (0.5 + rng.random(LM))).astype(np.float32),
)
st = State(grid, names=["PD", "DSG2", "TCU", "TCV"])
for name, shape in [("PGX_3D", (LM, JM, IM)), ("PGY_3D", (LM, JM, IM)),
                    ("PGNE_3D", (LM, JM, IM)), ("PGNW_3D", (LM, JM, IM))]:
    st.add(name, shape)
    st[name][...] = cp.asarray(rng.standard_normal(shape, dtype=np.float32) * 1e-3)
st["PD"][...] = cp.asarray(rng.uniform(5.0e4, 1.0e5, (JM, IM)).astype(np.float32))
st["DSG2"][...] = cp.asarray(rng.uniform(0.01, 0.05, LM).astype(np.float32))
st["TCU"][...] = cp.asarray(rng.standard_normal((LM, JM, IM), dtype=np.float32))
st["TCV"][...] = cp.asarray(rng.standard_normal((LM, JM, IM), dtype=np.float32))
tcu0, tcv0 = st["TCU"].copy(), st["TCV"].copy()

def run_case(first, restart_raw):
    cfg = types.SimpleNamespace(raw={"restart": restart_raw}, dt=60.0)
    ctx = Ctx(cfg=cfg, grid=grid, st=st, stream=None, first_step=first, dt=60.0)
    st["TCU"][...] = tcu0
    st["TCV"][...] = tcv0
    w.run(ctx)
    cp.cuda.Stream.null.synchronize()
    u, v = st["TCU"], st["TCV"]
    finite = bool(cp.isfinite(u).all() and cp.isfinite(v).all())
    return finite, u, v

# fresh first step: zeroed on the interior i0 in [1, IM-3], j0 in [1, JM-3]
fin, u, v = run_case(True, "false")
inner = (slice(None), slice(1, JM - 2), slice(1, IM - 2))
zeroed = bool((u[inner] == 0).all() and (v[inner] == 0).all())
print(f"fresh first step: finite={fin} interior_zeroed={zeroed} "
      f"outside_unchanged={bool((u[0] == tcu0[0]).all())}")
assert fin and zeroed

# restart first step: updated, not zeroed
fin_r, ur, vr = run_case(True, ".true.")
moved = float(cp.abs(ur[inner] - tcu0[inner]).max())
print(f"restart first step: finite={fin_r} max|dTCU|={moved:.3e}")
assert fin_r and moved > 0

# later step: updated
fin_n, un, vn = run_case(False, "false")
moved_n = float(cp.abs(un[inner] - tcu0[inner]).max())
print(f"later step: finite={fin_n} max|dTCU|={moved_n:.3e}")
assert fin_n and moved_n > 0

# flag parsing
flags = [w._restart_flag(types.SimpleNamespace(cfg=types.SimpleNamespace(raw={"restart": r})))
         for r in (".true.", "true", "T", "false", ".false.", False, True)]
print("restart flags:", flags)
assert flags == [True, True, True, False, False, False, True]
print("dyn-pgforce-uvupdate: full-size checks ok")
