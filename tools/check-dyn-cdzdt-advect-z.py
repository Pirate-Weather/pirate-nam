"""Full-size check of dyn-cdzdt-advect-z on the 12 km grid (954 x 835 x 60): the review defect.

Runs the real wrapper and NVRTC kernel and checks:
  - FAH is bound from ctx.tables["grid"]["fah"] (init-grid-consts), not from the grid object:
    a missing init-grid-consts table raises the named RuntimeError before any launch;
  - normal full-size step: no NaN/Inf, W updated in the interior, w = 0 on the lateral boundary strips.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp  # noqa: E402

from namgpu.driver import Ctx  # noqa: E402
from namgpu.dynamics import dyn_cdzdt_advect_z as w  # noqa: E402
from namgpu.state import State  # noqa: E402

IM, JM, LM = 954, 835, 60
rng = np.random.default_rng(7)
DT = 25.0 + 1.0 / 3.0
fah_host = (-DT / (3.0 * 1.2e4 * (1 + 0.1 * rng.random(JM)))).astype(np.float32)  # Fortran fah sign/scale
grid = types.SimpleNamespace(
    im=IM, jm=JM, lm=LM,
    pdsg1=(1000.0 * (0.5 + rng.random(LM))).astype(np.float32),
)
st = State(grid, names=["Z", "PDO", "DSG2", "W"])
for name, shape in [("PFX", (LM, JM, IM)), ("PFY", (LM, JM, IM)),
                    ("PFNE", (LM, JM, IM)), ("PFNW", (LM, JM, IM))]:
    st.add(name, shape)
    st[name][...] = cp.asarray(rng.standard_normal(shape, dtype=np.float32) * 1e-2)
st["Z"][...] = cp.asarray(rng.uniform(0.0, 1.0e4, (LM + 1, JM, IM)).astype(np.float32))
st["PDO"][...] = cp.asarray(rng.uniform(5.0e4, 1.0e5, (JM, IM)).astype(np.float32))
st["DSG2"][...] = cp.asarray(rng.uniform(0.01, 0.05, LM).astype(np.float32))
st["W"][...] = cp.asarray(rng.standard_normal((LM, JM, IM), dtype=np.float32) * 1e-3)
w0 = st["W"].copy()

# 1. missing init-grid-consts table must raise the named error, not a grid-attribute error
ctx0 = Ctx(cfg=types.SimpleNamespace(dt=DT, raw={}), grid=grid, st=st, stream=None, first_step=False, dt=DT)
ctx0.tables = {}
try:
    w.run(ctx0)
    raise AssertionError("expected RuntimeError without init-grid-consts fah")
except RuntimeError as e:
    print("missing table ->", str(e)[:90])
    assert "fah" in str(e) and "init-grid-consts" in str(e)

# 2. bound FAH from ctx.tables["grid"]["fah"]: full-size step
ctx = Ctx(cfg=types.SimpleNamespace(dt=DT, raw={}), grid=grid, st=st, stream=None, first_step=False, dt=DT)
ctx.tables = {"grid": {"fah": cp.asarray(fah_host)}}
st["W"][...] = w0
w.run(ctx)
cp.cuda.Stream.null.synchronize()
finite = bool(cp.isfinite(st["W"]).all())
inner = (slice(None), slice(2, JM - 2), slice(2, IM - 2))
moved = float(cp.abs(st["W"][inner] - w0[inner]).max())
bdy_zero = bool((st["W"][:, 0, :] == 0).all() and (st["W"][:, JM - 1, :] == 0).all()
                and (st["W"][:, :, 0] == 0).all() and (st["W"][:, :, IM - 1] == 0).all())
print(f"full-size step: finite={finite} max|dW| interior={moved:.3e} lateral w=0={bdy_zero}")
assert finite and moved > 0 and bdy_zero
print("dyn-cdzdt-advect-z: FAH binding and full-size no-NaN check ok")
