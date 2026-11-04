"""One full-size run of phys-fer-col-satcheck: compile, launch, check no NaN/Inf, and check that the
levels above LSFC are not written as CLEAR or RHO. CLEAR comes from the real phys-fer-col-setup kernel
(CLEAR_col), not from random data. The column prep that builds FER_LSFC and the condensate fields is not
built yet, so LSFC and the condensate/thickness fields are random plausible values here. Full size
954 x 835 x 60. Run under a pool lease from ~/nam-gpu:  source env.sh && python tools/check_phys_fer_col_satcheck.py
"""
import sys
import time
from types import SimpleNamespace

import cupy as cp
import numpy as np

from namgpu.state import State
from namgpu.physics.fer_hires import phys_fer_col_satcheck as unit
from namgpu.physics.fer_hires import phys_fer_col_setup as setup

grid = SimpleNamespace(im=954, jm=835, lm=60)
cfg = SimpleNamespace(nprecip=2, dt=25.0)
st = State(grid, names={"T"})
ctx = SimpleNamespace(st=st, grid=grid, stream=None, tables={}, cfg=cfg, dt=25.0)

shape3 = (60, 835, 954)
rng = np.random.default_rng(7)
st["T"][...] = cp.asarray(rng.uniform(215.0, 305.0, size=shape3).astype(np.float32))
st.add(unit.IN_FER["P"], shape3)[...] = cp.asarray(rng.uniform(2.0e4, 1.0e5, size=shape3).astype(np.float32))
st.add(unit.IN_FER["Q"], shape3)[...] = cp.asarray(rng.uniform(1.0e-5, 2.0e-2, size=shape3).astype(np.float32))
st.add(unit.IN_FER["WC"], shape3)[...] = cp.asarray((rng.uniform(0.0, 1.0e-3, size=shape3) * (rng.uniform(size=shape3) < 0.5)).astype(np.float32))
for key in ("QI", "QR", "QW"):
    st.add(unit.IN_FER[key], shape3)[...] = cp.asarray(rng.uniform(0.0, 1.0e-4, size=shape3).astype(np.float32))
st.add(unit.IN_FER["THICK"], shape3)[...] = cp.asarray(rng.uniform(1.0, 1.0e3, size=shape3).astype(np.float32))
lsfc_np = rng.integers(20, 61, size=(835, 954)).astype(np.int32)
st.add(unit.LSFC_NAME, (835, 954), "int32")[...] = cp.asarray(lsfc_np)
setup.run(ctx)   # real CLEAR_col (phys-fer-col-setup), consumed by the satcheck unit below

t0 = time.time()
out = unit.run(ctx)
cp.cuda.Stream.null.synchronize()
print(f"unit ran in {time.time() - t0:.2f} s (includes NVRTC compile)", flush=True)

bad = []
for name, arr in out.items():
    if arr.dtype.kind == "f" and not bool(cp.isfinite(arr).all()):
        bad.append(name)
print("arrays written:", len(out), "non-finite:", bad, flush=True)
clear = out[unit.OUT_CLEAR]
rho = out[unit.OUT_RHO]
print("CLEAR fraction:", float(clear.mean()), "RHO min/max (non-clear):",
      float(cp.where(clear == 0, rho, cp.inf).min()), float(cp.where(clear == 0, rho, -cp.inf).max()),
      flush=True)
above = cp.arange(60).reshape(60, 1, 1) >= cp.asarray(lsfc_np)[None]
leak = int(cp.count_nonzero(above & ((clear != 0) | (rho != 0))))
print("written above LSFC (must be 0):", leak, flush=True)
if bad or leak:
    sys.exit(1)
print("PASS", flush=True)
