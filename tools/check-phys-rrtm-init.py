"""Done-rule check for phys-rrtm-init (full size 954 x 835 x 60, one run, no NaN/Inf).

Run 1: no header SGM. The unit must refuse (no stand-in) with the named breakage.
Run 2: ctx.tables["vcoord_header"]["SGM"] supplied. TEST FIXTURE ONLY: the values are grid.sg2, not the NAM
       header SGM (no real header exists on the live path yet). Proves the kernels and launches are finite and
       that LTOP and PTOPC come out in range. Real SGM still needs the init unit.
"""
import sys
from pathlib import Path
import numpy as np
import cupy as cp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from namgpu import config, grid as gridmod, kernels, state as statemod
from namgpu.driver import Ctx
from namgpu.physics.rrtm import phys_rrtm_init as M

cfg = config.load()
g = gridmod.build(cfg)
st = statemod.State(g).randomize(seed=1)
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
kernels.module("phys_rrtm_init").compile()
try:
    with ctx.stream:
        M.run(ctx)
    print("RUN1 unexpectedly ran without SGM", flush=True)
    sys.exit(1)
except RuntimeError as e:
    print("RUN1 refused as designed:", str(e)[:160], flush=True)

ctx.tables = {"vcoord_header": {"SGM": np.asarray(g.sg2, np.float64)}}   # TEST FIXTURE, see docstring
ctx.stream.synchronize()
with ctx.stream:
    tbl = M.run(ctx)
ctx.stream.synchronize()
bad = [n for n, a in st.fields.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
print("RUN2 NaN/Inf fields:", bad, flush=True)
print("LTOP", tbl["ltop"].get().tolist(), "PTOPC", tbl["ptopc"].get().tolist(), flush=True)
print("AXSD[0]", float(tbl["axsd"][0]), "AXSD[99] (1 sigma)", float(tbl["axsd"][99]), "RSQR", float(tbl["rsqr"][0]), flush=True)
sys.exit(1 if bad else 0)
