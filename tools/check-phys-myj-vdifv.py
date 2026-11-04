"""Done-rule check for phys-myj-vdifv: compile the kernel, run the wrapper once on the full-size state."""
import sys, time
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cupy as cp
from namgpu import config, grid as gridmod, state as statemod
from namgpu.driver import Ctx
from namgpu import kernels
from namgpu.physics.myj import phys_myj_vdifv as w

cfg = config.load()
g = gridmod.build(cfg)
st = statemod.State(g).randomize(seed=1)
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
kernels.module("phys_myj_vdifv").compile()
t0 = time.perf_counter()
with ctx.stream:
    w.run(ctx)
ctx.stream.synchronize()
dt = time.perf_counter() - t0
bad = [n for n in ("U", "V") if not bool(cp.isfinite(st[n]).all())]
print("grid", g.im, g.jm, g.lm, "time_s", round(dt, 3), "nonfinite_fields", bad)
print("U range", float(st.U.min()), float(st.U.max()), "V range", float(st.V.min()), float(st.V.max()))
sys.exit(1 if bad else 0)
