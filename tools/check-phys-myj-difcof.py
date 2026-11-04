"""Done-rule check for phys-myj-difcof: compile the kernel, run the wrapper once on the full-size state.
Inputs come from the explicit fixture (no producer runs here); a missing producer must raise."""
import sys, time
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cupy as cp
from namgpu import config, grid as gridmod, state as statemod
from namgpu.driver import Ctx
from namgpu import kernels
from namgpu.physics.myj import phys_myj_difcof as w

cfg = config.load()
g = gridmod.build(cfg)
st = statemod.State(g).randomize(seed=1)
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
try:
    w.run(ctx)
    sys.exit("FAIL: run() accepted a state with no MYJ_GM producer")
except RuntimeError as e:
    print("refusal ok:", e)
w.fixture_inputs(ctx)
kernels.module("phys_myj_difcof").compile()
t0 = time.perf_counter()
with ctx.stream:
    out = w.run(ctx)
ctx.stream.synchronize()
dt = time.perf_counter() - t0
bad = [n for n in ("MYJ_AKM", "MYJ_AKH") if not bool(cp.isfinite(out[n]).all())]
top = float(cp.abs(out["MYJ_AKM"][-1]).max())
print("grid", g.im, g.jm, g.lm, "time_s", round(dt, 3), "nonfinite", bad, "level_LM_max_abs", top)
print("AKM range", float(out["MYJ_AKM"][:-1].min()), float(out["MYJ_AKM"][:-1].max()))
sys.exit(1 if bad or top != 0.0 else 0)
