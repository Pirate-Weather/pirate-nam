# One full-size run of bnd-write-bc; checks the edge mapping and finiteness of ctx.bc,
# a synthetic 4-D species check (the 4-D kernel path), and the call-site gates.
import sys, json
from pathlib import Path
import numpy as np
import cupy as cp
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
ROOT = Path(_ROOT)
sys.path.insert(0, str(ROOT))
from namgpu import config, grid as gridmod, state as statemod
from namgpu.driver import Ctx
from namgpu.boundary import bnd_write_bc as w

cfg = config.load()
g = gridmod.build(cfg)
st = statemod.State(g).randomize(seed=1)
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
ctx.bc = None
with ctx.stream:
    w.run(ctx, site="recomp_off", tboco=10800.0, recomp_tend=False)
ctx.stream.synchronize()
im, jm, lm = int(g.im), int(g.jm), int(g.lm)
lnsh, lnsv = int(cfg.lnsh), int(cfg.lnsv)
bad = [k for k, a in ctx.bc.items() if not bool(cp.isfinite(a).all())]
PD = cp.asnumpy(st["PD"]); U = cp.asnumpy(st["U"]); T = cp.asnumpy(st["T"])
B = {k: cp.asnumpy(v) for k, v in ctx.bc.items()}
res = {}
res["pd_N"] = bool(np.array_equal(B[("h2d","PD",0)][0,0,0], PD[jm-lnsh:jm, :]))
res["pd_S"] = bool(np.array_equal(B[("h2d","PD",1)][0,0,0], PD[0:lnsh, :]))
res["pd_W"] = bool(np.array_equal(B[("h2d","PD",2)][0,0,0], PD[:, 0:lnsh].T))
res["pd_E"] = bool(np.array_equal(B[("h2d","PD",3)][0,0,0], PD[:, im-lnsh:im].T))
res["t_N"] = bool(np.array_equal(B[("h3d","T",0)][0,0], T[:, jm-lnsh:jm, :]))
res["u_E_v3"] = bool(np.array_equal(B[("v3d","U",3)][0,0], U[:, :, im-lnsv-1:im-1].transpose(0, 2, 1)))
res["u_N_v3"] = bool(np.array_equal(B[("v3d","U",0)][0,0], U[:, jm-lnsv:jm, :]))

# 4-D species path: nspec = 2 synthetic field, same kernel; species s must land in buf[0, s].
Z4 = cp.asarray(np.random.default_rng(3).standard_normal((2, lm, jm, im)).astype(np.float32))
saved = dict(w.PARENT_BC_VARS)
w.PARENT_BC_VARS.clear(); w.PARENT_BC_VARS.update({"h4d": ["Z4"], "h2d": [], "h3d": [], "v2d": [], "v3d": []})
ctx4 = Ctx(cfg=cfg, grid=g, st={"Z4": Z4}, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
ctx4.bc = None
with ctx4.stream:
    w.run(ctx4, site="recomp_off", tboco=10800.0, recomp_tend=False)
ctx4.stream.synchronize()
w.PARENT_BC_VARS.clear(); w.PARENT_BC_VARS.update(saved)
Z = cp.asnumpy(Z4)
res["z4_species_N"] = all(bool(np.array_equal(cp.asnumpy(ctx4.bc[("h4d","Z4",0)])[0, s], Z[s, :, jm-lnsh:jm, :])) for s in range(2))

# Gates (no GPU work): each must raise.
def raises(fn):
    try:
        fn()
        return False
    except (ValueError, RuntimeError):
        return True
ctx5 = Ctx(cfg=cfg, grid=g, st=st, stream=None, dt=cfg.dt)
ctx5.bc = None
res["gate_unknown_site"] = raises(lambda: w.run(ctx5, site="nest", tboco=10800.0, recomp_tend=False))
ctx5.ntimestep = 5
res["gate_ntsd0_nonzero_step"] = raises(lambda: w.run(ctx5, site="ntsd0", tboco=10800.0, recomp_tend=True))
res["gate_tboco_missing"] = raises(lambda: w.run(ctx5, site="dfi_start", tboco=None, recomp_tend=True))
res["gate_recomp_mismatch"] = raises(lambda: w.run(ctx5, site="dfi_start", tboco=10800.0, recomp_tend=False))
res["gate_no_prior_buffer"] = raises(lambda: w.run(ctx5, site="dfi_start", tboco=10800.0, recomp_tend=True))

print(json.dumps({"nonfinite": bad, "keys": len(ctx.bc), "checks": res}, indent=1))
