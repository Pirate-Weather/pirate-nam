"""Done-rule check for drv-h-to-v-tend (CONTRACT.md section 6, parts 1 and 2), both call sites.

Full 954 x 835 x 60 domain, only the arrays this unit touches are allocated. Runs the wrapper for
branch "nphs" (DELTA_T = DT*NPHS) and branch "bmj" (DELTA_T = DT*NPRECIP), checks every written
array for NaN/Inf, checks one interior point against the Fortran formula for each branch, and
checks the driver plan order and the bmj DUDT_PHY/DVDT_PHY alias.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402

from namgpu import config, driver, grid as gridmod, state as statemod  # noqa: E402
from namgpu.driver import Ctx  # noqa: E402
from namgpu.physics.driver import phys_bmj_driver  # noqa: E402
from namgpu.steps import drv_h_to_v_tend as w  # noqa: E402

cfg = config.load()
g = gridmod.build(cfg)
st = statemod.State(g, names=["U", "V"])
im, jm, lm = int(g.im), int(g.jm), int(g.lm)
st.add("DUDT", (lm, jm, im))
st.add("DVDT", (lm, jm, im))
rng = cp.random.default_rng(1)
for n in ("U", "V", "DUDT", "DVDT"):
    st[n][...] = 60.0 * rng.random(st[n].shape, dtype=cp.float32) - 30.0
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

# plan order: nphs call in the turbulence block, bmj call after phys-bmj-driver and before fer
plan = driver.PHYSICS
i_nphs = plan.index(("drv-h-to-v-tend", "nphs"))
i_bmj = plan.index(("drv-h-to-v-tend", "nprecip", {"branch": "bmj"}))
i_glue = plan.index(("phys-bmj-driver", "nprecip"))
i_fer = plan.index(("phys-fer-driver", "nprecip"))
assert i_nphs < i_glue < i_bmj < i_fer, (i_nphs, i_glue, i_bmj, i_fer)
print("plan order ok:", i_nphs, i_glue, i_bmj, i_fer, flush=True)

# bmj glue alias: CUCNVC's DUDT_PHY/DVDT_PHY are the same arrays as DUDT/DVDT
phys_bmj_driver._scratch(ctx)
assert st["DUDT_PHY"] is st["DUDT"] and st["DVDT_PHY"] is st["DVDT"], "alias broken"
print("bmj alias ok", flush=True)

dt = np.float32(cfg.dt)
bad = []
for branch, nsteps in (("nphs", int(cfg.nphs)), ("bmj", int(cfg.nprecip))):
    u0 = st["U"].copy()
    t0 = time.perf_counter()
    with ctx.stream:
        w.run(ctx, branch=branch)
    ctx.stream.synchronize()
    print(f"{branch}: run ok {time.perf_counter() - t0:.3f} s  NSTEPS={nsteps} DT={float(dt)}",
          flush=True)
    dtt = np.float32(dt * np.float32(nsteps))
    i0, j0, l0 = 5, 7, 3   # zero-based interior point
    d = st["DUDT"]
    du = [float(x) for x in (d[l0, j0, i0], d[l0, j0, i0 + 1], d[l0, j0 + 1, i0], d[l0, j0 + 1, i0 + 1])]
    expect = float(u0[l0, j0, i0]) + 0.25 * sum(du) * float(dtt)
    got = float(st["U"][l0, j0, i0])
    print(f"  U point: expect {expect:.6f} got {got:.6f}", flush=True)
    if abs(got - expect) > 1e-3 * max(1.0, abs(expect)):
        bad.append(f"{branch}: formula mismatch")
    if not bool(cp.isfinite(st["U"]).all()):
        bad.append(f"{branch}: NaN/Inf in U")

bad += [f"NaN/Inf in {n}" for n in ("U", "V") if not bool(cp.isfinite(st[n]).all())]
print("result:", bad if bad else "no NaN/Inf, formula ok", flush=True)
sys.exit(1 if bad else 0)
