"""Check for phys-myj-prodq2 after the review fixes.

1. Refusal: run() with no MYJ_GM producer must raise (no zero-fill).
2. Full-size run with nonzero GM/GH so the TURBULENT branch runs. Every level K=1..LM-1 either
   takes the no-turbulence branch (EL set to EPSL) or the turbulent branch (EL left unchanged,
   unless the TKE floor clamps it to EPSL). Counting EL left unchanged is the witness that the
   turbulent arithmetic ran. Gate: more than half of the levels.
3. No NaN/Inf in Q2 or EL_PBL, and the lower boundary Q2(LM) equals
   max(B1**(2/3)*USTAR**2, EPSQ2(LM)) to float32 round-off.
"""
import sys, time
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cupy as cp
import numpy as np
from namgpu import config, grid as gridmod, state as statemod, kernels
from namgpu.driver import Ctx
from namgpu.physics.myj import phys_myj_prodq2 as w

cfg = config.load()
g = gridmod.build(cfg)
st = statemod.State(g).randomize(seed=1)
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
im, jm, lm = int(g.im), int(g.jm), int(g.lm)
shape3 = (lm, jm, im)

try:
    w.run(ctx)
    sys.exit("FAIL: run() accepted a state with no MYJ_GM producer")
except RuntimeError as e:
    print("refusal ok:", e)

rng = cp.random.default_rng(7)
st.fields["USTAR"] = rng.uniform(0.05, 1.0, (jm, im)).astype(cp.float32)
st.fields["EPSL"] = cp.full((lm,), 1.0, dtype=cp.float32)
st.fields["EPSQ2"] = cp.full((lm,), 0.02, dtype=cp.float32)
st.fields["Q2"] = rng.uniform(0.1, 1.0, shape3).astype(cp.float32)
st.fields["MYJ_GM"] = rng.uniform(0.001, 0.02, shape3).astype(cp.float32)
st.fields["MYJ_GH"] = rng.uniform(0.001, 0.02, shape3).astype(cp.float32)
st.fields["EL_PBL"] = rng.uniform(50.0, 500.0, shape3).astype(cp.float32)
el_before = st.fields["EL_PBL"].copy()
usq = cp.asnumpy(st.fields["USTAR"]).astype(np.float64) ** 2
epsq2_lm = float(cp.asnumpy(st.fields["EPSQ2"])[lm - 1])

kernels.module("phys_myj_prodq2").compile()
t0 = time.perf_counter()
with ctx.stream:
    out = w.run(ctx)
ctx.stream.synchronize()
dt = time.perf_counter() - t0

q2, el = out["Q2"], out["EL_PBL"]
nonfinite = [n for n, a in (("Q2", q2), ("EL_PBL", el)) if not bool(cp.isfinite(a).all())]
kk = slice(0, lm - 1)
turb_frac = float(cp.mean(el[kk] == el_before[kk]))
bnd_expect = np.maximum(11.87799326209552761 ** (2.0 / 3.0) * usq, epsq2_lm)
bnd_got = cp.asnumpy(q2[lm - 1]).astype(np.float64)
bnd_err = float(np.max(np.abs(bnd_got - bnd_expect) / np.maximum(bnd_expect, 1e-30)))
print("grid", im, jm, lm, "time_s", round(dt, 3), "nonfinite", nonfinite)
print("turbulent-branch fraction (EL unchanged, levels 1..LM-1):", round(turb_frac, 4))
print("lower boundary max rel err:", bnd_err)
bad = bool(nonfinite) or turb_frac <= 0.5 or bnd_err > 1e-5
sys.exit(1 if bad else 0)
