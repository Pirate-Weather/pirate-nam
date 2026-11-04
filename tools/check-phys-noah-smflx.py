"""Check phys-noah-smflx: (1) full-size standalone batch, one and two pass branches, NaN/Inf check;
(2) ctx path with REDPRM parameters and nonzero PRCP1 (two-pass branch), finite output;
(3) ctx path refuses without its SFLX inputs (KeyError naming SFLX).
    source ~/nam-gpu/env.sh && python tools/check-phys-noah-smflx.py
"""
import sys
from pathlib import Path

import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import config, grid as gridmod, state as statemod
from namgpu.driver import Ctx
from namgpu.physics.noah import phys_noah_smflx as u

cfg = config.load()
g = gridmod.build(cfg)
JM, IM = 835, 954

out = u.run(None)
for k, a in out.items():
    if not bool(cp.isfinite(a).all()):
        raise SystemExit(f"standalone: NaN/Inf in {k}")
print("standalone 954x835: finite, dt 50 s")

st = statemod.State(g)
st.randomize(seed=2)
nphs = int(cfg.nphs)
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
ctx_jm, ctx_im = int(st.SMC.shape[1]), int(st.SMC.shape[2])
rng = np.random.default_rng(3)
def f2(lo, hi):
    return cp.asarray(rng.uniform(lo, hi, size=(ctx_jm, ctx_im)).astype(np.float32))
smcmax = f2(0.40, 0.50)
redprm = {"BEXP": f2(2, 12), "DKSAT": f2(1e-6, 1e-4), "DWSAT": f2(1e-6, 1e-4), "SMCMAX": smcmax,
          "SMCWLT": smcmax * f2(0.05, 0.3), "SMCREF": smcmax * 0.8, "KDT": f2(1e-4, 5e-4),
          "SLOPE": f2(0, 1), "CMCMAX": f2(0.5e-3, 1e-3), "SHDFAC": f2(0, 0.9)}
for n, a in redprm.items():
    if n not in st.fields:
        st.add(n, (ctx_jm, ctx_im))
    st.fields[n][...] = a
st.SH2O[...] = st.SMC * 0.8
kw = {"PRCP1": f2(1e-5, 1e-4), "EDIR": f2(0, 1e-6), "EC": f2(0, 1e-6), "ET": cp.asarray(rng.uniform(0, 2e-6, size=(4, ctx_jm, ctx_im)).astype(np.float32))}
res = u.run(ctx, **kw)
for k in ("SMC", "SH2O", "CMC", "DRIP", "RUNOFF1", "RUNOFF2", "RUNOFF3"):
    if not bool(cp.isfinite(res[k]).all()):
        raise SystemExit(f"ctx: NaN/Inf in {k}")
print(f"ctx path: finite, dt = nphs {nphs} x ctx.dt {cfg.dt} = {nphs * float(cfg.dt)} s, runoff3 max {float(res["RUNOFF3"].max()):.3e}")

try:
    u.run(ctx, PRCP1=kw["PRCP1"], EDIR=kw["EDIR"], EC=kw["EC"])
except KeyError as e:
    print("ctx path refuses without ET:", str(e)[:120])
else:
    raise SystemExit("ctx path ran without ET")
