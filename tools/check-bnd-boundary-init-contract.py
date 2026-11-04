# bnd-boundary-init review fixes: the ctx.bc keys and shapes the consumer reads, the value slot
# against the source edges, the shape-checked buffer reuse, and one bnd-bocoh-blend pass on them.
# Full-size 954 x 835 x 60 random State, single process, one run each.
import sys
from pathlib import Path
import numpy as np
import cupy as cp

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
ROOT = Path(_ROOT)
sys.path.insert(0, str(ROOT))
from namgpu import config, grid as gridmod, state as statemod
from namgpu.driver import Ctx
from namgpu.boundary import bnd_boundary_init as init, bnd_bocoh_blend as blend

cfg = config.load()
g = gridmod.build(cfg)
st = statemod.State(g).randomize(seed=1)
im, jm, lm = int(g.im), int(g.jm), int(g.lm)
lnsh = int(cfg.lnsh)
stream = cp.cuda.Stream(non_blocking=True)
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt, bc={})
ctx.bc = {}
SIDE = ("S", "N", "W", "E")


def run_init():
    with stream:
        init.run(ctx)
    stream.synchronize()


run_init()
fails = []
for side in SIDE:
    for name in ("PD", "T", "Q", "CW"):
        key = f"{side}_{name}"
        if key not in ctx.bc:
            fails.append(f"missing {key}")
            continue
        a = ctx.bc[key]
        if name == "PD":
            want = (2, lnsh, im) if side in ("S", "N") else (2, jm, lnsh)
        else:
            want = (2, lm, lnsh, im) if side in ("S", "N") else (2, lm, jm, lnsh)
        if tuple(a.shape) != want:
            fails.append(f"{key} shape {tuple(a.shape)} want {want}")
        if float(cp.abs(a[1]).max()) != 0.0:
            fails.append(f"{key} tendency slot not zero")

# value slot against the Fortran source slices (0-based): S rows 0..lnsh-1, N rows jm-lnsh..jm-1,
# W cols 0..lnsh-1, E cols im-lnsh..im-1
PD = cp.asnumpy(st["PD"]).reshape(jm, im)
T = cp.asnumpy(st["T"]).reshape(lm, jm, im)


def cmp(key, ref):
    got = cp.asnumpy(ctx.bc[key][0])
    if not np.array_equal(got, ref):
        fails.append(f"{key} value slot differs from source")


cmp("S_PD", PD[0:lnsh, :])
cmp("N_PD", PD[jm - lnsh:jm, :])
cmp("W_PD", PD[:, 0:lnsh])
cmp("E_PD", PD[:, im - lnsh:im])
cmp("S_T", T[:, 0:lnsh, :])
cmp("N_T", T[:, jm - lnsh:jm, :])
cmp("W_T", T[:, :, 0:lnsh])
cmp("E_T", T[:, :, im - lnsh:im])

# stale-buffer defect: a changed lnsh must reallocate, never reuse the wider buffer
cfg.lnsh = 3
run_init()
if tuple(ctx.bc["S_PD"].shape) != (2, 3, im):
    fails.append(f"after lnsh=3, S_PD shape {tuple(ctx.bc['S_PD'].shape)} want (2, 3, {im})")
cfg.lnsh = lnsh
run_init()

# consumer: bnd-bocoh-blend must run on these keys with no KeyError or shape error and stay finite.
# bnd-bocoh-blend also reads S/N/W/E_Q2, which bnd-boundary-init does not produce (the Fortran
# blends only the hardwired T, Q, CW). For this check only, Q2 keys are filled from the current Q2
# strips so the blend's other inputs run; this is a test stub, not the product path.
Q2 = cp.asnumpy(st["Q2"]).reshape(lm, jm, im)
for side in SIDE:
    src = {"S": Q2[:, 0:lnsh, :], "N": Q2[:, jm - lnsh:jm, :],
           "W": Q2[:, :, 0:lnsh], "E": Q2[:, :, im - lnsh:im]}[side]
    ctx.bc[f"{side}_Q2"] = cp.asarray(np.stack([src, src]).astype(np.float32))
if g.pdsg1 is None:
    fails.append("grid.pdsg1 unset; blend not run (grid.set_vcoord missing)")
else:
    with stream:
        blend.run(ctx)
    stream.synchronize()
    bad = [n for n, a in st.fields.items()
           if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    if bad:
        fails.append(f"NaN/Inf after blend in {bad[:10]}")

print("keys", len(ctx.bc), "fails", fails if fails else "none")
