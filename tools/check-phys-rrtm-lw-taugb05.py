"""Compile phys-rrtm-lw-taugb05 and run its wrapper once on a full-size random state (954 x 835 x 60).

Pass = kernel compiles with NVRTC, the wrapper launches once, and taug_b05 / fracs_b05 contain no NaN or Inf.
"""
import time

import cupy as cp
import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_lw_taugb05 as W


class Grid:
    im, jm, lm = 954, 835, 60


class Fields:
    grid = Grid()

    def __init__(self):
        self.fields = {}

    def __getitem__(self, k):
        return self.fields[k]

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=dtype)
        return self.fields[name]


class Ctx:
    stream = None
    tables = None


IM, JM, LM = Grid.im, Grid.jm, Grid.lm
SHP = (LM, JM, IM)

kernels.module(W.UNIT).compile()
print("compile ok:", W.UNIT)

cp.random.seed(0)
st = Fields()


def frand(shape, lo, hi):
    return (lo + (hi - lo) * cp.random.random(shape, dtype=cp.float32)).astype(cp.float32)


def irand(lo, hi, shape):
    return cp.random.randint(lo, hi, size=shape, dtype=cp.int32)


lt = irand(20, 41, (JM, IM))
lay = cp.arange(LM, dtype=cp.int32)[:, None, None]
lower = lay < lt[None]
st.fields["colamt"] = frand((3,) + SHP, 0.5, 1.5)
st.fields["rfrate"] = frand((4,) + SHP, 0.5, 1.5)
st.fields["wx_ccl4"] = frand(SHP, 0.0, 0.1)
for n in ("fac00", "fac01", "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac", "minorfrac"):
    st.fields[n] = frand(SHP, 0.0, 1.0)
st.fields["jp"] = cp.where(lower, irand(1, 12, SHP), irand(13, 51, SHP)).astype(cp.int32)
st.fields["jt"] = irand(1, 6, SHP)
st.fields["jt1"] = irand(1, 5, SHP)
st.fields["indself"] = irand(1, 9, SHP)
st.fields["indfor"] = irand(1, 4, SHP)
st.fields["indminor"] = irand(1, 19, SHP)
st.fields["laytrop"] = lt
cp.cuda.Device().synchronize()

ctx = Ctx()
ctx.st = st
ctx.tables = {}
t0 = time.time()
taug, fracs = W.run(ctx)
cp.cuda.Device().synchronize()
print(f"wrapper run: {time.time() - t0:.2f} s (includes first-call NVRTC load)")
print("taug_b05", taug.shape, taug.dtype, "fracs_b05", fracs.shape, fracs.dtype)

bad = {}
for name, a in (("taug_b05", taug), ("fracs_b05", fracs)):
    n_bad = 0
    for l0 in range(0, LM, 10):
        n_bad += int(cp.count_nonzero(~cp.isfinite(a[:, l0:l0 + 10])))
    bad[name] = n_bad
    print(f"{name}: nonfinite values = {n_bad}, min = {float(a.min()):.4g}, max = {float(a.max()):.4g}")
assert all(v == 0 for v in bad.values()), bad
print("PASS: compile ok, one full-size launch, no NaN/Inf")
