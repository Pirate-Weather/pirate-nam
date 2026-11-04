"""Full-size check for init-static-missing after the review fix (no NaN, valid ranges, valid cells untouched)."""
import sys
import time
from types import SimpleNamespace

import numpy as np
import cupy as cp

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from namgpu.state import State  # noqa: E402
from namgpu.init import init_static_missing as M  # noqa: E402

IM, JM = 954, 835
rng = np.random.default_rng(7)
grid = SimpleNamespace(im=IM, jm=JM, lm=60)
st = State(grid, names={"IVGTYP", "ISLTYP", "ALBASE", "MXSNAL"})

ivg = rng.integers(1, 21, size=(JM, IM)).astype(np.int32)
isl = rng.integers(1, 10, size=(JM, IM)).astype(np.int32)
alb = rng.uniform(0.05, 0.8, size=(JM, IM)).astype(np.float32)
mxs = rng.uniform(0.6, 0.85, size=(JM, IM)).astype(np.float32)
# inject the failure modes: zeros (NAM reader default), NaN, GRIB-missing sentinel, bad categories
for arr, val in ((alb, 0.0), (alb, np.nan), (alb, -1.0e4), (mxs, 0.0), (mxs, np.nan)):
    idx = rng.choice(JM * IM, size=1000, replace=False)
    arr.reshape(-1)[idx] = val
ivg.reshape(-1)[rng.choice(JM * IM, 500, replace=False)] = 0
ivg.reshape(-1)[rng.choice(JM * IM, 500, replace=False)] = 99
isl.reshape(-1)[rng.choice(JM * IM, 500, replace=False)] = 0
st.IVGTYP[...] = cp.asarray(ivg)
st.ISLTYP[...] = cp.asarray(isl)
st.ALBASE[...] = cp.asarray(alb)
st.MXSNAL[...] = cp.asarray(mxs)

ctx = SimpleNamespace(st=st, grid=grid, stream=cp.cuda.Stream.null, tables={})
t0 = time.time()
M.run(ctx)
cp.cuda.Stream.null.synchronize()
dt = time.time() - t0

a = cp.asnumpy(st.ALBASE)
m = cp.asnumpy(st.MXSNAL)
iv = cp.asnumpy(st.IVGTYP)
isv = cp.asnumpy(st.ISLTYP)
tab = ctx.tables["noah_static"]
ok_alb = bool(np.all(np.isfinite(a)) and np.all((a > 0) & (a <= 1)))
ok_mxs = bool(np.all(np.isfinite(m)) and np.all((m > 0) & (m <= 1)))
ok_iv = bool(np.all((iv >= 1) & (iv <= tab["lucats"])))
ok_isl = bool(np.all((isv >= 1) & (isv <= tab["slcats"])))
valid_alb = (alb > 0) & (alb <= 1)
valid_mxs = (mxs > 0) & (mxs <= 1)
keep_alb = bool(np.array_equal(a[valid_alb], alb[valid_alb]))
keep_mxs = bool(np.array_equal(m[valid_mxs], mxs[valid_mxs]))
print(f"time {dt:.2f} s  grid {IM}x{JM}  lucats={tab['lucats']} slcats={tab['slcats']} bare={tab['bare']}")
print(f"ALBASE finite and in (0,1]: {ok_alb}; valid cells unchanged: {keep_alb}")
print(f"MXSNAL finite and in (0,1]: {ok_mxs}; valid cells unchanged: {keep_mxs}")
print(f"IVGTYP in 1..lucats: {ok_iv}; ISLTYP in 1..slcats: {ok_isl}")
print("PASS" if all([ok_alb, ok_mxs, ok_iv, ok_isl, keep_alb, keep_mxs]) else "FAIL")
