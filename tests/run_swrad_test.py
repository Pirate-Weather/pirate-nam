import time
import cupy as cp
from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_sw_swrad as m

t0 = time.time()
res = kernels.module(m.UNIT).compile()
print("compile ok", round(time.time() - t0, 2), "s", flush=True)
t1 = time.time()
r = m.selftest(seed=1)
cp.cuda.Stream.null.synchronize()
print("selftest day_columns", r["day_columns"], "bad", r["bad"], "secs", round(time.time() - t1, 1), flush=True)
free, total = cp.cuda.runtime.memGetInfo()
print("free GiB after", round(free / 2**30, 2), "of", round(total / 2**30, 2), flush=True)
