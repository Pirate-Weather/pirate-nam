"""Done-rule check for phys-rrtm-lw-rtrnmr: compile the kernel, run the wrapper once at full size."""
import time

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_lw_rtrnmr as w

mod = kernels.module("phys_rrtm_lw_rtrnmr")
mod.compile()
print("compile: ok", flush=True)
t0 = time.time()
res = w.run(None)
print("run(None) seconds:", round(time.time() - t0, 2), flush=True)
print("shape (im, jm, nlay):", res["shape"], flush=True)
print("finite per output:", res["finite"], flush=True)
assert all(res["finite"].values()), "non-finite output"
print("PASS", flush=True)
