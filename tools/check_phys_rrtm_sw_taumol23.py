"""Done-rule check for phys-rrtm-sw-taumol23: NVRTC compile, one full-size 954 x 835 x 60 pass, no NaN/Inf.

Usage (node-2, on a short pool lease): source ~/nam-gpu/env.sh; python3 tools/check_phys_rrtm_sw_taumol23.py
"""
import time

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_sw_taumol23 as unit

t0 = time.time()
kernels.get(unit.UNIT, unit.KERNEL)
print("compile ok", round(time.time() - t0, 2), "s", flush=True)
t1 = time.time()
bad = unit.selftest()
print("full-size pass", round(time.time() - t1, 2), "s", bad, flush=True)
