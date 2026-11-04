"""Done-rule check for phys-rrtm-aer-init (CONTRACT.md section 6, parts 1 and 2).

1. compiles the kernel file (NVRTC via CuPy);
2. runs phys-rrtm-rad-initialize (its physparam flags are the input) and then this wrapper ONCE;
   asserts the scalar outputs match the 12 km NAM configuration (iaer 11, iaer_mdl 0) and no
   NaN/Inf in every float table it writes.

This unit has no state arrays; its outputs are the set_spectrum tables and the module scalars, so the
full-size State of check_unit.py does not apply. Run: python tools/check-phys-rrtm-aer-init.py
"""
import sys

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cupy as cp
import numpy as np

from namgpu import kernels
from namgpu.driver import Ctx
from namgpu.physics.rrtm import phys_rrtm_aer_init as w
from namgpu.physics.rrtm import phys_rrtm_rad_initialize as radinit

kernels.module("phys_rrtm_aer_init").compile()
print("compile ok", flush=True)

ctx = Ctx(cfg=None, grid=None, st=None, stream=cp.cuda.Stream(non_blocking=True), dt=0.0)
with ctx.stream:
    radinit.run(ctx)
    tbl = w.run(ctx)
ctx.stream.synchronize()
flags = w.check_status(ctx)
print("flags", flags, flush=True)
assert flags["nswbnd"] == 14 and flags["nlwbnd"] == 16 and flags["nswlwbd"] == 30, flags
assert flags["do_spectrum"] is True and flags["do_volcaer"] is False, flags

for name in ("solfwv", "eirfwv", "s0intv"):
    a = tbl[name].get()
    fin = bool(np.isfinite(a).all())
    print(name, a.shape, "finite", fin, "min", float(a.min()), "max", float(a.max()), flush=True)
    assert fin, f"NaN/Inf in {name}"
sol = tbl["solfwv"].get()
assert (sol > 0).all(), "solfwv has unfilled wavenumbers (sum(nwvns0) must equal NWVTOT)"
assert np.float32(sol[0]) == np.float32(1.60000e-6) and np.float32(sol[-1]) == np.float32(7.46294e-5)
eir = tbl["eirfwv"].get()
assert (eir > 0).all()
print("solfwv[0]", float(sol[0]), "solfwv[-1]", float(sol[-1]), "eirfwv[0]", float(eir[0]),
      "eirfwv[-1]", float(eir[-1]), flush=True)
print("PASS", flush=True)
