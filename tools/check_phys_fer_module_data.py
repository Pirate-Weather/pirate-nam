"""One-shot check for unit phys-fer-module-data: compile, run the wrapper once at full size, assert the constants
match the host float32/int evaluation, the warn state holds the Fortran initial values, and no NaN/Inf appears."""
import sys
import time
from types import SimpleNamespace

import numpy as np

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cupy as cp  # noqa: E402

from namgpu.physics.fer_hires import phys_fer_module_data as U  # noqa: E402

IM, JM, LM = 954, 835, 60

kernels_mod = __import__("namgpu.kernels", fromlist=["module"])
mod = kernels_mod.module(U.UNIT)
for name in (U.KERNEL, U.KERNEL_WARN):
    mod.get_function(name)
print("compile ok: fer_module_data_publish, fer_warn_state_init")

rng = np.random.default_rng(0)
shape = (LM, JM, IM)
st = SimpleNamespace(
    F_ICE=cp.asarray(rng.random(shape, dtype=np.float32)),
    F_RAIN=cp.asarray(rng.random(shape, dtype=np.float32)),
    F_RIMEF=cp.asarray(rng.random(shape, dtype=np.float32)),
)
ctx = SimpleNamespace(cfg=SimpleNamespace(nprecip=2, dt=25.0), st=st, stream=None, tables={}, restart=False)

t0 = time.time()
out = U.run(ctx)
cp.cuda.Stream.null.synchronize()
print(f"run wall {time.time() - t0:.3f} s  real={len(U.REAL_NAMES)} int={len(U.INT_NAMES)}")

bad_nan = [k for k in (KEY for KEY in (U.KEY, U.KEY_INT, U.KEY_WARN))
           if not np.all(np.isfinite(cp.asnumpy(ctx.tables[k]).astype(np.float64)))]
print("NaN/Inf tables:", bad_nan if bad_nan else "none")
assert not bad_nan
for k in ("F_ICE", "F_RAIN", "F_RIMEF"):
    assert np.all(np.isfinite(cp.asnumpy(getattr(st, k)))), k

mismatch = U.verify(ctx)
print("constant mismatches:", mismatch if mismatch else "none")
assert not mismatch
warn_bad = U.verify_warn_state_initial(ctx)
print("warn state initial:", "ok" if not warn_bad else warn_bad)
assert not warn_bad

# A second call must not reset the live warn state.
ctx.tables[U.KEY_WARN][0] = 1.0
U.run(ctx)
cp.cuda.Stream.null.synchronize()
assert float(cp.asnumpy(ctx.tables[U.KEY_WARN])[0]) == 1.0, "warn state was reset by a second run"
print("second run keeps live warn state: ok")
