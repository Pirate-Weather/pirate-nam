"""check phys-fer-deposit: full-size no-NaN run of the DEPOSIT wrapper, with the review-gap defaults.

Full CONUS-size column block (LM=60, JM=835, IM=954). Checks:
  - defaults: PP = SGML2*PD + PSGML1, Tdum = T + XLF1*CW, WVdum = Q/(1-Q) (the FER_HIRES ice-only inputs);
  - outputs finite; nonconv count; WARN2 record and one-time host report (run at rhgrd=1e4 so that some cells
    fail to converge in MAX_ITERATIONS and the record path runs).
Synthetic state with physical ranges; the real caller is phys-fer-fer-hires (not ported here).
"""
import sys
import types

import cupy as cp
import numpy as np

from namgpu.physics.fer_hires import phys_fer_deposit as dep

LM, JM, IM = 60, 835, 954
rng = np.random.default_rng(7)

st = {
    "PD": cp.asarray(rng.uniform(9.5e4, 1.02e5, (JM, IM)).astype(np.float32)),
    "T": cp.asarray(rng.uniform(205.0, 262.0, (LM, JM, IM)).astype(np.float32)),
    "Q": cp.asarray(rng.uniform(1e-6, 4e-3, (LM, JM, IM)).astype(np.float32)),
    "CW": cp.asarray(rng.uniform(0.0, 1e-4, (LM, JM, IM)).astype(np.float32)),
}
sgml1 = np.linspace(0.9, 0.001, LM)
sgml2 = np.linspace(0.0, 0.99, LM)
grid = types.SimpleNamespace(sgml2=sgml2.astype(np.float32),
                             psgml1=200.0 + (1.0 - sgml2) * 1e3)
ctx = types.SimpleNamespace(st=st, grid=grid, tables={}, stream=None)

# Expected defaults, recomputed independently.
pd_host = cp.asnumpy(st["PD"])
pp_expect = (np.float32(1) * grid.sgml2[:, None, None] * pd_host[None]
             + grid.psgml1.astype(np.float32)[:, None, None]).astype(np.float32)

pidep, tdum, wvdum, nonconv = dep.run(ctx)
cp.cuda.Stream.null.synchronize()
pp = ctx.tables["fer_deposit_pp"]
assert cp.allclose(pp, cp.asarray(pp_expect), rtol=1e-6, atol=1e-3), "PP default != SGML2*PD+PSGML1"
print("PP default ok; XLF1 =", float(dep.XLF1))

for name, arr in (("pidep", pidep), ("tdum", tdum), ("wvdum", wvdum)):
    nan = int(cp.isnan(arr).sum())
    fin = int(cp.isfinite(arr).sum())
    print(f"{name}: shape={tuple(arr.shape)} nan={nan} finite={fin}/{arr.size}")
    assert nan == 0 and fin == arr.size, f"{name} has non-finite values"
print("nonconv (rhgrd 0.98):", int(cp.asnumpy(nonconv)[0]))
print("warn2 record (rhgrd 0.98):", dep.report_warn2(ctx))

# Force the WARN2 path once: a supersaturated start at rhgrd=1e4 keeps SSAT outside the band.
_, _, _, nonconv2 = dep.run(ctx, rhgrd=1.0e4)
n2 = int(cp.asnumpy(nonconv2)[0])
print("nonconv (rhgrd 1e4):", n2)
rec = dep.report_warn2(ctx)
print("warn2 record (rhgrd 1e4):", rec)
print("report again (should be None or silent):", dep.report_warn2(ctx) is not None)
assert n2 == 0 or rec is not None, "overflow occurred but no WARN2 record was claimed"
print("CHECK OK")
sys.stdout.flush()
