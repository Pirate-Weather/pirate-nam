"""One full-size run of phys-fer-col-setup on a random plausible column state: compile, launch, check no NaN/Inf.

Allocates only the fields this unit touches at full size (954 x 835 x 60), not the whole 10.6 GiB state,
because the shared 5090 holds another tenant. Run from the repository root.

Also checks the three gaps of the review: TK/TC reach the outputs, the LSFC guard counter stays 0, and
the rain and snow branches give the Fortran values on injected carry (branch probe, 4 cases).
"""
import sys
import time
from types import SimpleNamespace

import cupy as cp
import numpy as np

from namgpu.state import State
from namgpu.physics.fer_hires import phys_fer_col_setup as unit

grid = SimpleNamespace(im=954, jm=835, lm=60)
st = State(grid, names={"T"})
ctx = SimpleNamespace(st=st, grid=grid, stream=None, tables={}, cfg=None)

rng = np.random.default_rng(7)
T = st["T"]
T[...] = cp.asarray(rng.uniform(215.0, 305.0, size=(60, 835, 954)).astype(np.float32))
lsfc = st.add(unit.LSFC_NAME, (835, 954), "int32")
lsfc[...] = cp.asarray(rng.integers(40, 61, size=(835, 954)).astype(np.int32))

t0 = time.time()
out = unit.run(ctx)
cp.cuda.Stream.null.synchronize()
print(f"unit ran in {time.time() - t0:.2f} s (includes NVRTC compile)", flush=True)

fail = []
bad = []
for name, arr in out.items():
    if name == unit.BAD_NAME:
        continue
    if not bool(cp.isfinite(arr).all()):
        bad.append(name)
print("arrays written:", len(out), "non-finite:", bad, flush=True)
if bad:
    fail.append("non-finite")

lsfc_bad = int(out[unit.BAD_NAME][0])
print("FER_LSFC_BAD:", lsfc_bad, flush=True)
if lsfc_bad != 0:
    fail.append("lsfc out of range")

# Column levels 1..LSFC: TK is T, TC is T - 273.15. Levels above LSFC: defined defaults.
levels = cp.arange(60, dtype=cp.int32)[:, None, None]
mask = levels < lsfc[None]
tk_err = float(cp.abs(out["TK_col"] - T)[mask].max())
tc_err = float(cp.abs(out["TC_col"] - (T - 273.15))[mask].max())
print("TK_col vs T max err (column):", tk_err, " TC_col max err (column):", tc_err, flush=True)
if tk_err > 0.0 or tc_err > 1e-3:
    fail.append("TK/TC")
above = ~mask
dbz_above_ok = bool((out["DBZ_col"][above] == -20.0).all())
tk_above_ok = bool((out["TK_col"][above] == 0.0).all() and (out["CLEAR_col"][above] == 0.0).all())
print("levels above LSFC defaults ok:", dbz_above_ok and tk_above_ok, flush=True)
if not (dbz_above_ok and tk_above_ok):
    fail.append("defaults above LSFC")

carry = out[unit.CARRY_NAME]
print("carry arain/asnow max:", float(carry[0].max()), float(carry[1].max()),
      "indexs0c unique:", np.unique(cp.asnumpy(carry[5])).tolist(),
      "rhox0c:", float(carry[8].min()), flush=True)
dbz = out["DBZ_col"]
print("DBZ_col min/max:", float(dbz.min()), float(dbz.max()), flush=True)

# Branch probe: rain and snow branches with injected carry, one level. Columns: ARAIN, ASNOW,
# VRAIN1, VSNOW1, TK -> CLEAR, ARAIN, ASNOW, VRAIN1, VSNOW1, TC. CLIMIT = 1e-11.
cases = np.array([
    [0.5,   0.0,  6.0, 1.0, 280.0],   # rain above: not clear, vrain1 -> 0, arain kept, vsnow1 kept
    [0.0,   0.3,  6.0, 1.5, 250.0],   # snow above: not clear, vsnow1 -> 0, vrain1 kept
    [0.0,   0.0,  6.0, 1.5, 290.0],   # nothing above: clear, carries cleared, speeds kept
    [1e-12, 5e-11, 6.0, 1.5, 260.0],  # both below/above CLIMIT: arain cleared, snow branch taken
], dtype=np.float32)
expect = np.array([
    [0.0, 0.5,  0.0, 0.0, 1.0, 6.85],
    [0.0, 0.0,  0.3, 6.0, 0.0, -23.15],
    [1.0, 0.0,  0.0, 6.0, 1.5, 16.85],
    [0.0, 0.0,  5e-11, 6.0, 0.0, -13.15],
], dtype=np.float32)
got = cp.asnumpy(unit.run_branch_probe(cases))
print("branch probe got:\n", got, flush=True)
if not np.allclose(got, expect, rtol=1e-5, atol=1e-9):
    fail.append("branch probe")
    print("branch probe expected:\n", expect, flush=True)

if fail:
    print("FAIL:", fail, flush=True)
    sys.exit(1)
print("PASS", flush=True)
