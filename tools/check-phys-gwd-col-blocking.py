"""Done-rule check for phys-gwd-col-blocking: compile, one full-size run, NaN/Inf and witness checks.

Pass rules:
  - GUARD[0] == 0 (no point reads an unset BNV2LM entry; the Fortran value there is undefined).
  - NaN is allowed only at level LM of BNV2 and RI_N (the Fortran never defines that level).
    Every other float output is finite.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cupy as cp
from namgpu import kernels
from namgpu.physics.gwd import phys_gwd_col_blocking as w

kernels.module(w.UNIT).compile()
t0 = time.perf_counter()
out = w.run()
cp.cuda.Stream.null.synchronize()
dt = time.perf_counter() - t0

lm = out["DB"].shape[0]
nan_ok = {"BNV2", "RI_N"}
bad = []
for name, a in out.items():
    if a.dtype.kind != "f":
        continue
    if name in nan_ok:
        top = a[lm - 1]
        rest = a[: lm - 1]
        if not bool(cp.isfinite(rest).all()) or not bool(cp.isnan(top).all()):
            bad.append(name)
    elif not bool(cp.isfinite(a).all()):
        bad.append(name)

guard = int(out["GUARD"][0])
print("run seconds (incl. first compile/alloc):", round(dt, 3))
print("unexpected NaN/Inf outputs:", bad if bad else "none")
print("GUARD[0] (points reading unset BNV2LM, must be 0):", guard)
print("KBPS, KMPS:", int(out["KBPS"][0]), int(out["KMPS"][0]))
print("columns with IDXZB>0:", int((out["IDXZB"] > 0).sum()), "of", out["IDXZB"].size)
print("max |DB|:", float(cp.abs(out["DB"]).max()), " max |TAUP| (zeroed here):", float(cp.abs(out["TAUP"]).max()))
print("KREF range:", int(out["KREF"].min()), int(out["KREF"].max()))
print("OA range:", float(out["OA"].min()), float(out["OA"].max()))
sys.exit(1 if (bad or guard != 0) else 0)
