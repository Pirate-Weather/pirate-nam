"""Check for phys-rad-radconst (review fixes): the NAM JULIAN rule, no timestep drift, leap rule,
and the kernel output against the same Fortran formula in numpy float32.

    source ~/nam-gpu/env.sh
    python tools/check-phys-rad-radconst.py
"""
from __future__ import annotations

import datetime
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402

from namgpu.physics.driver import phys_rad_radconst as rc  # noqa: E402


class FakeSt:
    def __init__(self):
        self.fields = {}

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=dtype)
        return self.fields[name]


def ctx_at(start, ntimestep, st):
    return types.SimpleNamespace(tables={}, st=st, stream=None, start=start, ntimestep=ntimestep, dt=60.0)


def fortran_radconst(julian):
    f = np.float32
    degrad = f(3.1415926) / f(180.0)
    dpd = f(360.0) / f(365.0)
    sinob = np.sin(f(23.5) * degrad, dtype=f)
    sxlong = dpd * (f(julian) - f(80.0)) if julian >= 80.0 else dpd * (f(julian) + f(285.0))
    declin = np.arcsin(sinob * np.sin(sxlong * degrad, dtype=f), dtype=f)
    rjul = f(julian) * f(360.0) / f(365.0) * degrad
    eccfac = (f(1.000110) + f(0.034221) * np.cos(rjul, dtype=f) + f(0.001280) * np.sin(rjul, dtype=f)
              + f(0.000719) * np.cos(f(2.0) * rjul, dtype=f) + f(0.000077) * np.sin(f(2.0) * rjul, dtype=f))
    return declin, f(1370.0) * eccfac


def main():
    fails = []
    # 1. JULIAN follows TIME_MEASURE: 12Z 2 January -> 1.5 (module_RA_GFDL.F90 comment)
    j = float(rc.julian_nam(datetime.datetime(2023, 1, 2, 12)))
    if abs(j - 1.5) > 1e-6:
        fails.append(f"jan2 12Z julian {j} != 1.5")
    # 2. leap rule MOD(START_YEAR,4)==0: 2100 is NAM-leap (Feb 29), not Gregorian
    j2100 = float(rc.julian_nam(datetime.datetime(2100, 3, 1)))
    if abs(j2100 - 60.0) > 1e-6:
        fails.append(f"2100-03-01 00Z julian {j2100} != 60.0 (NAM leap: JULDAY 61 - 1)")
    j2024 = float(rc.julian_nam(datetime.datetime(2024, 3, 1)))
    if abs(j2024 - 60.0) > 1e-6:
        fails.append(f"2024-03-01 00Z julian {j2024} != 60.0")
    # 3. no timestep term: JULIAN is the same at step 0 and step 5000, and the kernel output follows it
    start = datetime.datetime(2023, 3, 31, 12)
    st = FakeSt()
    outs = []
    for n in (0, 5000):
        out = rc.run(ctx_at(start, n, st))
        outs.append(cp.asnumpy(out).copy())
    if not np.array_equal(outs[0], outs[1]):
        fails.append(f"output drifts with ntimestep: {outs[0]} vs {outs[1]}")
    if st.fields.get(rc.KEY) is None or len(st.fields) != 1:
        fails.append("scratch not allocated once in ctx.st")
    ref_declin, ref_solcon = fortran_radconst(89.5)
    got = outs[0]
    if not (np.isfinite(got).all() and abs(got[0] - ref_declin) < 1e-6 and abs(got[1] - ref_solcon) < 1e-3):
        fails.append(f"kernel {got} vs fortran formula ({ref_declin}, {ref_solcon})")
    print(f"julian 2023-03-31 12Z = {float(rc.julian_nam(start))}  declin={got[0]:.6f} rad  solcon={got[1]:.3f}")
    print(f"julian 2100-03-01 = {j2100}  2024-03-01 = {j2024}  jan2 12Z = {j}")
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
