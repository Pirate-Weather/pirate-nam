"""phys-rad-radconst: RADCONST, solar declination and solar constant for the forecast date.

Ported from nmm/module_RADIATION.F90:1000-1042 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE radconst.
Kernel: namgpu/kernels/phys_rad_radconst.cu (radconst, one thread, one launch).

STATUS: DEAD IN NAM. A case-insensitive grep over the nam.v4.2.11 source finds radconst only at its
definition (module_RADIATION.F90:1000-1042); nothing calls it. The live NMMB solar path is
sol_update_nmmb (radiation_astronomy_nmmb.f, solc0 table). This unit is a faithful port kept on the
record. Its outputs have no consumer and must never feed RRTM; phys-rad-driver does not depend on it.
SOLCON here is 1370*eccfac and is not the RRTM solar constant (ctx.cfg.solcon, 1368.22).

In:      julian from julian_nam(ctx.start). ctx.tables['julian'] overrides it only when a clock unit
         sets it (none does in this port).
Out:     ctx.st['RADCONST'] float32[2]: [DECLIN (rad), SOLCON (W m-2)]. Scratch, allocated once by
         ctx.st.add and reused on every call.
Launch:  one kernel launch on ctx.stream, no host sync, no per-call allocation.

JULIAN follows TIME_MEASURE (module_RA_GFDL.F90:255-292, called from module_SOLVER_GRID_COMP.F90:5268
with NTIMESTEP_rad). JULDAY is the day of year with February 29 days when MOD(START_YEAR,4)==0 (not the
Gregorian century rule). JULIAN = JULDAY-(1-SUM), SUM from START_HOUR/MINUTE/SECOND. TIME_MEASURE has
no timestep term: NAM's JULIAN is fixed at the start time of day and does not advance with the forecast.
(NAM's February length is a SAVE'd variable set from START_YEAR; START_YEAR is fixed for a run, so
the persistence never changes a value within one forecast.)
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rad_radconst"
KERNEL = "radconst"
KEY = "RADCONST"
OUT_DECLIN = 0   # radians
OUT_SOLCON = 1   # W m-2
NAM_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def julian_nam(when):
    """NAM JULIAN (float32) for a start datetime, as TIME_MEASURE computes it (module_RA_GFDL.F90:255-292).

    12Z 2 January -> JULDAY 2, SUM 0.5, JULIAN 1.5.
    """
    days = list(NAM_MONTH_DAYS)
    if when.year % 4 == 0:           # MOD(START_YEAR,4)==0
        days[1] = 29
    julday = sum(days[:when.month - 1]) + when.day
    f32 = np.float32
    total = f32(when.hour) + (f32(when.minute) + f32(when.second) / f32(60.0)) / f32(60.0)
    ssum = total / f32(24.0)
    return f32(julday) - (f32(1.0) - ssum)


def _julian(ctx) -> float:
    if "julian" in ctx.tables:
        return float(ctx.tables["julian"])
    if ctx.start is None:
        raise RuntimeError(
            "phys-rad-radconst: no date. The clock unit must set ctx.tables['julian'] "
            "or ctx.start must be set.")
    return float(julian_nam(ctx.start))


def run(ctx, xtime: float = 0.0):
    out = ctx.st.add(KEY, (2,), "float32")
    julian = np.float32(_julian(ctx))
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k((1, 1, 1), (1, 1, 1), (np.float32(xtime), julian, out))
    return out
