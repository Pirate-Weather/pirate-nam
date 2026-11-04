"""drv-rdtemp: RDTEMP, the radiative temperature change applied every step.

Ported from phys/module_RA_GFDL.F90:116-233 (RDTEMP), with ZENITH (1662-1805) and CAL_MON_DAY
(7823-7856) inlined in the kernel (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/drv_rdtemp.cu (rdtemp_apply, one thread per column, level loop inside).

Reads : GLAT, GLON [JM,IM] (radians), CZMEAN [JM,IM], RSWTT, RLWTT [LM,JM,IM] from ctx.st;
        T [LM,JM,IM] inout; NTSD = ctx.ntimestep (the driver's step counter, as NTIMESTEP), DT = ctx.dt.
Writes: T in place (every level on the ITS_B1..ITE_B1 x JTS_B1..JTE_B1 interior, ITS_B1=2, ITE_B1=IM-1);
        CZEN [JM,IM] in ctx.st on the tile ITS..ITE x JTS..JTE from the loop-limit table (ZENITH's
        output for this call; cells outside the tile keep their value, as in the Fortran).
Calendar: JULDAY, JULYR, the start hour (IHRST = START_HOUR) and the sticky leap flag (MONTH(2)=29
        of ZENITH/CAL_MON_DAY) come from ctx.start (the initial time, UTC) and are derived once per
        run, the same way NMMB derives JULDAY in module_RA_GFDL.F90:273-285, and cached in ctx.tables
        (no per-step host work).
Launch: one kernel on ctx.stream, grid_cols(IM, JM), no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.steps import drv_loop_limits as dll

UNIT = "drv_rdtemp"
KERNEL = "rdtemp_apply"
MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _calendar(ctx):
    """(JULDAY, JULYR, IHRST, LEAP_SEEN) for this run; computed once and cached.

    LEAP_SEEN is the sticky MONTH(2)=29 flag of ZENITH and CAL_MON_DAY. Fortran sets it on the
    first leap-year call and never clears it. A run has one JULYR (the start year), so the flag is
    the leap test of that year: the same sequence the Fortran process sees.
    """
    cal = ctx.tables.get("RDTEMP_CAL")
    if cal is not None:
        return cal
    s = ctx.start
    if s is None:
        raise RuntimeError("drv-rdtemp: ctx.start (the initial time) is not set")
    month = list(MONTH)
    leap = s.year % 4 == 0
    if leap:
        month[1] = 29
    julday = sum(month[: s.month - 1]) + s.day
    cal = (int(julday), int(s.year), int(s.hour), int(leap))
    ctx.tables["RDTEMP_CAL"] = cal
    return cal


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    julday, julyr, ihrst, leap_seen = _calendar(ctx)

    lim = dll.table(ctx)
    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
             (np.int32(ctx.ntimestep), np.float32(ctx.dt),
              np.int32(julday), np.int32(julyr), np.int32(ihrst), np.int32(leap_seen),
              st.GLAT, st.GLON, st.CZEN, st.CZMEAN, st.T, st.RSWTT, st.RLWTT,
              np.int32(im), np.int32(jm), np.int32(lm), lim))
