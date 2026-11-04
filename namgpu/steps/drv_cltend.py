"""drv-cltend: CLTEND, temperature change by cloud processes spread over NPRECIP steps.

Fortran: nmm/module_SOLVER_GRID_COMP.F90:11597-11681 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/drv_cltend.cu (cltend, one thread per (i, j, l), grid_2d).

Reads/writes: T, Told, Tadj [LM,JM,IM] in ctx.st (Told and Tadj are state-table fields,
parm/nam_solver_state.txt). T is inout in every mode; Told is written by mode -1, Tadj by mode 0.
Scalars: IM, JM, LM from ctx.grid; RDTPH = 1/NPRECIP from ctx.cfg.nprecip (single precision).

SOLVER_RUN calls it three times per cloud step (CONTRACT section 2). The icltend mode is REQUIRED
and comes from the driver plan (namgpu/driver.py PHYSICS), which carries the call-site cadence:
  icltend=-1  key "cltend_pre"  before convection, IF(CALL_PRECIP .AND. NPRECIP>1)  (SOLVER_RUN 5705)
  icltend=0   key "cltend_post" after GSMDRIVE, inside IF(CALL_PRECIP), NPRECIP>1   (SOLVER_RUN 5925)
  icltend=1   key "cltend_inc"  every step, NPRECIP>1 only, after the microphysics block (6005)
Fortran never calls CLTEND when NPRECIP==1 (all three call sites carry NPRECIP>1), so this wrapper
returns without touching the state in that case. There is no default mode: a call without icltend
is refused, because the three modes are not interchangeable (mode 0 right after mode -1 would
set Tadj = 0).

tools/check_unit.py runs this unit through the driver plan, so its three calls come from the
three plan entries, one per mode.

Launch: one kernel per mode on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_cltend"
KERNEL = "cltend"


def run(ctx, icltend: int):
    if icltend not in (-1, 0, 1):
        raise ValueError(f"drv-cltend: icltend must be -1, 0 or 1, got {icltend}")
    nprecip = int(ctx.cfg.nprecip)
    if nprecip <= 0:
        raise RuntimeError(f"drv-cltend: nprecip must be > 0, got {nprecip}")
    if nprecip <= 1:
        # Fortran: cld_tend1, cld_tend2, cld_tend3 are all IF(NPRECIP>1); CLTEND is not called.
        return
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    rdtph = np.float32(np.float32(1.0) / np.float32(nprecip))

    T = st.T
    Told = st.Told
    Tadj = st.Tadj

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, (np.int32(im), np.int32(jm), np.int32(lm),
                           np.int32(icltend), rdtph, T, Told, Tadj))
