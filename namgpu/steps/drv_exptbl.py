"""drv-exptbl: EXPTBL exponential look-up table, built once at init.

Ported from nmm/module_CONTROL.F90:1780-1820 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/drv_exptbl.cu (exptbl_build, one thread: the Fortran x recurrence).

Reads:   nothing (constants xmin=-30, xmax=30, kexm=10001 from the Fortran).
Writes:  ctx.tables['EXPF']          float32 [10001]   Fortran expf(1:kexm), index k-1
         ctx.tables['EXPTBL_SCAL']   float32 [4]       xmin, xmax, dex, rdex
         ctx.tables['EXPTBL_KEXM2']  int                kexm2 = 9999
Idempotent: the table is built on the first call and reused (ctx.tables is the once-only store).
Launch:  one kernel on ctx.stream, no host sync.
"""
from __future__ import annotations

import contextlib

from namgpu import kernels

UNIT = "drv_exptbl"
KERNEL = "exptbl_build"
KEXM = 10001


def run(ctx):
    import cupy as cp

    tables = ctx.tables
    if "EXPF" in tables:
        return
    expf = cp.empty((KEXM,), dtype=cp.float32)
    scal = cp.empty((4,), dtype=cp.float32)

    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern((1, 1, 1), (1, 1, 1), (expf, scal))

    tables["EXPF"] = expf
    tables["EXPTBL_SCAL"] = scal
    tables["EXPTBL_KEXM2"] = KEXM - 2
