"""drv-psitbl: PSITBL surface-layer stability-function table (sea and land, z/L table).

Fortran: nmm/module_CONTROL.F90:2189-2300 (psitbl), called from CONSTS (module_CONTROL.F90:342),
NAM nam.v4.2.11, 83df4e13c1. Kernel: namgpu/kernels/drv_psitbl.cu (drv_psitbl_table).

Reads:  nothing from ctx.st (the table depends only on the fixed z/L grid).
Writes: ctx.tables["psitbl"], a dict with device arrays
          "psim1", "psih1", "psim2", "psih2" : float32 [kztm]   (sea and land stability functions)
          "scal" : float32 [8] = ztmin1, ztmax1, ztmin2, ztmax2, dzeta1, dzeta2, fh01, fh02
        and host integers "kztm" (10001, module_CONTROL.F90:34) and "kztm2" (kztm-2).
Launch: one launch, one thread per table point, on ctx.stream, no host sync. Built once; later
calls reuse the arrays and skip the launch (the table is constant for the whole run).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_psitbl"
KERNEL = "drv_psitbl_table"
KZTM = 10001            # module_CONTROL.F90:34 kztm (size of the stability-function table)
KZTM2 = KZTM - 2        # module_CONTROL.F90:43 kztm2 = kztm - 2 (set by psitbl)


def run(ctx):
    import cupy as cp

    if ctx.tables is None:
        ctx.tables = {}
    tbl = ctx.tables.get("psitbl")
    if tbl is None:
        tbl = {
            "psim1": cp.zeros(KZTM, dtype=np.float32),
            "psih1": cp.zeros(KZTM, dtype=np.float32),
            "psim2": cp.zeros(KZTM, dtype=np.float32),
            "psih2": cp.zeros(KZTM, dtype=np.float32),
            "scal": cp.zeros(8, dtype=np.float32),
            "kztm": KZTM,
            "kztm2": KZTM2,
        }
        ctx.tables["psitbl"] = tbl

        kern = kernels.get(UNIT, KERNEL)
        block = (kernels.BLOCK_COL[0], 1, 1)
        grid = ((KZTM + block[0] - 1) // block[0], 1, 1)
        scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
        with scope:
            kern(grid, block, (
                tbl["psim1"], tbl["psih1"], tbl["psim2"], tbl["psih2"], tbl["scal"],
                np.int32(KZTM),
            ))
