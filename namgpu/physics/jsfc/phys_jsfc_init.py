"""phys-jsfc-init: JSFC_INIT, the Janjic surface-layer init (module_SF_JSFC.F90:1014-1251, nam.v4.2.11).

Fortran: JSFC_INIT(LOWLYR, USTAR, Z0, SEAMASK, XICE, IVGTYP, RESTART, ...). Kernel:
namgpu/kernels/phys_jsfc_init.cu (phys_jsfc_init_tables, phys_jsfc_init_ustar).

Reads:  ctx.st["SEAMASK"], ctx.st["XICE"], ctx.st["IVGTYP"] are NOT read on any write path (the
        vegetation-Z0 block is commented out in the source), so they are not touched.
Writes:
  ctx.tables["jsfc"], a dict with device arrays (the JSFC module's PSIM1/PSIH1/PSIM2/PSIH2,
  module_SF_JSFC.F90:69, not the CONSTS psitbl arrays of module_CONTROL):
      "psim1", "psih1", "psim2", "psih2" : float32 [KZTM]
      "scal" : float32 [8] = ztmin1, ztmax1, ztmin2, ztmax2, dzeta1, dzeta2, fh01, fh02
  and host integers "kztm" (10001) and "kztm2" (kztm-2).
  ctx.st["USTAR"] (float32 [JM, IM]) = 0.1 on i0 < IM-1 when restart is False (:1120).
  Z0 and LOWLYR are not written (see the kernel header).

Launches: two launches on ctx.stream, no host sync, no allocation after the first call. The table
kernel is 2 threads, one per table, each a serial 10001-step loop, because the Fortran accumulates
ZETA by repeated addition; it runs once per process (init), not on the step path.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_jsfc_init"
KZTM = 10001            # module_SF_JSFC.F90:65 KZTM (size of PSIM1/PSIH1/PSIM2/PSIH2)
KZTM2 = KZTM - 2        # module_SF_JSFC.F90:65 KZTM2 = KZTM - 2


def run(ctx, restart: bool = False):
    """Build the JSFC stability tables once per process and set USTAR on a cold start.

    Returns ctx.tables["jsfc"].
    """
    import cupy as cp

    if ctx.tables is None:
        ctx.tables = {}
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    tbl = ctx.tables.get("jsfc")
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
        ctx.tables["jsfc"] = tbl
        k_tables = kernels.get(UNIT, "phys_jsfc_init_tables")
        with scope:
            k_tables((1, 1, 1), (2, 1, 1), (
                tbl["psim1"], tbl["psih1"], tbl["psim2"], tbl["psih2"], tbl["scal"],
            ))

    ustar = ctx.st["USTAR"]
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    itf = min(im, im - 1)                 # ITF = MIN0(ITE, IDE-1) with ITE = IDE = IM
    k_ustar = kernels.get(UNIT, "phys_jsfc_init_ustar")
    with scope:
        k_ustar(kernels.grid_2d(im, jm, 1), kernels.BLOCK_2D,
                (ustar, np.int32(im), np.int32(jm), np.int32(itf), np.int32(bool(restart))))
    return tbl
