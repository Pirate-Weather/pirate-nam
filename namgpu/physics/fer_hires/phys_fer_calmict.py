"""phys-fer-calmict: CALMICT, Ferrier radar reflectivity at every (i, j, l) point (module_DIAGNOSE.F90:1589-1798).

Kernel: namgpu/kernels/phys_fer_calmict.cu, entry fer_calmict_eval (one thread per point, grid_2d).
The per-point body is the __device__ function nam_calmict(), which the diagnostics kernel
(MAX_FIELDS_HR, REFDMAX, module_DIAGNOSE.F90:1990-2016) calls once per level pair.

Inputs (CONTRACT names, float32, C order):
  P      [PMID]       [LM, JM, IM] pressure (Pa)
  T      [T]          [LM, JM, IM] temperature (K)
  Q      [Q]          [LM, JM, IM] specific humidity (kg/kg)
  C      [CW]         [LM, JM, IM] total condensate (kg/kg), C1D
  FI     [F_ICE]      [LM, JM, IM] ice fraction
  FR     [F_RAIN]     [LM, JM, IM] rain fraction
  FS     [F_RIMEF]    [LM, JM, IM] rime factor
  CUREFL [CUREFL]    [JM, IM] convective reflectivity (dBZ linear Z, the CUREFL argument)
ctx.tables (from phys-fer-init, not from here):
  fer_massi   float32 [951]    MASSI(50:1000), index m - 50
  fer_gpvs_tbpvs   float32 [7501]   TBPVS for FPVS (phys-fer-gpvs; built there if absent)
  fer_scalars dict: C1XPVS, C2XPVS, RQR_DRMIN, RQR_DRMAX, CN0R0, CN0R_DMRMIN, CN0R_DMRMAX
Output: DBZ1 float32 [LM, JM, IM] in ctx.tables["fer_calmict_dbz"] (allocated once, reused).
Launch: one kernel on ctx.stream (or the null stream), no host sync inside the wrapper.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_calmict"
KERNEL = "fer_calmict_eval"
OUT_KEY = "fer_calmict_dbz"


def run(ctx, P, T, Q, C, FI, FR, FS, CUREFL):
    import cupy as cp

    from namgpu.physics.fer_hires import phys_fer_gpvs

    shape = tuple(T.shape)
    lm, jm, im = shape
    if "fer_gpvs_tbpvs" not in ctx.tables:
        phys_fer_gpvs.run(ctx)
    out = ctx.tables.get(OUT_KEY)
    if out is None or out.shape != shape:
        out = cp.empty(shape, dtype=cp.float32)
        ctx.tables[OUT_KEY] = out
    assert CUREFL.shape == (jm, im)
    sc = ctx.tables["fer_scalars"]
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
          (P, T, Q, C, FI, FR, FS, CUREFL,
           ctx.tables["fer_massi"], ctx.tables["fer_gpvs_tbpvs"],
           np.float32(sc["C1XPVS"]), np.float32(sc["C2XPVS"]),
           np.float32(sc["RQR_DRMIN"]), np.float32(sc["RQR_DRMAX"]), np.float32(sc["CN0R0"]),
           np.float32(sc["CN0R_DMRMIN"]), np.float32(sc["CN0R_DMRMAX"]),
           out, np.int32(im), np.int32(jm), np.int32(lm)))
    return out
