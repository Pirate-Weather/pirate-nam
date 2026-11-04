"""phys-fer-condense: CONDENSE, cloud water condensation (module_MP_FER_HIRES.F90:2160-2225).

Kernel: namgpu/kernels/phys_fer_condense.cu. nam_condense() is a __device__ function there.
fer_condense_eval runs once over the full [LM, JM, IM] grid (grid_2d, one thread per cell) and binds the
production CONDENSE call exactly as module_MP_FER_HIRES.F90:1456-1461 does:
    IF (TC>=T_ICE .AND. (QW>EPSQ .OR. WV>QSWgrd)) PCOND = CONDENSE(PP, QW-PIACWI,
        TK+XLS1*PIDEP+XLF1*PIACWI, WV-PIDEP, RHgrd, I, J, L)      (else PCOND = 0)

Inputs (ctx), all [LM, JM, IM] float32 unless noted:
  TK      = st["T"]                       default
  WV      = st["Q"]                       default, water vapour mixing ratio
  QW      = st["CW"]                      default, cloud water
  PP      = P_col = PHMID(K) = SGML2(K)*PD(I,J) + PSGML1(K)  (module_MICROPHYSICS.F90:253,
            P_PHY=PHMID, module_MP_FER_HIRES.F90:378). Default: the scratch field "PMID" (st.add),
            built from ctx.grid.sgml2/psgml1 and st["PD"] on the device, no host sync.
  PIDEP   = DEPOSIT result (PIDEP), PIACWI = PIACWI, QSWGRD = saturation mixing ratio over water
            (grid), all written by phys-fer-col-processes. Default: ctx.tables["fer_col_processes"].
            If that unit has not run and the arrays are not passed, run() raises. It never
            substitutes zeros.
  RHgrd   = 0.98 (configfile rhgrd, the FER_HIRES argument).
  tbpvs0, C1XPVS0, C2XPVS0 = ctx.tables["fer_gpvs_tbpvs0"] and ctx.tables["fer_gpvs_coef"][2:4] from
            phys-fer-gpvs (GPVS_hr), read by the kernel. The host does not recompute them.
Outputs:
  ctx.tables["fer_condense_out"]  float32 [LM, JM, IM] PCOND (0 where the TC guard is false).
  ctx.tables["fer_condense_flag"] uint8   [LM, JM, IM] 1 where CONDENSE did not converge
                                  (the Fortran WARN3 condition). Read it off the step path.
Scratch is allocated once and reused. No host sync inside run().
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_condense"
KERNEL = "fer_condense_eval"
RHGRD = 0.98
KEY_OUT = "fer_condense_out"
KEY_FLAG = "fer_condense_flag"
KEY_MID = "fer_condense_mid_coef"


def _phmid(ctx, TK):
    """PHMID(K) = SGML2(K)*PD(I,J) + PSGML1(K) (module_MICROPHYSICS.F90:253, PDSL = PD(I,J)).

    Returns the "PMID" scratch field. The level coefficients are uploaded once and kept in ctx.tables.
    """
    import cupy as cp

    st = ctx.st
    mid = ctx.tables.get(KEY_MID)
    if mid is None:
        g = ctx.grid
        mid = (cp.asarray(np.asarray(g.sgml2, np.float32).reshape(-1, 1, 1)),
               cp.asarray(np.asarray(g.psgml1, np.float32).reshape(-1, 1, 1)))
        ctx.tables[KEY_MID] = mid
    sgml2, psgml1 = mid
    pmid = st.add("PMID", tuple(TK.shape))
    cp.multiply(sgml2, st["PD"][None, :, :], out=pmid)
    cp.add(pmid, psgml1, out=pmid)
    return pmid


def run(ctx, TK=None, WV=None, QW=None, PP=None, PIDEP=None, PIACWI=None, QSWGRD=None,
        RHgrd=RHGRD):
    import cupy as cp

    from namgpu.physics.fer_hires import phys_fer_gpvs

    st = ctx.st
    if TK is None:
        TK = st["T"]
    if WV is None:
        WV = st["Q"]
    if QW is None:
        QW = st["CW"]

    col = ctx.tables.get("fer_col_processes")
    if PIDEP is None or PIACWI is None or QSWGRD is None:
        if col is None:
            raise RuntimeError(
                "phys-fer-condense: PIDEP, PIACWI and QSWGRD come from phys-fer-col-processes, which has "
                "not run (no ctx.tables['fer_col_processes']). Run it first or pass the arrays.")
        PIDEP = col["PIDEP"] if PIDEP is None else PIDEP
        PIACWI = col["PIACWI"] if PIACWI is None else PIACWI
        QSWGRD = col["QSWGRD"] if QSWGRD is None else QSWGRD

    if PP is None:
        PP = _phmid(ctx, TK)

    tables = ctx.tables
    if "fer_gpvs_tbpvs0" not in tables or "fer_gpvs_coef" not in tables:
        phys_fer_gpvs.run(ctx)
    tbpvs0 = tables["fer_gpvs_tbpvs0"]
    coef = tables["fer_gpvs_coef"]

    LM, JM, IM = (int(s) for s in TK.shape)
    out = tables.get(KEY_OUT)
    if out is None or out.shape != TK.shape:
        out = cp.empty(TK.shape, dtype=cp.float32)
        tables[KEY_OUT] = out
    flag = tables.get(KEY_FLAG)
    if flag is None or flag.shape != TK.shape:
        flag = cp.empty(TK.shape, dtype=cp.uint8)
        tables[KEY_FLAG] = flag

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_2d(IM, JM, LM), kernels.BLOCK_2D,
          (PP, QW, TK, WV, PIDEP, PIACWI, QSWGRD, np.float32(RHgrd), tbpvs0, coef, out, flag,
           np.int32(IM), np.int32(JM), np.int32(LM)))
    return out
