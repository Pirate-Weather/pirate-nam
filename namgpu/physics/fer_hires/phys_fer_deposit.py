"""phys-fer-deposit: DEPOSIT, ice deposition at T < T_ICE (module_MP_FER_HIRES.F90:2231-2289).

Kernel: namgpu/kernels/phys_fer_deposit.cu, fer_deposit_eval, one thread per cell (grid_2d, level in
grid.z). The device function nam_deposit is what the FER_HIRES column unit (phys-fer-fer-hires) calls in
its ice branch; this wrapper runs it on every cell once (done-rule run and dependent-unit tests).
Pure device work: no host sync, no per-step allocation (scratch allocated on first call).

Inputs (ctx, or explicit arguments). The defaults are the real caller's inputs, module_MP_FER_HIRES.F90
ice-only branch (line ~1364-1378): PP = PHMID, Tdum = DUM1 = TK + XLF1*PIACW with PIACW = QW, and
WVdum = DUM2 = WV = Q/(1-Q):
    pp     float32 [LM, JM, IM]   PP = PHMID = SGML2(K)*PD(I,J) + PSGML1(K)   (module_MICROPHYSICS.F90:225, 253)
    tdum   float32 [LM, JM, IM]   Tdum = T + XLF1*CW, K; updated in place
    wvdum  float32 [LM, JM, IM]   WVdum = Q/(1-Q), water vapour mixing ratio; updated in place
    rhgrd  float                  RHgrd (FER_HIRES argument; 0.98 by default)
Ctx reads: st["PD"] [JM, IM], st["T"], st["Q"], st["CW"] (QW), grid.sgml2 and grid.psgml1 [LM].
Tables (ctx.tables, built by phys-fer-gpvs if absent):
    fer_gpvs_tbpvs float32 [NX=7501]; fer_gpvs_coef float32 [4] = {C1XPVS, C2XPVS, C1XPVS0, C2XPVS0}
Outputs:
    pidep   float32 [LM, JM, IM]   DEPOSIT return value (PIDEP)
    nonconv int32 [1]              number of cells that reached MAX_ITERATIONS in this call
    warn2   int32 [7]              first WARN2 overflow of the run (flag, I, J, L 1-based, TC, SSATinp, DWVinp
                                   as float bits); read by report_warn2 (host, not on the step path)
Returns (pidep, tdum, wvdum, nonconv).

Caller-side gate not in this unit: the Fortran calls DEPOSIT only where DUM2 > DUM (the ice-only branch).
This wrapper runs DEPOSIT on every cell; the FER_HIRES caller (phys-fer-fer-hires) owns that guard.
"""
from __future__ import annotations

import contextlib
import sys

import numpy as np

from namgpu import kernels
from namgpu.physics.fer_hires import phys_fer_gpvs

UNIT = "phys_fer_deposit"
KERNEL = "fer_deposit_eval"
RHGRD_DEFAULT = 0.98
# FER XLF1 = XLF/CP with FER's own XLF=3.3358e5 (fer_hires.cuh FER_XLF1), single precision division.
XLF1 = np.float32(np.float32(3.3358e5) / np.float32(1004.6))
WARN2_WORDS = 7


def _level_vec(tables, name, host, lm):
    """Device copy of a per-level host vector, shaped [LM, 1, 1] for broadcasting, cached in tables."""
    import cupy as cp

    arr = tables.get(name)
    if arr is None or int(arr.shape[0]) != lm:
        arr = cp.asarray(np.asarray(host, np.float32).reshape(lm, 1, 1))
        tables[name] = arr
    return arr


def _scratch(tables, name, shape):
    import cupy as cp

    arr = tables.get(name)
    if arr is None or tuple(arr.shape) != shape:
        arr = cp.empty(shape, dtype=cp.float32)
        tables[name] = arr
    return arr


def run(ctx, pp=None, tdum=None, wvdum=None, rhgrd: float = RHGRD_DEFAULT):
    import cupy as cp

    tables = ctx.tables
    if "fer_gpvs_tbpvs" not in tables:
        phys_fer_gpvs.run(ctx)
    tbpvs = tables["fer_gpvs_tbpvs"]
    coef = tables["fer_gpvs_coef"]
    st = ctx.st

    if pp is None:
        g = ctx.grid
        lm = int(len(g.sgml2))
        pd = st["PD"]                                         # [JM, IM], PDSL = PD(I,J)
        shape = (lm,) + tuple(pd.shape)
        pp = _scratch(tables, "fer_deposit_pp", shape)
        sgml2 = _level_vec(tables, "fer_deposit_sgml2", g.sgml2, lm)
        psgml1 = _level_vec(tables, "fer_deposit_psgml1", g.psgml1, lm)
        cp.multiply(sgml2, pd, out=pp)                        # SGML2(K)*PDSL
        cp.add(pp, psgml1, out=pp)                            # + PSGML1(K) = PHMID
    shape = tuple(pp.shape)

    if tdum is None:
        tdum = _scratch(tables, "fer_deposit_tdum", shape)
        cp.multiply(st["CW"], XLF1, out=tdum)                 # XLF1*PIACW, PIACW = QW
        cp.add(tdum, st["T"], out=tdum)                       # DUM1 = TK + XLF1*QW
    if wvdum is None:
        wvdum = _scratch(tables, "fer_deposit_wvdum", shape)
        q = st["Q"]
        cp.divide(q, np.float32(1.0) - q, out=wvdum)          # DUM2 = WV = Q/(1-Q)

    pidep = _scratch(tables, "fer_deposit_pidep", shape)
    nonconv = tables.get("fer_deposit_nonconv")
    if nonconv is None:
        nonconv = cp.zeros(1, dtype=cp.int32)
        tables["fer_deposit_nonconv"] = nonconv
    nonconv.fill(0)
    warn2 = tables.get("fer_deposit_warn2")
    if warn2 is None:
        warn2 = cp.zeros(WARN2_WORDS, dtype=cp.int32)
        tables["fer_deposit_warn2"] = warn2

    lm, jm, im = shape
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
          (pp, tdum, wvdum, pidep, tbpvs, coef, np.float32(rhgrd), nonconv, warn2,
           np.int32(im), np.int32(jm), np.int32(lm)))
    return pidep, tdum, wvdum, nonconv


def report_warn2(ctx):
    """Host report of the first WARN2 overflow of the run (Fortran: once per run, to stderr).

    Reads a few device words, so it is called off the step path (end of run or on demand). Prints the
    Fortran WARN2 line once per ctx. Returns the record dict if any cell overflowed in the run, else None.
    """
    import cupy as cp

    tables = ctx.tables
    warn2 = tables.get("fer_deposit_warn2")
    if warn2 is None:
        return None
    rec = cp.asnumpy(warn2)
    if int(rec[0]) == 0:
        return None
    tc, ssat, dwv = rec[4:7].view(np.float32).tolist()
    out = {"I": int(rec[1]), "J": int(rec[2]), "L": int(rec[3]),
           "TC": tc, "SSAT": ssat, "DWV": dwv}
    if not tables.get("fer_deposit_warn2_printed"):
        nonconv = tables.get("fer_deposit_nonconv")
        last = int(cp.asnumpy(nonconv)[0]) if nonconv is not None else -1
        print(f"WARN2: Too many iterations in function DEPOSIT;  I,J,L,TC,SSAT,DWV= "
              f"{out['I']} {out['J']} {out['L']} {tc} {ssat} {dwv}  (cells not converged in last call: {last})",
              file=sys.stderr)
        tables["fer_deposit_warn2_printed"] = True
    return out
