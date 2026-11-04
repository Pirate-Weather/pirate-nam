"""phys-noah-devap: DEVAP, direct bare-soil evaporation (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:2514-2554. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel file. The point
function is namgpu/kernels/phys_noah_devap.cuh, shared with EVAPO (phys_noah_evapo.cu), so the
DEVAP math exists once. The ArWen-derived starting point is Apache-2.0: namgpu/third_party/arwen/LICENSE.
Kernel: namgpu/kernels/phys_noah_devap.cu (phys_noah_devap, one thread per (i, j), one launch).

EDIR = FX * (1 - SHDFAC) * ETP1, with FX = clamp(SRATIO**FXEXP, 0, 1) and
SRATIO = (SH2O - SMCDRY) / (SMCMAX - SMCDRY) when SRATIO > 0, else FX = 0. The unit boundary applies
the EVAPO gate (ETP1 > 0 and SHDFAC < 1, module_LS_NOAHLSM.F90:2602-2603): outside it EDIR = 0.

Range: the caller's B1 limits, 1-based ITS_B1..ITE_B1 = 2..IM-1 and JTS_B1..JTE_B1 = 2..JM-1
(module_TURBULENCE.F90 call of NOAHLSM, same as phys-noah-driver). Points outside are not written.

ctx inputs, each bound from the state field its producer publishes (no placeholders):
    SH2O [NSOIL, JM, IM]   layer 1 = SH2O[0]         (phys-noah-sflx / phys-noah-init)
    ETP1 [JM, IM]          potential evaporation     (phys-noah-penman publishes it as ETP1)
    SHDFAC, SMCMAX, SMCDRY, FXEXP [JM, IM]           (phys-noah-redprm writes these into ctx.st)
A missing producer is a RuntimeError that names it. No constant or random stand-in is used on the
ctx path. EDIR is written into ctx.st.EDIR (created once). Without ctx, run() builds a full-size
954 x 835 synthetic batch for the done-rule check only.
No host sync, no per-step allocation, everything on the current stream.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_devap"
KERNEL = "phys_noah_devap"
BLOCK = (32, 8, 1)
NSOIL = 4

# 1-based B1 limits of the NOAHLSM call (module_TURBULENCE.F90), for an IM x JM domain.
def b1_limits(im: int, jm: int):
    return 2, im - 1, 2, jm - 1          # its, ite, jts, jte

# state field -> producer, for the ctx path.
REQUIRED = {
    "ETP1": "phys-noah-penman (publishes ETP1)",
    "SHDFAC": "phys-noah-redprm / phys-noah-driver",
    "SMCMAX": "phys-noah-redprm",
    "SMCDRY": "phys-noah-redprm",
    "FXEXP": "phys-noah-redprm",
}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def devap(etp1, sh2o, shdfac, smcmax, smcdry, fxexp, edir, *, im: int | None = None, jm: int | None = None):
    """EDIR[j, i] = FX * (1 - SHDFAC) * ETP1 inside the B1 range and the EVAPO gate. float32 [JM, IM] on device.

    sh2o is the layer-1 slice of the soil field: SH2O[0] of an [NSOIL, JM, IM] array (contiguous).
    """
    if im is None or jm is None:
        jm, im = int(etp1.shape[0]), int(etp1.shape[1])
    its, ite, jts, jte = b1_limits(im, jm)
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (etp1, sh2o, shdfac, smcmax, smcdry, fxexp, edir,
       np.int32(im), np.int32(jm), np.int32(its), np.int32(ite), np.int32(jts), np.int32(jte)))
    return edir


def _require(st, names):
    missing = [n for n in names if n not in st.fields]
    if missing:
        detail = "; ".join(f"{n} from {REQUIRED[n]}" for n in missing)
        raise RuntimeError(f"phys-noah-devap: state fields not produced yet: {detail}")


def run(ctx=None):
    """Done-rule entry. With ctx: bind ctx.st by the producer names and write EDIR. Without ctx: a
    full-size 954 x 835 synthetic batch (done-rule check only). Returns {"EDIR": array}."""
    import cupy as cp

    if ctx is not None:
        st = ctx.st
        jm, im = int(ctx.grid.jm), int(ctx.grid.im)
        _require(st, tuple(REQUIRED))
        sh2o = st.SH2O[0]
        edir = st.add("EDIR", (jm, im))
        devap(st.ETP1, sh2o, st.SHDFAC, st.SMCMAX, st.SMCDRY, st.FXEXP, edir, im=im, jm=jm)
        return {"EDIR": edir}

    jm, im = 835, 954
    rng = np.random.default_rng(1)
    sh2o = cp.asarray(rng.uniform(0.02, 0.44, size=(jm, im)).astype(np.float32))
    etp1 = cp.asarray(rng.uniform(0.0, 5.0e-4, size=(jm, im)).astype(np.float32))
    shdfac = cp.asarray(rng.uniform(0.0, 1.0, size=(jm, im)).astype(np.float32))
    smcmax = cp.asarray(rng.uniform(0.35, 0.55, size=(jm, im)).astype(np.float32))
    smcdry = cp.asarray(rng.uniform(0.01, 0.05, size=(jm, im)).astype(np.float32))
    fxexp = cp.asarray(rng.uniform(0.5, 2.0, size=(jm, im)).astype(np.float32))
    edir = cp.zeros((jm, im), dtype=cp.float32)
    devap(etp1, sh2o, shdfac, smcmax, smcdry, fxexp, edir, im=im, jm=jm)
    return {"EDIR": edir}
