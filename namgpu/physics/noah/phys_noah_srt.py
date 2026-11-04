"""phys-noah-srt: SRT, soil water RHS and tridiagonal coefficients (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:4603-4852 (SRT) and the
WDFCND it calls. Kernel: namgpu/kernels/phys_noah_srt.cu (phys_noah_srt, one thread per (i, j),
one launch, all NSOIL layers looped inside the thread). Adapted from ArWen third_party/arwen/core/
kernels/noah.cu (Apache-2.0); the infiltration cap follows the NAM Fortran (MIN(INFMAX, PX)).

Soil arrays are float32 [NSOIL=4, JM, IM] (CONTRACT section 1, layer 1 at the top). Per-column
scalars are float32 [JM, IM]. ZSOIL is float32 [NSOIL]. No host sync, no per-step allocation,
everything on the current stream.

run(ctx): the done-rule entry point. With ctx it reads ctx.st.STC for the soil temperature
grid shape; the SRT inputs that are not state-table fields yet come from scratch arrays. Without
ctx it builds a full-size random 954 x 835 batch (plausible soil ranges, nonzero rain on part of
the domain so both branches of the infiltration block run).
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_srt"
KERNEL = "phys_noah_srt"
BLOCK = (32, 8, 1)
NSOIL = 4
ZSOIL_DEFAULT = (-0.1, -0.4, -1.0, -2.0)

_scratch: dict = {}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def srt(rhstt, et, sh2o, sh2oa, sice, zsoil, edir, pcpdrp, dwsat, dksat, smcmax, bexp,
        smcwlt, slope, kdt, frzx, ai, bi, ci, runoff1, runoff2, dt):
    """RHSTT, AI, BI, CI ([NSOIL, JM, IM]) and RUNOFF1, RUNOFF2 ([JM, IM]) are written in place.

    Soil arrays are float32 device arrays [NSOIL, JM, IM]; per-column scalars are [JM, IM]; zsoil is
    [NSOIL]; dt is the time step in seconds.
    """
    nsoil_in, jm, im = (int(s) for s in sh2o.shape)
    if nsoil_in != NSOIL:
        raise ValueError(f"phys-noah-srt: state has NSOIL={nsoil_in}, kernel is built for NSOIL={NSOIL}")
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (rhstt, et, sh2o, sh2oa, sice, zsoil, edir, pcpdrp, dwsat, dksat, smcmax, bexp,
       smcwlt, slope, kdt, frzx, ai, bi, ci, runoff1, runoff2,
       np.float32(dt), np.int32(im), np.int32(jm)))
    return rhstt


def _scratch_arrays(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        soil = (NSOIL, jm, im)
        flat = (jm, im)
        _scratch[key] = {
            "RHSTT": cp.zeros(soil, dtype=cp.float32),
            "AI": cp.zeros(soil, dtype=cp.float32),
            "BI": cp.zeros(soil, dtype=cp.float32),
            "CI": cp.zeros(soil, dtype=cp.float32),
            "RUNOFF1": cp.zeros(flat, dtype=cp.float32),
            "RUNOFF2": cp.zeros(flat, dtype=cp.float32),
        }
    return _scratch[key]


def run(ctx=None, *, dt=25.0):
    """Done-rule entry: one launch on a full-size 954 x 835 column set. Returns the arrays it wrote."""
    import cupy as cp

    jm, im = 835, 954
    if ctx is not None:
        # Use the live grid shape when a state is present.
        jm, im = int(ctx.st.STC.shape[1]), int(ctx.st.STC.shape[2])
    rng = np.random.default_rng(1)

    def soil(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, size=(NSOIL, jm, im)).astype(np.float32))

    def flat(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, size=(jm, im)).astype(np.float32))

    smcmax = flat(0.35, 0.50)
    smcwlt = flat(0.02, 0.10)
    sh2oa = soil(0.10, 0.30)
    sh2o = soil(0.10, 0.30)
    sice = soil(0.0, 0.02)
    et = soil(-1.0e-5, 1.0e-5)
    # Rain on about half the columns so the infiltration and runoff branch runs.
    pcpdrp = flat(-1.0e-4, 2.0e-3) * cp.asarray((rng.uniform(size=(jm, im)) > 0.5).astype(np.float32))
    edir = flat(-1.0e-5, 1.0e-5)
    dwsat = flat(1.0e-6, 1.0e-4)
    dksat = flat(1.0e-7, 1.0e-5)
    bexp = flat(4.0, 10.0)
    slope = flat(0.1, 1.0)
    kdt = flat(1.0, 3.0)
    frzx = flat(0.1, 0.3)
    zsoil = cp.asarray(np.asarray(ZSOIL_DEFAULT, dtype=np.float32))

    out = _scratch_arrays(jm, im)
    srt(out["RHSTT"], et, sh2o, sh2oa, sice, zsoil, edir, pcpdrp, dwsat, dksat, smcmax, bexp,
        smcwlt, slope, kdt, frzx, out["AI"], out["BI"], out["CI"], out["RUNOFF1"],
        out["RUNOFF2"], dt)
    return out
