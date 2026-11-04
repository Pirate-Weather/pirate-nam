"""phys-noah-hstep: HSTEP, the soil temperature step with the ROSR12 tridiagonal solve (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:3153-3201 (HSTEP) and
:3666-3725 (ROSR12). Kernel: namgpu/kernels/phys_noah_hstep.cu (phys_noah_hstep, one thread per
(i, j), one launch, all NSOIL layers looped inside the thread).

Soil arrays are float32 [NSOIL=4, JM, IM] (CONTRACT section 1, layer 1 at the top), C-contiguous.
The INOUT Fortran arguments RHSTS, AI, BI and CI are written back in place with the Fortran values.
No host sync, no per-step allocation, everything on the current stream.

Callers and producers (module_LS_NOAHLSM.F90:3764-3776, SHFLX):
  ICE == 1: HRTICE(RHSTS, AI, BI, CI) then HSTEP;  otherwise HRT(RHSTS, AI, BI, CI) then HSTEP.
  HSTEP returns STCF only. The copy-back STC = STCF (Fortran 3778-3780) belongs to SHFLX and is not
  done here, so HSTEP alone is not the soil update.

run(ctx): the RHSTS, AI, BI, CI coefficients are NOT produced here. The caller must pass them (the
HRT or HRTICE outputs) by keyword. A ctx call without them raises instead of solving against zeros.
dt has no default: the caller must pass the DT that SHFLX receives from NOAHLSM.
Without ctx, run() builds a full-size 954 x 835 random batch (finiteness check only; these random
coefficients are not HRT/HRTICE output).
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_hstep"
KERNEL = "phys_noah_hstep"
BLOCK = (32, 8, 1)
NSOIL = 4

_scratch: dict = {}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def _check_soil(name: str, arr, jm: int, im: int) -> None:
    if tuple(arr.shape) != (NSOIL, jm, im):
        raise ValueError(f"phys-noah-hstep: {name} shape {tuple(arr.shape)} != {(NSOIL, jm, im)}")
    if str(arr.dtype) != "float32":
        raise ValueError(f"phys-noah-hstep: {name} dtype {arr.dtype} != float32")
    if not arr.flags.c_contiguous:
        raise ValueError(f"phys-noah-hstep: {name} is not C-contiguous (IDX3 layout)")


def hstep(stcin, stcf, rhsts, ai, bi, ci, dt):
    """STCF = STCIN + ROSR12 solution; RHSTS, AI, BI, CI are updated in place as the Fortran HSTEP does.

    All arrays are float32 C-contiguous device arrays of shape [NSOIL, JM, IM]; dt is the time step (s).
    """
    if int(stcin.shape[0]) != NSOIL:
        raise ValueError(f"phys-noah-hstep: stcin has {int(stcin.shape[0])} layers, NSOIL is {NSOIL}")
    _, jm, im = (int(s) for s in stcin.shape)
    for name, arr in (("STCIN", stcin), ("STCF", stcf), ("RHSTS", rhsts),
                      ("AI", ai), ("BI", bi), ("CI", ci)):
        _check_soil(name, arr, jm, im)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"phys-noah-hstep: dt must be a positive finite number, got {dt}")
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (stcin, stcf, rhsts, ai, bi, ci, np.float32(dt), np.int32(im), np.int32(jm)))
    return stcf


def _stcf_scratch(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        _scratch[key] = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
    return _scratch[key]


def _missing_coefs(rhsts, ai, bi, ci):
    return [n for n, a in (("rhsts", rhsts), ("ai", ai), ("bi", bi), ("ci", ci)) if a is None]


def run(ctx=None, *, stcin=None, stcf=None, rhsts=None, ai=None, bi=None, ci=None, dt=None):
    """Done-rule entry: one launch. Returns the arrays it wrote (STCF, RHSTS, AI, BI, CI).

    ctx path: STCIN is ctx.st.STC; rhsts, ai, bi and ci (HRT or HRTICE outputs) must be passed.
    dt is required on every path.
    """
    import cupy as cp

    if dt is None:
        raise ValueError("phys-noah-hstep: dt must be passed explicitly (SHFLX passes the DT it "
                         "receives from NOAHLSM); there is no default")

    if ctx is not None and stcin is None:
        stcin = ctx.st.STC
        jm, im = int(stcin.shape[1]), int(stcin.shape[2])
        missing = _missing_coefs(rhsts, ai, bi, ci)
        if missing:
            raise ValueError(
                "phys-noah-hstep: coefficients not bound: " + ", ".join(missing)
                + " (producers are phys-noah-hrt or phys-noah-hrtice, called before HSTEP as in SHFLX)")
        if stcf is None:
            stcf = _stcf_scratch(jm, im)
    elif stcin is None:
        jm, im = 835, 954
        rng = np.random.default_rng(1)

        def rnd(lo, hi):
            return cp.asarray(rng.uniform(lo, hi, size=(NSOIL, jm, im)).astype(np.float32))

        stcin = rnd(250.0, 290.0)
        rhsts = rnd(-0.5, 0.5)
        ai = rnd(-0.05, 0.05)
        bi = rnd(0.0, 0.05)
        ci = rnd(-0.05, 0.05)
        stcf = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
    else:
        jm, im = int(stcin.shape[1]), int(stcin.shape[2])
        missing = _missing_coefs(rhsts, ai, bi, ci)
        if missing:
            raise ValueError("phys-noah-hstep: coefficients not bound: " + ", ".join(missing))
        if stcf is None:
            stcf = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
    hstep(stcin, stcf, rhsts, ai, bi, ci, dt)
    return {"STCF": stcf, "RHSTS": rhsts, "AI": ai, "BI": bi, "CI": ci}
