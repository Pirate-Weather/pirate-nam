"""phys-noah-hrtice: HRTICE, sea-ice soil thermal RHS and tri-diagonal coefficients (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:3038-3150. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_noah_hrtice.cu (phys_noah_hrtice, one thread per (i, j), one launch).
Called by HRT in the sea-ice case (module_LS_NOAHLSM.F90:3764, ICE == 1).

Mask: ICE [JM, IM] int32. Only ICE == 1 cells get the sea-ice formulas; other cells are written 0.
ZSOIL is the sea-ice vector -3.*KZ/NSOIL from SFLX (module_LS_NOAHLSM.F90:1824-1826), built in the
kernel, so there is no per-run ZSOIL input.

Per-cell SFLX intermediates (the producers are SFLX, module_LS_NOAHLSM.F90:3363-3371, and PENMAN
and TDFCND before it). They are required inputs, never placeholders:
  SFCTMP, TH2, FDOWN, EMISSI, T24, RCH, RR, BETA, EPSCA, DF1 (final, after the VEGTYP override and
  the EXP(SBETA*SHDFAC) factor). YY and ZZ1 are computed in the kernel from these, as SFLX does.

Outputs RHSTS, AI, BI, CI are [NSOIL=4, JM, IM] float32 (layer 1 at index 0). STC is [NSOIL, JM, IM].
No host sync, no per-step allocation, everything on the current stream.

run(ctx): with ctx, the caller supplies ctx.sflx, a dict with the SFLX intermediates above plus
ice, and ctx.st.STC is read. If ctx.sflx is missing, run() raises: the SFLX unit (phys-noah-sflx,
still copy-arwen in UNITS.json) is the producer of those intermediates, and binding placeholders
would prove nothing. Without ctx, run() builds a full-size 954 x 835 random batch with the same
SFLX-consistent intermediates (T24 = SFCTMP**4, as SFLX computes it) and returns the outputs.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_hrtice"
KERNEL = "phys_noah_hrtice"
BLOCK = (32, 8, 1)
NSOIL = 4

SFLX_INPUTS = ("SFCTMP", "TH2", "FDOWN", "EMISSI", "T24", "RCH", "RR", "BETA", "EPSCA", "DF1")

_scratch: dict = {}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def hrtice(stc, ice, sf, rhsts, ai, bi, ci):
    """RHSTS, AI, BI, CI from STC [NSOIL, JM, IM] for ICE == 1 cells. sf: SFLX_INPUTS dict, float32 [JM, IM]."""
    jm, im = int(stc.shape[1]), int(stc.shape[2])
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (rhsts, stc, ice, sf["SFCTMP"], sf["TH2"], sf["FDOWN"], sf["EMISSI"], sf["T24"],
       sf["RCH"], sf["RR"], sf["BETA"], sf["EPSCA"], sf["DF1"], ai, bi, ci,
       np.int32(im), np.int32(jm)))
    return rhsts, ai, bi, ci


def _outputs(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        _scratch[key] = dict(
            rhsts=cp.zeros((NSOIL, jm, im), dtype=cp.float32),
            ai=cp.zeros((NSOIL, jm, im), dtype=cp.float32),
            bi=cp.zeros((NSOIL, jm, im), dtype=cp.float32),
            ci=cp.zeros((NSOIL, jm, im), dtype=cp.float32),
        )
    return _scratch[key]


def _check_sflx(sf: dict, jm: int, im: int) -> None:
    missing = [k for k in SFLX_INPUTS if k not in sf]
    if missing:
        raise ValueError(
            "phys-noah-hrtice: SFLX intermediates not bound: " + ", ".join(missing)
            + " (producer is phys-noah-sflx, not ported yet)")
    for k in SFLX_INPUTS:
        if tuple(sf[k].shape) != (jm, im):
            raise ValueError(f"phys-noah-hrtice: {k} shape {tuple(sf[k].shape)} != {(jm, im)}")


def run(ctx=None, *, stc=None, ice=None, sflx=None, rhsts=None, ai=None, bi=None, ci=None):
    """Done-rule entry: one launch. Returns the arrays it wrote (RHSTS, AI, BI, CI)."""
    import cupy as cp

    if stc is None and ctx is not None:
        jm, im = int(ctx.grid.jm), int(ctx.grid.im)
        stc = ctx.st.STC
        ice = ctx.sflx["ICE"]
        sflx = ctx.sflx
        _check_sflx(sflx, jm, im)
    elif stc is None:
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        stc = cp.asarray(rng.uniform(250.0, 271.0, size=(NSOIL, jm, im)).astype(np.float32))
        ice = cp.asarray((rng.uniform(size=(jm, im)) < 0.3).astype(np.int32))
        sfctmp = rng.uniform(260.0, 300.0, size=(jm, im)).astype(np.float32)
        sflx = {
            "SFCTMP": cp.asarray(sfctmp),
            "TH2": cp.asarray(rng.uniform(260.0, 300.0, size=(jm, im)).astype(np.float32)),
            "FDOWN": cp.asarray(rng.uniform(200.0, 400.0, size=(jm, im)).astype(np.float32)),
            "EMISSI": cp.asarray(np.full((jm, im), 0.95, dtype=np.float32)),
            "T24": cp.asarray((sfctmp * sfctmp * sfctmp * sfctmp).astype(np.float32)),
            "RCH": cp.asarray(rng.uniform(0.002, 0.006, size=(jm, im)).astype(np.float32)),
            "RR": cp.asarray(rng.uniform(1.0, 2.0, size=(jm, im)).astype(np.float32)),
            "BETA": cp.asarray(rng.uniform(0.0, 1.0, size=(jm, im)).astype(np.float32)),
            "EPSCA": cp.asarray(rng.uniform(-5.0, 5.0, size=(jm, im)).astype(np.float32)),
            "DF1": cp.asarray(rng.uniform(1.0, 3.5, size=(jm, im)).astype(np.float32)),
        }
    else:
        jm, im = int(stc.shape[1]), int(stc.shape[2])
        _check_sflx(sflx, jm, im)
    o = _outputs(jm, im)
    rhsts = o["rhsts"] if rhsts is None else rhsts
    ai = o["ai"] if ai is None else ai
    bi = o["bi"] if bi is None else bi
    ci = o["ci"] if ci is None else ci
    hrtice(stc, ice, sflx, rhsts, ai, bi, ci)
    return {"RHSTS": rhsts, "AI": ai, "BI": bi, "CI": ci}
