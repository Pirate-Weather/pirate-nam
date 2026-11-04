"""phys-rrtm-rsipath2: cloud liquid, ice, rain and snow paths and effective radii (rsipath2_tmp).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_clouds_nmmb.f:2779-2964 (subroutine
rsipath2_tmp, a modified Ferrier rsipath2). NOAA US government work; the NOAA history block is kept
as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_rsipath2.cu (rsipath2_tmp, one thread per column, one launch).

Inputs, by CONTRACT name (all float32, Fortran units):
  plyr   [LM, JM, IM]     layer mean pressure, mb (100 Pa)
  plvl   [LM+1, JM, IM]   interface pressure, mb; plvl[0] = top
  tlyr   [LM, JM, IM]     layer temperature, K
  qlyr   [LM, JM, IM]     specific humidity, gm/gm
  qcwat  [LM, JM, IM]     cloud liquid condensate
  qcice  [LM, JM, IM]     cloud ice condensate
  qrain  [LM, JM, IM]     rain water
Outputs (returned, and kept as state scratch RSIPATH2_<name>):
  cwatp, cicep, rainp, snowp, recwat, rerain, resnow, snden   all [LM, JM, IM] float32

The production caller (the NMMB RRTM cloud-input path) passes the arrays explicitly as keyword
arguments. With no arguments the wrapper builds the inputs from the State (for the done-rule check):
plvl = PINT/100, plyr = layer mean of PINT/100, tlyr = T, qlyr = Q, and the condensate CW split by
F_ICE and F_RAIN (liquid = (1-F_ICE) CW, rain = F_RAIN of the liquid part).

Not ported: the dummy arguments rrime, flgmin, lprnt and iflip (see the kernel header). recwat and
rerain are only written where qcwat > 0 or qrain > 0, as in the Fortran; the wrapper does not zero them.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_rsipath2"
KERNEL = "rsipath2_tmp"
OUT_NAMES = ("cwatp", "cicep", "rainp", "snowp", "recwat", "rerain", "resnow", "snden")
IN_NAMES = ("plyr", "plvl", "tlyr", "qlyr", "qcwat", "qcice", "qrain")


def _default_inputs(ctx):
    """Inputs from the State, for the standalone check (see the module docstring)."""
    st = ctx.st
    pint_pa = st["PINT"]                       # [LM+1, JM, IM] Pa
    plvl = (pint_pa / np.float32(100.0)).astype(np.float32)
    plyr = (0.5 * (pint_pa[:-1] + pint_pa[1:]) / np.float32(100.0)).astype(np.float32)
    cw = st["CW"]
    f_ice = st["F_ICE"]
    f_rain = st["F_RAIN"]
    liquid = (1.0 - f_ice) * cw
    return {
        "plyr": plyr, "plvl": plvl, "tlyr": st["T"], "qlyr": st["Q"],
        "qcwat": liquid * (1.0 - f_rain), "qcice": f_ice * cw, "qrain": liquid * f_rain,
    }


def _check(name, a, shape):
    if tuple(a.shape) != tuple(shape) or a.dtype != np.float32:
        raise ValueError(f"rsipath2: {name} must be float32 {tuple(shape)}, got {a.dtype} {a.shape}")


def run(ctx, *, out=None, **inputs) -> dict:
    """Launch rsipath2_tmp once for the whole domain. Returns {name: array} for OUT_NAMES."""
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if not inputs:
        inputs = _default_inputs(ctx)
    missing = [n for n in IN_NAMES if n not in inputs]
    if missing:
        raise ValueError(f"rsipath2: missing inputs {missing}")
    x = {n: inputs[n] for n in IN_NAMES}
    _check("plvl", x["plvl"], (lm + 1, jm, im))
    for n in IN_NAMES:
        if n != "plvl":
            _check(n, x[n], (lm, jm, im))

    # Outputs: caller's arrays if given, else scratch allocated once on the State and reused.
    shape = (lm, jm, im)
    outs = {}
    for n in OUT_NAMES:
        if out is not None and n in out:
            _check(n, out[n], shape)
            outs[n] = out[n]
        else:
            outs[n] = ctx.st.add(f"RSIPATH2_{n.upper()}", shape, dtype="float32")

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lm),
            x["plyr"], x["plvl"], x["tlyr"], x["qlyr"], x["qcwat"], x["qcice"], x["qrain"],
            outs["cwatp"], outs["cicep"], outs["rainp"], outs["snowp"],
            outs["recwat"], outs["rerain"], outs["resnow"], outs["snden"],
        ))
    return outs
