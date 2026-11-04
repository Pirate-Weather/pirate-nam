"""phys-rrtm-grrad-gases: the layer/level thermodynamics block of grrad_nmmb (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/grrad_nmmb.f:1397-1539. NOAA US government work, no
licence file; the NOAA/NCEP header is kept as the notice in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_grrad_gases.cu (phys_rrtm_grrad_gases, one thread per column,
one launch, every level loop inside the thread, float32, fmad on, no host sync).

Scope of this unit (the Fortran range): the statements between the ozone/cos-zenith/gas calls and
`nday = im`: the log(p) loops and the ivflip-dependent layer/level temperature and humidity build
(qlyr with the tem1d floor, tvly with the virtual-temperature factor, tlvl interpolation, tsfa).
The three CALLs in the range (getozn, coszmn_nmmb, getgases) are the sibling units
phys-rrtm-getozn, phys-rrtm-coszmn and phys-rrtm-getgases. Their outputs (olyr, coszen_loc,
coszdg, gasvmr) are not inputs of this block; the driver dispatches them in Fortran order before
this unit, and this wrapper does not run them.

Inputs, all bound from the producers (no stand-ins; a missing producer raises):
  RAD_PLYR   [LMK, JM, IM]  layer pressure      <- phys-rrtm-grrad-setup (Fortran plyr)
  RAD_PLVL   [LMP, JM, IM]  level pressure      <- phys-rrtm-grrad-setup (Fortran plvl)
  RAD_TLYR   [LMK, JM, IM]  layer temperature   <- phys-rrtm-grrad-setup (Fortran tlyr = tgrs)
  RAD_TSKN   [JM, IM]       skin temperature    <- phys-rrtm-grrad-setup (Fortran tskn = tsfc,
                            itsfc == 0, grrad_nmmb.f:1221-1232)
  T          [LM, JM, IM]   model temperature   (state table, Fortran tgrs = GT)
  Q          [LM, JM, IM]   model specific humidity (state table, Fortran qgrs = GQ)
Outputs (scratch, st.add):
  TLVL [LMP, JM, IM], TVLY [LMK, JM, IM], QLYR [LMK, JM, IM], TEM2DA [LMK, JM, IM],
  TEM2DB [LMP, JM, IM], TSFA [JM, IM]
  RAD_NDAY int32 [1] = IM (Fortran `nday = im`, grrad_nmmb.f:1534), consumed by the downstream
  sw/lw day-part and chunk-padding units.

ivflip: the NAM setting is iflip = 0 (input toa -> sfc, rad_initialize_nmmb.f:183 with the
call value from phys-rrtm-rad-initialize). Both branches are in the kernel.

run(ctx, ...) is called by the driver and tools/check_unit.py with a full context. There is no
context-free mode: the inputs are the setup outputs, and inventing them would hide a broken chain.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_grrad_gases"
KERNEL = "phys_rrtm_grrad_gases"
SCRATCH = ("TLVL", "TVLY", "QLYR", "TEM2DA", "TEM2DB", "TSFA")
SETUP_OUTPUTS = ("RAD_PLYR", "RAD_PLVL", "RAD_TLYR", "RAD_TSKN")
BLOCK = (128, 1, 1)


def _grid(im: int, jm: int):
    from namgpu.kernels import grid_cols
    return grid_cols(im, jm), BLOCK


def run(ctx, *, ivflip: int = 0):
    """Launch the unit once. Returns the arrays it wrote (dict of cupy arrays) plus 'nday'."""
    if ctx is None:
        raise ValueError("phys-rrtm-grrad-gases needs a context; its inputs are the grrad-setup outputs")
    st = ctx.st
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    lmk, lmp = lm + 1, lm + 2

    missing = [n for n in SETUP_OUTPUTS if n not in st.fields]
    if missing:
        raise KeyError(
            f"phys-rrtm-grrad-gases: {missing} not produced; run phys-rrtm-grrad-setup first "
            "(its outputs are the Fortran plyr, plvl, tlyr and tskn of grrad_nmmb)")

    plyr = st.fields["RAD_PLYR"]
    plvl = st.fields["RAD_PLVL"]
    tlyr = st.fields["RAD_TLYR"]
    tskn = st.fields["RAD_TSKN"]
    tgrs, qgrs = st.T, st.Q

    outs = {n: st.add(n, shp) for n, shp in (
        ("TLVL", (lmp, jm, im)), ("TVLY", (lmk, jm, im)), ("QLYR", (lmk, jm, im)),
        ("TEM2DA", (lmk, jm, im)), ("TEM2DB", (lmp, jm, im)), ("TSFA", (jm, im)))}

    # Fortran `nday = im` (grrad_nmmb.f:1534): kept on the device for the downstream units.
    nday_arr = st.add("RAD_NDAY", (1,), dtype="int32")
    nday_arr[...] = im

    grid, block = _grid(im, jm)
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    k = kernels.get(UNIT, KERNEL)
    with scope:
        k(grid, block,
          (plyr, plvl, tlyr, tskn, tgrs, qgrs,
           outs["TLVL"], outs["TVLY"], outs["QLYR"], outs["TEM2DA"], outs["TEM2DB"], outs["TSFA"],
           np.int32(im), np.int32(jm), np.int32(lm), np.int32(ivflip)))
    outs["RAD_NDAY"] = nday_arr
    outs["nday"] = int(im)
    return outs
