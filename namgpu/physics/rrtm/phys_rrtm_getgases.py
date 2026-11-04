"""phys-rrtm-getgases: getgases, gas volume-mixing-ratio profiles for the NMMB RRTM call (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_gases_nmmb.f:843-979, getgases.
NOAA US government work, no licence file; the NOAA/NCEP routine header is kept as the notice in
the kernel file namgpu/kernels/phys_rrtm_getgases.cu (one thread per column, every level loop inside).

Caller: grrad_nmmb.f:1419-1427 calls getgases once per row with plvl, xlon, xlat. Here the whole
domain is one grid, so the one launch covers every (i, j) column.

Inputs (device, float32):
  PLVL  [LMAX+1, JM, IM]  level pressure in mb (grrad setup output, LMAX = LM + 1, so LMAX+1 = LM + 2)
  GLON, GLAT [JM, IM]     longitude/latitude in radians (state table; must already be in the state)
  co2_glb  [1], gco2cyc [12], co2vmr_sav [12, 12, 24]  module data from gas_init
    (ctx.tables["rrtm_gases"], written by the gas-init unit phys-rrtm-gas-init)
  ico2flg (0, 1, 2) and ivflip (0, 1): from the same tables dict (gas-init writes them)
  kmonsav (1-based month, 1..12): written by phys-rrtm-gas-update. Required when ico2flg is 1 or 2,
    because the kernel reads gco2cyc[kmonsav-1].
Output (scratch, st.add): GASVMR [10, LMAX, JM, IM] = Fortran gasdat(IMAX, LMAX, NF_VGAS) in
the CONTRACT layout gasvmr[n, l, j, i].  Gas order: co2, n2o, ch4, o2, co, cfc11, cfc12, cfc22,
ccl4, cfc113.

Declared deviation (kernel phys_rrtm_getgases.cu): the ilon/ilat lower clamp to 1. Fortran has no
clamp and would read co2vmr_sav(...,0,...) for xlat more than about 15 deg past the pole; CONUS never
reaches that.

run(None) is the done-rule entry (full-size 954 x 835 x 60 with ico2flg = 2 so every branch runs,
plvl monotone and xlon/xlat in range, finite check on the output).
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_getgases"
KERNEL = "phys_rrtm_getgases"
NF_VGAS = 10            # radiation_gases_nmmb.f NF_VGAS
IMXCO2 = 24             # input co2 data lon points
JMXCO2 = 12             # input co2 data lat points
NMONTH = 12
PRESS_TOP = 200.0       # Pa, PT (contract section 5)
PRESS_SFC = 100000.0    # Pa
TABLE_KEYS = ("co2_glb", "gco2cyc", "co2vmr_sav", "ico2flg", "ivflip")


def _check_getgases_inputs(co2_glb, gco2cyc, co2vmr_sav, ico2flg, kmonsav, ivflip):
    """Refuses inputs the kernel would read out of bounds, or that gas-init never produces."""
    if ico2flg not in (0, 1, 2):
        raise ValueError(f"phys_rrtm_getgases: ico2flg={ico2flg} is not 0, 1 or 2")
    if ivflip not in (0, 1):
        raise ValueError(f"phys_rrtm_getgases: ivflip={ivflip} is not 0 or 1")
    if tuple(co2_glb.shape) != (1,):
        raise ValueError(f"phys_rrtm_getgases: co2_glb shape {tuple(co2_glb.shape)} is not (1,)")
    if tuple(gco2cyc.shape) != (NMONTH,):
        raise ValueError(f"phys_rrtm_getgases: gco2cyc shape {tuple(gco2cyc.shape)} is not ({NMONTH},)")
    if tuple(co2vmr_sav.shape) != (NMONTH, JMXCO2, IMXCO2):
        raise ValueError(f"co2vmr_sav shape {tuple(co2vmr_sav.shape)} does not match "
                         f"[{NMONTH}, {JMXCO2}, {IMXCO2}]")
    if ico2flg in (1, 2):
        if kmonsav is None or not (1 <= int(kmonsav) <= NMONTH):
            raise ValueError(f"phys_rrtm_getgases: kmonsav={kmonsav} must be 1..{NMONTH} when "
                             f"ico2flg={ico2flg} (phys-rrtm-gas-update sets the month)")


def getgases(plvl, xlon, xlat, gasvmr, *, co2_glb, gco2cyc, co2vmr_sav,
             ico2flg: int, kmonsav, ivflip: int, stream=None):
    """One launch over every (i, j). plvl [LMAX+1, JM, IM], xlon/xlat [JM, IM], gasvmr [10, LMAX, JM, IM]."""
    import contextlib

    jm, im = xlon.shape
    lmax = gasvmr.shape[1]
    if gasvmr.shape[0] != NF_VGAS or gasvmr.shape[2:] != (jm, im):
        raise ValueError(f"gasvmr shape {gasvmr.shape} does not match [10, LMAX, {jm}, {im}]")
    if plvl.shape != (lmax + 1, jm, im):
        raise ValueError(f"plvl shape {plvl.shape} does not match [{lmax + 1}, {jm}, {im}]")
    if xlat.shape != (jm, im):
        raise ValueError(f"xlat shape {xlat.shape} does not match [{jm}, {im}]")
    _check_getgases_inputs(co2_glb, gco2cyc, co2vmr_sav, ico2flg, kmonsav, ivflip)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            plvl, xlon, xlat, co2_glb, gco2cyc, co2vmr_sav, gasvmr,
            np.int32(im), np.int32(jm), np.int32(lmax),
            np.int32(ico2flg), np.int32(1 if kmonsav is None else int(kmonsav)), np.int32(ivflip),
        ))
    return gasvmr


def run(ctx=None, *, ico2flg: int = 2, ivflip: int = 0):
    """Launch the unit once. Returns the arrays it wrote (dict of cupy arrays).

    With ctx, reads PLVL, GLON, GLAT from ctx.st (GLON and GLAT must already be in the state; a missing
    one raises instead of zero-filling) and the gas-init module data from ctx.tables["rrtm_gases"]
    (co2_glb, gco2cyc, co2vmr_sav, ico2flg, ivflip; kmonsav from gas-update when ico2flg is 1 or 2).
    The scalar ico2flg/ivflip arguments are used only for the standalone run(None) check.
    """
    import cupy as cp

    if ctx is None:
        im, jm, lm = 954, 835, 60
        st = None
    else:
        im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
        st = ctx.st
    lmax = lm + 1                       # LMAX = LMK (grrad caller)

    def add(name, shape):
        if st is None:
            return cp.zeros(shape, dtype=cp.float32)
        return st.add(name, shape)

    if st is None:
        lev = np.linspace(PRESS_TOP, PRESS_SFC, lmax + 1).astype(np.float32)
        plvl = cp.asarray(np.broadcast_to(lev[:, None, None], (lmax + 1, jm, im)).copy())
        xlon = cp.asarray(np.random.default_rng(1).uniform(-np.pi, np.pi, (jm, im)).astype(np.float32))
        xlat = cp.asarray(np.random.default_rng(2).uniform(-1.2, 1.2, (jm, im)).astype(np.float32))
        co2_glb = cp.asarray(np.array([350.0e-6], dtype=np.float32))
        gco2cyc = cp.asarray(np.linspace(-2.0e-6, 2.0e-6, 12).astype(np.float32))
        co2vmr_sav = cp.asarray(
            (350.0e-6 + 2.0e-6 * np.random.default_rng(3).standard_normal((12, JMXCO2, IMXCO2)))
            .astype(np.float32))
        kmonsav = 6
    else:
        for name in ("GLON", "GLAT"):
            if name not in st.fields:
                raise RuntimeError(f"phys_rrtm_getgases: state field {name} is missing. The state must carry "
                                   "GLON and GLAT before getgases; a silent zero-fill would give xlon=xlat=0.")
        plvl = st.add("PLVL", (lmax + 1, jm, im))
        xlon = st.fields["GLON"]
        xlat = st.fields["GLAT"]
        tab = ctx.tables.get("rrtm_gases") if getattr(ctx, "tables", None) is not None else None
        if tab is None:
            raise RuntimeError("phys_rrtm_getgases needs ctx.tables['rrtm_gases'] from phys-rrtm-gas-init")
        missing = [k for k in TABLE_KEYS if k not in tab]
        if missing:
            raise RuntimeError(f"phys_rrtm_getgases: ctx.tables['rrtm_gases'] lacks {missing}. "
                               "phys-rrtm-gas-init must run first, and phys-rrtm-gas-update for kmonsav.")
        co2_glb = tab["co2_glb"]
        gco2cyc = tab["gco2cyc"]
        co2vmr_sav = tab["co2vmr_sav"]
        ico2flg = int(tab["ico2flg"])
        ivflip = int(tab["ivflip"])
        kmonsav = tab.get("kmonsav")
        kmonsav = None if kmonsav is None else int(kmonsav)

    gasvmr = add("GASVMR", (NF_VGAS, lmax, jm, im))
    getgases(plvl, xlon, xlat, gasvmr, co2_glb=co2_glb, gco2cyc=gco2cyc,
             co2vmr_sav=co2vmr_sav, ico2flg=ico2flg, kmonsav=kmonsav, ivflip=ivflip,
             stream=None if ctx is None else ctx.stream)
    return {"GASVMR": gasvmr}
