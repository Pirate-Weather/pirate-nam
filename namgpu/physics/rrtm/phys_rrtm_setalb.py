"""phys-rrtm-setalb: SETALB, surface albedo for the SW radiation call (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_surface_nmmb.f:407-725, subroutine setalb.
NOAA US government work; the NOAA header is kept as the notice in the kernel file
(namgpu/kernels/phys_rrtm_setalb.cu). Caller: grrad_nmmb.f:1818-1820 (`call setalb`, inside `if (lsswr)`),
fed by module_RA_RRTM.F90:905-918, which derives the arguments as follows (the wrapper does the same on device):
  SLMSK  = 2 where SICE > 0.5, else 1 - SM            (state SM is 1 = sea, 0 = land)   -> SETALB_SLMSK
  FICE   = SICE where SICE > 0.5, else 0                                                  -> SETALB_FICE
  SNWDPH = SI, snow depth in mm (module_RA_RRTM.F90:907; NOT the liquid snow SNO)         -> snowf
  ZORL   = Z0 * 100, metres to cm (module_RA_RRTM.F90:905)                                 -> SETALB_ZORLF
  COSZEN = coszen_loc, the COSZMN_NMMB output of the same radiation call (grrad_nmmb.f:1413,
           used at 1818). That is the phys-rrtm-dayparts coszen (ctx.tables["rrtm_dayparts"]), which
           phys-rrtm-grrad-sw runs before setalb. The instantaneous state CZEN is not used.
  TISFC  = TSKIN (module_RA_RRTM.F90:906, "change later if necessary")
  HPRIM  = STDH, terrain standard deviation in m (module_RA_RRTM.F90:911; state table STDH)
  SNCOVR = SNOWC, SNOALB = MXSNAL (ialbflg 1 only)

Kernel: phys_rrtm_setalb, one launch, one thread per (i, j) column (kernels.grid_cols).

Inputs, bound from ctx.st by NMMB state name (2-D [JM, IM], float32):
  SM, SICE, SI, Z0, TSKIN, STDH, SNOWC, MXSNAL   state tables (see the derivations above)
  coszen                                          ctx.tables["rrtm_dayparts"]["coszen"], reshaped [JM, IM]
  ALVSF, ALNSF, ALVWF, ALNWF                      albedo components, ialbflg 0, 1 and 2
  FACSF, FACWF                                    land-type fractions, ialbflg 0 and 1

Producers of the albedo inputs:
  ialbflg 2 (NAM regional path): phys-rrtm-grrad-clouds, its IALB=2 block (grrad_nmmb.f:1722-1794,
    Payne 1972 tables; FACSF = FACWF = 0 there, grrad_nmmb.f:1790-1791). That unit must run first.
  ialbflg 0 and 1: the caller's albedo and land-fraction arrays (grrad_nmmb.f arguments from
    module_RA_RRTM.F90). No port of that producer is in the tree yet. The missing inputs raise
    RuntimeError naming it: a named gap, not a silent zero.

ialbflg: read from ctx.cfg if present, else IALB_NAM_REGIONAL = 2. The NAM solver sets IALB = 2
for regional runs (module_SOLVER_GRID_COMP.F90:10676, "ONLY THIS ONE WORKS for regional").
Branch 2 reads only the four albedo components; it needs no snow, mask, ice, roughness or cos-zenith input.

Output: ctx.st.SFCALB [4, JM, IM] (scratch added here, State.add is idempotent):
  component 0 near-IR direct, 1 near-IR diffuse, 2 UV+vis direct, 3 UV+vis diffuse.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_setalb"
KERNEL = "phys_rrtm_setalb"
NF_ALBD = 4                  # radiation_surface_nmmb.f:94
IALB_NAM_REGIONAL = 2        # module_SOLVER_GRID_COMP.F90:10676

ALBEDO_ALL = ("ALVSF", "ALNSF", "ALVWF", "ALNWF")
ALBEDO_LAND = ("FACSF", "FACWF")
SURFACE_01 = ("SM", "SICE", "SI", "Z0", "TSKIN", "STDH")
SNOW_1 = ("SNOWC", "MXSNAL")


def _need_names(ialbflg):
    if ialbflg in (0, 1):
        names = list(SURFACE_01) + list(ALBEDO_ALL) + list(ALBEDO_LAND)
        if ialbflg == 1:
            names += list(SNOW_1)
        return names
    return list(ALBEDO_ALL)


def _need(st, names):
    missing = [n for n in names if st.fields.get(n) is None]
    if missing:
        raise RuntimeError(
            "phys-rrtm-setalb needs arrays not yet produced: " + ", ".join(missing)
            + " (ALVSF/ALNSF/ALVWF/ALNWF for ialbflg 2 come from phys-rrtm-grrad-clouds, its IALB=2 block "
              "grrad_nmmb.f:1722-1794, which must run first; FACSF/FACWF and the albedo components for "
              "ialbflg 0/1 have no producer in the tree yet: a named gap)")


def run(ctx, ialbflg: int | None = None):
    """Driver entry. Fills ctx.st.SFCALB [4, JM, IM] for the SW call; no host sync."""
    import cupy as cp

    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    st = ctx.st
    if ialbflg is None:
        ialbflg = int(getattr(ctx.cfg, "ialbflg", IALB_NAM_REGIONAL))

    _need(st, _need_names(ialbflg))
    f = st.fields
    sfcalb = st.add("SFCALB", (NF_ALBD, jm, im))
    alvsf, alnsf, alvwf, alnwf = f["ALVSF"], f["ALNSF"], f["ALVWF"], f["ALNWF"]

    if ialbflg in (0, 1):
        dp = ctx.tables.get("rrtm_dayparts")
        if dp is None:
            raise RuntimeError("phys-rrtm-setalb needs coszen_loc from phys-rrtm-dayparts "
                               "(grrad_nmmb.f:1413 feeds setalb at 1818); run dayparts first")
        coszf = dp["coszen"].reshape(jm, im)
        zorl_cm = st.add("SETALB_ZORLF", (jm, im))
        cp.multiply(f["Z0"], 100.0, out=zorl_cm)                 # Fortran zorlf is in cm
        slmsk = st.add("SETALB_SLMSK", (jm, im))
        fice = st.add("SETALB_FICE", (jm, im))
        sice = f["SICE"]
        icemask = sice > 0.5
        cp.subtract(1.0, f["SM"], out=slmsk)                     # SLMSK = 1 - SM
        cp.copyto(slmsk, 2.0, where=icemask)                     # SLMSK = 2 where SICE > 0.5
        cp.multiply(sice, icemask, out=fice)                     # FICE = SICE where SICE > 0.5, else 0
        snowf = f["SI"]                                          # snwdph, mm
        hprim = f["STDH"]
        tisfc = f["TSKIN"]                                       # caller sets TISFC = TSKIN
        tsknf = f["TSKIN"]
        if ialbflg == 1:
            sncovr, snoalb = f["SNOWC"], f["MXSNAL"]
        else:
            sncovr = snoalb = f["SI"]                            # not read on branch 0
        facsf, facwf = f["FACSF"], f["FACWF"]
    else:
        # Branch 2 reads only the four albedo components. The other kernel pointers are never
        # dereferenced there; a valid albedo array stands in for them.
        zorl_cm = snowf = sncovr = snoalb = coszf = hprim = tisfc = tsknf = alvsf
        slmsk = fice = facsf = facwf = alvsf

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            slmsk, snowf, sncovr, snoalb, zorl_cm, coszf, tsknf, hprim,
            alvsf, alnsf, alvwf, alnwf, facsf, facwf,
            fice, tisfc,
            np.int32(ialbflg), np.int32(im), np.int32(jm), sfcalb,
        ))
    return sfcalb
