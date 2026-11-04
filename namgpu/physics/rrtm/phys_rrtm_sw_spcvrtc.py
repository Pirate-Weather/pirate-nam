"""phys-rrtm-sw-spcvrtc: two-stream layer solution and column flux accumulation of the NMMB RRTM shortwave.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1): phys/radsw_main_nmmb.f:2144-2964 (spcvrtc) and
phys/radsw_main_nmmb.f:2977-3102 (swflux). Notice: the sw-rrtm3 header of radsw_main_nmmb.f (NCEP
modifications of the AER RRTM-SW code). Kernel: namgpu/kernels/phys_rrtm_sw_spcvrtc.cu.

Structure: one kernel launch per g-point (112 per call), each over every column (one thread per
column). The Fortran sums the g-points inside one call; the sum order is kept. Per-g-point launches
are a memory decision (the kernel header has the numbers), not a per-cell host loop.

Inputs, read from ctx.tables:
  "rrtm_sw_in"        dict of device arrays (float32):
                        cosz, sntz, ssolar, cf1, cf0   [JM, IM]
                        albbm, albdf                   [2, JM, IM]
                        cldfrc                         [NLAY, JM, IM]
  "rrtm_sw_gpoint"    callable jg (1-based, 1..112) -> (taug, taur), float32 [NLAY, JM, IM] for this g-point
  "rrtm_sw_sfluxzen"  float32 [NGPTSW, JM, IM] TOA flux per g-point and column, written by
                        phys-rrtm-sw-taumol (ctx.tables["rrtm_sw_sfluxzen"]); read per column in the kernel
  "rrtm_sw_band"      callable ib (1-based band index, 1..14) -> dict with float32 [NLAY, JM, IM]
                        tauae, ssaae, asyae, taucw, ssacw, asycw (the aerosol and cloud optics of
                        band ib; called once per band, the g-points are visited in band order)
  "rrtm_sw_exp_tbl"   optional float32 [NTBMX+1]; built here with the rswinit recipe if absent
Outputs, kept in ctx.tables["rrtm_sw_out"] (allocated once, zeroed by the kernel each call):
  fxupc, fxdnc, fxup0, fxdn0 float32 [NBDSW, NLP1, JM, IM] (band outermost, as Fortran fxup0(i,k,ib))
  ftoadc ftoauc ftoau0 fsfcuc fsfcu0 fsfcdc fsfcd0 suvbfc suvbf0 float32 [JM, IM]
  sfbmc sfdfc sfbm0 sfdf0 float32 [2, JM, IM]

Control: iswmode = 2 (PIFM, physparam.f:62). Band and g-point tables: radsw_main.f:312 (idxsfc),
radsw_param.f:113 (NTBMX) and radsw_param.f:145-157 (NGB), NG16..NG29 (radsw_param.f:105-109).
Day/night: a column is sunlit when cosz >= 0.0001 (grrad_nmmb.f:640-644, the Fortran NDAY set); other
columns get zero fluxes.
Layer order: radiation-internal, surface first (CONTRACT.md section 1 SW exception). The swrad seam
reverses the NMMB top-first state arrays before this unit and flips the fluxes back after it.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_spcvrtc"
KERNEL = "phys_rrtm_sw_spcvrtc"
NGPTSW = 112          # radsw_param.f:110
NBDSW = 14            # radsw_param.f NBANDS = 29 - 16 + 1
NBLOW = 16            # radsw_param.f NBLOW
NTBMX = 10000         # radsw_param.f:113
NLAY_MAX = 64         # kernel scratch limit (SWC_NLAY_MAX)
DAY_COSZ = 1.0e-4     # grrad_nmmb.f:640,644 sunlit threshold (NDAY set)
ISWMODE = 2           # physparam.f:62 iswmode
EXPEPS = np.float32(1.0e-20)  # radsw_main.f:1199 expeps

# NG16..NG29 (radsw_param.f:105-109) and NGB (radsw_param.f:145-157): band of each g-point.
_NG = {16: 6, 17: 12, 18: 8, 19: 8, 20: 10, 21: 10, 22: 2, 23: 10,
       24: 8, 25: 6, 26: 6, 27: 8, 28: 6, 29: 12}
_NGB = np.array([b for b in range(16, 30) for _ in range(_NG[b])], dtype=np.int32)
assert _NGB.shape == (NGPTSW,)

# idxsfc(nblow:nbhgh) (radsw_main.f:312): 0 = surface flux split evenly over both albedo bands.
_IDXSFC = {16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1,
           24: 0, 25: 2, 26: 2, 27: 2, 28: 2, 29: 1}


def build_exp_tbl() -> np.ndarray:
    """exp_tbl(0:NTBMX) as rswinit builds it (radsw_main.f:1275-1282): tfn, tau and exp in double
    (kind_phys = 8 in the reference build), stored float32 for this consumer. Matches
    phys_rrtm_sw_rswinit.cu rswinit_tables entry for entry up to the float32 cast."""
    t = np.empty(NTBMX + 1, dtype=np.float32)
    t[0] = 1.0
    bpade = 1.0 / 0.278
    i = np.arange(1, NTBMX, dtype=np.float64)
    tfn = i / (float(NTBMX) - i)
    tau = bpade * tfn
    t[1:NTBMX] = np.exp(-tau).astype(np.float32)
    t[NTBMX] = np.float32(EXPEPS)
    return t


def alloc_outputs(nlay: int, jm: int, im: int) -> dict:
    """Output arrays of spcvrtc, allocated once. The kernel zeroes them at the first g-point."""
    import cupy as cp

    nlp1 = nlay + 1

    def z4():
        return cp.zeros((NBDSW, nlp1, jm, im), dtype=cp.float32)

    def z3():
        return cp.zeros((2, jm, im), dtype=cp.float32)

    def z2():
        return cp.zeros((jm, im), dtype=cp.float32)

    return {
        "fxupc": z4(), "fxdnc": z4(), "fxup0": z4(), "fxdn0": z4(),
        "ftoadc": z2(), "ftoauc": z2(), "ftoau0": z2(), "fsfcuc": z2(),
        "fsfcu0": z2(), "fsfcdc": z2(), "fsfcd0": z2(),
        "sfbmc": z3(), "sfdfc": z3(), "sfbm0": z3(), "sfdf0": z3(),
        "suvbfc": z2(), "suvbf0": z2(),
    }


def spcvrtc(inp: dict, out: dict, gpoint, band, exp_tbl, sfluxzen, stream=None) -> dict:
    """All 112 g-points of spcvrtc, one launch each, in Fortran order (jg = 1..112)."""
    import cupy as cp  # noqa: F401  (device arrays arrive as cupy arrays)

    cosz = inp["cosz"]
    jm, im = cosz.shape
    nlay = inp["cldfrc"].shape[0]
    if nlay > NLAY_MAX:
        raise ValueError(f"phys-rrtm-sw-spcvrtc: NLAY {nlay} > kernel limit {NLAY_MAX}")
    if tuple(exp_tbl.shape) != (NTBMX + 1,):
        raise ValueError("phys-rrtm-sw-spcvrtc: exp_tbl must be float32 [NTBMX+1]")
    if tuple(sfluxzen.shape) != (NGPTSW, jm, im):
        raise ValueError("phys-rrtm-sw-spcvrtc: sfluxzen must be float32 [NGPTSW, JM, IM]")

    k = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    scope = stream if stream is not None else contextlib.nullcontext()
    cur_ib = None
    bdata = None
    with scope:
        for jg in range(1, NGPTSW + 1):
            jb = int(_NGB[jg - 1])
            ib = jb + 1 - NBLOW          # radsw_main.f:2144 ib = jb + 1 - nblow
            ibd = _IDXSFC[jb]            # radsw_main.f:2145 ibd = idxsfc(jb)
            if ib != cur_ib:
                bdata = band(ib)
                cur_ib = ib
            taug, taur = gpoint(jg)
            k(grid, kernels.BLOCK_COL, (
                inp["ssolar"], inp["cosz"], inp["sntz"], inp["albbm"], inp["albdf"], inp["cldfrc"],
                inp["cf1"], inp["cf0"], taug, taur,
                bdata["tauae"], bdata["ssaae"], bdata["asyae"],
                bdata["taucw"], bdata["ssacw"], bdata["asycw"],
                exp_tbl,
                out["fxupc"], out["fxdnc"], out["fxup0"], out["fxdn0"],
                out["ftoadc"], out["ftoauc"], out["ftoau0"],
                out["fsfcuc"], out["fsfcu0"], out["fsfcdc"], out["fsfcd0"],
                out["sfbmc"], out["sfdfc"], out["sfbm0"], out["sfdf0"],
                out["suvbfc"], out["suvbf0"],
                sfluxzen, np.int32(jg), np.int32(ib), np.int32(ibd),
                np.int32(jg == 1), np.int32(jg == NGPTSW),
                np.int32(nlay), np.int32(nlay + 1), np.int32(jm), np.int32(im),
                np.int32(ISWMODE)))
    return out


def run(ctx):
    """Driver entry: reads the upstream optics from ctx.tables (see the module docstring)."""
    import cupy as cp

    tables = ctx.tables
    inp = tables.get("rrtm_sw_in")
    gpoint = tables.get("rrtm_sw_gpoint")
    band = tables.get("rrtm_sw_band")
    if inp is None or gpoint is None or band is None:
        raise RuntimeError(
            "phys_rrtm_sw_spcvrtc needs ctx.tables rrtm_sw_in, rrtm_sw_gpoint and rrtm_sw_band "
            "from the cldprop, taumol and setalb units")
    sfluxzen = tables.get("rrtm_sw_sfluxzen")
    if sfluxzen is None:
        raise RuntimeError("phys_rrtm_sw_spcvrtc needs ctx.tables rrtm_sw_sfluxzen from phys-rrtm-sw-taumol")
    exp_tbl = tables.get("rrtm_sw_exp_tbl")
    if exp_tbl is None:
        exp_tbl = cp.asarray(build_exp_tbl())
        tables["rrtm_sw_exp_tbl"] = exp_tbl
    out = tables.get("rrtm_sw_out")
    if out is None:
        nlay, jm, im = inp["cldfrc"].shape
        out = alloc_outputs(nlay, jm, im)
        tables["rrtm_sw_out"] = out
    return spcvrtc(inp, out, gpoint, band, exp_tbl, sfluxzen, stream=getattr(ctx, "stream", None))
