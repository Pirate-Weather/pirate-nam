"""phys-rrtm-dayparts: DAYPARTS sunlit-point selection for NMMB RRTM (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/grrad_nmmb.f:606-656, subroutine dayparts, which
calls COSZMN_NMMB (phys/radiation_astronomy_nmmb.f:744-848). NOAA US government work; the NOAA header
of grrad_nmmb.f is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_dayparts.cu (three kernels plus one CuPy cumsum; see the kernel header).

Caller semantics: module_RA_RRTM.F90:879-881 calls dayparts once per row J with LENIVEC = IM points.
The run table is therefore built per row: each row's first run starts at its point 1, DP_START is
row-relative (1-based), and runs never cross a row boundary.

Inputs (flat, n = JM*IM, row-major: p = j*IM + i, 0-based):
  xlon, sinlat, coslat  float32[n]   longitude (radians), sin and cos of latitude (H points, flattened)
  solhr                 float        time after 00z in hours (ctx.clock.solhr, as coszmn uses it)
  dtswav                float        shortwave calling interval in HOURS (coszmn_nmmb doc)
  nrads                 int          number of zenith iterations per interval (NRADS)
  sol                   device float32[4] SOL_STATE = sollag, sindec, cosdec, anginc, written by the
                                     solar update (sol_update, phys-rrtm-sol-update); read on the device
Outputs (device, bound to ctx.tables["rrtm_dayparts"] by run()):
  coszen[n], coszdg[n]  float32      COSZMN_NMMB outputs for every H point
  dp_start[n], dp_len[n], dp_day[n] int32   run table in slot layout s = row*IM + k: run k of row j
                                     is valid for k < ndayparts[j]; dp_start is 1-based within the row,
                                     dp_len is the point count, dp_day is 0/1
  ndayparts[JM]         int32        number of runs in each row
Consumer contract (the grrad caller loop, Fortran grrad_nmmb.f:640-653 / module_RA_RRTM.F90:883-1223):
  for row j, for k < ndayparts[j]: slot = j*IM + k; IBEG = its + dp_start - 1; IEND = IBEG + dp_len - 1;
  dpd = dp_day. The caller loop body is not part of this unit.
One host call per routine; no per-cell host work, no per-step allocation (scratch cached per size).
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_dayparts"
SOL_STATE = "SOL_STATE"   # phys-rrtm-sol-update: float32[4] sollag, sindec, cosdec, anginc
K_COSZ = "phys_rrtm_dayparts_cosz"
K_SCATTER = "phys_rrtm_dayparts_scatter"
K_LEN = "phys_rrtm_dayparts_len"
KEY = "rrtm_dayparts"

_SCRATCH: dict = {}


def _scratch(n: int, jm: int):
    """Scratch and output arrays for n = jm*im points, allocated once per size."""
    import cupy as cp

    s = _SCRATCH.get(n)
    if s is None or s["ndayparts"].shape[0] != jm:
        s = {
            "coszen": cp.zeros(n, dtype=cp.float32),
            "coszdg": cp.zeros(n, dtype=cp.float32),
            "day": cp.zeros(n, dtype=cp.int32),
            "start": cp.zeros(n, dtype=cp.int32),
            "scan": cp.zeros(n, dtype=cp.int32),
            "dp_start": cp.zeros(n, dtype=cp.int32),
            "dp_len": cp.zeros(n, dtype=cp.int32),
            "dp_day": cp.zeros(n, dtype=cp.int32),
            "ndayparts": cp.zeros(jm, dtype=cp.int32),
        }
        _SCRATCH[n] = s
    return s


def dayparts(xlon, sinlat, coslat, solhr: float, dtswav: float, nrads: int,
             sol, im: int):
    """Run DAYPARTS row by row on n = JM*IM flattened points (row-major, IM points per row).

    xlon, sinlat, coslat are device float32 arrays of length n. Returns the scratch dict: coszen, coszdg,
    dp_start, dp_len, dp_day (slot layout, valid for k < ndayparts[j]), ndayparts[JM] (device).
    """
    import cupy as cp

    n = int(xlon.shape[0])
    if im < 1 or n % im != 0:
        raise ValueError(f"phys-rrtm-dayparts: n={n} is not a whole number of rows of im={im}")
    jm = n // im
    s = _scratch(n, jm)
    if n < 1:
        s["ndayparts"][:] = 0
        return s

    bs = kernels.BLOCK_COL[0]
    gs = kernels.grid_cols(n, 1)
    k_cosz = kernels.get(UNIT, K_COSZ)
    k_cosz((gs[0],), (bs,),
           (xlon, sinlat, coslat,
            np.float32(solhr), np.float32(dtswav), np.int32(nrads), sol,
            np.int32(n), np.int32(im), s["coszen"], s["coszdg"], s["day"], s["start"]))

    # Global inclusive scan of the run-start flags (CuPy library scan, device only). The row start is
    # always a run start, so scan[row start] - 1 is the run count before the row; the scatter subtracts it.
    s["scan"][...] = cp.cumsum(s["start"], dtype=cp.int32)
    k_scatter = kernels.get(UNIT, K_SCATTER)
    k_scatter((gs[0],), (bs,),
              (s["start"], s["scan"], s["day"], np.int32(n), np.int32(im),
               s["dp_start"], s["dp_day"]))

    k_len = kernels.get(UNIT, K_LEN)
    k_len((gs[0],), (bs,),
          (s["dp_start"], s["scan"], np.int32(n), np.int32(im), s["dp_len"], s["ndayparts"]))
    return s


def run(ctx):
    """Driver entry. Needs the solar astronomy state written by sol_update (phys-rrtm-sol-update) in
    ctx.st[SOL_STATE] = float32[4] sollag, sindec, cosdec, anginc (device); it raises if that is missing
    rather than guessing solar values. Lat/lon come from ctx.st GLAT/GLON (radians, H points, [JM, IM]).
    solhr is ctx.clock.solhr (the same value coszmn uses). Outputs are bound to ctx.tables["rrtm_dayparts"].
    """
    import cupy as cp

    if SOL_STATE not in ctx.st.fields:
        raise RuntimeError("phys-rrtm-dayparts needs SOL_STATE from sol_update "
                           "(phys-rrtm-sol-update) first; it is not guessed here")
    sol = ctx.st[SOL_STATE]
    # sin/cos of the latitude of THIS call's columns, every call (two elementwise kernels). A cache keyed on
    # id(ctx.st) went stale under the rrtm-call row chunks (one ChunkState object, new rows per chunk): every
    # chunk after the first used the first chunk's latitudes, so the day list drifted from coszmn's.
    glat = cp.asarray(ctx.st["GLAT"], dtype=cp.float32).reshape(-1)
    glon = cp.asarray(ctx.st["GLON"], dtype=cp.float32).reshape(-1)
    cache = {"xlon": glon, "sinlat": cp.sin(glat), "coslat": cp.cos(glat)}
    nrads = int(ctx.cfg.nrads)
    dtswav = float(nrads) * float(ctx.cfg.dt) / 3600.0
    im = int(ctx.grid.im)
    s = dayparts(cache["xlon"], cache["sinlat"], cache["coslat"],
                 float(ctx.clock.solhr), dtswav, nrads, sol, im)
    out = {"im": im, "jm": int(ctx.grid.jm), "coszen": s["coszen"], "coszdg": s["coszdg"],
           "dp_start": s["dp_start"], "dp_len": s["dp_len"], "dp_day": s["dp_day"],
           "ndayparts": s["ndayparts"]}
    ctx.tables[KEY] = out
    return out


def selftest(im: int = 954, jm: int = 835, seed: int = 0):
    """One launch on a full-size random but plausible state (CONTRACT section 6). Asserts no NaN/Inf
    and that the run table is row-consistent: each row's lengths sum to IM, every row starts at point 1,
    starts increase inside the row, and runs alternate in day within the row."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    n = im * jm
    lat = rng.uniform(np.deg2rad(20.0), np.deg2rad(55.0), n).astype(np.float32)
    lon = rng.uniform(np.deg2rad(-125.0), np.deg2rad(-65.0), n).astype(np.float32)
    xlon = cp.asarray(lon)
    sinlat = cp.asarray(np.sin(lat))
    coslat = cp.asarray(np.cos(lat))
    # Mid-day solar state, roughly 31 March: declination ~ +3 deg, equation of time ~ 0.
    sindec = float(np.sin(np.deg2rad(3.0)))
    cosdec = float(np.cos(np.deg2rad(3.0)))
    sol = cp.asarray(np.array([0.0, sindec, cosdec, 0.0], dtype=np.float32))   # SOL_STATE layout
    s = dayparts(xlon, sinlat, coslat, solhr=13.0, dtswav=48 * 25.0 / 3600.0, nrads=48,
                 sol=sol, im=im)
    nd = s["ndayparts"].get()
    starts = s["dp_start"].get().reshape(jm, im)
    lens = s["dp_len"].get().reshape(jm, im)
    days = s["dp_day"].get().reshape(jm, im)
    coszen = s["coszen"].get()
    coszdg = s["coszdg"].get()
    assert np.all(np.isfinite(coszen)) and np.all(np.isfinite(coszdg))
    assert np.all(nd >= 1), nd.min()
    daylit = 0
    for j in range(jm):
        k = int(nd[j])
        assert int(lens[j, :k].sum()) == im, (j, int(lens[j, :k].sum()))
        assert starts[j, 0] == 1 and np.all(np.diff(starts[j, :k]) > 0), j
        assert np.all(days[j, 1:k] != days[j, :k - 1]), j
        daylit += int(lens[j, :k][days[j, :k] == 1].sum())
    return {"ndayparts_max": int(nd.max()), "daylit_points": daylit, "n": n}
