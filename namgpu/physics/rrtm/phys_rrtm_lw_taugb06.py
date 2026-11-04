# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb06: RRTM LW band 6 optical depths and their Planck fractions (taumol group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:4602-4683, subroutine taugb06
(internal to taumol). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the header is kept in the kernel file
(namgpu/kernels/phys_rrtm_lw_taugb06.cu). Kernel: phys_rrtm_lw_taugb06, one thread per column.

Band 6 (820-980 cm-1): lower layers use H2O self and foreign continuum, the CO2 minor gas (with the
empirical CO2 adjustment above ratco2 = 3) and the CFC-11/CFC-12 minor terms; upper layers use the
CFC terms only. The band's 8 g-points are rows NS06 + ig of the 140-row taug / fracs (NS06 = 68).
This wrapper writes the 8-row band slab taug6[ig][k][col] and fracs6[ig][k][col]; the caller places the
slab at rows 68..75 of the full taug (the slab is contiguous in the [g][l][col] layout).

Tables: module_radlw_kgb06 from phys/radlw_datatb.f (the module the routine USEs), extracted by
tools/extract-rrtm-lw-kgb06.py to namgpu/data/rrtm_lw/kgb06.npz. The module_radlw_kgb06_nmmb block is not
used by any NAM routine and is not loaded here.

CO2 reference mixing ratios come from the module_radlw_ref table through rrtm_lw_ref.load_chi_mls()
(the real table, not the setcoef dict). The setcoef table is not read here.

Reads ctx.tables:
  "rrtm_lw_taumol_in"        dict of device inputs (CONTRACT layouts, see the kernel header):
      laytrop [jm, im] int32; jp, jt, jt1, indself, indfor, indminor [nlay, jm, im] int32;
      coldry [nlay, jm, im]; colamt [6, nlay, jm, im]; wx [4, nlay, jm, im];
      selffac, selffrac, forfac, forfrac, minorfrac, fac00, fac01, fac10, fac11 [nlay, jm, im] float32.
      Indices are the Fortran 1-based values; these are the keys phys-rrtm-lw-setcoef writes.
Writes ctx.tables["rrtm_lw_taugb06_out"] = {"taug6": [8, nlay, jm, im], "fracs6": [8, nlay, jm, im]},
allocated on first use if the caller did not provide it.

Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm.rrtm_lw_ref import load_chi_mls

UNIT = "phys_rrtm_lw_taugb06"
KERNEL = "phys_rrtm_lw_taugb06"
IN_KEY = "rrtm_lw_taumol_in"
TAB_KEY = "rrtm_lw_setcoef_tables"
OUT_KEY = "rrtm_lw_taugb06_out"
KGB_KEY = "rrtm_lw_kgb06"
CHI_KEY = "rrtm_lw_taugb06_chi_co2"   # (host chi array, device CO2 column) pair; see _co2_column()
NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb06.npz"

NG06 = 8        # module_radlw_parameters NG06 (radlw_param.f:107)
NS06 = 68       # module_radlw_parameters NS06 (radlw_param.f:114)
MAXGAS = 6      # module_radlw_parameters maxgas
MAXXSEC = 4     # module_radlw_parameters maxxsec
NREF = 59       # reference pressure levels (module_radlw_ref)
NGAS_REF = 7    # reference gases (module_radlw_ref chi_mls(59,7) in setcoef layout)


def load_kgb06():
    """Band-6 k-tables as device float32 arrays (uploaded once; host read is at init only)."""
    import cupy as cp

    with np.load(NPZ) as d:
        return {
            "absa": cp.asarray(d["absa"].astype(np.float32)),          # [65, 8]
            "selfref": cp.asarray(d["selfref"].astype(np.float32)),    # [10, 8]
            "forref": cp.asarray(d["forref"].astype(np.float32)),      # [4, 8]
            "ka_mco2": cp.asarray(d["ka_mco2"].astype(np.float32)),    # [19, 8]
            "fracrefa": cp.asarray(d["fracrefa"].astype(np.float32)),  # [8]
            "cfc11adj": cp.asarray(d["cfc11adj"].astype(np.float32)),  # [8]
            "cfc12": cp.asarray(d["cfc12"].astype(np.float32)),        # [8]
        }


def _check_shapes(inp, jm, im, nlay):
    if inp["laytrop"].shape != (jm, im):
        raise ValueError(f"taugb06: laytrop shape {inp['laytrop'].shape} != {(jm, im)}")
    for name in ("jp", "jt", "jt1", "indself", "indfor", "indminor", "coldry",
                 "selffac", "selffrac", "forfac", "forfrac", "minorfrac", "fac00", "fac01", "fac10", "fac11"):
        if inp[name].shape != (nlay, jm, im):
            raise ValueError(f"taugb06: {name} shape {inp[name].shape} != {(nlay, jm, im)}")
    if inp["colamt"].shape != (MAXGAS, nlay, jm, im):
        raise ValueError(f"taugb06: colamt shape {inp['colamt'].shape} != {(MAXGAS, nlay, jm, im)}")
    if inp["wx"].shape != (MAXXSEC, nlay, jm, im):
        raise ValueError(f"taugb06: wx shape {inp['wx'].shape} != {(MAXXSEC, nlay, jm, im)}")


def taugb06(inp: dict, chi_co2, kgb: dict, out: dict, nlay: int, jm: int, im: int, stream=None):
    """One launch over every (i, j) column. Arrays are device arrays in CONTRACT layout."""
    _check_shapes(inp, jm, im, nlay)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    ncol = im * jm
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(nlay), np.int32(ncol),
            inp["laytrop"], inp["jp"], inp["jt"], inp["jt1"], inp["indself"], inp["indfor"], inp["indminor"],
            inp["coldry"], inp["colamt"], inp["wx"],
            inp["selffac"], inp["selffrac"], inp["forfac"], inp["forfrac"], inp["minorfrac"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            chi_co2,
            kgb["absa"], kgb["selfref"], kgb["forref"], kgb["ka_mco2"],
            kgb["fracrefa"], kgb["cfc11adj"], kgb["cfc12"],
            out["taug6"], out["fracs6"],
        ))
    return out


def _co2_column(ctx):
    """Device CO2 column chi_mls[:, 1] (gas 2, [59] float32). Rebuilt whenever load_chi_mls() returns a
    different array object, so it can never go stale against the table."""
    import cupy as cp

    chi = load_chi_mls()
    cached = ctx.tables.get(CHI_KEY)
    if cached is None or cached[0] is not chi:
        cached = (chi, cp.asarray(np.ascontiguousarray(chi[:, 1])))
        ctx.tables[CHI_KEY] = cached
    return cached[1]


def run(ctx, stream=None):
    """Band-6 optical depths for the current taumol inputs; returns ctx.tables[OUT_KEY]."""
    import cupy as cp

    if ctx.tables is None:
        ctx.tables = {}
    inp = ctx.tables.get(IN_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_lw_taugb06 needs ctx.tables[{IN_KEY!r}] from the taumol driver")

    nlay, jm, im = inp["jp"].shape
    kgb = ctx.tables.get(KGB_KEY)
    if kgb is None:
        kgb = load_kgb06()
        ctx.tables[KGB_KEY] = kgb
    chi_co2 = _co2_column(ctx)

    out = ctx.tables.get(OUT_KEY)
    if out is None:
        out = {"taug6": cp.zeros((NG06, nlay, jm, im), dtype=cp.float32),
               "fracs6": cp.zeros((NG06, nlay, jm, im), dtype=cp.float32)}
        ctx.tables[OUT_KEY] = out
    return taugb06(inp, chi_co2, kgb, out, nlay, jm, im, stream=stream)


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1, rows: int = 32):
    """One full-size pass: the whole im x jm x nlay domain in row bands of `rows` (the full inputs do not
    fit beside other tenants on the shared card; the kernel is per column, so the bands are independent).
    Random but in-range state: jp 1..12 (ind1 <= 65), jt/jt1 1..3, indself 1..9, indfor 1..3,
    indminor 1..18 (ka_mco2 row <= 19), laytrop over its whole range 0..nlay, and the real chi_mls and
    k-tables. Asserts no NaN/Inf in taug6 and fracs6 and returns the value ranges."""
    import time

    import cupy as cp

    kernels.module(UNIT).compile()
    kgb = load_kgb06()
    chi_co2 = cp.asarray(np.ascontiguousarray(load_chi_mls()[:, 1]))
    rng = np.random.default_rng(seed)
    f32 = np.float32
    nbad = 0
    tmin, tmax = np.inf, -np.inf
    t0 = time.perf_counter()
    for j0 in range(0, jm, rows):
        jc = min(rows, jm - j0)
        shp = (nlay, jc, im)
        inp = {
            "laytrop": cp.asarray(rng.integers(0, nlay + 1, (jc, im)).astype(np.int32)),
            "jp": cp.asarray(rng.integers(1, 13, shp).astype(np.int32)),
            "jt": cp.asarray(rng.integers(1, 4, shp).astype(np.int32)),
            "jt1": cp.asarray(rng.integers(1, 4, shp).astype(np.int32)),
            "indself": cp.asarray(rng.integers(1, 10, shp).astype(np.int32)),
            "indfor": cp.asarray(rng.integers(1, 4, shp).astype(np.int32)),
            "indminor": cp.asarray(rng.integers(1, 19, shp).astype(np.int32)),
            "coldry": cp.asarray(rng.uniform(1.0e20, 1.0e24, shp).astype(f32)),
            "colamt": cp.asarray(rng.uniform(1.0e18, 1.0e24, (MAXGAS,) + shp).astype(f32)),
            "wx": cp.asarray(rng.uniform(0.0, 1.0e18, (MAXXSEC,) + shp).astype(f32)),
            "selffac": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
            "selffrac": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
            "forfac": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
            "forfrac": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
            "minorfrac": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
            "fac00": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
            "fac01": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
            "fac10": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
            "fac11": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
        }
        out = {"taug6": cp.zeros((NG06, nlay, jc, im), dtype=cp.float32),
               "fracs6": cp.zeros((NG06, nlay, jc, im), dtype=cp.float32)}
        taugb06(inp, chi_co2, kgb, out, nlay, jc, im)
        for k in ("taug6", "fracs6"):
            nbad += int(out[k].size - int(cp.isfinite(out[k]).sum()))
        tmin = min(tmin, float(out["taug6"].min()))
        tmax = max(tmax, float(out["taug6"].max()))
    cp.cuda.Stream.null.synchronize()
    assert nbad == 0, f"taugb06 selftest: {nbad} non-finite values"
    return {"nonfinite": nbad, "taug6_min": tmin, "taug6_max": tmax, "seconds": time.perf_counter() - t0}
