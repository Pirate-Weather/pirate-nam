# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb15: RRTM LW band 15 optical depths and Planck fractions (taumol group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:5980-6184, subroutine taugb15
(internal to taumol). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the header is kept in the kernel file
(namgpu/kernels/phys_rrtm_lw_taugb15.cu). Kernel: phys_rrtm_lw_taugb15, one thread per column.

Band 15 (2380-2600 cm-1): lower layers use the N2 and CO2 binary-species interpolation (the rfrate
ratio), the water vapour self and foreign continuum, and the N2 minor term; upper layers are zero.
The band's 2 g-points are rows NS15 + ig of the 140-row taug / fracs (NS15 = 136). This wrapper writes
the 2-row band slab taug15[ig][k][col] and fracs15[ig][k][col]; the caller places the slab at rows
136..137 of the full taug (the slab is contiguous in the [g][l][col] layout).

Tables: module_radlw_kgb15 from phys/radlw_datatb.f (the module the routine USEs at line 5988), extracted
by tools/extract-rrtm-lw-kgb15.py to namgpu/data/rrtm_lw/kgb15.npz. The module_radlw_kgb15_nmmb block
of radlw_datatb_nmmb.f has no ka_mn2 and is not the routine's table source.

Reads ctx.tables:
  "rrtm_lw_taumol_in"        dict of device inputs (CONTRACT layouts, see the kernel header):
      laytrop [jm, im] int32; jp, jt, jt1, indself, indfor, indminor [nlay, jm, im] int32;
      colamt [6, nlay, jm, im]; rfrate [2, 6, nlay, jm, im] (setcoef layout, rfrate(k,m,n));
      colbrd, scaleminor, selffac, selffrac, forfac, forfrac, minorfrac, fac00, fac01, fac10, fac11
      [nlay, jm, im] float32. Indices are the Fortran 1-based values; these are the keys
      phys-rrtm-lw-setcoef writes (rfrate, colbrd, scaleminor, selffac ... from its outputs).
  "rrtm_lw_setcoef_tables"   the setcoef table dict; chi_mls [59, 7] (module_radlw_ref, row = level,
      column = gas) supplies the N2/CO2 reference ratio refrat = chi_mls(4,1)/chi_mls(2,1).
Writes ctx.tables["rrtm_lw_taugb15_out"] = {"taug15": [2, nlay, jm, im], "fracs15": [2, nlay, jm, im]},
allocated on first use if the caller did not provide it.

Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation after the first call.
The refrat scalar and the k-tables are computed once at first use and cached in ctx.tables.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb15"
KERNEL = "phys_rrtm_lw_taugb15"
IN_KEY = "rrtm_lw_taumol_in"
TAB_KEY = "rrtm_lw_setcoef_tables"
OUT_KEY = "rrtm_lw_taugb15_out"
KGB_KEY = "rrtm_lw_kgb15"
REF_KEY = "rrtm_lw_taugb15_refrat"
NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb15.npz"

NG15 = 2        # module_radlw_parameters NG15 (radlw_param.f:109)
NS15 = 136      # module_radlw_parameters NS15 (radlw_param.f:116)
NSPA15 = 9      # radlw_main_nmmb.f:284, nspa(15)
MAXGAS = 6      # module_radlw_parameters maxgas
NRATES = 6      # radlw_param.f:99, NRATES
NREF = 59       # reference pressure levels (module_radlw_ref)
NGAS_REF = 7    # reference gases (module_radlw_ref chi_mls(7,59) in setcoef layout)


def load_kgb15():
    """Band-15 k-tables as device float32 arrays (uploaded once; host read is at init only)."""
    import cupy as cp

    with np.load(NPZ) as d:
        return {
            "absa": cp.asarray(d["absa"].astype(np.float32)),          # [585, 2]
            "selfref": cp.asarray(d["selfref"].astype(np.float32)),    # [10, 2]
            "forref": cp.asarray(d["forref"].astype(np.float32)),      # [4, 2]
            "ka_mn2": cp.asarray(d["ka_mn2"].astype(np.float32)),      # [19, 9, 2]
            "fracrefa": cp.asarray(d["fracrefa"].astype(np.float32)),  # [9, 2]
        }


def _check_shapes(inp, jm, im, nlay):
    if inp["laytrop"].shape != (jm, im):
        raise ValueError(f"taugb15: laytrop shape {inp['laytrop'].shape} != {(jm, im)}")
    for name in ("jp", "jt", "jt1", "indself", "indfor", "indminor", "colbrd", "scaleminor",
                 "selffac", "selffrac", "forfac", "forfrac", "minorfrac", "fac00", "fac01", "fac10", "fac11"):
        if inp[name].shape != (nlay, jm, im):
            raise ValueError(f"taugb15: {name} shape {inp[name].shape} != {(nlay, jm, im)}")
    if inp["colamt"].shape != (MAXGAS, nlay, jm, im):
        raise ValueError(f"taugb15: colamt shape {inp['colamt'].shape} != {(MAXGAS, nlay, jm, im)}")
    if inp["rfrate"].shape != (2, NRATES, nlay, jm, im):
        raise ValueError(f"taugb15: rfrate shape {inp['rfrate'].shape} != {(2, NRATES, nlay, jm, im)}")


def taugb15(inp: dict, refrat: float, kgb: dict, out: dict, nlay: int, jm: int, im: int, stream=None):
    """One launch over every (i, j) column. Arrays are device arrays in CONTRACT layout."""
    _check_shapes(inp, jm, im, nlay)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    ncol = im * jm
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(nlay), np.int32(ncol),
            np.float32(refrat), np.float32(refrat),
            inp["laytrop"], inp["jp"], inp["jt"], inp["jt1"], inp["indself"], inp["indfor"], inp["indminor"],
            inp["colamt"], inp["rfrate"], inp["colbrd"], inp["scaleminor"],
            inp["selffac"], inp["selffrac"], inp["forfac"], inp["forfrac"], inp["minorfrac"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            kgb["absa"], kgb["selfref"], kgb["forref"], kgb["ka_mn2"], kgb["fracrefa"],
            out["taug15"], out["fracs15"],
        ))
    return out


def run(ctx, stream=None):
    """Band-15 optical depths for the current taumol inputs; returns ctx.tables[OUT_KEY]."""
    import cupy as cp

    if ctx.tables is None:
        ctx.tables = {}
    inp = ctx.tables.get(IN_KEY)
    tab = ctx.tables.get(TAB_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_lw_taugb15 needs ctx.tables[{IN_KEY!r}] from the taumol driver")
    if tab is None or "chi_mls" not in tab:
        raise RuntimeError(f"phys_rrtm_lw_taugb15 needs ctx.tables[{TAB_KEY!r}]['chi_mls'] (module_radlw_ref)")

    nlay, jm, im = inp["jp"].shape
    if tab["chi_mls"].shape != (NREF, NGAS_REF):
        raise ValueError(f"taugb15: chi_mls shape {tab['chi_mls'].shape} != {(NREF, NGAS_REF)}")

    kgb = ctx.tables.get(KGB_KEY)
    if kgb is None:
        kgb = load_kgb15()
        ctx.tables[KGB_KEY] = kgb

    refrat = ctx.tables.get(REF_KEY)
    if refrat is None:
        # chi_mls(gas, level) in Fortran is chi_mls[level-1, gas-1] here: chi_mls(4,1) = [0, 3], chi_mls(2,1) = [0, 1].
        chi_row = cp.asnumpy(tab["chi_mls"][0]).astype(np.float32)
        refrat = float(np.float32(chi_row[3]) / np.float32(chi_row[1]))
        ctx.tables[REF_KEY] = refrat

    out = ctx.tables.get(OUT_KEY)
    if out is None:
        out = {"taug15": cp.zeros((NG15, nlay, jm, im), dtype=cp.float32),
               "fracs15": cp.zeros((NG15, nlay, jm, im), dtype=cp.float32)}
        ctx.tables[OUT_KEY] = out
    return taugb15(inp, refrat, kgb, out, nlay, jm, im, stream=stream)
