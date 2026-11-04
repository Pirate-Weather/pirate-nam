# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb13: TAUGB13, LW band 13 optical depths and Planck fractions (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:5666-5908 (SUBROUTINE taugb13).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the lw-rrtm3 header is kept in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_lw_taugb13.cu (rrtm_lw_taugb13, one thread per (i, j) column, one launch).

What it computes, per column: for each layer k the band-13 optical depth of the four g-points
(taug) and the Planck fraction of each g-point (fracs), band-local rows 1..4 (taugb13 rows NS13+1..NS13+4,
NS13 = 130; scatter_band13 places them in the full gpoint arrays). Lower atmosphere (k <= laytrop) uses the binary-species (H2O/CO2 with
their interpolation factors) k-table absa, the self and foreign continuum, the CO2 and CO minor-gas
tables and the CO2 adjustment; the upper atmosphere uses the O3 minor-gas table kb_mo3 only.

Inputs (device arrays in CONTRACT layout, Fortran (i,j,l) -> [l, j, i]):
  from the setcoef output dict (phys-rrtm-lw-setcoef, ctx.tables["rrtm_lw_setcoef_out"]):
    laytrop [JM, IM] int32; jp, jt, jt1, indself, indfor, indminor [nlay, JM, IM] int32;
    rfrate [2, 6, nlay, JM, IM]; fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac,
    minorfrac [nlay, JM, IM]
  plus, from the caller: colamt [7, nlay, JM, IM] and coldry [nlay, JM, IM].
    colamt planes 1..7 are the Fortran colamt(k,1..7) of radlw_main_nmmb.f: h2o, co2, o3, n2o, ch4,
    o2, co. Plane 7 (CO) is max(0, coldry*gasvmr(co)) when ilwrgas > 0 and 0 otherwise
    (radlw_main_nmmb.f:711 and :723). The Fortran maxgas is 7 (module_radlw_parameters, radlw_param.f:97).
Tables: chi_mls [59, 7] from ctx.tables["rrtm_lw_setcoef_tables"]; the band-13 k-tables from
namgpu/data/rrtm_lw/kgb13.npz (uploaded once into ctx.tables[TAB_KEY]).
Outputs (device float32, allocated once by alloc_outputs): taug13 and fracs13, band-local [4, nlay, JM, IM].
scatter_band13 writes them into the full gpoint arrays [nlay, NGPTLW=140, ...] at rows NS13..NS13+3
(zero-based), leaving every other row to its own band.

Notes:
  - the k-tables come from module_radlw_kgb13 in radlw_datatb.f, the module taugb13 USEs. The
    module_radlw_kgb13_nmmb block lacks the minor-gas tables. See tools/extract-rrtm-lw-kgb13.py.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb13"
KERNEL = "rrtm_lw_taugb13"
NG13 = 4
NS13 = 130                 # module_radlw_parameters NS13 (radlw_param.f:116)
NGPTLW = 140               # module_radlw_parameters NGPTLW (radlw_param.f:95)
NREF = 59
NGAS_REF = 7
NGAS_LW = 7                # Fortran maxgas (module_radlw_parameters, radlw_param.f:97)
NRATES = 6
NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb13.npz"
TAB_KEY = "rrtm_lw_taugb13_tables"
SETCOEF_TAB_KEY = "rrtm_lw_setcoef_tables"
OUT_KEY = "rrtm_lw_taugb13_out"

TABLE_KEYS = ("absa", "selfref", "forref", "fracrefa", "fracrefb", "ka_mco2", "ka_mco", "kb_mo3")
TABLE_SHAPES = {
    "absa": (585, NG13), "selfref": (10, NG13), "forref": (NG13, NG13), "fracrefa": (9, NG13),
    "fracrefb": (NG13,), "ka_mco2": (19, 9, NG13), "ka_mco": (19, 9, NG13), "kb_mo3": (19, NG13),
}
# The kernel reads the index inputs as int32 and the rest as float32. A wrong dtype of the right
# shape would be misread silently, so both are checked here.
INT_INPUTS = ("laytrop", "jp", "jt", "jt1", "indself", "indfor", "indminor")
FLOAT_INPUTS = ("coldry", "colamt", "rfrate", "fac00", "fac01", "fac10", "fac11", "selffac",
                "selffrac", "forfac", "forfrac", "minorfrac")


def alloc_outputs(nlay: int, jm: int, im: int):
    """Band-local outputs, allocated once: taug13 and fracs13 are [NG13, nlay, JM, IM]."""
    import cupy as cp

    return {
        "taug13": cp.zeros((NG13, nlay, jm, im), dtype=cp.float32),
        "fracs13": cp.zeros((NG13, nlay, jm, im), dtype=cp.float32),
    }


def scatter_band13(out: dict, taug_full, fracs_full):
    """Place band-13 rows into the full g-point arrays [nlay, NGPTLW, ...] at rows NS13..NS13+NG13-1.

    The full arrays may be a column chunk (any trailing shape that matches the band-local arrays), so
    a full-domain taumol can reuse this at a memory size that fits. One device copy each, no host sync.
    """
    tr = (1, 0) + tuple(range(2, out["taug13"].ndim))
    taug_full[:, NS13:NS13 + NG13] = out["taug13"].transpose(tr)
    fracs_full[:, NS13:NS13 + NG13] = out["fracs13"].transpose(tr)
    return taug_full, fracs_full


def _load_host(path: Path = NPZ) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        out = {k: np.ascontiguousarray(z[k], dtype=np.float32) for k in TABLE_KEYS}
    for k, shp in TABLE_SHAPES.items():
        if out[k].shape != shp:
            raise ValueError(f"{UNIT}: {k} has shape {out[k].shape}, expected {shp}")
    return out


def run(ctx):
    """Upload the band-13 k-tables into ctx.tables once; later calls return the cached dict."""
    import cupy as cp

    t = ctx.tables
    if TAB_KEY not in t:
        host = _load_host()
        t[TAB_KEY] = {k: cp.asarray(host[k]) for k in TABLE_KEYS}
    return t[TAB_KEY]


def _check_shapes(inp: dict, chi_mls, nlay: int, jm: int, im: int):
    if inp["laytrop"].shape != (jm, im):
        raise ValueError(f"{UNIT}: laytrop shape {inp['laytrop'].shape} != {(jm, im)}")
    for name in ("jp", "jt", "jt1", "indself", "indfor", "indminor", "coldry", "fac00", "fac01",
                 "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac", "minorfrac"):
        if inp[name].shape != (nlay, jm, im):
            raise ValueError(f"{UNIT}: {name} shape {inp[name].shape} != {(nlay, jm, im)}")
    if inp["colamt"].shape != (NGAS_LW, nlay, jm, im):
        raise ValueError(f"{UNIT}: colamt shape {inp['colamt'].shape} != {(NGAS_LW, nlay, jm, im)}")
    if inp["rfrate"].shape != (2, NRATES, nlay, jm, im):
        raise ValueError(f"{UNIT}: rfrate shape {inp['rfrate'].shape}")
    if chi_mls.shape != (NREF, NGAS_REF):
        raise ValueError(f"{UNIT}: chi_mls shape {chi_mls.shape} != {(NREF, NGAS_REF)}")
    for name in INT_INPUTS:
        if str(inp[name].dtype) != "int32":
            raise ValueError(f"{UNIT}: {name} dtype {inp[name].dtype} must be int32")
    for name in FLOAT_INPUTS:
        if str(inp[name].dtype) != "float32":
            raise ValueError(f"{UNIT}: {name} dtype {inp[name].dtype} must be float32")


def taugb13(inp: dict, chi_mls, tab: dict, out: dict, stream=None):
    """One launch over every (i, j) column. Arrays are device float32/int32 in CONTRACT layout.

    inp: the setcoef outputs (laytrop, jp, jt, jt1, rfrate, fac00..fac11, selffac, selffrac, indself,
         forfac, forfrac, indfor, minorfrac, indminor) plus colamt [7, ...] and coldry.
    chi_mls: device float32 [59, 7] (setcoef table).
    tab: the dict returned by run(ctx).
    out: the dict returned by alloc_outputs.
    """
    nlay = inp["coldry"].shape[0]
    jm, im = inp["coldry"].shape[1], inp["coldry"].shape[2]
    _check_shapes(inp, chi_mls, nlay, jm, im)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["laytrop"], inp["colamt"], inp["coldry"],
            inp["jp"], inp["jt"], inp["jt1"], inp["rfrate"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["selffac"], inp["selffrac"], inp["indself"],
            inp["forfac"], inp["forfrac"], inp["indfor"],
            inp["minorfrac"], inp["indminor"],
            chi_mls,
            tab["absa"], tab["selfref"], tab["forref"], tab["fracrefa"], tab["fracrefb"],
            tab["ka_mco2"], tab["ka_mco"], tab["kb_mo3"],
            np.int32(nlay), np.int32(im), np.int32(jm),
            out["taug13"], out["fracs13"],
        ))
    return out


if __name__ == "__main__":
    from types import SimpleNamespace

    ctx = SimpleNamespace(tables={}, stream=None)
    tab = run(ctx)
    for k in sorted(tab):
        print(k, tab[k].shape, tab[k].dtype)
