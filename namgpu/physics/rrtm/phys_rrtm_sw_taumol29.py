# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-sw-taumol29: TAUMOL29, SW band 29 (820-2600 cm-1) optical depths and Rayleigh depth.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:4683-4767 (SUBROUTINE taumol29).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the sw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sw_taumol29.cu (rrtm_sw_taumol29, one thread per (i, j) column, one launch).
Tables: namgpu/data/rrtm_sw/kgb29.npz (module_radsw_kgb29_nmmb, from tools/extract-rrtm-sw-kgb29.py).

What it computes, per column and layer k (k = 1..NLAY, Fortran order):
  taur29(k)            = colmol(k) * rayl                              (all twelve g-points)
  taug29(g, k), k <= laytrop: absa interpolation in ln(p) and T, with the h2o self- and foreign-continuum,
                       plus colamt(:,k,2)*absco2(g) (CO2).
  taug29(g, k), k >  laytrop: absb interpolation weighted by colamt(:,k,2) (CO2 band), plus
                       colamt(:,k,1)*absh2o(g) (H2O).
  g = 1..12 are the band's g-points NS29+1 .. NS29+12 = 101 .. 112 (NS29 = 100).

Inputs (device, CONTRACT layout, Fortran (i,j,l) -> [l, j, i]), band-29 slices:
  colmol, fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac, id0, id1  [NLAY, JM, IM]
  indself, indfor (int32)                                                           [NLAY, JM, IM]
  colamt  [6, NLAY, JM, IM]  (Fortran colamt(i,k,n); only n = 1 (h2o) and n = 2 (co2) are read)
  laytrop [JM, IM] int32
Tables (device, Fortran column-major flattened): absa (65*12), absb (235*12), selfref (10*12),
forref (4*12), absh2o (12), absco2 (12).
Outputs (device, allocated once):
  taug29 [12, NLAY, JM, IM]   band-29 g-points (plane g-1 = Fortran taug(:,:,NS29+g))
  taur29 [NLAY, JM, IM]

Band 29 writes only its own twelve planes; the 112-plane taug of the Fortran is not allocated.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol29"
KERNEL = "rrtm_sw_taumol29"
NG29 = 12
MSA29, MSB29, MSF29, MFR29 = 65, 235, 10, 4
MAXGAS = 6

IN_KEY = "rrtm_sw_taumol29_in"
TAB_KEY = "rrtm_sw_taumol29_tables"
OUT_KEY = "rrtm_sw_taumol29_out"

TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb29.npz"

INPUT_3D = ("colmol", "fac00", "fac01", "fac10", "fac11", "selffac", "selffrac",
            "forfac", "forfrac", "indself", "indfor", "id0", "id1")


def load_tables() -> dict:
    """The kgb29 k-tables, uploaded once. Fortran column-major order is kept: the flat index of
    absa(ms, g) is (ms-1) + 65*(g-1), and so on."""
    import cupy as cp

    z = np.load(TABLE_FILE)
    tab = {}
    for name, shape in (("absa", (MSA29, NG29)), ("absb", (MSB29, NG29)),
                        ("selfref", (MSF29, NG29)), ("forref", (MFR29, NG29))):
        arr = np.asarray(z[name], dtype=np.float32)
        assert arr.shape == shape, f"kgb29 {name} shape {arr.shape}"
        tab[name] = cp.asarray(np.ravel(arr, order="F"))
    for name in ("absh2o", "absco2"):
        arr = np.asarray(z[name], dtype=np.float32)
        assert arr.shape == (NG29,), f"kgb29 {name} shape {arr.shape}"
        tab[name] = cp.asarray(arr)
    return tab


def alloc_outputs(nlay: int, jm: int, im: int) -> dict:
    import cupy as cp

    return {
        "taug29": cp.zeros((NG29, nlay, jm, im), dtype=cp.float32),
        "taur29": cp.zeros((nlay, jm, im), dtype=cp.float32),
    }


def _check_shapes(inp: dict, nlay: int, jm: int, im: int) -> None:
    for name in INPUT_3D:
        if inp[name].shape != (nlay, jm, im):
            raise ValueError(f"taumol29: {name} shape {inp[name].shape} != {(nlay, jm, im)}")
    if inp["colamt"].shape != (MAXGAS, nlay, jm, im):
        raise ValueError(f"taumol29: colamt shape {inp['colamt'].shape} != {(MAXGAS, nlay, jm, im)}")
    if inp["laytrop"].shape != (jm, im):
        raise ValueError(f"taumol29: laytrop shape {inp['laytrop'].shape} != {(jm, im)}")


def taumol29(inp: dict, tab: dict, out: dict, stream=None) -> dict:
    """One launch over every (i, j) column. All arrays are device arrays in CONTRACT layout."""
    nlay, jm, im = inp["colmol"].shape
    _check_shapes(inp, nlay, jm, im)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["colmol"], inp["colamt"], inp["laytrop"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["selffac"], inp["selffrac"], inp["indself"],
            inp["forfac"], inp["forfrac"], inp["indfor"],
            inp["id0"], inp["id1"],
            tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
            tab["absh2o"], tab["absco2"],
            np.int32(nlay), np.int32(im), np.int32(jm),
            out["taug29"], out["taur29"],
        ))
    return out


def run(ctx):
    """Driver entry. The SW driver puts the band-29 inputs in ctx.tables[IN_KEY] (the outputs of setcoef
    and the taumol driver's id0/id1). The tables load once; the outputs are allocated once."""
    inp = ctx.tables.get(IN_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_sw_taumol29 needs ctx.tables[{IN_KEY!r}] from the SW driver (phys-rrtm-sw-swrad)")
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    nlay, jm, im = inp["colmol"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None or out["taur29"].shape != (nlay, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taumol29(inp, tab, out, stream=ctx.stream)


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1) -> dict:
    """One launch on a full-size plausible random state (954 x 835 x 60). Asserts no NaN/Inf in any output.
    Index draws cover the Fortran ranges: id0/id1 over the whole band-29 table, indself 1..9 (selfref has
    10 rows), indfor 1..4 (forref has 4 rows, so the clamped neighbour is exercised), laytrop 1..nlay."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    f32 = np.float32
    shp = (nlay, jm, im)

    def u(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, shp).astype(f32))

    inp = {
        "colmol": cp.asarray(np.exp(rng.uniform(np.log(1e20), np.log(1e25), shp)).astype(f32)),
        "colamt": cp.asarray(rng.uniform(0.0, 1.0e24, (MAXGAS,) + shp).astype(f32)),
        "laytrop": cp.asarray(rng.integers(1, nlay + 1, (jm, im)).astype(np.int32)),
        "fac00": u(0.0, 1.0), "fac01": u(0.0, 1.0), "fac10": u(0.0, 1.0), "fac11": u(0.0, 1.0),
        "selffac": u(0.0, 1.0), "selffrac": u(0.0, 1.0),
        "forfac": u(0.0, 1.0), "forfrac": u(0.0, 1.0),
        "indself": cp.asarray(rng.integers(1, 10, shp).astype(np.int32)),
        "indfor": cp.asarray(rng.integers(1, 5, shp).astype(np.int32)),
        "id0": cp.asarray(rng.integers(0, MSB29, shp).astype(np.int32)),
        "id1": cp.asarray(rng.integers(0, MSB29, shp).astype(np.int32)),
    }
    tab = load_tables()
    out = alloc_outputs(nlay, jm, im)
    taumol29(inp, tab, out)
    cp.cuda.Stream.null.synchronize()

    for k, a in out.items():
        assert bool(cp.isfinite(a).all()), f"NaN/Inf in taumol29 output {k}"
    return {k: float(out[k].mean()) for k in out}
