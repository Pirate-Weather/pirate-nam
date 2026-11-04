# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb12: TAUGB12, LW band 12 (1800-2080 cm-1) optical depths and Planck fractions.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:5482-5662 (SUBROUTINE taugb12).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the lw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_lw_taugb12.cu (rrtm_lw_taugb12, one thread per (i, j) column, one launch).

What it computes, per column: for the lower atmosphere (layers k = 1..laytrop of the column) the
band-12 g-point optical depths taug(122+ig, k) (8 g-points) from the water vapour / CO2 binary
species interpolation of the absa table, plus the self and foreign continuum terms, and the Planck
fractions fracs(122+ig, k); for the upper atmosphere (k = laytrop+1..nlay) both are zero.

Inputs come from the setcoef units (the LW driver fills them; this unit reads them):
  ctx.tables["rrtm_lw_setcoef_in"]["colamt"]       [MAXGAS=6, nlay, JM, IM]
  ctx.tables["rrtm_lw_setcoef_out"]: rfrate [2, 6, nlay, JM, IM], jp, jt, jt1, fac00, fac01, fac10,
      fac11, selffac, selffrac, indself, forfac, forfrac, indfor [nlay, JM, IM] (int32 for indices),
      laytrop [JM, IM] (int32)
  ctx.tables["rrtm_lw_setcoef_tables"]["chi_mls"]  [59, 7] (module_radlw_ref)
Tables (this unit, built once from namgpu/data/rrtm_lw/kgb12.npz, the module_radlw_kgb12 extract):
  absa [585, NG12=8], selfref [10, 8], forref [4, 8], fracrefa [9, 8]
Outputs (this unit, allocated once): ctx.tables["rrtm_lw_taugb12_out"] = {"taug", "fracs"},
  both [NGPTLW=140, nlay, JM, IM] float32. Only planes 123..130 (1-based, ns12 = 122) are written by
  this unit; the other g-points belong to the other band units.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb12"
KERNEL = "rrtm_lw_taugb12"
NG12 = 8
NS12 = 122
NGPTLW = 140
MAXGAS = 6
NRATES = 6
KGB12_NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb12.npz"

SETCOEF_IN = "rrtm_lw_setcoef_in"
SETCOEF_OUT = "rrtm_lw_setcoef_out"
SETCOEF_TAB = "rrtm_lw_setcoef_tables"
TAB_KEY = "rrtm_lw_taugb12_tables"
OUT_KEY = "rrtm_lw_taugb12_out"


def load_tables():
    """kgb12.npz -> device float32 arrays (absa, selfref, forref, fracrefa). Uploaded once."""
    import cupy as cp

    d = np.load(KGB12_NPZ)
    tab = {n: cp.asarray(np.ascontiguousarray(d[n], dtype=np.float32))
           for n in ("absa", "selfref", "forref", "fracrefa")}
    if tab["absa"].shape != (585, NG12) or tab["selfref"].shape != (10, NG12) \
            or tab["forref"].shape != (4, NG12) or tab["fracrefa"].shape != (9, NG12):
        raise ValueError("taugb12: kgb12.npz shapes do not match module_radlw_kgb12")
    return tab


def alloc_outputs(nlay: int, jm: int, im: int):
    """taug and fracs, allocated once. Every plane this unit owns is written on every call."""
    import cupy as cp

    return {
        "taug": cp.zeros((NGPTLW, nlay, jm, im), dtype=cp.float32),
        "fracs": cp.zeros((NGPTLW, nlay, jm, im), dtype=cp.float32),
    }


def taugb12(inp: dict, out: dict, tab: dict, stream=None):
    """One launch over every (i, j) column. inp/out/tab hold device arrays in CONTRACT layout."""
    nlay, jm, im = inp["fac00"].shape
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["colamt"], inp["rfrate"],
            inp["jp"], inp["jt"], inp["jt1"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["selffac"], inp["selffrac"], inp["indself"],
            inp["forfac"], inp["forfrac"], inp["indfor"],
            inp["laytrop"],
            tab["absa"], tab["selfref"], tab["forref"], tab["fracrefa"],
            inp["chi_mls"],
            np.int32(nlay), np.int32(im), np.int32(jm),
            out["taug"], out["fracs"],
        ))
    return out


def run(ctx):
    """Driver entry. Reads the setcoef inputs and outputs and the chi_mls table from ctx.tables,
    writes taug and fracs into ctx.tables[OUT_KEY]."""
    sin = ctx.tables.get(SETCOEF_IN)
    sout = ctx.tables.get(SETCOEF_OUT)
    stab = ctx.tables.get(SETCOEF_TAB)
    if sin is None or sout is None or stab is None:
        raise RuntimeError("phys_rrtm_lw_taugb12 needs the setcoef inputs, outputs and tables in ctx.tables")
    inp = {
        "colamt": sin["colamt"],
        "rfrate": sout["rfrate"],
        "jp": sout["jp"], "jt": sout["jt"], "jt1": sout["jt1"],
        "fac00": sout["fac00"], "fac01": sout["fac01"], "fac10": sout["fac10"], "fac11": sout["fac11"],
        "selffac": sout["selffac"], "selffrac": sout["selffrac"], "indself": sout["indself"],
        "forfac": sout["forfac"], "forfrac": sout["forfrac"], "indfor": sout["indfor"],
        "laytrop": sout["laytrop"],
        "chi_mls": stab["chi_mls"],
    }
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    nlay, jm, im = inp["fac00"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None or out["taug"].shape != (NGPTLW, nlay, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taugb12(inp, out, tab, stream=ctx.stream)


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1, rows: int = 32):
    """One full-size pass: compile, then the whole 954 x 835 x 60 domain in row bands of `rows`.
    The full taug and fracs are 140 x 60 x 835 x 954 x 4 B = 26.7 GB each, too large for the shared
    card, so the band runs per row band (the kernel is per column, so the bands are independent; the
    LW driver works per column batch the same way). Random but plausible state, real k-tables and chi_mls.
    Asserts no NaN/Inf in taug or fracs and that the upper-atmosphere planes are exactly zero."""
    import time

    import cupy as cp

    kernels.module(UNIT).compile()
    tab = load_tables()
    chi = np.load(KGB12_NPZ.parent / "kgb03.npz")["chi_mls"].astype(np.float32)   # (59, 7), module_radlw_ref
    chi_d = cp.asarray(chi)
    rng = np.random.default_rng(seed)
    f32 = np.float32
    nbad = 0
    t0 = time.perf_counter()
    for j0 in range(0, jm, rows):
        jc = min(rows, jm - j0)
        inp = {
            "colamt": cp.asarray(rng.uniform(1.0e18, 3.0e23, (MAXGAS, nlay, jc, im)).astype(f32)),
            "rfrate": cp.asarray(rng.uniform(0.5, 2.0, (2, NRATES, nlay, jc, im)).astype(f32)),
            "jp": cp.asarray(rng.integers(1, 12, (nlay, jc, im)).astype(np.int32)),
            "jt": cp.asarray(rng.integers(1, 5, (nlay, jc, im)).astype(np.int32)),
            "jt1": cp.asarray(rng.integers(1, 5, (nlay, jc, im)).astype(np.int32)),
            "fac00": cp.asarray(rng.uniform(0, 1, (nlay, jc, im)).astype(f32)),
            "fac01": cp.asarray(rng.uniform(0, 1, (nlay, jc, im)).astype(f32)),
            "fac10": cp.asarray(rng.uniform(0, 1, (nlay, jc, im)).astype(f32)),
            "fac11": cp.asarray(rng.uniform(0, 1, (nlay, jc, im)).astype(f32)),
            "selffac": cp.asarray(rng.uniform(0, 1e-2, (nlay, jc, im)).astype(f32)),
            "selffrac": cp.asarray(rng.uniform(0, 1, (nlay, jc, im)).astype(f32)),
            "indself": cp.asarray(rng.integers(1, 10, (nlay, jc, im)).astype(np.int32)),
            "forfac": cp.asarray(rng.uniform(0, 1e-2, (nlay, jc, im)).astype(f32)),
            "forfrac": cp.asarray(rng.uniform(0, 1, (nlay, jc, im)).astype(f32)),
            "indfor": cp.asarray(rng.integers(1, 4, (nlay, jc, im)).astype(np.int32)),
            "laytrop": cp.asarray(rng.integers(30, 56, (jc, im)).astype(np.int32)),
            "chi_mls": chi_d,
        }
        out = alloc_outputs(nlay, jc, im)
        taugb12(inp, out, tab)
        cp.cuda.Stream.null.synchronize()
        for name in ("taug", "fracs"):
            nbad += int((~cp.isfinite(out[name])).sum())
        # upper-atmosphere planes: layers k > laytrop of each column must be exactly zero
        k_idx = cp.arange(nlay, dtype=cp.int32)[:, None, None]
        upper = k_idx >= inp["laytrop"][None, :, :]
        for name in ("taug", "fracs"):
            band = out[name][NS12:NS12 + NG12]
            nbad += int((cp.abs(band) * upper[None]).sum() > 0)
    dt = time.perf_counter() - t0
    print(f"taugb12 selftest: {im}x{jm}x{nlay}, bands of {rows} rows, {dt:.2f} s, "
          f"non-finite or upper-nonzero values: {nbad}")
    if nbad:
        raise SystemExit(1)


if __name__ == "__main__":
    selftest()
