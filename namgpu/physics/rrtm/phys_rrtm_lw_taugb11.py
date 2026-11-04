# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb11: LW band 11 optical depths and Planck fractions (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:5391-5478 (SUBROUTINE taugb11).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the lw-rrtm3 package header is kept as the notice in the kernel file
(namgpu/kernels/phys_rrtm_lw_taugb11.cu). Kernel: phys_rrtm_lw_taugb11, one thread per column.

What it computes, per column and layer k (1-based Fortran):
  taug(NS11+ig, k) = colamt(k,1) * (interpolated band-11 absorption, lower or upper reference table)
                     + self continuum + foreign continuum + minor gas O2 term
  fracs(NS11+ig, k) = fracrefa(ig) (lower) or fracrefb(ig) (upper)
  for ig = 1..NG11 = 8. Lower layers are k <= laytrop (plog > 4.56), upper layers the rest.

Tables (device float32, C order reversed from Fortran module_radlw_kgb11, radlw_datatb.f:28426-29089):
  absa [65, 8], absb [235, 8], selfref [10, 8], forref [4, 8], fracrefa [8], fracrefb [8],
  ka_mo2 [19, 8], kb_mo2 [19, 8]. Extracted once by tools/extract_rrtm_lw_kgb11.py into
  namgpu/data/rrtm_lw/kgb11.npz and uploaded on first use into ctx.tables["rrtm_lw_kgb11"].

Inputs (ctx.tables, from the LW chain):
  ctx.tables["rrtm_lw_setcoef_out"]  laytrop [JM, IM]; jp, jt, jt1, indself, indfor, indminor,
                                     fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac,
                                     minorfrac, scaleminor  [nlay, JM, IM]   (phys-rrtm-lw-setcoef)
  ctx.tables["rrtm_lw_setcoef_in"]["colamt"]  [MAXGAS=6, nlay, JM, IM]

Output: written into the caller's buffers out["taug"], out["fracs"], each [R, nlay, JM, IM] float32,
with out["row0"] = the 0-based row of band-11 g-point 1. Band 11 writes rows row0 .. row0+7.
  Full-width caller (the lwrad/taumol owner, all 140 g-point rows): R = 140, row0 = NS11 = 114.
  Band-only buffer: R = 8, row0 = 0.
The full 140-row taug and fracs at 954 x 835 x 60 are about 26.8 GB each in float32, so the caller
must chunk columns or keep band-only buffers. This unit does not allocate those arrays.

Launch: one kernel launch on ctx.stream (grid_cols), no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb11"
KERNEL = "phys_rrtm_lw_taugb11"
NG11 = 8        # radlw_param_nmmb.f:154 (NG11 = 08)
NS11 = 114      # radlw_param_nmmb.f:163 (NS11 = 114), 0-based begin index in the 140 g-point rows
MAXGAS = 6
MSA11, MSB11, MSF11, MFR11, MMO11 = 65, 235, 10, 4, 19

TAB_KEY = "rrtm_lw_kgb11"
IN_KEY = "rrtm_lw_setcoef_in"
SETCOEF_OUT_KEY = "rrtm_lw_setcoef_out"
OUT_KEY = "rrtm_lw_taugb11_out"
DATA = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb11.npz"

TABLE_SHAPES = {
    "absa": (MSA11, NG11), "absb": (MSB11, NG11), "selfref": (MSF11, NG11),
    "forref": (MFR11, NG11), "fracrefa": (NG11,), "fracrefb": (NG11,),
    "ka_mo2": (MMO11, NG11), "kb_mo2": (MMO11, NG11),
}


def load_host() -> dict:
    """Band-11 k-tables from the extracted npz; shapes and dtype checked, values finite."""
    with np.load(DATA) as d:
        tab = {k: np.asarray(d[k]) for k in TABLE_SHAPES}
    for k, shape in TABLE_SHAPES.items():
        if tab[k].shape != shape or tab[k].dtype != np.float32:
            raise ValueError(f"{UNIT}: {k} is {tab[k].shape} {tab[k].dtype}, want {shape} float32")
        if not np.isfinite(tab[k]).all():
            raise ValueError(f"{UNIT}: non-finite values in {k}")
    return tab


def _device_tables(ctx):
    import cupy as cp

    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = {k: cp.asarray(v) for k, v in load_host().items()}
        ctx.tables[TAB_KEY] = tab
    return tab


def taugb11(inp: dict, tab: dict, out: dict, stream=None):
    """One launch over every (i, j) column. inp, tab, out hold device arrays in CONTRACT layout."""
    laytrop = inp["laytrop"]
    jm, im = laytrop.shape
    nlay = inp["jp"].shape[0]
    for name in ("jp", "jt", "jt1", "indself", "indfor", "indminor", "fac00", "fac01", "fac10",
                 "fac11", "selffac", "selffrac", "forfac", "forfrac", "minorfrac", "scaleminor"):
        if inp[name].shape != (nlay, jm, im):
            raise ValueError(f"taugb11: {name} shape {inp[name].shape} != {(nlay, jm, im)}")
    if inp["colamt"].shape != (MAXGAS, nlay, jm, im):
        raise ValueError(f"taugb11: colamt shape {inp['colamt'].shape}")
    rows = out["taug"].shape[0]
    row0 = int(out["row0"])
    if out["taug"].shape != (rows, nlay, jm, im) or out["fracs"].shape != (rows, nlay, jm, im):
        raise ValueError(f"taugb11: taug/fracs must be [R, {nlay}, {jm}, {im}]")
    if row0 < 0 or row0 + NG11 > rows:
        raise ValueError(f"taugb11: rows {row0}..{row0 + NG11 - 1} outside R={rows}")

    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            laytrop, inp["jp"], inp["jt"], inp["jt1"], inp["indself"], inp["indfor"], inp["indminor"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["selffac"], inp["selffrac"], inp["forfac"], inp["forfrac"],
            inp["minorfrac"], inp["scaleminor"], inp["colamt"],
            tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
            tab["fracrefa"], tab["fracrefb"], tab["ka_mo2"], tab["kb_mo2"],
            out["taug"], out["fracs"],
            np.int32(nlay), np.int32(im), np.int32(jm), np.int32(row0),
        ))
    return out


def run(ctx):
    """Driver entry. Needs the setcoef outputs and inputs (phys-rrtm-lw-setcoef) and a caller-owned
    taug/fracs buffer in ctx.tables["rrtm_lw_taugb11_out"] (see the module docstring for shapes)."""
    sc_out = ctx.tables.get(SETCOEF_OUT_KEY)
    sc_in = ctx.tables.get(IN_KEY)
    if sc_out is None or sc_in is None:
        raise RuntimeError(f"phys_rrtm_lw_taugb11 needs ctx.tables[{SETCOEF_OUT_KEY!r}] and "
                           f"ctx.tables[{IN_KEY!r}] from phys-rrtm-lw-setcoef")
    out = ctx.tables.get(OUT_KEY)
    if out is None:
        raise RuntimeError(f"phys_rrtm_lw_taugb11 needs ctx.tables[{OUT_KEY!r}] with taug, fracs, row0 "
                           "allocated by the LW taumol owner")
    inp = dict(sc_out)
    inp["colamt"] = sc_in["colamt"]
    tab = _device_tables(ctx)
    return taugb11(inp, tab, out, stream=ctx.stream)


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1):
    """One launch on a full-size random state with valid RRTM index ranges, band-only output buffers.
    Asserts no NaN or Inf in taug and fracs and that every written row is non-trivial."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    f32 = np.float32
    shape = (nlay, jm, im)
    laytrop_h = rng.integers(nlay // 2, nlay // 2 + 10, size=(jm, im)).astype(np.int32)
    k = np.arange(1, nlay + 1).reshape(nlay, 1, 1)
    lower = k <= laytrop_h[None, :, :]

    # Index ranges as produced by phys-rrtm-lw-setcoef (1-based Fortran values).
    jp = np.where(lower, rng.integers(1, 13, shape), rng.integers(13, 59, shape)).astype(np.int32)
    jt = rng.integers(1, 5, shape).astype(np.int32)
    jt1 = rng.integers(1, 5, shape).astype(np.int32)
    indself = np.where(lower, rng.integers(1, 10, shape), 0).astype(np.int32)
    indfor = np.where(lower, rng.integers(1, 3, shape), 3).astype(np.int32)
    indminor = rng.integers(1, 19, shape).astype(np.int32)

    def frac():
        return rng.uniform(0.0, 1.0, shape).astype(f32)

    colamt = rng.uniform(1.0e18, 1.0e24, (MAXGAS,) + shape).astype(f32)
    inp = {
        "laytrop": cp.asarray(laytrop_h),
        "jp": cp.asarray(jp), "jt": cp.asarray(jt), "jt1": cp.asarray(jt1),
        "indself": cp.asarray(indself), "indfor": cp.asarray(indfor), "indminor": cp.asarray(indminor),
        "fac00": cp.asarray(frac()), "fac01": cp.asarray(frac()),
        "fac10": cp.asarray(frac()), "fac11": cp.asarray(frac()),
        "selffac": cp.asarray(rng.uniform(1.0e20, 1.0e23, shape).astype(f32)),
        "selffrac": cp.asarray(frac()),
        "forfac": cp.asarray(rng.uniform(1.0e20, 1.0e23, shape).astype(f32)),
        "forfrac": cp.asarray(frac()),
        "minorfrac": cp.asarray(frac()),
        "scaleminor": cp.asarray(rng.uniform(1.0e-3, 1.0, shape).astype(f32)),
        "colamt": cp.asarray(colamt),
    }
    tab = {k_: cp.asarray(v) for k_, v in load_host().items()}
    out = {
        "taug": cp.zeros((NG11,) + shape, dtype=cp.float32),
        "fracs": cp.zeros((NG11,) + shape, dtype=cp.float32),
        "row0": 0,
    }
    taugb11(inp, tab, out)
    cp.cuda.Stream.null.synchronize()
    for name in ("taug", "fracs"):
        assert bool(cp.isfinite(out[name]).all()), f"NaN/Inf in {name}"
    for g in range(NG11):
        assert float(cp.abs(out["taug"][g]).max()) > 0.0, f"taug row {g} is all zero"
    return {"taug_mean": float(out["taug"].mean()), "fracs_mean": float(out["fracs"].mean())}


if __name__ == "__main__":
    print(UNIT, "selftest", selftest())
