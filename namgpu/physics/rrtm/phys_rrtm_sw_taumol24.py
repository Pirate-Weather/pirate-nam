"""phys-rrtm-sw-taumol24: TAUMOL24, SW band 24 (12850-16000 cm-1) optical depths and Rayleigh extinction.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:4277-4381 (SUBROUTINE taumol24).
The NOAA/AER sw-rrtm3 header of that file is the notice in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_sw_taumol24.cu, phys_rrtm_sw_taumol24 (one thread per (i, j) column,
one launch). Per column and layer k: for k <= laytrop the H2O/O2 binary-species interpolation of the
absa table, the O3 term, the H2O self and foreign continuum, and the Rayleigh extinction from rayla;
for k > laytrop the O2 column times the absb table, the O3 term and the Rayleigh extinction raylb.
Written g-point planes: 0-based 66..73 (NS24 = 66, NG24 = 8) of taug and taur.

Inputs, from the setcoef unit and the SW gas/cldprop unit (device float32, CONTRACT layout):
  ctx.tables["rrtm_sw_colamt"]   [6, NLAY, JM, IM]   column amounts of the six gases
  ctx.tables["rrtm_sw_colmol"]   [NLAY, JM, IM]      dry air + H2O column (Fortran colmol)
  ctx.tables["rrtm_sw_setcoef_out"]: laytrop [JM, IM] int32; jp, jt, jt1, indself, indfor [NLAY, JM, IM]
      int32; fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac [NLAY, JM, IM] float32
      (phys-rrtm-sw-setcoef, alloc_outputs layout)
Tables (built once from namgpu/data/rrtm_sw/kgb24.npz, the module_radsw_kgb24_nmmb extract):
  absa [8, 585], absb [8, 235], selfref [8, 10], forref [8, 3], rayla [9, 8], raylb [8],
  abso3a [8], abso3b [8]
Outputs: ctx.tables["rrtm_sw_taug"] and ctx.tables["rrtm_sw_taur"], both [112, NLAY, JM, IM] float32,
  allocated once and shared by the taumol units (each unit writes its own g-point planes; a re-allocation
  would erase the other units' planes, so it happens only when the shape changes).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol24"
KERNEL = "phys_rrtm_sw_taumol24"
NG24 = 8
NS24 = 66
NGPTSW = 112
MAXGAS = 6
MSA24, MSB24, MSF24, MFR24, MFX24 = 585, 235, 10, 3, 9
KGB24_NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb24.npz"

SETCOEF_OUT = "rrtm_sw_setcoef_out"
COLAMT_KEY = "rrtm_sw_colamt"
COLMOL_KEY = "rrtm_sw_colmol"
TAB_KEY = "rrtm_sw_taumol24_tables"
TAUG_KEY = "rrtm_sw_taug"
TAUR_KEY = "rrtm_sw_taur"


def load_tables():
    """kgb24.npz -> device float32 arrays in the kernel layouts. Uploaded once."""
    import cupy as cp

    d = np.load(KGB24_NPZ)
    shapes = {
        "absa": (NG24, MSA24), "absb": (NG24, MSB24), "selfref": (NG24, MSF24),
        "forref": (NG24, MFR24), "rayla": (MFX24, NG24), "raylb": (NG24,),
        "abso3a": (NG24,), "abso3b": (NG24,),
    }
    tab = {}
    for n, shp in shapes.items():
        a = np.ascontiguousarray(d[n], dtype=np.float32)
        if a.shape != shp:
            raise ValueError(f"taumol24: kgb24.npz {n} has shape {a.shape}, expected {shp}")
        tab[n] = cp.asarray(a)
    return tab


def alloc_outputs(nlay: int, jm: int, im: int):
    import cupy as cp

    return {
        "taug": cp.zeros((NGPTSW, nlay, jm, im), dtype=cp.float32),
        "taur": cp.zeros((NGPTSW, nlay, jm, im), dtype=cp.float32),
    }


def taumol24(inp: dict, out: dict, tab: dict, stream=None):
    """One launch over every (i, j) column. inp/out/tab hold device arrays in CONTRACT layout."""
    nlay, jm, im = inp["fac00"].shape
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["colamt"], inp["colmol"],
            inp["jp"], inp["jt"], inp["jt1"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["selffac"], inp["selffrac"], inp["indself"],
            inp["forfac"], inp["forfrac"], inp["indfor"],
            inp["laytrop"],
            tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
            tab["rayla"], tab["raylb"], tab["abso3a"], tab["abso3b"],
            out["taug"], out["taur"],
            np.int32(nlay), np.int32(jm), np.int32(im),
        ))
    return out


def run(ctx):
    """Driver entry. Reads the gas amounts, colmol and the setcoef outputs from ctx.tables and writes
    the band-24 planes of taug and taur into ctx.tables."""
    so = ctx.tables.get(SETCOEF_OUT)
    colamt = ctx.tables.get(COLAMT_KEY)
    colmol = ctx.tables.get(COLMOL_KEY)
    if so is None or colamt is None or colmol is None:
        raise RuntimeError("phys_rrtm_sw_taumol24 needs ctx.tables[rrtm_sw_setcoef_out, rrtm_sw_colamt, rrtm_sw_colmol]")
    inp = {
        "colamt": colamt, "colmol": colmol,
        "jp": so["jp"], "jt": so["jt"], "jt1": so["jt1"],
        "fac00": so["fac00"], "fac01": so["fac01"], "fac10": so["fac10"], "fac11": so["fac11"],
        "selffac": so["selffac"], "selffrac": so["selffrac"], "indself": so["indself"],
        "forfac": so["forfac"], "forfrac": so["forfrac"], "indfor": so["indfor"],
        "laytrop": so["laytrop"],
    }
    nlay, jm, im = inp["fac00"].shape
    if tuple(colamt.shape) != (MAXGAS, nlay, jm, im) or tuple(colmol.shape) != (nlay, jm, im):
        raise ValueError("phys_rrtm_sw_taumol24: colamt or colmol shape does not match setcoef outputs")
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    taug = ctx.tables.get(TAUG_KEY)
    taur = ctx.tables.get(TAUR_KEY)
    if taug is None or taur is None or tuple(taug.shape) != (NGPTSW, nlay, jm, im) \
            or tuple(taur.shape) != (NGPTSW, nlay, jm, im):
        out = alloc_outputs(nlay, jm, im)
        taug, taur = out["taug"], out["taur"]
        ctx.tables[TAUG_KEY] = taug
        ctx.tables[TAUR_KEY] = taur
    return taumol24(inp, {"taug": taug, "taur": taur}, tab, stream=ctx.stream)


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1, rows: int = 8):
    """Compile, then one pass over the full 954 x 835 x 60 domain in row bands of `rows`.

    Full-size taug and taur (112 planes each) are about 21 GiB apiece, too large for the shared card at
    once. The kernel is per column, so row bands are independent and the band-24 output is the same as a
    single full-domain launch. Random but plausible state: layer indices in their setcoef ranges, laytrop
    random in 1..nlay. Asserts no NaN/Inf in the eight written planes (66..73) of taug and taur."""
    import cupy as cp

    kernels.module(UNIT).compile()
    tab = load_tables()
    rng = np.random.default_rng(seed)
    f32 = np.float32
    nbad = 0
    for j0 in range(0, jm, rows):
        jc = min(rows, jm - j0)
        inp = {
            "colamt": cp.asarray(rng.uniform(1.0e18, 3.0e23, (MAXGAS, nlay, jc, im)).astype(f32)),
            "colmol": cp.asarray(rng.uniform(1.0e20, 1.0e25, (nlay, jc, im)).astype(f32)),
            "jp": cp.asarray(rng.integers(1, 59, (nlay, jc, im)).astype(np.int32)),
            "jt": cp.asarray(rng.integers(1, 5, (nlay, jc, im)).astype(np.int32)),
            "jt1": cp.asarray(rng.integers(1, 5, (nlay, jc, im)).astype(np.int32)),
            "fac00": cp.asarray(rng.uniform(0.0, 1.0, (nlay, jc, im)).astype(f32)),
            "fac01": cp.asarray(rng.uniform(0.0, 1.0, (nlay, jc, im)).astype(f32)),
            "fac10": cp.asarray(rng.uniform(0.0, 1.0, (nlay, jc, im)).astype(f32)),
            "fac11": cp.asarray(rng.uniform(0.0, 1.0, (nlay, jc, im)).astype(f32)),
            "selffac": cp.asarray(rng.uniform(0.0, 1.0e3, (nlay, jc, im)).astype(f32)),
            "selffrac": cp.asarray(rng.uniform(0.0, 1.0, (nlay, jc, im)).astype(f32)),
            "indself": cp.asarray(rng.integers(1, 10, (nlay, jc, im)).astype(np.int32)),
            "forfac": cp.asarray(rng.uniform(0.0, 1.0e3, (nlay, jc, im)).astype(f32)),
            "forfrac": cp.asarray(rng.uniform(0.0, 1.0, (nlay, jc, im)).astype(f32)),
            "indfor": cp.asarray(rng.integers(1, 3, (nlay, jc, im)).astype(np.int32)),
            "laytrop": cp.asarray(rng.integers(1, nlay + 1, (jc, im)).astype(np.int32)),
        }
        out = alloc_outputs(nlay, jc, im)
        taumol24(inp, out, tab)
        for name in ("taug", "taur"):
            planes = out[name][NS24:NS24 + NG24]
            nbad += int(cp.count_nonzero(~cp.isfinite(planes)).item())
    if nbad:
        raise RuntimeError(f"phys_rrtm_sw_taumol24 selftest: {nbad} NaN/Inf values in planes 66..73")
    return {"nan_inf": nbad, "domain": (im, jm, nlay)}
