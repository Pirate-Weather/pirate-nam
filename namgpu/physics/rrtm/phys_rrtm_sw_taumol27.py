"""phys-rrtm-sw-taumol27: TAUMOL27, SW band 27 (29000-38000 cm-1, low and high ozone) optical depths.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:4486-4554 (SUBROUTINE taumol27).
NOAA US government work; the sw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sw_taumol27.cu (rrtm_sw_taumol27, one thread per (i, j) column, one launch).

What it computes, per column and layer k (Fortran k = 1..nlay):
  taur(:, k, 87..94)  = colmol(k) * rayl(j)                                 (every layer)
  taug(:, k, 87..94)  = colamt(k, o3) * (ln(p) and T interpolation of the absa table, k < laytrop,
                                          or of the absb table, k >= laytrop)
  The g-points of band 27 are 0-based 86..93 of the 112-point array (NS27 = 86, NG27 = 8).
  No continuum and no species mixing in band 27. No solar source here: the band-27 solar flux is set by
  the SW driver, outside this routine.

Inputs (device, CONTRACT layout Fortran (i,j,l) -> [l, j, i]):
  ctx.tables["rrtm_sw_setcoef_out"]  (phys-rrtm-sw-setcoef): laytrop [JM, IM]; jp, jt, jt1 [nlay, JM, IM]
      int32; fac00, fac01, fac10, fac11 [nlay, JM, IM] float32.
  ctx.tables["rrtm_sw_columns"]  (SW driver): colmol [nlay, JM, IM]; colamt [MAXGAS=6, nlay, JM, IM]
      (plane 2 = o3 is read).
Tables (built once from namgpu/data/rrtm_sw/kgb27.npz, the module_radsw_kgb27_nmmb extract):
  absa [65, NG27=8], absb [235, 8], rayl [8].
Outputs: taug and taur, either [NG27=8, nlay, JM, IM] (default, plane j = band-27 g-point j+1) or the full
  [112, nlay, JM, IM] arrays of the SW driver, in which case plane0 = NS27 = 86 is passed.
  Only the NG27 planes of band 27 are written; the other planes belong to other band units.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol27"
KERNEL = "rrtm_sw_taumol27"
NG27 = 8
NS27 = 86
NGPT = 112
MAXGAS = 6
MSA27 = 65
MSB27 = 235
KGB27_NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb27.npz"

SETCOEF_OUT = "rrtm_sw_setcoef_out"
COLUMNS = "rrtm_sw_columns"
TAB_KEY = "rrtm_sw_taumol27_tables"
OUT_KEY = "rrtm_sw_taumol27_out"


def load_tables():
    """kgb27.npz -> device float32 arrays (absa, absb, rayl). Uploaded once."""
    import cupy as cp

    d = np.load(KGB27_NPZ)
    tab = {n: cp.asarray(np.ascontiguousarray(d[n], dtype=np.float32))
           for n in ("absa", "absb", "rayl")}
    if tab["absa"].shape != (MSA27, NG27) or tab["absb"].shape != (MSB27, NG27) \
            or tab["rayl"].shape != (NG27,):
        raise ValueError("taumol27: kgb27.npz shapes do not match module_radsw_kgb27_nmmb")
    return tab


def alloc_outputs(nlay: int, jm: int, im: int, nplanes: int = NG27):
    """taug and taur, allocated once. Every plane this unit owns is written on every call."""
    import cupy as cp

    return {
        "taug": cp.zeros((nplanes, nlay, jm, im), dtype=cp.float32),
        "taur": cp.zeros((nplanes, nlay, jm, im), dtype=cp.float32),
    }


def taumol27(inp: dict, out: dict, tab: dict, plane0: int = 0, stream=None):
    """One launch over every (i, j) column. inp/out/tab hold device arrays in CONTRACT layout."""
    nlay, jm, im = inp["fac00"].shape
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["colmol"], inp["colamt"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["jp"], inp["jt"], inp["jt1"], inp["laytrop"],
            tab["absa"], tab["absb"], tab["rayl"],
            np.int32(nlay), np.int32(im), np.int32(jm), np.int32(plane0),
            out["taug"], out["taur"],
        ))
    return out


def run(ctx):
    """Driver entry. Reads the setcoef outputs and the SW column state from ctx.tables and writes
    taug and taur for band 27 into ctx.tables[OUT_KEY] (NG27 planes, plane0 = 0)."""
    sout = ctx.tables.get(SETCOEF_OUT)
    cols = ctx.tables.get(COLUMNS)
    if sout is None or cols is None:
        raise RuntimeError("phys_rrtm_sw_taumol27 needs the setcoef outputs and the SW column state in ctx.tables")
    inp = {
        "colmol": cols["colmol"], "colamt": cols["colamt"],
        "fac00": sout["fac00"], "fac01": sout["fac01"], "fac10": sout["fac10"], "fac11": sout["fac11"],
        "jp": sout["jp"], "jt": sout["jt"], "jt1": sout["jt1"], "laytrop": sout["laytrop"],
    }
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    nlay, jm, im = inp["fac00"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None or out["taug"].shape != (NG27, nlay, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taumol27(inp, out, tab, plane0=0, stream=ctx.stream)


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1):
    """Compile, then one launch on the full 954 x 835 x 60 domain with a random but plausible state:
    real k-tables, pressure indices over the full range, laytrop drawn over 1..nlay.
    Asserts no NaN/Inf in taug and taur over every column, both branches included."""
    import cupy as cp

    kernels.module(UNIT).compile()
    tab = load_tables()
    rng = np.random.default_rng(seed)
    f32 = np.float32
    shp = (nlay, jm, im)
    inp = {
        "colmol": cp.asarray(rng.uniform(1.0e19, 1.0e25, shp).astype(f32)),
        "colamt": cp.asarray(rng.uniform(1.0e16, 1.0e22, (MAXGAS, nlay, jm, im)).astype(f32)),
        "fac00": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
        "fac01": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
        "fac10": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
        "fac11": cp.asarray(rng.uniform(0.0, 1.0, shp).astype(f32)),
        "jp": cp.asarray(rng.integers(1, 59, shp).astype(np.int32)),
        "jt": cp.asarray(rng.integers(1, 5, shp).astype(np.int32)),
        "jt1": cp.asarray(rng.integers(1, 5, shp).astype(np.int32)),
        "laytrop": cp.asarray(rng.integers(1, nlay + 1, (jm, im)).astype(np.int32)),
    }
    out = alloc_outputs(nlay, jm, im)
    taumol27(inp, out, tab, plane0=0)
    cp.cuda.Stream.null.synchronize()
    for k in ("taug", "taur"):
        assert bool(cp.isfinite(out[k]).all()), f"NaN/Inf in taumol27 output {k}"
    return {k: float(out[k].mean()) for k in out}
