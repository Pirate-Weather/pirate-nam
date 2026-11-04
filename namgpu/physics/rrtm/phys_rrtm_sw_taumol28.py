"""phys-rrtm-sw-taumol28: TAUMOL28, SW band 28 (38000-50000 cm-1, O3 and O2) optical depths and Rayleigh.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:4559-4678 (internal subroutine
taumol28 of taumol). NOAA US government work; the sw-rrtm3 header of that file is kept as the notice in
the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sw_taumol28.cu (rrtm_sw_taumol28, one thread per (i, j) column, one launch).

What it computes, per column and layer k (1-based Fortran k): taur(g, k) = colmol(k) * rayl for the six
band-28 g-points (rayl = 2.02e-05); for k <= laytrop the lower-atmosphere band-28 optical depth from the
O3/O2 binary species interpolation of absa (specmult = 8); for k >= laytrop and k > MINVAL(laytrop) the
upper-atmosphere form from absb (specmult = 4). Where both blocks reach the same layer, the upper form is
kept, as in the Fortran. Band 28 has no self- or foreign-continuum term. The planes written are the six
band-28 g-point planes of taug and taur, NS28 = 94 .. 99 of the driver's 112-plane arrays.

Inputs (ctx.tables, device arrays, CONTRACT layout):
  rrtm_sw_colamt     float32 [6, NLAY, JM, IM]   gas column amounts (cldprop/gas unit; dispatcher input)
  rrtm_sw_colmol     float32 [NLAY, JM, IM]      total column amount (from the SW driver)
  rrtm_sw_setcoef_out: laytrop [JM, IM] int32; jp, jt, jt1 [NLAY, JM, IM] int32;
      fac00, fac01, fac10, fac11 [NLAY, JM, IM] float32 (phys-rrtm-sw-setcoef)
Tables (built once from namgpu/data/rrtm_sw/kgb28.npz, the module_radsw_kgb28_nmmb extract):
  absa [6, 585], absb [6, 1175], rayl (2.02e-05), strrat28 (strrat(28) of module_radsw_sflux_nmmb)
Outputs: ctx.tables["rrtm_sw_taumol28_out"] = {"taug", "taur"}, float32 [NG28, NLAY, JM, IM] each,
allocated once. The driver passes its 112-plane arrays' view taug[NS28:NS28+NG28] (and taur likewise),
so the kernel's plane 0..5 are the band-28 planes.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol28"
KERNEL = "rrtm_sw_taumol28"
NG28 = 6
NS28 = 94
NGPTSW = 112
KGB28_NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb28.npz"

SETCOEF_OUT = "rrtm_sw_setcoef_out"
COLAMT_KEY = "rrtm_sw_colamt"
COLMOL_KEY = "rrtm_sw_colmol"
TAB_KEY = "rrtm_sw_taumol28_tables"
OUT_KEY = "rrtm_sw_taumol28_out"


def load_tables():
    """kgb28.npz -> device float32 tables plus the rayl and strrat(28) scalars. Uploaded once."""
    import cupy as cp

    d = np.load(KGB28_NPZ)
    tab = {n: cp.asarray(np.ascontiguousarray(d[n], dtype=np.float32)) for n in ("absa", "absb")}
    if tab["absa"].shape != (NG28, 585) or tab["absb"].shape != (NG28, 1175):
        raise ValueError("taumol28: kgb28.npz shapes do not match module_radsw_kgb28_nmmb")
    tab["rayl"] = np.float32(d["rayl"])
    tab["strrat28"] = np.float32(d["strrat28"])
    return tab


def alloc_outputs(nlay: int, jm: int, im: int):
    """taug and taur for the six band-28 planes, allocated once."""
    import cupy as cp

    return {
        "taug": cp.zeros((NG28, nlay, jm, im), dtype=cp.float32),
        "taur": cp.zeros((NG28, nlay, jm, im), dtype=cp.float32),
    }


def taumol28(inp: dict, out: dict, tab: dict, stream=None):
    """One launch over every (i, j) column. inp/out/tab hold device arrays in CONTRACT layout."""
    import cupy as cp

    nlay, jm, im = inp["colmol"].shape
    # MINVAL(laytrop) as a device int32 scalar, so the kernel reads it without a host sync.
    kmin = cp.min(inp["laytrop"]).astype(cp.int32).reshape(1)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["colamt"], inp["colmol"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["laytrop"], inp["jp"], inp["jt"], inp["jt1"],
            kmin,
            tab["absa"], tab["absb"],
            tab["rayl"], tab["strrat28"],
            np.int32(nlay), np.int32(im), np.int32(jm),
            out["taug"], out["taur"],
        ))
    return out


def run(ctx):
    """Driver entry. Reads colamt, colmol (ctx.tables) and the setcoef outputs; writes the band-28
    planes of taug and taur into ctx.tables[OUT_KEY]."""
    sout = ctx.tables.get(SETCOEF_OUT)
    colamt = ctx.tables.get(COLAMT_KEY)
    colmol = ctx.tables.get(COLMOL_KEY)
    if sout is None or colamt is None or colmol is None:
        raise RuntimeError("phys_rrtm_sw_taumol28 needs setcoef outputs, rrtm_sw_colamt and rrtm_sw_colmol in ctx.tables")
    inp = {
        "colamt": colamt, "colmol": colmol,
        "laytrop": sout["laytrop"],
        "jp": sout["jp"], "jt": sout["jt"], "jt1": sout["jt1"],
        "fac00": sout["fac00"], "fac01": sout["fac01"], "fac10": sout["fac10"], "fac11": sout["fac11"],
    }
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    nlay, jm, im = inp["colmol"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None:
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taumol28(inp, out, tab, stream=getattr(ctx, "stream", None))
