"""phys-rrtm-sw-taumol25: TAUMOL25, SW band 25 (16000-22650 cm-1) optical depths and Rayleigh scattering.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:4386-4445 (internal subroutine taumol25
of taumol). NOAA US government work; the sw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sw_taumol25.cu (rrtm_sw_taumol25, one thread per (i, j) column, one launch).

What it computes, per column and layer k (1-based Fortran k): taur(g, k) = colmol(k) * rayl(g) for the six
band-25 g-points; for k <= laytrop the H2O absorption interpolated from absa over the dispatcher index
id0/id1 (rrtm_sw_taumol_index, jp/jt/jt1 from setcoef), plus colamt(k,3) * abso3a; for k >= laytrop and
k > kmin the upper-atmosphere form colamt(k,3) * abso3b. Where both blocks reach the same layer the upper
form is kept, as in the Fortran. kmin is the domain-wide MINVAL(laytrop); the Fortran uses the 8-column
tile minimum. That overlap is a declared divergence: see the kernel header.

Output layout: taug and taur are [NG25 = 6, nlay, JM, IM] float32, so this unit allocates 6 planes
(1.15 GB each at 954 x 835 x 60), not the 112-plane driver array. Plane p is Fortran g-point NS25+1+p, i.e.
planes 74..79 (0-based) of the 112-plane layout. Nothing in the tree reads the buffer yet, so it is keyed to
this unit (OUT_KEY) and does not share the taumol dispatcher's 112-plane array.

Inputs (ctx.tables, device arrays, CONTRACT layout):
  rrtm_sw_colamt     float32 [6, NLAY, JM, IM]   gas column amounts (cldprop/gas unit; dispatcher input)
  rrtm_sw_colmol     float32 [NLAY, JM, IM]      total column amount (from the SW driver)
  rrtm_sw_setcoef_out: laytrop [JM, IM] int32; jp, jt, jt1 [NLAY, JM, IM] int32;
      fac00, fac01, fac10, fac11 [NLAY, JM, IM] float32 (phys-rrtm-sw-setcoef)
Tables (built once from namgpu/data/rrtm_sw/kgb25.npz, the module_radsw_kgb25_nmmb extract):
  absa [6, 65], rayl [6], abso3a [6], abso3b [6]
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol25"
KERNEL = "rrtm_sw_taumol25"
NG25 = 6
MSA25 = 65
NS25 = 74          # Fortran offset of this unit's g-points in the 112-plane layout (documentation only)
NGPTSW = 112
KGB25_NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb25.npz"

SETCOEF_OUT = "rrtm_sw_setcoef_out"
COLAMT_KEY = "rrtm_sw_colamt"
COLMOL_KEY = "rrtm_sw_colmol"
TAB_KEY = "rrtm_sw_taumol25_tables"
OUT_KEY = "rrtm_sw_taumol25_out"
KMIN_KEY = "rrtm_sw_taumol25_kmin"


def load_tables():
    """kgb25.npz -> device float32 tables. Uploaded once."""
    import cupy as cp

    d = np.load(KGB25_NPZ)
    tab = {n: cp.asarray(np.ascontiguousarray(d[n], dtype=np.float32))
           for n in ("absa", "rayl", "abso3a", "abso3b")}
    if tab["absa"].shape != (NG25, MSA25) or tab["rayl"].shape != (NG25,) \
            or tab["abso3a"].shape != (NG25,) or tab["abso3b"].shape != (NG25,):
        raise ValueError("taumol25: kgb25.npz shapes do not match module_radsw_kgb25_nmmb")
    return tab


def alloc_outputs(nlay: int, jm: int, im: int, nplanes: int = NG25):
    """taug and taur, allocated once: this unit's NG25 = 6 planes only."""
    import cupy as cp

    return {
        "taug": cp.zeros((nplanes, nlay, jm, im), dtype=cp.float32),
        "taur": cp.zeros((nplanes, nlay, jm, im), dtype=cp.float32),
    }


def taumol25(inp: dict, out: dict, tab: dict, kmin_buf, stream=None):
    """One launch over every (i, j) column. inp/out/tab hold device arrays in CONTRACT layout.
    out taug/taur are [NG25, nlay, JM, IM]; plane p is Fortran g-point NS25+1+p."""
    import cupy as cp

    nlay, jm, im = inp["colmol"].shape
    if out["taug"].shape[0] != NG25 or out["taur"].shape != out["taug"].shape:
        raise ValueError(f"taumol25: taug/taur must have {NG25} planes, got {out['taug'].shape}")
    # MINVAL(laytrop) as a device int32 scalar, written into a preallocated buffer (no host sync).
    kmin_buf[...] = cp.min(inp["laytrop"]).astype(cp.int32).reshape(1)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["colamt"], inp["colmol"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["laytrop"], inp["jp"], inp["jt"], inp["jt1"],
            kmin_buf,
            tab["absa"], tab["abso3a"], tab["abso3b"], tab["rayl"],
            np.int32(nlay), np.int32(im), np.int32(jm),
            out["taug"], out["taur"],
        ))
    return out


def run(ctx):
    """Driver entry. Reads colamt, colmol (ctx.tables) and the setcoef outputs; writes this unit's six
    taug and taur planes (Fortran planes 74..79) into ctx.tables[OUT_KEY]."""
    import cupy as cp

    sout = ctx.tables.get(SETCOEF_OUT)
    colamt = ctx.tables.get(COLAMT_KEY)
    colmol = ctx.tables.get(COLMOL_KEY)
    if sout is None or colamt is None or colmol is None:
        raise RuntimeError("phys_rrtm_sw_taumol25 needs setcoef outputs, rrtm_sw_colamt and rrtm_sw_colmol in ctx.tables")
    inp = {
        "colamt": colamt, "colmol": colmol,
        "laytrop": sout["laytrop"], "jp": sout["jp"], "jt": sout["jt"], "jt1": sout["jt1"],
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
    kmin_buf = ctx.tables.get(KMIN_KEY)
    if kmin_buf is None:
        kmin_buf = cp.zeros((1,), dtype=cp.int32)
        ctx.tables[KMIN_KEY] = kmin_buf
    return taumol25(inp, out, tab, kmin_buf, stream=getattr(ctx, "stream", None))
