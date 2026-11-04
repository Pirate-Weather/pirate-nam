"""phys-rrtm-sw-taumol26: TAUMOL26, SW band 26 (22650-29000 cm-1) optical depths and Rayleigh depths.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:4450-4481 (SUBROUTINE taumol26).
NOAA US government work; the sw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sw_taumol26.cu (rrtm_sw_taumol26, one thread per (i, j) column, one launch).
Tables: namgpu/data/rrtm_sw/kgb26.npz (module_radsw_kgb26_nmmb, from tools/extract-rrtm-sw-kgb26.py).

What it computes, per column and layer k (k = 1..NLAY, Fortran order), for each of the NG26 = 6 g-points j:
  taug26(j, k) = 0                      (band 26 has no gas absorption: the routine is a zero fill)
  taur26(j, k) = colmol(k) * rayl(j)    (Rayleigh scattering, a different coefficient per g-point)
g-point j = 1..6 is the band's planes NS26+1 .. NS26+6 = 81 .. 86 (NS26 = 80).

Inputs (device, CONTRACT layout, Fortran (i,j,l) -> [l, j, i]):
  colmol  [NLAY, JM, IM] float32   column amount of dry air (from the setcoef / gas unit)
Outputs (device, band-local, allocated once):
  taug26  [NG26 = 6, NLAY, JM, IM] float32, plane g-1 = Fortran taug(:,:,NS26+g)
  taur26  [NG26 = 6, NLAY, JM, IM] float32, plane g-1 = Fortran taur(:,:,NS26+g)

The 112-plane taug and taur of the Fortran (21 GB each at 954 x 835 x 60) are not allocated; this band
writes only its six planes. The spectral-sum unit maps g to NS26+g. This is the same convention as the
taumol22 unit.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol26"
KERNEL = "rrtm_sw_taumol26"
NG26 = 6

IN_KEY = "rrtm_sw_taumol26_in"
TAB_KEY = "rrtm_sw_taumol26_tables"
OUT_KEY = "rrtm_sw_taumol26_out"

TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb26.npz"


def load_tables() -> dict:
    """rayl(NG26) from kgb26.npz, uploaded once."""
    import cupy as cp

    z = np.load(TABLE_FILE)
    rayl = np.asarray(z["rayl"], dtype=np.float32)
    assert rayl.shape == (NG26,), f"kgb26 rayl shape {rayl.shape}"
    return {"rayl": cp.asarray(rayl)}


def alloc_outputs(nlay: int, jm: int, im: int) -> dict:
    import cupy as cp

    return {
        "taug26": cp.zeros((NG26, nlay, jm, im), dtype=cp.float32),
        "taur26": cp.zeros((NG26, nlay, jm, im), dtype=cp.float32),
    }


def _check_colmol(colmol) -> tuple[int, int, int]:
    """The raw kernel reads colmol by index, so dtype, rank and layout are checked here (a wrong one
    would be read silently wrong)."""
    import cupy as cp

    if not isinstance(colmol, cp.ndarray):
        raise TypeError(f"taumol26: colmol must be a CuPy device array, got {type(colmol).__name__}")
    if colmol.dtype != cp.float32:
        raise TypeError(f"taumol26: colmol must be float32 (CONTRACT section 1), got {colmol.dtype}")
    if colmol.ndim != 3:
        raise ValueError(f"taumol26: colmol must be [NLAY, JM, IM] (3-D), got shape {colmol.shape}")
    if not colmol.flags.c_contiguous:
        raise ValueError("taumol26: colmol must be C-contiguous in [NLAY, JM, IM] order")
    nlay, jm, im = colmol.shape
    if nlay < 1 or jm < 1 or im < 1:
        raise ValueError(f"taumol26: empty colmol shape {colmol.shape}")
    return nlay, jm, im


def taumol26(inp: dict, tab: dict, out: dict, stream=None) -> dict:
    """One launch over every (i, j) column. All arrays are device arrays in CONTRACT layout."""
    colmol = inp["colmol"]
    nlay, jm, im = _check_colmol(colmol)
    rayl = tab["rayl"]
    if tuple(rayl.shape) != (NG26,) or str(rayl.dtype) != "float32":
        raise ValueError(f"taumol26: rayl must be float32 shape ({NG26},), got {rayl.dtype} {tuple(rayl.shape)}")
    if tuple(out["taug26"].shape) != (NG26, nlay, jm, im) or tuple(out["taur26"].shape) != (NG26, nlay, jm, im):
        raise ValueError(f"taumol26: output shape does not match colmol {(nlay, jm, im)}")
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            colmol, rayl,
            np.int32(nlay), np.int32(im), np.int32(jm),
            out["taug26"], out["taur26"],
        ))
    return out


def run(ctx):
    """Driver entry. The SW driver puts colmol in ctx.tables[IN_KEY]. Tables load once; outputs are
    allocated once and reused."""
    inp = ctx.tables.get(IN_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_sw_taumol26 needs ctx.tables[{IN_KEY!r}] from the SW driver (phys-rrtm-sw-swrad)")
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    nlay, jm, im = inp["colmol"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None or tuple(out["taug26"].shape) != (NG26, nlay, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taumol26(inp, tab, out, stream=ctx.stream)
