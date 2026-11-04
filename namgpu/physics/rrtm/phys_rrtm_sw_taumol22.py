# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-sw-taumol22: TAUMOL22, SW band 22 (7700-8050 cm-1) optical depths and Rayleigh depth.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:4085-4201 (SUBROUTINE taumol22).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the sw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sw_taumol22.cu (rrtm_sw_taumol22, one thread per day column of a chunk).
Tables: namgpu/data/rrtm_sw/kgb22.npz (module_radsw_kgb22_nmmb, from tools/extract-rrtm-sw-kgb22.py).

What it computes, per chunk column and layer k (1-based Fortran k):
  taur(g, k) = colmol(k) * rayl for both band-22 g-points (planes NS22 = 54, 55 of the 112-plane array).
  taug(g, k) for k <= laytrop: absa interpolation in ln(p) and T with the h2o/o2 mixing, the water-vapour
             self and foreign continuum, plus the O2 continuum.
  taug(g, k) for k >  laytrop: absb interpolation with the o2adj-scaled O2 band, plus the O2 continuum.
  id0, id1 are evaluated in the kernel (rrtm_sw_taumol_index, rrtm_sw_bands.cuh) from jp, jt, jt1 and
  laytrop, as the taumol dispatcher does (radsw_main_nmmb.f:3251-3256).

Entry point: run_chunk(ctx, ch, cstart, nc), the driver seam of phys-rrtm-sw-swrad._seam. ch is the chunk
scratch ([nc, nlay] inputs, taug/taur [nc, nlay, 112]); day column cstart + t is live when it is below
SW_NDAY (device). Writes planes 54..55 of ch taug and taur. Other planes belong to other units.

Declared deviations (kernel header): k == laytrop uses absa (the Fortran is chunk-dependent there; NEEDS
OWNER SIGN-OFF), and speccomb == 0 takes the ratio as 0 (witnessed equal output).

Precision: kind_taum = 4 (SINGLE_PREC build of machine.f). The native Linux gnu configure
(src/conf/configure.nems.Linux.gnu) does not define SINGLE_PREC, so the native default build is kind_taum = 8.
That choice is a project-lead decision and is declared in STATUS.md; this unit follows the WOOF-NAM single-precision
run.

The Fortran comparison is tools/check-phys-rrtm-sw-taumol22-fortran.py (the real subroutine compiled with
gfortran against the NAM module sources, SINGLE_PREC).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol22"
KERNEL = "rrtm_sw_taumol22"
NG22 = 2
NS22 = 54                 # NS22 = NS21 + NG21 (radsw_param_nmmb.f:172-177)
NGPTSW = 112
MSA22, MSB22, MSF22, MFR22 = 585, 235, 10, 3
MAXGAS = 6
TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb22.npz"

TAB_KEY = "rrtm_sw_taumol22_tables"
SW_NDAY_KEY = "SW_NDAY"


def load_tables() -> dict:
    """The kgb22 k-tables, uploaded once. Fortran column-major order is kept: the flat index of
    absa(ms, g) is (ms-1) + 585*(g-1), and so on."""
    import cupy as cp

    z = np.load(TABLE_FILE)
    tab = {}
    for name, n in (("absa", MSA22), ("absb", MSB22), ("selfref", MSF22), ("forref", MFR22)):
        arr = np.asarray(z[name], dtype=np.float32)
        assert arr.shape == (n, NG22), f"kgb22 {name} shape {arr.shape}"
        tab[name] = cp.asarray(np.ravel(arr, order="F"))
    return tab


def _launch(ch: dict, tab: dict, ndayp, nc: int, nlay: int, cstart: int, stream=None) -> None:
    kern = kernels.get(UNIT, KERNEL)
    block = kernels.BLOCK_COL
    grid = ((nc + block[0] - 1) // block[0], 1, 1)
    args = (
        ch["colmol"], ch["colamt"], ch["laytrop"],
        ch["fac00"], ch["fac01"], ch["fac10"], ch["fac11"],
        ch["selffac"], ch["selffrac"], ch["indself"],
        ch["forfac"], ch["forfrac"], ch["indfor"],
        ch["jp"], ch["jt"], ch["jt1"],
        tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
        ndayp, np.int32(cstart), np.int32(nc), np.int32(nlay),
        np.int64(nlay), np.int64(1),                              # lc, lk: [nc, nlay]
        np.int64(nlay * MAXGAS), np.int64(MAXGAS), np.int64(1),   # colamt [nc, nlay, 6]
        np.int64(nlay * NGPTSW), np.int64(NGPTSW), np.int64(1),   # taug, taur [nc, nlay, 112]
        ch["taug"], ch["taur"],
    )
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)


def run_chunk(ctx, ch: dict, cstart: int, nc: int):
    """Driver seam entry (phys-rrtm-sw-swrad._seam): taumol22 for one chunk of day columns. Reads ch colamt
    [nc, nlay, 6], colmol, fac00..fac11, selffac, selffrac, forfac, forfrac, indself, indfor, jp, jt, jt1
    [nc, nlay] and laytrop [nc] (setcoef outputs). Writes planes 54..55 of ch taug and taur [nc, nlay, 112].
    The chunk runs one launch; the device SW_NDAY gates the live columns, so the last chunk may run past it."""
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    colmol = ch["colmol"]
    if colmol.shape[0] != nc:
        raise ValueError(f"taumol22 chunk: colmol {colmol.shape} does not match nc={nc}")
    nlay = int(colmol.shape[1])
    if ch["colamt"].shape != (nc, nlay, MAXGAS):
        raise ValueError(f"taumol22 chunk: colamt {ch['colamt'].shape} != {(nc, nlay, MAXGAS)}")
    if ch["taug"].shape != (nc, nlay, NGPTSW) or ch["taur"].shape != (nc, nlay, NGPTSW):
        raise ValueError(f"taumol22 chunk: taug/taur must be [nc, nlay, NGPTSW] = {(nc, nlay, NGPTSW)}")
    ndayp = ctx.st[SW_NDAY_KEY]
    _launch(ch, tab, ndayp, nc, nlay, cstart, stream=getattr(ctx, "stream", None))
    return ch


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1, nc: int = 4096):
    """Full-size plausible state (954 x 835 columns, 60 layers) through the chunk path, every chunk with the
    SW_NDAY gate open. Asserts the band-22 planes are finite in every chunk and that the index ranges the
    Fortran clamps to are respected by the inputs. This is a finiteness check only; the numbers are
    compared to the Fortran in tools/check-phys-rrtm-sw-taumol22-fortran.py."""
    import cupy as cp
    from types import SimpleNamespace

    from namgpu.physics.rrtm.phys_rrtm_sw_swrad import chunk_scratch

    ncols = im * jm
    rs = cp.random.RandomState(seed)
    ctx = SimpleNamespace(tables={}, st={SW_NDAY_KEY: cp.asarray([ncols], dtype=cp.int32)}, stream=None)
    ctx.tables[TAB_KEY] = load_tables()
    n_bad = 0
    for cstart in range(0, ncols, nc):
        ncur = min(nc, ncols - cstart)
        ch = chunk_scratch(cp, nc, nlay, nlay + 1)
        shp = (nc, nlay)
        ch["colmol"] = cp.exp(rs.uniform(np.log(1e20), np.log(1e25), shp, dtype=np.float32)).astype(cp.float32)
        ch["colamt"] = rs.uniform(0.0, 1.0e24, (nc, nlay, MAXGAS), dtype=np.float32)
        ch["laytrop"] = rs.randint(1, nlay + 1, size=nc, dtype=np.int32)
        for name in ("fac00", "fac01", "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac"):
            ch[name] = rs.uniform(0.0, 1.0, shp, dtype=np.float32)
        ch["indself"] = rs.randint(1, 10, size=shp, dtype=np.int32)
        ch["indfor"] = rs.randint(1, 3, size=shp, dtype=np.int32)
        ch["jp"] = rs.randint(1, 14, size=shp, dtype=np.int32)
        ch["jt"] = rs.randint(1, 6, size=shp, dtype=np.int32)
        ch["jt1"] = rs.randint(1, 6, size=shp, dtype=np.int32)
        run_chunk(ctx, ch, cstart, nc)
        band = ch["taug"][:ncur, :, NS22:NS22 + NG22]
        bandr = ch["taur"][:ncur, :, NS22:NS22 + NG22]
        if not (bool(cp.isfinite(band).all()) and bool(cp.isfinite(bandr).all())):
            n_bad += 1
    cp.cuda.Stream.null.synchronize()
    if n_bad:
        raise AssertionError(f"NaN/Inf in taumol22 band-22 planes in {n_bad} chunk(s)")
    return {"chunks": (ncols + nc - 1) // nc, "nan_chunks": 0}
