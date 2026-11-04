# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-sw-taumol: NMMB RRTM shortwave band dispatcher (taumol), dispatch work.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:3107-3383 (subroutine taumol).
The NOAA/AER header of that file is the notice in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_sw_taumol.cu, phys_rrtm_sw_taumol (one thread per column, one launch).
It writes the TOA solar flux per g-point, sfluxzen[112, JM, IM]. The per-band id0/id1 indices are
evaluated by each taumol## unit with rrtm_sw_taumol_index() (namgpu/kernels/rrtm_sw_bands.cuh).

ctx.tables inputs (written by other units, layouts in the kernel header):
  rrtm_sw_colamt   float32 [6, NLAY, JM, IM]   column amounts of the six gases (from the cldprop/gas unit)
  rrtm_sw_jp       int32   [NLAY, JM, IM]      reference pressure index (from the setcoef unit)
  rrtm_sw_laytrop  int32   [JM, IM]            tropopause layer index, 1-based (from the setcoef unit)
  rrtm_sw_sflux_sfluxref01  float32 [7, 1, 16]   phys-rrtm-sw-table-sflux (the producer's key names)
  rrtm_sw_sflux_sfluxref02  float32 [2, 5, 16]   phys-rrtm-sw-table-sflux
  rrtm_sw_sflux_sfluxref03  float32 [5, 9, 16]   phys-rrtm-sw-table-sflux
Producer gap (review of phys-rrtm-sw-table-sflux, not closed here): no unit in the tree writes
rrtm_sw_colamt, rrtm_sw_jp or rrtm_sw_laytrop into ctx.tables. setcoef keeps jp and laytrop in its own
output dict, and swrad keeps colamt in its per-chunk buffers, so run() raises until those are wired.
Output: ctx.tables["rrtm_sw_sfluxzen"], float32 [112, JM, IM], allocated once if missing.

Fortran sfluxzen is written for every column; the nday < CHK padding is dropped (one launch covers the
whole domain). The taumol16..29 calls are separate units, launched after this one by the driver.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol"
KERNEL = "phys_rrtm_sw_taumol"
NGPTSW = 112
MAXGAS = 6
MFS = (1, 5, 9)     # MFS01, MFS02, MFS03 (radsw_datatb_nmmb.f:1098)
MFB = (7, 2, 5)     # MFB01, MFB02, MFB03
NGMAX = 16
SCALEKUR = np.float32(50.15) / np.float32(48.37)   # radsw_datatb_nmmb.f:1121, single precision


def taumol_sfluxzen(colamt, jp, laytrop, sf01, sf02, sf03, sfluxzen, stream=None):
    """One launch over every column. Shapes as in the module docstring; sfluxzen is overwritten."""
    maxgas, nlay, jm, im = colamt.shape
    if maxgas != MAXGAS:
        raise ValueError(f"phys-rrtm-sw-taumol: colamt has {maxgas} gases, expected {MAXGAS}")
    if tuple(jp.shape) != (nlay, jm, im) or tuple(laytrop.shape) != (jm, im):
        raise ValueError("phys-rrtm-sw-taumol: shape mismatch for jp or laytrop")
    if tuple(sf01.shape) != (MFB[0], MFS[0], NGMAX) or tuple(sf02.shape) != (MFB[1], MFS[1], NGMAX) \
            or tuple(sf03.shape) != (MFB[2], MFS[2], NGMAX):
        raise ValueError("phys-rrtm-sw-taumol: sflux table shapes do not match radsw_datatb_nmmb.f")
    if tuple(sfluxzen.shape) != (NGPTSW, jm, im):
        raise ValueError("phys-rrtm-sw-taumol: sfluxzen shape mismatch")
    k = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (colamt, jp, laytrop, sf01, sf02, sf03, np.float32(SCALEKUR), sfluxzen,
           np.int32(nlay), np.int32(jm), np.int32(im)))
    return sfluxzen


def run(ctx):
    """Driver entry: TOA solar flux for every g-point of the 14 SW bands."""
    import cupy as cp

    t = ctx.tables
    try:
        colamt = t["rrtm_sw_colamt"]
        jp = t["rrtm_sw_jp"]
        laytrop = t["rrtm_sw_laytrop"]
        sf01 = t["rrtm_sw_sflux_sfluxref01"]
        sf02 = t["rrtm_sw_sflux_sfluxref02"]
        sf03 = t["rrtm_sw_sflux_sfluxref03"]
    except KeyError as e:
        raise RuntimeError(f"phys_rrtm_sw_taumol needs ctx.tables[{e.args[0]}]") from None
    _, nlay, jm, im = colamt.shape
    sfluxzen = t.get("rrtm_sw_sfluxzen")
    if sfluxzen is None or tuple(sfluxzen.shape) != (NGPTSW, jm, im):
        sfluxzen = cp.zeros((NGPTSW, jm, im), dtype=cp.float32)
        t["rrtm_sw_sfluxzen"] = sfluxzen
    return taumol_sfluxzen(colamt, jp, laytrop, sf01, sf02, sf03, sfluxzen, stream=ctx.stream)
