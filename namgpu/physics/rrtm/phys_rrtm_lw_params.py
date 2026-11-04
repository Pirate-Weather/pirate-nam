"""phys-rrtm-lw-params: LW control and band parameters of the NMMB RRTM longwave (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_param_nmmb.f:57-175: module_radlw_cntr_para_nmmb
(six control flags) and module_radlw_parameters_nmmb (band and g-point counts, begin indices, band
wavenumber limits). NOAA US government work; the header is kept in the kernel file
(namgpu/kernels/phys_rrtm_lw_params.cu). Kernel: phys_rrtm_lw_params, one thread, one launch.

Consumers: phys-rrtm-lw-rlwinit, phys-rrtm-lw-cldprop and phys-rrtm-lw-lwrad. The NMMB LW Fortran (radlw_main_nmmb.f)
USEs module_radlw_parameters (radlw_param.f), which is what this unit now holds. radlw_param_nmmb.f
(module_radlw_parameters_nmmb, NG01=8) is the GFS-family set and is not used by the NMMB LW.

Reads: nothing from ctx (the Fortran values are constants).
Writes: ctx.tables["rrtm_lw_params"], a dict of device arrays, and ctx.tables["rrtm_lw_ngb"] int32[140]:
  ctrl  int32[6]     ilwrate, iaerlw, irgaslw, icfclw, iflagliq, iflagice
  dims  int32[8]     NBANDS, NBDLW, NGPTLW, N5000, N200, MAXGAS, MAXXSEC, NPLNK
  ng    int32[16]    NG01..NG16 (g-points per band)
  ns    int32[16]    NS01..NS16 (0-based begin index of each band's g-points)
  wvn1  float32[16]  wvnlw1, cm-1 (band lower limit)
  wvn2  float32[16]  wvnlw2, cm-1 (band upper limit)

The Fortran type definitions (topflw_type, sfcflw_type, proflw_type; lines 115-135) hold no values and
have no device counterpart here: the flux fields they name are carried in ctx.st by the lwrad unit.

Host-side consistency (no device work, asserted at import): sum(NG) == NGPTLW == 140 and
NS(b) + NG(b) == NS(b+1) for every band, so the band table tiles the 140 g-points exactly.

Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation. Runs once at init.
"""
from __future__ import annotations

import contextlib

from namgpu import kernels

UNIT = "phys_rrtm_lw_params"
KERNEL = "phys_rrtm_lw_params"
KEY = "rrtm_lw_params"

# Control flags: physparam.f (ilwrate, ilwrgas, ilwcliq, ilwcice) as the NMMB LW uses them
# (radlw_main_nmmb.f:238). iaerlw and icfclw are kept from module_radlw_cntr_para_nmmb
# (radlw_param_nmmb.f:67-90); physparam.f does not define them.
CTRL = (2, 1, 1, 1, 1, 3)
# module_radlw_parameters (radlw_param.f:96-101, 108-116): the module radlw_main_nmmb.f USEs.
# DIMS = (NBANDS, NBDLW, NGPTLW, NTBL, NRATES, MAXGAS, MAXXSEC, NPLNK).
# MAXGAS = 7 here (radlw_param.f:99); setcoef and the band units carry 6 gas columns (see the
# phys-rrtm-lw-taumol notes): that mismatch is an open gap, not decided here.
DIMS = (16, 16, 140, 10000, 6, 7, 4, 181)
# NG01..NG16 (radlw_param.f:107-109), NS01..NS16 (radlw_param.f:114-116)
NG = (10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2)
NS = (0, 10, 22, 38, 52, 68, 76, 88, 96, 108, 114, 122, 130, 134, 136, 138)
# wvnlw1 / wvnlw2 (radlw_param.f:123-130), cm-1
WVN1 = (10., 351., 501., 631., 701., 821., 981., 1081., 1181., 1391., 1481., 1801., 2081., 2251., 2381., 2601.)
WVN2 = (350., 500., 630., 700., 820., 980., 1080., 1180., 1390., 1480., 1800., 2080., 2250., 2380., 2600., 3250.)

NBANDS, NBDLW, NGPTLW = DIMS[0], DIMS[1], DIMS[2]

# Host-side tiling check (import time, no GPU): the band table covers the 140 g-points exactly.
assert sum(NG) == NGPTLW == 140, "phys-rrtm-lw-params: sum(NG) must equal NGPTLW"
assert all(NS[b] + NG[b] == NS[b + 1] for b in range(NBANDS - 1)), "phys-rrtm-lw-params: NS/NG do not tile"
assert NS[-1] + NG[-1] == NGPTLW, "phys-rrtm-lw-params: last band does not end at NGPTLW"


def run(ctx):
    """Fill the LW parameter tables once; returns the dict ctx.tables[KEY] (device arrays)."""
    import cupy as cp

    if ctx.tables is None:
        ctx.tables = {}
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    tbl = ctx.tables.get(KEY)
    if tbl is None:
        tbl = {
            "ctrl": cp.zeros(6, dtype=cp.int32),
            "dims": cp.zeros(8, dtype=cp.int32),
            "ng": cp.zeros(16, dtype=cp.int32),
            "ns": cp.zeros(16, dtype=cp.int32),
            "wvn1": cp.zeros(16, dtype=cp.float32),
            "wvn2": cp.zeros(16, dtype=cp.float32),
        }
        ctx.tables[KEY] = tbl
    # ngb: band index (1-based) of each of the 140 g-points, the NGB data statement of
    # radlw_param.f:119-121. rtrn reads it as ctx.tables["rrtm_lw_ngb"] (phys_rrtm_lw_rtrn.py).
    if "rrtm_lw_ngb" not in ctx.tables:
        ctx.tables["rrtm_lw_ngb"] = cp.repeat(cp.arange(1, 17, dtype=cp.int32), cp.asarray(NG, dtype=cp.int32))

    k = kernels.get(UNIT, KERNEL)
    with scope:
        k((1, 1, 1), (1, 1, 1), (tbl["ctrl"], tbl["dims"], tbl["ng"], tbl["ns"], tbl["wvn1"], tbl["wvn2"]))
    return tbl


def check_status(ctx):
    """Host read of the device tables for tests only (not on the step path)."""
    import numpy as np

    tbl = ctx.tables[KEY]
    return {
        "ctrl": tuple(int(v) for v in tbl["ctrl"].get()),
        "dims": tuple(int(v) for v in tbl["dims"].get()),
        "ng": tuple(int(v) for v in tbl["ng"].get()),
        "ns": tuple(int(v) for v in tbl["ns"].get()),
        "wvn1": tuple(float(v) for v in np.asarray(tbl["wvn1"].get(), dtype=np.float32)),
        "wvn2": tuple(float(v) for v in np.asarray(tbl["wvn2"].get(), dtype=np.float32)),
    }
