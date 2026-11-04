# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-sw-params: SW control and band parameters of the NMMB RRTM shortwave (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_param_nmmb.f:51-189: module_radsw_cntr_para_nmmb
(six control flags) and module_radsw_parameters_nmmb (band and g-point counts, begin indices, band
wavenumber limits for bands 16..29). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the header is kept in the kernel file
(namgpu/kernels/phys_rrtm_sw_params.cu). Kernel: phys_rrtm_sw_params, one thread, one launch.

Consumers: phys-rrtm-sw-rswinit, phys-rrtm-sw-cldprop, phys-rrtm-sw-mcica-subcol, phys-rrtm-sw-setcoef,
phys-rrtm-sw-spcvrtc, phys-rrtm-sw-swflux, phys-rrtm-sw-taumol and the taumol16..29 units.

Reads: nothing from ctx (the Fortran values are constants).
Writes: ctx.tables["rrtm_sw_params"], a dict of device arrays:
  ctrl  int32[6]     iswrate, iaersw, imodsw, irgassw, iflagliq, iflagice
  dims  int32[9]     NBANDS, NBLOW, NBHGH, NGPT, NGMAX, MAXGAS, NSWSTR, NSWEND, NBDSW
  ng    int32[16]    g-points per band, band b at index b-16 (bands 16..29; slots 14, 15 unused = 0)
  ns    int32[16]    0-based begin index of each band's g-points, band b at index b-16
  wvn1  float32[16]  wvnum1, cm-1 (band lower limit), index b-16
  wvn2  float32[16]  wvnum2, cm-1 (band upper limit), index b-16

Band indexing: the Fortran NG(NBLOW:NBHGH) is indexed by band number 16..29, and wvnum1/wvnum2 are
NBANDS-long arrays in source DATA order (bands 16..28, then 820 cm-1 entry for band 29). Here every
table is stored so that slot b-16 is band b (bands 16..29); consumers must use the same convention.

Host-side consistency (no device work, asserted at import): sum(NG) == NGPT == 112 and
NS(b) + NG(b) == NS(b+1) for every band, with NS(29) + NG(29) == NGPT, so the band table tiles the 112
g-points exactly.

Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation. Runs once at init.
"""
from __future__ import annotations

import contextlib

from namgpu import kernels

UNIT = "phys_rrtm_sw_params"
KERNEL = "phys_rrtm_sw_params"
KEY = "rrtm_sw_params"

# module_radsw_cntr_para_nmmb (radsw_param_nmmb.f:62-82)
CTRL = (2, 1, 2, 1, 1, 3)
# module_radsw_parameters_nmmb (radsw_param_nmmb.f:147-156; NBDSW at line 156 is DIMS[8])
DIMS = (14, 16, 29, 112, 16, 6, 1, 14, 14)
# NG16..NG29 (radsw_param_nmmb.f:161-167), NS16..NS29 (radsw_param_nmmb.f:172-176); slots 14, 15 unused
NG = (6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12, 0, 0)
NS = (0, 6, 18, 26, 34, 44, 54, 56, 66, 74, 80, 86, 94, 100, 0, 0)
# wvnum1 / wvnum2 data (radsw_param_nmmb.f:180-185), cm-1, slot b-16 (DATA index k -> band 15+k -> slot k-1)
WVN1 = (2600., 3251., 4001., 4651., 5151., 6151., 7701., 8051., 12851., 16001., 22651., 29001., 38001., 820., 0., 0.)
WVN2 = (3250., 4000., 4650., 5150., 6150., 7700., 8050., 12850., 16000., 22650., 29000., 38000., 50000., 2600., 0., 0.)

NBANDS, NGPT = DIMS[0], DIMS[3]

# Host-side tiling check (import time, no GPU): the band table covers the 112 g-points exactly.
assert sum(NG[:NBANDS]) == NGPT == 112, "phys-rrtm-sw-params: sum(NG) must equal NGPT"
assert all(NS[b] + NG[b] == NS[b + 1] for b in range(NBANDS - 1)), "phys-rrtm-sw-params: NS/NG do not tile"
assert NS[NBANDS - 1] + NG[NBANDS - 1] == NGPT, "phys-rrtm-sw-params: last band does not end at NGPT"
assert DIMS[8] == NBANDS, "phys-rrtm-sw-params: NBDSW (dims[8]) must equal NBANDS"


def run(ctx):
    """Fill the SW parameter tables once; returns the dict ctx.tables[KEY] (device arrays)."""
    import cupy as cp

    if ctx.tables is None:
        ctx.tables = {}
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    tbl = ctx.tables.get(KEY)
    if tbl is None:
        tbl = {
            "ctrl": cp.zeros(6, dtype=cp.int32),
            "dims": cp.zeros(9, dtype=cp.int32),
            "ng": cp.zeros(16, dtype=cp.int32),
            "ns": cp.zeros(16, dtype=cp.int32),
            "wvn1": cp.zeros(16, dtype=cp.float32),
            "wvn2": cp.zeros(16, dtype=cp.float32),
        }
        ctx.tables[KEY] = tbl

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
