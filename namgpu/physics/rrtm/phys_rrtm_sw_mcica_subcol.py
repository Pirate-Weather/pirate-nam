# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-sw-mcica-subcol: McICA sub-column cloud generator of the NMMB RRTM shortwave.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:1827-1984 (mcica_subcol), called from
cldprop when isubcsw > 0. The NOAA/AER header of that file is the notice in the kernel file. The random
generator is written from the MT19937 paper, not copied from the LGPL mersenne_twister.f (see the kernel
header).

Kernel: namgpu/kernels/phys_rrtm_sw_mcica_subcol.cu, phys_rrtm_sw_mcica_subcol (one thread per column,
one launch).

NAM 12 km values (bound in namgpu/config.py, not defaults):
  isubcsw = 0   module_SOLVER_GRID_COMP.F90:10697 ISUBCSW=0. The McICA path is OFF on the NAM 12 km run:
                run() returns None and this unit does no work there.
  iovrsw  = 1   module_SOLVER_GRID_COMP.F90:10693 IOVR_SW=1 (max-random overlap).

Layer order (verified 2026-11-04, not assumed): the NMMB driver passes IFLIP = 0 (:10714) into
rad_initialize_nmmb, so radsw_main_nmmb.f reverses the NMMB top-first input to its internal array
(lines 732-944). The internal array runs surface-first. cldf and lcloudy are in that internal order:
index 0 is the LOWEST layer. This is the radiation exception in CONTRACT.md section 1. The producer is
phys-rrtm-sw-cldprop (k0 = 0 is Fortran k = 1 of the internal order).

Inputs (device):
  cldf    float32 [NLAY, JM, IM]   layer cloud fraction from cldprop, lowest layer first. The ftiny step
                                   (radsw_main_nmmb.f:1787-1792, cldf = 0 where cldf < 1e-12) is applied in
                                   the kernel, the consumer.
  ipseed  int32   [JM, IM]         permutation seed per column.
  iovrsw  int, 0 random, 1 max-random, 2 maximum (overlap flag of radsw_main_nmmb.f).
Output (device):
  lcloudy uint32  [NWORD = 4, NLAY, JM, IM] bit-packed sub-column flags: bit (g & 31) of word (g >> 5)
          is lcloudy(g+1, k0+1), for g-points 0..111.

Seed convention (DECLARED DEVIATION, not bitwise): Fortran 693-695 seeds ipsdsw0 + i with i the index
inside the CPU chunk (npts). The wrapper uses the 1-based linear index over the whole domain, j * IM + i.
That equals the Fortran only for a single-chunk run. Recorded here and in the receipt. The project ruled bitwise
out for this path.

OPEN ITEM (not closed): random_setseed in the LGPL mersenne_twister.f is not verified. The Fortran caller
comment says "advance random number generator by ipseed values" (1875-1882), and the kernel seeds without
an advance. Closing it needs a clean-room check (a reader of the module, or a reviewer, who writes the
seeding contract down without copying the code). Until then the seed path is unverified. The NAM 12 km
run never reaches it (isubcsw = 0).

run(ctx) does nothing when cfg.isubcsw is 0. For isubcsw == 1 the seed is ipsdsw0 + linear index
with ipsdsw0 = 1 (radsw_main_nmmb.f:349, 693-695). For isubcsw == 2 the caller must put the seed array in
ctx.tables["rrtm_sw_ipseed"] (icseed). The cldf array comes from ctx.tables["rrtm_sw_cldf"] (written by the
cldprop unit).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_mcica_subcol"
KERNEL = "phys_rrtm_sw_mcica_subcol"
NGPTSW = 112                  # radsw_param.f:110 NGPTSW
NWORD = 4                     # ceil(NGPTSW / 32)
NLAY_MAX = 128                # kernel scratch limit
IPSDSW0 = 1                   # radsw_main_nmmb.f:349 ipsdsw0


def new_lcloudy(nlay: int, jm: int, im: int):
    """Packed output buffer [NWORD, NLAY, JM, IM] uint32, allocated once by the caller."""
    import cupy as cp

    return cp.zeros((NWORD, nlay, jm, im), dtype=cp.uint32)


def _check_device_array(arr, name: str, dtype_name: str) -> None:
    """Refuse a wrong dtype or a non-contiguous array instead of letting the kernel misread it."""
    import cupy as cp

    if arr.dtype != cp.dtype(dtype_name):
        raise ValueError(f"phys-rrtm-sw-mcica-subcol: {name} must be {dtype_name}, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        raise ValueError(f"phys-rrtm-sw-mcica-subcol: {name} must be C-contiguous")


def mcica_subcol(cldf, ipseed, lcloudy, iovrsw: int, stream=None):
    """One launch over every column. cldf float32 [NLAY, JM, IM]; ipseed int32 [JM, IM];
    lcloudy uint32 [NWORD, NLAY, JM, IM] (output, overwritten)."""
    nlay, jm, im = cldf.shape
    if nlay > NLAY_MAX:
        raise ValueError(f"phys-rrtm-sw-mcica-subcol: NLAY {nlay} > kernel limit {NLAY_MAX}")
    if iovrsw not in (0, 1, 2):
        raise ValueError(f"phys-rrtm-sw-mcica-subcol: iovrsw {iovrsw} outside 0..2")
    if tuple(ipseed.shape) != (jm, im) or tuple(lcloudy.shape) != (NWORD, nlay, jm, im):
        raise ValueError("phys-rrtm-sw-mcica-subcol: shape mismatch for ipseed or lcloudy")
    _check_device_array(cldf, "cldf", "float32")
    _check_device_array(ipseed, "ipseed", "int32")
    _check_device_array(lcloudy, "lcloudy", "uint32")
    k = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (cldf, ipseed, lcloudy, np.int32(nlay), np.int32(jm), np.int32(im), np.int32(iovrsw)))
    return lcloudy


def run(ctx):
    """Driver entry, called only where cldprop calls mcica_subcol with isubcsw > 0.

    cfg.isubcsw and cfg.iovrsw are required: config.py binds them to the NAM values (ISUBCSW 0,
    IOVR_SW 1). A missing field raises rather than falling back to a default.
    """
    import cupy as cp

    isubcsw = int(ctx.cfg.isubcsw)
    if isubcsw <= 0:
        return None
    cldf = ctx.tables.get("rrtm_sw_cldf")
    if cldf is None:
        raise RuntimeError("phys_rrtm_sw_mcica_subcol needs ctx.tables[rrtm_sw_cldf] from the cldprop unit")
    nlay, jm, im = cldf.shape
    if isubcsw == 1:
        lin = cp.arange(1, jm * im + 1, dtype=cp.int32).reshape(jm, im)
        ipseed = IPSDSW0 + lin
    elif isubcsw == 2:
        ipseed = ctx.tables.get("rrtm_sw_ipseed")
        if ipseed is None:
            raise RuntimeError("isubcsw 2 needs ctx.tables[rrtm_sw_ipseed] (icseed)")
    else:
        raise ValueError(f"isubcsw {isubcsw} is not 1 or 2 (radsw_main_nmmb.f:693-699)")
    ipseed = cp.ascontiguousarray(ipseed, dtype=cp.int32)
    lcloudy = ctx.tables.get("rrtm_sw_lcloudy")
    if lcloudy is None or tuple(lcloudy.shape) != (NWORD, nlay, jm, im):
        lcloudy = new_lcloudy(nlay, jm, im)
        ctx.tables["rrtm_sw_lcloudy"] = lcloudy
    iovrsw = int(ctx.cfg.iovrsw)
    return mcica_subcol(cldf, ipseed, lcloudy, iovrsw, stream=ctx.stream)
