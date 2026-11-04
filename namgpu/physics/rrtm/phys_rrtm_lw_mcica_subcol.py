# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-mcica-subcol: McICA sub-column cloud generator of the NMMB RRTM longwave.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:1702-1858 (mcica_subcol), called from
cldprop at radlw_main_nmmb.f:1665-1693 when isubclw > 0. The NOAA/AER header of that file is the notice
in the kernel file. The random generator is written from the MT19937 paper, not copied from the LGPL
mersenne_twister.f (see the kernel header).

Kernels (namgpu/kernels/phys_rrtm_lw_mcica_subcol.cu):
  phys_rrtm_lw_mcica_subcol   one thread per column, one launch: lcloudy bits from cldf and the seed.
  phys_rrtm_lw_mcica_unpack   the lcloudy -> cldfmc step of cldprop (radlw_main_nmmb.f:1684-1691):
                              cldfmc(ig,k) = 1 if lcloudy(ig,k) else 0, in the layout rtrnmc reads.

Inputs (device):
  cldf    float32 [NLAY, JM, IM]   layer cloud fraction after the cldmin threshold, radlw INTERNAL order
                                   (index 0 = lowest layer; CONTRACT.md section 1, LW exception).
  ipseed  int32   [JM, IM]         permutation seed per column.
  iovrlw  int, 0 random, 1 max-random, 2 maximum (radlw_main_nmmb.f RLWINIT allows 0..2).
Outputs (device):
  lcloudy uint32  [NWORD = 5, NLAY, JM, IM] bit-packed sub-column flags: bit (g & 31) of word (g >> 5)
          is lcloudy(g+1, k0+1), for g-points 0..139.
  cldfmc  float32 [NLAY, NGPTLW, NCOL] for flat columns col0 .. col0+ncol-1, c = j * IM + i (0-based),
          the layout phys-rrtm-lw-rtrnmc reads. A full 954 x 835 domain at NLAY = 60 is about 26.7 GB,
          so the caller takes column chunks (run(ctx, col0, ncol)).

Configuration: run(ctx) reads isubclw and iovrlw from ctx.tables["rrtm_lw_ctl_host"], the values
phys-rrtm-lw-rlwinit set after its Fortran checks (radlw_main_nmmb.f:1229-1265). A missing rlwinit host
copy is a refusal, not a default. run() does nothing when isubclw is 0 (NAM default, physparam.f:166).
For isubclw == 1 the seed is ipsdlw0 + i with ipsdlw0 = ngptlw = 140 (radlw_main_nmmb.f:339, 629-631);
i is taken as the 1-based flat column index j * IM + i + 1. For isubclw == 2 the caller must put the
seed array in ctx.tables["rrtm_lw_ipseed"] (icseed).

Unverified: the column-to-(i, j) mapping of the Fortran npts loop (radlw_main_nmmb.f:630-646) is set by
the NMMB caller grrad.f, which is in the source tree but whose call site is not wired into WOOF-NAM yet.
The j-major flat order used here is an assumption and is not claimed correct until that call site is checked.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_mcica_subcol"
KERNEL = "phys_rrtm_lw_mcica_subcol"
UNPACK_KERNEL = "phys_rrtm_lw_mcica_unpack"
NGPTLW = 140                  # radlw_param_nmmb.f:141
NWORD = 5                     # ceil(NGPTLW / 32)
NLAY_MAX = 128                # kernel scratch limit
IPSDLW0 = NGPTLW              # radlw_main_nmmb.f:339
UNPACK_BLOCK = 256


def new_lcloudy(nlay: int, jm: int, im: int):
    """Packed output buffer [NWORD, NLAY, JM, IM] uint32, allocated once by the caller."""
    import cupy as cp

    return cp.zeros((NWORD, nlay, jm, im), dtype=cp.uint32)


def mcica_subcol(cldf, ipseed, lcloudy, iovrlw: int, stream=None):
    """One launch over every column. cldf float32 [NLAY, JM, IM]; ipseed int32 [JM, IM];
    lcloudy uint32 [NWORD, NLAY, JM, IM] (output, overwritten)."""
    nlay, jm, im = cldf.shape
    if nlay > NLAY_MAX:
        raise ValueError(f"phys-rrtm-lw-mcica-subcol: NLAY {nlay} > kernel limit {NLAY_MAX}")
    if iovrlw not in (0, 1, 2):
        raise ValueError(f"phys-rrtm-lw-mcica-subcol: iovrlw {iovrlw} outside 0..2 (RLWINIT stops here)")
    if tuple(ipseed.shape) != (jm, im) or tuple(lcloudy.shape) != (NWORD, nlay, jm, im):
        raise ValueError("phys-rrtm-lw-mcica-subcol: shape mismatch for ipseed or lcloudy")
    k = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (cldf, ipseed, lcloudy, np.int32(nlay), np.int32(jm), np.int32(im), np.int32(iovrlw)))
    return lcloudy


def unpack_cldfmc(lcloudy, col0: int, ncol: int, out=None, stream=None):
    """lcloudy uint32 [NWORD, NLAY, JM, IM] -> cldfmc float32 [NLAY, NGPTLW, ncol] for flat columns
    col0 .. col0+ncol-1 (c = j * IM + i). Writes into `out` when given (no allocation)."""
    import cupy as cp

    nwd, nlay, jm, im = lcloudy.shape
    if nwd != NWORD:
        raise ValueError(f"phys-rrtm-lw-mcica-unpack: lcloudy word count {nwd} != {NWORD}")
    ncol_all = jm * im
    if col0 < 0 or ncol <= 0 or col0 + ncol > ncol_all:
        raise ValueError(f"phys-rrtm-lw-mcica-unpack: columns {col0}..{col0 + ncol - 1} outside 0..{ncol_all - 1}")
    if out is None:
        out = cp.empty((nlay, NGPTLW, ncol), dtype=cp.float32)
    elif tuple(out.shape) != (nlay, NGPTLW, ncol) or out.dtype != cp.float32:
        raise ValueError("phys-rrtm-lw-mcica-unpack: out must be float32 [NLAY, NGPTLW, ncol]")
    # Word-set view of the chunk: [NWORD, NLAY, ncol] over the flat column axis.
    words = lcloudy.reshape(NWORD, nlay, ncol_all)[:, :, col0:col0 + ncol]
    if not words.flags.c_contiguous:
        words = cp.ascontiguousarray(words)
    n = nlay * NGPTLW * ncol
    grid = ((n + UNPACK_BLOCK - 1) // UNPACK_BLOCK, 1, 1)
    k = kernels.get(UNIT, UNPACK_KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (UNPACK_BLOCK, 1, 1),
          (words, out, np.int32(nlay), np.int32(ncol)))
    return out


def run(ctx, col0: int = 0, ncol: int | None = None):
    """Driver entry, called only where radlw calls cldprop with isubclw > 0.

    Returns the cldfmc chunk [NLAY, NGPTLW, ncol] for columns col0..col0+ncol-1 (default: the rest of the
    domain, about 27 GB at 60 levels for the full domain, so normally run in chunks), or None when isubclw is 0."""
    import cupy as cp

    ctl = ctx.tables.get("rrtm_lw_ctl_host")
    if ctl is None:
        raise RuntimeError("phys_rrtm_lw_mcica_subcol: ctx.tables[rrtm_lw_ctl_host] missing; "
                           "phys-rrtm-lw-rlwinit must run first (RLWINIT owns isubclw and iovrlw)")
    isubclw = int(ctl["isubclw"])
    if isubclw <= 0:
        return None                                   # Fortran calls mcica_subcol only if isubclw > 0
    # cldprop (phys_rrtm_lw_cldprop.run) writes this key; both units must agree on it.
    cldf = ctx.tables.get("rrtm_lw_cldprop_cldf")
    if cldf is None:
        raise RuntimeError("phys_rrtm_lw_mcica_subcol needs ctx.tables[rrtm_lw_cldprop_cldf] from the cldprop unit")
    nlay, jm, im = cldf.shape
    if isubclw == 1:
        lin = cp.arange(1, jm * im + 1, dtype=cp.int32).reshape(jm, im)
        ipseed = IPSDLW0 + lin
    elif isubclw == 2:
        ipseed = ctx.tables.get("rrtm_lw_ipseed")
        if ipseed is None:
            raise RuntimeError("isubclw 2 needs ctx.tables[rrtm_lw_ipseed] (icseed)")
    else:
        raise ValueError(f"isubclw {isubclw} is not 1 or 2 (radlw_main_nmmb.f:1254-1265)")
    lcloudy = ctx.tables.get("rrtm_lw_lcloudy")
    if lcloudy is None or tuple(lcloudy.shape) != (NWORD, nlay, jm, im):
        lcloudy = new_lcloudy(nlay, jm, im)
        ctx.tables["rrtm_lw_lcloudy"] = lcloudy
    iovrlw = int(ctl["iovrlw"])                       # RLWINIT-adjusted value
    mcica_subcol(cldf, ipseed.astype(cp.int32, copy=False), lcloudy, iovrlw, stream=ctx.stream)
    if ncol is None:
        ncol = jm * im - col0
    out = ctx.tables.get("rrtm_lw_cldfmc")
    if out is None or tuple(out.shape) != (nlay, NGPTLW, ncol):
        out = cp.empty((nlay, NGPTLW, ncol), dtype=cp.float32)
        ctx.tables["rrtm_lw_cldfmc"] = out
    return unpack_cldfmc(lcloudy, col0, ncol, out=out, stream=ctx.stream)
