"""phys-rrtm-gethml: high, mid, low, total and bl cloud fractions and cloud layer indices (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_clouds_nmmb.f:2167-2494, subroutine gethml.
NOAA US government work; the NOAA header of gethml is kept as the notice in the kernel file
(namgpu/kernels/phys_rrtm_gethml.cu). Kernel: rrtm_gethml, one thread per column, one launch.

Callers (radiation_clouds_nmmb.f:770, 1311, 1536, 2151) pass the cloud profile of the column and
get back the diagnostic cloud fractions and layer indices. This wrapper is that call.

Arguments (CuPy arrays, C order, float32; zero-based memory, Fortran 1-based levels):
  plyr    [NLAY, JM, IM]  layer mean pressure in MB (Fortran plyr(IX,NLAY)). From Pa use plyr_mb().
  ptop1   [4,    JM, IM]  cloud domain pressure limits in MB: sfc, low, mid, high. ptop1_mb() fills it
                          from NAM ptopc (1050, 650, 400, 0).
  cldtot  [NLAY, JM, IM]  total or stratiform cloud fraction, must be finite
  cldcnv  [NLAY, JM, IM]  convective cloud fraction, must be finite
  llyr    int             bl cloud layer, in the array's level order (1-based). From the sigma profile
                          use llyr_from_si() (radiation_clouds_nmmb.f:328-342), once at setup.
  ivflip  int, REQUIRED   vertical order of the arrays (physparam.f:164 sets 1 in NAM):
                          1 = index 1 is the surface (Fortran ivflip /= 0 branch);
                          0 = index 1 is the model top (ivflip == 0 branch, the CONTRACT layout).
                          Both branches are run and checked by tools/check-phys-rrtm-gethml.py.
  iovr    int, default 1  cloud overlap flag: 0 random, 1 max/ran (physparam.f iovrsw = iovrlw = 1).
  nlay    int, REQUIRED   the NLAY the calling Fortran call passes (LM + LTP at the NAM callers).

Outputs (added to ctx.st, reused on later calls; defined on every call):
  CLDS  [5, JM, IM]  float32  low, mid, high, total, bl cloud fractions (Fortran clds(IX,5))
  MTOP  [3, JM, IM]  int32    top layer index of low, mid, high (Fortran mtop(IX,3)); 0 if the domain
                              never closes in this column
  MBOT  [3, JM, IM]  int32    bottom layer index of low, mid, high; 0 if the domain never closes

No MPI, halo, OpenMP or print. No host sync, no per-step allocation (outputs and producer buffers are
created once through st.add). The producers below run on the host only at setup (llyr, once) or as one
GPU op (plyr_mb, ptop1_mb).
"""
from __future__ import annotations

import contextlib

from namgpu import kernels

UNIT = "phys_rrtm_gethml"
LTP = 1          # grrad_nmmb.f:210, extra top layer (LMK = LM + LTP)
NCLDS_OUT = 5    # clds(IX,5): low, mid, high, total, bl
NK_CLDS = 3      # radiation_clouds_nmmb.f:158
PTOPC_MB = (1050.0, 650.0, 400.0, 0.0)   # NAM ptopc, radiation_clouds_nmmb.f:164 (sfc, low, mid, high), mb
SI_BL = 0.9      # radiation_clouds_nmmb.f:331 and :339, bl cloud sigma cut


def _scratch(st, im: int, jm: int):
    st.add("CLDS", (NCLDS_OUT, jm, im))
    st.add("MTOP", (NK_CLDS, jm, im), dtype="int32")
    st.add("MBOT", (NK_CLDS, jm, im), dtype="int32")


def llyr_from_si(si, ivflip: int) -> int:
    """Bl cloud layer llyr (1-based, Fortran index) from sigma si(1..NLAY) in the array's level order.

    Mirrors radiation_clouds_nmmb.f:325-342. Runs once at setup on NLAY host values, not per step.
    """
    nlay = len(si)
    if nlay < 2:
        raise ValueError("phys-rrtm-gethml: llyr_from_si needs NLAY >= 2")
    if ivflip == 0:                       # toa -> sfc: do k = NLAY, 2, -1
        kl = nlay
        for k in range(nlay, 1, -1):
            kl = k
            if float(si[k - 1]) < SI_BL:
                break
        return kl
    if ivflip == 1:                       # sfc -> toa: do k = 2, NLAY
        kl = 2
        for k in range(2, nlay + 1):
            kl = k
            if float(si[k - 1]) < SI_BL:
                break
        return kl - 1
    raise ValueError("phys-rrtm-gethml: ivflip must be 0 or 1")


def _scope(ctx):
    # Producers run on the same stream as the kernel. Work on another stream has no ordering
    # against the kernel, so a producer on the default stream can race the launch.
    return ctx.stream if ctx.stream is not None else contextlib.nullcontext()


def ptop1_mb(ctx):
    """Cloud domain limits in mb from NAM ptopc. Fills the buffer once; call at setup."""
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    buf = ctx.st.add("PTOP1MB", (4, jm, im))
    with _scope(ctx):
        for n, v in enumerate(PTOPC_MB):
            buf[n] = v
    return buf


def plyr_mb(ctx, pres_pa, nlay: int):
    """Layer mean pressure in mb from pressure in Pa (one GPU divide into a reused buffer).

    The returned buffer is overwritten by the next call; copy it if two calls must coexist.
    """
    import cupy as cp

    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    if tuple(pres_pa.shape) != (int(nlay), jm, im):
        raise ValueError(f"phys-rrtm-gethml: pres_pa must be {(nlay, jm, im)}, got {tuple(pres_pa.shape)}")
    out = ctx.st.add("PLYRMB", (int(nlay), jm, im))
    with _scope(ctx):
        cp.divide(pres_pa, cp.float32(100.0), out=out)
    return out


def run(ctx, *, plyr, ptop1, cldtot, cldcnv, llyr: int, ivflip: int,
        iovr: int = 1, nlay: int):
    """One gethml call over the whole domain. Returns (CLDS, MTOP, MBOT) from ctx.st."""
    import cupy as cp

    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    nlay = int(nlay)
    st = ctx.st
    _scratch(st, im, jm)
    for name, arr, shape in (("plyr", plyr, (nlay, jm, im)),
                             ("ptop1", ptop1, (4, jm, im)),
                             ("cldtot", cldtot, (nlay, jm, im)),
                             ("cldcnv", cldcnv, (nlay, jm, im))):
        if tuple(arr.shape) != shape or arr.dtype != cp.float32:
            raise ValueError(f"phys-rrtm-gethml: {name} must be float32 {shape}, got "
                             f"{arr.dtype} {tuple(arr.shape)}")
    if ivflip not in (0, 1):
        raise ValueError("phys-rrtm-gethml: ivflip must be 0 or 1")
    if not (1 <= int(llyr) <= nlay):
        raise ValueError(f"phys-rrtm-gethml: llyr {llyr} outside 1..{nlay}")

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern = kernels.get(UNIT, "rrtm_gethml")
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
             (im, jm, nlay, int(ivflip), int(iovr), int(llyr),
              plyr, ptop1, cldtot, cldcnv, st.CLDS, st.MTOP, st.MBOT))
    return st.CLDS, st.MTOP, st.MBOT
