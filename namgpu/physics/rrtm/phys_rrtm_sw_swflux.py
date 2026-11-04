"""phys-rrtm-sw-swflux: swflux, upward and downward SW fluxes at layer interfaces (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:2977-3102, subroutine swflux.
NOAA US government work; the routine header is kept as the notice in the kernel file
namgpu/kernels/phys_rrtm_sw_swflux.cu. Kernel: phys_rrtm_sw_swflux, one thread per column, one launch
over the whole (IM, JM) grid. No host sync, no per-step allocation.

Callers: spcvrc and spcvrm (the SW cloud-overlap drivers) pass the per-column reflectances, transmittances
and beam transmittances for each interface and get the interface fluxes back. This wrapper is that call.

Arguments (CuPy float32, C order, level first as [NLP1, JM, IM]; Fortran k = 1 is the lowest interface
and is index 0):
  zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt   [NLP1, JM, IM]   Fortran (CHK, nlp1)
  nlay, nlp1 (= nlay + 1)                    ints
  nday                                       int, accepted for the call signature; read only under
                                             RUNTIME_CHECKING in the Fortran, so not used here
  lmask                                      not passed: the Fortran never references it
  outputs (written in place, float32):
    zfu, zfd                                 [NLP1, JM, IM]   Fortran (CHK, nlp1), intent(out)

NLP1 must be <= 64 (kernel SWF_MAXL). The 12 km NAM parent has LM = 60, LTP = 1, so NLAY = 61, NLP1 = 62.

run(None) is a labelled SELF-TEST: a random, plausible state at the full 954 x 835 x 60 grid (nlay = 61),
every column computed, every output finite. It is not physics and never reaches a State.
run(ctx) reads ZREFB, ZREFD, ZTRAB, ZTRAD, ZLDBT, ZTDBT from ctx.st under those names and adds ZFU, ZFD.
It refuses (RuntimeError) when any input is absent from ctx.st or was never written (State.add zero-fills,
so an unwritten input is all zeros). No synthetic inputs on the ctx path. The inputs come from the upstream
SW units (cldprop, setcoef, spcvrtc); until those run, run(ctx) raises.

Precision: the kernel is single precision (CONTRACT kfpt = single). The Fortran kind_phys is 8 in the NAM
build (phys/machine.f, selected_real_kind(13,60), no SINGLE_PREC guard in the gate0 tree), and 4 only under
-DSINGLE_PREC. The bitwise claim applies only to a reference built with that flag (KIND_PHYS_BUILD_FLAG);
the difference is a declared divergence pending the project lead's ruling, not a claimed match.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_swflux"
KERNEL = "phys_rrtm_sw_swflux"
LTP = 1             # extra top layer (grrad_nmmb.f), nlay = LM + LTP
KIND_PHYS_BUILD_FLAG = "SINGLE_PREC"   # the reference build flag under which kind_phys = 4
SWF_MAXL = 64       # kernel SWF_MAXL: largest nlp1 the kernel handles
IN_NAMES = ("ZREFB", "ZREFD", "ZTRAB", "ZTRAD", "ZLDBT", "ZTDBT")
OUT_NAMES = ("ZFU", "ZFD")


def _plausible(cp, shape, name):
    """Random values inside the physical range of each input (reflectance and transmittance in [0, 1])."""
    rng = cp.random
    if name in ("ZREFB", "ZREFD"):
        return (0.3 * rng.random(shape, dtype=cp.float32)).astype(cp.float32)
    if name in ("ZTRAB", "ZTRAD"):
        return (0.2 + 0.7 * rng.random(shape, dtype=cp.float32)).astype(cp.float32)
    if name == "ZLDBT":
        return (0.2 + 0.7 * rng.random(shape, dtype=cp.float32)).astype(cp.float32)
    return (0.2 + 0.8 * rng.random(shape, dtype=cp.float32)).astype(cp.float32)   # ZTDBT


def swflux(*, zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay: int, nday: int,
           out, stream=None):
    """One launch over every column. Writes out["ZFU"], out["ZFD"] in place and returns out."""
    import cupy as cp

    nlp1 = int(nlay) + 1
    if nlp1 > SWF_MAXL:
        raise ValueError(f"phys-rrtm-sw-swflux: nlp1 {nlp1} exceeds the kernel limit {SWF_MAXL}")
    nlp1_, jm, im = (int(s) for s in zrefb.shape)
    if nlp1_ != nlp1:
        raise ValueError(f"phys-rrtm-sw-swflux: zrefb has {nlp1_} levels, nlay {nlay} needs {nlp1}")
    for name, arr in (("zrefb", zrefb), ("zrefd", zrefd), ("ztrab", ztrab), ("ztrad", ztrad),
                      ("zldbt", zldbt), ("ztdbt", ztdbt), ("zfu", out["ZFU"]), ("zfd", out["ZFD"])):
        if tuple(arr.shape) != (nlp1, jm, im) or arr.dtype != cp.float32:
            raise ValueError(f"phys-rrtm-sw-swflux: {name} must be float32 {(nlp1, jm, im)}, "
                             f"got {arr.dtype} {tuple(arr.shape)}")

    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(nlay), np.int32(nlp1),
            zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt,
            out["ZFU"], out["ZFD"],
        ))
    return out


def run(ctx=None):
    """Done-rule entry. With no ctx: the full 954 x 835 grid, nlay = 61, random plausible inputs,
    one launch; returns the finite flags. With ctx: inputs from ctx.st, outputs added to ctx.st."""
    import cupy as cp

    if ctx is None:
        im, jm, lm = 954, 835, 60
        nlay = lm + LTP
        nlp1 = nlay + 1
        ins = {n: _plausible(cp, (nlp1, jm, im), n) for n in IN_NAMES}
        out = {n: cp.empty((nlp1, jm, im), cp.float32) for n in OUT_NAMES}
        swflux(zrefb=ins["ZREFB"], zrefd=ins["ZREFD"], ztrab=ins["ZTRAB"], ztrad=ins["ZTRAD"],
               zldbt=ins["ZLDBT"], ztdbt=ins["ZTDBT"], nlay=nlay, nday=im * jm, out=out)
        cp.cuda.Stream.null.synchronize()
        return {"finite": {n: bool(cp.isfinite(a).all()) for n, a in out.items()},
                "shape": (im, jm, nlay)}

    st = ctx.st
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    nlay = lm + LTP
    nlp1 = nlay + 1
    # Refuse, do not stand in: an input that no upstream unit has written is absent or all zero.
    # The all-zero check syncs once per ctx and is cached in ctx.tables (as phys-rrtm-getozn does).
    ins = {}
    for n in IN_NAMES:
        if n not in st.fields:
            raise RuntimeError(f"phys-rrtm-sw-swflux: state lacks {n}; the upstream SW units "
                               "(phys-rrtm-sw-cldprop, phys-rrtm-sw-setcoef, phys-rrtm-sw-spcvrtc) "
                               "must write it first (no synthetic inputs)")
        ins[n] = st.fields[n]
    if not ctx.tables.get("swflux_inputs_checked"):
        for n in IN_NAMES:
            if not bool(cp.any(ins[n] != 0).get()):
                raise RuntimeError(f"phys-rrtm-sw-swflux: {n} is all zero (never written by the upstream "
                                   "SW units); refusing to compute fluxes from empty reflectances")
        ctx.tables["swflux_inputs_checked"] = True
    out = {n: st.add(n, (nlp1, jm, im)) for n in OUT_NAMES}
    swflux(zrefb=ins["ZREFB"], zrefd=ins["ZREFD"], ztrab=ins["ZTRAB"], ztrad=ins["ZTRAD"],
           zldbt=ins["ZLDBT"], ztdbt=ins["ZTDBT"], nlay=nlay, nday=im * jm, out=out,
           stream=ctx.stream)
    return out
