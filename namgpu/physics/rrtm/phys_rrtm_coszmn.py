"""phys-rrtm-coszmn: COSZMN_NMMB, mean cosine of solar zenith over the SW call interval (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_astronomy_nmmb.f:744-848, coszmn_nmmb.
NOAA US government work; the NOAA routine header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_coszmn.cu (phys_rrtm_coszmn, one thread per (i, j), one launch, float32).

Caller: grrad_nmmb.f:1409-1414 calls coszmn_nmmb with coszen_loc and coszdg; grrad_nmmb.f:2201 and
module_RA_RRTM.F90:1142-1143 then hand them on as COSZDG_V and COSZEN_V. Here the whole domain is one grid.

Outputs (device, float32 [JM, IM]), written to the grrad scratch names that phys-rrtm-driver-post reads:
  GRRAD_COSZEN_V  daytime average over the SW interval  -> driver-post copies it to CZMEAN (module_RA_RRTM 1143)
  GRRAD_COSZDG_V  average over the whole SW interval    -> driver-post copies it to COSZDG  (module_RA_RRTM 1142)
Inputs: GLON, GLAT (state, radians); solar module variables sollag, sindec, cosdec from SOL_STATE
(float32[4], written by phys-rrtm-sol-update, read on the device by the kernel, no host copy).
Scalars: solhr = mod(fhr, 24) with fhr = ihr + imin/60 from the clock's JDAT (module_RA_RRTM.F90:857-863);
dtswav = NRADS*DT/3600 in hours with DT = cfg.dt (module_RA_RRTM.F90:840; nam_configfile dt_int 25, nrads 48).
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_coszmn"
KERNEL = "phys_rrtm_coszmn"
SOL_STATE = "SOL_STATE"            # phys-rrtm-sol-update: float32[4] = sollag, sindec, cosdec, anginc
OUT_DAYTIME = "GRRAD_COSZEN_V"     # grrad_nmmb COSZEN_V (daytime average)
OUT_WHOLE = "GRRAD_COSZDG_V"       # grrad_nmmb COSZDG_V (whole SW interval)


def coszmn(glon, glat, coszen, coszdg, solhr, dtswav, nrads, sol, stream=None):
    """One launch over every (i, j). glon, glat, coszen, coszdg: device float32 [JM, IM] (C order).
    sol: device float32[4] = sollag, sindec, cosdec, anginc."""
    jm, im = glon.shape
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            glon, glat, coszen, coszdg,
            np.float32(solhr), np.float32(dtswav), np.int32(nrads), sol,
            np.int32(im), np.int32(jm),
        ))
    return coszen, coszdg


def run(ctx):
    """Driver entry. Needs phys-rrtm-sol-update to have run this step (SOL_STATE in ctx.st)."""
    st = ctx.st
    shape = st["GLAT"].shape
    if SOL_STATE not in st.fields:
        raise RuntimeError("phys_rrtm_coszmn needs SOL_STATE from phys-rrtm-sol-update")
    coszen = st.add(OUT_DAYTIME, shape, dtype="float32")
    coszdg = st.add(OUT_WHOLE, shape, dtype="float32")
    sol = st[SOL_STATE]
    nrads = int(ctx.cfg.nrads)
    # FHSWR = (NRADS*DT)/3600 in kind_phys (single)
    dtswav = np.float32(np.float32(nrads) * np.float32(ctx.cfg.dt)) / np.float32(3600.0)
    # solhr = mod(fhr, f24), fhr = float(ihr) + float(imin)/60 from JDAT(5), JDAT(6)
    jdate = ctx.clock.jdate
    fhr = np.float32(jdate[4]) + np.float32(jdate[5]) / np.float32(60.0)
    solhr = np.float32(np.fmod(fhr, np.float32(24.0)))
    return coszmn(st["GLON"], st["GLAT"], coszen, coszdg, solhr, dtswav, nrads, sol, stream=ctx.stream)


def selftest(im: int = 954, jm: int = 835, nrads: int = 48, seed: int = 1):
    """One launch on a full-size plausible state; asserts the outputs are finite and in range."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    glon = cp.asarray(rng.uniform(-math.pi, math.pi, (jm, im)).astype(np.float32))
    glat = cp.asarray(rng.uniform(-1.0, 1.0, (jm, im)).astype(np.float32))
    coszen = cp.zeros((jm, im), dtype=cp.float32)
    coszdg = cp.zeros((jm, im), dtype=cp.float32)
    sol = cp.asarray(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    dtswav = np.float32(nrads * 25.0 / 3600.0)
    coszmn(glon, glat, coszen, coszdg, solhr=12.0, dtswav=dtswav, nrads=nrads, sol=sol)
    cp.cuda.Stream.null.synchronize()
    zen = coszen.get()
    dg = coszdg.get()
    assert np.all(np.isfinite(zen)) and np.all(np.isfinite(dg)), "NaN/Inf in coszmn output"
    assert zen.min() >= 0.0 and zen.max() <= 1.0 + 1e-6, (zen.min(), zen.max())
    assert dg.min() >= 0.0 and dg.max() <= 1.0 + 1e-6, (dg.min(), dg.max())
    return float(zen.mean()), float(dg.mean())


if __name__ == "__main__":
    print(selftest())
