"""drv-dfi-dolph: DOLPH, the Dolph-Chebyshev window for the TDFI digital filter weights.

Fortran: nmm/module_DIGITAL_FILTER_NMM.F90:929-1039 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE DOLPH
and its helper CHEBY, called from DIGITAL_FILTER_DYN_INIT_NMM (line 108) as
    taus = float(2*ndfistep) * dt ;  call dolph(dt, taus, nstep, dolph_wgts)
with nstep = ndfistep (line 91).
Kernel: namgpu/kernels/drv_dfi_dolph.cu (drv_dfi_dolph; one thread per NT = 0..M, one launch,
        no host sync, runs once at filter init).

Arguments (Fortran order): deltat (filter dt, s), taus (cut-off time scale, s), m (= nstep = ndfistep).
    defaults when not given: deltat = cfg.filt_dt (8.0 s), m = cfg.ndfistep (75 = 600 s / 8 s),
    taus = 2*m*deltat.
Writes: ctx.tables["dolph_wgts"], float32 [2*m+1]; index n = k + m holds the weight of time step
    k (k = -m..m), i.e. WINDOW(0:2M) of the Fortran, which the filter maps to dolph_wgts(-nstep:nstep).
Returns the device array. Scratch: none (the window is the output).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_dfi_dolph"
KERNEL = "drv_dfi_dolph"
BLOCK = (128, 1, 1)


def run(ctx, deltat=None, taus=None, m=None):
    import cupy as cp

    cfg = ctx.cfg
    if deltat is None:
        deltat = float(getattr(cfg, "filt_dt", 8.0))
    if m is None:
        m = int(getattr(cfg, "ndfistep", 75))
    m = int(m)
    if taus is None:
        taus = 2.0 * m * float(deltat)

    window = cp.zeros((2 * m + 1,), dtype=cp.float32)
    ctx.tables["dolph_wgts"] = window

    kern = kernels.get(UNIT, KERNEL)
    grid = ((m + 1 + BLOCK[0] - 1) // BLOCK[0], 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, BLOCK, (window, np.float32(deltat), np.float32(taus), np.int32(m)))
    return window
