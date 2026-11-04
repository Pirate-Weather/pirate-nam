"""phys-noah-csnow: CSNOW, snow thermal conductivity (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:2472-2511. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel file. The formula is
copied from ArWen gpuwm/core/kernels/noah.cu (Apache-2.0), see the kernel file header.
Kernel: namgpu/kernels/phys_noah_csnow.cu (phys_noah_csnow, one thread per (i, j), one launch).

SNCOND = 2 * 0.11631 * 0.328 * 10**(2.25 * DSNOW), the Dyachkova form of the executed
SUBROUTINE CSNOW. No host sync, no per-step allocation; the launch runs on ctx.stream.

State (CONTRACT sections 1 and 3): SNDENS (the Fortran DSNOW input) and SNCOND (the output) are
local to SNOPAC/SFLX in the Fortran, not state-table fields. With ctx they are registered on the
state with ctx.st.add("SNDENS", (JM, IM)) and ctx.st.add("SNCOND", (JM, IM)), allocated once and
reused every call. Nothing else is cached at module level.
  - SNDENS is written by its producer (snow density, SNEQV/SNOWH in SNOPAC/SFLX) before this unit runs.
  - SNCOND is read by the SFLX unit.

OPEN ITEM (owner: phys-noah-sflx, the SFLX wiring unit, which calls this unit): the Fortran calls
CSNOW only on the SNEQV /= 0 branch (module_LS_NOAHLSM.F90:1892-1905, and :1960). When SNEQV == 0
the Fortran sets SNCOND = 1.0 and never calls CSNOW. This unit computes every cell, so the SFLX
wiring must either launch only on SNEQV /= 0 cells or override SNCOND = 1.0 where SNEQV == 0.
SNDENS = SNEQV/SNOWH is also infinite where SNOWH = 0 and must be guarded by that unit.
Until the SFLX unit lands, SNCOND from this unit is not a final value on snow-free cells.

run(ctx): the done-rule entry point. With ctx it uses the ctx.st SNDENS and SNCOND fields on
ctx.stream. Without ctx it builds a full-size random 954 x 835 batch of snow densities (0.05-0.5)
and returns the outputs.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_csnow"
KERNEL = "phys_noah_csnow"
BLOCK = (32, 8, 1)
SNDENS_NAME = "SNDENS"
SNCOND_NAME = "SNCOND"


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def _check_field(name, a, shape):
    """The kernel reads and writes raw float* over a [JM, IM] C-order block. Anything else is refused
    here, because a float64, integer, strided or wrong-shape array would be read as float32 bytes and
    give silently wrong SNCOND."""
    if a.dtype != np.float32:
        raise ValueError(f"phys-noah-csnow: {name} must be float32, got {a.dtype} (kernel reads raw float*)")
    if tuple(a.shape) != tuple(shape):
        raise ValueError(f"phys-noah-csnow: {name} must have shape {tuple(shape)}, got {tuple(a.shape)}")
    if not a.flags.c_contiguous:
        raise ValueError(f"phys-noah-csnow: {name} must be C-contiguous (kernel indexes IDX2 = j*IM+i)")


def csnow(dsnow, sncond, stream=None):
    """SNCOND[j, i] = 2 * 0.11631 * 0.328 * 10**(2.25 * DSNOW[j, i]). Float32 [JM, IM] on device."""
    if dsnow.ndim != 2:
        raise ValueError(f"phys-noah-csnow: SNDENS must be 2-D [JM, IM], got ndim {dsnow.ndim}")
    jm, im = int(dsnow.shape[0]), int(dsnow.shape[1])
    _check_field("SNDENS", dsnow, (jm, im))
    _check_field("SNCOND", sncond, (jm, im))
    k = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        k(_grid(im, jm), BLOCK, (dsnow, sncond, np.int32(im), np.int32(jm)))
    return sncond


def run(ctx=None, *, dsnow=None, sncond=None):
    """Done-rule entry: one launch on a full-size 954 x 835 field. Returns the arrays it wrote."""
    import cupy as cp

    stream = None
    if ctx is not None:
        stream = ctx.stream
        jm, im = int(ctx.grid.jm), int(ctx.grid.im)
        if dsnow is None:
            dsnow = ctx.st.add(SNDENS_NAME, (jm, im))
        if sncond is None:
            sncond = ctx.st.add(SNCOND_NAME, (jm, im))
    else:
        if dsnow is None:
            jm, im = 835, 954
            rng = np.random.default_rng(1)
            dsnow = cp.asarray(rng.uniform(0.05, 0.5, size=(jm, im)).astype(np.float32))
        if sncond is None:
            sncond = cp.zeros(dsnow.shape, dtype=cp.float32)
    csnow(dsnow, sncond, stream)
    return {"SNCOND": sncond}
