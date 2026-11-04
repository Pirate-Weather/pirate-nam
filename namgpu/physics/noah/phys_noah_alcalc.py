"""phys-noah-alcalc: NOAHLSM snow-cover branch and ALCALC surface albedo (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:1976-2001 (caller branch)
and 2281-2328 (SUBROUTINE ALCALC). NOAA US government work; the module's NOAA header is kept as
the notice in the kernel file.
Kernel: namgpu/kernels/phys_noah_alcalc.cu (phys_noah_alcalc, one thread per (i, j), one launch).

The kernel is the NOAHLSM albedo branch, not just the ALCALC formula:
  sea ice (SICE > 0.5):   SNCOVR = 1.0,  ALBEDO = 0.65 (no ALCALC call)
  no snow (SNOW == 0):    SNCOVR = 0.0,  ALBEDO = ALB
  snow:                   SNCOVR = MIN(SNCOVR_from_SNFRAC, 0.98),
                          ALBEDO = ALB + SNCOVR*(SNOALB-ALB), clamped to SNOALB
Points outside the NOAHLSM B1 limits (i in [1, IM-2], j in [1, JM-2]) are not written.

Inputs from state: ALBASE (ALB), MXSNAL (SNOALB), SNOW (mm, SNEQV*1000), SICE, and SNCOVR as
written by the phys-noah-snfrac producer (SNUP scaling is the producer's). Outputs: ALBEDO and
the post-branch SNCOVR, written in place.

run(ctx): the done-rule entry point. With ctx it reads those fields and raises a named error if
a producer has not written one. Without ctx it builds a full-size 954 x 835 random batch with some
zero-snow and sea-ice points and returns the outputs.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_alcalc"
KERNEL = "phys_noah_alcalc"
BLOCK = kernels.BLOCK_2D


def _grid(im: int, jm: int):
    return kernels.grid_2d(im, jm, 1)


def alcalc(alb, snoalb, snow, sice, sncovr, albedo):
    """Apply the NOAHLSM albedo/snow-cover branch on [JM, IM] float32 device arrays.

    sncovr is the SNFRAC output on entry and the post-branch SNCOVR on return; albedo is written.
    """
    jm, im = int(alb.shape[0]), int(alb.shape[1])
    k = kernels.get(UNIT, KERNEL)
    # B1 limits, 0-based inclusive: i in [1, IM-2], j in [1, JM-2] (phys_noah_driver convention).
    k(_grid(im, jm), BLOCK,
      (alb, snoalb, snow, sice, sncovr, albedo,
       np.int32(1), np.int32(im - 2), np.int32(1), np.int32(jm - 2),
       np.int32(im), np.int32(jm)))
    return albedo


def _require(st, name: str):
    if name not in st.fields:
        raise RuntimeError(
            f"phys-noah-alcalc needs state field {name!r}; it is written by the producer "
            f"named in phys_noah_alcalc.py (SNCOVR: phys-noah-snfrac, SNOW/SICE: state table)")
    return st.fields[name]


def run(ctx=None, *, alb=None, snoalb=None, snow=None, sice=None, sncovr=None, albedo=None):
    """Done-rule entry: one launch on a full-size 954 x 835 field set. Returns the arrays it wrote."""
    import cupy as cp

    if ctx is not None and alb is None:
        st = ctx.st
        alb = _require(st, "ALBASE")
        snoalb = _require(st, "MXSNAL")
        snow = _require(st, "SNOW")
        sice = _require(st, "SICE")
        sncovr = _require(st, "SNCOVR")
        albedo = _require(st, "ALBEDO")
    elif alb is None:
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        alb = cp.asarray(rng.uniform(0.05, 0.4, size=(jm, im)).astype(np.float32))
        snoalb = cp.asarray(rng.uniform(0.6, 0.9, size=(jm, im)).astype(np.float32))
        snow_h = rng.uniform(0.0, 50.0, size=(jm, im)).astype(np.float32)
        snow_h[rng.uniform(size=(jm, im)) < 0.3] = 0.0          # bare ground points
        snow = cp.asarray(snow_h)
        sice_h = np.zeros((jm, im), dtype=np.float32)
        sice_h[rng.uniform(size=(jm, im)) < 0.05] = 1.0         # sea-ice points
        sice = cp.asarray(sice_h)
        sncovr = cp.asarray(rng.uniform(0.0, 1.0, size=(jm, im)).astype(np.float32))
        albedo = cp.zeros((jm, im), dtype=cp.float32)
    else:
        jm, im = int(alb.shape[0]), int(alb.shape[1])
        if albedo is None:
            albedo = cp.zeros((jm, im), dtype=cp.float32)
    alcalc(alb, snoalb, snow, sice, sncovr, albedo)
    return {"ALBEDO": albedo, "SNCOVR": sncovr}
