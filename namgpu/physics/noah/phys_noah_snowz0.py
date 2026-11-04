"""phys-noah-snowz0: SNOWZ0, total roughness length over snow (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:4508-4526. NOAA US
government work; the module NOAA header is kept as the notice in the kernel file. Start point:
ArWen core/kernels/noah.cu (Apache-2.0, namgpu/third_party/arwen/LICENSE), whose burial-depth blend is
not NAM's SNOWZ0 and is not used here.
Kernel: namgpu/kernels/phys_noah_snowz0.cu (phys_noah_snowz0, one thread per (i, j), one launch).

NAM SNOWZ0 is Z0 = Z0BRD. SNCOVR is accepted for the Fortran signature and not read.

Inputs: SNCOVR, Z0BRD, float32 [JM, IM]. Output: Z0 float32 [JM, IM].

run(ctx): the done-rule entry point. With ctx it uses ctx.st fields of the same names when they
exist and adds them otherwise (plausible test values). Without ctx it builds a full-size random
954 x 835 batch.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_snowz0"
KERNEL = "phys_noah_snowz0"
BLOCK = kernels.BLOCK_2D

# Test-only plausible ranges (background roughness over land, snow cover fraction).
_RANGES = {"SNCOVR": (0.0, 1.0), "Z0BRD": (0.01, 0.5)}


def _grid(im: int, jm: int):
    return kernels.grid_2d(im, jm, 1)


def snowz0(sncovr, z0brd, z0):
    """Z0[j, i] = Z0BRD[j, i] (SNCOVR unused by NAM SNOWZ0). Float32 [JM, IM] on device."""
    jm, im = int(sncovr.shape[0]), int(sncovr.shape[1])
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK, (sncovr, z0brd, z0, np.int32(im), np.int32(jm)))
    return z0


def _field(st, name, shape, rng):
    """Return ctx.st.<name>, adding it once with test values if the state has no such field."""
    import cupy as cp

    if name in st.fields:
        return st.fields[name]
    arr = st.add(name, shape)
    lo, hi = _RANGES[name]
    arr[...] = cp.asarray(rng.uniform(lo, hi, size=shape).astype(np.float32))
    return arr


def run(ctx=None):
    """Done-rule entry: one launch on a full-size 954 x 835 field set. Returns the arrays it wrote."""
    import cupy as cp

    rng = np.random.default_rng(1)
    if ctx is None:
        jm, im = 835, 954
        sncovr = cp.asarray(rng.uniform(*_RANGES["SNCOVR"], size=(jm, im)).astype(np.float32))
        z0brd = cp.asarray(rng.uniform(*_RANGES["Z0BRD"], size=(jm, im)).astype(np.float32))
        z0 = cp.zeros((jm, im), dtype=cp.float32)
    else:
        st = ctx.st
        jm, im = int(st.fields["T"].shape[1]), int(st.fields["T"].shape[2])
        shp = (jm, im)
        sncovr = _field(st, "SNCOVR", shp, rng)
        z0brd = _field(st, "Z0BRD", shp, rng)
        z0 = st.fields["Z0"] if "Z0" in st.fields else st.add("Z0", shp)
    snowz0(sncovr, z0brd, z0)
    return {"Z0": z0}
