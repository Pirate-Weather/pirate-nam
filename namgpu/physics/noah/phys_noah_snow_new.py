"""phys-noah-snow-new: SNOW_NEW, new-snow density and snow depth (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:4530-4600. NOAA US government
work; the module's NOAA header is kept as the notice in the kernel file. Start point: ArWen
third_party/arwen/core/kernels/noah.cu (Apache-2.0, notice kept in the kernel header).
Kernel: namgpu/kernels/phys_noah_snow_new.cu (phys_noah_snow_new, one thread per (i, j), one launch).
No host sync, no per-step allocation beyond the first-call scratch, everything on the current stream.

Inputs (float32 [JM, IM]): TEMP (K), RIMEF1, NEWSN (m), and the in/out SNOWH (m), SNDENS.
The Fortran dummy arguments T1, FFROZP and PRCP are not used by the executed body and are not inputs.
Outputs: SNOWH and SNDENS, updated in place.

run(ctx): the done-rule entry point. With ctx the arrays come from scratch allocated once; without ctx
it builds a full-size 954 x 835 random-but-plausible batch (TEMP 250-290 K, RIMEF1 1-25, NEWSN
0-0.02 m, SNOWH 0-1 m, SNDENS 0.05-0.5) and returns the outputs.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_snow_new"
KERNEL = "phys_noah_snow_new"
BLOCK = (32, 8, 1)

_scratch: dict = {}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def snow_new(temp, rimef1, newsn, snowh, sndens):
    """Update SNOWH and SNDENS in place for one new-snow increment. Float32 [JM, IM] on device."""
    jm, im = int(temp.shape[0]), int(temp.shape[1])
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK, (temp, rimef1, newsn, snowh, sndens, np.int32(im), np.int32(jm)))
    return snowh, sndens


def _scratch_arrays(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        _scratch[key] = {
            "TEMP": cp.zeros((jm, im), dtype=cp.float32),
            "RIMEF1": cp.zeros((jm, im), dtype=cp.float32),
            "NEWSN": cp.zeros((jm, im), dtype=cp.float32),
            "SNOWH": cp.zeros((jm, im), dtype=cp.float32),
            "SNDENS": cp.zeros((jm, im), dtype=cp.float32),
        }
    return _scratch[key]


def run(ctx=None, *, temp=None, rimef1=None, newsn=None, snowh=None, sndens=None):
    """Done-rule entry: one launch on a full-size 954 x 835 field. Returns the arrays it wrote."""
    import cupy as cp

    if temp is None:
        if ctx is not None:
            jm, im = int(ctx.grid.jm), int(ctx.grid.im)
        else:
            jm, im = 835, 954
        sc = _scratch_arrays(jm, im)
        rng = np.random.default_rng(1)
        sc["TEMP"][...] = cp.asarray(rng.uniform(250.0, 290.0, size=(jm, im)).astype(np.float32))
        sc["RIMEF1"][...] = cp.asarray(rng.uniform(1.0, 25.0, size=(jm, im)).astype(np.float32))
        sc["NEWSN"][...] = cp.asarray(rng.uniform(0.0, 0.02, size=(jm, im)).astype(np.float32))
        sc["SNOWH"][...] = cp.asarray(rng.uniform(0.0, 1.0, size=(jm, im)).astype(np.float32))
        sc["SNDENS"][...] = cp.asarray(rng.uniform(0.05, 0.5, size=(jm, im)).astype(np.float32))
        temp, rimef1, newsn = sc["TEMP"], sc["RIMEF1"], sc["NEWSN"]
        snowh, sndens = sc["SNOWH"], sc["SNDENS"]
    snow_new(temp, rimef1, newsn, snowh, sndens)
    return {"SNOWH": snowh, "SNDENS": sndens}
