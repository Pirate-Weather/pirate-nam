"""phys-noah-snowpack: SNOWPACK, snowpack compaction (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:4378-4505. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel file. The kernel
starts from the ArWen Noah snowpack body (namgpu/third_party/arwen/core/kernels/noah.cu, Apache-2.0)
and follows the NAM Fortran where they differ (NAM density limits 0.05-0.9, melt cap 0.9).
The SNOPAC call-site gate is in the kernel: columns with ESD <= 0 (or NaN) get ESD, SNOWH and
SNDENS set to 0 and no compaction, as the Fortran ELSE branch does.
Kernel: namgpu/kernels/phys_noah_snowpack.cu (phys_noah_snowpack, one thread per (i, j), one
launch, grid.z = 1).

Inputs TSNOW and TSOIL [JM, IM] (K); DTSEC scalar (s).
In/out ESD [JM, IM] (m water equivalent), SNOWH [JM, IM] (m) and SNDENS [JM, IM] (fraction of H2O
density), updated in place. ESD is INOUT because the ESD <= 0 branch resets it to zero.
No host sync, no per-step allocation, everything on the current stream.

run(ctx): the done-rule entry point. SNOWH, SNDENS, ESD, TSNOW and TSOIL are local to SFLX in
NAM, not state-table fields, so with ctx they come from scratch arrays allocated once (filled
with plausible random values for the done-rule check; SNOPAC, not yet ported, will fill ESD,
TSNOW and TSOIL). The random ESD spans negative values so the ESD <= 0 gate runs at full size. Without ctx it builds a full-size random 954 x 835 batch and returns the
outputs.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_snowpack"
KERNEL = "phys_noah_snowpack"
BLOCK = (32, 8, 1)
DTSEC_SNOWPACK = 50.0   # DTPHS = 2 * dt = 50 s (CONTRACT section 2, Noah TURBL cadence)

_scratch: dict = {}


def _grid(im: int, jm: int):
    return kernels.grid_2d(im, jm, 1)


def snowpack(esd, dtsec, snowh, sndens, tsnow, tsoil):
    """Updates ESD (ESD <= 0 reset), SNOWH and SNDENS in place from ESD, TSNOW, TSOIL [JM, IM] float32 (device)."""
    jm, im = int(esd.shape[0]), int(esd.shape[1])
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (esd, np.float32(dtsec), snowh, sndens, tsnow, tsoil, np.int32(im), np.int32(jm)))
    return snowh, sndens


def _random_batch(jm: int, im: int, seed: int):
    import cupy as cp

    rng = np.random.default_rng(seed)
    return dict(
        esd=cp.asarray(rng.uniform(-0.05, 0.3, size=(jm, im)).astype(np.float32)),
        snowh=cp.asarray(rng.uniform(0.01, 2.0, size=(jm, im)).astype(np.float32)),
        sndens=cp.asarray(rng.uniform(0.05, 0.5, size=(jm, im)).astype(np.float32)),
        tsnow=cp.asarray(rng.uniform(250.0, 275.0, size=(jm, im)).astype(np.float32)),
        tsoil=cp.asarray(rng.uniform(250.0, 275.0, size=(jm, im)).astype(np.float32)),
    )


def _scratch_arrays(jm: int, im: int):
    key = (jm, im)
    if key not in _scratch:
        _scratch[key] = _random_batch(jm, im, seed=2)
    return _scratch[key]


def run(ctx=None, *, esd=None, dtsec=DTSEC_SNOWPACK, snowh=None, sndens=None,
        tsnow=None, tsoil=None):
    """Done-rule entry: one launch on a full-size 954 x 835 field. Returns the arrays it wrote."""
    if esd is None and ctx is not None:
        jm, im = int(ctx.grid.jm), int(ctx.grid.im)
        s = _scratch_arrays(jm, im)
        esd, snowh, sndens = s["esd"], s["snowh"], s["sndens"]
        tsnow, tsoil = s["tsnow"], s["tsoil"]
    elif esd is None:
        jm, im = 835, 954
        s = _random_batch(jm, im, seed=1)
        esd, snowh, sndens = s["esd"], s["snowh"], s["sndens"]
        tsnow, tsoil = s["tsnow"], s["tsoil"]
    else:
        if tsnow is None or tsoil is None or snowh is None or sndens is None:
            raise ValueError("phys_noah_snowpack: pass esd, snowh, sndens, tsnow and tsoil together")
    snowpack(esd, dtsec, snowh, sndens, tsnow, tsoil)
    return {"ESD": esd, "SNOWH": snowh, "SNDENS": sndens}
