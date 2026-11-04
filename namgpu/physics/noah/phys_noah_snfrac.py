"""phys-noah-snfrac: SNFRAC, snow cover fraction (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:3931-3968. NOAA US
government work; the module NOAA header is kept as the notice in the kernel file. Start point:
ArWen core/kernels/noah.cu (Apache-2.0, third_party/arwen/LICENSE).
Kernel: namgpu/kernels/phys_noah_snfrac.cu (phys_noah_snfrac, one thread per (i, j), one launch).

SNCOVR = 1 - (exp(-SALP*r) - r*exp(-SALP)) with r = SNEQV/SNUP when SNEQV < SNUP, else 1.
The SFLX caller sets SNUP = 4*SNUP before the call (module_LS_NOAHLSM.F90:1989); that scaling is
done by the phys-noah-sflx unit, not here. No host sync, no per-step allocation, current stream.

Inputs: SNEQV, SNUP, SALP, SNOWH, all float32 [JM, IM] (SNOWH is accepted for the Fortran
signature and not read). Output: SNCOVR float32 [JM, IM].

run(ctx): the done-rule entry point. With ctx it reads SNEQV, SNUP, SALP and SNOWH from ctx.st and
raises KeyError if any is absent (no test values are substituted); SNCOVR is the output, allocated
on ctx.st once. Without ctx it builds a full-size random 954 x 835 batch for the standalone check.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_snfrac"
KERNEL = "phys_noah_snfrac"
BLOCK = kernels.BLOCK_2D

# Test-only plausible ranges (SNUP is the vegetation-class threshold, SALP the tuning value).
_RANGES = {"SNEQV": (0.0, 0.3), "SNUP": (0.02, 0.10), "SALP": (1.0, 4.0), "SNOWH": (0.0, 0.5)}


def _grid(im: int, jm: int):
    return kernels.grid_2d(im, jm, 1)


def snfrac(sneqv, snup, salp, snowh, sncovr):
    """SNCOVR[j, i] from SNEQV, SNUP, SALP (SNOWH unused). Float32 [JM, IM] on device."""
    if sneqv.ndim != 2:
        raise ValueError(f"phys-noah-snfrac: SNEQV must be 2-D [JM, IM], got ndim {sneqv.ndim}")
    jm, im = int(sneqv.shape[0]), int(sneqv.shape[1])
    for name, a in (("SNEQV", sneqv), ("SNUP", snup), ("SALP", salp), ("SNOWH", snowh), ("SNCOVR", sncovr)):
        _check_field(name, a, (jm, im))
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (sneqv, snup, salp, snowh, sncovr, np.int32(im), np.int32(jm)))
    return sncovr


def _state_field(st, name):
    """Return ctx.st.<name>. SNEQV, SNUP, SALP and SNOWH are Fortran inputs that the upstream
    units (SFLX, the state tables) must provide. A missing field is an error: a test value here
    would compute SNCOVR from invented inputs without any sign of it."""
    if name not in st.fields:
        raise KeyError(f"phys-noah-snfrac: state field {name} is absent; it must be provided before SNFRAC runs")
    return st.fields[name]


def _check_field(name, a, shape):
    """The kernel reads and writes raw float* over a [JM, IM] C-order block (IDX2 = j*IM+i)."""
    if a.dtype != np.float32:
        raise ValueError(f"phys-noah-snfrac: {name} must be float32, got {a.dtype} (kernel reads raw float*)")
    if tuple(a.shape) != tuple(shape):
        raise ValueError(f"phys-noah-snfrac: {name} must have shape {tuple(shape)}, got {tuple(a.shape)}")
    if not a.flags.c_contiguous:
        raise ValueError(f"phys-noah-snfrac: {name} must be C-contiguous (kernel indexes IDX2 = j*IM+i)")


def run(ctx=None):
    """Done-rule entry: one launch on a full-size 954 x 835 field set. Returns the arrays it wrote."""
    import cupy as cp

    rng = np.random.default_rng(1)
    if ctx is None:
        jm, im = 835, 954
        sneqv = cp.asarray(rng.uniform(*_RANGES["SNEQV"], size=(jm, im)).astype(np.float32))
        snup = cp.asarray(rng.uniform(*_RANGES["SNUP"], size=(jm, im)).astype(np.float32))
        salp = cp.asarray(rng.uniform(*_RANGES["SALP"], size=(jm, im)).astype(np.float32))
        snowh = cp.asarray(rng.uniform(*_RANGES["SNOWH"], size=(jm, im)).astype(np.float32))
        sncovr = cp.zeros((jm, im), dtype=cp.float32)
    else:
        st = ctx.st
        jm, im = int(st.fields["T"].shape[1]), int(st.fields["T"].shape[2])
        shp = (jm, im)
        sneqv = _state_field(st, "SNEQV")
        snup = _state_field(st, "SNUP")
        salp = _state_field(st, "SALP")
        snowh = _state_field(st, "SNOWH")
        sncovr = st.fields["SNCOVR"] if "SNCOVR" in st.fields else st.add("SNCOVR", shp)
    snfrac(sneqv, snup, salp, snowh, sncovr)
    return {"SNCOVR": sncovr}
