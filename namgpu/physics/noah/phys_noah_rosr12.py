"""phys-noah-rosr12: ROSR12, tridiagonal solver for the Noah LSM soil column.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:3666-3725. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel file. The kernel
starts from ArWen's noah_rosr12 (Apache-2.0) and follows the NAM Fortran operation order.
Kernel: namgpu/kernels/phys_noah_rosr12.cu (phys_noah_rosr12, one thread per (i, j) column,
one launch, grid_cols). Soil arrays are [NSOIL=4, JM, IM] float32, layer 1 at index 0.

Solves the NSOIL x NSOIL tridiagonal system with sub-diagonal A, diagonal B, super-diagonal C
(INOUT: C[NSOIL-1] is set to 0 as in the Fortran), right-hand side D, solution P and the forward
sweep DELTA (INOUT). Callers are SSTEP, HSTEP and SHFLX. Each of them builds its own per-layer
A, B, C, D inside its column loop; those coefficients are not state fields, so there is nothing
for ROSR12 to read from ctx.st. No host sync, no per-step allocation, everything on the current
stream.

run(ctx): the done-rule entry point. With ctx the caller must pass its real A, B, C, D (and
optionally P, DELTA) as keyword arguments. A ctx call without them raises KeyError naming the
producers. Nothing is zero-filled on the ctx path, so no run can divide 0 by 0. Without ctx it
builds a full-size 954 x 835 random batch that is diagonally dominant (|A| + |C| < B) so the
pivots stay finite, and returns the outputs.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_rosr12"
KERNEL = "phys_noah_rosr12"
NSOIL = 4

_PRODUCERS = ("SSTEP (phys_noah_sstep, Fortran 4855-4942), HSTEP (phys_noah_hstep, 3153-3201) and "
              "SHFLX (phys_noah_shflx) build A, B, C, D per column; none is a ctx.st field")


def _check(name: str, arr, shape: tuple) -> None:
    if tuple(arr.shape) != shape:
        raise ValueError(f"ROSR12 {name} has shape {tuple(arr.shape)}, expected {shape}")
    if str(arr.dtype) != "float32":
        raise ValueError(f"ROSR12 {name} has dtype {arr.dtype}, expected float32")
    if not arr.flags.c_contiguous:
        raise ValueError(f"ROSR12 {name} must be C-contiguous")


def rosr12(a, b, c, d, p, delta):
    """Solve per column. All arrays float32 [NSOIL, JM, IM] on device; p and delta are outputs,
    c is INOUT (its last layer is set to 0). Returns (p, delta)."""
    nsoil, jm, im = (int(s) for s in a.shape)
    if nsoil != NSOIL:
        raise ValueError(f"ROSR12 expects NSOIL={NSOIL}, got {nsoil}")
    shape = (nsoil, jm, im)
    for name, arr in (("A", a), ("B", b), ("C", c), ("D", d), ("P", p), ("DELTA", delta)):
        _check(name, arr, shape)
    k = kernels.get(UNIT, KERNEL)
    k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
      (a, b, c, d, p, delta, np.int32(im), np.int32(jm)))
    return p, delta


def residual_max(a, b, c_orig, d, p) -> float:
    """Test helper (host, float64): max |A x_{k-1} + B x_k + C x_{k+1} - D| over all columns, with
    x = P and C_orig the super-diagonal before the call (its last layer is taken as 0, as in the
    Fortran). Used by the review check only; not on any run path."""
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    C = np.asarray(c_orig, dtype=np.float64).copy()
    D = np.asarray(d, dtype=np.float64)
    X = np.asarray(p, dtype=np.float64)
    C[-1] = 0.0
    r = B * X + C * np.concatenate([X[1:], np.zeros_like(X[:1])], axis=0)
    r[1:] += A[1:] * X[:-1]
    return float(np.max(np.abs(r - D)))


def _scratch_check_inputs(name: str, val):
    if val is None:
        raise KeyError(f"phys-noah-rosr12 needs {name} from its caller; {_PRODUCERS}")
    return val


def run(ctx=None, *, a=None, b=None, c=None, d=None, p=None, delta=None):
    """Done-rule entry: one launch on a full-size 954 x 835 soil column batch.

    With ctx: a, b, c, d are required keyword arguments (the caller's real coefficients).
    Returns {"P": p, "DELTA": delta, "C": c} (C is the INOUT array, last layer zeroed).
    """
    import cupy as cp

    if a is None:
        if ctx is not None:
            _scratch_check_inputs("A", a)
            _scratch_check_inputs("B", b)
            _scratch_check_inputs("C", c)
            _scratch_check_inputs("D", d)
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        a = cp.asarray(rng.uniform(-0.5, 0.5, size=(NSOIL, jm, im)).astype(np.float32))
        b = cp.asarray(rng.uniform(2.0, 3.0, size=(NSOIL, jm, im)).astype(np.float32))
        c = cp.asarray(rng.uniform(-0.5, 0.5, size=(NSOIL, jm, im)).astype(np.float32))
        d = cp.asarray(rng.uniform(-1.0, 1.0, size=(NSOIL, jm, im)).astype(np.float32))
        p = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
        delta = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
    else:
        _scratch_check_inputs("B", b)
        _scratch_check_inputs("C", c)
        _scratch_check_inputs("D", d)
        if p is None:
            p = cp.zeros_like(a)
        if delta is None:
            delta = cp.zeros_like(a)
    rosr12(a, b, c, d, p, delta)
    return {"P": p, "DELTA": delta, "C": c}
