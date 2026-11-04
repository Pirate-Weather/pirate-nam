"""phys-bmj-spline: SPLINE, the one-dimensional cubic spline behind the BMJ convection tables.

Ported from phys/module_CU_BMJ.F90:2429-2547 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE spline,
called from bmj_init (table build, host-side init only, never on the step path).
Kernel: namgpu/kernels/phys_bmj_spline.cu (phys_bmj_spline_batch: one thread per spline, one launch).

Batch form: the Fortran routine is called once per table row; the port solves all rows of one table
in a single launch. Inputs are [nspl, nold] knots (xold ascending, yold values, y2 with the natural
ends y2(1) = y2(nold) = 0 set by the caller) and [nspl, nnew] evaluation points xnew. Outputs are
y2 (all knots) and ynew [nspl, nnew]. p and q are scratch [nspl, nold], cached per shape.

Launch: no host sync, no per-step allocation, no Python loop over cells; everything runs on the
current stream (ctx.stream inside `with ctx.stream:`).

Refused: nold < 3 (the Fortran requires nold >= 3; it would read xold(3) out of range).

run(ctx): the done-rule entry point. With no arrays given it builds a full-size plausible batch
(itbq = 152 tables of jtbq = 440 knots, the BMJ table dimensions in module_CU_BMJ.F90:66), runs the
spline once and returns the outputs; nothing is written to ctx.st.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_spline"
KERNEL = "phys_bmj_spline_batch"
BLOCK = 128

# module_CU_BMJ.F90:66  itb=76, jtb=134, itbq=152, jtbq=440
ITBQ = 152
JTBQ = 440

_scratch: dict = {}


def _scratch_pq(nspl: int, nold: int):
    """p and q scratch for one batch shape, allocated once and reused."""
    import cupy as cp

    key = (nspl, nold)
    if key not in _scratch:
        _scratch[key] = (cp.zeros((nspl, nold), dtype=cp.float32),
                         cp.zeros((nspl, nold), dtype=cp.float32))
    return _scratch[key]


def spline_batch(nspl, nold, nnew, xold, yold, y2, xnew, ynew, p=None, q=None):
    """Solve nspl natural cubic splines and evaluate them at xnew.

    xold, yold, y2: [nspl, nold] float32 device arrays (y2 is updated in place: all knots).
    xnew, ynew:     [nspl, nnew] float32 device arrays (ynew written in place).
    p, q:           optional [nspl, nold] scratch; allocated and cached when omitted.
    """
    if nold < 3:
        raise ValueError(f"phys-bmj-spline: nold must be >= 3 (got {nold})")
    if p is None or q is None:
        p, q = _scratch_pq(nspl, nold)
    k = kernels.get(UNIT, KERNEL)
    grid = ((nspl + BLOCK - 1) // BLOCK, 1, 1)
    k(grid, (BLOCK, 1, 1), (np.int32(nspl), np.int32(nold), np.int32(nnew),
                            xold, yold, y2, xnew, ynew, p, q))
    return y2, ynew


def run(ctx=None, *, xold=None, yold=None, y2=None, xnew=None, ynew=None, nspl=None, nold=None, nnew=None):
    """Done-rule entry: one launch on a full-size batch. Returns a dict of the arrays it wrote."""
    import cupy as cp

    if xold is None:
        nspl = ITBQ if nspl is None else nspl
        nold = JTBQ if nold is None else nold
        nnew = nold if nnew is None else nnew
        rng = np.random.default_rng(1)
        # ascending knots with plausible spacing, evaluation points inside [xold(1), xold(nold)]
        steps = rng.uniform(0.5, 1.5, size=(nspl, nold - 1)).astype(np.float32)
        xo = np.concatenate([np.zeros((nspl, 1), np.float32), np.cumsum(steps, axis=1)], axis=1)
        yo = (np.sin(xo * 0.05) + 0.1 * rng.standard_normal((nspl, nold))).astype(np.float32)
        xn = np.sort(rng.uniform(xo[:, :1], xo[:, -1:], size=(nspl, nnew)), axis=1).astype(np.float32)
        xold = cp.asarray(xo)
        yold = cp.asarray(yo)
        xnew = cp.asarray(xn)
        y2 = cp.zeros((nspl, nold), dtype=cp.float32)      # natural ends stay 0 (set by the caller)
        ynew = cp.zeros((nspl, nnew), dtype=cp.float32)
    else:
        nspl, nold = int(xold.shape[0]), int(xold.shape[1])
        nnew = int(xnew.shape[1])
        if ynew is None:
            ynew = cp.zeros((nspl, nnew), dtype=cp.float32)
        if y2 is None:
            y2 = cp.zeros((nspl, nold), dtype=cp.float32)
    spline_batch(nspl, nold, nnew, xold, yold, y2, xnew, ynew)
    return {"y2": y2, "ynew": ynew}
