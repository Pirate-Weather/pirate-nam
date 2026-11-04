"""drv-spline: batched natural cubic spline for the NMMB look-up tables (host-side init only).

Port of nmm/module_CONTROL.F90:2074-2188 (subroutine spline, Z. Janjic), nam.v4.2.11, 83df4e13c1.
Kernel: namgpu/kernels/drv_spline.cu, drv_spline_batch (one thread per spline).

Used by drv-tablep and drv-tablet to build the TABLEP/TABLET tables; they call spline_batch with
their own knot arrays.  The routine's outputs are y2 (second derivatives at the knots) and ynew
(values at xnew).  The caller sets y2 at the two end knots to 0 (natural spline), as the Fortran
routine requires.

Batch layout (all float32, C-order, device arrays):
    xold, yold, y2   [nspl, nold]   nold >= 3, xold ascending per row
    xnew, ynew       [nspl, nnew]
    p, q             [nspl, nold]   scratch, allocated here when not given
A single spline is one thread (the tridiagonal sweep is sequential in k); batching the independent
splines across threads is what makes this a parallel stage.  Launch only, no host sync.

run(ctx) is the done-rule self test on the grid: one spline per column of T (nspl = IM*JM,
nold = LM knots), evaluated at 2*LM-1 half-levels.  It raises if any output is not finite.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "drv_spline"
KERNEL = "drv_spline_batch"
BLOCK = (128, 1, 1)


def _need(name, a, shape):
    """Caller-supplied buffer must be a C-contiguous float32 array of exactly this shape: the kernel
    indexes it flat, so a wrong size or stride would write out of bounds or to the wrong knots."""
    import cupy as cp

    if a.dtype != cp.float32 or a.ndim != 2 or not a.flags.c_contiguous or tuple(a.shape) != shape:
        raise ValueError(f"drv-spline: {name} must be a C-contiguous float32 array of shape {shape} "
                         f"(got {a.dtype}, shape {tuple(a.shape)}, c_contiguous={a.flags.c_contiguous})")


def spline_batch(xold, yold, y2, xnew, ynew=None, p=None, q=None):
    """Fit nspl natural cubic splines and evaluate them at xnew.  Returns ynew (device, float32)."""
    import cupy as cp

    for name, a in (("xold", xold), ("yold", yold), ("y2", y2), ("xnew", xnew)):
        if a.dtype != cp.float32 or a.ndim != 2 or not a.flags.c_contiguous:
            raise ValueError(f"drv-spline: {name} must be a C-contiguous float32 [nspl, n] array")
    nspl, nold = (int(v) for v in xold.shape)
    if nold < 3:
        raise ValueError(f"drv-spline: nold must be >= 3 (got {nold})")
    if yold.shape != (nspl, nold) or y2.shape != (nspl, nold) or xnew.shape[0] != nspl:
        raise ValueError("drv-spline: xold, yold, y2 must share shape [nspl, nold]; xnew is [nspl, nnew]")
    nnew = int(xnew.shape[1])
    if nnew < 1:
        raise ValueError(f"drv-spline: nnew must be >= 1 (got {nnew}); the Fortran routine writes ynew(1) even when nnew=0, the kernel does not")
    if ynew is None:
        ynew = cp.empty((nspl, nnew), dtype=cp.float32)
    else:
        _need("ynew", ynew, (nspl, nnew))
    if p is None:
        p = cp.empty((nspl, nold), dtype=cp.float32)
    else:
        _need("p", p, (nspl, nold))
    if q is None:
        q = cp.empty((nspl, nold), dtype=cp.float32)
    else:
        _need("q", q, (nspl, nold))

    grid = ((nspl + BLOCK[0] - 1) // BLOCK[0], 1, 1)
    fn = kernels.get(UNIT, KERNEL)
    fn(grid, BLOCK, (np.int32(nspl), np.int32(nold), np.int32(nnew),
                     xold, yold, y2, xnew, ynew, p, q))
    return ynew


def run(ctx):
    """Init-chain stage: nothing to do. SPLINE is a subroutine that drv-tablep and drv-tablet call
    (spline_batch); it has no state of its own. Integrator fix 2026-11-04: run() used to be the self
    test below, which allocated a second full random State (10.6 GiB) inside the real init chain and
    ran the 16 GB card out of memory."""
    return None


def selftest(ctx):
    """Done-rule self test on a FULL-SIZE random state (CONTRACT.md section 6, step 2).

    State(grid).randomize() builds the 954 x 835 x 60 random state; every spline table is the T
    column of that state, p and q are passed explicitly (validated), and every array the kernel
    writes must be finite.
    """
    import cupy as cp
    from namgpu.state import State

    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    nspl = im * jm
    st = State(g).randomize()
    T = st.T                                                  # [LM, JM, IM], random 200-310 K
    yold = cp.ascontiguousarray(cp.transpose(T, (1, 2, 0)).reshape(nspl, lm))
    xold = cp.ascontiguousarray(
        cp.broadcast_to(cp.arange(lm, dtype=cp.float32), (nspl, lm)))
    y2 = cp.zeros((nspl, lm), dtype=cp.float32)              # natural: y2(1) = y2(nold) = 0
    nnew = 2 * lm - 1
    xnew = cp.ascontiguousarray(
        cp.broadcast_to(cp.linspace(0.0, lm - 1, nnew, dtype=cp.float32), (nspl, nnew)))
    p = cp.empty((nspl, lm), dtype=cp.float32)
    q = cp.empty((nspl, lm), dtype=cp.float32)
    ynew = spline_batch(xold, yold, y2, xnew, p=p, q=q)
    if not all(bool(cp.isfinite(a).all()) for a in (ynew, y2, p, q)):
        raise RuntimeError("drv-spline: NaN/Inf in spline output on the full-size test")
    return None
