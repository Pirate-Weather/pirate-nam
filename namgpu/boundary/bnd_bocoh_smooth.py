"""bnd-bocoh-smooth: optional spatial smoothing of the H boundary rows (S, N, W, E).

Ported from nmm/module_FLTBNDS.F90:5963-6387 (BOCOH smoothing block, nam.v4.2.11, 83df4e13c1).
Kernels: namgpu/kernels/bnd_bocoh_smooth.cu (bocoh_smooth_ww, bocoh_smooth_copy).

Inputs, the contract the producers actually make:
  ctx.bc["<SIDE>_<NAME>"]  strip buffers from bnd-boundary-init (NAME in PD, T, Q, CW; SIDE in S, N, W, E).
                           Leading time-slot axis of 2; index 0 is the value slot this unit reads and writes.
                           Value shapes: S, N: [lnsh, IM] (2-D) or [LM, lnsh, IM] (3-D);
                                         W, E: [JM, lnsh] or [LM, JM, lnsh].
  st[NAME]                 the model H fields (PD [JM, IM], T/Q/CW [LM, JM, IM]). Fortran boundary_init
                           fills the strips from the full-domain field (full_var) the same way.
Working full-domain plane (Fortran full_var): per call, one [nlev, JM, IM] scratch per NAME
  (st.add "BOCOH_FULL_<NAME>"), copied from st[NAME]. Its strip cells are then overwritten from the
  strip buffer value slot, so the stencil reads the boundary row/column values where the strip holds
  them and the model field one ring outside the strip. Smoothing then runs on this plane. Its strip
  cells are written back to the strip buffer value slot at the end. The wrapper does not write st[NAME].
  Declared mapping: this is how the port carries the smoothed strip into the strip buffers. It is not
  claimed to be bitwise to Fortran full_var persistence, which the Fortran blend reads directly.
  var_4d: NAM registers NVARS_BC_4D_H = 0 (module_SOLVER_GRID_COMP.F90:7658-7664), so no 4-D H
  variables exist and none are handled here.
  st.PINT[1:]              PINT levels 2..LM+1, i.e. Fortran pint(:,:,l+1), smoothed in place on the strip,
                           as Fortran does for pint.

NSMUD is 0 in the NAM configuration: BOCOH assigns nsmud=0 unconditionally (module_FLTBNDS.F90:5154;
the nsmud=lines-1 line above it is commented out) and nothing else assigns it, so the
`if(nsmud>=1)` block never runs. run() therefore launches nothing by default. The path is complete
for nsmud >= 1 and is exercised by tools/check-bnd-bocoh-smooth.py (full size, explicit nsmud=1).

Strip boxes (zero-based, inclusive): see strip_boxes(); lines = cfg.lnsh (5).
Launch: one kernel pair per (pass, side, array slab) on ctx.stream. No host sync, no per-cell Python;
the overlay and writeback are whole-strip device array copies. Scratch is allocated through st.add()
once, never per step.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab
from namgpu.boundary.bnd_boundary_init import H_2D, H_3D

UNIT = "bnd_bocoh_smooth"
K_WW = "bocoh_smooth_ww"
K_COPY = "bocoh_smooth_copy"

NSMUD = 0                     # Fortran nsmud (BOCOH line 5154): always 0 in the NAM configuration
SIDE_NAMES = ("S", "N", "W", "E")
H_NAMES = tuple(H_2D) + tuple(H_3D)


def strip_boxes(im: int, jm: int, lines: int):
    """Zero-based inclusive boxes (i0lo, i0hi, j0lo, j0hi) for the four sides, in Fortran order."""
    return (
        ("south", (1, im - 2, 1, lines - 1)),
        ("north", (1, im - 2, jm - lines, jm - 2)),
        ("west", (1, lines - 1, lines, jm - lines - 1)),
        ("east", (im - lines, im - 2, lines, jm - lines - 1)),
    )


def _strip_index(side: str, im: int, jm: int, lines: int):
    """(full-plane index, strip-buffer index), each a (rows, cols) pair over the last two axes.

    The full plane is zero-based [JM, IM]. The strip buffer is the bnd-boundary-init value slot:
    S/N [lines, IM] (rows = j row of the strip), W/E [JM, lines] (cols = strip columns).
    W/E cover rows lines..JM-lines-1 only, matching the Fortran W/E box, so the S/N corners are not
    overwritten by W/E."""
    all_ = slice(None)
    if side == "S":
        return (slice(0, lines), slice(0, im)), (all_, all_)
    if side == "N":
        return (slice(jm - lines, jm), slice(0, im)), (all_, all_)
    if side == "W":
        return (slice(lines, jm - lines), slice(0, lines)), (slice(lines, jm - lines), all_)
    return (slice(lines, jm - lines), slice(im - lines, im)), (slice(lines, jm - lines), all_)


def _bc_key(side: str, name: str) -> str:
    return f"{side}_{name}"


def _launch_slab(ctx, kww, kcopy, src, ww, nslab, box, im, jm):
    i0lo, i0hi, j0lo, j0hi = box
    ni, nj = i0hi - i0lo + 1, j0hi - j0lo + 1
    if ni <= 0 or nj <= 0:
        return
    bx, by, _ = kernels.BLOCK_2D
    grid = ((ni + bx - 1) // bx, (nj + by - 1) // by, nslab)
    i32 = np.int32
    args = (i32(im), i32(jm), i32(nslab), i32(i0lo), i32(i0hi), i32(j0lo), i32(j0hi))
    kww(grid, kernels.BLOCK_2D, args + (src, ww))
    kcopy(grid, kernels.BLOCK_2D, args + (src, ww))


def run(ctx, nsmud: int | None = None):
    """One call = the Fortran `do ks=1,nsmud` loop. nsmud defaults to NSMUD (0: nothing launched)."""
    nsmud = NSMUD if nsmud is None else int(nsmud)
    if nsmud < 1:
        return
    if slab.of(ctx) is not None:
        raise RuntimeError("bnd-bocoh-smooth: nsmud >= 1 smooths fixed global row bands and needs a halo "
                           "exchange between iterations; not built for row slabs (NAM runs nsmud = 0)")
    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lines = int(cfg.lnsh)
    bc = getattr(ctx, "bc", None)
    if bc is None:
        raise RuntimeError("bnd-bocoh-smooth: ctx.bc (H boundary strip buffers) is missing; "
                           "bnd-boundary-init must run first")
    missing = [_bc_key(s, n) for s in SIDE_NAMES for n in H_NAMES if _bc_key(s, n) not in bc]
    if missing:
        raise RuntimeError(f"bnd-bocoh-smooth: ctx.bc strip buffers missing {missing}; "
                           "bnd-boundary-init must run first")

    nlev = {n: (1 if n in H_2D else lm) for n in H_NAMES}
    ww = st.add("BOCOH_SMOOTH_WW", (lm, jm, im))
    pint_lev = st.PINT[1:]                      # pint(:,:,l+1), l = 1..LM (LM slabs, contiguous)
    kww = kernels.get(UNIT, K_WW)
    kcopy = kernels.get(UNIT, K_COPY)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    with scope:
        fulls = {}
        for name in H_NAMES:
            full = st.add(f"BOCOH_FULL_{name}", (nlev[name], jm, im))
            full[...] = st[name].reshape(nlev[name], jm, im)
            fulls[name] = full

        def plane(name):
            # 2-D: [JM, IM] view (the strip buffer has no level axis); 3-D: [LM, JM, IM].
            return fulls[name][0] if nlev[name] == 1 else fulls[name]

        # Overlay: strip cells of the working plane come from the strip buffer value slot.
        for side in SIDE_NAMES:
            fidx, bidx = _strip_index(side, im, jm, lines)
            for name in H_NAMES:
                plane(name)[..., fidx[0], fidx[1]] = bc[_bc_key(side, name)][0][..., bidx[0], bidx[1]]

        for _ks in range(nsmud):
            for _side, box in strip_boxes(im, jm, lines):
                # Fortran order inside a side: 2-D, 3-D, then PINT.
                for name in H_2D:
                    _launch_slab(ctx, kww, kcopy, fulls[name], ww, 1, box, im, jm)
                for name in H_3D:
                    _launch_slab(ctx, kww, kcopy, fulls[name], ww, lm, box, im, jm)
                _launch_slab(ctx, kww, kcopy, pint_lev, ww, lm, box, im, jm)

        # Writeback: smoothed strip cells go back to the strip buffer value slot.
        for side in SIDE_NAMES:
            fidx, bidx = _strip_index(side, im, jm, lines)
            for name in H_NAMES:
                bc[_bc_key(side, name)][0][..., bidx[0], bidx[1]] = plane(name)[..., fidx[0], fidx[1]]
