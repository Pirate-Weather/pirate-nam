"""io-quasipost-mslp: smoothMSLP, the Shuell/NAM membrane MSLP smoothing.

Port of nmm/module_QUASIPOST.F90:817-852 (nam.v4.2.11, 83df4e13c1), which calls
nmm/module_RELAX4E.F90 relax4e once (iterations = 1, relax coefficient 0.5, mask 1 everywhere).
The arithmetic and the edge/corner rules are in kernels/io_quasipost_mslp.cu.

Reads and writes:
    ctx.st.MEMBRANE_MSLP   [JM,IM] float32, Pa (H points). Smoothed in place, as Fortran does
                           (membrane_mslp(i,j) = relaxwork(i,j) at the end of smoothMSLP).
    scratch RELAXWORK_NEXT [JM,IM] float32, added with st.add() on the first call. It is the
                           relax4e local "next" array (the full domain here, single process).
Launches: two kernels on ctx.stream (relax pass, then copy back). No host sync, no per-step
allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "io_quasipost_mslp"
KERNEL_RELAX = "io_quasipost_mslp_relax"
KERNEL_COPY = "io_quasipost_mslp_copy"


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm = int(g.im), int(g.jm)

    # MEMBRANE_MSLP sits after the TRACKER marker in parm/nam_solver_state.txt, which
    # state.parse_state_table stops at, so it is not an attribute of st yet (state.py defect,
    # reported to the coordinator). Allocate it here with the table shape (rank 2, [JM, IM]).
    work = st.add("MEMBRANE_MSLP", (jm, im))
    next_ = st.add("RELAXWORK_NEXT", (jm, im))

    relax = kernels.get(UNIT, KERNEL_RELAX)
    copy = kernels.get(UNIT, KERNEL_COPY)
    grid = kernels.grid_2d(im, jm, 1)
    block = kernels.BLOCK_2D
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        relax(grid, block, (np.int32(im), np.int32(jm), work, next_))
        copy(grid, block, (np.int32(im), np.int32(jm), next_, work))
