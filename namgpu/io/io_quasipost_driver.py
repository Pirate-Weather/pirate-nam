"""io-quasipost-driver: QUASIPOST driver (in-model post fields for history).

Port of nmm/module_QUASIPOST.F90:68-255 (quasipost, quasipost_impl), nam.v4.2.11 83df4e13c1.
Kernel: namgpu/kernels/io_quasipost_driver.cu (the driver's own statements, the relax4e sweep
and the copies). The three routines it calls are separate units and run through their own
wrappers, in Fortran order:

  1. io-quasipost-3d      run(ctx)  calculate_3D: writes QP_PRESTV, QP_PRESZ, QP_GROUND_MASK,
                                    QP_GROUND_LEVEL (scratch, shapes below).
  2. (this unit)          the ipres loop with relax4e, writes QP_PRESTV and QP_GROUND_MASK back.
  3. io-quasipost-interp  run(ctx)  calculate_interP: reads QP_PRESTV, QP_PRESZ, QP_GROUND_LEVEL,
                                    state PINT and FIS; writes QP_PMSL. check_err(ctx) then raises on
                                    any NAM abort path (the one host read, at the end of this run).
  4. (this unit)          MEMBRANE_MSLP = QP_PMSL.
  5. io-quasipost-mslp    run(ctx)  smoothMSLP: smooths MEMBRANE_MSLP in place with a unit mask
                                    (Fortran smoothMSLP sets relaxmask=1 everywhere). It reads
                                    neither QP_RELAXMASK nor QP_RELAXWORK.
  6. (this unit)          P850Z = presZ(k850), P700Z = presZ(k700).

Scratch (st.add on first call, shared names so the deps can read them):
  QP_PRESTV       float32 [NPRES, JM, IM]   presTv (Tv on the 33 post levels, 20000 Pa first)
  QP_PRESZ        float32 [NPRES, JM, IM]   presZ (height on the post levels)
  QP_GROUND_MASK  int32   [NPRES, JM, IM]
  QP_GROUND_LEVEL int32   [JM, IM]
  QP_PMSL         float32 [JM, IM]
  QP_RELAXMASK    int32   [JM, IM]
  QP_RELAXWORK    float32 [JM, IM]
  QP_RELAX_NEXT   float32 [JM, IM]          second buffer for the relax4e sweeps
  QP_RUNNING      int32   [1]               device flag, 0 once the level loop has exited
  QP_ANY          int32   [1]               device flag, need_to_relax for the current level
State used: PINT, FIS (read by io-quasipost-interp); Z, T, Q, U10, V10 (read by io-quasipost-3d);
MEMBRANE_MSLP, P850Z, P700Z (written).

Launches: about 33 x (4 + 100) small launches on ctx.stream. No host sync in the level loop (the
exit decision is a device flag). The one host read is io_quasipost_interp.check_err, at the end of run.

Omitted from the Fortran, with the reason in the kernel header: MPI halo exchange, the abort
checks on the constant post_stdpres table, and the pmin reduction whose result is never read.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab

UNIT = "io_quasipost_driver"

NPRES = 33                 # npres: NCEP post standard pressure levels (Fortran post_stdpres)
K850 = 27 - 1              # Fortran k850 = 27 (85000 Pa), 0-based
K700 = 21 - 1              # Fortran k700 = 21 (70000 Pa), 0-based
NRELAX = 100               # relax4e(..., 100, ...)
RELAX_COEFF = 0.7          # relax4e(..., 0.7, ...)


def _scratch(st, jm: int, im: int):
    return dict(
        prestv=st.add("QP_PRESTV", (NPRES, jm, im)),
        presz=st.add("QP_PRESZ", (NPRES, jm, im)),
        ground_mask=st.add("QP_GROUND_MASK", (NPRES, jm, im), dtype="int32"),
        ground_level=st.add("QP_GROUND_LEVEL", (jm, im), dtype="int32"),
        pmsl=st.add("QP_PMSL", (jm, im)),
        relaxmask=st.add("QP_RELAXMASK", (jm, im), dtype="int32"),
        relaxwork=st.add("QP_RELAXWORK", (jm, im)),
        relax_next=st.add("QP_RELAX_NEXT", (jm, im)),
        running=st.add("QP_RUNNING", (1,), dtype="int32"),
        anyflag=st.add("QP_ANY", (1,), dtype="int32"),
    )


def run(ctx) -> None:
    # Deps are imported here so this module imports even before they land.
    from namgpu.io import io_quasipost_3d, io_quasipost_interp, io_quasipost_mslp

    st = ctx.st
    g = ctx.grid
    im, jm = int(g.im), int(g.jm)
    s = _scratch(st, jm, im)
    membrane = st.add("MEMBRANE_MSLP", (jm, im))
    p850z = st.add("P850Z", (jm, im))
    p700z = st.add("P700Z", (jm, im))

    kern = {k: kernels.get(UNIT, k) for k in (
        "qp_init", "qp_level_set", "qp_level_decide", "qp_copy_work",
        "qp_relax_sweep", "qp_finalize_level", "qp_copy_2d", "qp_copy_level_2d")}
    grid2 = kernels.grid_2d(im, jm, 1)
    block2 = kernels.BLOCK_2D
    grid1 = (1, 1, 1)
    block1 = (1, 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    with scope:
        # Step 1: calculate_3D (other unit).
        io_quasipost_3d.run(ctx)

        # relaxmask = 1, running = 1, anyflag = 0.
        kern["qp_init"](grid2, block2, (
            np.int32(im), np.int32(jm), s["relaxmask"], s["running"], s["anyflag"]))

        # ipres loop, Fortran 33 .. 1, with its EXIT.
        for ipres0 in reversed(range(NPRES)):
            s["anyflag"].fill(0)
            kern["qp_level_set"](grid2, block2, (
                np.int32(im), np.int32(jm), np.int32(ipres0),
                s["ground_mask"], s["relaxmask"], s["running"], s["anyflag"], *slab.kargs(ctx)))
            slab.allreduce(ctx, s["anyflag"], "max")   # row slab: need_to_relax is domain-wide
            kern["qp_level_decide"](grid1, block1, (s["running"], s["anyflag"]))
            kern["qp_copy_work"](grid2, block2, (
                np.int32(im), np.int32(jm), np.int32(ipres0),
                s["prestv"], s["relaxwork"], s["running"]))
            # relax4e: Jacobi sweeps, ping-pong; NRELAX is even, so the result is in relaxwork.
            src, dst = s["relaxwork"], s["relax_next"]
            for _ in range(NRELAX):
                kern["qp_relax_sweep"](grid2, block2, (
                    np.int32(im), np.int32(jm), np.float32(RELAX_COEFF),
                    s["relaxmask"], src, dst, s["running"], *slab.kargs(ctx)))
                slab.halo_exchange(ctx, dst, "qp_relax")   # row slab: the next sweep reads dst's neighbours
                src, dst = dst, src
            kern["qp_finalize_level"](grid2, block2, (
                np.int32(im), np.int32(jm), np.int32(ipres0),
                s["relaxmask"], s["ground_mask"], s["relaxwork"], s["prestv"], s["running"]))

        # Step 3: calculate_interP (other unit) -> QP_PMSL.
        io_quasipost_interp.run(ctx)

        # grid%membrane_MSLP = Pmsl.
        kern["qp_copy_2d"](grid2, block2, (
            np.int32(im), np.int32(jm), s["pmsl"], membrane))

        # Step 5: smoothMSLP (other unit) on MEMBRANE_MSLP (unit mask, its own scratch).
        io_quasipost_mslp.run(ctx)

        # p850z = presZ(k850), p700z = presZ(k700).
        kern["qp_copy_level_2d"](grid2, block2, (
            np.int32(im), np.int32(jm), np.int32(K850), s["presz"], p850z))
        kern["qp_copy_level_2d"](grid2, block2, (
            np.int32(im), np.int32(jm), np.int32(K700), s["presz"], p700z))
        # Output stage: the one host read in this unit. Raises on any calculate_interP abort path.
        io_quasipost_interp.check_err(ctx)
    ctx.quasipost_done = True             # io-history-fields writes MEMBRANE_MSLP as the history MSLP
