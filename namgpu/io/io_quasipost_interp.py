"""io-quasipost-interp: calculate_interP (nmm/module_QUASIPOST.F90:688-815), mode "port".

Interpolation of the membrane-MSLP inputs: below-ground extrapolation on the NCEP post standard
pressure levels and the sea-level pressure at Z = 0. One column per thread (grid_cols), one launch.
The math and the Fortran review are in kernels/io_quasipost_interp.cu.

Reads:   ctx.st.PINT            [LM+1, JM, IM]  interface pressure (Pa); index LM is the surface
         ctx.st.FIS             [JM, IM]        surface geopotential (m2 s-2)
         QP_PRESTV              [NPRES, JM, IM] standard-level Tv, after the driver relax4e pass
         QP_PRESZ               [NPRES, JM, IM] standard-level Z from io-quasipost-3d (in/out)
         QP_GROUND_LEVEL        [JM, IM] int32  first-guess ground index, 1-based, from io-quasipost-3d
Writes:  QP_PMSL                [JM, IM]        sea-level pressure (Pa); NaN in a column that hit a NAM abort
         QPI_ERR                [1] int32       0 ok; 1 = kp<1, 2 = dZ>=0 (the two NAM MPI_Abort paths)
Ordering: io-quasipost-3d must run before every call. It sets ctx.tables["qp3d_fresh"]; run() consumes
that token and refuses to run without it, because the fall-through reads QP_PRESZ at npres and
QP_GROUND_LEVEL, which only 3d writes.
Errors: check_err(ctx) is the host read of QPI_ERR. The driver calls it at the end of its run (output
stage) so an abort path raises instead of writing NaN into the history.
Domain: the kernel covers 1..IM x 1..JM. run() requires the grid to be the full domain.
npres = 33 (fixed by the NCEP post standard levels, not a configuration value).
Launch: one kernel on ctx.stream, no host sync inside run(), no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "io_quasipost_interp"
KERNEL = "io_quasipost_interp"
NPRES = 33
TOKEN = "qp3d_fresh"      # set by io_quasipost_3d.run, consumed by run()
ERR_TEXT = {
    1: "kp<1: the lowest model surface pressure is lower than the second lowest standard level "
       "(NAM MPI_Abort in calculate_interP)",
    2: "dZ>=0 in the hydrostatic step (NAM MPI_Abort in calculate_interP)",
}


def _require_full_domain(g) -> None:
    """The grid is the domain. Any tile bounds the grid carries must cover 1..IM and 1..JM, because the
    kernel writes every column of the domain and a sub-tile would write columns it does not own."""
    for lo, hi, n in (("ips", "ipe", int(g.im)), ("jps", "jpe", int(g.jm))):
        if hasattr(g, lo) and (int(getattr(g, lo)) != 1 or int(getattr(g, hi)) != n):
            raise RuntimeError(
                f"io-quasipost-interp: tile {lo}..{hi} is not the full domain 1..{n}")


def run(ctx):
    st = ctx.st
    g = ctx.grid
    _require_full_domain(g)
    if not ctx.tables.pop(TOKEN, False):
        raise RuntimeError(
            "io-quasipost-interp: io-quasipost-3d has not run since the last calculate_interP call; "
            "QP_PRESZ and QP_GROUND_LEVEL would be stale. The driver runs 3d first.")
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    pres_tv = st.add("QP_PRESTV", (NPRES, jm, im))
    pres_z = st.add("QP_PRESZ", (NPRES, jm, im))
    pmsl = st.add("QP_PMSL", (jm, im))
    ground = st.add("QP_GROUND_LEVEL", (jm, im), dtype="int32")
    err = st.add("QPI_ERR", (1,), dtype="int32")
    err[...] = 0

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        st.PINT, st.FIS, ground, pres_tv, pres_z, pmsl, err,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)


def check_err(ctx) -> None:
    """Host read of QPI_ERR (one int) on the output stage. Raises if a column hit a NAM abort path.
    The affected columns hold NaN in QP_PMSL; nothing downstream may use them."""
    err = ctx.st.fields.get("QPI_ERR")
    if err is None:
        return
    if ctx.stream is not None:
        ctx.stream.synchronize()
    code = int(err.get()[0]) if hasattr(err, "get") else int(err[0])
    if code:
        raise RuntimeError(
            f"io-quasipost-interp: ERR={code}: {ERR_TEXT.get(code, "unknown code")}")
