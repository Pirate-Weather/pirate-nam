"""io-quasipost-3d: 3-D diagnostic fields of NAM's unified post (calculate_3D).

Port of nmm/module_QUASIPOST.F90:257-686 (nam.v4.2.11, 83df4e13c1). Kernels in
kernels/io_quasipost_3d.cu; see the header there for the section map and the decisions.

Reads (ctx.st, device arrays):
    PINT [LM+1,JM,IM], Z, T, Q, U, V [LM,JM,IM], FIS [JM,IM], U10, V10 [JM,IM] (V points; the
    10 m section averages them to H points, CONTRACT.md section 1)
    ctx.grid.dxh [JM] (uploaded once into ctx.tables["dxh"]), ctx.grid.dyh (scalar)
Writes (added with st.add on the first call, float32 unless noted):
    QP_PRESTV, QP_PRESZ      [NPRES=33,JM,IM]  post-level Tv and Z (NPRES levels, level first)
    QP_GROUND_MASK           [NPRES,JM,IM]     int32 0/1
    QP_GROUND_LEVEL          [JM,IM]           int32 first post level above ground
    QP_KS850, QP_KS700, QP_KS500  [JM,IM]      int32 model level bracketing 850/700/500 mb
    QP_DUMMY1, QP_DUMMY2     [JM,IM]           the Fortran local dummy1/dummy2 (500-mb scratch)
    P500U, P500V             [JM,IM]           500-mb winds (H points)
    P700WIND, P700RV, P700U, P700V            700-mb wind magnitude, vorticity, u, v
    P850WIND, P850RV, P850U, P850V            850-mb wind magnitude, vorticity, u, v
    M10WIND, M10RV           [JM,IM]           10-m wind magnitude and vorticity
    QP_NBAD                  [2]               int32 counters: [0] weight out of range,
                                               [1] NaN m10rv (the Fortran aborts on these;
                                               the caller reads them at output time, not per step).
                                               PER CALL: run() zeroes them at its start, so they
                                               hold one run's counts, not cumulative totals.
Launches (one stream, no host sync, no per-step allocation after the first call):
    1 column kernel, then for 500/700/850 mb: interior, edge_i, edge_j, then m10.
Winds below the lowest mass level are read from model level LM (not level 1, the model top);
see the decisions block in kernels/io_quasipost_3d.cu.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "io_quasipost_3d"
NPRES = 33
K_COLUMN = "io_quasipost_3d_column"
K_INTERIOR = "io_quasipost_3d_interior"
K_EDGE_I = "io_quasipost_3d_edge_i"
K_EDGE_J = "io_quasipost_3d_edge_j"
K_M10 = "io_quasipost_3d_m10"

# post level indices (1-based, module_QUASIPOST.F90:23) for 850, 700, 500 mb
K850, K700, K500 = 27, 21, 13


def _dxh(ctx):
    """DXH(j), uploaded once; the grid row is a constant of the run."""
    if "dxh" not in ctx.tables:
        import cupy as cp
        ctx.tables["dxh"] = cp.asarray(np.ascontiguousarray(ctx.grid.dxh, dtype=np.float32))
    return ctx.tables["dxh"]


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if im < 4 or jm < 4:
        # The boundary copies read IM-2 and JM-2; below 4 they index column/row 0.
        raise RuntimeError(f"io-quasipost-3d: grid {im}x{jm} too small for the boundary copies")

    presTv = st.add("QP_PRESTV", (NPRES, jm, im))
    presZ = st.add("QP_PRESZ", (NPRES, jm, im))
    gmask = st.add("QP_GROUND_MASK", (NPRES, jm, im), "int32")
    glev = st.add("QP_GROUND_LEVEL", (jm, im), "int32")
    ks850 = st.add("QP_KS850", (jm, im), "int32")
    ks700 = st.add("QP_KS700", (jm, im), "int32")
    ks500 = st.add("QP_KS500", (jm, im), "int32")
    dummy1 = st.add("QP_DUMMY1", (jm, im))
    dummy2 = st.add("QP_DUMMY2", (jm, im))
    p500u = st.add("P500U", (jm, im))
    p500v = st.add("P500V", (jm, im))
    p700wind = st.add("P700WIND", (jm, im))
    p700rv = st.add("P700RV", (jm, im))
    p700u = st.add("P700U", (jm, im))
    p700v = st.add("P700V", (jm, im))
    p850wind = st.add("P850WIND", (jm, im))
    p850rv = st.add("P850RV", (jm, im))
    p850u = st.add("P850U", (jm, im))
    p850v = st.add("P850V", (jm, im))
    m10wind = st.add("M10WIND", (jm, im))
    m10rv = st.add("M10RV", (jm, im))
    nbad = st.add("QP_NBAD", (2,), "int32")

    dxh = _dxh(ctx)
    dyh = np.float32(g.dyh)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        nbad.fill(0)
        # Section 1: T/Q/Z to the post levels, ground level, below-ground extrapolation.
        kernels.get(UNIT, K_COLUMN)(
            ((im * jm + 127) // 128, 1, 1), kernels.BLOCK_COL, (
                np.int32(im), np.int32(jm), np.int32(lm),
                st["PINT"], st["Z"], st["T"], st["Q"], st["FIS"],
                presTv, presZ, gmask, glev, ks850, ks700, ks500, nbad,
            ))

        # Section 2: winds on 500, 700, 850 mb (Fortran windloop k = 0, 1, 2).
        levels = (
            (K500, ks500, p500u, p500v, dummy1, dummy2),
            (K700, ks700, p700u, p700v, p700wind, p700rv),
            (K850, ks850, p850u, p850v, p850wind, p850rv),
        )
        grid2d = kernels.grid_2d(im, jm, 1)
        kint = kernels.get(UNIT, K_INTERIOR)
        kedge_i = kernels.get(UNIT, K_EDGE_I)
        kedge_j = kernels.get(UNIT, K_EDGE_J)
        for kd, ksx, pu, pv, pw, pr in levels:
            kint(grid2d, kernels.BLOCK_2D, (
                np.int32(im), np.int32(jm), np.int32(lm), np.int32(kd),
                st["PINT"], st["U"], st["V"], dxh, dyh, ksx,
                pu, pv, pw, pr,
            ))
            kedge_i(((jm - 3 + 127) // 128, 3, 1), (128, 1, 1), (
                np.int32(im), np.int32(jm), pu, pv, pw, pr,
            ))
            kedge_j(((im + 127) // 128, 3, 1), (128, 1, 1), (
                np.int32(im), np.int32(jm), pu, pv, pw, pr,
            ))

        # Section 3: 10-m wind magnitude and vorticity.
        kernels.get(UNIT, K_M10)(grid2d, kernels.BLOCK_2D, (
            np.int32(im), np.int32(jm),
            st["U10"], st["V10"], dxh, dyh, m10wind, m10rv, nbad,
        ))
    # Ordering token for io-quasipost-interp.run (QP_PRESZ and QP_GROUND_LEVEL are fresh from here).
    ctx.tables["qp3d_fresh"] = True
    return None
