"""dyn-pgforce-pgf: pressure-gradient force components behind H points (pcx, pcy, pcne, pcnw)
with the nonhydrostatic correction, plus the pgx/pgy/pgne/pgnw scratch sums.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:249-289 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_pgforce_pgf.cu (dyn_pgforce_pgf, one thread per (i, j), level in grid.z).

Reads : PINT [LM+1,JM,IM] from ctx.st; APEL_3D, FIM_3D, DFI_3D (filled by dyn-pgforce-geopot).
Writes: PCX, PCY, PCNE, PCNW [LM,JM,IM] in ctx.st; PGX_3D, PGY_3D, PGNE_3D, PGNW_3D are
        [LM,JM,IM] scratch (the Fortran pgx/pgy/pgne/pgnw are 2-D and reused each level; the
        divcor and uvupdate units read the level slice).
        All of these are not NAM state-table names and are added once with st.add.
One launch on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_pgforce_pgf"
KERNEL = "dyn_pgforce_pgf"


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape3 = (lm, jm, im)

    pcx = st.add("PCX", shape3)
    pcy = st.add("PCY", shape3)
    pcne = st.add("PCNE", shape3)
    pcnw = st.add("PCNW", shape3)
    pgx = st.add("PGX_3D", shape3)
    pgy = st.add("PGY_3D", shape3)
    pgne = st.add("PGNE_3D", shape3)
    pgnw = st.add("PGNW_3D", shape3)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        st.PINT, st.add("FIM_3D", shape3), st.add("APEL_3D", shape3), st.add("DFI_3D", shape3),
        pcx, pcy, pcne, pcnw, pgx, pgy, pgne, pgnw,
        np.int32(im), np.int32(jm), np.int32(lm),
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
