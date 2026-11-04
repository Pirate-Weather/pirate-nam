"""bnd-bocoh-blend: write H-point boundary values into PD, T, Q, CW and PINT, blended over the
lnsh rows on each side, then the four corners and the outer-edge PINT.

Ported from nmm/module_FLTBNDS.F90:5355-5962 (inside BOCOH, nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/bnd_bocoh_blend.cu (strip x4, corners, pint_edges).

Reads : ctx.bc value slot of S_PD, S_T, S_Q, S_CW (and the N_, W_, E_ equivalents),
        PD, T, Q, CW, PINT, DSG2 from ctx.st; PDSG1 and pt from ctx.grid (PDSG1 is uploaded
        once and cached in ctx.tables); lnsh from ctx.cfg.lnsh.
Writes: PD, T, Q, CW, PINT in ctx.st (edge strips and corners, then the outer PINT rows).

H boundary variables are the parent's hardwired list (module_SOLVER_GRID_COMP.F90:7658-7664):
2-D PD and 3-D T, Q, CW. Q2 is not a parent H boundary variable and is not blended.

ctx.bc contract (CONTRACT.md section 6, "H-point boundary layout"): a dict keyed "<SIDE>_<VAR>"
(SIDE in S, N, W, E; VAR in PD, T, Q, CW), written by bnd-boundary-init. Each entry is float32 with
a leading time-slot axis of length 2; index 0 is the value slot this unit reads. Value-slot shapes
(slot excluded):
    S and N, 2-D PD : [lnsh, IM]          3-D T, Q, CW : [LM, lnsh, IM]
    W and E, 2-D PD : [JM, lnsh]          3-D T, Q, CW : [LM, JM, lnsh]
A missing key or a wrong shape raises: no silent zero boundary.

Loop limits: single process, its = ids = 1 and ite = ide = IM, jts = jds = 1 and jte = jde = JM,
so the Fortran its_h2/ite_h2/jts_h2/jte_h2 span the whole domain (module_DM_PARALLEL.F90).

Launch order (all on ctx.stream, no host sync): strip S, strip N, strip W, strip E, corners,
pint_edges. This is the Fortran order: a later side reads what an earlier side wrote.
No per-step allocation: the only upload is PDSG1, once.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels, slab

UNIT = "bnd_bocoh_blend"
SIDES = ("S", "N", "W", "E")
H_VARS_2D = ("PD",)
H_VARS_3D = ("T", "Q", "CW")


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _bc_value(ctx, side: str, var: str, shape: tuple):
    key = f"{side}_{var}"
    if ctx.bc is None or key not in ctx.bc:
        raise KeyError(
            f"bnd-bocoh-blend: ctx.bc[{key!r}] missing (bnd-boundary-init writes it)")
    arr = ctx.bc[key]
    if tuple(arr.shape[1:]) != tuple(shape):
        raise ValueError(f"bnd-bocoh-blend: ctx.bc[{key!r}] shape {arr.shape}, want (2,)+{shape}")
    return arr[0]  # value slot (Fortran index 1 of the time axis)


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsh = int(ctx.cfg.lnsh)
    if g.pdsg1 is None:
        raise RuntimeError(
            "bnd-bocoh-blend: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    pd, t, q, cw, pint = st["PD"], st["T"], st["Q"], st["CW"], st["PINT"]
    dsg2 = st["DSG2"]
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    pt = np.float32(g.pt)

    strip = kernels.get(UNIT, "bnd_bocoh_blend_strip")
    block = (128, 1, 1)
    s_bdy, n_bdy = slab.edges(ctx)      # row slab: S strips on the first rank, N on the last
    with ctx.stream:
        for side_id, side in enumerate(SIDES):
            if (side == "S" and not s_bdy) or (side == "N" and not n_bdy):
                continue
            if side in ("S", "N"):
                s2, s3 = (lnsh, im), (lm, lnsh, im)
                nb = im
            else:
                s2, s3 = (jm, lnsh), (lm, jm, lnsh)
                nb = jm
            bc_pd = _bc_value(ctx, side, "PD", s2)
            bc_t = _bc_value(ctx, side, "T", s3)
            bc_q = _bc_value(ctx, side, "Q", s3)
            bc_cw = _bc_value(ctx, side, "CW", s3)
            grid = ((nb + block[0] - 1) // block[0], lnsh, 1)
            strip(grid, block, (
                np.int32(side_id), np.int32(im), np.int32(jm), np.int32(lm), np.int32(lnsh),
                pt, dsg2, pdsg1, bc_pd, bc_t, bc_q, bc_cw,
                pd, t, q, cw, pint, *slab.kargs(ctx),
            ))

        corners = kernels.get(UNIT, "bnd_bocoh_blend_corners")
        corners((1, 1, 1), (4, 1, 1), (
            np.int32(im), np.int32(jm), np.int32(lm), pd, t, q, cw,
            np.int32((1 if s_bdy else 0) | (2 if n_bdy else 0)),
        ))

        edges = kernels.get(UNIT, "bnd_bocoh_blend_pint_edges")
        edges(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lm), pt, dsg2, pdsg1, pd, pint, *slab.kargs(ctx),
        ))
