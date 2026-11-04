"""dyn-adv1-horiz-uv: momentum horizontal advection in ADV1 (secadv, 2nd order), the 4th-order
diagonal branch (secadv false) and the Adams-Bashforth advection of u1d/v1d.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:1867-2021 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_adv1_horiz_uv.cu (adv1_uv_u4_diag, adv1_uv_u4_edge, adv1_uv_u4_copy,
adv1_uv_horiz). Level is blockIdx.z; every launch covers all levels.

Reads:   U1D, V1D [LM,JM,IM] (extrapolated u1d/v1d, set before the coriolis section of ADV1);
         PDOP_UV [JM,IM] (8-point pdop, written by dyn-adv1-vert-uv earlier in the step; must run first);
         PFXX1, PFYX1, PFNEX1, PFNWX1, PFXY1, PFYY1, PFNEY1, PFNWY1 [LM,JM,IM] (flux scratch from
         dyn-adv1-coriolis); DSG2, PDSG1 [LM]; DXV, RDXV, FAD [JM]; DYV, RDYV scalars.
Writes:  TCU, TCV [LM,JM,IM] (added to, as Fortran). Only when secadv is false: U1D and V1D are
         overwritten by the 4th-order copy, using scratch U2D, V2D [LM,JM,IM] (allocated then, never on
         the default path).
Limits:  the regional single-tile case (ids=1, ide=IM, its=1, ite=IM). Global (global = true) is refused:
         swapwn/polewn and the global pf zeroing are not ported (CONTRACT section 2, global false).
Launch:  on ctx.stream; no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.dynamics.dyn_adv1_vert_uv import _device_vector

UNIT = "dyn_adv1_horiz_uv"


def _grid_const(st, name: str, host):
    """Upload a host grid constant once and keep it on the device in the state."""
    if name not in st.fields:
        import cupy as cp
        st.fields[name] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return st.fields[name]


def _fad(g, dt: float) -> np.ndarray:
    """Grid factor fad(j) = -dt/(3*dxv(j)*dyv*2*2) (module_CONTROL.F90:643), single precision in
    the Fortran order.  Rows 1, 2, JM-1, JM (1-based) are zero (module_CONTROL.F90:581 init, the
    between-poles loop covers jds+2..jde-2 only)."""
    dxv = np.asarray(g.dxv, dtype=np.float32)
    dyv = np.float32(g.dyv)
    den = np.float32(3.0) * dxv * dyv * np.float32(2.0) * np.float32(2.0)
    fad = (np.float32(-dt) / den).astype(np.float32)
    fad[0:2] = 0.0
    fad[-2:] = 0.0
    return fad


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if ctx.cfg.global_:
        raise RuntimeError(
            "dyn-adv1-horiz-uv: global = true needs swapwn/polewn and the global pf zeroing, "
            "which are not ported (CONTRACT section 2 assumes global = false)")

    # Regional tile limits, single process (module_CONTROL.F90:250-275, module_DM_PARALLEL looplimits).
    lnsad = int(ctx.cfg.lnsh) + 2                  # module_CONTROL.F90:260
    ids, ide, jds, jde = 1, im, 1, jm
    its, ite, jts, jte = 1, im, 1, jm
    adv_standard = (jte >= jds + 1 + lnsad and jts <= jde - 1 - lnsad
                    and ite >= ids + 1 + lnsad and its <= ide - 1 - lnsad)
    ibeg = max(its, ids + 1 + lnsad)               # module_DYNAMICS_ROUTINES.F90 UV block
    iend = min(ite, ide - 2 - lnsad)
    jbeg = max(jts, jds + 1 + lnsad)
    jend = min(jte, jde - 2 - lnsad)

    shape3 = (lm, jm, im)
    secadv = bool(ctx.cfg.secadv)

    u1d = st.add("U1D", shape3)
    v1d = st.add("V1D", shape3)
    pf = {n: st.add(n, shape3) for n in
          ("PFXX1", "PFYX1", "PFNEX1", "PFNWX1", "PFXY1", "PFYY1", "PFNEY1", "PFNWY1")}

    dxv = _grid_const(st, "DXV", g.dxv)
    rdxv = _grid_const(st, "RDXV", np.float32(1.0) / np.asarray(g.dxv, dtype=np.float32))
    # FAD is int_state%FAD: the init-grid-consts row, scaled in place by the TDFI (drv-filter-consts).
    # Computing it here from ctx.dt cached the first call's dt (the filter's -8 s) for the whole run.
    from namgpu.steps import drv_filter_consts  # noqa: PLC0415
    drv_filter_consts.bind(ctx)
    fad = st.fields["FAD"]
    dsg2 = st.DSG2
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)   # same cache and invalidation as vert-uv
    rdyv = np.float32(1.0) / np.float32(g.dyv)

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        if not secadv:
            # 4th-order branch (module_DYNAMICS_ROUTINES.F90:1869-1931), dead for the NAM config.
            u2d = st.add("U2D", shape3)
            v2d = st.add("V2D", shape3)
            kernels.get(UNIT, "adv1_uv_u4_diag")(
                kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
                (np.int32(im), np.int32(jm), np.int32(lm), u1d, v1d, u2d, v2d))
            edge_grid = ((max(im, jm) + 255) // 256, 1, lm)
            for edge in range(4):                  # s, n, w, e: Fortran order
                kernels.get(UNIT, "adv1_uv_u4_edge")(
                    edge_grid, (256, 1, 1),
                    (np.int32(im), np.int32(jm), np.int32(lm), np.int32(edge),
                     u1d, v1d, u2d, v2d))
            kernels.get(UNIT, "adv1_uv_u4_copy")(
                kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
                (np.int32(im), np.int32(jm), np.int32(lm), u2d, v2d, u1d, v1d))

        if adv_standard:
            pdop_uv = st.fields.get("PDOP_UV")
            if pdop_uv is None:
                raise RuntimeError("dyn-adv1-horiz-uv: PDOP_UV missing; dyn-adv1-vert-uv must run first in the step")
            kernels.get(UNIT, "adv1_uv_horiz")(
                kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
                (np.int32(im), np.int32(jm), np.int32(lm),
                 np.int32(ibeg - 1), np.int32(iend - 1),
                 np.int32(jbeg - 1), np.int32(jend - 1),
                 np.float32(g.dyv), rdyv,
                 dxv, rdxv, fad, dsg2, pdsg1,
                 pdop_uv, u1d, v1d,
                 pf["PFXX1"], pf["PFYX1"], pf["PFNEX1"], pf["PFNWX1"],
                 pf["PFXY1"], pf["PFYY1"], pf["PFNEY1"], pf["PFNWY1"],
                 st.TCU, st.TCV))
