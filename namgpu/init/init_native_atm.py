"""init-native-atm: decoded NAM bgrd3d00 hybrid-level fields -> NMMB state (cold start).

Unit: init-native-atm (group init, mode new). Kernel: namgpu/kernels/init_native_atm.cu, one
launch of kernels.grid_2d(IM, JM, LM). No host sync, no per-step allocation.

Inputs (float32 device arrays decoded from the GRIB2 bgrd3d00 file and placed in ctx.st.fields;
layout [LM, JM, IM] for 3-D, [JM, IM] for PSFC; l = 1 is the model top):
  TMP (K), SPFH (kg/kg), UGRD, VGRD (m/s, as NPS writes them), CLMR, CICE, RWMR, SNMR (kg/kg),
  PSFC (Pa).
Mapping flags come from the NPS namelist (cfg.direct_temp, cfg.do_clouds, cfg.vcoord; NAM CONUS:
direct_temp=.true., do_clouds=.false., vcoord=1). See the kernel header for each term's source.

Outputs written into ctx.st (CONTRACT names): T, Q, U, V, CW, W, Q2, F_ICE, F_RAIN, F_RIMEF
(3-D [LM, JM, IM]) and PD ([JM, IM]).

Not done here: PINT and the vertical coordinate (init-vcoord-recover), the GRIB2 decode itself
(init-grib2-decode), and the boundary strips.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "init_native_atm"   # kernel module name = namgpu/kernels/<UNIT>.cu (UNITS.json id with '-' -> '_')
KERNEL = "init_native_atm"
REQUIRED = ("TMP", "SPFH", "UGRD", "VGRD", "CLMR", "CICE", "RWMR", "SNMR")
Q2_DEFAULT = 0.02   # module_SOLVER_GRID_COMP.F90:941 (NMM cold start, constant Q2)


def _field(st, name, shape):
    if name not in st.fields:
        raise KeyError(f"{UNIT}: input {name} is missing from the state (decode bgrd3d00 first)")
    a = st.fields[name]
    if tuple(a.shape) != tuple(shape):
        raise ValueError(f"{UNIT}: {name} has shape {tuple(a.shape)}, expected {tuple(shape)}")
    if np.dtype(a.dtype) != np.float32:
        raise TypeError(f"{UNIT}: {name} has dtype {a.dtype}, the kernel reads float32")
    return a


def run(ctx):
    import cupy as cp

    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    d3 = (lm, jm, im)
    d2 = (jm, im)

    inp = {n: _field(st, n, d3) for n in REQUIRED}
    psfc = _field(st, "PSFC", d2)

    # NPS CWM gate (module_vinterp_routines.F90, "CWM definition"): do_clouds AND maxval(CLWMR) > 0.
    # Kept on the device so there is no host sync.
    if cfg.do_clouds:
        cw_on = (cp.max(inp["CLMR"]) > 0).astype(cp.int32).reshape(1)
    else:
        cw_on = cp.zeros(1, dtype=cp.int32)

    out = {n: st.add(n, d3) for n in ("T", "Q", "U", "V", "CW", "W", "Q2",
                                       "F_ICE", "F_RAIN", "F_RIMEF")}
    pd = st.add("PD", d2)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        inp["TMP"], inp["SPFH"], inp["UGRD"], inp["VGRD"], inp["CLMR"], inp["CICE"],
        inp["RWMR"], inp["SNMR"], psfc, cw_on,
        out["T"], out["Q"], out["U"], out["V"], out["CW"], out["W"], out["Q2"],
        out["F_ICE"], out["F_RAIN"], out["F_RIMEF"], pd,
        np.float32(g.pt), np.float32(g.pdtop), np.float32(Q2_DEFAULT),
        np.int32(cfg.vcoord), np.int32(bool(cfg.direct_temp)),
        np.int32(im), np.int32(jm), np.int32(lm),
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
        warm_start(ctx)


def warm_start(ctx):
    """Use the analysis' own condensate, ice/rain/rime fractions and TKE when the source has them.

    The archived NAM analysis (bgrd3d00) is the cycled NDAS/GSI state the operational NAM forecast
    started from, and it carries TCOND (total condensate = NMMB CWM), FICE, FRAIN, RIME (the Ferrier
    F_ICE, F_RAIN, F_RIMEF) and TKE. A cold CW = 0, Q2 = 0.02 start would throw that away and spin
    clouds up from nothing, which the archived NAM forecast did not do. So, where sources.py found
    them (ctx.tables["native_warm"]): CW = TCOND, F_ICE = FICE, F_RAIN = FRAIN, F_RIMEF = max(RIME, 1),
    Q2 = max(2 TKE, 0.02) (NMMB Q2 is twice the TKE; 0.02 is the EPSQ2 floor,
    module_INIT_READ_BIN.F90:237-240). Sources without these records (GFS) keep the cold-start values.
    """
    from namgpu.init import sources  # noqa: PLC0415
    if ctx.tables.get("parent_warm"):
        # One-way nest from our own history (parent_start.species): the parent's Ferrier fractions on the nest
        # levels; CW is the cold rule's species sum, which is the parent's CWM. Applied after drv-physics-initialize
        # (apply_parent_warm), because phys-fer-init's cold-start fill (F_ICE = F_RAIN = 0, F_RIMEF = 1) runs there.
        return ["F_ICE", "F_RAIN"]
    return sources.decode_warm(ctx)


def apply_parent_warm(ctx):
    """After drv-physics-initialize: put the parent's F_ICE / F_RAIN back over phys-fer-init's cold-start zeros.

    Without this the parent's snow and rain start as cloud water: the first 3 km nest from the real 12 km parent
    (2026-11-04) had QSNOW = QRAIN = 0 and no echo at f00 where the parent had 48.8 dBZ, and 26 dBZ at f01.
    F_RIMEF stays at the fill's 1 (unrimed; the history has no rime field). One-way nests from our history only.
    """
    pw = ctx.tables.pop("parent_warm", None)
    if not pw:
        return []
    st = ctx.st
    st["F_ICE"][...] = pw["F_ICE"]
    st["F_RAIN"][...] = pw["F_RAIN"]
    return ["F_ICE", "F_RAIN"]
