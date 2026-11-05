"""init-derive-state: derived prognostics after the native NAM init read (CONTRACT section 1-3).

Fortran: nmm/module_INIT_READ_BIN.F90 (PINT, UP/VP/TP, DWDT, O3, Q2, W, TRACERS_PREV, TOLD; the cold-start block is
lines 156-638, the restart block 639-2763), nmm/module_SOLVER_GRID_COMP.F90 (SOLVER_INITIALIZE allocation fills
REFL_10CM, TADJ, QSH; OMGALF at :942; THS at :10240), nmm/module_DYNAMICS_ROUTINES.F90 (RTOP at :213, Z recursion at
:3032-3069). NAM nam.v4.2.11, 83df4e13c1. Term-by-term notes are in kernels/init_derive_state.cu.
Kernels: namgpu/kernels/init_derive_state.cu (all on ctx.stream, no host sync).

Reads : SG1, SG2 (LM+1), DSG1, DSG2 (LM), PD, FIS (read-only input from the init read, module_INIT_READ_BIN.F90:
        1549-1553; the wrapper checks it is present), TSKIN, T, Q, CW, Q2, U, V; slot indices from ctx.tables["indx"]
        (drv-tracer-index, Fortran 1-based), or built from cfg.
Writes, both paths: PINT, Z (LM+1 interfaces; skipped on restart, the file supplies them); RTOP (LM); TRACERS (slots
        INDX_Q, INDX_CW, INDX_Q2); THS (2-D, cold only). RTOP is added with st.add on first call if absent.
Writes, cold start only (restart reads each from file): DWDT, OMGALF, UP, VP, TP, TOLD, W, Q2, O3 (LM); TRACERS_PREV;
        REFL_10CM (= DBZmin), TADJ (= 0), QSH (2-D, = 0).
restart: ctx.restart (bool, default False). A restart leaves every file-read field to the restart reader.
Not derived here: F_ICE/F_RAIN/F_RIMEF cold fills (owner phys-fer-init); the T/U/V pre-read sentinels are overwritten by
the init read.
Single-card whole-domain tile: IDE = IM, JDE = JM (CONTRACT section 1), used for the O3 box centre.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels
from namgpu.steps.drv_tracer_index import host_index

UNIT = "init_derive_state"
GI = np.float32(1.0 / 9.8060226)      # NMMB_G (module_CONSTANTS), GI = 1/G


def _slot(st, name, shape):
    """Existing State array, or a unit-owned one added on first call."""
    f = st.fields.get(name)
    return f if f is not None else st.add(name, shape)


EPSQ2_FLOOR = np.float32(0.02)   # module_INIT_READ_BIN.F90:237-240 (NEMSIO path :399-402, same rule)


def init_epsq2(ctx):
    """EPSQ2 per level as the Fortran init read sets it: EPSQ2(K-1)=0.02 for K=2..LM and
    EPSQ2(LM)=EPSQ2(LM-1). Writes st.EPSQ2 and ctx.tables["EPSQ2"], the device vector that the
    dynamics wrappers (dyn-adv2-q2-mid, dyn-mono-q2-back, dyn-hdiff-coef) read."""
    import cupy as cp
    st = ctx.st
    lm = int(ctx.grid.lm)
    if lm < 2:
        raise ValueError(f"init-derive-state: EPSQ2 rule needs LM >= 2, got {lm}")
    host = np.empty(lm, dtype=np.float32)
    host[: lm - 1] = EPSQ2_FLOOR
    host[lm - 1] = host[lm - 2]
    dev = st["EPSQ2"]
    if tuple(dev.shape) != (lm,):
        raise ValueError(f"init-derive-state: State EPSQ2 shape {tuple(dev.shape)} != ({lm},)")
    dev[...] = cp.asarray(host)
    ctx.tables["EPSQ2"] = dev
    # EPSL(K-1)=SQRT(EPSQ2(K-1)*0.5) for K=2..LM (module_INIT_READ_BIN.F90:238, NEMSIO :400). The
    # MYJPBL floor state (phys-myj-prodq2, mixlen, myjpbl) reads it; nothing else wrote it, so it
    # stayed zero. EPSL(LM) is not set by the Fortran; it mirrors level LM-1 as EPSQ2 does.
    # Restricted-names States (the dyn check tools) do not carry EPSL; the production State does.
    eps = st.fields.get("EPSL")
    if eps is not None:
        if tuple(eps.shape) != (lm,):
            raise ValueError(f"init-derive-state: State EPSL shape {tuple(eps.shape)} != ({lm},)")
        epsl = np.empty(lm, dtype=np.float32)
        epsl[: lm - 1] = np.sqrt(host[: lm - 1] * np.float32(0.5))
        epsl[lm - 1] = epsl[lm - 2]
        eps[...] = cp.asarray(epsl)


def _tracer_slots(ctx):
    tab = ctx.tables.get("indx") if hasattr(ctx, "tables") else None
    if tab is None:
        tab = host_index(ctx.cfg.microphysics, ctx.cfg.raw.get("num_tracers_chem", 0))
    # Fortran INDX values are 1-based; the TRACERS array is zero-based in the slot dimension.
    return (int(tab["INDX_Q"]) - 1, int(tab["INDX_CW"]) - 1, int(tab["INDX_Q2"]) - 1)


def _keep_bits(ctx) -> int:
    """Cold-start fields the start read from its analysis and that this unit must not overwrite (sources.decode_warm
    records them in ctx.tables["native_warm_kept"]): bit 1 Q2 (the analysis TKE), bit 2 W (its DZDT). Breakage it
    fixes: a native NAM start decoded TKE into Q2 and then this unit's cold branch wrote 0.02 over it (TKE 0.01
    everywhere at 0 h against the analysis' mean 0.109), and W started at 0 (NAM start study, 2026-11-05)."""
    kept = ctx.tables.get("native_warm_kept") or ()
    return (1 if "Q2" in kept else 0) | (2 if "W" in kept else 0)


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    pt, pdtop = np.float32(g.pt), np.float32(g.pdtop)
    restart = bool(ctx.restart)
    restart_i = np.int32(restart)
    n_q, n_cw, n_q2 = _tracer_slots(ctx)
    init_epsq2(ctx)

    fis = st.fields.get("FIS")
    if fis is None or tuple(fis.shape) != (jm, im):
        raise ValueError("init-derive-state: FIS must be read by the init read before this unit "
                         "(module_INIT_READ_BIN.F90:1549-1553) with shape (JM, IM)")

    lev = (lm, jm, im)
    pint = _slot(st, "PINT", (lm + 1, jm, im))
    z = _slot(st, "Z", (lm + 1, jm, im))
    rtop = _slot(st, "RTOP", lev)
    told = _slot(st, "TOLD", lev)
    omgalf = _slot(st, "OMGALF", lev)
    dwdt = _slot(st, "DWDT", lev)
    up = _slot(st, "UP", lev)
    vp = _slot(st, "VP", lev)
    tp = _slot(st, "TP", lev)
    w = _slot(st, "W", lev)
    o3 = _slot(st, "O3", lev)
    refl = _slot(st, "REFL_10CM", lev)
    tadj = _slot(st, "TADJ", lev)
    qsh = _slot(st, "QSH", (jm, im))
    tracers = st["TRACERS"]
    tracers_prev = _slot(st, "TRACERS_PREV", tracers.shape)
    ths = _slot(st, "THS", (jm, im))

    k_pint = kernels.get(UNIT, "init_derive_pint")
    k_lev = kernels.get(UNIT, "init_derive_levels")
    k_trc = kernels.get(UNIT, "init_derive_tracers")
    k_z = kernels.get(UNIT, "init_derive_z")
    k_ths = kernels.get(UNIT, "init_derive_ths")
    k_fill = kernels.get(UNIT, "init_derive_cold_fill")
    grid_lev = kernels.grid_2d(im, jm, lm)
    grid_2d1 = kernels.grid_2d(im, jm, 1)
    grid_cols = kernels.grid_cols(im, jm)
    blk2 = kernels.BLOCK_2D
    blkc = kernels.BLOCK_COL

    n3 = lm * jm * im
    n2 = jm * im
    with ctx.stream:
        k_pint(grid_cols, blkc, (
            st["DSG1"], st["DSG2"], st["PD"], pint, pt, pdtop, restart_i,
            np.int32(im), np.int32(jm), np.int32(lm)))
        k_lev(grid_lev, blk2, (
            st["T"], st["Q"], st["CW"], st["U"], st["V"], pint,
            rtop, dwdt, omgalf, up, vp, tp, told, w, o3, st["Q2"],
            restart_i, np.int32(_keep_bits(ctx)), np.int32(im), np.int32(jm),      # IDE, JDE: single-card whole domain (IDE = IM)
            np.int32(im), np.int32(jm), np.int32(lm)))
        k_trc(grid_lev, blk2, (
            st["Q"], st["CW"], st["Q2"], tracers, tracers_prev,
            np.int32(n_q), np.int32(n_cw), np.int32(n_q2), restart_i,
            np.int32(im), np.int32(jm), np.int32(lm)))
        k_z(grid_cols, blkc, (
            fis, st["PD"], st["T"], st["Q"], st["CW"], pint, st["DSG1"], st["DSG2"],
            z, pdtop, GI, restart_i, np.int32(im), np.int32(jm), np.int32(lm)))
        k_ths(grid_2d1, blk2, (
            st["TSKIN"], st["PD"], st["SG1"], st["SG2"], ths, pdtop, pt, restart_i,
            np.int32(im), np.int32(jm), np.int32(lm)))
        if not restart:
            # SOLVER_INITIALIZE values that a cold start keeps (the restart reader supplies them otherwise).
            k_fill(((max(n3, n2) + 255) // 256,), (256,), (
                refl, tadj, qsh, np.int32(n3), np.int32(n2)))
    bind_tracer_views(st, n_q, n_cw, n_q2)


def bind_tracer_views(st, n_q, n_cw, n_q2):
    """Q, CW and Q2 ARE the TRACERS slots INDX_Q, INDX_CW, INDX_Q2 (Fortran pointers,
    module_SOLVER_INTERNAL_STATE.F90:1382-1404: int_state%Q => TRACERS_ARR(INDX_Q slice), same for
    CW and Q2). The State allocates Q, CW, Q2 as separate arrays, so after init_derive_tracers copied
    them into TRACERS the two copies diverged: ADV2/UPDATES advected only TRACERS, physics and history
    used Q/CW/Q2, so water vapour, condensate and TKE were never advected (fix physics-convection
    1150Z: no Gulf moisture return, no warm-sector convection). Bind the names to the slots, as Fortran
    does. Called after init_derive_tracers copied Q/CW/Q2 into the slots (both
    paths), so the values carry over; idempotent."""
    tr = st["TRACERS"]
    for name, slot in (("Q", n_q), ("CW", n_cw), ("Q2", n_q2)):
        view = tr[slot]
        cur = st.fields.get(name)
        if cur is not None and cur is not view:
            if tuple(cur.shape) != tuple(view.shape):
                raise ValueError(f"init-derive-state: {name} shape {tuple(cur.shape)} != TRACERS slot "
                                 f"{tuple(view.shape)}")
        st.fields[name] = view
