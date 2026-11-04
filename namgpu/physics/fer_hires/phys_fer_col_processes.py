"""phys-fer-col-processes: MP_FER_HIRES column processes (module_MP_FER_HIRES.F90:1321-1478).

Ported from phys/module_MP_FER_HIRES.F90:1321-1478 (NAM nam.v4.2.11, 83df4e13c1). Kernel:
namgpu/kernels/phys_fer_col_processes.cu (phys_fer_col_processes), one launch, grid_2d(IM, JM, LM),
one thread per (i, j, l) cell. The statement map and the notice are in the kernel header.

Every kernel input is bound from its producer by bind_inputs(), on the device, before the launch:
  T, PMID, FER_Q, FER_QW       column state (T_col, P_col, Q_col, QW_col), in ctx.st
  FER_SC_RHO                   phys-fer-col-satcheck, RHO (Fortran line 967), 0 on clear levels
  fer_fpvs, fer_fpvs0          phys-fer-fpvs, FPVS and FPVS0 of T (lines 916-933 read them)
  fer_col_ice_out slots        phys-fer-col-ice-props (lines 1029-1320 and 1037-1052): TK2, DIFFUS,
                               DYNVIS, THERM_COND, GAMMAS, VEL_INC, QLICE, QTICE, NLICE, NSMICE,
                               PILOSS, NINUCLEI, INDEXS, RCW
A missing producer raises before anything is launched; no input defaults to zero or random data.
DWVI is in/out and keeps whatever value cols already holds. Fortran DWVI keeps its value when neither
ice branch assigns it, so the caller must keep cols across levels and calls.

Tables and scalars come from ctx.tables["fer"], built once by phys-fer-init (phys_fer_init._build):
  tbpvs, tbpvs0 [7501]; accri, venti1, venti2 [951] (index M - 50); growth [35] (MY_GROWTH_NMM);
  scalars: CIACW, ARAUT, BRAUT, QAUT0, ARCW, C1XPVS, C2XPVS, C1XPVS0, C2XPVS0; dtph.

cols: dict of float32 device arrays [LM, JM, IM] (CONTRACT section 1 layout); INDEXS is int32.
The arrays are allocated once by alloc(). The flags and warn2 scratch live in ctx.tables:
  FLAGS [8] int32: 0 DEPOSIT non-converged count, 1 INDEXS out-of-range count, 2 CONDENSE non-converged.
  WARN2 [8] int32: first DEPOSIT non-convergence record (phys_fer_deposit layout).
RHgrd is the configfile rhgrd (0.98, CONTRACT section 5).

Launch: one kernel on ctx.stream (or the null stream), no host sync, scalars passed by value.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_col_processes"
KERNEL = "phys_fer_col_processes"
RHGRD = 0.98
T0C = np.float32(273.15)           # FER_T0C (fer_hires.cuh)
EP2 = np.float32(287.04 / 461.6)   # FER_EP2 (fer_hires.cuh)
NFLAGS = 8
FLAGS_KEY = "fer_col_processes_flags"
WARN2_KEY = "fer_col_processes_warn2"

# phys-fer-col-ice-props output slots (phys_fer_col_ice_props.cu, FER_OUT_*).
SLOT = {"INDEXS": 2, "VEL_INC": 5, "QLICE": 8, "QTICE": 9, "NLICE": 11, "NSMICE": 13,
        "PILOSS": 15, "RCW": 34, "NINUCLEI": 35, "TK2": 26, "DYNVIS": 28, "THERM_COND": 29,
        "DIFFUS": 30, "GAMMAS": 31}
ICE_FROM_PRODUCER = ("VEL_INC", "QLICE", "QTICE", "NLICE", "NSMICE", "PILOSS", "NINUCLEI",
                     "TK2", "DYNVIS", "THERM_COND", "DIFFUS", "GAMMAS")

IN_F32 = ("TK", "PP", "WV", "QW", "RHO", "RRHO", "TK2", "QSI", "QSWGRD", "QSIGRD", "DIFFUS",
          "DYNVIS", "THERM_COND", "GAMMAS", "VEL_INC", "NLICE", "NSMICE", "NINUCLEI", "QLICE",
          "QTICE", "PILOSS")
IO_F32 = ("RCW", "DWVI")
OUT_F32 = ("PRAUT", "PIACW", "PIACWI", "PCOND", "PIDEP", "PINIT", "TCC", "QSW0", "DWV0")
ALL_F32 = IN_F32 + IO_F32 + OUT_F32


def alloc(ctx, shape):
    """Scratch arrays for the unit, allocated once and kept in ctx.tables["fer_col_processes"]."""
    import cupy as cp

    key = "fer_col_processes"
    cols = ctx.tables.get(key)
    if cols is None or cols["TK"].shape != tuple(shape):
        cols = {name: cp.zeros(shape, dtype=cp.float32) for name in ALL_F32}
        cols["INDEXS"] = cp.full(shape, 50, dtype=cp.int32)
        ctx.tables[key] = cols
    if FLAGS_KEY not in ctx.tables:
        ctx.tables[FLAGS_KEY] = cp.zeros(NFLAGS, dtype=cp.int32)
        ctx.tables[WARN2_KEY] = cp.zeros(NFLAGS, dtype=cp.int32)
    return cols


def _field(st, name):
    try:
        return st[name]
    except KeyError:
        raise RuntimeError(f"phys-fer-col-processes: ctx.st[{name!r}] is missing; run its producer first") from None


def bind_inputs(ctx, cols, rhgrd: float = RHGRD):
    """Fill every kernel input from its producer's output (device copies and elementwise math only)."""
    import cupy as cp

    from namgpu.physics.fer_hires import phys_fer_col_ice_props as ice_props
    from namgpu.physics.fer_hires import phys_fer_fpvs

    st = ctx.st
    T = _field(st, "T")
    PP = _field(st, "PMID")
    Q = _field(st, "FER_Q")
    QW = _field(st, "FER_QW")
    RHO = _field(st, "FER_SC_RHO")
    ice = ctx.tables.get(ice_props.OUT_KEY)
    if ice is None:
        raise RuntimeError("phys-fer-col-processes: ctx.tables['fer_col_ice_out'] is missing; "
                           "run phys-fer-col-ice-props first")
    shape = tuple(T.shape)
    if tuple(ice.shape) != (ice_props.NSLOT,) + shape:
        raise ValueError(f"phys-fer-col-processes: fer_col_ice_out has shape {tuple(ice.shape)}, "
                         f"expected {(ice_props.NSLOT,) + shape}")

    fpvs, fpvs0, _, _ = phys_fer_fpvs.run(ctx)   # FPVS(TK), FPVS0(TK) (lines 916-933)
    one = np.float32(1.0)
    tc = T - T0C                                  # line 915
    esw = cp.minimum(np.float32(1000.0) * fpvs0, np.float32(0.99) * PP)        # line 925
    qsw = EP2 * esw / (PP - esw)                                              # line 926
    esi = cp.minimum(np.float32(1000.0) * fpvs, np.float32(0.99) * PP)        # line 931
    qsi = cp.where(tc < 0, EP2 * esi / (PP - esi), qsw)                       # lines 930-933
    wv = Q / (one - Q)                                                        # line 918
    rho_pos = RHO > 0
    rho_safe = cp.where(rho_pos, RHO, one)
    rrho = cp.where(rho_pos, one / rho_safe, np.float32(0.0))                 # line 968 (0 on clear levels)

    cols["TK"][...] = T
    cols["PP"][...] = PP
    cols["WV"][...] = wv
    cols["QW"][...] = QW
    cols["RHO"][...] = RHO
    cols["RRHO"][...] = rrho
    cols["QSI"][...] = qsi
    cols["QSWGRD"][...] = np.float32(rhgrd) * qsw                              # line 939
    cols["QSIGRD"][...] = np.float32(rhgrd) * qsi                              # line 940
    for name in ICE_FROM_PRODUCER:
        cols[name][...] = ice[SLOT[name]]
    cols["INDEXS"][...] = ice[SLOT["INDEXS"]].astype(cp.int32)
    cols["RCW"][...] = ice[SLOT["RCW"]]
    return cols


def run(ctx, cols=None, rhgrd: float = RHGRD):
    fer = ctx.tables["fer"]
    sc = fer["scalars"]
    if cols is None:
        g = ctx.grid
        cols = alloc(ctx, (g.lm, g.jm, g.im))
    if FLAGS_KEY not in ctx.tables:
        alloc(ctx, tuple(cols["TK"].shape))
    bind_inputs(ctx, cols, rhgrd)
    lm, jm, im = cols["TK"].shape
    k = kernels.get(UNIT, KERNEL)
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.float32(fer["dtph"]), np.float32(rhgrd),
        np.float32(sc["CIACW"]), np.float32(sc["ARAUT"]), np.float32(sc["BRAUT"]),
        np.float32(sc["QAUT0"]), np.float32(sc["ARCW"]),
        np.float32(sc["C1XPVS"]), np.float32(sc["C2XPVS"]),
        np.float32(sc["C1XPVS0"]), np.float32(sc["C2XPVS0"]),
        *[cols[name] for name in IN_F32],
        cols["INDEXS"],
        fer["accri"], fer["venti1"], fer["venti2"], fer["growth"],
        fer["tbpvs"], fer["tbpvs0"],
        *[cols[name] for name in IO_F32],
        *[cols[name] for name in OUT_F32],
        ctx.tables[FLAGS_KEY], ctx.tables[WARN2_KEY],
    )
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D, args)
    return cols
