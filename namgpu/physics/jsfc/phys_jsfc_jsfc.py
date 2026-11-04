"""phys-jsfc-jsfc: JSFC surface-layer driver, three launches on ctx.stream, no host sync.

Ported from phys/module_SF_JSFC.F90:80-354 (NAM nam.v4.2.11, 83df4e13c1), the subroutine JSFC.
Kernels: namgpu/kernels/phys_jsfc_jsfc.cu, entries phys_jsfc_jsfc_pre (Fortran :99-178) and
phys_jsfc_jsfc_post (:203-220). The CALL SFCDIF (:181-200) is phys-jsfc-sfcdif, launched between
them with the same arguments the Fortran call passes.

Caller: module_TURBULENCE.F90:798-802 (sfclay JSFC branch). The Fortran arguments map to the
state names the turbulence driver (phys_turbl_driver.py) owns:
  HT <- SFCZ (Z interface heights), TSK <- TSFC (surface temperature from THS/EXNSFC),
  U <- U_PHY, V <- V_PHY (H-point winds from TURBULENCE:703), QSFC <- QSH,
  MAVAIL <- MAVAIL (the NMM constant 1, set by the turbulence driver; ones when absent),
  TH2 outputs: T02 -> T2, TH02 -> TH2X, Q02 -> Q2X; SFCDIF TH02/Q02 -> TSHLTR/QSHLTR.
  Surface fluxes: HFX -> TWBS, QFX -> QWBS, FLX_LH -> ELFLX; RMOL, CT, RIB as named.

Tile: every stage runs over the module_TURBULENCE B1 tile (ctx.tables["NMMB_LIMITS"]). Columns
outside it are not read or written, as Fortran leaves them.

Refusals: a missing input field or table is a RuntimeError that names it. Nothing is filled with
synthetic values. The full-size no-NaN check is tests/check_phys_jsfc_jsfc_fullsize.py.

Launch: on ctx.stream, no host sync, no per-step allocation (scratch and outputs are created once,
by name, with st.add).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab
from namgpu.physics.jsfc import phys_jsfc_sfcdif

UNIT = "phys_jsfc_jsfc"
KERNEL_PRE = "phys_jsfc_jsfc_pre"
KERNEL_POST = "phys_jsfc_jsfc_post"
LIMITS_KEY = "NMMB_LIMITS"
TABLE_KEY = "jsfc"

# State fields the Fortran JSFC arguments are read from (inputs).
REQUIRED = ("SFCZ", "TSFC", "XLAND", "Z0BASE", "VEGFRC", "SNOWC", "EPSQ2", "PHINT", "DZ", "PHMID",
            "TH", "T", "Q", "QC", "U_PHY", "V_PHY", "Q2", "THZ0", "QSH", "UZ0", "VZ0", "QZ0",
            "USTAR", "Z0", "AKHS", "AKMS")

# INOUT state (JSFC INTENT(INOUT)), by Fortran name -> state name. PBLH and RIB and RMOL are
# zero-allocated on first use (Fortran allocates them with the domain).
INOUT_STATE = {"QSFC": "QSH", "THZ0": "THZ0", "QZ0": "QZ0", "UZ0": "UZ0", "VZ0": "VZ0",
               "USTAR": "USTAR", "Z0": "Z0", "RMOL": "RMOL", "AKHS": "AKHS", "AKMS": "AKMS",
               "PBLH": "PBLH", "RIB": "RIB"}

# OUT fields, by the SFCDIF output name (the CALL SFCDIF argument order) -> state name.
SFCDIF_OUT_STATE = {"CT": "CT", "RLMO": "RMOL", "RIB": "RIB", "CHS": "CHS", "CHS2": "CHS2",
                    "CQS2": "CQS2", "HFX": "TWBS", "QFX": "QWBS", "FLX_LH": "ELFLX",
                    "FLHC": "FLHC", "FLQC": "FLQC", "QGH": "QGH", "CPM": "CPM", "U10": "U10",
                    "V10": "V10", "TH02": "TSHLTR", "TH10": "TH10", "Q02": "QSHLTR",
                    "Q10": "Q10", "PSHLTR": "PSHLTR"}

# Outputs written by the post stage (T02, TH02, Q02 of JSFC).
POST_OUT_STATE = {"T02": "T2", "TH02": "TH2X", "Q02": "Q2X"}

# Scratch between the stages, one [JM, IM] array per name (state name is "JSFC_" + name).
SCRATCH = ("SEAMASK", "THSK", "PSFC", "TZ0", "ZSL", "ULOW", "VLOW", "TLOW", "THLOW", "THELOW",
           "QLOW", "CWMLOW", "PLOW")


def _ones(ctx, jm, im):
    """MAVAIL = 1 when the state has no MAVAIL field (NMM sets it to 1; kept once in ctx.tables)."""
    import cupy as cp

    key = "JSFC_MAVAIL_ONE"
    if key not in ctx.tables:
        ctx.tables[key] = cp.ones((jm, im), dtype=cp.float32)
    return ctx.tables[key]


def run(ctx, ntsd=None):
    """The three launches over the whole B1 tile. ntsd defaults to ctx.ntimestep (the turbulence driver
    calls run(ctx) with no argument, and the NTSD==0 branches depend on the step). Returns the output dict."""
    if ntsd is None:
        ntsd = int(ctx.ntimestep)
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    tables = ctx.tables

    if LIMITS_KEY not in tables:
        raise RuntimeError("phys-jsfc-jsfc: ctx.tables['NMMB_LIMITS'] missing (drv-loop-limits must run first)")
    if TABLE_KEY not in tables:
        raise RuntimeError("phys-jsfc-jsfc: ctx.tables['jsfc'] missing (phys-jsfc-init must run first)")
    missing = [n for n in REQUIRED if n not in st.fields]
    if missing:
        raise RuntimeError("phys-jsfc-jsfc: state fields missing: %s (refused, no synthetic fallback)"
                           % ", ".join(missing))

    f = st.fields
    for name in list(INOUT_STATE.values()) + list(SFCDIF_OUT_STATE.values()) + list(POST_OUT_STATE.values()):
        st.add(name, (jm, im))
    scr = {n: st.add("JSFC_" + n, (jm, im)) for n in SCRATCH}
    mavail = f["MAVAIL"] if "MAVAIL" in f else _ones(ctx, jm, im)
    lim = tables[LIMITS_KEY]
    if LIMITS_KEY == "NMMB_LIMITS":
        lim = slab.local_limits(ctx)   # row slab: j limits in local rows (one card: the same table)

    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    i32 = np.int32
    with scope:
        # stage 1: :99-178
        kernels.get(UNIT, KERNEL_PRE)(grid, block, (
            i32(im), i32(jm), i32(lm), i32(ntsd), lim,
            f["EPSQ2"], f["SFCZ"], f["TSFC"], f["XLAND"], f["DZ"], f["PHMID"], f["PHINT"],
            f["TH"], f["T"], f["Q"], f["QC"], f["U_PHY"], f["V_PHY"], f["Q2"],
            f["THZ0"], f["USTAR"], f["PBLH"], f["CT"],
            scr["SEAMASK"], scr["THSK"], scr["PSFC"], scr["TZ0"], scr["ZSL"],
            scr["ULOW"], scr["VLOW"], scr["TLOW"], scr["THLOW"], scr["THELOW"],
            scr["QLOW"], scr["CWMLOW"], scr["PLOW"],
        ))

        # stage 2: CALL SFCDIF (:181-200), same argument roles as the Fortran call
        arrs = {
            "SEAMASK": scr["SEAMASK"], "THS": scr["THSK"], "PSFC": scr["PSFC"], "TZ0": scr["TZ0"],
            "ZSL": scr["ZSL"], "PBLH": f["PBLH"], "WETM": mavail, "Z0BASE": f["Z0BASE"],
            "ULOW": scr["ULOW"], "VLOW": scr["VLOW"], "TLOW": scr["TLOW"], "THLOW": scr["THLOW"],
            "THELOW": scr["THELOW"], "QLOW": scr["QLOW"], "CWMLOW": scr["CWMLOW"],
            "PLOW": scr["PLOW"],
            "QS": f["QSH"], "UZ0": f["UZ0"], "VZ0": f["VZ0"], "THZ0": f["THZ0"], "QZ0": f["QZ0"],
            "USTAR": f["USTAR"], "Z0": f["Z0"], "AKMS": f["AKMS"], "AKHS": f["AKHS"],
        }
        outs = {k: f[v] for k, v in SFCDIF_OUT_STATE.items()}
        phys_jsfc_sfcdif.launch(im, jm, ntsd, tables[TABLE_KEY], arrs, outs, lim, stream=ctx.stream)

        # stage 3: :203-220
        kernels.get(UNIT, KERNEL_POST)(grid, block, (
            i32(im), i32(jm), lim, scr["PSFC"], f["TSHLTR"], f["TH10"], f["QSHLTR"], f["Q10"],
            f["TH2X"], f["T2"], f["Q2X"],
        ))

    return {n: f[n] for n in list(INOUT_STATE.values()) + list(SFCDIF_OUT_STATE.values())
            + list(POST_OUT_STATE.values())}
