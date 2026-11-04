"""phys-bmj-col-deep-tend: deep convection heating, moistening and precipitation of the BMJ scheme.

Fortran: phys/module_CU_BMJ.F90:1180-1316 (nam.v4.2.11, 83df4e13c1), the heating/moistening loop of
the cloud-efficiency body (1191-1213), the deep-convection block (1222-1290, dtdt, dqdt, optional
mmntdeep momentum, pcpcol) and the start of the else branch that sets cldefi (1295-1315).
Kernel: namgpu/kernels/phys_bmj_col_deep_tend.cu, entry phys_bmj_col_deep_tend_columns, one thread per
column (the device function bmj_col_deep_tend is what phys_bmj_bmjdrv calls per column).

Pass structure. Fortran runs the efficiency body inside `cloud_efficiency : do itrefi=1,itrefi_max`
(itrefi_max = 3, line 69), and the deep block and else branch once, after the loop. This unit is one
pass of the loop body plus, on the last pass, the post-loop block. The caller drives the loop:

    for p in range(1, ITREFI_MAX + 1):
        phys_bmj_col_deep_iter.deep_iter(...)            # updates BMJ_TREFK, BMJ_QREFK in place
        phys_bmj_col_deep_tend.run(ctx, itrefi_pass=p)   # efi feedback; deep block on p == ITREFI_MAX

Passes before ITREFI_MAX write BMJ_EFI, BMJ_DIFT, BMJ_DIFQ, BMJ_DENTPY, BMJ_PRECK, which the next pass
reads (BMJ_EFI feeds deep-iter's dspbk/dsp0k/dsptk). Only the last pass writes DTDT, DQDT, PCPCOL,
FEFI, CLDEFI, ISWAP. A run() call without itrefi_pass is refused.

Inputs (ctx.st, all REQUIRED; the wrapper never fills them with synthetic values):
  BMJ_TK, BMJ_QK         [LM,JM,IM] float32  column T, Q (phys-bmj-col-prep)
  BMJ_DPRS, BMJ_EL       [LM,JM,IM] float32  layer pressure thickness, EL
  BMJ_TREFK, BMJ_QREFK   [LM,JM,IM] float32  reference T, Q (phys-bmj-col-deep-ref; deep-iter in place)
  BMJ_PK                 [LM,JM,IM] float32  Exner-like pressure profile
  BMJ_SM                 [JM,IM]    float32  land mask sm
  BMJ_PKT, BMJ_RDP0T     [JM,IM]    float32  PKT and RDP0T (deep-ref)
  BMJ_LTOP, BMJ_LB, BMJ_L0 [JM,IM] int32    1-based Fortran levels (deep-ref / cloud-top)
  U_PHY, V_PHY           [LM,JM,IM] float32  column winds (momentum branch, mmntdeep off)
  CLDEFI                 [JM,IM]    float32  cloud efficiency (inout; set here on both branches)
Outputs (ctx.st, added once with st.add):
  BMJ_DIFT, BMJ_DIFQ     [LM,JM,IM] float32  dift, difq (kept for the shallow units)
  BMJ_DTDT, BMJ_DQDT     [LM,JM,IM] float32  deep heating and moistening tendencies (K/s, kg/kg/s)
  BMJ_EFI, BMJ_DENTPY, BMJ_PRECK, BMJ_FEFI, BMJ_PCPCOL [JM,IM] float32
  BMJ_SUMDP              [JM,IM]    float32  sumdp, the enthalpy-loop sum of dprs over ltop..lb (set by the
                                              kernel each pass, as the Fortran enthalpy loop does)
  BMJ_ISWAP              [JM,IM]    int32    iswap (0 on the deep branch; see the kernel header)

Momentum (mmntdeep): BMJ sets mmntdeep = .false. unconditionally (module_CU_BMJ.F90:565), so the
DUDT/DVDT branch never runs. The kernel still takes dudt, dvdt pointers; the wrapper passes two private
scratch arrays that are never written and never exposed, so DUDT_PHY and DVDT_PHY are not touched,
exactly as the Fortran leaves the caller's arrays untouched.

Column domain and gating (Fortran bmjdrv 213-249). bmj is called only when icldck = mod(ntsd, ncnvc) == 0
or ntsd == 0; the wrapper returns without writing when it is not. The kernel loops over the _b1 limits
(module_DM_PARALLEL looplimits: its_b1 = max(its, ids+1), ite_b1 = min(ite, ide-1), likewise jts_b1,
jte_b1); the wrapper builds the int32 mask act[JM,IM] from them (1-based i in 2..IM-1, j in 2..JM-1) and
the kernel skips columns with act == 0. cu_act_flag (bmjdrv, set at 232-234) is owned by bmjdrv, not here.

Scalars: CP, CPRLG = cp/(row*g*elwv), TAUK = dtcnvc/(trel*1), RDTCNVC = 1/dtcnvc with dtcnvc = dt * nprecip,
UVSCALD = 0.01 (Fortran local), nodeep from the configfile. SLOPES and SLOPEL are evaluated in float32
as Fortran parameters (1.-efmnts)/(1.-efimn) and (1.-efmntl)/(1.-efimn), kfpt single.

Launch: one kernel on ctx.stream, no host sync, no per-step allocation (outputs and scratch added or cached
once).

Refusals:
  - LM must be 60 (the NAM 12 km configuration, CONTRACT section 5).
  - Every REQUIRED input must be in ctx.st, or run() raises naming it (BMJ_LTOP/LB/L0 absent would
    otherwise be zero and the kernel would write COL_AT(dift, 0) out of bounds).
  - itrefi_pass must be given and be in 1..ITREFI_MAX.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_col_deep_tend"
KERNEL = "phys_bmj_col_deep_tend_columns"
BMJ_LM = 60
ITREFI_MAX = 3  # module_CU_BMJ.F90:69 (itrefi_max=3)

# Constants (module_CU_BMJ.F90 parameters and the bmjdrv call constants), single precision (kfpt).
CP = np.float32(1004.6)
G = np.float32(9.8060226)
ROW = np.float32(1.0e3)
ELWV = np.float32(2.501e6)
TREL = np.float32(2400.0)
EFIFC = np.float32(5.0)
EFIMN = np.float32(0.20)
EFMNTL = np.float32(0.70)
EFMNTS = np.float32(0.70)
EPSNTP = np.float32(0.0001)
EPSPR = np.float32(1.0e-7)
STEFI = np.float32(1.0)
UVSCALD = np.float32(0.01)
# Fortran parameters slopel=(1.-efmntl)/(1.-efimn), slopes=(1.-efmnts)/(1.-efimn), evaluated in kfpt.
SLOPES = np.float32(np.float32(1.0) - EFMNTS) / np.float32(np.float32(1.0) - EFIMN)
SLOPEL = np.float32(np.float32(1.0) - EFMNTL) / np.float32(np.float32(1.0) - EFIMN)

LEVEL_IN = ("BMJ_TK", "BMJ_QK", "BMJ_DPRS", "BMJ_EL", "BMJ_TREFK", "BMJ_QREFK", "BMJ_PK")  # [LM,JM,IM] f32
COLUMN_IN = {  # name -> dtype, shape [JM, IM]
    "BMJ_SM": "float32", "BMJ_PKT": "float32", "BMJ_RDP0T": "float32",
    "BMJ_LTOP": "int32", "BMJ_LB": "int32", "BMJ_L0": "int32",
}
LEVEL_OUT = ("BMJ_DIFT", "BMJ_DIFQ", "BMJ_DTDT", "BMJ_DQDT")  # [LM,JM,IM] f32
COLUMN_OUT = {  # name -> dtype, shape [JM, IM]
    "BMJ_EFI": "float32", "BMJ_SUMDP": "float32", "BMJ_DENTPY": "float32", "BMJ_PRECK": "float32",
    "BMJ_FEFI": "float32", "BMJ_PCPCOL": "float32", "BMJ_ISWAP": "int32",
}
EXTERNAL_IN = ("U_PHY", "V_PHY", "CLDEFI")

_cache: dict = {}


def _flag(raw: dict, key: str) -> np.int32:
    return np.int32(1 if bool(raw[key]) else 0)


def _private(name: str, shape, dtype):
    """Private scratch, allocated once per shape and never exposed in ctx.st."""
    import cupy as cp

    key = (name, tuple(shape), dtype)
    if key not in _cache:
        _cache[key] = cp.zeros(shape, dtype=dtype)
    return _cache[key]


def _b1_mask(im: int, jm: int):
    """int32 act[JM,IM]: 1 where the Fortran _b1 limits (its_b1..ite_b1, jts_b1..jte_b1) include the column."""
    import cupy as cp

    key = ("act", im, jm)
    if key not in _cache:
        i = cp.arange(1, im + 1)[None, :]
        j = cp.arange(1, jm + 1)[:, None]
        _cache[key] = (((i >= 2) & (i <= im - 1)) & ((j >= 2) & (j <= jm - 1))).astype(cp.int32)
    return _cache[key]


def run(ctx, *, itrefi_pass: int):
    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm != BMJ_LM:
        raise RuntimeError(f"phys-bmj-col-deep-tend: lm={lm}; the kernel is sized for {BMJ_LM} levels")
    if itrefi_pass is None or not (1 <= int(itrefi_pass) <= ITREFI_MAX):
        raise RuntimeError(
            f"phys-bmj-col-deep-tend: itrefi_pass={itrefi_pass!r}; the caller must pass the pass number "
            f"1..{ITREFI_MAX} of the itrefi loop (the deep block runs only on pass {ITREFI_MAX})")
    itrefi_pass = int(itrefi_pass)

    missing = [n for n in LEVEL_IN + tuple(COLUMN_IN) + EXTERNAL_IN if n not in st.fields]
    if missing:
        raise RuntimeError(
            "phys-bmj-col-deep-tend: required inputs not in ctx.st (the upstream unit must write them): "
            + ", ".join(missing))

    for name in LEVEL_OUT:
        if name not in st.fields:
            st.add(name, (lm, jm, im), dtype="float32")
    for name, dt in COLUMN_OUT.items():
        if name not in st.fields:
            st.add(name, (jm, im), dtype=dt)

    ntsd = int(ctx.ntimestep)
    ncnvc = int(cfg.nprecip)
    # Fortran bmjdrv 213-226: bmj runs when icldck == 0 or ntsd == 0. Nothing is written otherwise.
    if not (ntsd == 0 or ntsd % ncnvc == 0):
        return st["BMJ_DTDT"], st["BMJ_DQDT"]
    if itrefi_pass == 1:
        st["BMJ_ISWAP"][...] = 0  # Fortran 574: iswap=0 once per bmj call, before the itrefi loop

    dtcnvc = np.float32(float(cfg.dt) * ncnvc)
    cprlg = np.float32(CP / (ROW * G * ELWV))
    tauk = np.float32(dtcnvc / (TREL * np.float32(1.0)))
    rdtcnvc = np.float32(np.float32(1.0) / dtcnvc)
    raw = cfg.raw
    mmntdeep = np.int32(0)  # BMJ sets mmntdeep = .false. (module_CU_BMJ.F90:565)
    last_pass = np.int32(1 if itrefi_pass == ITREFI_MAX else 0)

    f = st.fields
    dudt_scratch = _private("dudt", (lm, jm, im), "float32")  # never written: mmntdeep is off
    dvdt_scratch = _private("dvdt", (lm, jm, im), "float32")
    act = _b1_mask(im, jm)
    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lm),
            f["BMJ_TK"], f["BMJ_QK"], f["BMJ_DPRS"], f["BMJ_EL"],
            f["BMJ_TREFK"], f["BMJ_QREFK"], f["BMJ_PK"], f["U_PHY"], f["V_PHY"],
            f["BMJ_DIFT"], f["BMJ_DIFQ"], f["BMJ_DTDT"], f["BMJ_DQDT"],
            dudt_scratch, dvdt_scratch,
            f["BMJ_LTOP"], f["BMJ_LB"], f["BMJ_L0"],
            f["BMJ_SUMDP"], f["BMJ_SM"], f["BMJ_PKT"], f["BMJ_RDP0T"],
            f["BMJ_EFI"], f["CLDEFI"], f["BMJ_DENTPY"], f["BMJ_PRECK"],
            f["BMJ_FEFI"], f["BMJ_PCPCOL"], f["BMJ_ISWAP"],
            mmntdeep, _flag(raw, "nodeep"),
            CP, cprlg, tauk, rdtcnvc, UVSCALD,
            EFIFC, EFIMN, EPSNTP, EPSPR, EFMNTL, EFMNTS, SLOPES, SLOPEL, STEFI,
            act, last_pass,
        ))
    return f["BMJ_DTDT"], f["BMJ_DQDT"]
