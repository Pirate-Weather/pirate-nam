"""phys-bmj-col-shallow-adjust: BMJ shallow convection adjustment for every shallow column.

Ported from phys/module_CU_BMJ.F90:1450-2067 (NAM nam.v4.2.11, 83df4e13c1), the shallow part of
subroutine bmj. Kernel: namgpu/kernels/phys_bmj_col_shallow_adjust.cu, entry
phys_bmj_col_shallow_adjust_columns, one launch, one thread per column (grid_cols), on ctx.stream.
The same column code is the __device__ function bmj_col_shallow_adjust, which the BMJ column scheme
(bmjdrv) calls for each column. The statement map and the undefined-read guards are in the kernel header.

Inputs are read from the state names the upstream units write (SOURCE below). Every input is required:
a missing one raises, naming the producers, and nothing is filled with synthetic values. Producers:
    phys-bmj-col-prep        BMJ_TK, BMJ_QK, BMJ_RXNER, BMJ_FLAG (flag rows 1 plume, 2 shallow),
                             PHMID (prsmid; the driver writes it too), U_PHY, V_PHY (state),
                             BMJ_DTDT, BMJ_DQDT, BMJ_DUDT, BMJ_DVDT (zeroed per column, Fortran
                             bmjdrv 272-277, before any BMJ column unit)
    phys-bmj-col-cloudtop    BMJ_CT_DPRS (dprs = PINT(k+1)-PINT(k)); final post-restore state after the
                             maximum-instability block (Fortran 906-939): BMJ_CT_LBOT (lbot after the
                             restore, Fortran 914), BMJ_CT_PBOT, BMJ_CT_PSP, BMJ_CT_THBT, BMJ_CPE, BMJ_DTV,
                             BMJ_THES (equal to thescnv on the cape>0 path; the kernel only adjusts when
                             lbot >= 1, which is the cape>0 path)
    phys-bmj-col-shallow-top BMJ_LTOP, BMJ_PTOP: the shallow cloud-top search (Fortran 1317-1449). In/out.
                             The shallow-top unit does not yet write these into ctx.st, so the wrapper
                             refuses until it does.

Read and written [JM, IM]: BMJ_LBOT (int32, a copy of BMJ_CT_LBOT that the kernel adjusts), BMJ_LTOP
(int32), BMJ_PTOP (float32): shallow-top values in, adjusted values out.
Tables (ctx.tables, BMJ_INIT, phys-bmj-init): bmj_ptbl [JTB, ITB], bmj_qs0, bmj_sqs [JTB],
bmj_sthe, bmj_the0 [ITB], bmj_ttbl [ITB, JTB], bmj_stheq, bmj_the0q [ITBQ], bmj_ttblq [ITBQ, JTBQ].
Scalars: NPRECIP (cfg.nprecip) and DT (cfg.dt) give DTCNVC = NPRECIP*DT; RDTCNVC = 1/DTCNVC;
TAUKSC = DTCNVC/(trel*1.0) with trel = 2400 (module_CU_BMJ.F90:570); UVSCALS1 = UVSCALS2 = 0.01
(line 102-104); MMNTSHAL1 = MMNTSHAL2 = .true. (lines 566-567). newall, newswap, newupup, nodeep come
from cfg.raw (the configfile switches bmjdrv passes; nodeep defaults to false). Constants from
nmmb_common.cuh.

Writes: BMJ_DTDT, BMJ_DQDT, BMJ_DUDT, BMJ_DVDT [LM, JM, IM]. These are the prep-zeroed fields; the kernel
writes ltop..lbot of shallow columns only, so deep-tend results on deep columns are never touched.
BMJ_LBOT, BMJ_LTOP [JM, IM] int32, BMJ_PTOP [JM, IM].

Undefined Fortran reads, guarded in the kernel (kernel header has the full list): lbot < 1 (no shallow
column, nothing written); ltop = 1 (t(ltp1) and q(ltp1) would read index 0); lbot = lmh (trefk(lbot+1),
pk(lbot+1) would read past the top). The guarded columns get the quick-exit values (lbot = 0, ltop = LM,
ptop = pbot), as the Fortran own exits do. The kernel leaves the column alone for lbot < 1.

Done rule (check_unit path): run(ctx, standins=True) fills the missing producer names with physically
shaped stand-ins and zeroed tendency scratch, and says so. The default run(ctx) never stands in: it
raises on any missing input. The check tool is tools/check-phys-bmj-col-shallow-adjust.py.

Launch: one kernel on ctx.stream, no host sync, no Python loop over cells, no per-step allocation after the
first call. Refused: LM must be 60 (the kernel per-column arrays are sized for the 12 km CONUS config).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_col_shallow_adjust"
KERNEL = "phys_bmj_col_shallow_adjust_columns"
BMJ_LM = 60

# Scalars (nmmb_common.cuh; module_CU_BMJ.F90 and module_CONSTANTS values, single precision).
R_D = np.float32(287.04)
R_V = np.float32(461.6)
CP = np.float32(1004.6)
CAPPA = np.float32(R_D / CP)
P608 = np.float32(R_V / R_D - np.float32(1.0))
A2 = np.float32(17.2693882)
A3 = np.float32(273.15)
A4 = np.float32(35.86)
PQ0 = np.float32(379.90516)
TIW = np.float32(273.15)
ELWV = np.float32(2.501e6)
ELIV = np.float32(2.850e6)
TREL = np.float32(2400.0)          # module_CU_BMJ.F90:57 trel
UVSCALS = np.float32(0.01)         # uvscals1 = uvscals2 (lines 102-104)

# Kernel input name -> state name the producer writes. Identity where the producer uses the same name.
SOURCE = {
    "BMJ_PRSMID": "PHMID",
    "BMJ_TK": "BMJ_TK", "BMJ_QK": "BMJ_QK", "BMJ_RXNER": "BMJ_RXNER",
    "BMJ_DPRS": "BMJ_CT_DPRS",
    "BMJ_THESCNV": "BMJ_THES",
    "BMJ_CPE": "BMJ_CPE", "BMJ_DTV": "BMJ_DTV",
    "BMJ_PBOT": "BMJ_CT_PBOT", "BMJ_PSP": "BMJ_CT_PSP", "BMJ_THBT": "BMJ_CT_THBT",
    "BMJ_FLAG": "BMJ_FLAG",
    "U_PHY": "U_PHY", "V_PHY": "V_PHY",
}
LBOT_SOURCE = "BMJ_CT_LBOT"                       # copied into BMJ_LBOT, which the kernel adjusts
SHALLOW_TOP_IN = ("BMJ_LTOP", "BMJ_PTOP")         # written by phys-bmj-col-shallow-top (not wired yet)
TENDENCY_FROM_PREP = ("BMJ_DTDT", "BMJ_DQDT", "BMJ_DUDT", "BMJ_DVDT")
TABLE_KEYS = ("bmj_ptbl", "bmj_qs0", "bmj_sqs", "bmj_sthe", "bmj_the0", "bmj_ttbl",
              "bmj_stheq", "bmj_the0q", "bmj_ttblq")


def _put(st, name, value):
    """Done-rule stand-in only: add a producer array when the name is not there yet."""
    if name in st.fields:
        return st[name]
    arr = st.add(name, tuple(value.shape), dtype=value.dtype.name)
    arr[...] = value
    return arr


def _standins(st, im, jm, lm):
    """Done-rule stand-ins under the producer names (physically shaped; see the module docstring)."""
    import cupy as cp

    rng = cp.random.default_rng(11)
    shape3, shape2 = (lm, jm, im), (jm, im)
    if "PHMID" not in st.fields:
        base = cp.asarray(np.geomspace(3000.0, 1.0e5, lm).astype(np.float32))[:, None, None]
        _put(st, "PHMID", (base * (1.0 + 0.01 * (rng.random(shape3, dtype=cp.float32) - 0.5))).astype(cp.float32))
    pm = st["PHMID"]
    _put(st, "BMJ_CT_DPRS", (0.0612 * pm).astype(cp.float32))
    _put(st, "BMJ_TK", (220.0 + 85.0 * rng.random(shape3, dtype=cp.float32)).astype(cp.float32))
    _put(st, "BMJ_QK", (1.0e-6 + 0.02 * rng.random(shape3, dtype=cp.float32) ** 2).astype(cp.float32))
    _put(st, "BMJ_RXNER", cp.power(cp.float32(1.0e5) / pm, CAPPA).astype(cp.float32))
    _put(st, "BMJ_THES", (300.0 + 40.0 * rng.random(shape3, dtype=cp.float32)).astype(cp.float32))
    _put(st, "BMJ_CPE", rng.standard_normal(shape3, dtype=cp.float32))
    _put(st, "BMJ_DTV", rng.standard_normal(shape3, dtype=cp.float32))
    _put(st, "U_PHY", (60.0 * rng.random(shape3, dtype=cp.float32) - 30.0).astype(cp.float32))
    _put(st, "V_PHY", (60.0 * rng.random(shape3, dtype=cp.float32) - 30.0).astype(cp.float32))
    lbot = cp.random.randint(5, lm, size=shape2).astype(cp.int32)          # [5, lm-1]
    ltop = (rng.random(shape2, dtype=cp.float32) * (lbot - 4).astype(cp.float32)).astype(cp.int32) + 2
    _put(st, LBOT_SOURCE, lbot)                                             # ltop in [2, lbot-3]
    _put(st, "BMJ_LTOP", ltop)
    _put(st, "BMJ_PTOP", cp.take_along_axis(pm, (ltop - 1)[None], axis=0)[0].astype(cp.float32))
    _put(st, "BMJ_CT_PBOT", cp.take_along_axis(pm, (lbot - 1)[None], axis=0)[0].astype(cp.float32))
    _put(st, "BMJ_CT_PSP", (5.0e4 + 5.0e4 * rng.random(shape2, dtype=cp.float32)).astype(cp.float32))
    _put(st, "BMJ_CT_THBT", (290.0 + 20.0 * rng.random(shape2, dtype=cp.float32)).astype(cp.float32))
    flag = cp.zeros((7, jm, im), dtype=cp.int32)
    flag[1] = (rng.random(shape2, dtype=cp.float32) < 0.2).astype(cp.int32)   # plume
    flag[2] = 1                                                                # shallow everywhere
    _put(st, "BMJ_FLAG", flag)


def run(ctx, *, standins: bool = False):
    from namgpu.physics.bmj import phys_bmj_init

    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm != BMJ_LM:
        raise RuntimeError(f"phys-bmj-col-shallow-adjust: lm={lm}; the kernel is sized for {BMJ_LM} levels")
    if "bmj_ptbl" not in ctx.tables:
        phys_bmj_init.run(ctx)
    for key in TABLE_KEYS:
        if key not in ctx.tables:
            raise RuntimeError(f"phys-bmj-col-shallow-adjust: ctx.tables['{key}'] missing (phys-bmj-init)")

    shape3, shape2 = (lm, jm, im), (jm, im)
    if standins:
        _standins(st, im, jm, lm)
        for name in TENDENCY_FROM_PREP:
            st.add(name, shape3, dtype="float32")   # zero scratch stand-in for the prep zeroing

    need = sorted(set(SOURCE.values()) | {LBOT_SOURCE} | set(SHALLOW_TOP_IN) | set(TENDENCY_FROM_PREP))
    missing = [n for n in need if n not in st.fields]
    if missing:
        raise RuntimeError(
            "phys-bmj-col-shallow-adjust: required inputs not in ctx.st (their producers must run first: "
            "phys-bmj-col-prep, phys-bmj-col-cloudtop, phys-bmj-col-shallow-top; the BMJ_LTOP/BMJ_PTOP "
            "writer is not wired yet): " + ", ".join(missing))

    raw = cfg.raw
    nprecip = int(cfg.nprecip)
    dtcnvc = np.float32(nprecip) * np.float32(cfg.dt)
    rdtcnvc = np.float32(1.0) / dtcnvc
    tauksc = np.float32(dtcnvc / (TREL * np.float32(1.0)))

    # Tendencies are zeroed by phys-bmj-col-prep before the BMJ column units. They are not zeroed here,
    # so the deep heating written by phys-bmj-col-deep-tend on deep columns survives this unit.
    dtdt, dqdt = st["BMJ_DTDT"], st["BMJ_DQDT"]
    dudt, dvdt = st["BMJ_DUDT"], st["BMJ_DVDT"]

    # lbot is adjusted in place by the kernel; copy the cloud-top value so the producer array is not modified.
    lbot = st.add("BMJ_LBOT", shape2, dtype="int32")
    lbot[...] = st[LBOT_SOURCE]
    ltop = st["BMJ_LTOP"]
    ptop = st["BMJ_PTOP"]

    def src(name):
        return st[SOURCE.get(name, name)]

    flag = src("BMJ_FLAG")
    tab = ctx.tables

    def flag_of(v):
        return np.int32(1 if bool(v) else 0)

    i32 = np.int32
    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            i32(im), i32(jm), i32(lm), i32(lm),
            flag_of(raw["newall"]), flag_of(raw["newswap"]), flag_of(raw["newupup"]),
            flag_of(raw["nodeep"]), i32(1), i32(1),                       # mmntshal1, mmntshal2
            A2, A3, A4, CAPPA, CP, P608, PQ0, TIW, ELWV, ELIV,
            rdtcnvc, tauksc, UVSCALS, UVSCALS,
            flag[2], flag[1],                                              # shallow, plume
            src("BMJ_PBOT"), src("BMJ_PSP"), src("BMJ_THBT"),
            src("BMJ_PRSMID"), src("BMJ_DPRS"), src("BMJ_TK"), src("BMJ_QK"), src("BMJ_RXNER"),
            src("BMJ_THESCNV"), src("BMJ_CPE"), src("BMJ_DTV"), src("U_PHY"), src("V_PHY"),
            dtdt, dqdt, dudt, dvdt,
            lbot, ltop, ptop,
            tab["bmj_ptbl"], tab["bmj_qs0"], tab["bmj_sqs"], tab["bmj_sthe"], tab["bmj_the0"],
            tab["bmj_ttbl"], tab["bmj_stheq"], tab["bmj_the0q"], tab["bmj_ttblq"],
        ))
    return {"BMJ_DTDT": dtdt, "BMJ_DQDT": dqdt, "BMJ_DUDT": dudt, "BMJ_DVDT": dvdt,
            "BMJ_LBOT": lbot, "BMJ_LTOP": ltop, "BMJ_PTOP": ptop}
