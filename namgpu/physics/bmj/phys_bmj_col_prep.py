"""phys-bmj-col-prep: BMJ arguments, locals and column preparation (BMJ subroutine, lines 399-597).

Ported from phys/module_CU_BMJ.F90:399-597 (NAM nam.v4.2.11, 83df4e13c1). Kernel:
namgpu/kernels/phys_bmj_col_prep.cu (phys_bmj_col_prep), one launch, one thread per column
(grid_cols), no host sync, no per-step allocation after the first call.

Reads (state, CONTRACT names):
    T       float32 [LM, JM, IM]     TCOL
    Q       float32 [LM, JM, IM]     QCOL (the kernel clamps to MAX(EPSQ, Q) as bmjdrv does)
    PINT    float32 [LM+1, JM, IM]   PSFC = PINT[LM] (phint(i,j,lm+1))
    PHMID   float32 [LM, JM, IM]     PRSMID (PCOL). Written by the BMJ driver unit (CUCNVC prep,
                                     module_CONVECTION.F90:355, phys_bmj_driver.cu:138). If the
                                     driver has not added it, this wrapper computes the same CUCNVC
                                     formula: PHMID = SGML2(K)*PD + PSGML1(K) (PDSL = PD(I,J)).
    EXNER   float32 [LM, JM, IM]     EXNERCOL. Same ownership as PHMID. Otherwise computed as the
                                     CUCNVC formula EXNER = (PHMID*1.E-5)**CAPPA, CAPPA = R_D/CP
                                     (module_CONVECTION.F90:362, module_CONSTANTS cappa=r/cp).

Writes (scratch, added with st.add, zero on first use; per level [LM, JM, IM] unless noted).
    BMJ_SCAL slot 16 (pcpcol) binding: the Fortran PCPCOL is intent(inout) in BMJ. This unit writes
    its Fortran zero (module_CU_BMJ.F90:580). The BMJ column chain (deep/shallow, phys-bmj-col-*)
    accumulates it, and the RAINCV step (module_CU_BMJ.F90:268, phys-bmj-bmjdrv, RAINCV =
    PCPCOL*1.E3/NCNVC) reads it. Any unit that reads convective precipitation must read
    BMJ_SCAL[16] (the seed of that chain) and not a second copy. phys-bmj-bmjdrv status is todo; its
    kernel seeds its own pcpcol register at 0 (phys_bmj_bmjdrv.cu) and must take the seed from slot
    16 when it lands.
    BMJ_TK BMJ_QK BMJ_RXNER BMJ_CPECNV BMJ_DTVCNV BMJ_THES BMJ_THESP BMJ_THESCNV
    BMJ_QSATK BMJ_RHK BMJ_DUDT BMJ_DVDT
    BMJ_DTDT BMJ_DQDT  zeroed here per column, before any BMJ column unit: Fortran bmjdrv zeroes
                       dqdt, dtdt, dudt, dvdt for every column before bmj (module_CU_BMJ.F90:272-277).
                       phys-bmj-col-deep-tend writes them on deep columns and
                       phys-bmj-col-shallow-adjust on shallow columns; neither zeroes a whole field.
    BMJ_SCAL   float32 [18, JM, IM]  per-column scalars: 0 cprlg, 1 elocp, 2 rcp, 3 a23m4l,
                                     4 uvscald, 5 uvscals1, 6 uvscals2, 7 rdtcnvc, 8 depmin,
                                     9 rh_deep, 10 tauk, 11 tauksc, 12 cup, 13 dsp0, 14 dspb,
                                     15 dspt, 16 pcpcol, 17 tref(1)
    BMJ_FLAG   int32  [7, JM, IM]    0 deep, 1 plume, 2 shallow, 3 mmntdeep, 4 mmntshal1,
                                     5 mmntshal2, 6 iswap

Constants (nmmb_common.cuh, the Fortran literals): A2 17.2693882, A3 273.15, A4 35.86, CP 1004.6,
G 9.8060226, ELWV 2.501e6, PQ0 379.90516, EPSQ 1e-12. DTCNVC = NPRECIP * dt = 2 * dt (50 s).
EPSQ provenance (verified): module_SOLVER_GRID_COMP.F90:65 imports EPSQ from MODULE_CONSTANTS
(module_CONSTANTS.F90:27, epsq=1.e-12) and passes it as the CUCNVC EPSQ argument (line 5775). The
1.e-20 in module_DYNAMICS_ROUTINES is the dynamics floor and is not the BMJ argument.
NODEEP = 0 (the configfile does not set nodeep, so the 0.75 critical humidity applies).

Launch: no host sync, no per-step allocation after the first call, everything on ctx.stream.
Refused: nothing; the routine has no branch that the 12 km configuration skips.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_col_prep"
KERNEL = "phys_bmj_col_prep"
BLOCK = 128

NSC = 18  # BMJ_SCAL slot 16 is pcpcol (see the binding above)
NFL = 7
LEVEL_OUT = (
    "BMJ_TK", "BMJ_QK", "BMJ_RXNER", "BMJ_CPECNV", "BMJ_DTVCNV", "BMJ_THES", "BMJ_THESP",
    "BMJ_THESCNV", "BMJ_QSATK", "BMJ_RHK", "BMJ_DUDT", "BMJ_DVDT", "BMJ_DTDT", "BMJ_DQDT",
)

# nmmb_common.cuh values (NMMB module_CONSTANTS and module_CU_BMJ inputs)
A2 = 17.2693882
A3 = 273.15
A4 = 35.86
CP = 1004.6
G = 9.8060226
ELWV = 2.501e6
PQ0 = 379.90516
EPSQ = 1.0e-12
NPRECIP = 2
R_D = 287.04
CAPPA = np.float32(R_D / CP)     # module_CONSTANTS cappa = r/cp, stored as real*4


def _driver_inputs(st, g, im, jm, lm, shape3):
    """PHMID and EXNER as the CUCNVC prep computes them (module_CONVECTION.F90:355 and 362).

    The phys-bmj-driver unit writes the same two arrays (phys_bmj_driver.cu:138-144). When it has
    run, its values are used. When it has not, the CUCNVC formula is evaluated here from the state
    and the vertical constants: PHMID(K) = SGML2(K)*PDSL + PSGML1(K) with PDSL = PD(I,J), and
    EXNER = (PHMID*1.E-5)**CAPPA. Vectorised over the grid (one CuPy expression, no host loop).
    """
    import cupy as cp

    if "PHMID" in st.fields and "EXNER" in st.fields:
        return st.fields["PHMID"], st.fields["EXNER"]
    sgml2 = cp.asarray(np.ascontiguousarray(g.sgml2, dtype=np.float32))
    psgml1 = cp.asarray(np.ascontiguousarray(g.psgml1, dtype=np.float32))
    pdsl = st["PD"]
    phmid = st.add("PHMID", shape3)
    exner = st.add("EXNER", shape3)
    phmid[...] = sgml2[:, None, None] * pdsl[None, :, :] + psgml1[:, None, None]
    exner[...] = cp.power(phmid * np.float32(1.0e-5), CAPPA)
    return phmid, exner


def run(ctx):
    import cupy as cp

    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape3 = (lm, jm, im)

    phmid, exner = _driver_inputs(st, g, im, jm, lm, shape3)
    outs = [st.add(name, shape3) for name in LEVEL_OUT]
    scal = st.add("BMJ_SCAL", (NSC, jm, im))
    flag = st.add("BMJ_FLAG", (NFL, jm, im), "int32")

    dtcnvc = np.float32(NPRECIP * float(ctx.dt))
    k = kernels.get(UNIT, KERNEL)
    grid = ((im * jm + BLOCK - 1) // BLOCK, 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (BLOCK, 1, 1),
          (np.int32(im), np.int32(jm), np.int32(lm),
           st["T"], st["Q"], phmid, exner, st["PINT"],
           np.int32(0),
           np.float32(A2), np.float32(A3), np.float32(A4),
           np.float32(CP), np.float32(G), np.float32(ELWV),
           dtcnvc, np.float32(PQ0), np.float32(EPSQ),
           *outs, scal, flag))

    written = dict(zip(LEVEL_OUT, outs))
    written["BMJ_SCAL"] = scal
    written["BMJ_FLAG"] = flag
    return written
