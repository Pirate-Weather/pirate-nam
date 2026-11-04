// nam-gpu unit phys-bmj-bmjdrv: BMJDRV per-column setup, BMJ call, tendencies and precipitation out.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module header kept as the notice: MODULE MODULE_CU_BMJ (NMMB convection packages, nam.v4.2.11).
// Ported to CUDA for nam-gpu from phys/module_CU_BMJ.F90:117-395 (nam.v4.2.11, 83df4e13c1).
//
// Scope of this file: the bmjdrv subroutine body only. The routine calls BMJ (module_CU_BMJ.F90:399-2067),
// which is the work of units phys-bmj-col-prep ... phys-bmj-col-shallow-adjust; that call is made here
// through bmj_column(), a __device__ function those units define in their own kernel files. This file
// does not define it and does not compile on its own until those units land.
//
// Fortran -> kernel map (bmjdrv line numbers, module_CU_BMJ.F90, nam.v4.2.11 83df4e13c1):
//   219       ICLDCK = mod(NTSD, NCNVC); guard (icldck == 0 or ntsd == 0)            -> icldck guard
//   225-227   IMD, JMD (used only by the commented-out selection at 312-313); PRINT_DIAG = .false.
//                                                                                    -> print_diag
//   230-236   CU_ACT_FLAG = .true. over its..ite, jts..jte (full domain)            -> CU_ACT_FLAG
//   238       DTCNVC = DT*NCNVC                                                      -> dtcnvc
//   239       RDXEQ = 1/dxh((jds+jde)/2+1); its only use (248) is commented out      -> rdxeq (computed, unused)
//   248       FRESP = FRES (the RDXEQ scaling is commented out)                      -> fresp
//   251-263   DSP base values scaled by FRESP, FR, FSL, FSS                          -> dsp* / *s*
//   265-270   SLOP* (the slopes)                                                     -> slop*
//   272-277   DQDT, DTDT, DUDT, DVDT zeroed for k=1..lm                              -> local arrays
//   279-282   RAINCV = 0, PCPCOL = 0, PSFC = PHINT(lm+1), PTOP = PHINT(1)            -> psfc, ptop
//   286       SEAMASK = XLAND - 1                                                    -> seamask
//   290-298   UCOL, VCOL, QCOL=MAX(EPSQ,Q), TCOL, PCOL=PHMID, EXNERCOL, DPCOL=PHINT(k+1)-PHINT(k)
//   303-304   LMH = LM, LPBL = KPBL                                                  -> lmh, lpbl
//   315-326   CALL BMJ                                                               -> bmj_column()
//   331-334   DUDT_PHY, DVDT_PHY = DUDT, DVDT                                        -> out arrays
//   339-342   RTHCUTEN = DTDT/EXNER, RQCUTEN = DQDT                                  -> out arrays
//   348       RAINCV = PCPCOL*1.E3/NCNVC                                             -> raincv
//   352-353   CUTOP = LM+1-LTOP, CUBOT = LM+1-LBOT                                   -> cutop, cubot
//   357-383   print_diag debug block (writes to unit 6)                              -> ported below (printf)
//   Dummy arguments DYH and TH are never read in bmjdrv; they are bound for parity with the CUCNVC call.
//   OpenMP directives (zj$omp) dropped: one thread per column in CUDA.
//
// Index ranges: single process, regional, full domain 1..IM x 1..JM. its_b1 = max(its, ids+1) = 2 and
// ite_b1 = min(ite, ide-1) = IM-1 (1-based; module_DM_PARALLEL.F90:799-804), so the b1 loop is 0-based
// i0 in [1, IM-2], j0 in [1, JM-2]. Columns outside that range get the flag and the prep values only, as
// in Fortran.
//
// Layout (CONTRACT section 1): 3-D A[l0, j0, i0] via IDX3; 2-D [j0, i0] via IDX2; PHINT [LM+1, JM, IM];
// the Fortran level k maps to k0 = k-1; the Fortran interface index lm+1 maps to PHINT level LM (0-based).
// LM is fixed at BMJ_LM = 60 (NAM 12 km, CONTRACT section 5); the wrapper refuses any other LM so the
// per-column local arrays are sized at compile time.
//
// Numerics: float32 (kfpt = single). Literals follow the Fortran single-precision constants. The
// parameter values are copied from the module parameter block. FMA on (CONTRACT section 3).
// No host sync, no allocation. Column arrays live in thread-local memory.

#include "nmmb_common.cuh"

#define BMJ_LM 60

// module_CU_BMJ.F90 module parameters used by bmjdrv (efimn and the dsp*_base values).
#define BMJ_EFIMN       0.20f
#define BMJ_DSPBFL_BASE -3875.0f
#define BMJ_DSP0FL_BASE -5875.0f
#define BMJ_DSPTFL_BASE -1875.0f
#define BMJ_DSPBFS_BASE -3875.0f
#define BMJ_DSP0FS_BASE -5875.0f
#define BMJ_DSPTFS_BASE -1875.0f

// BMJ call (module_CU_BMJ.F90:399-2067, subroutine bmj): bmj_column() in phys_bmj_column.cuh, one
// device function transliterated from the Fortran (2026-11-04), with the BMJ_INIT tables in BmjTbl.
#include "phys_bmj_column.cuh"

extern "C" __global__ void phys_bmj_bmjdrv_column(
    const int IM, const int JM, const int LM,
    const int ntsd, const int ncnvc,
    const float dt, const float dyh,
    const float fres, const float fr, const float fsl, const float fss,
    const int entrain, const int newall, const int newswap, const int newupup, const int nodeep,
    const float a2, const float a3, const float a4, const float cappa, const float cp,
    const float eliv, const float elwv, const float epsq, const float g, const float p608,
    const float pq0, const float r_d, const float tiw,
    const float* __restrict__ PHINT,      // [LM+1, JM, IM]
    const float* __restrict__ PHMID,      // [LM, JM, IM]
    const float* __restrict__ EXNER,      // [LM, JM, IM]
    const float* __restrict__ Q,          // [LM, JM, IM]
    const float* __restrict__ T,          // [LM, JM, IM]
    const float* __restrict__ U_PHY,      // [LM, JM, IM]
    const float* __restrict__ V_PHY,      // [LM, JM, IM]
    const float* __restrict__ TH,         // [LM, JM, IM] dummy argument, never read in bmjdrv
    const float* __restrict__ XLAND,      // [JM, IM]
    const float* __restrict__ DXH,        // [JM] grid row spacing
    const int*   __restrict__ KPBL,       // [JM, IM]
    float* __restrict__ CLDEFI,           // [JM, IM] inout
    int*   __restrict__ CU_ACT_FLAG,      // [JM, IM] out (1 = true)
    float* __restrict__ RAINCV,           // [JM, IM] out
    float* __restrict__ CUTOP,            // [JM, IM] out
    float* __restrict__ CUBOT,            // [JM, IM] out
    float* __restrict__ DUDT_PHY,         // [LM, JM, IM] out
    float* __restrict__ DVDT_PHY,         // [LM, JM, IM] out
    float* __restrict__ RTHCUTEN,         // [LM, JM, IM] out
    float* __restrict__ RQCUTEN,          // [LM, JM, IM] out
    // BMJ_INIT tables (phys-bmj-init, ctx.tables["bmj_*"])
    const float* __restrict__ tb_ptbl, const float* __restrict__ tb_qs0, const float* __restrict__ tb_sqs,
    const float* __restrict__ tb_sthe, const float* __restrict__ tb_the0, const float* __restrict__ tb_ttbl,
    const float* __restrict__ tb_stheq, const float* __restrict__ tb_the0q, const float* __restrict__ tb_ttblq)
{
    BmjTbl tb;
    tb.ptbl = tb_ptbl; tb.qs0 = tb_qs0; tb.sqs = tb_sqs; tb.sthe = tb_sthe; tb.the0 = tb_the0;
    tb.ttbl = tb_ttbl; tb.stheq = tb_stheq; tb.the0q = tb_the0q; tb.ttblq = tb_ttblq;
    COLUMN_IJ(IM, JM)
    const size_t p2 = IDX2(i0, j0, IM);

    // Fortran 219: the convection-step test. The early return in CUCNVC already limits calls to
    // NCNVC multiples, so this guard only matters if the wrapper is called on another step.
    const int icldck = ntsd % ncnvc;
    if (!(icldck == 0 || ntsd == 0)) return;

    // Fortran 230-236: cu_act_flag over the full tile (its..ite, jts..jte).
    CU_ACT_FLAG[p2] = 1;

    // Fortran 239: RDXEQ = 1/dxh((jds+jde)/2+1). Single process: jds=1, jde=JM, so the 0-based index
    // is (1+JM)/2. Its only use (Fortran 248) is commented out, so the value has no effect on output.
    const float rdxeq = 1.0f / DXH[(1 + JM) / 2];
    (void)rdxeq;
    // Dummy argument DYH is declared in bmjdrv but never read; bound for signature parity.
    (void)dyh;

    // The b1 loop (Fortran 247-249) covers only the interior. Outside it the outputs keep the values
    // the prep kernel wrote (raincv 0, cutop/cubot 999, tendencies 0).
    const bool interior = (i0 >= 1 && i0 <= IM - 2 && j0 >= 1 && j0 <= JM - 2);
    if (!interior) return;

    // Fortran 238: DTCNVC = DT*NCNVC.
    const float dtcnvc = dt * (float)ncnvc;

    // Fortran 248: FRESP = FRES (the RDXEQ scaling is commented out in the source).
    const float fresp = fres;

    // Fortran 251-270: deficit saturation pressures by resolution and land/sea factors.
    const float dspbfl = BMJ_DSPBFL_BASE * fresp * fr;
    const float dsp0fl = BMJ_DSP0FL_BASE * fresp * fr;
    const float dsptfl = BMJ_DSPTFL_BASE * fresp * fr;
    const float dspbfs = BMJ_DSPBFS_BASE * fresp;
    const float dsp0fs = BMJ_DSP0FS_BASE * fresp;
    const float dsptfs = BMJ_DSPTFS_BASE * fresp;

    const float dspbsl = dspbfl * fsl;
    const float dsp0sl = dsp0fl * fsl;
    const float dsptsl = dsptfl * fsl;
    const float dspbss = dspbfs * fss;
    const float dsp0ss = dsp0fs * fss;
    const float dsptss = dsptfs * fss;

    const float slopbl = (dspbfl - dspbsl) / (1.0f - BMJ_EFIMN);
    const float slop0l = (dsp0fl - dsp0sl) / (1.0f - BMJ_EFIMN);
    const float sloptl = (dsptfl - dsptsl) / (1.0f - BMJ_EFIMN);
    const float slopbs = (dspbfs - dspbss) / (1.0f - BMJ_EFIMN);
    const float slop0s = (dsp0fs - dsp0ss) / (1.0f - BMJ_EFIMN);
    const float slopts = (dsptfs - dsptss) / (1.0f - BMJ_EFIMN);

    // Fortran 272-277: tendencies zeroed; 279-282 raincv, pcpcol, psfc, ptop.
    float dqdt[BMJ_LM], dtdt[BMJ_LM], dudt[BMJ_LM], dvdt[BMJ_LM];
    float dpcol[BMJ_LM], pcol[BMJ_LM], qcol[BMJ_LM], tcol[BMJ_LM];
    float exnercol[BMJ_LM], ucol[BMJ_LM], vcol[BMJ_LM];
    for (int k0 = 0; k0 < BMJ_LM; ++k0) {
        dqdt[k0] = 0.0f;
        dtdt[k0] = 0.0f;
        dudt[k0] = 0.0f;
        dvdt[k0] = 0.0f;
    }
    float pcpcol = 0.0f;
    RAINCV[p2] = 0.0f;
    const float psfc = PHINT[IDX3(i0, j0, LM, IM, JM)];   // PHINT(i,j,lm+1)
    const float ptop = PHINT[IDX3(i0, j0, 0, IM, JM)];    // PHINT(i,j,1)

    // Fortran 286: bmj land mask, 1.0 for sea, 0.0 for land.
    const float seamask = XLAND[p2] - 1.0f;

    // Fortran 290-298: 1-D column arrays. QCOL uses EPSQ as the floor (Fortran MAX(EPSQ, Q)).
    for (int k0 = 0; k0 < LM; ++k0) {
        const size_t pk = IDX3(i0, j0, k0, IM, JM);
        ucol[k0] = U_PHY[pk];
        vcol[k0] = V_PHY[pk];
        qcol[k0] = fmaxf(epsq, Q[pk]);
        tcol[k0] = T[pk];
        pcol[k0] = PHMID[pk];
        exnercol[k0] = EXNER[pk];
        dpcol[k0] = PHINT[IDX3(i0, j0, k0 + 1, IM, JM)] - PHINT[pk];
    }

    // Fortran 303-304.
    const int lmh = LM;
    const int lpbl = KPBL[p2];

    // Fortran 315-326: the BMJ scheme for this column. print_diag is .false. (Fortran 227; the
    // selection at 312-313 is commented out), so it is passed as 0.
    const bool print_diag = false;
    int lbot = 0;
    int ltop = 0;
    float pcpcol_out = pcpcol;
    bmj_column(entrain, newall, newswap, newupup, nodeep, print_diag ? 1 : 0,
               i0 + 1, j0 + 1, LM, lmh, lpbl, ntsd,
               &lbot, &ltop,
               a2, a3, a4, cappa, &CLDEFI[p2], cp, dtcnvc, eliv, elwv,
               g, p608, pq0, r_d, tiw,
               psfc, ptop, seamask,
               dspbfl, dsp0fl, dsptfl, dspbfs, dsp0fs, dsptfs,
               dspbsl, dsp0sl, dsptsl, dspbss, dsp0ss, dsptss,
               slopbl, slop0l, sloptl, slopbs, slop0s, slopts,
               dpcol, pcol, qcol, tcol, exnercol, ucol, vcol,
               dqdt, dtdt, dudt, dvdt, &pcpcol_out, tb);

    // Fortran 331-334: momentum tendencies.
    for (int k0 = 0; k0 < LM; ++k0) {
        const size_t pk = IDX3(i0, j0, k0, IM, JM);
        DUDT_PHY[pk] = dudt[k0];
        DVDT_PHY[pk] = dvdt[k0];
    }

    // Fortran 339-342: heating (dtdt / exner, the RTHCUTEN is theta tendency) and moistening tendencies.
    for (int k0 = 0; k0 < LM; ++k0) {
        const size_t pk = IDX3(i0, j0, k0, IM, JM);
        RTHCUTEN[pk] = dtdt[k0] / EXNER[pk];
        RQCUTEN[pk] = dqdt[k0];
    }

    // Fortran 348: precipitation from meters to mm per step: PCPCOL*1.E3/NCNVC.
    RAINCV[p2] = pcpcol_out * 1.e3f / (float)ncnvc;

    // Fortran 352-353: convective cloud top and bottom as level indices counted from the top.
    CUTOP[p2] = (float)(LM + 1 - ltop);
    CUBOT[p2] = (float)(LM + 1 - lbot);

    // Fortran 357-383: print_diag debug block (writes to unit 6). Fortran 225-226 sets IMD and JMD and
    // 227 sets PRINT_DIAG = .false.; the selection that would set it true (312-313) is commented out, so
    // this block is unreachable, exactly as in the source. Ported to device printf.
    const int imd = (1 + IM) / 2;   // Fortran (ims+ime)/2, single process
    const int jmd = (1 + JM) / 2;   // Fortran (jms+jme)/2, single process
    (void)imd;
    (void)jmd;
    if (print_diag) {
        float delt = 0.0f;
        float delq = 0.0f;
        float plyr = 0.0f;
        if (lbot > 0 && ltop < lbot) {
            for (int k = ltop; k <= lbot; ++k) {
                plyr = plyr + dpcol[k - 1];
                delq = delq + dpcol[k - 1] * dtcnvc * fabsf(dqdt[k - 1]);
                delt = delt + dpcol[k - 1] * dtcnvc * fabsf(dtdt[k - 1]);
            }
            delq = delq / plyr;
            delt = delt / plyr;
        }
        printf("{cu3 i,j,pcpcol,dtavg,dqavg,plyr,lbot,ltop,cubot,cutop = %4d%4d%12.4e%12.4e%12.4e%7.2f%3d%3d%3d%3d\n",
               i0 + 1, j0 + 1, pcpcol_out, delt, 1000.0f * delq, 0.01f * plyr,
               lbot, ltop, (int)CUBOT[p2], (int)CUTOP[p2]);
        if (plyr > 0.0f) {
            for (int k = lbot; k >= ltop; --k) {
                printf("{cu3a k,dt,dq,dp = %3d%12.4e%12.4e%7.2f\n",
                       k, dtcnvc * dtdt[k - 1], 1000.0f * dtcnvc * dqdt[k - 1], 0.01f * dpcol[k - 1]);
            }
        }
    }
}
