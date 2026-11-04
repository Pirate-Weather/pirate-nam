// nam-gpu unit phys-bmj-col-prep: BMJ arguments, locals and column preparation (BMJ subroutine,
// lines 399-597), one thread per column.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran module has no licence block; its module header is kept here as the notice:
//   MODULE MODULE_CU_BMJ (Betts-Miller-Janjic convection, NMMB, nam.v4.2.11).
// Ported to CUDA for nam-gpu from phys/module_CU_BMJ.F90:399-597 (nam.v4.2.11, 83df4e13c1).
//
// Scope: the preparation statements of SUBROUTINE bmj that run before the buoyancy search
// (line 600 onward is unit phys-bmj-col-cloudbase / cloudtop). The locals that the later BMJ
// units read are column-persistent, so they are written to device scratch (CONTRACT section 1,
// st.add) instead of being kept in registers:
//   per level  [LM, JM, IM]  BMJ_TK, BMJ_QK, BMJ_RXNER, BMJ_CPECNV, BMJ_DTVCNV, BMJ_THES,
//                            BMJ_THESP, BMJ_THESCNV, BMJ_QSATK, BMJ_RHK, BMJ_DUDT, BMJ_DVDT
//   scalars    [NSC, JM, IM] BMJ_SCAL (slot list below)
//   flags      [NFL, JM, IM] int32 BMJ_FLAG (logical switches and ISWAP, 0 or 1)
// The BMJ call itself (bmjdrv, module_CU_BMJ.F90:117-395) passes the column arrays in; here the
// per-column inputs are the full-domain fields the BMJ driver prepares: T (TCOL), Q (QCOL, clamped
// to EPSQ), PHMID (PRSMID = PCOL), EXNER (EXNERCOL) and PINT (PSFC = PINT at the surface level).
//
// Fortran -> kernel map (module_CU_BMJ.F90 line numbers, verified with grep -n):
//   399-436 subroutine bmj header and dummy arguments -> kernel arguments (scalars by value,
//                                    arrays device pointers; A2..DTCNVC, PQ0, EPSQ as scalars)
//   441-539 declarations and parameters -> scratch arrays and slot constants below; no statement
//   541-544 cprlg, elocp, rcp, a23m4l -> BMJ_SCAL slots 0..3
//   546-548 uvscald, uvscals1, uvscals2 = 0.01 -> slots 4..6
//   550     rdtcnvc = 1/dtcnvc -> slot 7
//   551     depmin = psh*psfc*rsfcp, psfc = PINT[LM] at this column -> slot 8
//   553-555 deep, plume, shallow = .false. -> BMJ_FLAG slots 0..2 = 0
//   557-561 rh_deep = 0.90 if nodeep else 0.75 -> slot 9
//   565-567 mmntdeep = .false., mmntshal1/2 = .true. -> BMJ_FLAG slots 3..5 = 0, 1, 1
//   569-570 tauk, tauksc = dtcnvc/(trel*1.0), trel = 2400 -> slots 10, 11
//   574     iswap = 0 -> BMJ_FLAG slot 6
//   576-580 cup, dsp0, dspb, dspt, pcpcol = 0 -> slots 12..16 (pcpcol is intent(inout)).
//                                    Slot 16 is the binding seed of the PCPCOL chain; the RAINCV
//                                    step (module_CU_BMJ.F90:268) must read BMJ_SCAL[16] (see the
//                                    wrapper docstring). Not a private copy.
//   581     tref(1) = t(1) -> slot 17 (only level 1 of tref is set here)
//   583-596 loop l = 1..lmh (lmh = lm, bmjdrv line 303):
//     584-585 tk = t, qk = q        -> BMJ_TK, BMJ_QK (q is the bmjdrv qcol = max(epsq, q); the
//                                     clamp here is idempotent)
//     586     rxner = 1/exner       -> BMJ_RXNER
//     587-593 cpecnv, dtvcnv, thes, thesp, thescnv, dudt, dvdt = 0 -> BMJ_CPECNV, BMJ_DTVCNV,
//                                     BMJ_THES, BMJ_THESP, BMJ_THESCNV, BMJ_DUDT, BMJ_DVDT
//     594     qsatk = pq0/prsmid*exp(a2*(tk-a3)/(tk-a4)) -> BMJ_QSATK (same operation order)
//     595     rhk = qk/qsatk        -> BMJ_RHK
//
// Constants (single precision, kfpt): ROW = 1.e3; PSH = 20000; RSFCP = 1./101300.;
// TREL = 2400; ELIWV = 2.683e6; EPSQ = 1.e-12 (module_CONSTANTS, nmmb_common.cuh). EPSQ is the
// CUCNVC argument, passed from module_SOLVER_GRID_COMP.F90:5775 (verified, not the 1.e-20 of
// module_DYNAMICS_ROUTINES).
// Layout (CONTRACT sections 1 and 3): 3-D A[l0, j0, i0] via IDX3; 2-D [j0, i0] via IDX2.
// Numerics: float32 throughout, expf for exp, FMA contraction on (fast numerics).
// No host sync, no printf, no device malloc.

#include "nmmb_common.cuh"

#define BMJP_ROW      1.0e3f
#define BMJP_ELIWV    2.683e6f
#define BMJP_PSH      20000.0f
#define BMJP_RSFCP    (1.0f / 101300.0f)
#define BMJP_TREL     2400.0f
#define BMJP_RH_DEEP_NODEEP  0.90f
#define BMJP_RH_DEEP         0.75f
#define BMJP_UVSCALE  0.01f

// Per-column scalar slots of BMJ_SCAL (NSC = 18) and flag slots of BMJ_FLAG (NFL = 7).
#define BMJ_NSC 18
#define BMJ_NFL 7
#define BMJS_CPRLG 0
#define BMJS_ELOCP 1
#define BMJS_RCP 2
#define BMJS_A23M4L 3
#define BMJS_UVSCALD 4
#define BMJS_UVSCALS1 5
#define BMJS_UVSCALS2 6
#define BMJS_RDTCNVC 7
#define BMJS_DEPMIN 8
#define BMJS_RH_DEEP 9
#define BMJS_TAUK 10
#define BMJS_TAUKSC 11
#define BMJS_CUP 12
#define BMJS_DSP0 13
#define BMJS_DSPB 14
#define BMJS_DSPT 15
#define BMJS_PCPCOL 16
#define BMJS_TREF1 17
#define BMJF_DEEP 0
#define BMJF_PLUME 1
#define BMJF_SHALLOW 2
#define BMJF_MMNTDEEP 3
#define BMJF_MMNTSHAL1 4
#define BMJF_MMNTSHAL2 5
#define BMJF_ISWAP 6

extern "C" __global__ void phys_bmj_col_prep(
    const int IM, const int JM, const int LM,
    const float* __restrict__ T,        // [LM, JM, IM] TCOL
    const float* __restrict__ Q,        // [LM, JM, IM] Q (clamped to EPSQ here)
    const float* __restrict__ PHMID,    // [LM, JM, IM] PRSMID (PCOL)
    const float* __restrict__ EXNER,    // [LM, JM, IM] EXNERCOL
    const float* __restrict__ PINT,     // [LM+1, JM, IM] interface pressure; PSFC = PINT[LM]
    const int   NODEEP,
    const float A2, const float A3, const float A4,
    const float CP, const float G, const float ELWV,
    const float DTCNVC, const float PQ0, const float EPSQ,
    float* __restrict__ BMJ_TK,
    float* __restrict__ BMJ_QK,
    float* __restrict__ BMJ_RXNER,
    float* __restrict__ BMJ_CPECNV,
    float* __restrict__ BMJ_DTVCNV,
    float* __restrict__ BMJ_THES,
    float* __restrict__ BMJ_THESP,
    float* __restrict__ BMJ_THESCNV,
    float* __restrict__ BMJ_QSATK,
    float* __restrict__ BMJ_RHK,
    float* __restrict__ BMJ_DUDT,
    float* __restrict__ BMJ_DVDT,
    float* __restrict__ BMJ_DTDT,      // zeroed per column (Fortran bmjdrv 272-277)
    float* __restrict__ BMJ_DQDT,
    float* __restrict__ BMJ_SCAL,       // [BMJ_NSC, JM, IM]
    int*   __restrict__ BMJ_FLAG)       // [BMJ_NFL, JM, IM]
{
    COLUMN_IJ(IM, JM)
    const size_t p2 = IDX2(i0, j0, IM);
    const size_t sc = (size_t)JM * (size_t)IM;     // stride of one scalar / flag slot

    // ---- column scalars (Fortran 494-534) ----
    const float cprlg  = CP / (BMJP_ROW * G * ELWV);
    const float elocp  = BMJP_ELIWV / CP;
    const float rcp    = 1.0f / CP;
    const float a23m4l = A2 * (A3 - A4) * ELWV;
    const float rdtcnvc = 1.0f / DTCNVC;
    const float psfc   = PINT[IDX3(i0, j0, LM, IM, JM)];          // phint(i,j,lm+1)
    const float depmin = BMJP_PSH * psfc * BMJP_RSFCP;
    const float rh_deep = NODEEP ? BMJP_RH_DEEP_NODEEP : BMJP_RH_DEEP;
    const float tauk   = DTCNVC / (BMJP_TREL * 1.0f);
    const float tauksc = DTCNVC / (BMJP_TREL * 1.0f);

    BMJ_SCAL[BMJS_CPRLG * sc + p2]   = cprlg;
    BMJ_SCAL[BMJS_ELOCP * sc + p2]   = elocp;
    BMJ_SCAL[BMJS_RCP * sc + p2]     = rcp;
    BMJ_SCAL[BMJS_A23M4L * sc + p2]  = a23m4l;
    BMJ_SCAL[BMJS_UVSCALD * sc + p2]  = BMJP_UVSCALE;
    BMJ_SCAL[BMJS_UVSCALS1 * sc + p2] = BMJP_UVSCALE;
    BMJ_SCAL[BMJS_UVSCALS2 * sc + p2] = BMJP_UVSCALE;
    BMJ_SCAL[BMJS_RDTCNVC * sc + p2] = rdtcnvc;
    BMJ_SCAL[BMJS_DEPMIN * sc + p2]  = depmin;
    BMJ_SCAL[BMJS_RH_DEEP * sc + p2] = rh_deep;
    BMJ_SCAL[BMJS_TAUK * sc + p2]    = tauk;
    BMJ_SCAL[BMJS_TAUKSC * sc + p2]  = tauksc;
    BMJ_SCAL[BMJS_CUP * sc + p2]     = 0.0f;
    BMJ_SCAL[BMJS_DSP0 * sc + p2]    = 0.0f;
    BMJ_SCAL[BMJS_DSPB * sc + p2]    = 0.0f;
    BMJ_SCAL[BMJS_DSPT * sc + p2]    = 0.0f;
    BMJ_SCAL[BMJS_PCPCOL * sc + p2]  = 0.0f;
    BMJ_SCAL[BMJS_TREF1 * sc + p2]   = T[IDX3(i0, j0, 0, IM, JM)];   // tref(1) = t(1)

    // ---- logical switches and ISWAP (Fortran 507-527) ----
    BMJ_FLAG[BMJF_DEEP * sc + p2]      = 0;   // deep = .false.
    BMJ_FLAG[BMJF_PLUME * sc + p2]     = 0;   // plume = .false.
    BMJ_FLAG[BMJF_SHALLOW * sc + p2]   = 0;   // shallow = .false.
    BMJ_FLAG[BMJF_MMNTDEEP * sc + p2]  = 0;   // mmntdeep = .false.
    BMJ_FLAG[BMJF_MMNTSHAL1 * sc + p2] = 1;   // mmntshal1 = .true.
    BMJ_FLAG[BMJF_MMNTSHAL2 * sc + p2] = 1;   // mmntshal2 = .true.
    BMJ_FLAG[BMJF_ISWAP * sc + p2]     = 0;   // iswap = 0

    // ---- per-level preparation, l = 1..lmh with lmh = lm (Fortran 536-550) ----
    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t p3 = IDX3(i0, j0, l0, IM, JM);
        const float tk = T[p3];
        const float qk = fmaxf(EPSQ, Q[p3]);           // bmjdrv qcol = max(epsq, q)
        const float rxner = 1.0f / EXNER[p3];
        const float prs = PHMID[p3];
        const float qsatk = PQ0 / prs * expf(A2 * (tk - A3) / (tk - A4));
        const float rhk = qk / qsatk;

        BMJ_TK[p3] = tk;
        BMJ_QK[p3] = qk;
        BMJ_RXNER[p3] = rxner;
        BMJ_CPECNV[p3] = 0.0f;
        BMJ_DTVCNV[p3] = 0.0f;
        BMJ_THES[p3] = 0.0f;
        BMJ_THESP[p3] = 0.0f;
        BMJ_THESCNV[p3] = 0.0f;
        BMJ_DUDT[p3] = 0.0f;
        BMJ_DVDT[p3] = 0.0f;
        BMJ_DTDT[p3] = 0.0f;
        BMJ_DQDT[p3] = 0.0f;
        BMJ_QSATK[p3] = qsatk;
        BMJ_RHK[p3] = rhk;
    }
}
