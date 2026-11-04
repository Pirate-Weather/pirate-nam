// phys_fer_condense.cu -- WOOF-NAM unit phys-fer-condense: CONDENSE, cloud water condensation
// (Asai 1965 iteration) used by the Ferrier-Aligo microphysics.
//
// NOTICE: NAM source file phys/module_MP_FER_HIRES.F90 (NOAA/NCEP/EMC, US Government work, no licence
// file in the NAM clone). Notice is the module changelog block (lines 1-82, "Modified fer_hires
// microphysics - 11 July 2016 version") and the module header, as in fer_hires.cuh.
//
// Ported to CUDA for WOOF-NAM from phys/module_MP_FER_HIRES.F90:2160-2225 (nam.v4.2.11, 83df4e13c1).
//
// Statement map (every statement of CONDENSE has a counterpart below, or a reason):
//   2160          FUNCTION CONDENSE(PP,QW,TK,WV,RHgrd,I,J,L) -> nam_condense (device), same argument order
//   2164-2165     HIGH_PRES, RHLIMIT=.001 (default-real literal stored in a HIGH_PRES parameter), RHLIMIT1=-RHLIMIT
//                 -> FER_RHLIMIT = (double)0.001f
//   2168-2175     declarations: COND, SSAT, WCdum HIGH_PRES (double); the rest single. Precision of each
//                 operation is kept (see Precision below).
//   2181-2184     NITER (loop variable), DWVinp (kept: returned by the non-convergence branch), SSATinp (debug
//                 only, dropped with the WARN3 print), I/J/L (used only by the WARN3 print, dropped)
//   2192-2194     Tdum=TK, WVdum=WV, WCdum=QW
//   2195          ESW=MIN(1000.*FPVS0(Tdum),0.99*PP)       -> nam_fpvs0 from phys-fer-fpvs
//   2196          WS=RHgrd*EPS*ESW/(PP-ESW)
//   2197-2198     DWV=WVdum-WS; SSAT=DWV/WS
//   2199          CONDENSE=0.
//   2204-2215     DO NITER loop: COND, COND=MAX(COND,-WCdum), Tdum/WVdum/WCdum/CONDENSE updates, ESW/WS/DWV/SSAT
//                 recomputed, and the two EXIT tests (SSAT in [RHLIMIT1,RHLIMIT]; SSAT<RHLIMIT1 with WCdum<=EPSQ)
//   2218-2223     IF (NITER>MAX_ITERATIONS): the WARN3 write(0,*) is host-only and is dropped (no printf in kernels);
//                 CONDENSE=DWVinp is kept; SSAT=0. is a dead store (SSAT is local, not returned).
//   2225          END FUNCTION -> return CONDENSE
//
// Loop exit semantics: the Fortran DO loop leaves NITER = MAX_ITERATIONS+1 = 11 only when no EXIT was taken,
// so "not converged after FER_MAX_ITERATIONS" here is exactly the Fortran NITER > MAX_ITERATIONS branch.
//
// Precision (as declared in the Fortran): COND, SSAT, WCdum and RHLIMIT are HIGH_PRES (double).
// Each mixed expression is evaluated in double where Fortran promotes it, and rounded to single where
// Fortran assigns to a single variable (Tdum, WVdum, CONDENSE). Single-only expressions stay float.
//
// Calling convention: the standalone evaluation fer_condense_eval (one thread per cell, grid_2d) takes the
// production CONDENSE inputs (PP, QW, TK, WV, PIDEP, PIACWI, QSWGRD) and binds DUM1/DUM2/DUM3 and the
// TC guard itself, as module_MP_FER_HIRES.F90:1456-1461 does. Outputs: PCOND and a non-convergence flag.
#include "nmmb_common.cuh"
#include "fer_hires.cuh"
#include "phys_fer_fpvs.cu"   // nam_fpvs0 (unit phys-fer-fpvs); its eval kernel is unused here

// RHLIMIT = .001 in the Fortran source is a default-real literal in a HIGH_PRES parameter.
#define FER_RHLIMIT ((double)0.001f)

// CONDENSE(PP,QW,TK,WV,RHgrd). *converged is false on the Fortran NITER > MAX_ITERATIONS branch.
__device__ inline float nam_condense(float PP, float QW, float TK, float WV, float RHgrd,
                                     const float* __restrict__ tbpvs0,
                                     float c1xpvs0, float c2xpvs0, bool* converged)
{
    const double RHLIMIT  = FER_RHLIMIT;
    const double RHLIMIT1 = -RHLIMIT;
    const double EPSQ_d   = (double)FER_EPSQ;   // EPSQ is single in the module; compared as double
    const float  EPS      = FER_EP2;            // EPS = EP_2
    const float  XLV1     = FER_XLV1;
    const float  XLV3     = FER_XLV3;

    float  Tdum  = TK;
    float  WVdum = WV;
    double WCdum = (double)QW;

    float ESW = fminf(1000.0f * nam_fpvs0(Tdum, tbpvs0, c1xpvs0, c2xpvs0), 0.99f * PP);
    float WS  = RHgrd * EPS * ESW / (PP - ESW);
    float DWV = WVdum - WS;
    double SSAT = (double)(DWV / WS);
    float CONDENSE = 0.0f;
    const float DWVinp = DWV;                  // debug copy of the initial deficit, used by the non-convergence branch

    bool done = false;
    for (int niter = 1; niter <= FER_MAX_ITERATIONS; ++niter) {
        // COND=DWV/(1.+XLV3*WS/(Tdum*Tdum))  (all single in Fortran, then stored to HIGH_PRES COND)
        double COND = (double)(DWV / (1.0f + XLV3 * WS / (Tdum * Tdum)));
        COND = fmax(COND, -WCdum);                              // limit cloud water evaporation
        Tdum  = (float)((double)Tdum + (double)XLV1 * COND);    // Tdum=Tdum+XLV1*COND
        WVdum = (float)((double)WVdum - COND);                  // WVdum=WVdum-COND
        WCdum = WCdum + COND;                                   // WCdum=WCdum+COND (double)
        CONDENSE = (float)((double)CONDENSE + COND);            // CONDENSE=CONDENSE+COND
        ESW = fminf(1000.0f * nam_fpvs0(Tdum, tbpvs0, c1xpvs0, c2xpvs0), 0.99f * PP);
        WS  = RHgrd * EPS * ESW / (PP - ESW);
        DWV = WVdum - WS;
        SSAT = (double)(DWV / WS);
        if (SSAT >= RHLIMIT1 && SSAT <= RHLIMIT) { done = true; break; }   // SSAT near zero
        if (SSAT < RHLIMIT1 && WCdum <= EPSQ_d) { done = true; break; }   // SSAT<0 and no cloud water
    }

    *converged = done;
    if (!done) {
        // Fortran NITER > MAX_ITERATIONS: too many iterations (possible numerical instability).
        // The WARN3 host message is dropped; it is reported through the per-cell flag instead.
        CONDENSE = DWVinp;
    }
    return CONDENSE;
}

// Production binding (module_MP_FER_HIRES.F90:1456-1461), done here so the unit takes the same inputs the
// Fortran does, not raw T/Q/CW:
//   TC = TK - T0C (the column kernel's tc, line 915);  guard TC>=T_ICE .AND. (QW>EPSQ .OR. WV>QSWgrd)
//   DUM1 = QW - PIACWI;  DUM2 = TK + XLS1*PIDEP + XLF1*PIACWI;  DUM3 = WV - PIDEP
//   PCOND = CONDENSE(PP, DUM1, DUM2, DUM3, RHgrd, ...). PCOND is 0 when the guard is false (the column
//   kernel initialises pcond to 0 on each cell, as the Fortran does for the level).
// coef = GPVS_hr coefficients {C1XPVS, C2XPVS, C1XPVS0, C2XPVS0} from phys-fer-gpvs (single source of truth).
// flag[p] = 1 where CONDENSE did not converge (Fortran WARN3 condition), else 0.
// Evaluation kernel: one thread per (i, j, l) cell; all [LM, JM, IM] float32 arrays use CONTRACT layout.
extern "C" __global__ void fer_condense_eval(const float* __restrict__ PP,
                                             const float* __restrict__ QW,
                                             const float* __restrict__ TK,
                                             const float* __restrict__ WV,
                                             const float* __restrict__ PIDEP,
                                             const float* __restrict__ PIACWI,
                                             const float* __restrict__ QSWGRD,
                                             float RHgrd,
                                             const float* __restrict__ tbpvs0,
                                             const float* __restrict__ coef,
                                             float* __restrict__ out,
                                             unsigned char* __restrict__ flag,
                                             int IM, int JM, int LM)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t p = IDX3(i0, j0, l0, IM, JM);

    const float tk     = TK[p];
    const float wv     = WV[p];
    const float qw     = QW[p];
    const float pidep  = PIDEP[p];
    const float piacwi = PIACWI[p];
    const float qswgrd = QSWGRD[p];

    out[p]  = 0.0f;
    flag[p] = 0;
    const float tc = tk - FER_T0C;
    if (tc >= FER_T_ICE && (qw > FER_EPSQ || wv > qswgrd)) {
        const float dum1 = qw - piacwi;
        const float dum2 = tk + FER_XLS1 * pidep + FER_XLF1 * piacwi;
        const float dum3 = wv - pidep;
        bool converged = true;
        out[p]  = nam_condense(PP[p], dum1, dum2, dum3, RHgrd, tbpvs0, coef[2], coef[3], &converged);
        flag[p] = converged ? 0 : 1;
    }
}
