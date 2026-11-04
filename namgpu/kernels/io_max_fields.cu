// WOOF-NAM unit io-max-fields: MAX_FIELDS_HR, the hourly maxima, minima and running averages of
// the NMMB diagnostics: REFDMAX (1 km AGL simulated reflectivity), REFDMAX263K (-10 C level),
// PRATEMAX, FPRATEMAX, UPVVELMAX, DNVVELMAX, TLMAX/TLMIN, T02MAX/T02MIN, RH02MAX/RH02MIN,
// U10MAX, V10MAX, SPD10MAX, and the averages T10, T10AVG, PSFCAVG, AKHSAVG, AKMSAVG, SNOAVG.
// The updraft helicity maxima (UPHLMAX25/03 and minima) are CALC_UPHLCY, unit io-uphlcy, which
// the wrapper runs before this kernel (the two CALL sites of MAX_FIELDS_HR).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). The routine
// MAX_FIELDS_HR in nmm/module_DIAGNOSE.F90 has no author block of its own; the module and its
// constants are NCEP's. Ported to CUDA for WOOF-NAM from nmm/module_DIAGNOSE.F90:2190-2518
// (nam.v4.2.11, 83df4e13c1). Constants from nmm/module_CONSTANTS.F90 (see nmmb_common.cuh).
//
// Mapping: one thread per (i, j) column (kernels.grid_cols). Each thread runs the Fortran column
// loops (PMID, UPINDX, the 1 km and -10 C searches, the vertical loops) on its own column, so no
// array scratch is needed and LM is not limited. The Fortran loops over ITS_B1:ITE_B1 and
// JTS_B1:JTE_B1 (1-based 2..IM-1, 2..JM-1) are the thread guard b1 below (0-based i0 in [1, IM-2]).
// Single-process, one card: ITS = JTS = 1, ITE = IM, JTE = JM.
//
// Fortran-to-CUDA notes.
//  - W dummy: MAX_FIELDS_HR receives int_state%W_TOT as its W argument (module_SOLVER_GRID_COMP.F90:5049).
//    The wrapper binds st["W_TOT"] to this W, so UPVVELMAX/DNVVELMAX read W_TOT.
//  - PMID(I,J,L) = PSGML1(L) + SGML2(L)*PD(I,J) is inlined where used (pure function of the column).
//  - ZINTSFC = FIS/g, g = NMMB_G (module_CONSTANTS g).
//  - UPINDX, DECLARED NOT WITNESSED: no level of the column crosses 10000 Pa in the Fortran L_LOOP
//    (module_DIAGNOSE.F90 ~2189-2212), so the Fortran value is undefined. This kernel takes
//    UPINDX = LM+1 (no level counted in UPVVELMAX/DNVVELMAX). proof-bar rule 1 needs a
//    witness and his sign-off before this counts as ported.
//  - REFDMAX 1 km vloop, DECLARED NOT WITNESSED, STALE PATH NOT REPRODUCED: with no 1000 m AGL
//    crossing in the column (module_DIAGNOSE.F90 ~2236-2255) the Fortran reads ZMIDP1, ZMIDloc and
//    DBZ1 left over from the previous column. Reproducing that needs the previous column's values
//    in a sequential column order, which the one-thread-per-column launch does not have. This
//    kernel skips the update for such a column. Same ruling as UPINDX applies.
//  - REFDMAX263K: the search with DBZ1 = DBZmin defaults; no undefined read.
//  - T, Q, CW, F_RAIN, F_ICE, F_RIMEF, HTOP, FIRST_NMM and MY_DOMAIN_ID are not used by the routine
//    body (Q1D(:) is a dead store of Q, T1D/P1D are dead stores; not ported).
//  - CUREFL: CU_A*(RDTPHS*CPRATE)**CU_B only if CPRATE > 0.
//  - FPVS0 (module_MP_FER_HIRES.F90:2894-2911): table lookup on TBPVS0, the real GPVS_hr table built
//    by phys-fer-gpvs (ctx.tables["fer_gpvs_tbpvs0"]), with C1XPVS0 and C2XPVS0 passed as scalars and
//    the Magnus fallback outside [180, 330] K.
//  - NCOUNT: zeroed by the wrapper at the hourly reset (NTIMESTEP == 0 or MOD(NTIMESTEP,
//    NSTEPS_PER_RESET) == 0, module_SOLVER_GRID_COMP.F90:4963-4996) and incremented before the
//    launch (Fortran NCOUNT=NCOUNT+1 precedes the averages); the kernel gets the incremented value.
//  - b1 guard: ITS_B1 = MAX(ITS, IDS+1) and ITE_B1 = MIN(ITE, IDE-1) (module_SOLVER_GRID_COMP.F90:523-524),
//    so 2..IM-1 and 2..JM-1 for one tile. The port is single process; a multi-tile layout would need
//    these bounds from the tile.
//  - Fortran MAX/MIN map to fmaxf/fminf. Fortran U10**2. is written as x*x.
//
// Launch: grid_cols(IM, JM), BLOCK_COL. No host sync, no allocation, no printf.

#include "nmmb_common.cuh"

// module_DIAGNOSE / module_CONSTANTS / module_MP_FER_HIRES constants (single precision).
#define MF_CU_A       300.0f                           // CU_A (module_DIAGNOSE)
#define MF_CU_B       1.4f                             // CU_B (module_DIAGNOSE)
#define MF_DBZMIN     (-20.0f)                         // dbzmin (module_CONSTANTS)
#define MF_ZMIN       0.01f                            // Zmin (module_DIAGNOSE)
#define MF_RCP        (NMMB_R_D / NMMB_CP)             // RCP = R_D/CP
#define MF_CPV        (4.0f * NMMB_R_V)                // cpv = 4*r_v (module_CONSTANTS)
#define MF_EPSILON    (NMMB_R_D / NMMB_R_V)            // EPSILON = R_D/R_V
#define MF_ONE_MINUS_EPSILON (1.0f - MF_EPSILON)
#define MF_R_FACTOR   (1.0f / MF_EPSILON - 1.0f)
#define MF_CP_FACTOR  (MF_CPV / NMMB_CP - 1.0f)
#define MF_P00_INV    1.0e-5f
#define MF_XMIN       180.0f                           // module_MP_FER_HIRES XMIN
#define MF_XMAX       330.0f                           // module_MP_FER_HIRES XMAX
#define MF_NX         7501                             // module_MP_FER_HIRES NX

// FPVS0 (module_MP_FER_HIRES.F90:2894-2911): table lookup on TBPVS0 (1-based in Fortran; index
// JX1 here is 0-based: Fortran TBPVS0(JX1) -> tbpvs0[jx1-1], TBPVS0(JX1+1) -> tbpvs0[jx1]).
__device__ __forceinline__ float mf_fpvs0(float t, const float* __restrict__ tbpvs0,
                                          float c1xpvs0, float c2xpvs0)
{
    if (t >= MF_XMIN && t <= MF_XMAX) {
        const float xj1 = fminf(fmaxf(c1xpvs0 + c2xpvs0 * t, 1.0f), (float)MF_NX);
        const int jx1 = (int)fminf(xj1, (float)(MF_NX - 1));
        const float a = tbpvs0[jx1 - 1];
        const float b = tbpvs0[jx1];
        return a + (xj1 - (float)jx1) * (b - a);
    }
    // Magnus-Tetens formula for water saturation (Murray, 1967), kPa.
    return 0.61078f * expf(17.2694f * (t - 273.16f) / (t - 35.86f));
}

// CUREFL = CU_A*CUPRATE**CU_B with CUPRATE = RDTPHS*CPRATE, only where CPRATE > 0.
__device__ __forceinline__ float mf_cu_refl(float cprate, float rdtphs)
{
    float curefl = 0.0f;
    if (cprate > 0.0f) {
        const float cuprate = rdtphs * cprate;
        curefl = MF_CU_A * powf(cuprate, MF_CU_B);
    }
    return curefl;
}

// The LL loop: Z units of the two bounding levels, convective part added to the stratiform one.
__device__ __forceinline__ void mf_add_convective(float dbz[2], float curefl)
{
    for (int ll = 0; ll < 2; ++ll) {
        float refl = 0.0f;
        if (dbz[ll] > MF_DBZMIN) refl = powf(10.0f, 0.1f * dbz[ll]);
        dbz[ll] = curefl + refl;
    }
}

// Z (mm6 m-3) to dBZ: IF (Z > ZMIN) 10*ALOG10(Z) ELSE DBZmin.
__device__ __forceinline__ float mf_dbz_out(float z)
{
    return (z > MF_ZMIN) ? 10.0f * log10f(z) : MF_DBZMIN;
}

extern "C" __global__ void io_max_fields(
    const float* __restrict__ T,          // [LM,JM,IM]
    const float* __restrict__ W,          // [LM,JM,IM] W_TOT (Fortran W dummy)
    const float* __restrict__ Z,          // [LM,JM,IM]
    const float* __restrict__ REFL_10CM,  // [LM,JM,IM]
    const float* __restrict__ PINT,       // [LM+1,JM,IM]
    const float* __restrict__ PD,         // [JM,IM]
    const float* __restrict__ PREC,       // [JM,IM]
    const float* __restrict__ CPRATE,     // [JM,IM]
    const float* __restrict__ T2,         // [JM,IM]
    const float* __restrict__ U10,        // [JM,IM]
    const float* __restrict__ V10,        // [JM,IM]
    const float* __restrict__ PSHLTR,     // [JM,IM]
    const float* __restrict__ TSHLTR,     // [JM,IM]
    const float* __restrict__ QSHLTR,     // [JM,IM]
    const float* __restrict__ SGML2,      // [LM]
    const float* __restrict__ PSGML1,     // [LM]
    const float* __restrict__ SR,         // [JM,IM]
    const float* __restrict__ FIS,        // [JM,IM]
    const float* __restrict__ TH10,       // [JM,IM]
    const float* __restrict__ AKHS,       // [JM,IM]
    const float* __restrict__ AKMS,       // [JM,IM]
    const float* __restrict__ SNO,        // [JM,IM]
    const float* __restrict__ TBPVS0,     // [NX] GPVS_hr table (phys-fer-gpvs)
    float* __restrict__ REFDMAX,          // [JM,IM] INOUT
    float* __restrict__ REFDMAX263K,      // [JM,IM] INOUT
    float* __restrict__ PRATEMAX,         // [JM,IM] INOUT
    float* __restrict__ FPRATEMAX,        // [JM,IM] INOUT
    float* __restrict__ UPVVELMAX,        // [JM,IM] INOUT
    float* __restrict__ DNVVELMAX,        // [JM,IM] INOUT
    float* __restrict__ TLMAX,            // [JM,IM] INOUT
    float* __restrict__ TLMIN,            // [JM,IM] INOUT
    float* __restrict__ T02MAX,           // [JM,IM] INOUT
    float* __restrict__ T02MIN,           // [JM,IM] INOUT
    float* __restrict__ RH02MAX,          // [JM,IM] INOUT
    float* __restrict__ RH02MIN,          // [JM,IM] INOUT
    float* __restrict__ U10MAX,           // [JM,IM] INOUT
    float* __restrict__ V10MAX,           // [JM,IM] INOUT
    float* __restrict__ SPD10MAX,         // [JM,IM] INOUT
    float* __restrict__ T10,              // [JM,IM] OUT
    float* __restrict__ T10AVG,           // [JM,IM] INOUT
    float* __restrict__ PSFCAVG,          // [JM,IM] INOUT
    float* __restrict__ AKHSAVG,          // [JM,IM] INOUT
    float* __restrict__ AKMSAVG,          // [JM,IM] INOUT
    float* __restrict__ SNOAVG,           // [JM,IM] INOUT
    const float dt,
    const int nphs,
    const int ntsd,
    const int ncount,                     // NCOUNT already incremented by the wrapper
    const float c1xpvs0,                  // C1XPVS0 (GPVS_hr)
    const float c2xpvs0,                  // C2XPVS0 (GPVS_hr)
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)
    const size_t c = IDX2(i0, j0, im);

    // DTPHS = DT*NPHS, RDTPHS = 3.6e6/DTPHS (single precision, as the Fortran REAL).
    const float dtphs = dt * (float)nphs;
    const float rdtphs = 3.6e6f / dtphs;

    const float pd = PD[c];
    const float zint = FIS[c] / NMMB_G;   // ZINTSFC = FIS/g

    // UPINDX: first level from the top with PLOW >= 10000 and PUP <= 10000 (Fortran L_LOOP).
    int upindx = lm + 1;
    for (int l0 = 0; l0 < lm - 1; ++l0) {
        const float plow = PSGML1[l0 + 1] + SGML2[l0 + 1] * pd;   // PMID(L+1)
        const float pup  = PSGML1[l0] + SGML2[l0] * pd;           // PMID(L)
        if (plow >= 10000.0f && pup <= 10000.0f) { upindx = l0 + 1; break; }
    }

    // Interior guard of the Fortran ITS_B1:ITE_B1, JTS_B1:JTE_B1 loops (1-based 2..IM-1, 2..JM-1).
    const bool b1 = (i0 >= 1) && (i0 <= im - 2) && (j0 >= 1) && (j0 <= jm - 2);

    if (b1) {
        // REFDMAX: 1000 m AGL. vloop (1-based L = 8..LM-1 -> l0 = 7..lm-2). The first level pair
        // with Z(L+1)-ZINTSFC <= 1000 and Z(L)-ZINTSFC >= 1000 gives the bounding levels.
        int lfound = -1;
        for (int l0 = 7; l0 <= lm - 2; ++l0) {
            const float zl  = Z[IDX3(i0, j0, l0, im, jm)];
            const float zl1 = Z[IDX3(i0, j0, l0 + 1, im, jm)];
            if ((zl1 - zint) <= 1000.0f && (zl - zint) >= 1000.0f) { lfound = l0; break; }
        }
        if (lfound >= 0) {
            float dbz[2];
            dbz[0] = REFL_10CM[IDX3(i0, j0, lfound, im, jm)];       // DBZ1(1) = REFL(L)
            dbz[1] = REFL_10CM[IDX3(i0, j0, lfound + 1, im, jm)];   // DBZ1(2) = REFL(L+1)
            const float zmidp1  = Z[IDX3(i0, j0, lfound, im, jm)];      // ZMIDP1 = Z(L)
            const float zmidloc = Z[IDX3(i0, j0, lfound + 1, im, jm)];  // ZMIDloc = Z(L+1)
            mf_add_convective(dbz, mf_cu_refl(CPRATE[c], rdtphs));
            const float fact = (1000.0f + zint - zmidloc) / (zmidloc - zmidp1);
            const float dbz1avg = dbz[1] + (dbz[1] - dbz[0]) * fact;
            REFDMAX[c] = fmaxf(REFDMAX[c], mf_dbz_out(dbz1avg));
        }

        // REFDMAX263K: search from the bottom (L = LM..2, l0 = lm-1..1) for T(L) >= 263.15 and
        // T(L-1) <= 263.15; defaults DBZ1 = DBZmin.
        float dbz263[2] = {MF_DBZMIN, MF_DBZMIN};
        for (int l0 = lm - 1; l0 >= 1; --l0) {
            if (T[IDX3(i0, j0, l0, im, jm)] >= 263.15f &&
                T[IDX3(i0, j0, l0 - 1, im, jm)] <= 263.15f) {
                dbz263[0] = REFL_10CM[IDX3(i0, j0, l0, im, jm)];
                dbz263[1] = REFL_10CM[IDX3(i0, j0, l0 - 1, im, jm)];
                break;
            }
        }
        mf_add_convective(dbz263, mf_cu_refl(CPRATE[c], rdtphs));
        const float dbz263avg = fmaxf(dbz263[0], dbz263[1]);   // maxval(DBZ1)
        REFDMAX263K[c] = fmaxf(REFDMAX263K[c], mf_dbz_out(dbz263avg));
    }

    // UPVVELMAX / DNVVELMAX: W over the levels at or below UPINDX (all levels in the domain).
    {
        float upv = UPVVELMAX[c];
        float dnv = DNVVELMAX[c];
        for (int l0 = 0; l0 < lm; ++l0) {
            if (l0 + 1 >= upindx) {
                const float wv = W[IDX3(i0, j0, l0, im, jm)];
                upv = fmaxf(upv, wv);
                dnv = fminf(dnv, wv);
            }
        }
        UPVVELMAX[c] = upv;
        DNVVELMAX[c] = dnv;
    }

    // TLMAX / TLMIN: lowest layer temperature T(LM).
    {
        const float tlow = T[IDX3(i0, j0, lm - 1, im, jm)];
        TLMAX[c] = fmaxf(TLMAX[c], tlow);
        TLMIN[c] = fminf(TLMIN[c], tlow);
    }

    // Shelter (2 m) and 10 m extremes: only when NTSD > 0 (Fortran IF (NTSD > 0)).
    if (ntsd > 0) {
        const float qsh = QSHLTR[c];
        const float psh = PSHLTR[c];
        const float tsh = TSHLTR[c];
        const float cappa_moist = MF_RCP * (1.0f + qsh * MF_R_FACTOR) / (1.0f + qsh * MF_CP_FACTOR);
        const float t02 = tsh * powf(MF_P00_INV * psh, cappa_moist);
        T02MAX[c] = fmaxf(T02MAX[c], t02);
        T02MIN[c] = fminf(T02MIN[c], t02);

        const float vapor_press = psh * qsh / (MF_EPSILON + qsh * MF_ONE_MINUS_EPSILON);
        const float sat_vapor_press = 1.0e3f * mf_fpvs0(t02, TBPVS0, c1xpvs0, c2xpvs0);
        const float rh02 = fminf(vapor_press / sat_vapor_press, 0.99f);
        RH02MAX[c] = fmaxf(RH02MAX[c], rh02);
        RH02MIN[c] = fminf(RH02MIN[c], rh02);

        const float u10 = U10[c];
        const float v10 = V10[c];
        const float magw2 = u10 * u10 + v10 * v10;
        if (magw2 > SPD10MAX[c]) {
            U10MAX[c] = u10;
            V10MAX[c] = v10;
            SPD10MAX[c] = magw2;
        }
    }

    // Running averages over the NCOUNT calls (the Fortran divides after each update).
    {
        const float nm1 = (float)(ncount - 1);
        const float nn = (float)ncount;

        const float term = -0.273133f / T2[c];
        const float p10 = PSHLTR[c] * expf(term);
        const float t10v = TH10[c] * powf(p10 / 1.0e5f, MF_RCP);
        T10[c] = t10v;
        float a = T10AVG[c] * nm1 + t10v;
        T10AVG[c] = a / nn;

        a = PSFCAVG[c] * nm1 + PINT[IDX3(i0, j0, lm, im, jm)];   // PINT(I,J,LM+1)
        PSFCAVG[c] = a / nn;

        a = AKHSAVG[c] * nm1 + AKHS[c];
        AKHSAVG[c] = a / nn;

        a = AKMSAVG[c] * nm1 + AKMS[c];
        AKMSAVG[c] = a / nn;

        if (SNO[c] > 0.0f) {
            a = SNOAVG[c] * nm1 + 1.0f;
        } else {
            a = SNOAVG[c] * nm1;
        }
        SNOAVG[c] = a / nn;
    }

    // Maximum precipitation rate (total, frozen).
    PRATEMAX[c] = fmaxf(PRATEMAX[c], rdtphs * PREC[c]);
    FPRATEMAX[c] = fmaxf(FPRATEMAX[c], rdtphs * SR[c] * PREC[c]);
}
