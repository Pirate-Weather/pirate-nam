// phys_rrtm_driver_post.cu -- RRTM driver, the post-grrad_nmmb half: output copies, radiation-step
// cloud diagnostics (ICWP /= -1) and the old NAM/GFDL cloud-cover diagnostic (ICWP == -1).
//
// NOTICE: NAM source file phys/module_RA_RRTM.F90 (NOAA/NCEP/EMC, US Government work, no licence file).
// Ported to CUDA for WOOF-NAM from phys/module_RA_RRTM.F90:1113-1418 (nam.v4.2.11, 83df4e13c1).
//
// Where this range sits (UNITS.json split of SUBROUTINE RRTM):
//   phys-rrtm-driver-prep  module_RA_RRTM.F90:98-878    column inputs, P8W/P_PHY, old-cloud inputs (CSMID, CCMID), CU_cloud
//   phys-rrtm-driver-call  module_RA_RRTM.F90:879-1112  dayparts, zenith angles, call setup
//   THIS UNIT              1113-1418   the grrad_nmmb call site (the call is the phys-rrtm-grrad-* units, run before this
//                                      kernel), its outputs, the cloud diagnostics, and the ICWP == -1 cloud block.
//
// Kernels (one launch each, no host sync, no device allocation):
//   rrtm_post_outputs    one thread per column (j, i); loops over l. Fortran 1142-1214 (per-column copies and the
//                        ICWP /= -1 cloud block), 1232-1240 (SIGT4 on LW steps). Runs every RRTM call.
//   rrtm_post_cloud_old  one thread per column; Fortran 1255-1417 (ICWP == -1 only; the NAM 12 km run has ICWP = 1).
//
// Fortran-to-CUDA review (every assignment in 1113-1418 has a counterpart; the Fortran line is in brackets):
//   COSZDG(IRANGE,J) = COSZDG_V            [1142]   -> rrtm_post_outputs, COSZDG
//   CZMEAN(IRANGE,J) = COSZEN_V            [1143]   -> CZMEAN
//   RLWTT(:,:,L) = HLW(:,L), RSWTT = SWH   [1145-1148] -> RLWTT, RSWTT (l loop)
//   RLWIN = SFCFLW%dnfxc                   [1164]   -> RLWIN
//   RSWIN = SFCFSW%dnfxc                   [1165]   -> RSWIN
//   RSWOUT = SFCFSW%upfxc                  [1166]   -> RSWOUT
//   RSWINC = SFCFSW%dnfx0                  [1167]   -> RSWINC
//   RLWTOA = TOPFLW%upfxc                  [1169]   -> RLWTOA
//   RSWTOA = TOPFSW%upfxc                  [1170]   -> RSWTOA
//   ARG_CW = MAXVAL(CW(IDEX,J,1:LM))       [1183]   -> arg_cw (per column; IDEX is the column i)
//   CLDCOV_V = 0, CLDSA_V(1:5) = 0 if ARG_CW < EPSQ2 [1185-1194] -> in place on the grrad scratch (Fortran locals)
//   CLDFRA(:,:,L) = CLDCOV_V, CSMID(:,:,L) = CLDCOV_V [1198-1199] -> CLDFRA, CSMID
//   CFRACL/CFRACM/CFRACH = CLDSA_V(1..3)   [1202-1204]  -> CFRACL, CFRACM, CFRACH
//   ACFRST += CLDSA_V(4); NCFRST += 1      [1210-1211]  -> ACFRST, NCFRST
//   WHERE (CU_cloud) ACFRCV += CLDSA_V(4); NCFRCV += 1 [1213-1214] -> ACFRCV (if CU_cloud), NCFRCV (always)
//   IF (LSSAV) ... ELSE PRINT/STOP         [1176-1221]  -> the wrapper refuses lssav false (the Fortran STOPs)
//   SIGT4 = STBOLT*T*T*T*T at LM           [1232-1240]  -> SIGT4, on LW steps (MOD(NTIMESTEP,NRADL)==0), all columns
//   ICWP == -1 block                       [1255-1417]  -> rrtm_post_cloud_old (below)
//
// Inputs that are not this unit's outputs (the wrapper binds each one to its producer field, see the wrapper):
//   grrad_nmmb outputs (Fortran argument names, module_RA_RRTM.F90:1131-1132): HLW = htrlw and SWH = HTRSW [LM,JM,IM];
//     COSZDG_V, COSZEN_V = GRRAD_COSZDG_V, GRRAD_COSZEN_V [JM,IM]; SFCFLW_DNFXC = sfcflw_dnfxc and TOPFLW_UPFXC =
//     topflw_upfxc [JM,IM]; SFCFSW[0,1,3] (upfxc, dnfxc, dnfx0) and TOPFSW[0] (upfxc) are the packed components of
//     SFCFSW [4,JM,IM] and TOPFSW [3,JM,IM]; CLDCOV_V = cldcov [LM,JM,IM] and CLDSA_V = cldsa [5,JM,IM] (both
//     zeroed in place where the Fortran zeroes its local copies).
//   state: CW, T, PD, DSG2, SGML2 (NMMB tables); CU_CLOUD (int32 [JM,IM]) from ctx.tables['rrtm_driver_call'],
//     written by phys-rrtm-driver-call (Fortran CU_cloud is set at module_RA_RRTM.F90:1074 and 1087).
//   ICWP == -1 only: CSMID, CCMID [LM,JM,IM] (pre-grrad block, module_RA_RRTM.F90:554-837), PDSG1/PSGML1 (grid),
//     PT, PTOPC [4] and LTOP [3] (phys-rrtm-init, RRTM_INIT), CNCLD (module logical, passed as cncld).
//
// Recomputed here, not read: P8W and P_PHY (module_RA_RRTM.F90:480-490, the driver-prep recursion). Same formulas,
// same evaluation order, computed per column because the old-cloud block needs them per cell.
//
// Constants that are RRTM's own PARAMETERs, not nmmb_common: EPSQ2 = 1.E-8 (module_RA_RRTM.F90:286; nmmb_common
// NMMB_EPSQ2 is 0.02 and belongs to another unit), CTHK = 20000 Pa (DATA, :307). STBOLT from nmmb_common (5.67051E-8).
//
// Not ported, with reasons:
//   - MALVL and CLDMAX (ICWP == -1 block) are assigned and never read in this range; kept out of the kernel.
//   - CUCLD, CU_DEPTH and the GR1 convective bogus are in driver-prep, not in this range.
//   - The DO J / dayparts loop (879-1224) is CPU chunking of grrad (CHNK_RRTM). Columns are independent, so the
//     full-domain launch gives the same per-column values.
//   - The LSSAV = .FALSE. branch STOPs in the Fortran; the wrapper raises instead (it never runs with lssav false).

#include "nmmb_common.cuh"

#define RRTM_EPSQ2   1.0e-8f     // EPSQ2, module_RA_RRTM.F90:286
#define RRTM_CTHK    20000.0f    // CTHK(1:3), module_RA_RRTM.F90:307
#define RRTM_LMAX    64          // local-array capacity; the wrapper refuses LM > 64 (NAM LM = 60)

// ---------------------------------------------------------------------------------------------
// Kernel 1: output copies, the ICWP /= -1 cloud block, SIGT4. One thread per column.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void rrtm_post_outputs(
    const int IM, const int JM, const int LM,
    const int icwp, const int lssav, const int lw_step,
    const float* __restrict__ HLW, const float* __restrict__ SWH,
    const float* __restrict__ COSZDG_V, const float* __restrict__ COSZEN_V,
    const float* __restrict__ SFCFLW_DNFXC, const float* __restrict__ SFCFSW_DNFXC,
    const float* __restrict__ SFCFSW_UPFXC, const float* __restrict__ SFCFSW_DNFX0,
    const float* __restrict__ TOPFLW_UPFXC, const float* __restrict__ TOPFSW_UPFXC,
    float* CLDCOV_V, float* CLDSA_V,
    const float* __restrict__ CW, const float* __restrict__ T,
    const int* __restrict__ CU_CLOUD,
    float* RLWTT, float* RSWTT, float* CLDFRA, float* CSMID,
    float* RLWIN, float* RSWIN, float* RSWINC, float* RSWOUT, float* RLWTOA, float* RSWTOA,
    float* CZMEAN, float* SIGT4, float* COSZDG,
    float* CFRACL, float* CFRACM, float* CFRACH,
    float* ACFRST, float* ACFRCV, int* NCFRST, int* NCFRCV)
{
    COLUMN_IJ(IM, JM)
    const size_t n2 = IDX2(i0, j0, IM);          // 2-D [j, i]
    const size_t plane = (size_t)JM * IM;        // stride of nc in CLDSA_V [5, JM, IM]

    // ---- 2-D copies from the grrad outputs (1142-1143, 1164-1170) ----
    COSZDG[n2]  = COSZDG_V[n2];
    CZMEAN[n2]  = COSZEN_V[n2];
    RLWIN[n2]   = SFCFLW_DNFXC[n2];
    RSWIN[n2]   = SFCFSW_DNFXC[n2];
    RSWOUT[n2]  = SFCFSW_UPFXC[n2];
    RSWINC[n2]  = SFCFSW_DNFX0[n2];
    RLWTOA[n2]  = TOPFLW_UPFXC[n2];
    RSWTOA[n2]  = TOPFSW_UPFXC[n2];

    // ---- heating-rate copies, DO L=1,LM (1145-1148) ----
    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t p = IDX3(i0, j0, l0, IM, JM);
        RLWTT[p] = HLW[p];
        RSWTT[p] = SWH[p];
    }

    // ---- cloud block for ICWP /= -1 (1176-1214) ----
    if (icwp != -1) {
        if (lssav) {
            // ARG_CW = MAXVAL(CW(IDEX,J,1:LM)), total condensate of this column
            float arg_cw = -3.402823466e38f;
            for (int l0 = 0; l0 < LM; ++l0) {
                const float v = CW[IDX3(i0, j0, l0, IM, JM)];
                if (v > arg_cw) arg_cw = v;
            }
            // Eliminate cloud fraction where condensate < EPSQ2 (1180-1194)
            if (arg_cw < RRTM_EPSQ2) {
                for (int l0 = 0; l0 < LM; ++l0) CLDCOV_V[IDX3(i0, j0, l0, IM, JM)] = 0.0f;
                for (int nc = 0; nc < 5; ++nc) CLDSA_V[(size_t)nc * plane + n2] = 0.0f;
            }
            // CLDFRA and CSMID from the (possibly zeroed) grrad cloud cover (1197-1200)
            for (int l0 = 0; l0 < LM; ++l0) {
                const size_t p = IDX3(i0, j0, l0, IM, JM);
                CLDFRA[p] = CLDCOV_V[p];
                CSMID[p]  = CLDCOV_V[p];
            }
            CFRACL[n2] = CLDSA_V[0 * plane + n2];          // CLDSA_V(1:im,1)
            CFRACM[n2] = CLDSA_V[1 * plane + n2];          // CLDSA_V(1:im,2)
            CFRACH[n2] = CLDSA_V[2 * plane + n2];          // CLDSA_V(1:im,3)
            // Accumulators (1210-1214): ACFRCV only where CU_cloud, NCFRCV always
            ACFRST[n2] = ACFRST[n2] + CLDSA_V[3 * plane + n2];
            NCFRST[n2] = NCFRST[n2] + 1;
            if (CU_CLOUD[n2] != 0) ACFRCV[n2] = ACFRCV[n2] + CLDSA_V[3 * plane + n2];
            NCFRCV[n2] = NCFRCV[n2] + 1;
        }
    }

    // ---- SIGT4 on LW steps (1232-1240): STBOLT*TDUM*TDUM*TDUM*TDUM, TDUM = T(I,J,LM) ----
    if (lw_step) {
        const float tdum = T[IDX3(i0, j0, LM - 1, IM, JM)];
        SIGT4[n2] = NMMB_STBOLT * tdum * tdum * tdum * tdum;
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel 2: the ICWP == -1 (old NAM/GFDL) cloud-cover diagnostic (1255-1417). One thread per column.
// Writes CFRACL/CFRACM/CFRACH and the accumulators, and the RRTM-local CLDCFR, MTOP, MBOT (3 levels).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void rrtm_post_cloud_old(
    const int IM, const int JM, const int LM, const int cncld,
    const float PT,
    const float* __restrict__ PD,
    const float* __restrict__ DSG2, const float* __restrict__ SGML2,
    const float* __restrict__ PDSG1, const float* __restrict__ PSGML1,
    const float* __restrict__ CSMID, const float* __restrict__ CCMID,
    const float* __restrict__ PTOPC, const int* __restrict__ LTOP,
    float* CLDCFR, int* MTOP, int* MBOT,
    float* CFRACL, float* CFRACM, float* CFRACH,
    float* ACFRST, float* ACFRCV, int* NCFRST, int* NCFRCV)
{
    COLUMN_IJ(IM, JM)
    if (LM > RRTM_LMAX) return;                  // refused by the wrapper; capacity of the local arrays
    const size_t n2 = IDX2(i0, j0, IM);
    const size_t plane = (size_t)JM * IM;
    const int LM1 = LM - 1;

    // Level arrays, 1-based as in the Fortran: P8W(1..LM+1), P_PHY(1..LM), CLDAMT(0..LM).
    float p8w[RRTM_LMAX + 2];
    float pphy[RRTM_LMAX + 1];
    float cldamt[RRTM_LMAX + 1];

    // P8W and P_PHY (module_RA_RRTM.F90:480-490), recomputed for this column
    const float pd = PD[n2];
    p8w[0] = 0.0f;                               // never read: CLDAMT(0) = 0 short-circuits the only use
    p8w[1] = PT;
    for (int k = 1; k <= LM; ++k) {
        p8w[k + 1] = p8w[k] + PDSG1[k - 1] + DSG2[k - 1] * pd;
        pphy[k]    = SGML2[k - 1] * pd + PSGML1[k - 1];
    }

    // CLDAMT(0..LM) = 0 for the cell (1262-1263); CLDCFR, MTOP, MBOT zeroed per call (524-536)
    for (int L = 0; L <= LM; ++L) cldamt[L] = 0.0f;
    float cldcfr[3] = {0.0f, 0.0f, 0.0f};
    int mtop[3] = {0, 0, 0};
    int mbot[3] = {0, 0, 0};

    // DO 480 NLVL = 1,3
    for (int nlvl = 1; nlvl <= 3; ++nlvl) {
        const int lltop = LM + 1 - LTOP[nlvl - 1];      // LLTOP = LM+1-LTOP(NLVL)
        if (lltop >= LM) continue;                      // GO TO 480
        int llbot;
        if (nlvl > 1) {
            llbot = LM + 1 - LTOP[nlvl - 2] - 1;        // LLBOT = LM+1-LTOP(NLVL-1)-1
            llbot = min(llbot, LM1);
        } else {
            llbot = LM1;
        }

        // DO 435: CLDAMT = max(CSMID, CCMID) over LLTOP..LLBOT
        for (int L = lltop; L <= llbot; ++L) {
            const float c = CSMID[IDX3(i0, j0, L - 1, IM, JM)];
            const float m = CCMID[IDX3(i0, j0, L - 1, IM, JM)];
            cldamt[L] = fmaxf(c, m);
        }

        // Total cloud fraction in this pressure domain (1297-1383)
        float cl1 = 0.0f, cl2 = 0.0f;
        int kbt1 = llbot, kbt2 = llbot, kth1 = 0, kth2 = 0;
        // DO 450 LL = LLTOP,LLBOT with L = LLBOT-LL+LLTOP: L runs from LLBOT down to LLTOP
        for (int L = llbot; L >= lltop; --L) {
            const float cr1 = cldamt[L];
            const bool bitx = (p8w[L] >= PTOPC[nlvl]) && (p8w[L] < PTOPC[nlvl - 1]) && (cldamt[L] > 0.0f);
            if (!bitx) continue;                        // IF(.NOT.BIT1) GO TO 450

            // first cloud layer / consecutive cloud layer (BITY, BITZ from the first test)
            if (kth2 <= 0) {                            // BITY = BITX .AND. KTH2 <= 0
                kbt2 = L;
                kth2 = 1;
            } else {                                    // BITZ = BITX .AND. KTH2 > 0
                const int ktop1 = kbt2 - kth2 + 1;
                const float dpcl = pphy[kbt2] - pphy[ktop1];
                if (dpcl < RRTM_CTHK) kth2 = kth2 + 1;
                else                  kbt2 = kbt2 - 1;
            }
            if (bitx) cl2 = fmaxf(cl2, cr1);            // AMAX1(CL2, CR1)

            // separated / domain-boundary layers: random overlap (second BITY)
            const bool bity = bitx && ((cldamt[L - 1] <= 0.0f) || (p8w[L - 1] < PTOPC[nlvl]));
            if (!bity) continue;                        // IF(.NOT.BIT2) GO TO 450
            const bool bitz = bity && (cl1 > 0.0f);
            const bool bitw = bity && (cl1 <= 0.0f);

            if (bitz) {
                kbt1 = (int)((cl1 * (float)kbt1 + cl2 * (float)kbt2) / (cl1 + cl2));
                kth1 = (int)((cl1 * (float)kth1 + cl2 * (float)kth2) / (cl1 + cl2)) + 1;
                cl1  = cl1 + cl2 - cl1 * cl2;
            }
            if (bitw) {
                kbt1 = kbt2;
                kth1 = kth2;
                cl1  = cl2;
            }
            // BITY: start a new domain-boundary layer
            kbt2 = llbot;
            kth2 = 0;
            cl2  = 0.0f;
        }

        cldcfr[nlvl - 1] = fminf(1.0f, cl1);            // AMIN1(1.0, CL1)
        mtop[nlvl - 1]   = min(kbt1, kbt1 - kth1 + 1);
        mbot[nlvl - 1]   = kbt1;
    }                                                   // 480 CONTINUE

    // RRTM-local outputs (3 levels, [3, JM, IM])
    for (int k = 0; k < 3; ++k) {
        CLDCFR[(size_t)k * plane + n2] = cldcfr[k];
        MTOP[(size_t)k * plane + n2]   = mtop[k];
        MBOT[(size_t)k * plane + n2]   = mbot[k];
    }

    // CFRACL/CFRACM/CFRACH = CLDCFR(1..3) (1398-1400)
    const float cfracl = cldcfr[0];
    const float cfracm = cldcfr[1];
    const float cfrach = cldcfr[2];
    CFRACL[n2] = cfracl;
    CFRACM[n2] = cfracm;
    CFRACH[n2] = cfrach;

    if (cncld) {
        // Maximum stratiform / convective cloud fraction over the column (1403-1412)
        float cfsmax = 0.0f, cfcmax = 0.0f;
        for (int l0 = 0; l0 < LM; ++l0) {
            const float c = CSMID[IDX3(i0, j0, l0, IM, JM)];
            const float m = CCMID[IDX3(i0, j0, l0, IM, JM)];
            cfsmax = fmaxf(cfsmax, c);
            cfcmax = fmaxf(cfcmax, m);
        }
        ACFRST[n2] = ACFRST[n2] + cfsmax;
        NCFRST[n2] = NCFRST[n2] + 1;
        ACFRCV[n2] = ACFRCV[n2] + cfcmax;
        NCFRCV[n2] = NCFRCV[n2] + 1;
    } else {
        // Grid-scale cloudiness only (1414-1417): random overlap of the three layers
        const float cfravg = 1.0f - (1.0f - cfracl) * (1.0f - cfracm) * (1.0f - cfrach);
        ACFRST[n2] = ACFRST[n2] + cfravg;
        NCFRST[n2] = NCFRST[n2] + 1;
    }
}
