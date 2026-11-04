// phys_turbl_driver.cu -- TURBL outer driver glue for nam-gpu (column prep, H->V wind averaging,
// surface-layer / LSM / PBL / GWD call sequencing, flux and state updates).
//
// Notice: module_TURBULENCE.F90 in NAM nam.v4.2.11 (MatthewPyle-NOAA/NAM @ 83df4e13c1, NMMB
// physics, NOAA/NCEP/EMC) keeps its author headers (BLACK, JANJIC, WANG, VASIC, YL). There is no
// licence file in the NAM tree; the NOAA source notice in NOTICE applies. NAM is US Government
// work. The routine's own header block (Fortran lines 1-30) is kept as that notice.
//
// Ported to CUDA for nam-gpu from nmm/module_TURBULENCE.F90:89-1420 (nam.v4.2.11, 83df4e13c1).
//
// Scheme selection (parm/nam_configfile: turbulence myj, sfc_layer myj, land_surface noah,
// gwdflg true, microphysics fer_hires, np3d 3):
//   PBL_PHYSICS = MYJPBLSCHEME (2), SFCLAY_PHYSICS = JSFCSCHEME (2), SURFACE_PHYSICS = LSMSCHEME (2).
//   The GFS PBL (9, 93), GFDL surface layer (88), LISS (101) and GFDL slab branches are not
//   selected by the 12 km configuration and are not called here.
//   The urban branch (UCMCALL == 1) is not ported: it is not reached with UCMCALL 0.
//   The SST_UPDATE block is dead code in the Fortran (SST_UPDATE = 0) and is not ported.
//   The TSHLTR range check and the TURBL error prints are diagnostics only (Fortran WRITE(0,*)).
//   The LPBL search (turbulence /= myj) is ported in full, gated by a host flag.
//
// Dependency units (their wrappers run between the kernels below, in Fortran call order):
//   phys-jsfc-jsfc (JSFC), phys-noah-driver (NOAHLSM), phys-noah-sfcdiags (SFCDIAGS),
//   phys-myj-myjpbl (MYJPBL), phys-gwd-driver (GWD_DRIVER).
// Scratch names handed to them are the Fortran names (DZ, Z, PHINT, PHMID, TH, EXNER, U_PHY, V_PHY,
// RTHBLTEN, RQBLTEN, RQCBLTEN, RQIBLTEN, DUDT_GWD, DVDT_GWD, ...), see the wrapper docstring.
//
// Index convention (CONTRACT.md 1): Fortran (i,j,l) with 1-based l -> [l0, j0, i0] with l0 = l-1.
// Full-domain loops cover i0 in [0, IM-1], j0 in [0, JM-1]. The B1 loops (its_b1 = max(its, ids+1),
// ite_b1 = min(ite, ide-1), likewise j, module_DM_PARALLEL.F90:799-804) cover i0 in [1, IM-2],
// j0 in [1, JM-2]. Single card, no halo: its = ids = 1, ite = ide = IM.
//
// Constants (module_CONSTANTS via nmmb_common.cuh): XLV = ELWV, G, R_D, CAPPA, P608, STBOLT, EPSQ.
// RHOWATER = 1000 (module_CONSTANTS rhowater). Single precision (kfpt). Fast numerics: FMA on,
// IEEE division. One thread per column (or per cell) with the level loop inside; no host sync,
// no malloc.

#include "nmmb_common.cuh"

#define TURBL_RHOWATER 1000.0f   /* module_CONSTANTS rhowater */

// ---------------------------------------------------------------------------------------------
// 1. Column prep: TURBL cell block (Fortran 556-572), NTSD == 0 block (574-586), Q2 floor and
//    EXCH_H/DZ zero (588-600), Z zero (602-614), and the per-column prep with the vertical
//    working arrays (Fortran 630-790).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_prep_column(
    float* __restrict__ SM,            // [JM,IM] INOUT  (bandaid when ntsd0)
    const float* __restrict__ SST,     // [JM,IM]
    float* __restrict__ THS,           // [JM,IM] INOUT  skin potential temperature
    float* __restrict__ TSFC,          // [JM,IM] INOUT  skin temperature
    float* __restrict__ VGFRCK,        // [JM,IM] OUT    100 * VEGFRC
    const float* __restrict__ VEGFRC,  // [JM,IM]
    const float* __restrict__ SNO,     // [JM,IM]
    const float* __restrict__ SI,      // [JM,IM]
    float* __restrict__ SNOW,          // [JM,IM] OUT
    float* __restrict__ SNOWH,         // [JM,IM] OUT
    float* __restrict__ XLAND,         // [JM,IM] OUT    SM + 1
    float* __restrict__ T2,            // [JM,IM] OUT    = TSFC (pre-update)
    const float* __restrict__ Z0,      // [JM,IM]
    float* __restrict__ Z0BASE,        // [JM,IM] OUT when ntsd0
    const float* __restrict__ SICE,    // [JM,IM]
    float* __restrict__ Q2,            // [LM,JM,IM] INOUT floor EPSQ2
    float* __restrict__ EXCH_H,        // [LM,JM,IM] OUT 0
    float* __restrict__ DZ,            // [LM,JM,IM] OUT
    float* __restrict__ Z,             // [LM+1,JM,IM] OUT
    const float* __restrict__ FIS,     // [JM,IM]
    const float* __restrict__ PD,      // [JM,IM]
    const float* __restrict__ T,       // [LM,JM,IM]
    const float* __restrict__ Q,       // [LM,JM,IM]
    const float* __restrict__ PREC,    // [JM,IM]
    const float* __restrict__ DDATA,   // [JM,IM]  read only when pcpflg
    const float* __restrict__ CZEN,    // [JM,IM]
    const float* __restrict__ CZMEAN,  // [JM,IM]
    const float* __restrict__ SIGT4,   // [JM,IM]
    const float* __restrict__ RLWIN,   // [JM,IM]
    const float* __restrict__ RSWIN,   // [JM,IM]
    const float* __restrict__ RSWOUT,  // [JM,IM]
    const float* __restrict__ F_RIMEF, // [LM,JM,IM]
    float* __restrict__ RAIN,          // [JM,IM] OUT
    float* __restrict__ RAINBL,        // [JM,IM] OUT 0
    float* __restrict__ FACTRS,        // [JM,IM] OUT
    float* __restrict__ RLW_DN_SFC,    // [JM,IM] OUT
    float* __restrict__ RSW_NET_SFC,   // [JM,IM] OUT
    float* __restrict__ RSW_DN_SFC,    // [JM,IM] OUT
    float* __restrict__ SFCZ,          // [JM,IM] OUT
    float* __restrict__ EXNSFC,        // [JM,IM] OUT
    float* __restrict__ TH2X,          // [JM,IM] OUT
    float* __restrict__ Q2X,           // [JM,IM] OUT
    float* __restrict__ RIMEF,         // [JM,IM] OUT
    float* __restrict__ PHINT,         // [LM+1,JM,IM] OUT
    float* __restrict__ PHMID,         // [LM,JM,IM] OUT
    float* __restrict__ RR,            // [LM,JM,IM] OUT
    float* __restrict__ EXNER,         // [LM,JM,IM] OUT
    float* __restrict__ TH,            // [LM,JM,IM] OUT
    float* __restrict__ DELP,          // [LM,JM,IM] OUT
    float* __restrict__ RTHBLTEN,      // [LM,JM,IM] OUT 0
    float* __restrict__ RQBLTEN,       // [LM,JM,IM] OUT 0
    float* __restrict__ RQCBLTEN,      // [LM,JM,IM] OUT 0
    float* __restrict__ RQIBLTEN,      // [LM,JM,IM] OUT 0
    const float* __restrict__ SG2,     // [LM+1]
    const float* __restrict__ PSG1,    // [LM+1]
    const float* __restrict__ SGML2,   // [LM]
    const float* __restrict__ PSGML1,  // [LM]
    const float* __restrict__ EPSQ2,   // [LM]
    const float sno_factr,             // SNO_FACTR for SNOWH (0.001 for noah)
    const int pcpflg,
    const int ntsd0,                   // NTSD == 0
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)
    const size_t c2 = IDX2(i0, j0, im);
    const float g_inv = 1.0f / NMMB_G;

    // ---- first cell block ----
    VGFRCK[c2] = 100.0f * VEGFRC[c2];
    SNOW[c2] = SNO[c2];
    SNOWH[c2] = SI[c2] * sno_factr;
    XLAND[c2] = SM[c2] + 1.0f;
    T2[c2] = TSFC[c2];

    // ---- NTSD == 0 block ----
    if (ntsd0) {
        Z0BASE[c2] = Z0[c2];
        if (SM[c2] > 0.5f && SICE[c2] > 0.5f) SM[c2] = 0.0f;   // Bandaid
    }

    // ---- Q2 floor, EXCH_H and DZ zero ----
    for (int k0 = 0; k0 < lm; ++k0) {
        const size_t c3 = IDX3(i0, j0, k0, im, jm);
        Q2[c3] = fmaxf(Q2[c3], EPSQ2[k0]);
        EXCH_H[c3] = 0.0f;
        DZ[c3] = 0.0f;
    }
    // ---- Z zero, levels 1..LM+1 ----
    for (int k0 = 0; k0 <= lm; ++k0) {
        Z[IDX3(i0, j0, k0, im, jm)] = 0.0f;
    }

    // ---- per-column prep ----
    const float pdsl = PD[c2];
    const float psfc = SG2[lm] * pdsl + PSG1[lm];        // SG2(LM+1), PSG1(LM+1)
    PHINT[IDX3(i0, j0, lm, im, jm)] = psfc;               // PHINT(LM+1) = PSFC
    const float exnsfc = powf(1.0e5f / psfc, NMMB_CAPPA);
    EXNSFC[c2] = exnsfc;
    THS[c2] = (SST[c2] * exnsfc) * SM[c2] + THS[c2] * (1.0f - SM[c2]);
    TSFC[c2] = THS[c2] / exnsfc;
    SFCZ[c2] = FIS[c2] * g_inv;

    // precipitation for the surface budget (PCPFLG off in NAM: PREC * RHOWATER)
    if (pcpflg && DDATA[c2] < 100.0f) {
        RAIN[c2] = DDATA[c2] * TURBL_RHOWATER;
    } else {
        RAIN[c2] = PREC[c2] * TURBL_RHOWATER;
    }
    RAINBL[c2] = 0.0f;

    const float plm = SGML2[lm - 1] * pdsl + PSGML1[lm - 1];   // SGML2(LM), PSGML1(LM)
    TH2X[c2] = T[IDX3(i0, j0, lm - 1, im, jm)] * powf(1.0e5f / plm, NMMB_CAPPA);
    Q2X[c2] = Q[IDX3(i0, j0, lm - 1, im, jm)];
    RIMEF[c2] = fmaxf(1.0f, F_RIMEF[IDX3(i0, j0, lm - 1, im, jm)]);

    // long and shortwave fluxes at the ground
    const float czmean = CZMEAN[c2];
    if (czmean > 0.0f) {
        FACTRS[c2] = CZEN[c2] / czmean;
    } else {
        FACTRS[c2] = 0.0f;
    }
    const float sigt4 = SIGT4[c2];
    float factrl;
    if (sigt4 > 0.0f) {
        const float tlmh = T[IDX3(i0, j0, lm - 1, im, jm)];
        factrl = NMMB_STBOLT * tlmh * tlmh * tlmh * tlmh / sigt4;
    } else {
        factrl = 0.0f;
    }
    RLW_DN_SFC[c2] = RLWIN[c2] * factrl;
    RSW_NET_SFC[c2] = (RSWIN[c2] - RSWOUT[c2]) * FACTRS[c2];
    RSW_DN_SFC[c2] = RSWIN[c2] * FACTRS[c2];

    Z[IDX3(i0, j0, lm, im, jm)] = SFCZ[c2];              // Z(LM+1) = SFCZ

    // ---- vertical working arrays, K = LM..1 ----
    for (int k0 = lm - 1; k0 >= 0; --k0) {
        const size_t c3 = IDX3(i0, j0, k0, im, jm);
        const float plyr = SGML2[k0] * pdsl + PSGML1[k0];
        const float ql = fmaxf(Q[c3], NMMB_EPSQ);
        const float tl = T[c3];
        RR[c3] = plyr / (NMMB_R_D * tl);
        const float rxner = powf(1.0e5f / plyr, NMMB_CAPPA);
        EXNER[c3] = 1.0f / rxner;
        TH[c3] = tl * rxner;
        PHINT[c3] = SG2[k0] * pdsl + PSG1[k0];           // PHINT(K)
        PHMID[c3] = plyr;

        RQCBLTEN[c3] = 0.0f;
        RQIBLTEN[c3] = 0.0f;
        RTHBLTEN[c3] = 0.0f;
        RQBLTEN[c3] = 0.0f;

        const float dphi = PHINT[IDX3(i0, j0, k0 + 1, im, jm)] - PHINT[c3];
        DZ[c3] = tl * (NMMB_P608 * ql + 1.0f) * NMMB_R_D * dphi / (plyr * NMMB_G);
        Z[c3] = Z[IDX3(i0, j0, k0 + 1, im, jm)] + DZ[c3];
        DELP[c3] = dphi;
    }
}

// ---------------------------------------------------------------------------------------------
// 2. Wind averaging to mass points and zeroing of the TURBL tendencies (Fortran 527-540 zeros,
//    718-737 H->V average on the B1 region). U_PHY/V_PHY = (U or V at the four V points around
//    the mass point) * 0.25, the same operation order as the Fortran.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_uv_phy(
    const float* __restrict__ U,       // [LM,JM,IM]
    const float* __restrict__ V,       // [LM,JM,IM]
    float* __restrict__ U_PHY,         // [LM,JM,IM] OUT
    float* __restrict__ V_PHY,         // [LM,JM,IM] OUT
    float* __restrict__ DUDT,          // [LM,JM,IM] OUT 0
    float* __restrict__ DVDT,          // [LM,JM,IM] OUT 0
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    for (int k0 = 0; k0 < lm; ++k0) {
        const size_t c3 = IDX3(i0, j0, k0, im, jm);
        U_PHY[c3] = 0.0f;
        V_PHY[c3] = 0.0f;
        DUDT[c3] = 0.0f;
        DVDT[c3] = 0.0f;
        if (b1) {
            const size_t cim = IDX3(i0 - 1, j0, k0, im, jm);
            const size_t cjm = IDX3(i0, j0 - 1, k0, im, jm);
            const size_t cmm = IDX3(i0 - 1, j0 - 1, k0, im, jm);
            U_PHY[c3] = (U[c3] + U[cim] + U[cjm] + U[cmm]) * 0.25f;
            V_PHY[c3] = (V[c3] + V[cim] + V[cjm] + V[cmm]) * 0.25f;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Surface pre-state: ONE (Fortran 659-662 and 713-721), RMOL and SFCEVPX zero (659-662),
//    TWBS/QWBS zero, the lowest-layer state THLOW/TLOW/QLOW/CWMLOW, and the PBLH clamp
//    (Fortran 762-772), and the B1 CHS/CPM/CHS2/QGH zero and RAINBL accumulation (Fortran 722-732).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_sfc_pre(
    float* __restrict__ ONE,           // [JM,IM] OUT
    const float* __restrict__ MAVAIL,  // [JM,IM]
    const int one_from_mavail,         // LAND_SURFACE /= 'nmm'
    float* __restrict__ RMOL,          // [JM,IM] OUT 0
    float* __restrict__ SFCEVPX,       // [JM,IM] OUT 0
    float* __restrict__ TWBS,          // [JM,IM] OUT 0
    float* __restrict__ QWBS,          // [JM,IM] OUT 0
    float* __restrict__ THLOW,         // [JM,IM] OUT
    float* __restrict__ TLOW,          // [JM,IM] OUT
    float* __restrict__ QLOW,          // [JM,IM] INOUT
    float* __restrict__ CWMLOW,        // [JM,IM] OUT
    const float* __restrict__ CW,      // [LM,JM,IM]
    const float* __restrict__ T,       // [LM,JM,IM]
    const float* __restrict__ Q,       // [LM,JM,IM]
    const float* __restrict__ TH,      // [LM,JM,IM]
    const float* __restrict__ Z,       // [LM+1,JM,IM]
    float* __restrict__ PBLH,          // [JM,IM] INOUT
    float* __restrict__ QGH,           // [JM,IM] OUT 0 (B1)
    float* __restrict__ CHS,           // [JM,IM] OUT 0 (B1)
    float* __restrict__ CPM,           // [JM,IM] OUT 0 (B1)
    float* __restrict__ CHS2,          // [JM,IM] OUT 0 (B1)
    float* __restrict__ RAINBL,        // [JM,IM] INOUT (B1)
    const float* __restrict__ RAIN,    // [JM,IM]
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)
    const size_t c2 = IDX2(i0, j0, im);
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);

    ONE[c2] = one_from_mavail ? MAVAIL[c2] : 1.0f;
    RMOL[c2] = 0.0f;
    SFCEVPX[c2] = 0.0f;

    TWBS[c2] = 0.0f;
    QWBS[c2] = 0.0f;
    const size_t cl = IDX3(i0, j0, lm - 1, im, jm);
    THLOW[c2] = TH[cl];
    TLOW[c2] = T[cl];
    QLOW[c2] = fmaxf(Q[cl], NMMB_EPSQ);
    const float qlowx = QLOW[c2] / (1.0f - QLOW[c2]);
    QLOW[c2] = qlowx / (1.0f + qlowx);
    CWMLOW[c2] = CW[cl];
    PBLH[c2] = fmaxf(PBLH[c2], 0.0f);
    PBLH[c2] = fminf(PBLH[c2], Z[IDX3(i0, j0, lm - 1, im, jm)]);

    if (b1) {
        QGH[c2] = 0.0f;
        CHS[c2] = 0.0f;
        CPM[c2] = 0.0f;
        CHS2[c2] = 0.0f;
        RAINBL[c2] = RAINBL[c2] + RAIN[c2];
        RAINBL[c2] = fmaxf(RAINBL[c2], 0.0f);
    }
}

// ---------------------------------------------------------------------------------------------
// 4. PSFC_OUT save (Fortran 738-744), inside the sfc_and_sfclyr block (NTSD == 1 or mod == 0).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_sfc_save(
    const float* __restrict__ PHINT,   // [LM+1,JM,IM]
    float* __restrict__ PSFC_OUT,      // [JM,IM] OUT
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    if (b1) PSFC_OUT[IDX2(i0, j0, im)] = PHINT[IDX3(i0, j0, lm, im, jm)];
}

// ---------------------------------------------------------------------------------------------
// 5. Noah post-step (Fortran 1000-1010 region after NOAHLSM): SFCEVPX accumulation, SFCEXC,
//    SOILTB from the bottom soil layer STC(NSOIL). STC is [NSOIL, JM, IM].
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_noah_post(
    const float* __restrict__ QWBS,    // [JM,IM]
    const float* __restrict__ CHS,     // [JM,IM]
    const float* __restrict__ STC,     // [NSOIL,JM,IM]
    float* __restrict__ SFCEVPX,       // [JM,IM] INOUT
    float* __restrict__ SFCEXC,        // [JM,IM] OUT
    float* __restrict__ SOILTB,        // [JM,IM] OUT
    const float dtbl,
    const int nsoil,
    const int im,
    const int jm)
{
    COLUMN_IJ(im, jm)
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    if (!b1) return;
    const size_t c2 = IDX2(i0, j0, im);
    SFCEVPX[c2] = SFCEVPX[c2] + QWBS[c2] * dtbl;
    SFCEXC[c2] = CHS[c2];
    SOILTB[c2] = STC[((size_t)(nsoil - 1) * jm + j0) * im + i0];
}

// ---------------------------------------------------------------------------------------------
// 6. RAINBL reset in mm (Fortran 1030-1035), after the sfc_select block.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_rainbl_reset(
    float* __restrict__ RAINBL,        // [JM,IM] OUT 0 (B1)
    const int im,
    const int jm)
{
    COLUMN_IJ(im, jm)
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    if (b1) RAINBL[IDX2(i0, j0, im)] = 0.0f;
}

// ---------------------------------------------------------------------------------------------
// 7. AKHS_OUT / AKMS_OUT (Fortran 1046-1053): the surface exchange coefficients times half the
//    lowest layer depth, kept for the next step. Full domain.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_akhs_out(
    const float* __restrict__ DZ,      // [LM,JM,IM]
    const float* __restrict__ AKHS,    // [JM,IM]
    const float* __restrict__ AKMS,    // [JM,IM]
    float* __restrict__ AKHS_OUT,      // [JM,IM] OUT
    float* __restrict__ AKMS_OUT,      // [JM,IM] OUT
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)
    const size_t c2 = IDX2(i0, j0, im);
    const float dzhalf = 0.5f * DZ[IDX3(i0, j0, lm - 1, im, jm)];
    AKHS_OUT[c2] = AKHS[c2] * dzhalf;
    AKMS_OUT[c2] = AKMS[c2] * dzhalf;
}

// ---------------------------------------------------------------------------------------------
// 8. Uncomputed-edge fill of the 10 m and 2 m diagnostics (Fortran 1094-1155). The four global
//    boundaries are applied in Fortran order W, E, S, N, so the host launches side = 0..3 in
//    that order (corner cells are written by the later sides, as in Fortran).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_edge_fill(
    float* __restrict__ TH10,
    float* __restrict__ Q10,
    float* __restrict__ U10,
    float* __restrict__ V10,
    float* __restrict__ TSHLTR,
    float* __restrict__ QSHLTR,
    const int side,                    // 0 W, 1 E, 2 S, 3 N
    const int im,
    const int jm)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    int di, dj, si, sj;
    if (side == 0 || side == 1) {
        if (t >= jm) return;
        dj = t; sj = t;
        di = (side == 0) ? 0 : im - 1;
        si = (side == 0) ? 1 : im - 2;
    } else {
        if (t >= im) return;
        di = t; si = t;
        dj = (side == 2) ? 0 : jm - 1;
        sj = (side == 2) ? 1 : jm - 2;
    }
    const size_t dst = IDX2(di, dj, im);
    const size_t src = IDX2(si, sj, im);
    TH10[dst] = TH10[src];
    Q10[dst] = Q10[src];
    U10[dst] = U10[src];
    V10[dst] = V10[src];
    TSHLTR[dst] = TSHLTR[src];
    QSHLTR[dst] = QSHLTR[src];
}

// ---------------------------------------------------------------------------------------------
// 9. LPBL: model layer containing the PBL top, only when turbulence /= myj (Fortran 1110-1137).
//    The Fortran row loop with its early exit gives, per column, the first (highest-index,
//    scanning from LM down) layer with PBLH <= Z - SFCZ. LPBL stores the 1-based layer number.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_lpbl(
    const float* __restrict__ Z,       // [LM+1,JM,IM]
    const float* __restrict__ SFCZ,    // [JM,IM]
    const float* __restrict__ PBLH,    // [JM,IM]
    int* __restrict__ LPBL,            // [JM,IM] OUT
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)
    // Fortran LPBL loops run over ITS_B1..ITE_B1, JTS_B1..JTE_B1 only (module_TURBULENCE.F90 ~1210-1236);
    // LPBL is INTENT(OUT) and edge values are left untouched.
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    if (!b1) return;
    const size_t c2 = IDX2(i0, j0, im);
    int lp = -1000;
    for (int k0 = lm - 1; k0 >= 0; --k0) {
        const float altitude = Z[IDX3(i0, j0, k0, im, jm)] - SFCZ[c2];
        if (PBLH[c2] <= altitude && lp < 0) lp = k0 + 1;
    }
    LPBL[c2] = lp;
}

// ---------------------------------------------------------------------------------------------
// 10. Snow bookkeeping after the LPBL block (Fortran 1158-1167): SNO = SNOW, SI = SNOWH * factor.
//     NAM uses noah, so factor = 1000 (SNO_FACTR = 1000 when not LISS).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_sno_update(
    const float* __restrict__ SNOW,    // [JM,IM]
    const float* __restrict__ SNOWH,   // [JM,IM]
    float* __restrict__ SNO,           // [JM,IM] OUT
    float* __restrict__ SI,            // [JM,IM] OUT
    const float sno_factr2,
    const int im,
    const int jm)
{
    COLUMN_IJ(im, jm)
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    if (!b1) return;
    const size_t c2 = IDX2(i0, j0, im);
    SNO[c2] = SNOW[c2];
    SI[c2] = SNOWH[c2] * sno_factr2;
}

// ---------------------------------------------------------------------------------------------
// 11. Diagnostic radiation accumulation (Fortran 1181-1204): RADOT, ASW*, ALW*, THS, PREC.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_rad_accum(
    const float* __restrict__ EPSR,    // [JM,IM]
    const float* __restrict__ TSFC,    // [JM,IM]
    const float* __restrict__ EXNSFC,  // [JM,IM]
    const float* __restrict__ RSWIN,   // [JM,IM]
    const float* __restrict__ RSWOUT,  // [JM,IM]
    const float* __restrict__ RSWTOA,  // [JM,IM]
    const float* __restrict__ RLWTOA,  // [JM,IM]
    const float* __restrict__ FACTRS,  // [JM,IM]
    const float* __restrict__ RLW_DN_SFC, // [JM,IM]
    float* __restrict__ RADOT,         // [JM,IM] OUT
    float* __restrict__ ASWIN,         // [JM,IM] INOUT
    float* __restrict__ ASWOUT,        // [JM,IM] INOUT
    float* __restrict__ ASWTOA,        // [JM,IM] INOUT
    float* __restrict__ ALWIN,         // [JM,IM] INOUT
    float* __restrict__ ALWOUT,        // [JM,IM] INOUT
    float* __restrict__ ALWTOA,        // [JM,IM] INOUT
    float* __restrict__ THS,           // [JM,IM] OUT
    float* __restrict__ PREC,          // [JM,IM] OUT 0
    const int im,
    const int jm)
{
    COLUMN_IJ(im, jm)
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    if (!b1) return;
    const size_t c2 = IDX2(i0, j0, im);
    const float tsfc2 = TSFC[c2] * TSFC[c2];
    RADOT[c2] = EPSR[c2] * NMMB_STBOLT * tsfc2 * tsfc2;

    ASWIN[c2] = ASWIN[c2] + RSWIN[c2] * FACTRS[c2];
    ASWOUT[c2] = ASWOUT[c2] - RSWOUT[c2] * FACTRS[c2];
    ASWTOA[c2] = ASWTOA[c2] + RSWTOA[c2] * FACTRS[c2];
    ALWIN[c2] = ALWIN[c2] + RLW_DN_SFC[c2];
    ALWOUT[c2] = ALWOUT[c2] - RADOT[c2];
    ALWTOA[c2] = ALWTOA[c2] + RLWTOA[c2];
    THS[c2] = TSFC[c2] * EXNSFC[c2];
    PREC[c2] = 0.0f;
}

// ---------------------------------------------------------------------------------------------
// 12. GWD tendencies added to the PBL tendencies (Fortran 1305-1314). Full domain.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_dudt_add(
    const float* __restrict__ DUDT_GWD,   // [LM,JM,IM]
    const float* __restrict__ DVDT_GWD,   // [LM,JM,IM]
    float* __restrict__ DUDT,             // [LM,JM,IM] INOUT
    float* __restrict__ DVDT,             // [LM,JM,IM] INOUT
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)
    for (int k0 = 0; k0 < lm; ++k0) {
        const size_t c3 = IDX3(i0, j0, k0, im, jm);
        DUDT[c3] = DUDT[c3] + DUDT_GWD[c3];
        DVDT[c3] = DVDT[c3] + DVDT_GWD[c3];
    }
}

// ---------------------------------------------------------------------------------------------
// 13. Temperature, humidity, condensate update (Fortran 1342-1385): T, Q, QC, QI, CWM, F_ICE,
//     F_RAIN from the PBL tendencies. fer_hires: F_QC, F_QR, F_QS, F_QG true, F_QI false, and
//     QGraup is zero because FER_MIC is true (Fortran "IF(F_QG .AND. .NOT.FER_MIC)").
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_tq_update(
    float* __restrict__ T,             // [LM,JM,IM] INOUT
    float* __restrict__ Q,             // [LM,JM,IM] INOUT
    float* __restrict__ QC,            // [LM,JM,IM] INOUT
    float* __restrict__ QI,            // [LM,JM,IM] INOUT (only when f_qi)
    const float* __restrict__ QR,      // [LM,JM,IM]
    const float* __restrict__ QS,      // [LM,JM,IM]
    const float* __restrict__ QG,      // [LM,JM,IM]
    float* __restrict__ CWM,           // [LM,JM,IM] OUT
    float* __restrict__ F_ICE,         // [LM,JM,IM] OUT
    float* __restrict__ F_RAIN,        // [LM,JM,IM] OUT
    const float* __restrict__ EXNER,   // [LM,JM,IM]
    const float* __restrict__ RTHBLTEN,
    const float* __restrict__ RQBLTEN,
    const float* __restrict__ RQCBLTEN,
    const float* __restrict__ RQIBLTEN,
    const float dtphs,
    const int f_qr,
    const int f_qi,
    const int f_qs,
    const int f_qg,
    const int fer_mic,
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    if (!b1) return;
    for (int k0 = 0; k0 < lm; ++k0) {
        const size_t c3 = IDX3(i0, j0, k0, im, jm);
        T[c3] = T[c3] + dtphs * RTHBLTEN[c3] * EXNER[c3];
        Q[c3] = Q[c3] + RQBLTEN[c3] * dtphs;
        QC[c3] = fmaxf(0.0f, QC[c3] + RQCBLTEN[c3] * dtphs);
        const float qcw = QC[c3];
        float qrain = 0.0f, qci = 0.0f, qsnow = 0.0f, qgraup = 0.0f;
        if (f_qr) qrain = QR[c3];
        if (f_qi) {
            qci = fmaxf(0.0f, QI[c3] + RQIBLTEN[c3] * dtphs);
            QI[c3] = qci;
        }
        if (f_qs) qsnow = QS[c3];
        if (f_qg && !fer_mic) qgraup = QG[c3];
        CWM[c3] = qcw + qrain + qci + qsnow + qgraup;
        F_ICE[c3] = 0.0f;
        F_RAIN[c3] = 0.0f;
        if (CWM[c3] > NMMB_EPSQ) F_ICE[c3] = (qci + qsnow + qgraup) / CWM[c3];
        if (qrain > NMMB_EPSQ) F_RAIN[c3] = qrain / (qcw + qrain);
    }
}

// ---------------------------------------------------------------------------------------------
// 14. Instantaneous and accumulated surface fluxes (Fortran 1420-1440): sign convention,
//     SFCSHX, SFCLHX, SFCEVP, POTEVP, POTFLX, SUBSHX.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_flux_accum(
    float* __restrict__ TWBS,          // [JM,IM] INOUT
    float* __restrict__ QWBS,          // [JM,IM] INOUT
    const float* __restrict__ CHKLOWQ,// [JM,IM]
    const float* __restrict__ SM,      // [JM,IM]
    const float* __restrict__ GRNFLX,  // [JM,IM]
    float* __restrict__ SFCSHX,        // [JM,IM] INOUT
    float* __restrict__ SFCLHX,        // [JM,IM] INOUT
    float* __restrict__ SFCEVP,        // [JM,IM] INOUT
    float* __restrict__ POTEVP,        // [JM,IM] INOUT
    float* __restrict__ POTFLX,        // [JM,IM] OUT
    float* __restrict__ SUBSHX,        // [JM,IM] INOUT
    const float dtphs,
    const int im,
    const int jm)
{
    COLUMN_IJ(im, jm)
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    if (!b1) return;
    const size_t c2 = IDX2(i0, j0, im);
    const float xlv = NMMB_ELWV;
    const float xlvrw = dtphs / (xlv * TURBL_RHOWATER);
    const float factor = -xlv * TURBL_RHOWATER / dtphs;

    TWBS[c2] = -TWBS[c2];
    QWBS[c2] = -QWBS[c2] * xlv * CHKLOWQ[c2];
    SFCSHX[c2] = SFCSHX[c2] + TWBS[c2];
    SFCLHX[c2] = SFCLHX[c2] + QWBS[c2];
    SFCEVP[c2] = SFCEVP[c2] - QWBS[c2] * xlvrw;
    POTEVP[c2] = POTEVP[c2] - QWBS[c2] * SM[c2] * xlvrw;
    POTFLX[c2] = POTEVP[c2] * factor;
    SUBSHX[c2] = SUBSHX[c2] + GRNFLX[c2];
}

// ---------------------------------------------------------------------------------------------
// 15. Hour counters (Fortran 1446-1456): APHTIM, ARDSW, ARDLW, ASRFC += 1. Full domain.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void turbl_counters(
    float* __restrict__ APHTIM,
    float* __restrict__ ARDSW,
    float* __restrict__ ARDLW,
    float* __restrict__ ASRFC,
    const int im,
    const int jm)
{
    COLUMN_IJ(im, jm)
    const size_t c2 = IDX2(i0, j0, im);
    APHTIM[c2] = APHTIM[c2] + 1.0f;
    ARDSW[c2] = ARDSW[c2] + 1.0f;
    ARDLW[c2] = ARDLW[c2] + 1.0f;
    ASRFC[c2] = ASRFC[c2] + 1.0f;
}
