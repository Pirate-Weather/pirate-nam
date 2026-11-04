// nam-gpu unit phys-jsfc-jsfc: JSFC surface-layer driver (Janjic), one thread per surface column.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module and routine kept as the notice:
//   module module_SF_JSFC  (NMMB, nam.v4.2.11)
//   subroutine JSFC        (driver: vertical setup, PBLH, call to SFCDIF, 2 m / 10 m supersaturation removal)
// Ported to CUDA for nam-gpu from phys/module_SF_JSFC.F90:80-354 (nam.v4.2.11, 83df4e13c1).
//
// Apache-2.0 portions: the shelter supersaturation block and the surface-state layout follow
// ArWen third_party/arwen/core/kernels/myjsfc.cu (Apache-2.0, see third_party/arwen/LICENSE).
// That ArWen kernel is WRF's MYJ surface layer, not this NMMB routine, so its MYJ similarity code
// is not carried over.
//
// JSFC runs as three launches on one stream, with no host sync between them:
//   1. phys_jsfc_jsfc_pre      (this file)  :99-178 preparations, LPBL/PBLH, lowest layer, TZ0
//   2. phys_jsfc_sfcdif_column (phys-jsfc-sfcdif, phys_jsfc_sfcdif.cu)  CALL SFCDIF, :181-200
//   3. phys_jsfc_jsfc_post     (this file)  :203-220 TH02/T02/Q02 and 2 m / 10 m supersaturation removal
// The wrapper (phys_jsfc_jsfc.py) binds the SFCDIF outputs to the same JSFC state the Fortran
// CALL SFCDIF arguments name (TH02 -> TSHLTR, Q02 -> QSHLTR, TH10, Q10, PSHLTR).
//
// Tile: JSFC runs over ITS_B1..ITE_B1 x JTS_B1..JTE_B1 (1-based; module_TURBULENCE.F90 passes
// ITS_B1,ITE_B1,JTS_B1,JTE_B1 with ITS_B1 = MAX(ITS,IDS+1), ITE_B1 = MIN(ITE,IDE-1)). Every stage
// reads the tile from ctx.tables["NMMB_LIMITS"] and leaves columns outside it untouched, as Fortran does.
//
// Layout (CONTRACT.md sections 1 and 3): Fortran A(i,j,l) is A[(l-1)*S + c], c = (j-1)*IM + (i-1),
// S = IM*JM. Per-column scalars are [JM,IM] at index c. PHINT has LM+1 levels, PHINT(LM+1) is the
// surface pressure. One thread per column (kernels.grid_cols); no host sync, no device allocation,
// no printf.
//
// Constants: module_CONSTANTS (cp 1004.6, g 9.8060226, elwv 2.501e6, r_d 287.04, pq0, a2, a3 273.15,
// a4 35.86, cappa = r_d/cp) via nmmb_common.cuh; FH = 1.10 and ELOCP = 2.72e6/CP from the JSFC module.

#include "nmmb_common.cuh"

#define JSFC_FH        1.10f
#define JSFC_ELOCP     (2.72e6f / NMMB_CP)
#define JSFC_GOCP02    (NMMB_G / NMMB_CP * 2.0f)
#define JSFC_GOCP10    (NMMB_G / NMMB_CP * 10.0f)
#define JSFC_RCAP      (1.0f / NMMB_CAPPA)

// True when the 0-based column (i0, j0) lies in the Fortran B1 tile (1-based limits from lim).
__device__ __forceinline__ bool jsfc_in_tile(const int* __restrict__ lim, int i0, int j0)
{
    const int I = i0 + 1;
    const int J = j0 + 1;
    return I >= lim[DL_ITS_B1] && I <= lim[DL_ITE_B1] && J >= lim[DL_JTS_B1] && J <= lim[DL_JTE_B1];
}

// ---- stage 1: :99-178 ----
extern "C" __global__ void phys_jsfc_jsfc_pre(
    int IM, int JM, int LM, int ntsd, const int* __restrict__ lim,
    const float* __restrict__ EPSQ2,      // [LM]
    const float* __restrict__ HT,         // [JM,IM]  Fortran caller SFCZ
    const float* __restrict__ TSK,        // [JM,IM]  Fortran caller TSFC
    const float* __restrict__ XLAND,      // [JM,IM]
    const float* __restrict__ DZ,         // [LM,JM,IM]
    const float* __restrict__ PHMID,      // [LM,JM,IM]
    const float* __restrict__ PHINT,      // [LM+1,JM,IM]
    const float* __restrict__ TH,         // [LM,JM,IM]
    const float* __restrict__ T,          // [LM,JM,IM]
    const float* __restrict__ Q,          // [LM,JM,IM]
    const float* __restrict__ QC,         // [LM,JM,IM]
    const float* __restrict__ U,          // [LM,JM,IM]  U_PHY (H points)
    const float* __restrict__ V,          // [LM,JM,IM]  V_PHY (H points)
    const float* __restrict__ Q2,         // [LM,JM,IM]
    const float* __restrict__ THZ0,       // [JM,IM]  INOUT in JSFC, read here for TZ0
    float* __restrict__ USTAR,            // [JM,IM]  INOUT, NTSD==0 reset
    float* __restrict__ PBLH,             // [JM,IM]  INOUT, written here
    float* __restrict__ CT,               // [JM,IM]  CT=0 (:122-124)
    float* __restrict__ SEAMASK,          // scratch for SFCDIF
    float* __restrict__ THSK,             // scratch for SFCDIF
    float* __restrict__ PSFC_S,           // scratch for SFCDIF, reused by stage 3
    float* __restrict__ TZ0,              // scratch for SFCDIF
    float* __restrict__ ZSL,              // scratch for SFCDIF
    float* __restrict__ ULOW,             // scratch for SFCDIF
    float* __restrict__ VLOW,             // scratch for SFCDIF
    float* __restrict__ TLOW,             // scratch for SFCDIF
    float* __restrict__ THLOW,            // scratch for SFCDIF
    float* __restrict__ THELOW,           // scratch for SFCDIF
    float* __restrict__ QLOW,             // scratch for SFCDIF
    float* __restrict__ CWMLOW,           // scratch for SFCDIF
    float* __restrict__ PLOW)             // scratch for SFCDIF
{
    const size_t S = (size_t)IM * (size_t)JM;
    const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= S) return;
    const int j0 = (int)(c / (size_t)IM);
    const int i0 = (int)(c - (size_t)j0 * IM);
    if (!jsfc_in_tile(lim, i0, j0)) return;

    // Fortran 1-based level k -> offset (k-1)*S + c
#define LV(a, k)  a[((size_t)((k) - 1)) * S + c]

    // ---- :99-124 preparations ----
    const float ht = HT[c];
    const float tsk = TSK[c];
    const float psfc = PHINT[(size_t)LM * S + c];       // PHINT(I,J,LMH+1)
    const float thsk = tsk / powf(psfc * 1.0e-5f, NMMB_CAPPA);
    const float apesfc = powf(psfc * 1.0e-5f, NMMB_CAPPA);
    if (ntsd == 0) USTAR[c] = 0.1f;                     // :119-121
    CT[c] = 0.0f;                                       // :122-124

    // ---- :143-168 height of layer interfaces and PBL height ----
    // ZINT(K) = ZINT(K+1) + DZ(K) from K=LM down (same float ops as Fortran), ZINT(LM+1) = HT.
    // The LPBL scan is K = LM-1 .. 1 and takes the first K with Q2(K) <= EPSQ2(K)*FH, which in
    // this descending loop is the first hit recorded.
    float z = ht;
    float zlm = 0.0f;
    float zlpbl = 0.0f;
    int lpbl_found = 0;
    for (int k = LM; k >= 1; --k) {
        z = z + LV(DZ, k);
        if (k == LM) zlm = z;
        if (!lpbl_found && k <= LM - 1 && LV(Q2, k) <= EPSQ2[k - 1] * JSFC_FH) {
            lpbl_found = 1;
            zlpbl = z;
        }
    }
    if (!lpbl_found) zlpbl = z;                         // LPBL=1: ZHK(1) = ZINT(1)
    const float pblh = zlpbl - ht;                      // :110 PBLH = ZHK(LPBL) - ZHK(LMH+1)
    PBLH[c] = pblh;

    // ---- :170-178 lowest layer ----
    const float plow = LV(PHMID, LM);
    const float tlow = LV(T, LM);
    const float thlow = LV(TH, LM);
    const float qlow = LV(Q, LM);
    const float cwmlow = LV(QC, LM);
    const float thelow = (cwmlow * (-JSFC_ELOCP / tlow) + 1.0f) * thlow;
    const float zsl = (zlm - ht) * 0.5f;                // (ZHK(LMH)-ZHK(LMH+1))*0.5

    // :292-299 TZ0 (uses THZ0 before SFCDIF writes it)
    const float tz0 = (ntsd == 0) ? tsk : THZ0[c] * apesfc;

    SEAMASK[c] = XLAND[c] - 1.0f;                       // SM=XLAND-1
    THSK[c] = thsk;
    PSFC_S[c] = psfc;
    TZ0[c] = tz0;
    ZSL[c] = zsl;
    ULOW[c] = LV(U, LM);
    VLOW[c] = LV(V, LM);
    TLOW[c] = tlow;
    THLOW[c] = thlow;
    THELOW[c] = thelow;
    QLOW[c] = qlow;
    CWMLOW[c] = cwmlow;
    PLOW[c] = plow;
#undef LV
}

// ---- stage 3: :203-220 ----
extern "C" __global__ void phys_jsfc_jsfc_post(
    int IM, int JM, const int* __restrict__ lim,
    const float* __restrict__ PSFC_S,     // [JM,IM]  from stage 1
    const float* __restrict__ TSHLTR,     // [JM,IM]  SFCDIF TH02
    const float* __restrict__ TH10,       // [JM,IM]  SFCDIF TH10
    float* __restrict__ QSHLTR,           // [JM,IM]  SFCDIF Q02 in, clamped out
    float* __restrict__ Q10,              // [JM,IM]  SFCDIF Q10 in, clamped out
    float* __restrict__ TH2X,             // [JM,IM]  TH02 = TSHLTR
    float* __restrict__ T2,               // [JM,IM]  T02 = TH02*APESFC
    float* __restrict__ Q2X)              // [JM,IM]  Q02 = QSHLTR/(1-QSHLTR)
{
    const size_t S = (size_t)IM * (size_t)JM;
    const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= S) return;
    const int j0 = (int)(c / (size_t)IM);
    const int i0 = (int)(c - (size_t)j0 * IM);
    if (!jsfc_in_tile(lim, i0, j0)) return;

    const float psfc = PSFC_S[c];
    const float apesfc = powf(psfc * 1.0e-5f, NMMB_CAPPA);
    const float th02 = TSHLTR[c];                       // TH02(I,J)=TSHLTR(I,J)
    const float th10 = TH10[c];
    float qshltr = QSHLTR[c];
    float q10 = Q10[c];

    const float rapa = apesfc;
    const float rapa02 = rapa - JSFC_GOCP02 / th02;
    const float rapa10 = rapa - JSFC_GOCP10 / th10;
    const float t02p = th02 * rapa02;
    const float t10p = th10 * rapa10;
    const float p02p = powf(rapa02, JSFC_RCAP) * 1.0e5f;
    const float p10p = powf(rapa10, JSFC_RCAP) * 1.0e5f;
    const float qs02 = NMMB_PQ0 / p02p * expf(NMMB_A2 * (t02p - NMMB_A3) / (t02p - NMMB_A4));
    const float qs10 = NMMB_PQ0 / p10p * expf(NMMB_A2 * (t10p - NMMB_A3) / (t10p - NMMB_A4));
    if (qshltr > qs02) qshltr = qs02;
    if (q10 > qs10) q10 = qs10;
    const float q02 = qshltr / (1.0f - qshltr);         // Q02(I,J)=QSHLTR/(1-QSHLTR)

    QSHLTR[c] = qshltr;
    Q10[c] = q10;
    TH2X[c] = th02;
    T2[c] = th02 * apesfc;                              // T02(I,J)=TH02(I,J)*APESFC
    Q2X[c] = q02;
}
