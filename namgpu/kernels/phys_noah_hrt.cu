// phys_noah_hrt.cu -- HRT: soil heat-equation right-hand side and tridiagonal coefficients (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:2785-3035 (nam.v4.2.11, 83df4e13c1).
// The helpers FRH2O, TBND, TDFCND, TMPAVG and SNKSRC come from noah_soil_helpers.cuh (single source,
// shared with phys-noah-tbnd and phys-noah-tmpavg).
//
// Start point and notice: the column structure and the top-layer/middle-layer formulas were
// transcribed from ArWen core/kernels/noah.cu noah_hrt (Apache-2.0, third_party/arwen/LICENSE),
// which transcribes WRF's Noah LSM. The ArWen Apache-2.0 notice applies to that transcribed text.
//
// Fortran HRT(RHSTS,STC,SMC,SMCMAX,NSOIL,ZSOIL,YY,ZZ1,TBOT,ZBOT,PSISAT,SH2O,DT,BEXP,F1,DF1,
// QUARTZ,CSOIL,AI,BI,CI,VEGTYP,IVEGSRC). F1 is not referenced in the routine and is not an
// argument here. ITAVG is set to .TRUE. at line 2817 and never changed, so the ITAVG=.FALSE.
// branches are dead; they are kept below under `if (!itavg)` so every path of the routine has
// a counterpart.
// Fortran 1-based layer K maps to CUDA index k = K-1 (0..3). Fortran "K.EQ.1" tests become k == 0,
// "K.EQ.NSOIL" becomes k == NSOIL-1. ZSOIL(K) is zsoil[k].
//
// Layout: soil fields [NSOIL, JM, IM] (CONTRACT section 1, layer 1 at the top). Per-column
// inputs (SMCMAX, YY, ZZ1, TBOT, ZBOT, PSISAT, BEXP, DF1, QUARTZ, CSOIL, VEGTYP) are [JM, IM].
// ZSOIL is the shared [NSOIL] layer depth vector. SH2O is INTENT(INOUT): updated in place by SNKSRC.
//
// Sea ice (SFLX ICE == 1, module_LS_NOAHLSM.F90:3762-3764): Fortran calls HRTICE, not HRT, for those
// columns and leaves SH2O alone. This kernel takes the ICE mask and returns at once for ICE == 1
// cells, so their RHSTS/AI/BI/CI/SH2O are not written here. The wrapper runs phys_noah_hrtice first
// (it writes the ICE == 1 cells), then this kernel fills the rest.
//
// One thread per column (grid_cols), one launch, no host sync.
// Float32 throughout (kfpt=single). Fast numerics (fmad on, IEEE division).

#include "nmmb_common.cuh"
#include "noah_soil_helpers.cuh"

#define NAM_HRT_NSOIL 4
#define NAM_HRT_T0      273.15f     // HRT / module T0
#define NAM_HRT_CAIR    1004.0f     // HRT CAIR
#define NAM_HRT_CICE    2.106e6f    // HRT CICE
#define NAM_HRT_CH2O    4.2e6f      // HRT CH2O

// ---------------------------------------------------------------- HRT (2785-3035)
// One soil column. Fortran K = 2..NSOIL is 0-based k = 1..nsoil-1 after the top layer.
__device__ static void nam_hrt_column(float* rhsts, const float* stc, const float* smc,
                                      float smcmax, const float* zsoil, float yy, float zz1,
                                      float tbot, float zbot, float psisat, float* sh2o,
                                      float dt, float bexp, float df1, float quartz,
                                      float csoil, int vegtyp, int ivegsrc,
                                      float* ai, float* bi, float* ci)
{
    const int nsoil = NAM_HRT_NSOIL;
    const float t0 = NAM_HRT_T0, cair = NAM_HRT_CAIR, cice = NAM_HRT_CICE, ch2o = NAM_HRT_CH2O;
    const int itavg = 1;                       // ITAVG = .TRUE. (line 2817), never changed

    // urban switch (2805-2814): IW = 13 for IVEGSRC == 1, else 1
    int iw = 1;
    if (ivegsrc == 1) iw = 13;
    float csoil_loc;
    if (vegtyp == iw) csoil_loc = 3.0e6f;
    else csoil_loc = csoil;

    // -------- top soil layer (K = 1) --------
    float hcpct = sh2o[0] * ch2o + (1.0f - smcmax) * csoil_loc
                  + (smcmax - smc[0]) * cair + (smc[0] - sh2o[0]) * cice;
    float ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct);
    bi[0] = -ci[0] + df1 / (0.5f * zsoil[0] * zsoil[0] * hcpct * zz1);
    float dtsdz = (stc[0] - stc[1]) / (-0.5f * zsoil[1]);
    float ssoil = df1 * (stc[0] - yy) / (0.5f * zsoil[0] * zz1);
    float denom = zsoil[0] * hcpct;
    rhsts[0] = (df1 * dtsdz - ssoil) / denom;
    float qtot = -1.0f * rhsts[0] * denom;
    float tsnsr;
    float sice = smc[0] - sh2o[0];
    float tsurf = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    float tbk = nam_tbnd(stc[0], stc[1], zsoil, zbot, 0);
    float tavg;
    if (itavg) {
        if ((sice > 0.0f) || (stc[0] < t0) || (tsurf < t0) || (tbk < t0)) {
            tavg = nam_tmpavg(tsurf, stc[0], tbk, zsoil, 0);
            tsnsr = nam_snksrc(qtot, tavg, smc[0], sh2o[0], zsoil, smcmax, psisat, bexp, dt, 0);
            rhsts[0] = rhsts[0] - tsnsr / denom;
        }
    } else {
        if ((sice > 0.0f) || (stc[0] < t0)) {
            tsnsr = nam_snksrc(qtot, stc[0], smc[0], sh2o[0], zsoil, smcmax, psisat, bexp, dt, 0);
            rhsts[0] = rhsts[0] - tsnsr / denom;
        }
    }

    // -------- layers K = 2 .. NSOIL (0-based k = 1 .. nsoil-1) --------
    float ddz2 = 0.0f;
    float df1k = df1;
    float dtsdz2 = 0.0f;
    float df1n = 0.0f;
    float tbk1 = 0.0f;
    for (int k = 1; k < nsoil; ++k) {
        hcpct = sh2o[k] * ch2o + (1.0f - smcmax) * csoil_loc
                + (smcmax - smc[k]) * cair + (smc[k] - sh2o[k]) * cice;
        if (k != nsoil - 1) {                  // Fortran K /= NSOIL: middle layers
            df1n = nam_tdfcnd(smc[k], quartz, smcmax, sh2o[k]);
            if (vegtyp == iw) df1n = 3.24f;
            denom = 0.5f * (zsoil[k - 1] - zsoil[k + 1]);
            dtsdz2 = (stc[k] - stc[k + 1]) / denom;
            ddz2 = 2.0f / (zsoil[k - 1] - zsoil[k + 1]);
            ci[k] = -df1n * ddz2 / ((zsoil[k - 1] - zsoil[k]) * hcpct);
            if (itavg) tbk1 = nam_tbnd(stc[k], stc[k + 1], zsoil, zbot, k);
        } else {                               // bottom layer (K = NSOIL)
            df1n = nam_tdfcnd(smc[k], quartz, smcmax, sh2o[k]);
            if (vegtyp == iw) df1n = 3.24f;
            denom = 0.5f * (zsoil[k - 1] + zsoil[k]) - zbot;
            dtsdz2 = (stc[k] - tbot) / denom;
            ci[k] = 0.0f;
            if (itavg) tbk1 = nam_tbnd(stc[k], tbot, zsoil, zbot, k);
        }
        denom = (zsoil[k] - zsoil[k - 1]) * hcpct;
        rhsts[k] = (df1n * dtsdz2 - df1k * dtsdz) / denom;
        qtot = -1.0f * denom * rhsts[k];
        sice = smc[k] - sh2o[k];
        if (itavg) {
            tavg = nam_tmpavg(tbk, stc[k], tbk1, zsoil, k);
            if ((sice > 0.0f) || (stc[k] < t0) || (tbk < t0) || (tbk1 < t0)) {
                tsnsr = nam_snksrc(qtot, tavg, smc[k], sh2o[k], zsoil, smcmax, psisat, bexp, dt, k);
                rhsts[k] = rhsts[k] - tsnsr / denom;
            }
        } else {
            if ((sice > 0.0f) || (stc[k] < t0)) {
                tsnsr = nam_snksrc(qtot, stc[k], smc[k], sh2o[k], zsoil, smcmax, psisat, bexp, dt, k);
                rhsts[k] = rhsts[k] - tsnsr / denom;
            }
        }
        ai[k] = -df1k * ddz / ((zsoil[k - 1] - zsoil[k]) * hcpct);
        bi[k] = -(ai[k] + ci[k]);
        tbk = tbk1;
        df1k = df1n;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

extern "C" __global__ void phys_noah_hrt(
    float* __restrict__ rhsts,             // RHSTS  [NSOIL, JM, IM] (OUT)
    const float* __restrict__ stc,         // STC    [NSOIL, JM, IM] (IN)
    const float* __restrict__ smc,         // SMC    [NSOIL, JM, IM] (IN)
    float* __restrict__ sh2o,              // SH2O   [NSOIL, JM, IM] (INOUT)
    const float* __restrict__ smcmax,      // SMCMAX [JM, IM]
    const float* __restrict__ zsoil,       // ZSOIL  [NSOIL] layer depths, shared (IN)
    const float* __restrict__ yy,          // YY     [JM, IM]
    const float* __restrict__ zz1,         // ZZ1    [JM, IM]
    const float* __restrict__ tbot,        // TBOT   [JM, IM]
    const float* __restrict__ zbot,        // ZBOT   [JM, IM]
    const float* __restrict__ psisat,      // PSISAT [JM, IM]
    const float* __restrict__ bexp,        // BEXP   [JM, IM]
    const float* __restrict__ df1,         // DF1    [JM, IM]
    const float* __restrict__ quartz,      // QUARTZ [JM, IM]
    const float* __restrict__ csoil,       // CSOIL  [JM, IM]
    const int* __restrict__ vegtyp,        // VEGTYP [JM, IM]
    const int* __restrict__ ice,           // ICE    [JM, IM] (IN); ICE == 1 cells belong to HRTICE
    float* __restrict__ ai,                // AI     [NSOIL, JM, IM] (OUT)
    float* __restrict__ bi,                // BI     [NSOIL, JM, IM] (OUT)
    float* __restrict__ ci,                // CI     [NSOIL, JM, IM] (OUT)
    int ivegsrc,                           // IVEGSRC (IN)
    float dt,                              // DT (IN)
    int im, int jm)
{
    COLUMN_IJ(im, jm)
    const size_t n2 = IDX2(i0, j0, im);
    if (ice[n2] == 1) return;              // sea ice column: HRTICE owns it (SFLX 3762-3764)
    float rh[NAM_HRT_NSOIL], a[NAM_HRT_NSOIL], b[NAM_HRT_NSOIL], c[NAM_HRT_NSOIL];
    float w[NAM_HRT_NSOIL];                // local copy of SH2O for the in-place SNKSRC update
    float s[NAM_HRT_NSOIL], m[NAM_HRT_NSOIL];
    for (int k = 0; k < NAM_HRT_NSOIL; ++k) {
        const size_t nk = IDX3(i0, j0, k, im, jm);
        s[k] = stc[nk];
        m[k] = smc[nk];
        w[k] = sh2o[nk];
    }
    nam_hrt_column(rh, s, m, smcmax[n2], zsoil, yy[n2], zz1[n2], tbot[n2], zbot[n2],
                   psisat[n2], w, dt, bexp[n2], df1[n2], quartz[n2], csoil[n2],
                   vegtyp[n2], ivegsrc, a, b, c);
    for (int k = 0; k < NAM_HRT_NSOIL; ++k) {
        const size_t nk = IDX3(i0, j0, k, im, jm);
        rhsts[nk] = rh[k];
        sh2o[nk] = w[k];
        ai[nk] = a[k];
        bi[nk] = b[k];
        ci[nk] = c[k];
    }
}
