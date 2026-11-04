// phys_noah_hrtice.cu -- HRTICE: sea-ice soil thermal RHS and tri-diagonal coefficients (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:3038-3150 (nam.v4.2.11, 83df4e13c1).
//
// Source: SUBROUTINE HRTICE (RHSTS,STC,NSOIL,ZSOIL,YY,ZZ1,DF1,AI,BI,CI), called from HRT at
// module_LS_NOAHLSM.F90:3764 in the sea-ice case (ICE == 1). Every executed statement is ported.
//
// Inputs that SFLX computes per point and HRTICE needs (module_LS_NOAHLSM.F90:3363-3371, the
// sea-ice branch of SFLX), all bound per cell here:
//   YYNUM = FDOWN - EMISSI*SIGMA*T24
//   YY    = SFCTMP + (YYNUM/RCH + TH2 - SFCTMP - BETA*EPSCA)/RR
//   ZZ1   = DF1/(-0.5*ZSOIL(1)*RCH*RR) + 1.0
//   DF1   = (TDFCND DF1, VEGTYP override) * EXP(SBETA*SHDFAC)   [passed per cell as the final value]
// ZSOIL is the sea-ice vector set in SFLX (module_LS_NOAHLSM.F90:1824-1826): ZSOIL(KZ) =
// -3.*KZ/NSOIL, equal layers summing to 3 m. It is only used for ICE == 1 cells, so it is computed
// here from that formula and no land-mass vector is involved. ICE == 1 is the mask: cells with
// ICE /= 1 are not sea ice, HRTICE is never called for them in NAM, and this kernel writes 0.
//
// Constants: HCPCT = 1.72396E+6 (PARAMETER, single); TBOT = 271.16 (DATA, SAVE, single);
// SIGMA = 5.67E-8 (module PARAMETER). NSOIL = 4. Fortran REAL = float (kfpt=single).
//
// This is the sea-ice HRTICE. The ArWen noah.cu nglac_hrtice is the glacial variant (different
// HCPCT form, DF1K/DF1N, ZBOT = -25 m) and is not the NAM routine; it is not used here.
//
// Layout: one thread per (i, j) column, grid.z = 1 (CONTRACT section 3). Soil fields are
// [NSOIL, JM, IM] float32 with layer 1 (top) at index 0. Per-cell inputs are [JM, IM] float32
// except ice, which is int32. Outputs (INTENT OUT): RHSTS, AI, BI, CI.

#include "nmmb_common.cuh"

#define NAM_NSOIL_HRTICE 4
#define NAM_HRTICE_HCPCT 1.72396E+6f
#define NAM_HRTICE_TBOT  271.16f
#define NAM_HRTICE_SIGMA 5.67E-8f

extern "C" __global__ void phys_noah_hrtice(
    float* __restrict__ rhsts,         // RHSTS  RHS of soil temperature tendency (OUT) [NSOIL, JM, IM]
    const float* __restrict__ stc,     // STC    soil temperature (IN)                  [NSOIL, JM, IM]
    const int* __restrict__ ice,       // ICE    sea-ice flag, HRTICE runs only where ICE == 1 [JM, IM]
    const float* __restrict__ sfctmp,  // SFCTMP air temperature (IN)                   [JM, IM]
    const float* __restrict__ th2,     // TH2    2 m potential temperature (IN)         [JM, IM]
    const float* __restrict__ fdown,   // FDOWN  downward longwave (IN)                 [JM, IM]
    const float* __restrict__ emissi,  // EMISSI surface emissivity (IN)                [JM, IM]
    const float* __restrict__ t24,     // T24    SFCTMP**4 from SFLX (IN)               [JM, IM]
    const float* __restrict__ rch,     // RCH    RHO*CP*CH from PENMAN (IN)             [JM, IM]
    const float* __restrict__ rr,      // RR     PENMAN RR (IN)                         [JM, IM]
    const float* __restrict__ beta,    // BETA   evap factor (IN)                       [JM, IM]
    const float* __restrict__ epsca,   // EPSCA  PENMAN EPSCA (IN)                      [JM, IM]
    const float* __restrict__ df1,     // DF1    final sea-ice thermal diffusivity (IN) [JM, IM]
    float* __restrict__ ai,            // AI     tri-diagonal sub-diagonal (OUT)        [NSOIL, JM, IM]
    float* __restrict__ bi,            // BI     tri-diagonal diagonal (OUT)            [NSOIL, JM, IM]
    float* __restrict__ ci,            // CI     tri-diagonal super-diagonal (OUT)      [NSOIL, JM, IM]
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;

    const size_t cell = IDX2(i0, j0, im);
    const size_t stride = (size_t)jm * im;   // distance between soil layers

    if (ice[cell] != 1) {                    // not sea ice: HRTICE is not called in NAM
        for (int k = 0; k < NAM_NSOIL_HRTICE; ++k) {
            const size_t o = (size_t)k * stride + cell;
            rhsts[o] = 0.0f;
            ai[o] = 0.0f;
            bi[o] = 0.0f;
            ci[o] = 0.0f;
        }
        return;
    }

    // ZSOIL sea-ice vector (SFLX, module_LS_NOAHLSM.F90:1824-1826): -3.*KZ/NSOIL, KZ = 1..NSOIL
    float z[NAM_NSOIL_HRTICE], t[NAM_NSOIL_HRTICE];
    float r[NAM_NSOIL_HRTICE], a[NAM_NSOIL_HRTICE], b[NAM_NSOIL_HRTICE], c[NAM_NSOIL_HRTICE];
    for (int k = 0; k < NAM_NSOIL_HRTICE; ++k) {
        z[k] = -3.0f * (float)(k + 1) / (float)NAM_NSOIL_HRTICE;
        t[k] = stc[(size_t)k * stride + cell];
    }

    // YY and ZZ1 (SFLX, module_LS_NOAHLSM.F90:3369-3371), per cell, with the sea-ice ZSOIL(1)
    const float yynum = fdown[cell] - emissi[cell] * NAM_HRTICE_SIGMA * t24[cell];
    const float yyc = sfctmp[cell] + (yynum / rch[cell] + th2[cell] - sfctmp[cell]
                                      - beta[cell] * epsca[cell]) / rr[cell];
    const float zz1c = df1[cell] / (-0.5f * z[0] * rch[cell] * rr[cell]) + 1.0f;
    const float dfc = df1[cell];

    const float hcpct = NAM_HRTICE_HCPCT;
    const float tbot = NAM_HRTICE_TBOT;
    const float zbot = z[NAM_NSOIL_HRTICE - 1];                 // ZBOT = ZSOIL(NSOIL)

    // top layer (Fortran K = 1)
    float ddz = 1.0f / (-0.5f * z[1]);                          // DDZ = 1/(-0.5*ZSOIL(2))
    a[0] = 0.0f;                                                // AI(1) = 0
    c[0] = (dfc * ddz) / (z[0] * hcpct);                        // CI(1)
    b[0] = -c[0] + dfc / (0.5f * z[0] * z[0] * hcpct * zz1c);   // BI(1)
    float dtsdz = (t[0] - t[1]) / (-0.5f * z[1]);               // DTSDZ
    float ssoil = dfc * (t[0] - yyc) / (0.5f * z[0] * zz1c);    // SSOIL
    r[0] = (dfc * dtsdz - ssoil) / (z[0] * hcpct);              // RHSTS(1)

    // layers K = 2 .. NSOIL (zero-based k = 1 .. NSOIL-1)
    float ddz2 = 0.0f;                                          // DDZ2 = 0
    for (int k = 1; k < NAM_NSOIL_HRTICE; ++k) {
        float dtsdz2;
        if (k != NAM_NSOIL_HRTICE - 1) {                        // K /= NSOIL
            const float denom = 0.5f * (z[k - 1] - z[k + 1]);
            dtsdz2 = (t[k] - t[k + 1]) / denom;
            ddz2 = 2.0f / (z[k - 1] - z[k + 1]);
            c[k] = -(dfc * ddz2 / ((z[k - 1] - z[k]) * hcpct)); // CI(K)
        } else {                                                // K == NSOIL: TBOT boundary
            dtsdz2 = (t[k] - tbot) / (0.5f * (z[k - 1] + z[k]) - zbot);
            c[k] = 0.0f;                                        // CI(NSOIL) = 0
        }
        const float denom2 = (z[k] - z[k - 1]) * hcpct;
        r[k] = (dfc * dtsdz2 - dfc * dtsdz) / denom2;           // RHSTS(K)
        a[k] = -(dfc * ddz / ((z[k - 1] - z[k]) * hcpct));      // AI(K)
        b[k] = -(a[k] + c[k]);                                  // BI(K)
        dtsdz = dtsdz2;
        ddz = ddz2;
    }

    for (int k = 0; k < NAM_NSOIL_HRTICE; ++k) {
        const size_t o = (size_t)k * stride + cell;
        rhsts[o] = r[k];
        ai[o] = a[k];
        bi[o] = b[k];
        ci[o] = c[k];
    }
}
