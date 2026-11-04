// phys_noah_snksrc.cu -- SNKSRC: phase-change sink/source term of the soil heat equation (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the source
// module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90), HISTORY LOG 2008-07-28
// Vasic. The routine's own comment block (Koren et al. 1999, JGR 104(D16), eqns 17-18) is kept in
// the Fortran source.
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:3960-4047 (nam.v4.2.11, 83df4e13c1).
// Start point: namgpu/third_party/arwen/core/kernels/noah.cu noah_snksrc (Apache-2.0, ArWen), checked
// statement by statement against the Fortran. FRH2O is the device function in phys_noah_frh2o.cuh.
//
// Fortran SUBROUTINE SNKSRC(TSNSR,TAVG,SMC,SH2O,ZSOIL,NSOIL,SMCMAX,PSISAT,BEXP,DT,K,QTOT),
// every executed statement, for soil layer K (1-based; k0 = K-1 here):
//   DZ = -ZSOIL(1)                       if K == 1
//   DZ = ZSOIL(K-1) - ZSOIL(K)           otherwise
//   FREE = FRH2O(TAVG, SMC, SH2O, SMCMAX, BEXP, PSISAT)
//   XH2O = SH2O + QTOT*DT / (DH2O*HLICE*DZ)
//   IF (XH2O < SH2O .AND. XH2O < FREE):  XH2O = SH2O if FREE > SH2O, else FREE
//   IF (XH2O > SH2O .AND. XH2O > FREE):  XH2O = SH2O if FREE < SH2O, else FREE
//   IF (XH2O < 0)   XH2O = 0
//   IF (XH2O > SMC) XH2O = SMC
//   TSNSR = -DH2O*HLICE*DZ*(XH2O - SH2O) / DT
//   SH2O  = XH2O
// Constants (PARAMETER): DH2O = 1.0E3, HLICE = 3.335E5.
//
// ZSOIL is COLUMN DEPENDENT through ICE. module_LS_NOAHLSM.F90:1822-1827 (SFLX): on a sea-ice column
// (ICE == 1) ZSOIL(KZ) = -3*KZ/NSOIL (three metres in equal layers); otherwise ZSOIL(1) = -SLDPTH(1)
// and the layers are cumulative (NAM: -0.1, -0.4, -1.0, -2.0). This kernel evaluates ZSOIL per column
// from the ICE flag, so both column types are right. The non-ice depths come in as the shared
// [NSOIL] vector, which is what the NAM SLDPTH gives.
//
// The call-site gate is ported in-kernel. Fortran calls SNKSRC only when the layer's conditions hold
// (module_LS_NOAHLSM.F90 ~2895-2905 for the top layer K = 1, ~3007-3010 for K = 2..NSOIL, with ITAVG
// = .TRUE. at line 2828, the only live path):
//   top layer:    (SICE > 0) .OR. (STC(1) < T0) .OR. (TSURF < T0) .OR. (TBND(1) < T0)
//   layer K >= 2: (SMC(K)-SH2O(K) > 0) .OR. (STC(K) < T0) .OR. (TBND(K-1) < T0) .OR. (TBND(K) < T0)
// where TBND(k) is the TBND interface temperature below 0-based layer k (phys-noah-tbnd output),
// TSURF = (YY + (ZZ1-1)*STC(1))/ZZ1 (SFLX), and SICE of the top layer is SMC(1)-SH2O(1). When the
// gate is false the Fortran does not call SNKSRC: TSNSR is not used, SH2O is left unchanged, and
// RHSTS takes no correction. This kernel writes TSNSR = 0 and leaves SH2O unchanged in that case,
// which is the same update. The temperature-average branch (TMPAVG) is the one the caller computes;
// the STC-variant (ITAVG = .FALSE., lines ~2903 and ~3014) is dead in NAM and is not ported.
//
// Inputs are per soil layer, as the Fortran call receives them: TAVG (the TMPAVG of the layer) and
// QTOT (the heat flux from HRT for that layer). SH2O is updated in place; each thread touches only
// its own layer. SMCMAX, BEXP and PSISAT are per column (REDPRM). T0 = 273.15 K.
//
// Layout: 3-D point kernel, one thread per (i, j, soil layer), soil layer on grid.z (grid_2d with
// lm = NSOIL). Soil arrays are [NSOIL, JM, IM] with layer 1 at the top (CONTRACT section 1).
// NAM_NSOIL = 4 is fixed here; the wrapper refuses any other NSOIL.
// Float32 throughout (Fortran REAL = float, kfpt = single). No host interaction.

#include "nmmb_common.cuh"
#include "phys_noah_frh2o.cuh"

#define NAM_NSOIL 4
#define NAM_SNKSRC_DH2O  1.0000E3f
#define NAM_SNKSRC_HLICE 3.3350E5f
#define NAM_SNKSRC_T0    273.15f

// ZSOIL(k0+1) for this column (SFLX, module_LS_NOAHLSM.F90:1822-1827).
__device__ __forceinline__ float nam_zsoil(const float* zsoil_nam, int ice, int k0)
{
    if (ice == 1) return -3.0f * (float)(k0 + 1) / (float)NAM_NSOIL;
    return zsoil_nam[k0];
}

// The SNKSRC body for one layer, given the layer thickness DZ (from ZSOIL of this column).
__device__ __forceinline__ float nam_snksrc(float qtot, float tavg, float smc,
                                            float& sh2o, float dz,
                                            float smcmax, float psisat, float bexp,
                                            float dt)
{
    const float dh2o = NAM_SNKSRC_DH2O;
    const float hlice = NAM_SNKSRC_HLICE;

    const float freev = nam_frh2o(tavg, smc, sh2o, smcmax, bexp, psisat);

    float xh2o = sh2o + qtot * dt / (dh2o * hlice * dz);
    if (xh2o < sh2o && xh2o < freev) {
        if (freev > sh2o) xh2o = sh2o;
        else              xh2o = freev;
    }
    if (xh2o > sh2o && xh2o > freev) {
        if (freev < sh2o) xh2o = sh2o;
        else              xh2o = freev;
    }
    if (xh2o < 0.0f) xh2o = 0.0f;
    if (xh2o > smc)  xh2o = smc;

    const float tsnsr = -dh2o * hlice * dz * (xh2o - sh2o) / dt;
    sh2o = xh2o;
    return tsnsr;
}

extern "C" __global__ void phys_noah_snksrc(
    const float* __restrict__ tavg,     // TAVG   soil layer temperature from TMPAVG (K)  [NSOIL, JM, IM]
    const float* __restrict__ qtot,     // QTOT   heat flux into the layer from HRT (W m-2) [NSOIL, JM, IM]
    const float* __restrict__ smc,      // SMC    total soil moisture                     [NSOIL, JM, IM]
    float* __restrict__ sh2o,           // SH2O   unfrozen soil moisture, updated in place [NSOIL, JM, IM]
    const float* __restrict__ stc,      // STC    soil temperature (gate)                 [NSOIL, JM, IM]
    const float* __restrict__ tbnd,     // TBND   interface temperature below layer k (gate) [NSOIL, JM, IM]
    const float* __restrict__ tsurf,    // TSURF  soil column top temperature (gate)      [JM, IM]
    const float* __restrict__ zsoil,    // ZSOIL  non-ice layer bottom depths (m, negative) [NSOIL]
    const int* __restrict__ ice,        // ICE    sea-ice flag (1 = ice column)           [JM, IM]
    const float* __restrict__ smcmax,   // SMCMAX saturation moisture (REDPRM)            [JM, IM]
    const float* __restrict__ bexp,     // BEXP   soil B parameter (REDPRM)               [JM, IM]
    const float* __restrict__ psis,     // PSISAT saturated matric potential (REDPRM)     [JM, IM]
    float dt,                           // DT     SNKSRC time step: DTBL = DT*NPHS = 50 s on NAM 12 km
    float* __restrict__ tsnsr,          // TSNSR  phase-change heat source/sink (W m-3)   [NSOIL, JM, IM] (output)
    int im, int jm)
{
    POINT_IJL                              // l0 = soil layer (blockIdx.z)
    if (i0 >= im || j0 >= jm || l0 >= NAM_NSOIL) return;
    const size_t c = IDX2(i0, j0, im);
    const size_t n = IDX3(i0, j0, l0, im, jm);

    // Call-site gate (the Fortran CALL SNKSRC condition), evaluated with the pre-update SH2O.
    const float t0 = NAM_SNKSRC_T0;
    const float sice = smc[n] - sh2o[n];
    const float tbk_below = tbnd[n];                         // TBND below this layer
    bool gate;
    if (l0 == 0) {
        gate = (sice > 0.0f) || (stc[n] < t0) || (tsurf[c] < t0) || (tbk_below < t0);
    } else {
        const float tbk_above = tbnd[IDX3(i0, j0, l0 - 1, im, jm)];
        gate = (sice > 0.0f) || (stc[n] < t0) || (tbk_above < t0) || (tbk_below < t0);
    }
    if (!gate) {
        tsnsr[n] = 0.0f;                   // Fortran does not call SNKSRC: SH2O unchanged
        return;
    }

    // Layer thickness from this column's ZSOIL (ICE-aware).
    const int ic = ice[c];
    float dz;
    if (l0 == 0) dz = -nam_zsoil(zsoil, ic, 0);
    else         dz = nam_zsoil(zsoil, ic, l0 - 1) - nam_zsoil(zsoil, ic, l0);

    tsnsr[n] = nam_snksrc(qtot[n], tavg[n], smc[n], sh2o[n], dz,
                          smcmax[c], psis[c], bexp[c], dt);
}
