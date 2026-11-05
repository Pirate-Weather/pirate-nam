// io_history_fields.cu -- wrfout-shaped history fields from the NMMB state (unit io-history-fields,
// mode "new": no Fortran counterpart; the mapping follows the NAM sources named below).
//
// Ported to CUDA for WOOF-NAM from nam.v4.2.11 (83df4e13c1) sorc/nam_nems_nmmb.fd/src/atmos/nmm:
//   module_SOLVER_GRID_COMP.F90:11144-11186  UPDATE_WATER, CASE ('fer','fer_hires'), spec_adv=F branch:
//                                            QC=(1-F_RAIN)*(1-F_ICE)*CWM, QR=F_RAIN*(1-F_ICE)*CWM,
//                                            QS=F_ICE*CWM, all zero when CWM <= EPSQ. QI is not set.
//   Z is MIDLAYER height (module_DYNAMICS_ROUTINES.F90:3075, z(i,j,l)=zlo+dz/2; dyn-cdzdt-nonhydro writes it
//   every step). Z[LM] is not used: the surface interface is FIS/g. See io_history_fields.py header.
//   Constants: NMMB_G (module_CONSTANTS g), CAPPA = R_D/CP, EPSQ (nmmb_common.cuh). P00 = 1.0e5 Pa is the
//   WRF theta reference (not an NMMB constant; used only for the output theta).
//   Pressure at mid layers: PMID(l) = PSGML1(l) + SGML2(l)*PD (grid.py psgml1, same as PGFORCE), PINT as state.
//
// Layout: state arrays are [l, j, i] (interfaces LM+1), PD/FIS/ACPREC_TOT/CUPREC_TOT/UE10/VE10 are [j, i].
// Mixing ratios: NMMB Q and CW are specific (kg/kg); WRF outputs are mixing ratios, so each moisture
// output is x/(1-Q). Vapour and condensate species use the same factor.
#include "nmmb_common.cuh"

#ifndef NMMB_P00
#define NMMB_P00 100000.0f
#endif

// Mid-layer fields: pressure, theta (written as theta-300, WRF's T), vapour and condensate mixing ratios.
// QICE is zero under fer_hires (ice is all QS, see UPDATE_WATER above); kept so the output set is WRF's.
extern "C" __global__ void hist_moist_theta(
    int IM, int JM, int LM,
    const float* __restrict__ T, const float* __restrict__ Q, const float* __restrict__ CW,
    const float* __restrict__ F_ICE, const float* __restrict__ F_RAIN,
    const float* __restrict__ PD,
    const float* __restrict__ PSGML1, const float* __restrict__ SGML2,
    float* __restrict__ P_OUT, float* __restrict__ T_OUT,
    float* __restrict__ QVAPOR, float* __restrict__ QCLOUD, float* __restrict__ QRAIN,
    float* __restrict__ QICE, float* __restrict__ QSNOW)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t n = IDX3(i0, j0, l0, IM, JM);
    const size_t n2 = IDX2(i0, j0, IM);

    const float pmid = PSGML1[l0] + SGML2[l0] * PD[n2];
    const float t = T[n];
    const float q = Q[n];
    const float cw = CW[n];
    const float fi = F_ICE[n];
    const float fr = F_RAIN[n];
    const float inv = 1.0f / (1.0f - q);

    const float theta = t * powf(NMMB_P00 / pmid, NMMB_CAPPA);

    float qc = 0.0f, qr = 0.0f, qs = 0.0f;
    if (cw > NMMB_EPSQ) {
        const float liqw = (1.0f - fi) * cw;
        qc = (1.0f - fr) * liqw;
        qr = fr * liqw;
        qs = fi * cw;
    }

    P_OUT[n]   = pmid;
    T_OUT[n]   = theta - 300.0f;
    QVAPOR[n]  = q * inv;
    QCLOUD[n]  = qc * inv;
    QRAIN[n]   = qr * inv;
    QICE[n]    = 0.0f;
    QSNOW[n]   = qs * inv;
}

// hist_moist_theta with one output per launch (io_history_fields.StreamField): the same expressions, stored only
// for WHICH (0 P, 1 T, 2 QVAPOR, 3 QCLOUD, 4 QRAIN, 5 QICE, 6 QSNOW), so the writer can stream one 3-D buffer.
extern "C" __global__ void hist_moist_one(
    int IM, int JM, int LM, int WHICH,
    const float* __restrict__ T, const float* __restrict__ Q, const float* __restrict__ CW,
    const float* __restrict__ F_ICE, const float* __restrict__ F_RAIN,
    const float* __restrict__ PD,
    const float* __restrict__ PSGML1, const float* __restrict__ SGML2,
    float* __restrict__ OUT)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t n = IDX3(i0, j0, l0, IM, JM);
    const size_t n2 = IDX2(i0, j0, IM);

    const float pmid = PSGML1[l0] + SGML2[l0] * PD[n2];
    const float t = T[n];
    const float q = Q[n];
    const float cw = CW[n];
    const float fi = F_ICE[n];
    const float fr = F_RAIN[n];
    const float inv = 1.0f / (1.0f - q);

    const float theta = t * powf(NMMB_P00 / pmid, NMMB_CAPPA);

    float qc = 0.0f, qr = 0.0f, qs = 0.0f;
    if (cw > NMMB_EPSQ) {
        const float liqw = (1.0f - fi) * cw;
        qc = (1.0f - fr) * liqw;
        qr = fr * liqw;
        qs = fi * cw;
    }

    switch (WHICH) {
        case 0: OUT[n] = pmid; break;
        case 1: OUT[n] = theta - 300.0f; break;
        case 2: OUT[n] = q * inv; break;
        case 3: OUT[n] = qc * inv; break;
        case 4: OUT[n] = qr * inv; break;
        case 5: OUT[n] = 0.0f; break;
        default: OUT[n] = qs * inv; break;
    }
}

// Geopotential at the LM+1 interfaces, WRF PH on bottom_top_stag: PH = g * zint. The state Z is MIDLAYER
// height (dyn-cdzdt-nonhydro, Fortran z = zlo + dz/2; init-derive-state writes the same convention), so the
// interfaces are rebuilt from the surface: zint(LM) = FIS/g, zint(l) = 2*Z(l) - zint(l+1). One thread per column.
extern "C" __global__ void hist_ph(
    int IM, int JM, int LM,
    const float* __restrict__ Z, const float* __restrict__ FIS, float* __restrict__ PH)
{
    COLUMN_IJ(IM, JM)
    float zb = FIS[IDX2(i0, j0, IM)] * (1.0f / NMMB_G);
    PH[IDX3(i0, j0, LM, IM, JM)] = NMMB_G * zb;
    for (int l0 = LM - 1; l0 >= 0; --l0) {
        const float zt = 2.0f * Z[IDX3(i0, j0, l0, IM, JM)] - zb;
        PH[IDX3(i0, j0, l0, IM, JM)] = NMMB_G * zt;
        zb = zt;
    }
}

// Surface fields: PSFC = PINT(LM+1) (bottom interface), HGT = FIS/g. Precipitation from the running totals
// since start (io_history_fields.py "Accumulators"): RAINC = CUPREC_TOT (convective), RAINNC = ACPREC_TOT -
// CUPREC_TOT (grid-scale), both converted m -> mm. Not the NPREC bucket CUPREC.
extern "C" __global__ void hist_surface(
    int IM, int JM, int LM,
    const float* __restrict__ PINT, const float* __restrict__ FIS,
    const float* __restrict__ ACPREC_TOT, const float* __restrict__ CUPREC_TOT,
    float* __restrict__ PSFC, float* __restrict__ HGT,
    float* __restrict__ RAINNC, float* __restrict__ RAINC)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;
    const size_t n2 = IDX2(i0, j0, IM);

    PSFC[n2] = PINT[IDX3(i0, j0, LM, IM, JM)];
    HGT[n2] = FIS[n2] / NMMB_G;
    const float cu = CUPREC_TOT[n2] * 1000.0f;
    RAINC[n2] = cu;
    RAINNC[n2] = ACPREC_TOT[n2] * 1000.0f - cu;
}
