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

// Reflectivity as NAM's post writes it: the microphysics REFL_10CM plus the BMJ convective-rain reflectivity
// (woof-nam 0.1.3, 2026-11-05). NCEP's NAM post (nam_nceppost.fd MDLFLD.f 247-345 and 2675-2683, as cited in
// the NAM start study) adds, in Z units, Z = CU_A * R**CU_B (CU_A 300, CU_B 1.4; Marshall-Palmer-type Z-R, the
// same constants as module_DIAGNOSE's CUREFL, see io_max_fields.cu) for the convective rain rate R (mm/h) =
// CPRATE * 3.6e6 / (DT*NPHS), from the convective cloud top (CNVTOP, a bottom-up layer number, 1 = lowest) down
// to the ground: full strength at and below the freezing level (the highest level with T >= 273.15 K, else the
// lowest level), falling off above it as 10**(-2 dz / max(1000 m, zct - zfrz)), i.e. 20 dBZ weaker at the cloud
// top. REFC is the column maximum of that sum, REFD its value at a level. Written from that description and the
// published Z-R constants, not from the post source.
// Levels with no convective term keep REFL_10CM exactly (no dBZ -> Z -> dBZ round trip), so a column without
// convective rain is byte-identical to the plain microphysics field.
// One thread per column. write3 != 0: OUT3 [LM,JM,IM] gets every level; COMP [JM,IM] always gets the maximum.
#define HR_CU_A   300.0f
#define HR_CU_B   1.4f
#define HR_DBZMIN (-20.0f)
#define HR_ZMIN   0.01f          // 10**(0.1*DBZMIN)
#define HR_TFRZ   273.15f
extern "C" __global__ void hist_refl_post(
    int IM, int JM, int LM,
    const float* __restrict__ REFL, const float* __restrict__ T, const float* __restrict__ Z,
    const float* __restrict__ CPRATE, const float* __restrict__ CNVTOP, const float rdtphs,
    const int write3, float* __restrict__ OUT3, float* __restrict__ COMP)
{
    COLUMN_IJ(IM, JM)
    const size_t c = IDX2(i0, j0, IM);
    const float cuprate = rdtphs * CPRATE[c];
    const int cnvtop = (int)CNVTOP[c];
    const bool has = (cuprate > 0.0f) && (cnvtop >= 1);
    float cmax = -1.0e30f;
    if (!has) {
        for (int l0 = 0; l0 < LM; ++l0) {
            const float r = REFL[IDX3(i0, j0, l0, IM, JM)];
            if (write3) OUT3[IDX3(i0, j0, l0, IM, JM)] = r;
            cmax = fmaxf(cmax, r);
        }
        COMP[c] = cmax;
        return;
    }
    // freezing level: the highest level (smallest l0, state levels run top-down) with T >= TFRZ, else the lowest
    float zfrz = Z[IDX3(i0, j0, LM - 1, IM, JM)];
    for (int l0 = 0; l0 < LM; ++l0) {
        if (T[IDX3(i0, j0, l0, IM, JM)] >= HR_TFRZ) { zfrz = Z[IDX3(i0, j0, l0, IM, JM)]; break; }
    }
    const int kct = min(max(cnvtop - 1, 0), LM - 1);      // bottom-up 0-based
    const int lct = LM - 1 - kct;                           // state (top-down) index of the cloud-top layer
    const float zct = Z[IDX3(i0, j0, lct, IM, JM)];
    const float curefl_s = HR_CU_A * powf(cuprate, HR_CU_B);
    const float curefl_i = -2.0f / fmaxf(1000.0f, zct - zfrz);
    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t p = IDX3(i0, j0, l0, IM, JM);
        float r = REFL[p];
        if (l0 >= lct) {
            const float delz = Z[p] - zfrz;
            const float fctr = (delz <= 0.0f) ? 1.0f : powf(10.0f, curefl_i * delz);
            const float zr = powf(10.0f, 0.1f * r);
            const float zt = zr + fctr * curefl_s;
            r = fmaxf(HR_DBZMIN, 10.0f * log10f(fmaxf(HR_ZMIN, zt)));
        }
        if (write3) OUT3[p] = r;
        cmax = fmaxf(cmax, r);
    }
    COMP[c] = cmax;
}
