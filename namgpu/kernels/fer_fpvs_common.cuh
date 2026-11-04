// fer_fpvs_common.cuh -- the exact saturation vapour pressure functions FPVSX and FPVSX0 of
// MP_FER_HIRES, shared by every unit that needs them (phys-fer-gpvs builds the table from them,
// phys-fer-fpvs and its callers evaluate them). One definition, so the formula cannot drift.
//
// NOTICE: NOAA/NCEP NAM source module_MP_FER_HIRES.F90 (US government work, no licence file in the
// NAM clone). Ported to CUDA for nam-gpu from phys/module_MP_FER_HIRES.F90:2915-2971 (FPVSX) and
// 2974-2989 (FPVSX0) (nam.v4.2.11, 83df4e13c1).
//
// Fortran FPVSX (2915-2971) chooses the water branch (XA, XB) for T >= TTP and the ice branch (XAI, XBI)
// otherwise. FPVSX0 (2974-2989) is the liquid-only form. In FPVSX0 the Fortran declares XAI=-DLDT/RV and
// XBI=XA+HSUB/(RV*TTP); they are never used by the returned value and are kept here as dead locals.
// Constants: TTP, HVAP, PSAT, CLIQ, CVAP, CICE, HSUB from the FPVSX parameter blocks; R_V from
// nmmb_common.cuh (module_CONSTANTS).
#ifndef NAMGPU_FER_FPVS_COMMON_CUH
#define NAMGPU_FER_FPVS_COMMON_CUH

#include "nmmb_common.cuh"

#define FPVSX_TTP   2.7316E+2f
#define FPVSX_HVAP  2.5000E+6f
#define FPVSX_PSAT  6.1078E+2f
#define FPVSX_CLIQ  4.1855E+3f
#define FPVSX_CVAP  1.8460E+3f
#define FPVSX_CICE  2.1060E+3f
#define FPVSX_HSUB  2.8340E+6f

// FPVSX(T): exact integrated Clausius-Clapeyron, liquid for T >= TTP, ice below. Returns kPa.
__device__ inline float nam_fpvsx(float T)
{
    const float TTP = FPVSX_TTP, HVAP = FPVSX_HVAP, PSAT = FPVSX_PSAT;
    const float CLIQ = FPVSX_CLIQ, CVAP = FPVSX_CVAP, CICE = FPVSX_CICE, HSUB = FPVSX_HSUB;
    const float PSATK = PSAT * 1.E-3f;                        // Fortran PSATK = PSAT*1.E-3
    const float DLDT = CVAP - CLIQ;
    const float XA = -DLDT / NMMB_R_V;
    const float XB = XA + HVAP / (NMMB_R_V * TTP);
    const float DLDTI = CVAP - CICE;
    const float XAI = -DLDTI / NMMB_R_V;
    const float XBI = XAI + HSUB / (NMMB_R_V * TTP);
    const float TR = TTP / T;
    if (T >= TTP) {
        return PSATK * powf(TR, XA) * expf(XB * (1.0f - TR));
    } else {
        return PSATK * powf(TR, XAI) * expf(XBI * (1.0f - TR));
    }
}

// FPVSX0(T): exact liquid-only form. Returns kPa.
__device__ inline float nam_fpvsx0(float T)
{
    const float TTP = FPVSX_TTP, HVAP = FPVSX_HVAP, PSAT = FPVSX_PSAT;
    const float CLIQ = FPVSX_CLIQ, CVAP = FPVSX_CVAP;
    const float PSATK = PSAT * 1.E-3f;
    const float DLDT = CVAP - CLIQ;
    const float XA = -DLDT / NMMB_R_V;
    const float XB = XA + HVAP / (NMMB_R_V * TTP);
    const float TR = TTP / T;
    return PSATK * powf(TR, XA) * expf(XB * (1.0f - TR));
}

#endif // NAMGPU_FER_FPVS_COMMON_CUH
