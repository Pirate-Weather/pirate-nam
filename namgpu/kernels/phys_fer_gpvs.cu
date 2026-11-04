// phys_fer_gpvs.cu -- GPVS_hr: saturation vapour pressure lookup table for the Ferrier-Aligo
// microphysics (table TBPVS / TBPVS0, plus the linear-interpolation scalars C1XPVS, C2XPVS,
// C1XPVS0, C2XPVS0).
//
// NOTICE: NAM source file phys/module_MP_FER_HIRES.F90 (NOAA/NCEP/EMC, US Government work, no
// licence file). Module header kept as the notice:
//   MODULE MODULE_MP_FER_HIRES (Ferrier-Aligo high-resolution microphysics, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from phys/module_MP_FER_HIRES.F90:2791-2836 (nam.v4.2.11, 83df4e13c1).
// FPVSX / FPVSX0 that GPVS_hr calls (module_MP_FER_HIRES.F90:2915-2989) are defined once in
// fer_fpvs_common.cuh and shared with the phys-fer-fpvs unit.
//
// Fortran (every statement of 2791-2836 and its counterpart):
//   XINC=(XMAX-XMIN)/(NX-1)                 -> xinc (float; NX-1 converted to real)
//   C1XPVS=1.-XMIN/XINC; C2XPVS=1./XINC     -> coef[0], coef[1]
//   C1XPVS0=1.-XMIN/XINC; C2XPVS0=1./XINC   -> coef[2], coef[3]  (identical to C1XPVS/C2XPVS)
//   DO JX=1,NX: X=XMIN+(JX-1)*XINC; T=X    -> one thread per JX (jx0 = JX-1), X computed the same way
//   TBPVS(JX)=FPVSX(T); TBPVS0(JX)=FPVSX0(T) -> tbpvs[jx0], tbpvs0[jx0]
// Module constants used (module_MP_FER_HIRES.F90 local PARAMETERs and module_CONSTANTS r_v):
//   NX=7501, XMIN=180.0, XMAX=330.0; RV = R_V = 461.6 (NMMB_R_V).
//
// Notes on the Fortran that is kept as written:
//   - FPVSX0 declares XAI=-DLDT/RV and XBI=XA+HSUB/(RV*TTP) but never uses them (only XA, XB and the
//     TTP branch-free formula are used). They are not computed here: they have no effect on output.
//   - FPVSX chooses the water branch (XA, XB) for T >= TTP and the ice branch (XAI, XBI) otherwise.
//   - The Fortran real PARAMETER expressions are evaluated in single precision here (float), with the
//     same operation order; the compiler's constant folding may differ in the last bit (not claimed bitwise).
// Output: tbpvs, tbpvs0 are float32[NX] (CuPy 1-D). coef is float32[4] = {C1XPVS, C2XPVS, C1XPVS0, C2XPVS0}.
// One launch, ceil(NX / blockDim) blocks, one thread per table entry. No host sync, no allocation.

#include "nmmb_common.cuh"
#include "fer_fpvs_common.cuh"

#define FER_NX     7501
#define FER_XMIN   180.0f
#define FER_XMAX   330.0f

// GPVS_hr: one thread per table entry JX = jx0+1 (jx0 = 0 .. NX-1).
extern "C" __global__ void gpvs_hr(float* __restrict__ tbpvs,
                                   float* __restrict__ tbpvs0,
                                   float* __restrict__ coef)
{
    const float XMIN = FER_XMIN;
    const float XMAX = FER_XMAX;
    const float XINC = (XMAX - XMIN) / (float)(FER_NX - 1);

    const int jx0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (jx0 == 0) {
        coef[0] = 1.f - XMIN / XINC;   // C1XPVS
        coef[1] = 1.f / XINC;          // C2XPVS
        coef[2] = 1.f - XMIN / XINC;   // C1XPVS0
        coef[3] = 1.f / XINC;          // C2XPVS0
    }
    if (jx0 >= FER_NX) return;

    const float X = XMIN + (float)jx0 * XINC;   // Fortran X=XMIN+(JX-1)*XINC
    tbpvs[jx0]  = nam_fpvsx(X);                 // TBPVS(JX)=FPVSX(T)
    tbpvs0[jx0] = nam_fpvsx0(X);                // TBPVS0(JX)=FPVSX0(T)
}
