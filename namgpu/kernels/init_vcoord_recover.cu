// init_vcoord_recover.cu -- per-level least-squares fit of the NMMB hybrid mid-layer
// coefficients PSGML1(l) and SGML2(l) from the analysis mid-layer pressure PRES(i,j,l) against
// PD(i,j) = PSFC - PDTOP - PT.
//
// PD identity (Fortran, no PD-from-PSFC assignment exists in the clone; the two surface
// reconstructions with SG1(LM+1) = SG2(LM+1) = 1 and PSG1(LM+1) = SG1*PDTOP + PT give):
//   module_TURBULENCE.F90:557   PSFC = SG2(LM+1)*PD + PSG1(LM+1)
//   module_RADIATION.F90:287    PHINT(LM+1) = SG2(LM+1)*PD + PSG1(LM+1)
// so PD = PSFC - PDTOP - PT. (module_NESTING.F90:10543 forms a parent PSFC as PD + PT; it is a
// different offset and is not used by this unit. Noted for the nesting unit to resolve.)
//
// Notice: no NAM Fortran counterpart for the fit. The NAM nam.v4.2.11 source (MatthewPyle-NOAA/
// NAM @ 83df4e13c1, NOAA/NCEP/EMC, US Government work, see NOTICE) reads SG1/SG2/SGML1/SGML2/
// PDTOP from the input file header (module_INIT_READ_BIN.F90:177-185, 720-751;
// module_INIT_READ_NEMSIO.F90:190-198, 914-944). The port's header path is in
// namgpu/init/init_vcoord_recover.py and takes precedence over this fit when a header is present.
// This unit is UNITS.json id init-vcoord-recover (mode new). CONTRACT.md has no section-4 row for
// it; section 4 lists where the module outputs go, and the outputs written here are SG1, SG2,
// DSG1, DSG2, SGML1, SGML2, PSG1, PSGML1, PDSG1 and PDTOP.
//
// Model per level l (zero-based l0), over the b1 columns (CONTRACT.md section 0, _b1 limits per
// module_DM_PARALLEL looplimits: zero-based i0 in [1, IM-2], j0 in [1, JM-2]; the boundary ring
// is excluded because the NAM b1 range is the interior of the domain):
//     PRES(i,j,l) ~= a_l + b_l * PD(i,j)      a_l = PSGML1(l), b_l = SGML2(l)
// Ordinary least squares, one level at a time. Sums are accumulated in double; PD and PRES are
// read as float (kfpt = single) exactly as the state holds them.
//
// Kernels:
//   init_vcoord_fit_partial : grid (30 x 105 x LM) of 32 x 8 blocks, blockIdx.z = level.
//                             Columns outside the b1 range contribute zero. Each block reduces
//                             its tile in shared memory and adds the block total into acc[q][l]
//                             with one atomicAdd per quantity per block.
//   init_vcoord_solve       : one thread per level; writes psgml1[l] and sgml2[l] (float).
// acc must be zeroed before init_vcoord_fit_partial (the wrapper does it once per call).
//
// Guard (named breakage): if PD is constant over the b1 domain the slope has no value. The
// solve kernel writes NaN for that level (it does not write a zero slope: b = 0 makes the
// interface recurrence put SG2(LM+1) = 0 instead of 1, so PINT(LM+1) != PSFC and nothing
// downstream notices). The wrapper checks finiteness on the host and raises.
//
// No host sync inside the kernels, no malloc, no printf. Fast numerics (fmad on, IEEE division).

#include "nmmb_common.cuh"

#define INIT_VCOORD_BX 32
#define INIT_VCOORD_BY 8
#define INIT_VCOORD_NT (INIT_VCOORD_BX * INIT_VCOORD_BY)

extern "C" __global__ void init_vcoord_fit_partial(
    const float* __restrict__ pres,    // [LM, JM, IM]  analysis mid-layer pressure (Pa)
    const float* __restrict__ psfc,    // [JM, IM]      surface pressure (Pa)
    double* __restrict__ acc,          // [4, LM]       sums: Sx=PD, Sy=PRES, Sxx, Sxy
    const float pt,                    // model top pressure (Pa), configfile pt
    const float pdtop,                 // pressure depth of the pressure range (Pa)
    const int im,
    const int jm,
    const int lm)
{
    __shared__ double s_x[INIT_VCOORD_NT];
    __shared__ double s_y[INIT_VCOORD_NT];
    __shared__ double s_xx[INIT_VCOORD_NT];
    __shared__ double s_xy[INIT_VCOORD_NT];

    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // b1 columns: 1 <= i0 <= IM-2, 1 <= j0 <= JM-2 (zero-based), CONTRACT.md section 0.
    const bool in_b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2 && l0 < lm);
    double x = 0.0, y = 0.0, xx = 0.0, xy = 0.0;
    if (in_b1) {
        // PD = PSFC - PDTOP - PT   (float, as the state holds it; see the identity above)
        const float pd = psfc[IDX2(i0, j0, im)] - pdtop - pt;
        const float pr = pres[IDX3(i0, j0, l0, im, jm)];
        x  = (double)pd;
        y  = (double)pr;
        xx = x * x;
        xy = x * y;
    }
    s_x[tid] = x;
    s_y[tid] = y;
    s_xx[tid] = xx;
    s_xy[tid] = xy;
    __syncthreads();

    // Tree reduction over the 256 threads of the block.
    for (int s = INIT_VCOORD_NT / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_x[tid]  += s_x[tid + s];
            s_y[tid]  += s_y[tid + s];
            s_xx[tid] += s_xx[tid + s];
            s_xy[tid] += s_xy[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&acc[0 * lm + l0], s_x[0]);
        atomicAdd(&acc[1 * lm + l0], s_y[0]);
        atomicAdd(&acc[2 * lm + l0], s_xx[0]);
        atomicAdd(&acc[3 * lm + l0], s_xy[0]);
    }
}

extern "C" __global__ void init_vcoord_solve(
    const double* __restrict__ acc,    // [4, LM] sums from init_vcoord_fit_partial
    const double n,                    // number of b1 columns (IM-2)*(JM-2)
    float* __restrict__ psgml1,        // [LM] out: intercept a_l = PSGML1(l)  (Pa)
    float* __restrict__ sgml2,         // [LM] out: slope b_l = SGML2(l)
    const int lm)
{
    const int l0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (l0 >= lm) return;

    const double sx  = acc[0 * lm + l0];
    const double sy  = acc[1 * lm + l0];
    const double sxx = acc[2 * lm + l0];
    const double sxy = acc[3 * lm + l0];

    const double mean_x = sx / n;
    const double mean_y = sy / n;
    const double cxx = sxx - sx * sx / n;      // centred sums
    const double cxy = sxy - sx * sy / n;

    // Undetermined slope (constant PD): NaN, refused by the host wrapper. Relative test against
    // the uncentred sum of squares.
    const double qnan = __longlong_as_double(0x7ff8000000000000LL);
    double a = qnan, b = qnan;
    if (cxx > 1.0e-12 * (sxx > 1.0 ? sxx : 1.0)) {
        b = cxy / cxx;
        a = mean_y - b * mean_x;
    }

    psgml1[l0] = (float)a;
    sgml2[l0]  = (float)b;
}
