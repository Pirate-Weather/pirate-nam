// dyn_vtoa.cu -- omega-alpha term of the T equation and the interface-pressure update.
//
// NOTICE: NAM source file nmm/module_DYNAMICS_ROUTINES.F90 (NOAA/NCEP/EMC, US Government work,
// no licence file). This file has no author/history block of its own; see NOTICE at the repo root.
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:2191-2347 (nam.v4.2.11, 83df4e13c1).
//
// Fortran routine VTOA(lm, dt, ef4t, pt, sg2, psdt, dwdt, rtop, omgalf, pint, tdiv, tct).
//
// Mapping (CONTRACT.md section 1): Fortran A(i,j,l) -> A[(l-1), (j-1), (i-1)] with IDX3.
//   PINT, interfaces: LM+1 levels. DWDT, RTOP, OMGALF, TDIV, TCT: LM levels. PSDT: [JM, IM].
//   SG2: LM+1 values (sg2(l) -> sg2[l-1]).
//
// Loop limits: NMMB _b1 limits on a single full domain (module_DM_PARALLEL.F90 looplimits,
// ids = 1, ide = IM): its_b1 = max(its, ids+1) = 2, ite_b1 = min(ite, ide-1) = IM-1, and the same
// in j. So the kernel updates only 1-based i in [2, IM-1], j in [2, JM-1] (zero-based [1, IM-2]).
//
// Column dependence: the l-sweep of PINT is a recurrence along each column (pint(l+1) uses
// pint(l) of the same column, tpm/tpmp carry the old interface sums). Columns are independent
// (all other inputs are read-only, and omgalf/tct/pint are written only at their own (i,j,l)).
// So one thread per column runs the l-loop exactly as the Fortran does, with the same
// order of reads and writes (tpmp is formed from the OLD pint(l+1), pint(l+2) before pint(l+1)
// is overwritten).
//
// Fortran lines not mapped to a term: the ENABLE_SMP OpenMP blocks (lines 2256-2270 and
// 2335-2342) are host threading only (jstart/jstop = jts_b1/jte_b1 when OpenMP is off); the
// commented-out tct=0 block (2274-2280) is commented out in the source and is not run.
// NMMB's TCT is accumulated (tct = tct + toa), not reset here.

#include "nmmb_common.cuh"

// One Fortran 1-based (i, j, l) index, for an array with LEVELS levels in the leading dimension.
__device__ __forceinline__ size_t vtoa_ix(int i0, int j0, int l1, int im, int jm)
{
    return IDX3(i0, j0, l1 - 1, im, jm);
}

extern "C" __global__ void dyn_vtoa(
    const int lm,                          // total number of levels (Fortran LM)
    const int im,                          // IM
    const int jm,                          // JM
    const float dt,                        // dynamics time step
    const float ef4t,                      // int_state%EF4T: 0.5*dt/cp (CONSTS), rescaled on DFI steps
    const float pt,                        // pressure at model top
    const float* __restrict__ sg2,         // [LM+1] delta sigmas (sg2(1)..sg2(LM+1))
    const float* __restrict__ psdt,        // [JM, IM] surface pressure tendency
    const float* __restrict__ dwdt,        // [LM, JM, IM] nonhydrostatic correction factor
    const float* __restrict__ rtop,        // [LM, JM, IM] RT/p
    float* __restrict__ omgalf,            // [LM, JM, IM] omega-alpha, updated in place
    float* __restrict__ pint,              // [LM+1, JM, IM] interface pressure, updated in place
    const float* __restrict__ tdiv,        // [LM, JM, IM] integrated horizontal mass divergence
    float* __restrict__ tct)               // [LM, JM, IM] time change of temperature, accumulated
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= im * jm) return;
    const int i0 = col % im;
    const int j0 = col / im;
    // NMMB _b1 limits for the full domain: 1-based i, j in [2, IM-1], [2, JM-1].
    if (i0 < 1 || i0 > im - 2 || j0 < 1 || j0 > jm - 2) return;

    // ---- Fortran lines 2283-2301: first level (l = 1), the surface-pressure interface ----
    // The sweep needs both pint(2) and pint(3) before pint(2) is overwritten.
    pint[vtoa_ix(i0, j0, 1, im, jm)] = pt;                       // pint(i,j,1) = pt
    float tpm = pt + pint[vtoa_ix(i0, j0, 2, im, jm)];           // tpm = pt + pint(i,j,2), old pint(2)

    float dwdtp = dwdt[vtoa_ix(i0, j0, 1, im, jm)];
    float tpmp = pint[vtoa_ix(i0, j0, 2, im, jm)] + pint[vtoa_ix(i0, j0, 3, im, jm)];   // old values
    float toa = -tdiv[vtoa_ix(i0, j0, 1, im, jm)] * rtop[vtoa_ix(i0, j0, 1, im, jm)] * dwdtp * ef4t;

    omgalf[vtoa_ix(i0, j0, 1, im, jm)] = omgalf[vtoa_ix(i0, j0, 1, im, jm)] + toa;
    tct[vtoa_ix(i0, j0, 1, im, jm)] = tct[vtoa_ix(i0, j0, 1, im, jm)] + toa;

    pint[vtoa_ix(i0, j0, 2, im, jm)] = psdt[IDX2(i0, j0, im)] * (sg2[0] + sg2[1]) * dwdtp * dt
                                     + tpm - pint[vtoa_ix(i0, j0, 1, im, jm)];
    tpm = tpmp;

    // ---- Fortran lines 2303-2320: levels l = 2 .. LM-1 ----
    for (int l = 2; l <= lm - 1; ++l) {
        dwdtp = dwdt[vtoa_ix(i0, j0, l, im, jm)];

        tpmp = pint[vtoa_ix(i0, j0, l + 1, im, jm)] + pint[vtoa_ix(i0, j0, l + 2, im, jm)];   // old values
        toa = -(tdiv[vtoa_ix(i0, j0, l - 1, im, jm)] + tdiv[vtoa_ix(i0, j0, l, im, jm)])
              * rtop[vtoa_ix(i0, j0, l, im, jm)] * dwdtp * ef4t;

        omgalf[vtoa_ix(i0, j0, l, im, jm)] = omgalf[vtoa_ix(i0, j0, l, im, jm)] + toa;
        tct[vtoa_ix(i0, j0, l, im, jm)] = tct[vtoa_ix(i0, j0, l, im, jm)] + toa;

        pint[vtoa_ix(i0, j0, l + 1, im, jm)] = psdt[IDX2(i0, j0, im)] * (sg2[l - 1] + sg2[l]) * dwdtp * dt
                                             + tpm - pint[vtoa_ix(i0, j0, l, im, jm)];
        tpm = tpmp;
    }

    // ---- Fortran lines 2322-2334: last level l = LM (no tpmp carried) ----
    dwdtp = dwdt[vtoa_ix(i0, j0, lm, im, jm)];
    toa = -(tdiv[vtoa_ix(i0, j0, lm - 1, im, jm)] + tdiv[vtoa_ix(i0, j0, lm, im, jm)])
          * rtop[vtoa_ix(i0, j0, lm, im, jm)] * dwdtp * ef4t;

    omgalf[vtoa_ix(i0, j0, lm, im, jm)] = omgalf[vtoa_ix(i0, j0, lm, im, jm)] + toa;
    tct[vtoa_ix(i0, j0, lm, im, jm)] = tct[vtoa_ix(i0, j0, lm, im, jm)] + toa;

    pint[vtoa_ix(i0, j0, lm + 1, im, jm)] = psdt[IDX2(i0, j0, im)] * (sg2[lm - 1] + sg2[lm]) * dwdtp * dt
                                          + tpm - pint[vtoa_ix(i0, j0, lm, im, jm)];
}
