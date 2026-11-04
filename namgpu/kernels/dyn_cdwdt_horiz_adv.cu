// nam-gpu unit dyn-cdwdt-horiz-adv: horizontal advection of w (dw/dt, horizontal part).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:3653-3727 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine cdwdt, section "advection of w", inside the do l=1,lm loop that starts at 3651;
// dwdt = DWDT, w = W, pfx/pfy/pfne/pfnw = mass fluxes, fah = FAH, dsg2/pdsg1/pdo as in CONTRACT):
//   [A] l, j=jts_b1..jte_h1, i=its_b1..ite_h1 (wx, wy, wne, wnw, recomputed on the fly):
//         wx(i,j)  = (w(i,j,l)-w(i-1,j,l))*pfx(i,j,l)
//         wy(i,j)  = (w(i,j,l)-w(i,j-1,l))*pfy(i,j,l)
//         wne(i,j) = (w(i,j,l)-w(i-1,j-1,l))*pfne(i,j,l)
//         wnw(i,j) = (w(i-1,j,l)-w(i,j-1,l))*pfnw(i,j,l)
//   [B] ww(i,j)=0 over ims..ime, jms..jme (ww is never read in this section; dropped, no output)
//   [C] l, j=jts_b1..jte_b1, fahp = -fah(j)/dt, i=its_b1..ite_b1:
//         advec = ((wx(i,j)+wx(i+1,j)+wy(i,j)+wy(i,j+1)) +
//                  (wne(i+1,j+1)+wne(i,j)+wnw(i,j+1)+wnw(i+1,j))*0.25)*fahp
//                 /(dsg2(l)*pdo(i,j)+pdsg1(l))
//         clip: advec = max(-epsvw, min(epsvw, advec))     (epsvw = 9999.)
//         dwdt(i,j,l) = advec + dwdt(i,j,l)
//       The dwdtmx/dwdtmn/imx/jmx/lmx/imn/jmn/lmn/kp/kn tracking (lines 3670-3691) only feeds the
//       print formats 1300/1400 (lines 3705-3706), which the Fortran never writes: dead in NAM, dropped.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   its_b1 = 2 .. ite_b1 = IM-1, jts_b1 = 2 .. jte_b1 = JM-1 (1-based)
//   -> zero-based advec points i0 in [1, IM-2], j0 in [1, JM-2]. The flux fields are needed at
//      i0, i0+1, j0, j0+1 and i0-1, j0-1, all inside the flux range (1-based 2..IM, 2..JM), so every
//      neighbour is evaluated from W and the mass fluxes exactly as the Fortran flux loop does.
//
// Design: one thread per (i, j, l) (grid_2d, POINT_IJL). Each level is independent in this section
// (no vertical recurrence), so one launch covers all levels. The flux values are recomputed at each
// neighbour from W and PF* with the same expression and operation order as Fortran [A], so no
// scratch array is needed and no thread reads another thread's output. dwdt is written once per
// point (read-modify-write of its own element), so there is no race. Fast numerics: fmad allowed.

#include "nmmb_common.cuh"

// Flux helpers: same expressions and operand order as the Fortran flux loop [A].
__device__ __forceinline__ float cdwdt_wx(const float* __restrict__ w, const float* __restrict__ pfx,
                                          int i0, int j0, int l0, int IM, int JM)
{
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    return (w[c] - w[IDX3(i0 - 1, j0, l0, IM, JM)]) * pfx[c];
}

__device__ __forceinline__ float cdwdt_wy(const float* __restrict__ w, const float* __restrict__ pfy,
                                          int i0, int j0, int l0, int IM, int JM)
{
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    return (w[c] - w[IDX3(i0, j0 - 1, l0, IM, JM)]) * pfy[c];
}

__device__ __forceinline__ float cdwdt_wne(const float* __restrict__ w, const float* __restrict__ pfne,
                                           int i0, int j0, int l0, int IM, int JM)
{
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    return (w[c] - w[IDX3(i0 - 1, j0 - 1, l0, IM, JM)]) * pfne[c];
}

__device__ __forceinline__ float cdwdt_wnw(const float* __restrict__ w, const float* __restrict__ pfnw,
                                           int i0, int j0, int l0, int IM, int JM)
{
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    return (w[IDX3(i0 - 1, j0, l0, IM, JM)] - w[IDX3(i0, j0 - 1, l0, IM, JM)]) * pfnw[c];
}

extern "C" __global__ void cdwdt_horiz_adv(
    const int IM, const int JM, const int LM,
    const float dt,                      // DT (dynamics step, 25 s)
    const float epsvw,                   // epsvw = 9999. (module_DYNAMICS_ROUTINES cdwdt parameter)
    const float* __restrict__ fah,       // FAH   [JM]          in (Fortran fah(jds:jde), j -> j0)
    const float* __restrict__ dsg2,      // DSG2  [LM]          in
    const float* __restrict__ pdsg1,     // PDSG1 [LM]          in
    const float* __restrict__ pdo,       // PDO   [JM,IM]       in
    const float* __restrict__ w,         // W     [LM,JM,IM]    in
    const float* __restrict__ pfx,       // PFX   [LM,JM,IM]    in (mass flux, x)
    const float* __restrict__ pfy,       // PFY   [LM,JM,IM]    in (mass flux, y)
    const float* __restrict__ pfne,      // PFNE  [LM,JM,IM]    in (mass flux, ne)
    const float* __restrict__ pfnw,      // PFNW  [LM,JM,IM]    in (mass flux, nw)
    float* __restrict__ dwdt)            // DWDT  [LM,JM,IM]    inout
{
    POINT_IJL   // i0, j0, l0 (l0 = blockIdx.z)

    // b1 range only (Fortran its_b1..ite_b1, jts_b1..jte_b1 = 2..IM-1, 2..JM-1)
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;


    const float fahp = -fah[j0] / dt;              // fahp=-fah(j)/dt
    const float denom = dsg2[l0] * pdo[IDX2(i0, j0, IM)] + pdsg1[l0];  // dsg2(l)*pdo(i,j)+pdsg1(l)

    const float sum_x = cdwdt_wx(w, pfx, i0, j0, l0, IM, JM)
                      + cdwdt_wx(w, pfx, i0 + 1, j0, l0, IM, JM)
                      + cdwdt_wy(w, pfy, i0, j0, l0, IM, JM)
                      + cdwdt_wy(w, pfy, i0, j0 + 1, l0, IM, JM);

    const float sum_d = cdwdt_wne(w, pfne, i0 + 1, j0 + 1, l0, IM, JM)
                      + cdwdt_wne(w, pfne, i0, j0, l0, IM, JM)
                      + cdwdt_wnw(w, pfnw, i0, j0 + 1, l0, IM, JM)
                      + cdwdt_wnw(w, pfnw, i0 + 1, j0, l0, IM, JM);

    float advec = ((sum_x + sum_d * 0.25f) * fahp) / denom;

    // limiter (Fortran lines 3684-3689)
    if (advec > epsvw) advec = epsvw;
    if (advec < -epsvw) advec = -epsvw;

    const size_t c = IDX3(i0, j0, l0, IM, JM);
    dwdt[c] = advec + dwdt[c];                     // dwdt(i,j,l)=advec+dwdt(i,j,l)
}
