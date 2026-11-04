// WOOF-NAM unit dyn-adv2-vert: Crank-Nicolson (uncentred w1/w2) vertical advection of the tracer group in ADV2.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:4463-4569 (nam.v4.2.11, 83df4e13c1).
//   Subroutine adv2, the vertical Crank-Nicolson block only (pdops smoothing, pvvlo/vvlo/cms forward
//   sweep, rsts forward sweep, level-lm closure, tcs back substitution).
//   The horizontal Adams-Bashforth part that follows (Fortran 4570 onward: ss1/ssx/ssy/ssne/ssnw and the
//   tcs accumulation) belongs to the horizontal ADV2 unit and is not in this kernel.
//
// Fortran (subroutine adv2, lm = LM, dtq = dt*0.25*idtadt, wb = 1, wa = (1-wb)*0.25, w1 = 0.9, w2 = 2-w1):
//   pdops(i,j)  = (pdop(i,j-1)+pdop(i-1,j)+pdop(i+1,j)+pdop(i,j+1))*wa + pdop(i,j)*wb
//                 with pdop = (pd+pdo)*0.5 at each point (recomputed here, same arithmetic as the
//                 Fortran pdop array that NAM fills just before this block)
//   [F1] l=1:   pvvlo = ((psgdt(j-1)+psgdt(i-1)+psgdt(i+1)+psgdt(j+1))*wa + psgdt(i,j)*wb)*dtq
//               vvlo = pvvlo/(dsg2(1)*pdops + pdsg1(1)); cms = -vvlo*w2+1; rcms(1)=1/cms; crs(1)=vvlo*w2
//               rsts(1,ks) = (-vvlo*w1)*(s1(2,ks)-s1(1,ks)) + s1(1,ks)
//   [F2] l=2..lm-1: rdp = 1/(dsg2(l)*pdops+pdsg1(l)); pvvup=pvvlo; pvvlo = smoothed psgdt(l)*dtq
//               vvup = pvvup*rdp; vvlo = pvvlo*rdp; cf = -vvup*w2*rcms(l-1)
//               cms = -crs(l-1)*cf + ((vvup-vvlo)*w2+1); rcms(l)=1/cms; crs(l)=vvlo*w2
//               rsts(l,ks) = -rsts(l-1,ks)*cf + s1(l,ks) - (s1(l,ks)-s1(l-1,ks))*vvup*w1
//                            - (s1(l+1,ks)-s1(l,ks))*vvlo*w1
//   [F3] l=lm:  pvvup=pvvlo; vvup = pvvup/(dsg2(lm)*pdops+pdsg1(lm)); cf = -vvup*w2*rcms(lm-1)
//               cms = -crs(lm-1)*cf + (vvup*w2+1); rcms(lm)=1/cms; crs(lm)=0
//               rsts(lm,ks) = -rsts(lm-1,ks)*cf + s1(lm,ks) - (s1(lm,ks)-s1(lm-1,ks))*vvup*w1
//               tcs(lm,ks) = rsts(lm,ks)*rcms(lm) - s1(lm,ks)
//   [F4] l=lm-1..1 (back substitution, per species):
//               tcs(l,ks) = (-crs(l)*(s1(l+1,ks)+tcs(l+1,ks)) + rsts(l,ks))*rcms(l) - s1(l,ks)
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, regional, ids=1 ide=IM jds=1 jde=JM):
//   the vertical block runs on its_b1..ite_b1 and jts_b1..jte_b1 (1-based 2..IM-1, 2..JM-1), so i0 in [1, IM-2],
//   j0 in [1, JM-2]. Boundary rows and columns are not written by the Fortran; this kernel does not write them.
//   The MPI/OpenMP looplimits split (ENABLE_SMP) is dropped: one thread per column.
//
// Scratch / in-place use (CONTRACT section 1):
//   rcms, crs -> ADV2_RCMS, ADV2_CRS [LM,JM,IM] scratch (species independent, rewritten per species pass,
//                each column touched only by its own thread).
//   rsts      -> the Fortran rsts(l,ks) array is held in tcs(l,ks) during the forward sweep. tcs(l) holds
//                rsts(l) until the back substitution reads it and overwrites it, level by level from the top
//                (same trick as dyn-adv1-vert-t). The Fortran tcs is fully rewritten at levels 1..lm, so its
//                previous contents are not used.
//   Each species is swept in its own pass over the levels. The geometry (rcms, crs, pvvlo, vvlo) does not depend
//   on the species, so repeating it per species yields identical values.
//
// Tracer layout: TRACERS-shaped arrays are [NTR, LM, JM, IM]; species n0 (zero-based) = Fortran ks - 1.
// Single precision (kfpt = single). FMA contraction on (NVRTC --fmad=true), IEEE division.

#include "nmmb_common.cuh"

// Fortran pdop = (pd+pdo)*0.5
__device__ __forceinline__ float adv2_pdop(const float* __restrict__ pd, const float* __restrict__ pdo,
                                            int i0, int j0, int IM)
{
    return (pd[IDX2(i0, j0, IM)] + pdo[IDX2(i0, j0, IM)]) * 0.5f;
}

// Smoothed psgdt at zero-based level l0: (psgdt(i,j-1)+psgdt(i-1,j)+psgdt(i+1,j)+psgdt(i,j+1))*wa + psgdt(i,j)*wb
__device__ __forceinline__ float adv2_psgdt_smooth(const float* __restrict__ psgdt, int i0, int j0, int l0,
                                                    int IM, int JM, float wa, float wb)
{
    return ((psgdt[IDX3(i0, j0 - 1, l0, IM, JM)] + psgdt[IDX3(i0 - 1, j0, l0, IM, JM)]
           + psgdt[IDX3(i0 + 1, j0, l0, IM, JM)] + psgdt[IDX3(i0, j0 + 1, l0, IM, JM)]) * wa
           + psgdt[IDX3(i0, j0, l0, IM, JM)] * wb);
}

extern "C" __global__ void adv2_vert(
    const int IM, const int JM, const int LM, const int NTR,
    const int kss0, const int kse0,          // zero-based inclusive tracer range (Fortran kss-1 .. kse-1)
    const float dt, const int idtadt,        // DT and IDTADT (or IDTADTQ for the Q call)
    const float* __restrict__ pd,            // PD           [JM,IM]          in
    const float* __restrict__ pdo,           // PDO          [JM,IM]          in
    const float* __restrict__ psgdt,         // PSGDT        [LM+1,JM,IM]     in (levels 0..LM-2 used)
    const float* __restrict__ dsg2,          // DSG2         [LM]             in
    const float* __restrict__ pdsg1,         // PDSG1        [LM]             in
    const float* __restrict__ s1,            // TRACERS_SQRT [NTR,LM,JM,IM]   in (Fortran s1)
    float* __restrict__ rcms,                // ADV2_RCMS    [LM,JM,IM]       scratch
    float* __restrict__ crs,                 // ADV2_CRS     [LM,JM,IM]       scratch
    float* __restrict__ tcs)                 // TRACERS_TEND [NTR,LM,JM,IM]   out (rsts held here during sweep)
{
    COLUMN_IJ(IM, JM)

    // b1 range: boundary rows and columns are not written by the Fortran
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    const float wb  = 1.0f;                          // weighting factor
    const float wa  = (1.0f - wb) * 0.25f;           // weighting factor
    const float w1  = 0.9f;                          // crank-nicholson uncentering
    const float w2  = 2.0f - w1;                     // = 2.-w1
    const float dtq = dt * 0.25f * (float)idtadt;    // dtq = dt*0.25*idtadt

    // pdops: smoothed hydrostatic pressure difference at h points (species independent)
    const float pdops = ((adv2_pdop(pd, pdo, i0, j0 - 1, IM) + adv2_pdop(pd, pdo, i0 - 1, j0, IM))
                       + adv2_pdop(pd, pdo, i0 + 1, j0, IM) + adv2_pdop(pd, pdo, i0, j0 + 1, IM)) * wa
                       + adv2_pdop(pd, pdo, i0, j0, IM) * wb;

#define T4(l0_) IDX4(i0, j0, (l0_), n0, IM, JM, LM)
#define S3(l0_) IDX3(i0, j0, (l0_), IM, JM)

    for (int n0 = kss0; n0 <= kse0; ++n0) {

        // ---- [F1] level 1 (l0 = 0): geometry and rsts(1) --------------------------------------------
        float pvvlo = adv2_psgdt_smooth(psgdt, i0, j0, 0, IM, JM, wa, wb) * dtq;
        float vvlo  = pvvlo / (dsg2[0] * pdops + pdsg1[0]);
        float cms   = -vvlo * w2 + 1.0f;
        rcms[S3(0)] = 1.0f / cms;
        crs[S3(0)]  = vvlo * w2;
        const float s1_1 = s1[T4(0)];
        const float s1_2 = s1[T4(1)];
        tcs[T4(0)] = (-vvlo * w1) * (s1_2 - s1_1) + s1_1;     // rsts(1) held in tcs(1) until F4

        // ---- [F2] levels 2..lm-1 (l0 = 1..LM-2) ----------------------------------------------------
        for (int l0 = 1; l0 < LM - 1; ++l0) {
            const float rdp   = 1.0f / (dsg2[l0] * pdops + pdsg1[l0]);
            const float pvvup = pvvlo;
            pvvlo = adv2_psgdt_smooth(psgdt, i0, j0, l0, IM, JM, wa, wb) * dtq;
            const float vvup = pvvup * rdp;
            vvlo = pvvlo * rdp;
            const float cf = -vvup * w2 * rcms[S3(l0 - 1)];
            cms = -crs[S3(l0 - 1)] * cf + ((vvup - vvlo) * w2 + 1.0f);
            rcms[S3(l0)] = 1.0f / cms;
            crs[S3(l0)]  = vvlo * w2;

            const float s1_bel = s1[T4(l0 - 1)];
            const float s1_cur = s1[T4(l0)];
            const float s1_abv = s1[T4(l0 + 1)];
            const float rsts_bel = tcs[T4(l0 - 1)];
            tcs[T4(l0)] = -rsts_bel * cf + s1_cur
                        - (s1_cur - s1_bel) * vvup * w1
                        - (s1_abv - s1_cur) * vvlo * w1;
        }

        // ---- [F3] level lm (l0 = LM-1) ---------------------------------------------------------------
        {
            const int lt = LM - 1;
            const float pvvup = pvvlo;
            const float vvup  = pvvup / (dsg2[lt] * pdops + pdsg1[lt]);
            const float cf    = -vvup * w2 * rcms[S3(lt - 1)];
            cms = -crs[S3(lt - 1)] * cf + (vvup * w2 + 1.0f);
            rcms[S3(lt)] = 1.0f / cms;
            crs[S3(lt)]  = 0.0f;

            const float s1_bel = s1[T4(lt - 1)];
            const float s1_top = s1[T4(lt)];
            const float rsts_bel = tcs[T4(lt - 1)];
            const float rsts_top = -rsts_bel * cf + s1_top - (s1_top - s1_bel) * vvup * w1;
            tcs[T4(lt)] = rsts_top * rcms[S3(lt)] - s1_top;
        }

        // ---- [F4] back substitution, levels lm-1..1 (l0 = LM-2..0) -----------------------------------
        for (int l0 = LM - 2; l0 >= 0; --l0) {
            const float rsts_l = tcs[T4(l0)];                  // rsts(l), stored in F1/F2
            tcs[T4(l0)] = (-crs[S3(l0)] * (s1[T4(l0 + 1)] + tcs[T4(l0 + 1)]) + rsts_l) * rcms[S3(l0)]
                        - s1[T4(l0)];
        }
    }
#undef T4
#undef S3
}
