// phys_rrtm_gethml.cu -- WOOF-NAM unit phys-rrtm-gethml: subroutine gethml (high, mid, low, total
// and boundary-layer cloud fractions, plus the cloud top/bottom layer indices per domain).
//
// NOTICE: NOAA / NCEP NAM source (US government work, no licence file in the NAM clone). Header of
// phys/radiation_clouds_nmmb.f, subroutine gethml (nam.v4.2.11, 83df4e13c1), kept as the notice:
//   subroutine gethml
//   abstract: compute high, mid, low, total, and boundary cloud fractions and cloud top/bottom
//   layer indices for model diagnostic output. the three cloud domain boundaries are defined by
//   ptopc. the cloud overlapping method is defined by control flag 'iovr'.
//   program history log: 04-29-2004 yu-tai hou - separated to become individule subprogram.
// Ported to CUDA for WOOF-NAM from phys/radiation_clouds_nmmb.f:2167-2494 (nam.v4.2.11, 83df4e13c1).
//
// Fortran lines covered:
//   2167-2212  argument list and declarations            -> kernel signature
//   2213-2222  clds(i,1:5) = 0                           -> zero the five outputs
//   2224-2226  cl1 = cl2 = 1                             -> per-column scalars (one thread per column)
//   2228-2236  kstr/kend/kinc from ivflip                -> loop bounds
//   2238-2271  iovr == 0 (random overlap) total and bl   -> first block, iovr == 0 branch
//   2272-2300  iovr /= 0 (max/ran overlap) total and bl  -> first block, iovr /= 0 branch
//   2305-2404  ivflip == 0 high/mid/low sweep (toa->sfc) -> second block, ivflip == 0 branch
//   2405-2487  ivflip /= 0 high/mid/low sweep (sfc->toa) -> second block, ivflip /= 0 branch
//   2490       return
// The module variables are inputs: climit = 0.001, ovcst = 1.0 (radiation_clouds_nmmb.f:167-168, see GETHML_OVCST),
// NK_CLDS = 3 (line 158), iovr and llyr (lines 212-213 and 328-342, set by the caller of gethml).
// Both vertical orders are ported; ivflip selects one at run time, as the Fortran does.
//
// Layout (CONTRACT section 1): the Fortran (IX,NLAY) arrays are [NLAY, JM, IM] in CuPy order, the
// column index (i, j) is one thread, and level k (1-based, Fortran) is read at zero-based k - 1.
// The output clds is [5, JM, IM] (low, mid, high, total, bl), mtop and mbot are [3, JM, IM] int32.
// Fortran indices are kept 1-based inside the kernel so every comparison reads like the source.
//
// Fast numerics: float (kind_phys single in this port). Halo / MPI / OpenMP: none (single card, full
// domain). The iovr branch and the ivflip branch are both kept.
#include "nmmb_common.cuh"

#define GETHML_CLIMIT   0.001f
// ovcst = 1.0 - 1.0e-8 is a default-real constant expression in the NAM physics build (no -fdefault-real-8
// there), so it evaluates to exactly 1.0 before conversion to kind_phys. 1.0f is the same value. The min()
// cap is a no-op in the Fortran too.
#define GETHML_OVCST    1.0f
#define GETHML_NK_CLDS  3
// Cloud inputs must be finite (caller contract, checked in tools/check-phys-rrtm-gethml.py). fminf/fmaxf
// ignore a NaN operand; the NAM min/max order for NaN is compiler-defined and not in the source.

// Fortran NINT: nearest integer, halves rounded away from zero (roundf does the same).
__device__ inline int gethml_nint(float x) { return (int)roundf(x); }

// Column accessors. c is the column offset inside one [n, JM, IM] plane.
#define GH_LV(a, k) (a)[((size_t)((k) - 1)) * plane + c]     // level k, 1-based
#define GH_NV(a, n) (a)[((size_t)((n) - 1)) * plane + c]     // component n, 1-based

extern "C" __global__ void rrtm_gethml(
    const int im, const int jm, const int nlay,
    const int ivflip, const int iovr, const int llyr,
    const float* __restrict__ plyr,     // [NLAY, JM, IM] layer mean pressure (mb)
    const float* __restrict__ ptop1,    // [4, JM, IM] cloud domain pressure limits (sfc, low, mid, high)
    const float* __restrict__ cldtot,   // [NLAY, JM, IM] total or stratiform cloud fraction
    const float* __restrict__ cldcnv,   // [NLAY, JM, IM] convective cloud fraction
    float* __restrict__ clds,           // [5, JM, IM] low, mid, high, total, bl
    int* __restrict__ mtop,             // [3, JM, IM] top layer index per domain (low, mid, high)
    int* __restrict__ mbot)             // [3, JM, IM] bottom layer index per domain
{
    COLUMN_IJ(im, jm)
    const size_t plane = (size_t)jm * (size_t)im;
    const size_t c = (size_t)j0 * (size_t)im + (size_t)i0;

    // ---- clds = 0 (2213-2222) ----
    for (int n = 1; n <= 5; ++n) GH_NV(clds, n) = 0.0f;
    // mtop/mbot are written on every call: a domain that never closes reads 0, not stale values.
    for (int n = 1; n <= GETHML_NK_CLDS; ++n) { GH_NV(mtop, n) = 0; GH_NV(mbot, n) = 0; }

    // ---- total and bl clouds (2224-2300) ----
    float cl1 = 1.0f;
    float cl2 = 1.0f;

    int kstr, kend, kinc;
    if (ivflip == 0) {                  // input data from toa to sfc (2228-2231)
        kstr = nlay; kend = 1; kinc = -1;
    } else {                            // input data from sfc to toa (2232-2235)
        kstr = 1; kend = nlay; kinc = 1;
    }

    if (iovr == 0) {                    // random overlap (2238-2271)
        for (int k = kstr; (kinc > 0) ? (k <= kend) : (k >= kend); k += kinc) {
            const float ccur = fminf(GETHML_OVCST, fmaxf(GH_LV(cldtot, k), GH_LV(cldcnv, k)));
            if (ccur >= GETHML_CLIMIT) cl1 = cl1 * (1.0f - ccur);
            if (k == llyr) GH_NV(clds, 5) = 1.0f - cl1;        // save bl cloud
        }
        GH_NV(clds, 4) = 1.0f - cl1;                             // save total cloud
    } else {                            // max/ran overlap (2272-2300)
        for (int k = kstr; (kinc > 0) ? (k <= kend) : (k >= kend); k += kinc) {
            const float ccur = fminf(GETHML_OVCST, fmaxf(GH_LV(cldtot, k), GH_LV(cldcnv, k)));
            if (ccur >= GETHML_CLIMIT) {                         // cloudy layer
                cl2 = fminf(cl2, (1.0f - ccur));
            } else {                                             // clear layer
                cl1 = cl1 * cl2;
                cl2 = 1.0f;
            }
            if (k == llyr) GH_NV(clds, 5) = 1.0f - cl1 * cl2;   // save bl cloud
        }
        GH_NV(clds, 4) = 1.0f - cl1 * cl2;                       // save total cloud
    }

    // ---- high, mid and low clouds (2305-2487) ----
    // cl1, cl2 are reset for the second sweep (2306-2307 / 2409-2410).
    cl1 = 0.0f;
    cl2 = 0.0f;
    int kbt1, kbt2, kth1, kth2, idom;

    if (ivflip == 0) {                  // input data from toa to sfc (2305-2404)
        kbt1 = nlay; kbt2 = nlay;
        kth1 = 0;    kth2 = 0;
        idom = 1;
        GH_NV(mbot, 1) = nlay;
        GH_NV(mtop, 1) = nlay;

        for (int k = nlay; k >= 1; --k) {
            int id = idom;
            int id1 = id + 1;

            const float pcur = GH_LV(plyr, k);
            const float ccur = fminf(GETHML_OVCST, fmaxf(GH_LV(cldtot, k), GH_LV(cldcnv, k)));

            float pnxt, cnxt;
            if (k > 1) {
                pnxt = GH_LV(plyr, k - 1);
                cnxt = fminf(GETHML_OVCST, fmaxf(GH_LV(cldtot, k - 1), GH_LV(cldcnv, k - 1)));
            } else {
                pnxt = -1.0f;
                cnxt = 0.0f;
            }

            if (id < GETHML_NK_CLDS && pcur < GH_NV(ptop1, id1)) {
                id = id + 1;
                id1 = id1 + 1;
                idom = id;
            }

            if (ccur >= GETHML_CLIMIT) {
                if (kth2 == 0) kbt2 = k;
                kth2 = kth2 + 1;

                if (iovr == 0) {
                    cl2 = cl2 + ccur - cl2 * ccur;
                } else {
                    cl2 = fmaxf(cl2, ccur);
                }

                if (cnxt < GETHML_CLIMIT || pnxt < GH_NV(ptop1, id1)) {
                    kbt1 = gethml_nint((cl1 * (float)kbt1 + cl2 * (float)kbt2) / (cl1 + cl2));
                    kth1 = gethml_nint((cl1 * (float)kth1 + cl2 * (float)kth2) / (cl1 + cl2));
                    cl1 = cl1 + cl2 - cl1 * cl2;

                    kbt2 = k - 1;
                    kth2 = 0;
                    cl2 = 0.0f;
                }
            }

            if (pnxt < GH_NV(ptop1, id1)) {
                GH_NV(clds, id) = cl1;
                GH_NV(mtop, id) = min(kbt1, kbt1 - kth1 + 1);
                GH_NV(mbot, id) = kbt1;

                cl1 = 0.0f;
                kbt1 = k - 1;
                kth1 = 0;

                if (id1 <= GETHML_NK_CLDS) {
                    GH_NV(mbot, id1) = kbt1;
                    GH_NV(mtop, id1) = kbt1;
                }
            }
        }
    } else {                            // input data from sfc to toa (2405-2487)
        kbt1 = 1; kbt2 = 1;
        kth1 = 0; kth2 = 0;
        idom = 1;
        GH_NV(mbot, 1) = 1;
        GH_NV(mtop, 1) = 1;

        for (int k = 1; k <= nlay; ++k) {
            int id = idom;
            int id1 = id + 1;

            const float pcur = GH_LV(plyr, k);
            const float ccur = fminf(GETHML_OVCST, fmaxf(GH_LV(cldtot, k), GH_LV(cldcnv, k)));

            float pnxt, cnxt;
            if (k < nlay) {
                pnxt = GH_LV(plyr, k + 1);
                cnxt = fminf(GETHML_OVCST, fmaxf(GH_LV(cldtot, k + 1), GH_LV(cldcnv, k + 1)));
            } else {
                pnxt = -1.0f;
                cnxt = 0.0f;
            }

            if (id < GETHML_NK_CLDS && pcur < GH_NV(ptop1, id1)) {
                id = id + 1;
                id1 = id1 + 1;
                idom = id;
            }

            if (ccur >= GETHML_CLIMIT) {
                if (kth2 == 0) kbt2 = k;
                kth2 = kth2 + 1;

                if (iovr == 0) {
                    cl2 = cl2 + ccur - cl2 * ccur;
                } else {
                    cl2 = fmaxf(cl2, ccur);
                }

                if (cnxt < GETHML_CLIMIT || pnxt < GH_NV(ptop1, id1)) {
                    kbt1 = gethml_nint((cl1 * (float)kbt1 + cl2 * (float)kbt2) / (cl1 + cl2));
                    kth1 = gethml_nint((cl1 * (float)kth1 + cl2 * (float)kth2) / (cl1 + cl2));
                    cl1 = cl1 + cl2 - cl1 * cl2;

                    kbt2 = k + 1;
                    kth2 = 0;
                    cl2 = 0.0f;
                }
            }

            if (pnxt < GH_NV(ptop1, id1)) {
                GH_NV(clds, id) = cl1;
                GH_NV(mtop, id) = max(kbt1, kbt1 + kth1 - 1);
                GH_NV(mbot, id) = kbt1;

                cl1 = 0.0f;
                kbt1 = k + 1;
                kth1 = 0;

                if (id1 <= GETHML_NK_CLDS) {
                    GH_NV(mbot, id1) = kbt1;
                    GH_NV(mtop, id1) = kbt1;
                }
            }
        }
    }
}
