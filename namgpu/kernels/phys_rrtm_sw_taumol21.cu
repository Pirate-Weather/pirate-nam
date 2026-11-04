// phys_rrtm_sw_taumol21.cu -- TAUMOL21: SW band 21 (6150-7700 cm-1) g-point optical depths and
// Rayleigh optical depths of the NMMB RRTM shortwave (RRTM group). One thread per (i, j) column;
// the layer loop k = 1..nlay runs inside the thread; every term of the Fortran routine is present.
//
// NOAA notice: phys/radsw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, US government work
// (no licence file; see NOTICE). The sw-rrtm3 package header of that file is kept as the notice:
// !!!!!  sw-rrtm3 radiation package description                                              !!!!!
// !   this package includes ncep's modifications of the rrtm-sw radiation code from aer inc.   !
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:3948-4080 (SUBROUTINE taumol21).
// Tables: module_radsw_kgb21_nmmb (phys/radsw_datatb_nmmb.f:10915-14755), extracted by
// tools/extract_rrtm_sw_kgb21.py into namgpu/data/rrtm_sw/kgb21.npz and passed in by the wrapper.
// Constants written as literals: rayl (module_radsw_kgb21, radsw_datatb_nmmb.f:10945), strrat(21)
// (radsw_datatb_nmmb.f:1108-1110, sixth value of the band-16..29 list), nspa(21) = 9 and
// nspb(21) = 5 (radsw_main_nmmb.f:308-309), NG21 = 10, NS21 = 44 (radsw_param_nmmb.f:159-175),
// eps = 1e-6 and oneminus = 1 - eps (radsw_main_nmmb.f:291-292), f_one = 1 (radsw_main_nmmb.f:299).
//
// Index conventions (CONTRACT.md section 1, Fortran A(i,j,l) -> C [l, j, i]):
//   colmol, fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac      [nlay, JM, IM]
//   jp, jt, jt1, indself, indfor                                                [nlay, JM, IM] int32
//   colamt                                                                      [MAXGAS=6, nlay, JM, IM]
//                                                                               (planes 0 and 1 used:
//                                                                                h2o and co2)
//   laytrop                                                                     [JM, IM] int32
//   absa [585, NG21], absb [1175, NG21], selfref [10, NG21], forref [4, NG21]  (g index fastest;
//                                                                               Fortran absa(ind, ig))
//   taug, taur                                                                  [NOUT, nlay, JM, IM];
//                                                                               the NG21 planes this
//                                                                               routine writes are
//                                                                               plane0 .. plane0+9
//                                                                               (plane0 = 44 for the
//                                                                               full 112-g-point array,
//                                                                               0 for an NG21-plane array).
//
// Translation notes:
//   - The Fortran driver (taumol, radsw_main_nmmb.f:3236-3257) computes id0/id1 with the lower
//     formula where k <= laytrop and the upper formula where k > laytrop. taumol21 reads those values;
//     this kernel recomputes them per layer with the same rule, so the upper loop at k == laytrop
//     reads the lower-formula id0/id1, as the Fortran does.
//   - taumol21 runs two loops over the chunk. The lower loop writes k <= laytrop (WHERE k .LE.
//     laytrop); the upper loop, DO k = MINVAL(laytrop)+1, NLAY with WHERE k .GE. laytrop, then
//     overwrites k == laytrop. The kernel therefore uses the lower expression for k < laytrop and the
//     upper expression for k >= laytrop, per column. The one case that differs is a column whose
//     laytrop equals the minimum laytrop of its CHK chunk: the upper loop does not reach k == laytrop
//     there, so the lower value stays. That depends on chunk composition (a CPU chunking detail), so
//     it is not reproduced.
//   - taur (tauray = colmol * rayl) is written for every layer k = 1..nlay, as in the Fortran.
//   - No solar source is computed here; the band-21 solar flux is the driver's sfluxzen (taumol
//     case default, radsw_main_nmmb.f:3300 onward), outside this routine.
//   - Fortran MOD(specmult, 1.0) is fmodf(specmult, 1.0f); INT() truncates toward zero, as the C cast.
//   - Single precision everywhere (kind_taum = 4 in machine.f), fmad on, no host sync, no allocation.

#include "nmmb_common.cuh"

#define NG21_SW     10
#define NS21_SW     44               // first 0-based g-point of band 21 in the 112-point array
#define MSA21_SW    585
#define MSB21_SW    1175
#define NSPA21_SW   9
#define NSPB21_SW   5
#define RAYL21_SW   9.41e-09f        // module_radsw_kgb21 rayl, radsw_datatb_nmmb.f:10945
#define STRRAT21_SW 4.53210e-03f     // strrat(21), radsw_datatb_nmmb.f:1108-1110
#define ONEMINUS_SW (1.0f - 1.0e-6f)  // oneminus = 1.0 - eps, eps = 1.0e-6 (radsw_main_nmmb.f:291-292); float32 expression, equal to 0.999999f
#define F_ONE_SW    1.0f             // radsw_main_nmmb.f:299

extern "C" __global__ void rrtm_sw_taumol21(
    const float* __restrict__ colmol,
    const float* __restrict__ colamt,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const int*   __restrict__ jp,
    const int*   __restrict__ jt,
    const int*   __restrict__ jt1,
    const int*   __restrict__ laytrop,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const int*   __restrict__ indself,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const int*   __restrict__ indfor,
    const float* __restrict__ absa,
    const float* __restrict__ absb,
    const float* __restrict__ selfref,
    const float* __restrict__ forref,
    int nlay, int IM, int JM, int plane0,
    float* __restrict__ taug,
    float* __restrict__ taur)
{
    COLUMN_IJ(IM, JM)

    const int lt = laytrop[IDX2(i0, j0, IM)];

    for (int l0 = 0; l0 < nlay; ++l0) {
        const size_t c = IDX3(i0, j0, l0, IM, JM);      // [l, j, i] offset of this layer
        const int k1 = l0 + 1;                           // Fortran k
        const float colh = colamt[IDX4(i0, j0, l0, 0, IM, JM, nlay)];   // colamt(:,k,1), h2o
        const float colc = colamt[IDX4(i0, j0, l0, 1, IM, JM, nlay)];   // colamt(:,k,2), co2

        // taur: tauray = colmol * rayl for every layer (k = 1..nlay).
        const float tauray = colmol[c] * RAYL21_SW;
        for (int jg = 0; jg < NG21_SW; ++jg)
            taur[IDX4(i0, j0, l0, plane0 + jg, IM, JM, nlay)] = tauray;

        // Driver id0/id1 (radsw_main_nmmb.f:3251-3257): lower formula where k <= laytrop, upper
        // formula where k > laytrop. taumol21 uses these values in both of its loops.
        const int jpv = jp[c], jtv = jt[c], jt1v = jt1[c];
        int id0, id1;
        if (k1 <= lt) {
            id0 = ((jpv - 1) * 5 + (jtv - 1)) * NSPA21_SW;
            id1 = (jpv * 5 + (jt1v - 1)) * NSPA21_SW;
        } else {
            id0 = ((jpv - 13) * 5 + (jtv - 1)) * NSPB21_SW;
            id1 = ((jpv - 12) * 5 + (jt1v - 1)) * NSPB21_SW;
        }

        const float f00 = fac00[c], f01 = fac01[c], f10 = fac10[c], f11 = fac11[c];

        if (k1 < lt) {
            // ---- lower loop (k < laytrop): absa table, water vapour self and foreign continuum
            const float speccomb = colh + STRRAT21_SW * colc;
            const float specmult = 8.0f * fminf(ONEMINUS_SW, colh / speccomb);
            const int js = 1 + (int)specmult;
            const float fs = fmodf(specmult, F_ONE_SW);
            const float fs1 = F_ONE_SW - fs;
            const float fac000 = fs1 * f00, fac010 = fs1 * f10, fac100 = fs * f00, fac110 = fs * f10;
            const float fac001 = fs1 * f01, fac011 = fs1 * f11, fac101 = fs * f01, fac111 = fs * f11;

            const int ind01 = max(1, min(MSA21_SW, id0 + js));
            const int ind02 = min(MSA21_SW, ind01 + 1);
            const int ind03 = min(MSA21_SW, ind01 + 9);
            const int ind04 = min(MSA21_SW, ind01 + 10);
            const int ind11 = max(1, min(MSA21_SW, id1 + js));
            const int ind12 = min(MSA21_SW, ind11 + 1);
            const int ind13 = min(MSA21_SW, ind11 + 9);
            const int ind14 = min(MSA21_SW, ind11 + 10);

            const int inds = indself[c];
            const int indf = indfor[c];
            const int indsp = inds + 1;
            const int indfp = indf + 1;
            const float sfac = selffac[c], sfrac = selffrac[c];
            const float ffac = forfac[c], ffrac = forfrac[c];

            for (int jg = 0; jg < NG21_SW; ++jg) {
                const float a01 = absa[(size_t)(ind01 - 1) * NG21_SW + jg];
                const float a02 = absa[(size_t)(ind02 - 1) * NG21_SW + jg];
                const float a03 = absa[(size_t)(ind03 - 1) * NG21_SW + jg];
                const float a04 = absa[(size_t)(ind04 - 1) * NG21_SW + jg];
                const float a11 = absa[(size_t)(ind11 - 1) * NG21_SW + jg];
                const float a12 = absa[(size_t)(ind12 - 1) * NG21_SW + jg];
                const float a13 = absa[(size_t)(ind13 - 1) * NG21_SW + jg];
                const float a14 = absa[(size_t)(ind14 - 1) * NG21_SW + jg];
                const float s_lo = selfref[(size_t)(inds - 1) * NG21_SW + jg];
                const float s_hi = selfref[(size_t)(indsp - 1) * NG21_SW + jg];
                const float f_lo = forref[(size_t)(indf - 1) * NG21_SW + jg];
                const float f_hi = forref[(size_t)(indfp - 1) * NG21_SW + jg];

                const float tg = speccomb
                    * (fac000 * a01 + fac100 * a02 + fac010 * a03 + fac110 * a04
                     + fac001 * a11 + fac101 * a12 + fac011 * a13 + fac111 * a14)
                    + colh * (sfac * (s_lo + sfrac * (s_hi - s_lo))
                            + ffac * (f_lo + ffrac * (f_hi - f_lo)));
                taug[IDX4(i0, j0, l0, plane0 + jg, IM, JM, nlay)] = tg;
            }
        }

        if (k1 >= lt) {
            // ---- upper loop (k >= laytrop): absb table, foreign continuum only
            const float speccomb = colh + STRRAT21_SW * colc;
            const float specmult = 4.0f * fminf(ONEMINUS_SW, colh / speccomb);
            const int js = 1 + (int)specmult;
            const float fs = fmodf(specmult, F_ONE_SW);
            const float fs1 = F_ONE_SW - fs;
            const float fac000 = fs1 * f00, fac010 = fs1 * f10, fac100 = fs * f00, fac110 = fs * f10;
            const float fac001 = fs1 * f01, fac011 = fs1 * f11, fac101 = fs * f01, fac111 = fs * f11;

            const int ind01 = max(1, min(MSB21_SW, id0 + js));
            const int ind02 = min(MSB21_SW, ind01 + 1);
            const int ind03 = min(MSB21_SW, ind01 + 5);
            const int ind04 = min(MSB21_SW, ind01 + 6);
            const int ind11 = max(1, min(MSB21_SW, id1 + js));
            const int ind12 = min(MSB21_SW, ind11 + 1);
            const int ind13 = min(MSB21_SW, ind11 + 5);
            const int ind14 = min(MSB21_SW, ind11 + 6);

            const int indf = indfor[c];
            const int indfp = indf + 1;
            const float ffac = forfac[c], ffrac = forfrac[c];

            for (int jg = 0; jg < NG21_SW; ++jg) {
                const float b01 = absb[(size_t)(ind01 - 1) * NG21_SW + jg];
                const float b02 = absb[(size_t)(ind02 - 1) * NG21_SW + jg];
                const float b03 = absb[(size_t)(ind03 - 1) * NG21_SW + jg];
                const float b04 = absb[(size_t)(ind04 - 1) * NG21_SW + jg];
                const float b11 = absb[(size_t)(ind11 - 1) * NG21_SW + jg];
                const float b12 = absb[(size_t)(ind12 - 1) * NG21_SW + jg];
                const float b13 = absb[(size_t)(ind13 - 1) * NG21_SW + jg];
                const float b14 = absb[(size_t)(ind14 - 1) * NG21_SW + jg];
                const float f_lo = forref[(size_t)(indf - 1) * NG21_SW + jg];
                const float f_hi = forref[(size_t)(indfp - 1) * NG21_SW + jg];

                const float tg = speccomb
                    * (fac000 * b01 + fac100 * b02 + fac010 * b03 + fac110 * b04
                     + fac001 * b11 + fac101 * b12 + fac011 * b13 + fac111 * b14)
                    + colh * ffac * (f_lo + ffrac * (f_hi - f_lo));
                taug[IDX4(i0, j0, l0, plane0 + jg, IM, JM, nlay)] = tg;
            }
        }
    }
}
