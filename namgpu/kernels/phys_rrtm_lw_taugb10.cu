// phys_rrtm_lw_taugb10.cu -- WOOF-NAM unit phys-rrtm-lw-taugb10: subroutine taugb10, the RRTM
// longwave band 10 (1390-1480 cm-1; low key h2o, high key h2o) optical depths and Planck fractions.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOTICE: NOAA / NCEP NAM source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone). Header of
// phys/radlw_main_nmmb.f (lw-rrtm3 package, NCEP modifications of RRTM-LW from AER; nam.v4.2.11,
// 83df4e13c1), kept as the notice:
//   lw-rrtm3 radiation package description. this package includes ncep's modifications of the
//   rrtm-lw radiation code from aer inc.
// Ported to CUDA for WOOF-NAM from phys/radlw_main_nmmb.f:5317-5387 (nam.v4.2.11, 83df4e13c1).
//
// Fortran coverage (whole subroutine, 5317-5387):
//   use module_radlw_kgb10          -> k-tables absa, absb, selfref, forref, fracrefa, fracrefb
//                                      (kgb10.npz from radlw_datatb.f:27968-28421, see the wrapper)
//   laytrop loop over k=1..laytrop   -> lower-atmosphere loop, l0 in [0, laytrop)
//     ind0, ind1, inds, indf, ind0p, ind1p, indsp, indfp   -> same index arithmetic, 1-based
//     tauself  = selffac*(selfref(ig,inds) + selffrac*(selfref(ig,indsp)-selfref(ig,inds)))
//     taufor   = forfac*(forref(ig,indf) + forfrac*(forref(ig,indfp)-forref(ig,indf)))
//     taug(ns10+ig,k) = colamt(k,1)*(fac00*absa(ig,ind0)+fac10*absa(ig,ind0p)
//                       +fac01*absa(ig,ind1)+fac11*absa(ig,ind1p)) + tauself + taufor
//     fracs(ns10+ig,k) = fracrefa(ig)
//   k=laytrop+1..nlay               -> upper-atmosphere loop, l0 in [laytrop, nlay)
//     ind0 = ((jp-13)*5+(jt-1))*nspb(10)+1, ind1 = ((jp-12)*5+(jt1-1))*nspb(10)+1
//     taufor only (no self continuum above the tropopause)
//     taug(ns10+ig,k) = colamt(k,1)*(fac00*absb(ig,ind0)+...) + taufor
//     fracs(ns10+ig,k) = fracrefb(ig)
//
// Constants (module values, radlw_param_nmmb.f:138-160, radlw_main_nmmb.f:282-285):
//   NG10 = 6 g-points in band 10, NS10 = 108 band-10 offset (ns10 + ig is the g-point, ig 1-based),
//   nspa(10) = 1 and nspb(10) = 1 (one reference atmosphere per band-10 pressure/temperature cell).
//   Fortran integer indices (jp, jt, jt1, indself, indfor) are 1-based, as setcoef produces them.
//
// Index conventions in this kernel (the 0-based C form of the Fortran 1-based indices):
//   absa, absb:   C [ind, ig], row-major, so absa(ig,ind) is absa[(ind-1)*NG10 + (ig-1)].
//   selfref:      C [inds, ig]; forref C [indf, ig]; fracrefa/fracrefb C [ig].
//   Input arrays [nlay, JM, IM] use IDX3(i0,j0,l0,IM,JM); colamt is [MAXGAS=6, nlay, JM, IM] and
//   only colamt(k,1) (the h2o slab, n = 0) is read, so its index is the same IDX3 offset.
//   taug, fracs: [nlay, NGPT_LW=140, JM, IM]; the band-10 g-points are written at g = NS10+ig-1.
//
// Vertical order (review gap 3): this unit runs inside radlw on radlw's INTERNAL order, the same
// order phys_rrtm_lw_setcoef.cu produces: index 0 = the LOWEST layer, Fortran k = l0+1 (RRTM order).
// It reads jp, jt, jt1, indself, indfor, colamt, fac*, selffac, forfac at l0 = k-1 and writes taug
// and fracs at the same l0. There is NO flip inside this unit. The NAM top-first order (CONTRACT
// section 1) is applied outside radlw. OPEN GAP, not closed here: phys_rrtm_lw_rtrn.cu reads
// Fortran layer k at slot LM-k (top-first), which does not match this order. The gas-sum step that
// feeds rtrn is phys-rrtm-lw-taumol (status todo), so the flip cannot be checked until it lands.
// Owner: taumol / rtrn. Until then the band-10 layers are in RRTM order, as setcoef is.
//   The other 124 g-points of taug/fracs belong to other bands and are not written here.
//
// Layout note: one thread per (i, j) column. Each thread runs the full l-loop, with the lower and
// upper branches selected per column by its own laytrop, exactly as the Fortran does.
//
// Precision (review gap 4): float everywhere. The pinned NAM build is NOT single precision. Its
// configure FFLAGS carry -r8 and SINGLE_PREC is not defined (sorc/nam_nems_nmmb.fd/src/atmos/phys/
// machine.f: kind_phys = 8), so the Fortran taugb10 is double. The port is single by design (fast
// numerics). This is a declared divergence and goes in the unit receipt.
// Expression order is the Fortran order (left-to-right sums), FMA contraction is left to NVRTC
// (--fmad=true).

#include "nmmb_common.cuh"

#define NG10_BAND 6
#define NS10_OFF  108
#define NGPT_LW   140
#define NSPA10    1
#define NSPB10    1

extern "C" __global__ void rrtm_lw_taugb10(
    const int im,
    const int jm,
    const int nlay,
    const int* __restrict__ laytrop,        // [JM, IM]
    const int* __restrict__ jp,             // [nlay, JM, IM]
    const int* __restrict__ jt,             // [nlay, JM, IM]
    const int* __restrict__ jt1,            // [nlay, JM, IM]
    const float* __restrict__ colamt,       // [MAXGAS, nlay, JM, IM]; slab n=0 (h2o) read
    const float* __restrict__ fac00,        // [nlay, JM, IM]
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const int* __restrict__ indself,        // [nlay, JM, IM], Fortran 1-based
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const int* __restrict__ indfor,         // [nlay, JM, IM], Fortran 1-based
    const float* __restrict__ absa,         // [MSA10=65, NG10]
    const float* __restrict__ absb,         // [MSB10=235, NG10]
    const float* __restrict__ selfref,      // [MSF10=10, NG10]
    const float* __restrict__ forref,       // [MFR10=4, NG10]
    const float* __restrict__ fracrefa,     // [NG10]
    const float* __restrict__ fracrefb,     // [NG10]
    float* __restrict__ taug,               // [nlay, NGPT_LW, JM, IM]
    float* __restrict__ fracs)              // [nlay, NGPT_LW, JM, IM]
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= im * jm) return;
    const int i0 = col % im;
    const int j0 = col / im;

    const size_t plane = (size_t)jm * (size_t)im;     // stride of one layer in [l, j, i]
    const size_t ij = (size_t)j0 * (size_t)im + (size_t)i0;
    // laytrop is the Fortran count in [0, nlay]. Review gap 5: Fortran has no check. This clamp only
    // stops a bad count from indexing outside the layer range. Valid inputs are unchanged.
    const int lay = min(max(laytrop[ij], 0), nlay);   // IDX2(i0, j0, IM)

    // Output index of [l0, g0, j0, i0] in the [nlay, NGPT_LW, JM, IM] arrays.
#define OUTIDX(l0_, g0_) ((((size_t)(l0_)) * NGPT_LW + (size_t)(g0_)) * plane + ij)
    // Input index of [l0, j0, i0] in the [nlay, JM, IM] arrays.
#define INIDX(l0_) ((size_t)(l0_) * plane + ij)

    // --- lower atmosphere: k = 1..laytrop (l0 = 0..laytrop-1)
    for (int l0 = 0; l0 < lay; ++l0) {
        const size_t c = INIDX(l0);
        const int jpk = jp[c];
        const int jtk = jt[c];
        const int jt1k = jt1[c];
        const int inds = indself[c];
        const int indf = indfor[c];

        // Fortran: ind0 = ((jp-1)*5 + (jt-1))*nspa(10) + 1 ; ind1 = (jp*5 + (jt1-1))*nspa(10) + 1
        const int ind0 = ((jpk - 1) * 5 + (jtk - 1)) * NSPA10 + 1;
        const int ind1 = (jpk * 5 + (jt1k - 1)) * NSPA10 + 1;
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;
        const int indsp = inds + 1;
        const int indfp = indf + 1;

        const float cl = colamt[c];                   // colamt(k,1)
        const float fc00 = fac00[c], fc01 = fac01[c], fc10 = fac10[c], fc11 = fac11[c];
        const float sfac = selffac[c], sfrac = selffrac[c];
        const float ffac = forfac[c], ffrac = forfrac[c];

        for (int ig = 0; ig < NG10_BAND; ++ig) {      // ig = ig0 + 1
            // Fortran selfref(ig,inds), selfref(ig,indsp): C row = index - 1
            const float sref0 = selfref[(size_t)(inds - 1) * NG10_BAND + ig];
            const float sref1 = selfref[(size_t)(indsp - 1) * NG10_BAND + ig];
            const float tauself = sfac * (sref0 + sfrac * (sref1 - sref0));

            const float fref0 = forref[(size_t)(indf - 1) * NG10_BAND + ig];
            const float fref1 = forref[(size_t)(indfp - 1) * NG10_BAND + ig];
            const float taufor = ffac * (fref0 + ffrac * (fref1 - fref0));

            const float a00 = absa[(size_t)(ind0 - 1) * NG10_BAND + ig];
            const float a10 = absa[(size_t)(ind0p - 1) * NG10_BAND + ig];
            const float a01 = absa[(size_t)(ind1 - 1) * NG10_BAND + ig];
            const float a11 = absa[(size_t)(ind1p - 1) * NG10_BAND + ig];

            const size_t g = (size_t)(NS10_OFF + ig);
            taug[OUTIDX(l0, g)] = cl * (fc00 * a00 + fc10 * a10 + fc01 * a01 + fc11 * a11)
                                  + tauself + taufor;
            fracs[OUTIDX(l0, g)] = fracrefa[ig];
        }
    }

    // --- upper atmosphere: k = laytrop+1..nlay (l0 = laytrop..nlay-1)
    for (int l0 = lay; l0 < nlay; ++l0) {
        const size_t c = INIDX(l0);
        const int jpk = jp[c];
        const int jtk = jt[c];
        const int jt1k = jt1[c];
        const int indf = indfor[c];

        // Fortran: ind0 = ((jp-13)*5 + (jt-1))*nspb(10) + 1 ; ind1 = ((jp-12)*5 + (jt1-1))*nspb(10) + 1
        const int ind0 = ((jpk - 13) * 5 + (jtk - 1)) * NSPB10 + 1;
        const int ind1 = ((jpk - 12) * 5 + (jt1k - 1)) * NSPB10 + 1;
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;
        const int indfp = indf + 1;

        const float cl = colamt[c];                   // colamt(k,1)
        const float fc00 = fac00[c], fc01 = fac01[c], fc10 = fac10[c], fc11 = fac11[c];
        const float ffac = forfac[c], ffrac = forfrac[c];

        for (int ig = 0; ig < NG10_BAND; ++ig) {
            const float fref0 = forref[(size_t)(indf - 1) * NG10_BAND + ig];
            const float fref1 = forref[(size_t)(indfp - 1) * NG10_BAND + ig];
            const float taufor = ffac * (fref0 + ffrac * (fref1 - fref0));

            const float b00 = absb[(size_t)(ind0 - 1) * NG10_BAND + ig];
            const float b10 = absb[(size_t)(ind0p - 1) * NG10_BAND + ig];
            const float b01 = absb[(size_t)(ind1 - 1) * NG10_BAND + ig];
            const float b11 = absb[(size_t)(ind1p - 1) * NG10_BAND + ig];

            const size_t g = (size_t)(NS10_OFF + ig);
            taug[OUTIDX(l0, g)] = cl * (fc00 * b00 + fc10 * b10 + fc01 * b01 + fc11 * b11)
                                  + taufor;
            fracs[OUTIDX(l0, g)] = fracrefb[ig];
        }
    }

#undef OUTIDX
#undef INIDX
}
