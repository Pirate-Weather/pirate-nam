// WOOF-NAM two-way nest: nest -> parent feedback in the overlap, every parent step.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Ported to CUDA for WOOF-NAM from nam.v4.2.11 (83df4e13c1):
//   nmm/module_NESTING.F90:12671-12904 GENERATE_2WAY_DATA        -> nest_2way_gen, nest_2way_sfc
//   nmm/module_PARENT_CHILD_CPL_COMP.F90:22032-22458 PARENT_2WAY_UPDATE -> nest_2way_update
//   stencils module_NESTING.F90:107-114 (even ratio: h->H 3, v->V 2, sfc->H 3; odd: 3, 3, 3);
//   child index of a parent update point (CHILDREN_SEND_PARENTS_2WAY_DATA, CPL 12005-12014):
//     IC_H = (I - I_SW)*RATIO + 1, IC_V = IC_H + RATIO/2 (same for J).
// The update point set (CHILD_2WAY_BOOKKEEPING, module_NESTING.F90:12355-12670) is built once on the host
// (namgpu/nest/twoway.py). The ESMF/MPI exchange (CHILDREN_SEND_PARENTS_2WAY_DATA, PARENTS_RECV_CHILD_2WAY_DATA)
// is not ported: one process holds both domains; the box means go into a device buffer the parent reads at the
// start of its next step, as in NMMB.
// Fields fed back (parm/nam_nests.txt, BUILD_2WAY_BUNDLE): T, Q, CW on H points, U, V on V points. PD is not fed
// back; the parent's PD and FIS and the nest's box-mean PD and FIS only set the vertical re-spline.
// As written in PARENT_2WAY_UPDATE, the one surface pair (the nest H-point FIS and PD box means, and the parent's
// H-point FIS(I,J) and PD(I,J)) serves every variable, U and V included.
//
// Layout: [LM][JM][IM] C order, level 0 = model top. buf[var][LM][NPTS]: var 0 T, 1 Q, 2 CW, 3 U, 4 V.
// sfc[NPTS][2]: nest FIS box mean, nest PD box mean on the parent H point.
#include "nmmb_common.cuh"
#include "nest_spline.cuh"

#define IX2(i, j, IM) ((size_t)(j) * (IM) + (i))
#define IX3(i, j, l, IM, JM) (((size_t)(l) * (JM) + (j)) * (IM) + (i))

// GENERATE_2WAY_DATA: box means of the nest variables over an NxN stencil around each parent update point.
// grid (ceil(npts/bx), lm, 5): blockIdx.z = variable.
extern "C" __global__ void nest_2way_gen(
    const float* __restrict__ cT, const float* __restrict__ cQ, const float* __restrict__ cCW,
    const float* __restrict__ cU, const float* __restrict__ cV, int imc, int jmc, int lm,
    int npts, const int* __restrict__ ich, const int* __restrict__ jch, int half_v, int nst_h, int nst_v,
    float* __restrict__ buf)
{
    int np = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y, v = blockIdx.z;
    if (np >= npts || l >= lm) return;
    const float* src = v == 0 ? cT : v == 1 ? cQ : v == 2 ? cCW : v == 3 ? cU : cV;
    const int n = v < 3 ? nst_h : nst_v;
    int ic = ich[np], jc = jch[np];
    if (v >= 3) { ic += half_v; jc += half_v; }       // IC_V = IC_H + RATIO/2
    const int i0 = ic - n / 2, j0 = jc - n / 2;      // N_STENCIL_0 = N_STENCIL/2
    const float recip = 1.0f / (float)(n * n);
    float s = 0.0f;
    for (int j = j0; j < j0 + n; ++j)
        for (int i = i0; i < i0 + n; ++i)
            s += src[IX3(i, j, l, imc, jmc)];
    buf[((size_t)v * lm + l) * npts + np] = s * recip;
}

// GENERATE_2WAY_DATA, INTERPOLATE_SFC branch for the H points: FIS and PD box means, N_STENCIL_0 = (N+1)/2 - 1.
extern "C" __global__ void nest_2way_sfc(
    const float* __restrict__ cFIS, const float* __restrict__ cPD, int imc,
    int npts, const int* __restrict__ ich, const int* __restrict__ jch, int nst_sfc, float* __restrict__ sfc)
{
    int np = blockIdx.x * blockDim.x + threadIdx.x;
    if (np >= npts) return;
    const int n0 = (nst_sfc + 1) / 2 - 1;
    const int i0 = ich[np] - n0, j0 = jch[np] - n0;
    const float recip = 1.0f / (float)(nst_sfc * nst_sfc);
    float pds = 0.0f, fis = 0.0f;
    for (int j = j0; j < j0 + nst_sfc; ++j)
        for (int i = i0; i < i0 + nst_sfc; ++i) {
            pds += cPD[IX2(i, j, imc)];
            fis += cFIS[IX2(i, j, imc)];
        }
    sfc[2 * (size_t)np + 0] = fis * recip;
    sfc[2 * (size_t)np + 1] = pds * recip;
}

// PARENT_2WAY_UPDATE: where either surface is above sea level (FIS > 1), re-spline the nest column from the nest
// mean pressures to the parent's (mass-weighted bottom layers and HYPERBOLA below the nest's lowest mid-layer);
// then parent = WGT_CHILD*nest + (1 - WGT_CHILD)*parent. One thread per parent update column; ip/jp are the
// parent's 0-based indices of the point.
extern "C" __global__ void nest_2way_update(
    float* __restrict__ pT, float* __restrict__ pQ, float* __restrict__ pCW,
    float* __restrict__ pU, float* __restrict__ pV,
    const float* __restrict__ pPD, const float* __restrict__ pFIS, int imp, int jmp, int lm,
    const float* __restrict__ sg1, const float* __restrict__ sg2, float pdtop, float pt,
    int npts, const int* __restrict__ ip, const int* __restrict__ jp,
    const float* __restrict__ buf, const float* __restrict__ sfc, float wgt_child)
{
    int np = blockIdx.x * blockDim.x + threadIdx.x;
    if (np >= npts) return;
    const float wgt_parent = 1.0f - wgt_child;
    const int i = ip[np], j = jp[np];
    const size_t h = IX2(i, j, imp);
    const float fis_c = sfc[2 * (size_t)np], pd_c = sfc[2 * (size_t)np + 1];
    const bool adjust = fis_c > 1.0f || pFIS[h] > 1.0f;
    float pmc[NEST_LMAX + 1], pmp[NEST_LMAX], vx[NEST_LMAX + 1], y2[NEST_LMAX + 1], vo[NEST_LMAX];
    int nlev = lm;
    bool extrap = false;
    if (adjust) {
        float pdtop_pt = sg1[0] * pdtop + pt;
        float hi_c = sg2[0] * pd_c + pdtop_pt;
        float hi_p = sg2[0] * pPD[h] + pdtop_pt;
        for (int l = 0; l < lm; ++l) {
            pdtop_pt = sg1[l + 1] * pdtop + pt;
            float lo = sg2[l + 1] * pd_c + pdtop_pt;
            pmc[l] = 0.5f * (hi_c + lo);
            hi_c = lo;
            lo = sg2[l + 1] * pPD[h] + pdtop_pt;
            pmp[l] = 0.5f * (hi_p + lo);
            hi_p = lo;
        }
        if (pmp[lm - 1] > pmc[lm - 1]) {
            extrap = true;
            nlev = lm + 1;
            pmc[lm] = pmp[lm - 1];
            // the mass-weighted bottom pressures below are shared by every variable (computed once here, as the
            // Fortran recomputes them identically per variable from the same PMID_CHILD)
        }
    }
    float pmc_lm = 0.f, pmc_lm1 = 0.f;
    if (extrap) {
        pmc_lm = 0.5f * (pmc[lm - 2] + pmc[lm - 1]);
        pmc_lm1 = (pmc[lm - 3] + pmc[lm - 2] + pmc[lm - 1]) / 3.0f;
    }
    float* dst[5] = {pT, pQ, pCW, pU, pV};
    const size_t plane = (size_t)imp * jmp;
    for (int v = 0; v < 5; ++v) {
        const float* col = buf + (size_t)v * lm * npts + np;     // stride npts per level
        if (adjust) {
            float xo[NEST_LMAX + 1];
            for (int l = 0; l < nlev; ++l) xo[l] = pmc[l];
            if (!extrap) {
                for (int l = 0; l < lm; ++l) vx[l] = col[(size_t)l * npts];
            } else {
                for (int l = 0; l < lm - 2; ++l) vx[l] = col[(size_t)l * npts];
                float c_lm = col[(size_t)(lm - 1) * npts], c_lm1 = col[(size_t)(lm - 2) * npts],
                      c_lm2 = col[(size_t)(lm - 3) * npts];
                float prod_lm = c_lm * pmc[lm - 1], prod_lm1 = c_lm1 * pmc[lm - 2], prod_lm2 = c_lm2 * pmc[lm - 3];
                vx[lm - 1] = (prod_lm + prod_lm1) / (pmc[lm - 1] + pmc[lm - 2]);
                vx[lm - 2] = (prod_lm + prod_lm1 + prod_lm2) / (pmc[lm - 1] + pmc[lm - 2] + pmc[lm - 3]);
                xo[lm - 1] = pmc_lm;
                xo[lm - 2] = pmc_lm1;
                float r_delp = 1.0f / (xo[lm - 1] - xo[lm - 2]);
                float delp = pmp[lm - 1] - xo[lm - 1];
                float c1 = (vx[lm - 1] - vx[lm - 2]) * r_delp;
                float factor = NEST_HYPER_A / (delp + NEST_HYPER_A);
                vx[lm] = vx[lm - 1] + c1 * delp * factor;
            }
            for (int l = 0; l <= lm; ++l) y2[l] = 0.0f;
            nmmb_spline(nlev, xo, vx, y2, lm, pmp, vo);
        } else {
            for (int l = 0; l < lm; ++l) vo[l] = col[(size_t)l * npts];
        }
        float* a = dst[v];
        for (int l = 0; l < lm; ++l) {
            size_t o = (size_t)l * plane + h;               // V variables: parent V point (i, j)
            a[o] = vo[l] * wgt_child + a[o] * wgt_parent;
        }
    }
}
