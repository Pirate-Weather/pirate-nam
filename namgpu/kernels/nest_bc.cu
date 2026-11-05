// WOOF-NAM concurrent nest: parent -> nest lateral boundary values from the live parent, every parent step.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Ported to CUDA for WOOF-NAM from nmm/module_PARENT_CHILD_CPL_COMP.F90 (nam.v4.2.11, 83df4e13c1):
//   20489-21029 PARENT_UPDATE_CHILD_PSFC   -> nest_bc_psfc
//   21030-21680 PARENT_UPDATE_CHILD_BNDRY  -> nest_bc_h (T, Q, CW on H points), nest_bc_v (U, V on V points)
//   21681-22031 PRESSURE_ON_NEST_BNDRY_V   -> the 4-point means of PD inside nest_bc_v
// and the tendency of nmm/module_SOLVER_GRID_COMP.F90:8155-9298 UPDATE_BC_TENDS:
//   BND(...,2) = (BND_DATA - BND(...,1)) * RECIP, RECIP = 1/(DT*PARENT_CHILD_TIME_RATIO),
// written straight into the nest's boundary strips (slot 1), so no file and no host copy is involved.
// The ESMF/MPI exchange around these routines (CHILDREN_RECV_PARENT_DATA, PARENTS_SEND_CHILD_DATA) is not
// ported: one process holds both domains.
//
// Layout (CuPy C order, 0-based): parent and nest 2-D fields [JM][IM], 3-D [LM][JM][IM], level 0 = model top
// (Fortran L = 1). SG1/SG2 [LM+1] interfaces, PSGML1/SGML2 [LM] mid-layers (PMID = PSGML1 + SGML2*PD).
// V point (i, j) sits NE of H point (i, j) (PD_V = 0.25*(PD(i,j)+PD(i+1,j)+PD(i,j+1)+PD(i+1,j+1))).
// One thread per boundary point. The point tables (nest i, j; parent SW corner iw, js; bilinear weights
// SW, SE, NW, NE; strip side and offset) are built once on the host from
// PARENT_TO_CHILD_INTERP_FACTORS (CPL 15871-16761), see namgpu/nest/twoway.py.
// Strip arrays are the nest ctx.bc entries "<S|N|W|E>_<VAR>", [2][NL][lines][along]: slot 0 the carried value
// (advanced by bnd-bocoh/bocov-update), slot 1 the tendency written here. ptr[] holds the slot-0 base of each
// (var, side); off[] the point's offset in the level-0 plane; lstr[side] the level stride; slot 1 = +LM*lstr.
#include "nmmb_common.cuh"
#include "nest_spline.cuh"

#define IX2(i, j, IM) ((size_t)(j) * (IM) + (i))
#define IX3(i, j, l, IM, JM) (((size_t)(l) * (JM) + (j)) * (IM) + (i))

__device__ __forceinline__ float bil(const float* a, size_t sw, size_t se, size_t nw, size_t ne, const float* w)
{
    return w[0] * a[sw] + w[1] * a[se] + w[2] * a[nw] + w[3] * a[ne];
}

// PARENT_UPDATE_CHILD_PSFC: the parent's surface pressure at the nest's own terrain height on the nest
// boundary H points (interface geopotentials built hydrostatically from T, Q, FIS at the 4 parent points;
// quadratic extrapolation where the nest terrain is below the parent's). pdb[j][i] = PSFC - PT.
extern "C" __global__ void nest_bc_psfc(
    const float* __restrict__ pPD, const float* __restrict__ pFIS,
    const float* __restrict__ pT, const float* __restrict__ pQ,
    int imp, int jmp, int lm,
    const float* __restrict__ sg1, const float* __restrict__ sg2, float pdtop, float pt,
    const float* __restrict__ cFIS, int imc,
    int n, const int* __restrict__ ic, const int* __restrict__ jc,
    const int* __restrict__ iw, const int* __restrict__ js, const float* __restrict__ w4,
    float* __restrict__ pdb)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float* w = w4 + 4 * (size_t)k;
    const int i0 = iw[k], j0 = js[k];
    const size_t sw = IX2(i0, j0, imp), se = IX2(i0 + 1, j0, imp);
    const size_t nw = IX2(i0, j0 + 1, imp), ne = IX2(i0 + 1, j0 + 1, imp);
    const size_t plane = (size_t)imp * jmp;
    float pint[NEST_LMAX + 1], phi[NEST_LMAX + 1];

    float px_sw = pPD[sw] + pt, px_se = pPD[se] + pt, px_nw = pPD[nw] + pt, px_ne = pPD[ne] + pt;
    pint[lm] = w[0] * px_sw + w[1] * px_se + w[2] * px_nw + w[3] * px_ne;
    float log_pbot = logf(pint[lm]);
    phi[lm] = bil(pFIS, sw, se, nw, ne, w);
    for (int l = lm - 1; l >= 0; --l) {                 // Fortran L = NLEV..1, upward
        float pdtop_pt = pdtop * sg1[l] + pt;
        px_sw = sg2[l] * pPD[sw] + pdtop_pt;
        px_se = sg2[l] * pPD[se] + pdtop_pt;
        px_nw = sg2[l] * pPD[nw] + pdtop_pt;
        px_ne = sg2[l] * pPD[ne] + pdtop_pt;
        pint[l] = w[0] * px_sw + w[1] * px_se + w[2] * px_nw + w[3] * px_ne;
        const size_t o = (size_t)l * plane;
        float t = bil(pT + o, sw, se, nw, ne, w);
        float q = bil(pQ + o, sw, se, nw, ne, w);
        float log_ptop = logf(pint[l]);
        phi[l] = phi[l + 1] + NMMB_R_D * t * (1.0f + NMMB_P608 * q) * (log_pbot - log_ptop);
        log_pbot = log_ptop;
    }
    const float fis_c = cFIS[IX2(ic[k], jc[k], imc)];
    float psfc = pint[lm];
    if (fis_c < phi[lm]) {                               // nest terrain below the parent's
        float c1 = (pint[lm] - pint[lm - 1]) / (phi[lm] - phi[lm - 1]);
        float c2 = (c1 - (pint[lm - 1] - pint[lm - 2]) / (phi[lm - 1] - phi[lm - 2])) / (phi[lm] - phi[lm - 2]);
        float d = fis_c - phi[lm];
        psfc = c2 * d * d + c1 * d + pint[lm];
    } else {
        for (int l = lm - 1; l >= 0; --l) {              // Fortran L = LM..1: PHI(L) is the top of layer L
            if (fis_c < phi[l]) {
                float lp1 = logf(pint[l + 1]);
                float dl = (lp1 - logf(pint[l])) / (phi[l + 1] - phi[l]);
                psfc = expf(lp1 + dl * (fis_c - phi[l + 1]));
                break;
            }
        }
    }
    pdb[IX2(ic[k], jc[k], imc)] = psfc - pt;
}

// The vertical half of PARENT_UPDATE_CHILD_BNDRY for one column: parent mid-layer pressures pmid[] and the
// variable col[] (both horizontally interpolated) -> the nest mid-layer pressures of pdb_c, with the
// mass-weighted HYPERBOLA extrapolation below the parent's lowest mid-layer. out[LM].
__device__ __forceinline__ void column_to_nest(int lm, const float* pmid, const float* col, float pdb_c,
                                               const float* psgml1, const float* sgml2, float* out)
{
    float pin[NEST_LMAX + 1], vin[NEST_LMAX + 1], y2[NEST_LMAX + 1], pch[NEST_LMAX];
    for (int l = 0; l < lm; ++l) pch[l] = psgml1[l] + sgml2[l] * pdb_c;
    for (int l = 0; l < lm; ++l) { pin[l] = pmid[l]; vin[l] = col[l]; }
    int nlev = lm;
    if (pch[lm - 1] > pin[lm - 1]) {
        nlev = lm + 1;
        pin[lm] = pch[lm - 1];
        float r_delp = 1.0f / (pin[lm - 1] - pin[lm - 2]);
        float delp = pch[lm - 1] - pin[lm - 1];
        float c1 = (vin[lm - 1] - vin[lm - 2]) * r_delp;
        float factor = NEST_HYPER_A / (delp + NEST_HYPER_A);
        vin[lm] = vin[lm - 1] + c1 * delp * factor;
    }
    for (int l = 0; l <= lm; ++l) y2[l] = 0.0f;
    nmmb_spline(nlev, pin, vin, y2, lm, pch, out);
}

// T, Q, CW (and the PD tendency) on the nest boundary H points. ptr: [var*4 + side], var 0 PD, 1 T, 2 Q, 3 CW.
extern "C" __global__ void nest_bc_h(
    const float* __restrict__ pPD, const float* __restrict__ pT, const float* __restrict__ pQ,
    const float* __restrict__ pCW, int imp, int jmp, int lm,
    const float* __restrict__ sg1, const float* __restrict__ sg2,
    const float* __restrict__ psgml1, const float* __restrict__ sgml2, float pdtop, float pt,
    const float* __restrict__ pdb, int imc,
    int n, const int* __restrict__ ic, const int* __restrict__ jc,
    const int* __restrict__ iw, const int* __restrict__ js, const float* __restrict__ w4,
    const int* __restrict__ side, const long long* __restrict__ off,
    const unsigned long long* __restrict__ ptr, const long long* __restrict__ lstr, float recip)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float* w = w4 + 4 * (size_t)k;
    const int i0 = iw[k], j0 = js[k];
    const size_t sw = IX2(i0, j0, imp), se = IX2(i0 + 1, j0, imp);
    const size_t nw = IX2(i0, j0 + 1, imp), ne = IX2(i0 + 1, j0 + 1, imp);
    const size_t plane = (size_t)imp * jmp;
    const int sd = side[k];
    const long long ls = lstr[sd];
    const long long o0 = off[k];
    float pint[NEST_LMAX + 1], pmid[NEST_LMAX], col[NEST_LMAX + 1], out[NEST_LMAX];

    // interface pressures (PINT_INTERP_LO) and mid-layer pressures (PMID_INTERP), bottom up as the Fortran
    pint[lm] = w[0] * (pPD[sw] + pt) + w[1] * (pPD[se] + pt) + w[2] * (pPD[nw] + pt) + w[3] * (pPD[ne] + pt);
    for (int l = lm - 1; l >= 0; --l) {
        float pdtop_pt = sg1[l] * pdtop + pt;
        float hi = w[0] * (sg2[l] * pPD[sw] + pdtop_pt) + w[1] * (sg2[l] * pPD[se] + pdtop_pt)
                 + w[2] * (sg2[l] * pPD[nw] + pdtop_pt) + w[3] * (sg2[l] * pPD[ne] + pdtop_pt);
        pmid[l] = 0.5f * (hi + pint[l + 1]);
        pint[l] = hi;
    }
    const float pdb_c = pdb[IX2(ic[k], jc[k], imc)];

    // PD: the boundary PD is the parent's PD on the nest terrain (PDB)
    {
        float* a = (float*)ptr[0 * 4 + sd];
        a[ls + o0] = (pdb_c - a[o0]) * recip;          // 2-D strip: slot stride = lines*along = lstr
    }
    const float* src[3] = {pT, pQ, pCW};
    for (int v = 0; v < 3; ++v) {
        const float* s = src[v];
        for (int l = 0; l < lm; ++l) col[l] = bil(s + (size_t)l * plane, sw, se, nw, ne, w);
        if (v == 0 && col[lm - 1] < col[lm - 2]) {     // 'T-nestbc': mass-weighted lowest two layers
            float dp1 = pint[lm] - pint[lm - 1], dp2 = pint[lm - 1] - pint[lm - 2], dp3 = pint[lm - 2] - pint[lm - 3];
            float p1 = col[lm - 1] * dp1, p2 = col[lm - 2] * dp2, p3 = col[lm - 3] * dp3;
            col[lm - 1] = (p1 + p2 + p3) / (dp1 + dp2 + dp3);
            col[lm - 2] = (p2 + p3) / (dp2 + dp3);
        }
        column_to_nest(lm, pmid, col, pdb_c, psgml1, sgml2, out);
        float* a = (float*)ptr[(v + 1) * 4 + sd];
        const long long slot = (long long)lm * ls;
        for (int l = 0; l < lm; ++l) {
            long long o = o0 + (long long)l * ls;
            a[slot + o] = (out[l] - a[o]) * recip;
        }
    }
}

// U, V on the nest boundary V points. Parent PD on V points and the nest PDB on V points are 4-point means of the
// H values (PRESSURE_ON_NEST_BNDRY_V; PD_V of the parent as computed there). iw/js are parent V indices.
// ptr: [var*4 + side], var 0 U, 1 V.
extern "C" __global__ void nest_bc_v(
    const float* __restrict__ pPD, const float* __restrict__ pU, const float* __restrict__ pV,
    int imp, int jmp, int lm,
    const float* __restrict__ sg1, const float* __restrict__ sg2,
    const float* __restrict__ psgml1, const float* __restrict__ sgml2, float pdtop, float pt,
    const float* __restrict__ pdb, int imc,
    int n, const int* __restrict__ ic, const int* __restrict__ jc,
    const int* __restrict__ iw, const int* __restrict__ js, const float* __restrict__ w4,
    const int* __restrict__ side, const long long* __restrict__ off,
    const unsigned long long* __restrict__ ptr, const long long* __restrict__ lstr, float recip)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float* w = w4 + 4 * (size_t)k;
    const int i0 = iw[k], j0 = js[k];
    const size_t sw = IX2(i0, j0, imp), se = IX2(i0 + 1, j0, imp);
    const size_t nw = IX2(i0, j0 + 1, imp), ne = IX2(i0 + 1, j0 + 1, imp);
    const size_t plane = (size_t)imp * jmp;
    const int sd = side[k];
    const long long ls = lstr[sd];
    const long long o0 = off[k];
    float pint[NEST_LMAX + 1], pmid[NEST_LMAX], col[NEST_LMAX + 1], out[NEST_LMAX];

    auto pdv = [&](int i, int j) -> float {
        return 0.25f * (pPD[IX2(i, j, imp)] + pPD[IX2(i + 1, j, imp)] + pPD[IX2(i, j + 1, imp)]
                        + pPD[IX2(i + 1, j + 1, imp)]);
    };
    const float d_sw = pdv(i0, j0), d_se = pdv(i0 + 1, j0), d_nw = pdv(i0, j0 + 1), d_ne = pdv(i0 + 1, j0 + 1);
    pint[lm] = w[0] * (d_sw + pt) + w[1] * (d_se + pt) + w[2] * (d_nw + pt) + w[3] * (d_ne + pt);
    for (int l = lm - 1; l >= 0; --l) {
        float pdtop_pt = sg1[l] * pdtop + pt;
        float hi = w[0] * (sg2[l] * d_sw + pdtop_pt) + w[1] * (sg2[l] * d_se + pdtop_pt)
                 + w[2] * (sg2[l] * d_nw + pdtop_pt) + w[3] * (sg2[l] * d_ne + pdtop_pt);
        pmid[l] = 0.5f * (hi + pint[l + 1]);
        pint[l] = hi;
    }
    const int ci = ic[k], cj = jc[k];
    const float pdb_c = (pdb[IX2(ci, cj, imc)] + pdb[IX2(ci + 1, cj, imc)] + pdb[IX2(ci, cj + 1, imc)]
                         + pdb[IX2(ci + 1, cj + 1, imc)]) * 0.25f;
    const float* src[2] = {pU, pV};
    for (int v = 0; v < 2; ++v) {
        const float* s = src[v];
        for (int l = 0; l < lm; ++l) col[l] = bil(s + (size_t)l * plane, sw, se, nw, ne, w);
        column_to_nest(lm, pmid, col, pdb_c, psgml1, sgml2, out);
        float* a = (float*)ptr[v * 4 + sd];
        const long long slot = (long long)lm * ls;
        for (int l = 0; l < lm; ++l) {
            long long o = o0 + (long long)l * ls;
            a[slot + o] = (out[l] - a[o]) * recip;
        }
    }
}
