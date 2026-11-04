// WOOF-NAM unit dyn-hdiff-coef: HDIFF setup and the grand vertical loop that builds the deformation
// coefficient DEF3D and the slope/coast masks FMLX_3D / FMLY_3D consumed by dyn-hdiff-apply.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:2531-2833 (nam.v4.2.11, 83df4e13c1).
//
// Fortran subroutine hdiff (1-based indices, full domain IM x JM x LM, one card, regional, hydro off):
//   [A] l=1..lm, boundary zeroing: def3d(i,1,l)=0 (s_bdy), def3d(i,jm,l)=0 (n_bdy),
//       def3d(1,j,l)=0 (w_bdy), def3d(im,j,l)=0 (e_bdy). On one card all four flags are true
//       (module_CONTROL s_bdy=(jts==jds) and the matching n/w/e tests).
//   [B] vertical_loop_1, l=1..lm, i=2..im-1, j=2..jm-1 (its_b1_h1..ite_b1_h2, jts_b1_h1..jte_b1_h2):
//       deft, defs (deformation), defvp (vertical deformation), q2trm (2tke term, scq2 = 50),
//       defc (regional floor: 0.015*(3-l+1)/3 for l<=3, else 0), defm (9999), defp = sqrt(2*defhp +
//       defvfc*defvp) clipped to [defc, defm]. Written to def3d(i,j,l).
//   [C] vertical_loop_3 slope masks, l=1..lm, i=2..im, j=2..jm (its_b1..ite_h2, jts_b1..jte_h2):
//       fmlx, fmly = 1 unless (l > lpt2 and not hydro); then 1 only if the slope is <= slopec (0.05)
//       or a sea-ice/sea-mask edge is crossed. The hkx/hky sums and the gradient/diffusion terms that
//       follow the masks (Fortran 2834-2915) belong to dyn-hdiff-apply.
// Global-only branch (global=true: defm, defc with lpt2/3) is kept; the global-only SWAPHN/POLEHN
// halo calls after the first loop are not ported (global=false in the 12 km configuration).
// No line of 2531-2833 is skipped except: the OpenMP directives, timef/timers, and the commented-out
// defz diagnostic (no effect on the state).
//
// Index ranges: single process, halo-free. Zero-based thread index i0 = i-1, j0 = j-1, l0 = l-1.
// Fields: U, V, W, Q2, Z(levels 1..lm) and every other 3-D field [LM,JM,IM]; SM, SICE [JM,IM];
// DXV, RARE, RDXH [JM] (index j-1); EPSQ2 [LM] (index l-1); LPT2 scalar.
//
// One thread per (i, j, l): grid_2d launch, grid.z = LM. No host sync, no device allocation.

#include "nmmb_common.cuh"

#define F3(p, i, j, l) (p)[IDX3((i) - 1, (j) - 1, (l) - 1, im, jm)]
#define F2(p, i, j)    (p)[IDX2((i) - 1, (j) - 1, im)]

extern "C" __global__ void dyn_hdiff_coef(
    const float* __restrict__ u,       // U     [LM,JM,IM]
    const float* __restrict__ v,       // V     [LM,JM,IM]
    const float* __restrict__ w,       // W     [LM,JM,IM]
    const float* __restrict__ q2,      // Q2    [LM,JM,IM]
    const float* __restrict__ z,       // Z     [LM(+1),JM,IM], levels 1..LM read
    const float* __restrict__ sm,      // SM    [JM,IM]  sea mask
    const float* __restrict__ sice,    // SICE  [JM,IM]  sea-ice fraction
    const float* __restrict__ dxv,     // DXV   [JM]
    const float* __restrict__ rare,    // RARE  [JM]     1/(dxh(j)*dyh)
    const float* __restrict__ rdxh,    // RDXH  [JM]     1/dxh(j)
    const float* __restrict__ epsq2,   // EPSQ2 [LM]     floor value of 2tke per level
    float* __restrict__ def3d,         // out DEF3D [LM,JM,IM]
    float* __restrict__ fmlx,          // out FMLX_3D [LM,JM,IM]
    float* __restrict__ fmly,          // out FMLY_3D [LM,JM,IM]
    const float dyh,
    const float rdyh,
    const int im,
    const int jm,
    const int lm,
    const int lpt2,
    const int hydro,
    const int global,
    const int JOFF, const int JMG)     // row slab (nmmb_common.cuh): domain tests on the global row
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;

    const int i = i0 + 1;
    const int j = j0 + 1;
    const int l = l0 + 1;

    // Constants of the routine (module_DYNAMICS_ROUTINES parameter block, single precision).
    const float defvfc = 2.0f;     // vertical deformation weight
    const float scq2   = 50.0f;    // 2tke weighting factor
    const float slopec = 0.05f;    // critical slope
    const float defm   = global ? 1.6875e-4f : 9999.0f;   // deformation cap (global / regional)

    // [A] + [B]: boundary rows and columns are zeroed; interior cells get the deformation.
    const int jg1 = j + JOFF;          // global Fortran j
    const bool bdy = (jg1 == 1) || (jg1 == JMG) || (i == 1) || (i == im);
    if (!bdy && NMMB_SLAB_EDGE(j0, jm, jg1 - 1, JMG, 1)) return;   // interior reads j-1..j+1
    if (bdy) {
        F3(def3d, i, j, l) = 0.0f;
    } else {
        float defc;
        if (global) {
            if (l > lpt2 / 3) {
                defc = 0.0f;
            } else {
                defc = defm * 0.01f;
            }
        } else {
            if (l <= 3) {
                defc = 0.015f * (float)(3 - l + 1) / 3.0f;
            } else {
                defc = 0.0f;
            }
        }

        const float rj = rare[j0];
        const float dxvj = dxv[j0];
        const float dxvjm = dxv[j0 - 1];
        const float rdxhj = rdxh[j0];

        const float deft = ((F3(u, i, j - 1, l) + F3(u, i, j, l)
                             - F3(u, i - 1, j - 1, l) - F3(u, i - 1, j, l)) * dyh
                            - (F3(v, i - 1, j, l) + F3(v, i, j, l)) * dxvj
                            + (F3(v, i - 1, j - 1, l) + F3(v, i, j - 1, l)) * dxvjm) * rj;
        const float defs = ((F3(u, i - 1, j, l) + F3(u, i, j, l)) * dxvj
                            - (F3(u, i - 1, j - 1, l) + F3(u, i, j - 1, l)) * dxvjm
                            + (F3(v, i, j - 1, l) + F3(v, i, j, l)
                               - F3(v, i - 1, j - 1, l) - F3(v, i - 1, j, l)) * dyh) * rj;

        float def1 = 0.0f, def2 = 0.0f, def3 = 0.0f, def4 = 0.0f;
        if (!hydro) {
            def1 = (F3(w, i, j, l) - F3(w, i, j - 1, l)) * rdyh;
            def2 = (F3(w, i, j, l) - F3(w, i - 1, j, l)) * rdxhj;
            def3 = (F3(w, i + 1, j, l) - F3(w, i, j, l)) * rdxhj;
            def4 = (F3(w, i, j + 1, l) - F3(w, i, j, l)) * rdyh;
        }

        const float q2v = F3(q2, i, j, l);
        float q2trm = 0.0f;
        if (q2v > epsq2[l0]) {
            q2trm = scq2 * q2v * rj;
        }

        const float defhp = deft * deft + defs * defs + q2trm;
        const float defvp = def1 * def1 + def2 * def2 + def3 * def3 + def4 * def4;

        float defp = sqrtf(defhp * 2.0f + defvp * defvfc);
        defp = fmaxf(defp, defc);
        defp = fminf(defp, defm);
        F3(def3d, i, j, l) = defp;
    }

    // [C]: slope masks, range i=2..im, j=2..jm (its_b1..ite_h2, jts_b1..jte_h2).
    if (i >= 2 && j >= 2) {
        float fx = 1.0f;
        float fy = 1.0f;
        if (l > lpt2 && !hydro) {
            const bool cilinx = (F2(sice, i - 1, j) != F2(sice, i, j))
                             || (F2(sm, i - 1, j) != F2(sm, i, j));
            const bool ciliny = (F2(sice, i, j - 1) != F2(sice, i, j))
                             || (F2(sm, i, j - 1) != F2(sm, i, j));
            const float slopx = fabsf((F3(z, i, j, l) - F3(z, i - 1, j, l)) * rdxh[j0]);
            const float slopy = fabsf((F3(z, i, j, l) - F3(z, i, j - 1, l)) * rdyh);
            fx = (slopx <= slopec || cilinx) ? 1.0f : 0.0f;
            fy = (slopy <= slopec || ciliny) ? 1.0f : 0.0f;
        }
        F3(fmlx, i, j, l) = fx;
        F3(fmly, i, j, l) = fy;
    }
}

#undef F3
#undef F2
