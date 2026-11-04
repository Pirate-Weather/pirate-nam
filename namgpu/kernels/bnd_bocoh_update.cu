// WOOF-NAM unit bnd-bocoh-update: BOCOH setup. Advances the boundary working objects on the H
// points of the S/N and W/E strips by their tendencies: working += tendency * dt.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module header kept as the notice:
//   module module_fltbnds  (NMMB boundary handling, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_FLTBNDS.F90:5064-5354 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine bocoh, lines 5064-5354), H-point boundary variables only:
//   [S] if(s_bdy): for each 2-d var nv:  jb=1..lnsh, ib=its_h2..ite_h2:
//         var_2d(nv)%south(ib,jb,1) = var_2d(nv)%south(ib,jb,1) + var_2d(nv)%south(ib,jb,2)*dt
//       3-d vars: same with l=1..lm inside (south(ib,jb,l,k)); 4-d vars: nl=lbnd..ubnd outside l.
//   [N] if(n_bdy): identical update on the north strips.
//   [W] if(w_bdy): jb=jts_h2..jte_h2, ib=1..lnsh; same update on west strips.
//   [E] if(e_bdy): identical update on the east strips.
//   The ib/jb loops cover the full strip: its_h2 = max(its-2,ids) = 1 and ite_h2 = min(ite+2,ide)
//   = IM for the one-card layout (module_DM_PARALLEL looplimits); jts_h2..jte_h2 = 1..JM.
//   Each element update reads only its own working and tendency entries, so the order of the
//   S/N/W/E blocks and of the variables is immaterial; the kernels run them in any order.
//   NOT in this unit: the blend block (Fortran 5355-5962) is bnd-bocoh-blend, which reads the
//   blend weights wh(k) (bocoh_wh in bnd_bocoh_blend.cu) and writes PD, T, Q, CW, Q2, PINT and
//   the corners; the smoothing block (5963-6387) is bnd-bocoh-smooth and is dead while nsmud = 0.
//   wq(k) is initialised at 5157-5160 and never read in BOCOH. lines, nsmud and the loop counters
//   have no effect on the working update.
//
// Layout (CONTRACT.md section 1, Fortran (i,j,...) -> CuPy [..., j, i], zero-based):
//   2-d strip  south/north: [2, lnsh, IM]          west/east: [2, JM, lnsh]   (k = 0 working, 1 tendency)
//   3-d strip  south/north: [2, LM, lnsh, IM]      west/east: [2, LM, JM, lnsh]
//   4-d strip  south/north: [NL, 2, LM, lnsh, IM]  west/east: [NL, 2, LM, JM, lnsh]
//   with NL = ubnd - lbnd + 1 of that variable (nl index shifted by lbnd).
// The wrapper passes, for each edge and rank class, a table of device pointers (one per variable),
// plus ni (fast index extent) and nj (slow index extent). Element (k, [l,] j, i) lives at
// k*(lm*nj*ni) + l*(nj*ni) + j*ni + i for 3-d, and the 2-d form drops l.
// Halo: Fortran allocates SOUTH(IMS:IME,...) with halo columns outside 1..IM. Only ib = 1..IM
// (strip width ib = 1..lnsh on W/E) is stored and read, because one card gives its_h2 = 1 and
// ite_h2 = IM. The halo is not stored; a multi-card layout must add its offset explicitly.
// No host sync, no device allocation; one launch per (edge, rank class).

#include "nmmb_common.cuh"

// 2-d strips: grid (ni, nj, nvar), blockIdx.z selects the variable.
extern "C" __global__ void bocoh_strip_2d(
    const int nvar, const int ni, const int nj,
    const unsigned long long* __restrict__ tbl, const float dt)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int nv = blockIdx.z;
    if (nv >= nvar || i0 >= ni || j0 >= nj) return;
    float* p = (float*)(size_t)tbl[nv];
    const size_t plane = (size_t)nj * (size_t)ni;
    const size_t c = (size_t)j0 * (size_t)ni + (size_t)i0;
    p[c] = p[c] + p[plane + c] * dt;      // working += tendency * dt
}

// 3-d strips: grid (ni, nj, lm), blockIdx.z is the level; the variables loop inside the thread.
extern "C" __global__ void bocoh_strip_3d(
    const int nvar, const int ni, const int nj, const int lm,
    const unsigned long long* __restrict__ tbl, const float dt)
{
    POINT_IJL
    if (i0 >= ni || j0 >= nj || l0 >= lm) return;
    const size_t plane = (size_t)nj * (size_t)ni;
    const size_t s = (size_t)lm * plane;                       // stride of k (working/tendency)
    const size_t c = (size_t)l0 * plane + (size_t)j0 * (size_t)ni + (size_t)i0;
    for (int nv = 0; nv < nvar; ++nv) {
        float* p = (float*)(size_t)tbl[nv];
        p[c] = p[c] + p[s + c] * dt;
    }
}

// 4-d strips: grid (ni, nj, lm); each variable has NL = nlc[nv] entries in the outermost index.
extern "C" __global__ void bocoh_strip_4d(
    const int nvar, const int ni, const int nj, const int lm,
    const unsigned long long* __restrict__ tbl, const int* __restrict__ nlc, const float dt)
{
    POINT_IJL
    if (i0 >= ni || j0 >= nj || l0 >= lm) return;
    const size_t plane = (size_t)nj * (size_t)ni;
    const size_t s = (size_t)lm * plane;
    const size_t c = (size_t)l0 * plane + (size_t)j0 * (size_t)ni + (size_t)i0;
    for (int nv = 0; nv < nvar; ++nv) {
        float* p = (float*)(size_t)tbl[nv];
        for (int n = 0; n < nlc[nv]; ++n) {
            const size_t b = (size_t)n * 2 * s + c;
            p[b] = p[b] + p[b + s] * dt;
        }
    }
}
