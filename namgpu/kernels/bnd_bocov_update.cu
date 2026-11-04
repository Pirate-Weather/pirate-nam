// bnd_bocov_update.cu -- advance the V-point boundary working objects on the S/N and W/E strips
// by their tendencies: value(1) += tendency(2) * dt, for every 2-D and 3-D V-point boundary
// variable, every strip element and every level.
//
// NOTICE: NAM source file nmm/module_FLTBNDS.F90 (NOAA/NCEP/EMC, US Government work, no licence
// file). Keep the NOAA author headers of that file as notice; see NOTICE at the repo root.
// Ported to CUDA for WOOF-NAM from nmm/module_FLTBNDS.F90:6392-6589 (nam.v4.2.11, 83df4e13c1).
//
// Fortran routine BOCOV(lm, lnsv, dt, nvars_bc_2d_v, nvars_bc_3d_v, bnd_vars_v), update block only.
//   south(ib,jb,[l,]1) = south(ib,jb,[l,]1) + south(ib,jb,[l,]2) * dt
//   same for north, west(ib,jb,[l,]k), east(ib,jb,[l,]k)
//
// Layout (CONTRACT section 1): every Fortran strip array (a fixed number of boundary rows as the
// last-but-one dimension, time/tendency slot k = 1, 2 as the first dimension) is held as a
// C-ordered float32 array with the Fortran indices reversed:
//   S/N 2-D  Fortran (ib, jb, k)    -> [k, jb, ib]    shape (2, LNSV, IM)
//   S/N 3-D  Fortran (ib, jb, l, k) -> [k, l, jb, ib] shape (2, LM, LNSV, IM)
//   W/E 2-D  Fortran (ib, jb, k)    -> [k, jb, ib]    shape (2, JM, LNSV)
//   W/E 3-D  Fortran (ib, jb, l, k) -> [k, l, jb, ib] shape (2, LM, JM, LNSV)
// Here k = 0 is Fortran slot 1 (working value) and k = 1 is Fortran slot 2 (tendency).
//
// Loop limits (module_DM_PARALLEL.F90 looplimits, single full domain ids = 1, ide = IM, jde = JM,
// its = ids, jts = jds):
//   S/N strips: ib from its_h2 = max(its-2, ids) = 1 to min(ite_h2, ide-1) = IM-1  (count IM-1)
//               jb = 1..LNSV (all boundary rows)
//   W/E strips: jb from jts_h2 (W) or jts_h1 (E) = 1 to min(jte_h2, jde-1) = JM-1  (count JM-1)
//               ib = 1..LNSV
// Zero-based: the wrapper passes n_along = IM-1 (S/N) or JM-1 (W/E) and the full strip extents.
//
// Every element is updated once and depends on nothing else, so one thread per element is exact
// and the order of the Fortran nv/l/jb/ib loops does not change the result.
//
// Launch: S/N kernels: x = along-strip (ib), y = boundary row (jb), z = variable (2-D) or
//         variable*LM + level (3-D).  W/E kernels: x = along-strip (jb), y = boundary column (ib),
//         z the same.  The pointer table holds the device addresses of the NVARS arrays of one
//         face (south, north, west or east), in the Fortran order of bnd_vars_v%var_2d/var_3d.
//
// Single precision: dt is kfpt, so the update is float32 with fmad enabled (contraction allowed,
// as in the fast-numerics default of CONTRACT section 3).

#include "nmmb_common.cuh"

// ---- S / N faces, 2-D variables -------------------------------------------------------------
// arr[k, jb, ib] with k in {0,1}, jb in [0, nb), ib in [0, ext).
extern "C" __global__ void bocov_sn_2d(
    const int nvars,                         // number of 2-D V-point variables
    const int nb,                            // LNSV boundary rows (across the strip)
    const int ext,                           // allocated extent along the strip (IM)
    const int n_along,                       // IM-1: ib = 1..IM-1 (Fortran)
    const float dt,                          // dynamics time step
    const unsigned long long* __restrict__ ptrs)   // device pointers, one per variable
{
    const int ib0 = blockIdx.x * blockDim.x + threadIdx.x;   // along the strip
    const int jb0 = blockIdx.y * blockDim.y + threadIdx.y;   // boundary row
    const int v   = blockIdx.z;                              // variable
    if (v >= nvars || jb0 >= nb || ib0 >= n_along) return;
    float* a = (float*)(ptrs[v]);
    const size_t i1 = ((size_t)0 * nb + jb0) * ext + ib0;    // value slot (Fortran k = 1)
    const size_t i2 = ((size_t)1 * nb + jb0) * ext + ib0;    // tendency slot (Fortran k = 2)
    a[i1] = a[i1] + a[i2] * dt;
}

// ---- S / N faces, 3-D variables -------------------------------------------------------------
// arr[k, l, jb, ib]; grid z = variable * LM + level.
extern "C" __global__ void bocov_sn_3d(
    const int nvars,
    const int lm,                            // LM levels
    const int nb,                            // LNSV
    const int ext,                           // IM
    const int n_along,                       // IM-1
    const float dt,
    const unsigned long long* __restrict__ ptrs)
{
    const int ib0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int jb0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int vz  = blockIdx.z;
    const int v   = vz / lm;
    const int l0  = vz - v * lm;
    if (v >= nvars || jb0 >= nb || ib0 >= n_along) return;
    float* a = (float*)(ptrs[v]);
    const size_t i1 = (((size_t)0 * lm + l0) * nb + jb0) * ext + ib0;
    const size_t i2 = (((size_t)1 * lm + l0) * nb + jb0) * ext + ib0;
    a[i1] = a[i1] + a[i2] * dt;
}

// ---- W / E faces, 2-D variables -------------------------------------------------------------
// arr[k, jb, ib] with jb in [0, ext) (the along-strip index is jb here), ib in [0, nb).
extern "C" __global__ void bocov_we_2d(
    const int nvars,
    const int nb,                            // LNSV boundary columns (across the strip)
    const int ext,                           // allocated extent along the strip (JM)
    const int n_along,                       // JM-1: jb = 1..JM-1 (Fortran)
    const float dt,
    const unsigned long long* __restrict__ ptrs)
{
    const int jb0 = blockIdx.x * blockDim.x + threadIdx.x;   // along the strip
    const int ib0 = blockIdx.y * blockDim.y + threadIdx.y;   // boundary column
    const int v   = blockIdx.z;
    if (v >= nvars || ib0 >= nb || jb0 >= n_along) return;
    float* a = (float*)(ptrs[v]);
    const size_t i1 = ((size_t)0 * ext + jb0) * nb + ib0;
    const size_t i2 = ((size_t)1 * ext + jb0) * nb + ib0;
    a[i1] = a[i1] + a[i2] * dt;
}

// ---- W / E faces, 3-D variables -------------------------------------------------------------
// arr[k, l, jb, ib]; grid z = variable * LM + level.
extern "C" __global__ void bocov_we_3d(
    const int nvars,
    const int lm,
    const int nb,                            // LNSV
    const int ext,                           // JM
    const int n_along,                       // JM-1
    const float dt,
    const unsigned long long* __restrict__ ptrs)
{
    const int jb0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int ib0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int vz  = blockIdx.z;
    const int v   = vz / lm;
    const int l0  = vz - v * lm;
    if (v >= nvars || ib0 >= nb || jb0 >= n_along) return;
    float* a = (float*)(ptrs[v]);
    const size_t i1 = (((size_t)0 * lm + l0) * ext + jb0) * nb + ib0;
    const size_t i2 = (((size_t)1 * lm + l0) * ext + jb0) * nb + ib0;
    a[i1] = a[i1] + a[i2] * dt;
}
