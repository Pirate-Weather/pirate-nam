// dyn_pgforce_pgf.cu -- pressure-gradient force components behind H points, with the
// nonhydrostatic correction (NMMB module_DYNAMICS_ROUTINES.F90 PGFORCE, lines 249-289).
//
// Notice: nmm/module_DYNAMICS_ROUTINES.F90 carries no PROGRAM HISTORY LOG / author block in the
// NAM source at 83df4e13c1; the NAM notice in NOTICE applies (NOAA/NCEP/EMC, US Government work).
//
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:249-289 (nam.v4.2.11, 83df4e13c1).
//
// Fortran source (vertical_loop body, one level l; all levels are independent once the
// geopotential unit dyn-pgforce-geopot has filled FIM_3D, APEL_3D, DFI_3D):
//   ppx  = (fim(i,j)-fim(i-1,j))*(pint(i,j,l+1)+pint(i-1,j,l+1)-pint(i,j,l)-pint(i-1,j,l))*0.5
//   pcx  = (dfi(i-1,j)+dfi(i,j))*(apel(i,j)-apel(i-1,j))            pgx = ppx+pcx
//   ppy  = (fim(i,j)-fim(i,j-1))*(pint(i,j,l+1)+pint(i,j-1,l+1)-pint(i,j,l)-pint(i,j-1,l))*0.5
//   pcy  = (dfi(i,j-1)+dfi(i,j))*(apel(i,j)-apel(i,j-1))            pgy = ppy+pcy
//   ppne = (fim(i,j)-fim(i-1,j-1))*(pint(i-1,j-1,l+1)+pint(i,j,l+1)-pint(i-1,j-1,l)-pint(i,j,l))*0.5
//   ppnw = (fim(i-1,j)-fim(i,j-1))*(pint(i-1,j,l+1)+pint(i,j-1,l+1)-pint(i-1,j,l)-pint(i,j-1,l))*0.5
//   pcne = (dfi(i-1,j-1)+dfi(i,j))*(apel(i,j)-apel(i-1,j-1))
//   pcnw = (dfi(i,j-1)+dfi(i-1,j))*(apel(i-1,j)-apel(i,j-1))
//   pgne = ppne+pcne                                                   pgnw = ppnw+pcnw
//
// Loop limits (module_DM_PARALLEL.F90 looplimits, single process, ids=its=1, ide=ite=IM):
//   pcx/pgx  : i = its_b1..ite_h2 = 2..IM,  j = jts..jte_h2 = 1..JM   (i0 >= 1)
//   pcy/pgy  : i = its..ite_h2    = 1..IM,  j = jts_b1..jte_h2 = 2..JM (j0 >= 1)
//   pcne/pgne, pcnw/pgnw: i = 2..IM, j = 2..JM                       (i0 >= 1 and j0 >= 1)
// The Fortran "pgx/pgy/pgne/pgnw" are 2-D scratch reused every level. Here they are 3-D scratch
// (PGX_3D, PGY_3D, PGNE_3D, PGNW_3D, [LM, JM, IM]) so that all levels run in one launch; the
// divcor and uvupdate units read the level slice PGX_3D[l0] where Fortran reads pgx(i,j) at l.
//
// Fortran-to-CUDA review (every term of lines 249-289):
//   - the 2-D copies apel/fim/dfi of apel_3d/fim_3d/dfi_3d (lines 240-246 region) are read
//     directly from the 3-D arrays: the copy is an identity, no term dropped;
//   - pcx/pcy/pcne/pcnw (3-D outputs) and pgx/pgy/pgne/pgnw (scratch) are all written;
//   - PINT is read at levels l and l+1 (interface arrays have LM+1 levels, PINT[0] = PT);
//   - no halo exchange, no OpenMP, no print: nothing skipped.
//   - `first`, `global`, `dt`, `ntimestep` are not referenced in lines 249-289 (the first-pass
//     switch is in uvupdate, lines 452+; the global divergence branch is in divcor).
//
// Launch: kernels.grid_2d(IM, JM, LM) with BLOCK_2D; blockIdx.z = level (l0 = l-1, model top = 0).
// Layout: every 3-D array is [LM, JM, IM] (IDX3), PINT is [LM+1, JM, IM].

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_pgforce_pgf(
    const float* __restrict__ PINT,      // [LM+1, JM, IM]  pressure at interfaces (Pa)
    const float* __restrict__ FIM_3D,    // [LM, JM, IM]    geopotential at layer middle (from geopot)
    const float* __restrict__ APEL_3D,   // [LM, JM, IM]    pressure at layer middle (from geopot)
    const float* __restrict__ DFI_3D,    // [LM, JM, IM]    delta geopotential / 2 (from geopot)
    float* __restrict__ PCX,             // [LM, JM, IM]    second term, x
    float* __restrict__ PCY,             // [LM, JM, IM]    second term, y
    float* __restrict__ PCNE,            // [LM, JM, IM]    second term, ne
    float* __restrict__ PCNW,            // [LM, JM, IM]    second term, nw
    float* __restrict__ PGX_3D,          // [LM, JM, IM]    pgf scratch, x
    float* __restrict__ PGY_3D,          // [LM, JM, IM]    pgf scratch, y
    float* __restrict__ PGNE_3D,         // [LM, JM, IM]    pgf scratch, ne
    float* __restrict__ PGNW_3D,         // [LM, JM, IM]    pgf scratch, nw
    int IM, int JM, int LM)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;   // Fortran i-1
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;   // Fortran j-1
    const int l0 = blockIdx.z;                              // Fortran l-1 (0 = top)
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;

    const size_t c = IDX3(i0, j0, l0, IM, JM);

#define PI3(ii, jj, ll) PINT[IDX3((ii), (jj), (ll), IM, JM)]
#define FI(ii, jj)      FIM_3D[IDX3((ii), (jj), l0, IM, JM)]
#define AP(ii, jj)      APEL_3D[IDX3((ii), (jj), l0, IM, JM)]
#define DF(ii, jj)      DFI_3D[IDX3((ii), (jj), l0, IM, JM)]

    // ---- x: pcx, pgx (Fortran i = 2..IM, j = 1..JM) ------------------------------------
    if (i0 >= 1) {
        const float ppx = (FI(i0, j0) - FI(i0 - 1, j0))
                        * (PI3(i0, j0, l0 + 1) + PI3(i0 - 1, j0, l0 + 1)
                         - PI3(i0, j0, l0) - PI3(i0 - 1, j0, l0)) * 0.5f;
        const float pcx = (DF(i0 - 1, j0) + DF(i0, j0)) * (AP(i0, j0) - AP(i0 - 1, j0));
        PCX[c] = pcx;
        PGX_3D[c] = ppx + pcx;
    }

    // ---- y: pcy, pgy (Fortran i = 1..IM, j = 2..JM) ------------------------------------
    if (j0 >= 1) {
        const float ppy = (FI(i0, j0) - FI(i0, j0 - 1))
                        * (PI3(i0, j0, l0 + 1) + PI3(i0, j0 - 1, l0 + 1)
                         - PI3(i0, j0, l0) - PI3(i0, j0 - 1, l0)) * 0.5f;
        const float pcy = (DF(i0, j0 - 1) + DF(i0, j0)) * (AP(i0, j0) - AP(i0, j0 - 1));
        PCY[c] = pcy;
        PGY_3D[c] = ppy + pcy;
    }

    // ---- ne and nw: pcne, pcnw, pgne, pgnw (Fortran i = 2..IM, j = 2..JM) --------------
    if (i0 >= 1 && j0 >= 1) {
        const float ppne = (FI(i0, j0) - FI(i0 - 1, j0 - 1))
                         * (PI3(i0 - 1, j0 - 1, l0 + 1) + PI3(i0, j0, l0 + 1)
                          - PI3(i0 - 1, j0 - 1, l0) - PI3(i0, j0, l0)) * 0.5f;
        const float ppnw = (FI(i0 - 1, j0) - FI(i0, j0 - 1))
                         * (PI3(i0 - 1, j0, l0 + 1) + PI3(i0, j0 - 1, l0 + 1)
                          - PI3(i0 - 1, j0, l0) - PI3(i0, j0 - 1, l0)) * 0.5f;
        const float pcne = (DF(i0 - 1, j0 - 1) + DF(i0, j0)) * (AP(i0, j0) - AP(i0 - 1, j0 - 1));
        const float pcnw = (DF(i0, j0 - 1) + DF(i0 - 1, j0)) * (AP(i0 - 1, j0) - AP(i0, j0 - 1));
        PCNE[c] = pcne;
        PCNW[c] = pcnw;
        PGNE_3D[c] = ppne + pcne;
        PGNW_3D[c] = ppnw + pcnw;
    }

#undef PI3
#undef FI
#undef AP
#undef DF
}
