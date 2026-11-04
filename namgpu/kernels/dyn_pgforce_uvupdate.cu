// dyn_pgforce_uvupdate.cu -- PGFORCE first-pass switch and the pressure-gradient u/v tendency
// update (NMMB module_DYNAMICS_ROUTINES.F90 PGFORCE, lines 452-493).
//
// Notice: nmm/module_DYNAMICS_ROUTINES.F90 carries no PROGRAM HISTORY LOG / author block in the
// NAM source at 83df4e13c1; the NAM notice in NOTICE applies (NOAA/NCEP/EMC, US Government work).
//
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:452-493 (nam.v4.2.11, 83df4e13c1).
//
// Fortran source (vertical_loop body, one level l):
//   if(.not.first.or.restart) then                              ! UPDATE = 1
//     rdv = rdyv*dt
//     do j = jts_b1..jte_b2 ; rdu = rdxv(j)*dt
//       do i = its_b1..ite_b2
//         apd  = (pd(i,j)+pd(i+1,j)+pd(i,j+1)+pd(i+1,j+1))*0.25
//         rpdp = 0.3333333333/(dsg2(l)*apd+pdsg1(l))
//         tcu(i,j,l) = -((pgx(i+1,j)+pgx(i+1,j+1)) + (pgne(i+1,j+1)-pgnw(i+1,j+1))*0.5)*rdu*rpdp + tcu(i,j,l)
//         tcv(i,j,l) = -((pgy(i,j+1)+pgy(i+1,j+1)) + (pgne(i+1,j+1)+pgnw(i+1,j+1))*0.5)*rdv*rpdp + tcv(i,j,l)
//     enddo; enddo
//   else                                                        ! UPDATE = 0 (first step, no restart)
//     tcu(i,j,l) = 0 ; tcv(i,j,l) = 0   over the same i, j range
//   endif
//
// Notes on this unit (see the wrapper docstring):
//   - The routine writes the PGF TENDENCIES TCU/TCV, not U/V. U/V are advanced later by
//     UPDATEUV (dyn-updateuv, module_DYNAMICS_ROUTINES.F90 ~1830). cfc/bfc (Adams-Bashforth
//     1.533) do not appear in lines 452-493; they are not used here.
//   - Loop limits from module_DM_PARALLEL.F90 looplimits with ids=its=1, ide=ite=IM (single
//     process), 1-based: its_b1 = 2, ite_b2 = IM-2, jts_b1 = 2, jte_b2 = JM-2. In 0-based C:
//     i0 in [1, IM-3], j0 in [1, JM-3].
//   - pgx/pgy/pgne/pgnw are the 3-D scratch written by dyn-pgforce-pgf (PGX_3D etc.), read at
//     level l0 (blockIdx.z). pd is [JM, IM]. dsg2, pdsg1 are [LM] (level 0 = top).
//   - rdxv(j) is a [JM] vector (1/dxv(j)); rdyv and dt are scalars.
//   - `first`/`restart` are folded into the int UPDATE argument by the wrapper.
//   - No halo exchange, no OpenMP, no print: nothing skipped.
//
// Fortran-to-CUDA review (every term of lines 452-493):
//   apd, rpdp, rdu, rdv, tcu update, tcv update, the else-branch zeroing: all present below.
//
// Launch: kernels.grid_2d(IM, JM, LM) with BLOCK_2D; blockIdx.z = level (l0 = l-1, top = 0).
// Layout: 3-D arrays [LM, JM, IM] (IDX3); PD [JM, IM] (IDX2); PINT not used here.

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_pgforce_uvupdate(
    const float* __restrict__ PGX_3D,    // [LM, JM, IM]  pgf scratch, x     (from dyn-pgforce-pgf)
    const float* __restrict__ PGY_3D,    // [LM, JM, IM]  pgf scratch, y
    const float* __restrict__ PGNE_3D,   // [LM, JM, IM]  pgf scratch, ne
    const float* __restrict__ PGNW_3D,   // [LM, JM, IM]  pgf scratch, nw
    const float* __restrict__ PD,        // [JM, IM]      surface pressure thickness
    const float* __restrict__ RDXV,      // [JM]          1/dxv(j)
    const float* __restrict__ DSG2,      // [LM]          delta sigma (sg2)
    const float* __restrict__ PDSG1,     // [LM]          delta pressure (pdtop-based)
    float* __restrict__ TCU,             // [LM, JM, IM]  u pressure-gradient tendency (in/out)
    float* __restrict__ TCV,             // [LM, JM, IM]  v pressure-gradient tendency (in/out)
    float RDYV,                          // 1/dyv
    float DT,                            // dynamics time step (s)
    int UPDATE,                          // 1 if (.not.first .or. restart), else 0
    int IM, int JM, int LM)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;   // Fortran i-1
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;   // Fortran j-1
    const int l0 = blockIdx.z;                              // Fortran l-1 (0 = top)

    // Fortran its_b1..ite_b2 = 2..IM-2 and jts_b1..jte_b2 = 2..JM-2 (1-based).
    if (i0 < 1 || i0 > IM - 3 || j0 < 1 || j0 > JM - 3 || l0 >= LM) return;

    const size_t c = IDX3(i0, j0, l0, IM, JM);

    if (!UPDATE) {
        // else branch of the first-pass switch: zero the tendencies.
        TCU[c] = 0.0f;
        TCV[c] = 0.0f;
        return;
    }

    const float rdv = RDYV * DT;
    const float rdu = RDXV[j0] * DT;

    const float apd = (PD[IDX2(i0,     j0,     IM)] + PD[IDX2(i0 + 1, j0,     IM)]
                     + PD[IDX2(i0,     j0 + 1, IM)] + PD[IDX2(i0 + 1, j0 + 1, IM)]) * 0.25f;
    const float rpdp = 0.3333333333f / (DSG2[l0] * apd + PDSG1[l0]);

    const size_t cxp  = IDX3(i0 + 1, j0,     l0, IM, JM);   // pgx(i+1,j), pgy(i+1,j)
    const size_t cxpy = IDX3(i0 + 1, j0 + 1, l0, IM, JM);   // (i+1,j+1)
    const size_t cyp  = IDX3(i0,     j0 + 1, l0, IM, JM);   // pgy(i,j+1)

    TCU[c] = -((PGX_3D[cxp] + PGX_3D[cxpy])
              + (PGNE_3D[cxpy] - PGNW_3D[cxpy]) * 0.5f) * rdu * rpdp
              + TCU[c];

    TCV[c] = -((PGY_3D[cyp] + PGY_3D[cxpy])
              + (PGNE_3D[cxpy] + PGNW_3D[cxpy]) * 0.5f) * rdv * rpdp
              + TCV[c];
}
