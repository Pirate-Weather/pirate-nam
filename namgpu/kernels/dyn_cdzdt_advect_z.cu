// dyn_cdzdt_advect_z.cu -- grand horizontal loop of CDZDT: mass fluxes of height on H points,
// horizontal advection of height into W, and w = 0 on the lateral boundary rows/columns.
//
// NOTICE: NAM source file nmm/module_DYNAMICS_ROUTINES.F90 (NOAA/NCEP/EMC, US Government work,
// no licence file). This file has no author/history block of its own; see NOTICE at the repo root.
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:3140-3235 (nam.v4.2.11, 83df4e13c1).
//
// Fortran routine CDZDT, section "grand horizontal loop" (lines 3140-3235), inside the
// vertical_loop over l = 1..LM. Fortran arrays per CONTRACT.md section 1:
//   A(i,j,l) -> A[(l-1), (j-1), (i-1)] via IDX3; H-point arrays Z, PFX, PFY, PFNE, PFNW, W
//   [LM(+1), JM, IM]; PDO [JM, IM]; DSG2, PDSG1 [LM]; FAH [JM] (grid constant, jds:jde).
//
// Scratch: the Fortran zx, zy, zne, znw are 2-D work arrays recomputed for every l. Here each
// flux is evaluated where it is used, with the same expression and operand order as the Fortran
// (lines 3166-3169). No value is reused from another thread, so the result equals the Fortran
// section term by term.
//
// Loop limits (single full domain, module_DM_PARALLEL.F90 looplimits with ids=1, ide=IM, jds=1, jde=JM):
//   flux loop   : its_b1..ite_h1 = 2..IM,   jts_b1..jte_h1 = 2..JM   (1-based)
//   w update    : its_b1..ite_b1 = 2..IM-1, jts_b1..jte_b1 = 2..JM-1 (1-based)
// All fluxes read by the w update (i in 2..IM-1, j in 2..JM-1) lie inside the flux range.
//
// Boundary zeroing (Fortran 3189-3228): the flags s_bdy, n_bdy, w_bdy, e_bdy are true for the one
// process that owns the whole domain (module_CONTROL.F90:542-545: its==ids, ite==ide, jts==jds,
// jte==jde). The kernel receives them as ints; for the 12 km CONUS parent all four are 1 and
// global is 0 (CONTRACT section 5, global = false). The zeroing runs after the update, as in
// Fortran; each (i,j,l) is written by one thread only, so the order is preserved.
//
// Fortran lines not mapped to a term:
//   OpenMP directives at 3147-3149 and 3230 (ENABLE_SMP only, host threading, disabled here).
//   Line 3160 "dpup = pdsg1(l)": dpup is assigned and never used in this section (dead value).
//   The global-only branch is absent: global = false.

#include "nmmb_common.cuh"

// zx(i,j) = (z(i,j,l) - z(i-1,j,l)) * pfx(i,j,l)                                   (Fortran 3167)
__device__ __forceinline__ float cdzdt_zx(const float* __restrict__ z, const float* __restrict__ pfx,
                                          int i0, int j0, int l0, int im, int jm)
{
    const size_t c = IDX3(i0, j0, l0, im, jm);
    return (z[c] - z[IDX3(i0 - 1, j0, l0, im, jm)]) * pfx[c];
}

// zy(i,j) = (z(i,j,l) - z(i,j-1,l)) * pfy(i,j,l)                                   (Fortran 3168)
__device__ __forceinline__ float cdzdt_zy(const float* __restrict__ z, const float* __restrict__ pfy,
                                          int i0, int j0, int l0, int im, int jm)
{
    const size_t c = IDX3(i0, j0, l0, im, jm);
    return (z[c] - z[IDX3(i0, j0 - 1, l0, im, jm)]) * pfy[c];
}

// zne(i,j) = (z(i,j,l) - z(i-1,j-1,l)) * pfne(i,j,l)                               (Fortran 3169)
__device__ __forceinline__ float cdzdt_zne(const float* __restrict__ z, const float* __restrict__ pfne,
                                           int i0, int j0, int l0, int im, int jm)
{
    const size_t c = IDX3(i0, j0, l0, im, jm);
    return (z[c] - z[IDX3(i0 - 1, j0 - 1, l0, im, jm)]) * pfne[c];
}

// znw(i,j) = (z(i-1,j,l) - z(i,j-1,l)) * pfnw(i,j,l)                               (Fortran 3170)
__device__ __forceinline__ float cdzdt_znw(const float* __restrict__ z, const float* __restrict__ pfnw,
                                           int i0, int j0, int l0, int im, int jm)
{
    const size_t c = IDX3(i0, j0, l0, im, jm);
    return (z[IDX3(i0 - 1, j0, l0, im, jm)] - z[IDX3(i0, j0 - 1, l0, im, jm)]) * pfnw[c];
}

extern "C" __global__ void dyn_cdzdt_advect_z(
    const float* __restrict__ z,       // [LM+1, JM, IM] heights (levels 1..LM used), from dyn-cdzdt-nonhydro
    const float* __restrict__ pfx,     // [LM, JM, IM] mass flux, x direction (PGFORCE)
    const float* __restrict__ pfy,     // [LM, JM, IM] mass flux, y direction
    const float* __restrict__ pfne,    // [LM, JM, IM] mass flux, NE direction
    const float* __restrict__ pfnw,    // [LM, JM, IM] mass flux, NW direction
    const float* __restrict__ fah,     // [JM] grid factor, fah(j)
    const float* __restrict__ dsg2,    // [LM] delta sigmas
    const float* __restrict__ pdsg1,   // [LM] delta pressures
    const float* __restrict__ pdo,     // [JM, IM] sigma range pressure difference (previous step)
    float* __restrict__ w,             // [LM, JM, IM] W, updated in place
    const float dt,                    // dynamics time step (DT, 25 s)
    const int im,                      // IM
    const int jm,                      // JM
    const int lm,                      // LM
    const int global,                  // 0 for regional (global = false)
    const int s_bdy,                   // this process touches the south boundary (1) or not (0)
    const int n_bdy,                   // north boundary
    const int w_bdy,                   // west boundary
    const int e_bdy)                   // east boundary
{
    POINT_IJL   // i0 = x, j0 = y, l0 = z: one thread per (i, j) at level l
    if (i0 >= im || j0 >= jm || l0 >= lm) return;

    const size_t c = IDX3(i0, j0, l0, im, jm);

    // ---- Fortran 3167-3184: advection of height, interior points (1-based i,j in 2..IM-1, 2..JM-1) ----
    if (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2) {
        // fahp = -fah(j)/dt       (Fortran 3177, per j; fah indexed from 1 at jds)
        const float fahp = -fah[j0] / dt;

        // The eight flux terms, same operands as the Fortran expression at (i,j):
        const float zx_ij      = cdzdt_zx (z, pfx,  i0,     j0,     l0, im, jm);   // zx(i,j)
        const float zx_ip1j    = cdzdt_zx (z, pfx,  i0 + 1, j0,     l0, im, jm);   // zx(i+1,j)
        const float zy_ij      = cdzdt_zy (z, pfy,  i0,     j0,     l0, im, jm);   // zy(i,j)
        const float zy_ijp1    = cdzdt_zy (z, pfy,  i0,     j0 + 1, l0, im, jm);   // zy(i,j+1)
        const float zne_ip1jp1 = cdzdt_zne(z, pfne, i0 + 1, j0 + 1, l0, im, jm);   // zne(i+1,j+1)
        const float zne_ij     = cdzdt_zne(z, pfne, i0,     j0,     l0, im, jm);   // zne(i,j)
        const float znw_ijp1   = cdzdt_znw(z, pfnw, i0,     j0 + 1, l0, im, jm);   // znw(i,j+1)
        const float znw_ip1j   = cdzdt_znw(z, pfnw, i0 + 1, j0,     l0, im, jm);   // znw(i+1,j)

        // w = ((zx(i,j)+zx(i+1,j)+zy(i,j)+zy(i,j+1)) + (zne(i+1,j+1)+zne(i,j)+znw(i,j+1)+znw(i+1,j))*0.25)*fahp
        //     / (dsg2(l)*pdo(i,j)+pdsg1(l)) + w(i,j,l)                                (Fortran 3179-3184)
        const float s_flux = (zx_ij + zx_ip1j + zy_ij + zy_ijp1)
                           + (zne_ip1jp1 + zne_ij + znw_ijp1 + znw_ip1j) * 0.25f;
        const float den = dsg2[l0] * pdo[IDX2(i0, j0, im)] + pdsg1[l0];
        w[c] = (s_flux * fahp) / den + w[c];
    }

    // ---- Fortran 3189-3228: w = 0 along the lateral boundaries (not for global) ----
    if (!global) {
        // south: j = jts..jts+1  (1-based 1..2)
        if (s_bdy && j0 <= 1) w[c] = 0.0f;
        // north: j = jte-1..jte  (1-based JM-1..JM)
        if (n_bdy && j0 >= jm - 2) w[c] = 0.0f;
        // west: i = its..its+1   (1-based 1..2)
        if (w_bdy && i0 <= 1) w[c] = 0.0f;
        // east: i = ite-1..ite   (1-based IM-1..IM)
        if (e_bdy && i0 >= im - 2) w[c] = 0.0f;
    }
}
