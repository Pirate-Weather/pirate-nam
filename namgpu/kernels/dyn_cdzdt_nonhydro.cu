// dyn_cdzdt_nonhydro.cu -- CDZDT setup and the nonhydrostatic equation: height Z of the layers
// and the local dw/dt and w from the mass field.
//
// NOTICE: NAM source file nmm/module_DYNAMICS_ROUTINES.F90 (NOAA/NCEP/EMC, US Government work,
// no licence file). The routine has no author/history block of its own; see NOTICE at the repo root.
//
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:2920-3139 (nam.v4.2.11, 83df4e13c1).
//
// Fortran subroutine cdzdt(global, hydro, lm, dt, dsg2, pdsg1, fah, fis, pd, pdo, psgdt, cw, q,
// rtop, t, pint, dwdt, pdwdt, w, baro, z, pfne, pfnw, pfx, pfy). Covered:
//   2920-2990  constants rg = 1/g, rdt = 1/dt, trog = 2 r/g; interface start zlo = fis*rg, wlo = 0
//              (the Fortran zlo/wlo are 2-D scratch; in this kernel they are per-column registers)
//   2993-3019  nonhydrostatic equation: do l = lm,1,-1 (upward accumulation from the surface)
//              pdwdt = dwdt; dwdt = w; dz, dw from pint(l), pint(l+1), t, q, cw, pd, pdo, rtop;
//              z = dz*0.5 + zlo; w = dw*0.5 + wlo; zlo, wlo advanced to the upper interface
//   3021       if (hydro) return      (config hydro = false; the branch is kept, see below)
//   3040-3139  non-hydrostatic vertical correction on the b1 limits: ttb = 0; for l = 1..lm-1:
//              tta = (z(l+1)-z(l))*psgdt(l)*0.5; w(l) = (tta+ttb)/(dsg2(l)*pdo+pdsg1(l)) + w(l);
//              ttb = tta; then w(lm) = ttb/(dsg2(lm)*pdo+pdsg1(lm)) + w(lm)
//   Not mapped: baro (declared intent(out) in the argument list but never assigned in this
//              routine, so the kernel does not write it); fah, pfne, pfnw, pfx, pfy (declared, not
//              read here: the nonhydrostatic path uses only the arrays listed below); global (the
//              global-only branch is not used in the regional 12 km run, CONTRACT section 2).
//              The ENABLE_SMP OpenMP directives and the looplimits split are host threading only.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   its_h1 = 1 .. ite_h2 = IM and jts_h1 = 1 .. jte_h2 = JM: the whole domain for the nonhydrostatic
//            equation (part A), every column i0 in [0, IM-1], j0 in [0, JM-1].
//   its_b1 = 2 .. ite_b1 = IM-1 and jts_b1 = 2 .. jte_b1 = JM-1 (1-based): i0 in [1, IM-2],
//            j0 in [1, JM-2] for the vertical correction (part B). Outside that, w keeps the
//            part-A value, as in the Fortran.
//
// Thread mapping: one thread per column (CONTRACT section 3, grid_cols). The whole l-recursion of
// both parts runs inside the thread: the column is the only data dependency (zlo, wlo, ttb).
// Every array element is written by exactly one thread, so there is no reduction and no ordering
// between threads.
//
// Layout (CONTRACT section 1): Fortran A(i,j,l) -> IDX3(i-1, j-1, l-1, IM, JM) for 3-D fields;
// interface fields (PINT) have LM+1 levels, PINT(i,j,l) -> level l-1 and PINT(i,j,l+1) -> level l;
// 2-D fields -> IDX2. dsg2, pdsg1 are 1-D [LM], index l-1.
//
// Constants: rg = 1/g and trog = 2 r/g, g = 9.8060226 and r = 287.04 (module_CONSTANTS). The
// literal 0.608 in the dz expression is the Fortran single-precision literal. Single precision
// throughout (kfpt = single); FMA contraction on, IEEE division.
//
// Hydro flag: the Fortran returns after part A when hydro is true. The kernel takes the same flag
// as an argument (hydro != 0 skips part B), so the hydrostatic configuration keeps the Fortran
// meaning. The 12 km NAM run uses hydro = false.

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_cdzdt_nonhydro(
    const int IM,
    const int JM,
    const int LM,
    const int hydro,                         // Fortran hydro (logical): nonzero = hydrostatic
    const float dt,                          // Fortran dt (dynamics time step)
    const float* __restrict__ dsg2,          // DSG2 [LM]
    const float* __restrict__ pdsg1,         // PDSG1 [LM]
    const float* __restrict__ fis,           // FIS [JM,IM] surface geopotential
    const float* __restrict__ pd,            // PD [JM,IM]
    const float* __restrict__ pdo,           // PDO [JM,IM]
    const float* __restrict__ psgdt,         // PSGDT [LM,JM,IM] (levels 1..LM-1 used)
    const float* __restrict__ cw,            // CW [LM,JM,IM]
    const float* __restrict__ q,             // Q [LM,JM,IM]
    const float* __restrict__ rtop,          // RTOP [LM,JM,IM] (rt/p)
    const float* __restrict__ t,             // T [LM,JM,IM]
    const float* __restrict__ pint,          // PINT [LM+1,JM,IM]
    float* __restrict__ dwdt,                // DWDT [LM,JM,IM], updated in place (inout)
    float* __restrict__ pdwdt,               // PDWDT [LM,JM,IM], written (previous DWDT)
    float* __restrict__ w,                   // W [LM,JM,IM], updated in place (inout)
    float* __restrict__ z)                   // Z [LM,JM,IM], written (layer heights)
{
    // One thread per column; the l-loop runs top-down in the Fortran index order. The ranges
    // covered are the full domain (part A), see the header.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= IM * JM) return;
    const int i0 = col % IM;
    const int j0 = col / IM;

    const float rg = 1.0f / NMMB_G;                // rg = 1./g
    const float rdt = 1.0f / dt;                   // rdt = 1./dt
    const float trog = 2.0f * NMMB_R_D / NMMB_G;   // trog = 2.*r/g

    // ---- Part A: nonhydrostatic equation, do l = lm,1,-1 (Fortran 2993-3019) -----------------
    float zlo = fis[IDX2(i0, j0, IM)] * rg;  // zlo(i,j) = fis(i,j)*rg
    float wlo = 0.0f;                        // wlo(i,j) = 0.
    const float pdv = pd[IDX2(i0, j0, IM)];
    const float pdov = pdo[IDX2(i0, j0, IM)];

    for (int l0 = LM - 1; l0 >= 0; --l0) {
        const size_t ix = IDX3(i0, j0, l0, IM, JM);
        pdwdt[ix] = dwdt[ix];                // pdwdt(i,j,l) = dwdt(i,j,l)
        dwdt[ix] = w[ix];                    // dwdt(i,j,l)  = w(i,j,l)

        // dz = (q*0.608+(1.-cw))*t*trog*(dsg2*pd+pdsg1)/(pint(l)+pint(l+1))
        const float dz =
            (q[ix] * 0.608f + (1.0f - cw[ix])) * t[ix] * trog
            * (dsg2[l0] * pdv + pdsg1[l0])
            / (pint[IDX3(i0, j0, l0, IM, JM)] + pint[IDX3(i0, j0, l0 + 1, IM, JM)]);
        // dw = (dz - rtop*(dsg2*pdo+pdsg1)*rg)*rdt
        const float dw = (dz - rtop[ix] * (dsg2[l0] * pdov + pdsg1[l0]) * rg) * rdt;

        const float zup = zlo + dz;
        const float wup = wlo + dw;
        z[ix] = dz * 0.5f + zlo;             // z(i,j,l) = dz*0.5 + zlo
        w[ix] = dw * 0.5f + wlo;             // w(i,j,l) = dw*0.5 + wlo

        zlo = zup;
        wlo = wup;
    }

    // Hydrostatic configuration stops here (Fortran "if(hydro) return").
    if (hydro != 0) return;

    // ---- Part B: vertical correction of w on the b1 limits (Fortran 3040-3139) --------------
    // NMMB _b1 limits for the full domain: 1-based i, j in [2, IM-1], [2, JM-1].
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    float ttb = 0.0f;                        // ttb(i,j) = 0.
    for (int l0 = 0; l0 <= LM - 2; ++l0) {   // Fortran l = 1..lm-1
        const size_t ix = IDX3(i0, j0, l0, IM, JM);
        // tta(i,j) = (z(i,j,l+1)-z(i,j,l))*psgdt(i,j,l)*0.5
        const float tta = (z[IDX3(i0, j0, l0 + 1, IM, JM)] - z[ix]) * psgdt[ix] * 0.5f;
        // w(i,j,l) = (tta+ttb)/(dsg2*pdo+pdsg1) + w(i,j,l)
        w[ix] = (tta + ttb) / (dsg2[l0] * pdov + pdsg1[l0]) + w[ix];
        ttb = tta;                           // ttb(i,j) = tta(i,j)
    }
    // w(i,j,lm) = ttb/(dsg2(lm)*pdo+pdsg1(lm)) + w(i,j,lm)
    const size_t ixl = IDX3(i0, j0, LM - 1, IM, JM);
    w[ixl] = ttb / (dsg2[LM - 1] * pdov + pdsg1[LM - 1]) + w[ixl];
}
