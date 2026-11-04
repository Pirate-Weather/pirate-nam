// dyn_ddamp_damp.cu -- divergence damping of U and V (NMMB DDAMP, divergence-damping branch).
//
// Notice: module_DYNAMICS_ROUTINES.F90 in NAM nam.v4.2.11 (MatthewPyle-NOAA/NAM @ 83df4e13c1,
// NMMB dynamics, NOAA/NCEP/EMC) has no author or history header block of its own; the NOAA
// source notice in NOTICE applies. NAM is US Government work with no licence file.
//
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:1000-1036 (nam.v4.2.11, 83df4e13c1).
//
// Two kernels, both launched every call (every step):
//
// ddamp_apd -- the ddamp preamble, Fortran lines 890-898, executed on EVERY ddamp call (it sits
//   before the extmod test, so it is not part of the extmode-only branch):
//     do j=2,JM-2 ; do i=2,IM-2
//       apd(i,j)=((pd(i,j)+pd(i+1,j)+pd(i,j+1)+pd(i+1,j+1))*cfc
//                +(pdo(i,j)+pdo(i+1,j)+pdo(i,j+1)+pdo(i+1,j+1))*bfc)*0.25
//   cfc=1.533, bfc=1.-cfc (module_DYNAMICS_ROUTINES.F90:75-76). PD and PDO change every step, so
//   APD must be recomputed every step. A once-seeded APD would be the step-1 value on every later
//   step.
//
// ddamp_damp -- Fortran lines 1003-1036 (the ELSE branch of the extmod test at lines 904-997):
//   1003-1015  fcim  = freerun ? (dpl<dpb ? fint*dpl+dfac : 1.) : assimfc,
//              dpl = sg1(l+1)*pdtop + sg2(l+1)*10000.
//   1017       dvd   = ddmpv
//   1018-1029  do j=jts_b1,jte_b2 ; dud = ddmpu(j)
//   1020-1027  rdpd = fcim/(dsg2(l)*apd(i,j)+pdsg1(l))
//              u(i,j,l) = (div(i+1,j,l)+div(i+1,j+1,l)-div(i,j,l)-div(i,j+1,l))*dud*rdpd + u(i,j,l)
//              v(i,j,l) = (div(i,j+1,l)+div(i+1,j+1,l)-div(i,j,l)-div(i+1,j,l))*dvd*rdpd + v(i,j,l)
//
// NOTE: extmod is the .true. parameter declared at line 855, so the else branch is not reached in
// NAM 4.2.11. It is ported in full because the unit is defined as this line range.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM, jds=1 jde=JM):
//   its_b1 = 2, ite_b2 = IM-2, jts_b1 = 2, jte_b2 = JM-2 (1-based)
//   -> zero-based i0 in [1, IM-3], j0 in [1, JM-3]; l0 = 0..LM-1 is the Fortran l-1.
//
// Layout (CONTRACT section 1): X(i,j,l) in IDX3(i0,j0,l0,IM,JM); 2-D [j,i] via IDX2.
// One thread per (i, j, l) point (grid_2d, POINT_IJL). Each thread reads only div and writes its own
// (i,j,l) elements of u and v, so there is no race. Fast numerics: fmad contraction is allowed.

#include "nmmb_common.cuh"

// module_DYNAMICS_ROUTINES.F90:75-76, single precision
#define DDAMP_CFC 1.533f
#define DDAMP_BFC (1.0f - DDAMP_CFC)

extern "C" __global__ void ddamp_apd(
    const int IM, const int JM,
    const float* __restrict__ pd,     // PD  [JM,IM]
    const float* __restrict__ pdo,    // PDO [JM,IM]
    float* __restrict__ apd)          // APD [JM,IM] out (recomputed every call)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < 1 || i0 > IM - 3 || j0 < 1 || j0 > JM - 3) return;

    const size_t c00 = IDX2(i0,     j0,     IM);
    const size_t c10 = IDX2(i0 + 1, j0,     IM);
    const size_t c01 = IDX2(i0,     j0 + 1, IM);
    const size_t c11 = IDX2(i0 + 1, j0 + 1, IM);

    const float s_pd  = pd[c00]  + pd[c10]  + pd[c01]  + pd[c11];
    const float s_pdo = pdo[c00] + pdo[c10] + pdo[c01] + pdo[c11];
    apd[c00] = (s_pd * DDAMP_CFC + s_pdo * DDAMP_BFC) * 0.25f;
}

extern "C" __global__ void ddamp_damp(
    const int IM, const int JM, const int LM,
    const float ddmpv,                // DDMPV scalar (dvd)
    const float pdtop,                // PDTOP scalar
    const float dfac,                 // preamble: dfac = 9.
    const float dpb,                  // preamble: dpb  = 1500.
    const float fint,                 // preamble: fint = (1.-dfac)/dpb
    const float assimfc,              // local parameter assimfc = 5.0
    const int freerun,                // logical freerun (configfile freerun: FREE -> true)
    const float* __restrict__ dsg2,   // DSG2  [LM]
    const float* __restrict__ pdsg1,  // PDSG1 [LM]
    const float* __restrict__ sg1,    // SG1   [LM+1]
    const float* __restrict__ sg2,    // SG2   [LM+1]
    const float* __restrict__ ddmpu,  // DDMPU [JM]  Fortran ddmpu(jds:jde), index j -> j0
    const float* __restrict__ apd,    // APD   [JM,IM] from ddamp_apd, this call
    const float* __restrict__ div,    // DIV   [LM,JM,IM] in
    float* __restrict__ u,            // U     [LM,JM,IM] inout
    float* __restrict__ v)            // V     [LM,JM,IM] inout
{
    POINT_IJL   // i0, j0, l0 (l0 = blockIdx.z)

    // Fortran its_b1..ite_b2 and jts_b1..jte_b2 (1-based 2..IM-2, 2..JM-2)
    if (i0 < 1 || i0 > IM - 3 || j0 < 1 || j0 > JM - 3) return;

    // fcim for this level (Fortran l = l0+1; sg1(l+1) -> sg1[l0+1])
    float fcim;
    if (freerun) {
        const float dpl = sg1[l0 + 1] * pdtop + sg2[l0 + 1] * 10000.0f;
        if (dpl < dpb) {
            fcim = fint * dpl + dfac;
        } else {
            fcim = 1.0f;
        }
    } else {
        fcim = assimfc;
    }

    const float dvd = ddmpv;
    const float dud = ddmpu[j0];
    const float rdpd = fcim / (dsg2[l0] * apd[IDX2(i0, j0, IM)] + pdsg1[l0]);

    // the four div neighbours used by both the u and the v stencil
    const float d_ij   = div[IDX3(i0,     j0,     l0, IM, JM)];
    const float d_i1j  = div[IDX3(i0 + 1, j0,     l0, IM, JM)];
    const float d_ij1  = div[IDX3(i0,     j0 + 1, l0, IM, JM)];
    const float d_i1j1 = div[IDX3(i0 + 1, j0 + 1, l0, IM, JM)];

    const size_t c3 = IDX3(i0, j0, l0, IM, JM);

    // u(i,j,l) = (div(i+1,j,l)+div(i+1,j+1,l)-div(i,j,l)-div(i,j+1,l))*dud*rdpd + u(i,j,l)
    const float su = ((d_i1j + d_i1j1) - d_ij) - d_ij1;
    u[c3] = su * dud * rdpd + u[c3];

    // v(i,j,l) = (div(i,j+1,l)+div(i+1,j+1,l)-div(i,j,l)-div(i+1,j,l))*dvd*rdpd + v(i,j,l)
    const float sv = ((d_ij1 + d_i1j1) - d_ij) - d_i1j;
    v[c3] = sv * dvd * rdpd + v[c3];
}
