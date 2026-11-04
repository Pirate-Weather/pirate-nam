// nam-gpu unit dyn-ddamp-extmode: DDAMP external-mode divergence damping (vertically integrated)
// and the internal-mode U/V divergence-damping update.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:801-999 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine ddamp, extmod = .true. parameter; lm = LM, dvd = ddmpv, freerun, fcxm):
//   [A] b1 range (its_b1..ite_b2, jts_b1..jte_b2):  apd(i,j) = ((pd sum)*cfc + (pdo sum)*bfc)*0.25
//   [B] dive range (its_b1..ite_b1_h2, jts_b1..jte_b1_h2):  dive(i,j) = sum over l of div(i,j,l)
//       (summed l = 1..lm in Fortran order: l=1 first, then dive = div(l) + dive)
//   [C] b1..b2 range:  rdpd = fcxm/(sg1(lm+1)*pdtop + sg2(lm+1)*apd);
//       rddu = (dive(i+1,j)+dive(i+1,j+1)-dive(i,j)-dive(i,j+1))*ddmpu(j)*rdpd
//       rddv = (dive(i,j+1)+dive(i+1,j+1)-dive(i,j)-dive(i+1,j))*ddmpv*rdpd
//   [D] every level l, b1..b2 range:  fcim (freerun: dpl = sg1(l+1)*pdtop + sg2(l+1)*1e4,
//       fcim = fint*dpl+dfac if dpl < dpb else 1; not freerun: fcim = assimfc = 5);
//       rdpd = fcim/(dsg2(l)*apd + pdsg1(l));
//       u(i,j,l) = ((div(i+1,j,l)+div(i+1,j+1,l)-div(i,j,l)-div(i,j+1,l))*ddmpu(j))*rdpd + rddu(i,j) + u
//       v(i,j,l) = ((div(i,j+1,l)+div(i+1,j+1,l)-div(i,j,l)-div(i+1,j,l))*ddmpv)*rdpd + rddv(i,j) + v
//   The `else` branch of the Fortran (extmod .false., internal-only damping) is NOT ported: `extmod`
//   is a compile-time PARAMETER .true. in the source, so that branch is unreachable in every build.
//   The Fortran `ENABLE_SMP` OpenMP split is dropped (one card, one thread per column / point).
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM,
// 1-based): its_b1=2, ite_b1=IM-1, its_b2=3, ite_b2=IM-2, ite_b1_h2=IM-1, jts_b1=2, jte_b1=JM-1,
// jte_b2=JM-2, jte_b1_h2=JM-1. Zero-based (i0 = i-1, j0 = j-1):
//   apd, rddu, rddv, u, v:   i0 in [1, IM-3],  j0 in [1, JM-3]
//   dive:                    i0 in [1, IM-2],  j0 in [1, JM-2]
// Scratch (not in the NAM state tables): APD, DIVE, RDDU, RDDV, all [JM,IM]; allocated by the wrapper.
//
// Launches (one stream, ordered): ddamp_apd_dive (one thread per column), ddamp_rdd (point, l=1),
// ddamp_uv (point, one thread per (i,j,l)). No host sync.

#include "nmmb_common.cuh"

// Kernel constants, Fortran single precision (kfpt = single).
#define DDAMP_CFC     1.533f
#define DDAMP_DFAC    9.0f
#define DDAMP_DPB     1500.0f
#define DDAMP_ASSIMFC 5.0f

// [A] + [B]: apd (mass-weighted surface pressure average) and dive (column-summed divergence).
extern "C" __global__ void ddamp_apd_dive(
    const int IM, const int JM, const int LM,
    const float* __restrict__ pd,       // PD   [JM,IM]
    const float* __restrict__ pdo,      // PDO  [JM,IM]
    const float* __restrict__ div,      // DIV  [LM,JM,IM]
    float* __restrict__ apd,            // scratch [JM,IM]  out
    float* __restrict__ dive)           // scratch [JM,IM]  out
{
    COLUMN_IJ(IM, JM)
    const float bfc = 1.0f - DDAMP_CFC;
    const size_t c2 = IDX2(i0, j0, IM);

    // [B] dive over i0 in [1, IM-2], j0 in [1, JM-2]
    if (i0 >= 1 && i0 <= IM - 2 && j0 >= 1 && j0 <= JM - 2) {
        float s = div[IDX3(i0, j0, 0, IM, JM)];
        for (int l0 = 1; l0 < LM; ++l0) {
            s = div[IDX3(i0, j0, l0, IM, JM)] + s;
        }
        dive[c2] = s;
    }

    // [A] apd over i0 in [1, IM-3], j0 in [1, JM-3]
    if (i0 >= 1 && i0 <= IM - 3 && j0 >= 1 && j0 <= JM - 3) {
        const float pdsum = pd[IDX2(i0, j0, IM)] + pd[IDX2(i0 + 1, j0, IM)]
                          + pd[IDX2(i0, j0 + 1, IM)] + pd[IDX2(i0 + 1, j0 + 1, IM)];
        const float pdosum = pdo[IDX2(i0, j0, IM)] + pdo[IDX2(i0 + 1, j0, IM)]
                           + pdo[IDX2(i0, j0 + 1, IM)] + pdo[IDX2(i0 + 1, j0 + 1, IM)];
        apd[c2] = (pdsum * DDAMP_CFC + pdosum * bfc) * 0.25f;
    }
}

// [C]: external-mode divergence factors rddu, rddv (2-D, l-independent).
extern "C" __global__ void ddamp_rdd(
    const int IM, const int JM, const int LM,
    const float fcxm,
    const float pdtop,
    const float dvd,                    // ddmpv scalar
    const float* __restrict__ sg1,      // SG1 [LM+1]
    const float* __restrict__ sg2,      // SG2 [LM+1]
    const float* __restrict__ ddmpu,    // DDMPU [JM]  (Fortran jds:jde, zero-based j0)
    const float* __restrict__ apd,      // scratch [JM,IM]
    const float* __restrict__ dive,     // scratch [JM,IM]
    float* __restrict__ rddu,           // scratch [JM,IM]  out
    float* __restrict__ rddv)           // scratch [JM,IM]  out
{
    POINT_IJL
    if (i0 < 1 || i0 > IM - 3 || j0 < 1 || j0 > JM - 3) return;
    const size_t c2 = IDX2(i0, j0, IM);

    const float rdpd = fcxm / (sg1[LM] * pdtop + sg2[LM] * apd[c2]);
    const float dud = ddmpu[j0];

    rddu[c2] = (dive[IDX2(i0 + 1, j0, IM)] + dive[IDX2(i0 + 1, j0 + 1, IM)]
              - dive[IDX2(i0, j0, IM)] - dive[IDX2(i0, j0 + 1, IM)]) * dud * rdpd;
    rddv[c2] = (dive[IDX2(i0, j0 + 1, IM)] + dive[IDX2(i0 + 1, j0 + 1, IM)]
              - dive[IDX2(i0, j0, IM)] - dive[IDX2(i0 + 1, j0, IM)]) * dvd * rdpd;
}

// [D]: internal-mode damping of U and V on every level, plus the external-mode add-back.
// grid_2d over (IM, JM, LM): blockIdx.z is the level (zero-based l0 = Fortran l - 1).
extern "C" __global__ void ddamp_uv(
    const int IM, const int JM, const int LM,
    const int freerun,                  // 1 = FREERUN (fcim from the top-to-pressure ramp)
    const float pdtop,
    const float dvd,                    // ddmpv scalar
    const float* __restrict__ sg1,      // SG1  [LM+1]
    const float* __restrict__ sg2,      // SG2  [LM+1]
    const float* __restrict__ dsg2,     // DSG2 [LM]
    const float* __restrict__ pdsg1,    // PDSG1 [LM]
    const float* __restrict__ ddmpu,    // DDMPU [JM]
    const float* __restrict__ apd,      // scratch [JM,IM]
    const float* __restrict__ div,      // DIV [LM,JM,IM]
    const float* __restrict__ rddu,     // scratch [JM,IM]
    const float* __restrict__ rddv,     // scratch [JM,IM]
    float* __restrict__ u,              // U [LM,JM,IM]  inout
    float* __restrict__ v)              // V [LM,JM,IM]  inout
{
    POINT_IJL
    if (i0 < 1 || i0 > IM - 3 || j0 < 1 || j0 > JM - 3 || l0 >= LM) return;

    const float dpb = DDAMP_DPB;
    const float dfac = DDAMP_DFAC;
    const float fint = (1.0f - dfac) / dpb;

    // fcim for this level (Fortran l = l0 + 1)
    float fcim;
    if (freerun) {
        const float dpl = sg1[l0 + 1] * pdtop + sg2[l0 + 1] * 10000.0f;
        if (dpl < dpb) {
            fcim = fint * dpl + dfac;
        } else {
            fcim = 1.0f;
        }
    } else {
        fcim = DDAMP_ASSIMFC;
    }

    const size_t c2 = IDX2(i0, j0, IM);
    const size_t p = IDX3(i0, j0, l0, IM, JM);
    const float dud = ddmpu[j0];
    const float rdpd = fcim / (dsg2[l0] * apd[c2] + pdsg1[l0]);

    u[p] = ((div[IDX3(i0 + 1, j0,     l0, IM, JM)] + div[IDX3(i0 + 1, j0 + 1, l0, IM, JM)]
           - div[IDX3(i0,     j0,     l0, IM, JM)] - div[IDX3(i0,     j0 + 1, l0, IM, JM)]) * dud) * rdpd
         + rddu[c2] + u[p];
    v[p] = ((div[IDX3(i0,     j0 + 1, l0, IM, JM)] + div[IDX3(i0 + 1, j0 + 1, l0, IM, JM)]
           - div[IDX3(i0,     j0,     l0, IM, JM)] - div[IDX3(i0 + 1, j0,     l0, IM, JM)]) * dvd) * rdpd
         + rddv[c2] + v[p];
}
