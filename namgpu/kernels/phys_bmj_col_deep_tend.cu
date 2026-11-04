// WOOF-NAM unit phys-bmj-col-deep-tend: deep convection heating, moistening, momentum (mmntdeep
// branch, off by default in the NAM configuration) and precipitation of the BMJ scheme, for one
// column. Fortran range phys/module_CU_BMJ.F90:1180-1316 (the heating/moistening loop inside the
// cloud-efficiency iteration, the deep-convection block, and the start of the shallow-cloud else
// branch that sets cldefi).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module and routine header kept as the notice:
//   module module_cu_bmj  (NMMB, nam.v4.2.11)
//   subroutine bmj  (deep convection; BMJ scheme)
//   programer: z. janjic, yugoslav fed. hydromet. inst., beograd
// Ported to CUDA for WOOF-NAM from phys/module_CU_BMJ.F90:1180-1316 (nam.v4.2.11, 83df4e13c1).
//
// Pass structure (review of phys-bmj-col-deep-tend). In Fortran the cloud-efficiency body
// (1191-1213) sits inside `cloud_efficiency : do itrefi=1,itrefi_max` (1088-1216), and the deep block
// (1222-1290) and the else branch (1295-1315) run ONCE, after the loop, with the efi of the last
// pass. last_pass = 0 runs the efficiency part only (writes efi, dift, difq, dentpy, preck), so the
// next itrefi pass reads efi through the caller. last_pass = 1 also runs the deep or else branch.
// The itrefi loop itself is driven by the wrapper (namgpu.physics.bmj.phys_bmj_col_deep_tend).
// Fortran iswap is set to 0 at 574 (once per bmj call, in the wrapper) and to 0 again at 1225 (deep
// branch only). The else branch never assigns it, so this kernel does not write it there.
//
// Two entry points:
//   bmj_col_deep_tend                 __device__ function, the Fortran body for one column. The
//                                     BMJ column driver (phys_bmj_bmjdrv) calls it per thread.
//   phys_bmj_col_deep_tend_columns    extern "C" __global__, one thread per column, the launchable
//                                     form used by the wrapper namgpu.physics.bmj.phys_bmj_col_deep_tend.
//
// Column data layout: a per-level Fortran array A(l) (1-based) of one column is read/written as
// base[(l-1)*stride]. The launchable kernel uses base = field + column and stride = IM*JM, which is
// the CONTRACT layout A[l, j, i] with column index c = j*IM + i. The column driver may pass a
// thread-local array with stride 1. Per-column scalars are passed as pointers to the column element.
//
// Fortran dummy/locals -> arguments (names as in the Fortran):
//   tk, qk            column T, Q (tk, qk locals of BMJ, set from T, Q in the prep unit)
//   dprs, el          layer pressure thickness, EL (parcel efficiency weight)
//   trefk, qrefk      reference T and Q profiles from the deep-reference / iteration units
//   pk, pkt, rdp0t    Exner-like pressure profile, PKT = pk(ltop), RDP0T = 1/(pk0-pkt)
//   u, v              column winds (for the momentum branch)
//   dift, difq        out: T and Q differences trefk-tk, qrefk-qk scaled by tauk (kept for later units)
//   dtdt, dqdt        inout: heating and moistening tendencies on the deep branch (else unchanged)
//   dudt, dvdt        inout: momentum tendencies on the mmntdeep branch (else unchanged)
//   ltop, lb, l0      cloud top, cloud base, deep-convection base level indices (1-based)
//   sm                land/sea factor of the column
//   sumdp             inout: layer-pressure sum (read by AVRGT, rewritten on the mmntdeep branch)
//   efi, cldefi       out: cloud efficiency; cloud efficiency for the shallow-cldefi else branch
//   dentpy, preck     out: enthalpy and precipitation increments of this iteration
//   fefi, pcpcol      out: deep efficiency factor and convective precipitation of the column
//   iswap             out: swap flag, 0 on the deep branch (Fortran IWAP=0 "deep convection, no swap")
//   mmntdeep, nodeep  logical switches (0/1)
//   cp, cprlg, tauk, rdtcnvc, uvscald, efifc, efimn, epsntp, epspr, efmntl, efmnts, slopes, slopel,
//   stefi             scalars: CP, CPRLG = cp/(row*g*elwv), TAUK = dtcnvc/(trel*1), RDTCNVC = 1/dtcnvc,
//                     UVSCALD = 0.01, and the BMJ module parameters (EFIFC, EFIMN, EPSNTP, EPSPR,
//                     EFMNTL, EFMNTS, SLOPES, SLOPEL, STEFI).
//
// Index handling: all loops keep the Fortran 1-based bounds (ltop..lb, l0..lb, ltop..l0-1); the
// comparisons `l0 > ltop` and `l0 < lb` are kept as written.
//
// Numerics: float32 (NMMB kfpt = single), FMA contraction on (fast numerics, CONTRACT section 3).
// Evaluation order follows the Fortran expressions left to right. No host sync, no printf, no
// device allocation.

#include "nmmb_common.cuh"

#define COL_AT(p, l) ((p)[((l) - 1) * stride])

__device__ void bmj_col_deep_tend(
    const float* __restrict__ tk, const float* __restrict__ qk,
    const float* __restrict__ dprs, const float* __restrict__ el,
    const float* __restrict__ trefk, const float* __restrict__ qrefk,
    const float* __restrict__ pk, const float* __restrict__ u, const float* __restrict__ v,
    float* __restrict__ dift, float* __restrict__ difq,
    float* __restrict__ dtdt, float* __restrict__ dqdt,
    float* __restrict__ dudt, float* __restrict__ dvdt,
    int stride, int ltop, int lb, int l0, int mmntdeep, int nodeep,
    float sm, float* __restrict__ sumdp, float pkt, float rdp0t,
    float cp, float cprlg, float tauk, float rdtcnvc, float uvscald,
    float efifc, float efimn, float epsntp, float epspr,
    float efmntl, float efmnts, float slopes, float slopel, float stefi,
    float* __restrict__ efi_out, float* __restrict__ cldefi_out,
    float* __restrict__ dentpy_out, float* __restrict__ preck_out,
    float* __restrict__ fefi_out, float* __restrict__ pcpcol_out,
    int* __restrict__ iswap_out, int last_pass)
{
    // sumdp: the enthalpy-loop sum of dprs over ltop..lb (Fortran 1139-1144, the last iter=2 value,
    // which does not depend on trefk). Computed here so the caller does not supply it.
    float sdp = 0.0f;
    for (int l = ltop; l <= lb; ++l) {
        sdp = COL_AT(dprs, l) + sdp;
    }
    *sumdp = sdp;

    // ---- heating, moistening, precipitation (Fortran 1180-1216, inside cloud_efficiency) ----
    float avrgt = 0.0f;
    float preck = 0.0f;
    float dsq = 0.0f;
    float dst = 0.0f;

    for (int l = ltop; l <= lb; ++l) {
        float tkl = COL_AT(tk, l);
        float diftl = (COL_AT(trefk, l) - tkl) * tauk;
        float difql = (COL_AT(qrefk, l) - COL_AT(qk, l)) * tauk;
        float avrgtl = (tkl + tkl + diftl);
        float dpot = COL_AT(dprs, l) / avrgtl;
        dst = diftl * dpot + dst;
        dsq = difql * COL_AT(el, l) * dpot + dsq;
        avrgt = avrgtl * COL_AT(dprs, l) + avrgt;
        preck = diftl * COL_AT(dprs, l) + preck;
        COL_AT(dift, l) = diftl;
        COL_AT(difq, l) = difql;
    }

    dst = (dst + dst) * cp;
    dsq = dsq + dsq;
    float dentpy = dst + dsq;
    avrgt = avrgt / (*sumdp + *sumdp);

    float drheat = (preck * sm + fmaxf(1.0e-7f, preck) * (1.0f - sm)) * cp / avrgt;
    drheat = fmaxf(drheat, 1.0e-20f);
    float efi = efifc * dentpy / drheat;
    efi = fminf(efi, 1.0f);
    efi = fmaxf(efi, efimn);
    // ---- end of the cloud-efficiency loop body ----

    *efi_out = efi;
    *dentpy_out = dentpy;
    *preck_out = preck;

    // Passes before the last of itrefi_max only feed efi forward (see the pass structure above).
    if (!last_pass) return;

    if (dentpy >= epsntp && preck > epspr && !nodeep) {
        // ---- deep convection (Fortran 1222-1290) ----
        int iswap = 0;   // deep convection, no swap
        float cldefi = efi;
        *cldefi_out = cldefi;

        float fefi;
        if (sm > 0.5f) {
            fefi = (cldefi - efimn) * slopes + efmnts;
        } else {
            fefi = (cldefi - efimn) * slopel + efmntl;
        }

        fefi = (dentpy - epsntp) * fefi / dentpy;
        preck = preck * fefi;

        // update precipitation and tendencies of temperature and moisture
        float cup = preck * cprlg;
        *pcpcol_out = cup;

        for (int l = ltop; l <= lb; ++l) {
            COL_AT(dtdt, l) = COL_AT(dift, l) * fefi * rdtcnvc;
            COL_AT(dqdt, l) = COL_AT(difq, l) * fefi * rdtcnvc;
        }

        if (mmntdeep) {
            float facuv = fefi * rdtcnvc * uvscald;
            if (l0 > ltop && l0 < lb) {
                float ubar = 0.0f;
                float vbar = 0.0f;
                *sumdp = 0.0f;

                for (int l = l0; l <= lb; ++l) {
                    ubar = COL_AT(u, l) * COL_AT(dprs, l) + ubar;
                    vbar = COL_AT(v, l) * COL_AT(dprs, l) + vbar;
                    *sumdp = COL_AT(dprs, l) + *sumdp;
                }

                float rdpsum = 1.0f / *sumdp;
                ubar = ubar * rdpsum;
                vbar = vbar * rdpsum;

                for (int l = l0; l <= lb; ++l) {
                    COL_AT(dudt, l) = (ubar - COL_AT(u, l)) * facuv;
                    COL_AT(dvdt, l) = (vbar - COL_AT(v, l)) * facuv;
                }

                float dum = ubar - COL_AT(u, l0);
                float dvm = vbar - COL_AT(v, l0);

                for (int l = ltop; l <= l0 - 1; ++l) {
                    COL_AT(dudt, l) = (COL_AT(pk, l) - pkt) * dum * rdp0t * facuv;
                    COL_AT(dvdt, l) = (COL_AT(pk, l) - pkt) * dvm * rdp0t * facuv;
                }
            } else {
                float ubar = 0.0f;
                float vbar = 0.0f;
                *sumdp = 0.0f;

                for (int l = ltop; l <= lb; ++l) {
                    ubar = COL_AT(u, l) * COL_AT(dprs, l) + ubar;
                    vbar = COL_AT(v, l) * COL_AT(dprs, l) + vbar;
                    *sumdp = COL_AT(dprs, l) + *sumdp;
                }

                float rdpsum = 1.0f / *sumdp;
                ubar = ubar * rdpsum;
                vbar = vbar * rdpsum;

                for (int l = ltop; l <= lb; ++l) {
                    COL_AT(dudt, l) = (ubar - COL_AT(u, l)) * facuv;
                    COL_AT(dvdt, l) = (vbar - COL_AT(v, l)) * facuv;
                }
            }
        }

        *fefi_out = fefi;
        *preck_out = preck;
        *iswap_out = iswap;
    } else {
        // ---- no deep convection: the cloud-top reduction is in the shallow units; this branch
        // sets the cloud efficiency used there (Fortran 1295-1315). The commented iteration
        // blocks of the Fortran (ltop+3, shallow cloud-top search) are not active in the NAM
        // configuration and are not part of this unit.
        *cldefi_out = efimn * sm + stefi * (1.0f - sm);
    }
}

#undef COL_AT

extern "C" __global__ void phys_bmj_col_deep_tend_columns(
    int IM, int JM, int LM,
    const float* __restrict__ tk, const float* __restrict__ qk,
    const float* __restrict__ dprs, const float* __restrict__ el,
    const float* __restrict__ trefk, const float* __restrict__ qrefk,
    const float* __restrict__ pk, const float* __restrict__ u, const float* __restrict__ v,
    float* __restrict__ dift, float* __restrict__ difq,
    float* __restrict__ dtdt, float* __restrict__ dqdt,
    float* __restrict__ dudt, float* __restrict__ dvdt,
    const int* __restrict__ ltop, const int* __restrict__ lb, const int* __restrict__ l0,
    float* __restrict__ sumdp, const float* __restrict__ sm,
    const float* __restrict__ pkt, const float* __restrict__ rdp0t,
    float* __restrict__ efi, float* __restrict__ cldefi,
    float* __restrict__ dentpy, float* __restrict__ preck,
    float* __restrict__ fefi, float* __restrict__ pcpcol, int* __restrict__ iswap,
    int mmntdeep, int nodeep,
    float cp, float cprlg, float tauk, float rdtcnvc, float uvscald,
    float efifc, float efimn, float epsntp, float epspr,
    float efmntl, float efmnts, float slopes, float slopel, float stefi,
    const int* __restrict__ act, int last_pass)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int ncol = IM * JM;
    if (c >= ncol) return;
    (void)LM;
    // Fortran loops run over the _b1 limits (module_DM_PARALLEL looplimits: its_b1 = max(its, ids+1),
    // ite_b1 = min(ite, ide-1)) and only on convection steps. The wrapper builds act[c] from both.
    if (!act[c]) return;

    bmj_col_deep_tend(
        tk + c, qk + c, dprs + c, el + c, trefk + c, qrefk + c, pk + c, u + c, v + c,
        dift + c, difq + c, dtdt + c, dqdt + c, dudt + c, dvdt + c,
        ncol, ltop[c], lb[c], l0[c], mmntdeep, nodeep,
        sm[c], sumdp + c, pkt[c], rdp0t[c],
        cp, cprlg, tauk, rdtcnvc, uvscald,
        efifc, efimn, epsntp, epspr, efmntl, efmnts, slopes, slopel, stefi,
        efi + c, cldefi + c, dentpy + c, preck + c, fefi + c, pcpcol + c, iswap + c, last_pass);
}
