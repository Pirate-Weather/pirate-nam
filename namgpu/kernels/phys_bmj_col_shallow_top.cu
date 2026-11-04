// WOOF-NAM unit phys-bmj-col-shallow-top: shallow cloud top search (virtual-temperature excess) and
// gather of the shallow convection points, for the BMJ convection scheme.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module and routine header kept as the notice:
//   module module_cu_bmj  (NMMB, nam.v4.2.11)
//   subroutine bmj(...)  -- programer: z. janjic, yugoslav fed. hydromet. inst., beograd
// Ported to CUDA for WOOF-NAM from phys/module_CU_BMJ.F90:1317-1449 (nam.v4.2.11, 83df4e13c1), with the
// per-column state that the Fortran sets before the deep test (555, 953-955, 1225-1226) written here too.
//
// Per-column paths (Fortran control flow of subroutine bmj):
//   bypass = 1   depth < depmin (953-955): shallow=.true., deep=.false., goto 600. The cloud-top search
//                is NOT run for these columns (it would read stale pk/dtv). Writes shallow=1, deep=0.
//   deep_ok = 1  deep convection (1223 test true, 1225-1226): shallow=0, deep=1, iswap=0, cldefi=efi.
//   otherwise    failed-deep branch, Fortran 1316-1405:
//     1317-1324  cldefi = efimn*sm + stefi*(1-sm) (1316), ptpk = max(pshu, pk(lbot) - depmin)
//     1328-1332  if(.not.newswap): ltop = (last l with pk(l) <= ptpk) + 1, else ltop unchanged
//     1384-1394  if(.not.newswap): walk down from lbot-1 to ltop while dtv(l) > 0; ltop = min(ltp1, lbot)
//     1403-1405  iswap = 1, ptop = pk(ltop), shallow = .true., deep = .false.
//     1436-1440  print_diag write: debug print, dropped (no print inside kernels)
//     1444       if(.not.shallow) return: the gather. Here it is the per-column mask `shallow` that the
//                shallow-adjust unit (phys-bmj-col-shallow-adjust, 1450-2067) reads; no compaction.
// depmin (Fortran 551) = psh*psfc*rsfcp is recomputed here with the same operation order (psh*psfc)*rsfcp.
//
// Inputs that other units produce (passed in, not recomputed):
//   pk(l)   = prsmid(l) copy (Fortran 996, pk(l)=pkl, pkl=prsmid(l)); dtv(l) = DTVCNV after the restore
//             (Fortran 919-921); lbot = LBOTCNV after the cloud-base unit (Fortran 914). These are caller
//             contracts. The producers (deep-ref, cloudtop, deep-tend) are separate units; this kernel
//             does not check them.
//   deep_ok = the result of the test at 1223 (dentpy>=epsntp .and. preck>epspr .and. .not.nodeep), 0/1.
//             It is only meaningful where bypass = 0 (the 1223 test is reached only when depth >= depmin).
//   bypass  = 1 where depth < depmin (Fortran 953 shallow-only columns), 0/1. Supplied by the caller.
//   efi     = Fortran efi (1210-1213, clamped to [efimn, 1]) per column; used only where deep_ok = 1.
//   ltop    = cloud top from phys-bmj-col-cloudtop. Fortran cloudtop (885) clamps it to >= 1; this kernel
//             clamps the input to >= 1 too, so pk(ltop) (Fortran 1405) never reads pk(0).
//   sm, psfc (2-D, the surface pressure of the column), efimn, stefi (scalars, configfile/bmj_init).
// Scalars psh = 20000., pshu = 45000., rsfcp = 1./101300. are module defaults (module_CU_BMJ.F90:39,
// 103); the caller passes them so the single-source constants stay in nmmb_common.cuh.
//
// Layout (CONTRACT section 1): 3-D [LM, JM, IM] (l = 0 is Fortran l = 1, model top), 2-D and int
// fields [JM, IM]. Fortran 1-based l maps to index l-1; Fortran pk(lbot) is pk[lbot-1].
// One thread per column with an l-loop (CONTRACT section 3, kernels.grid_cols). Fast numerics
// (fmad on), float everywhere (kfpt = single). No host sync, no device malloc, no printf.

#include "nmmb_common.cuh"

// Device body: one column. Returns nothing; updates the column state in place.
__device__ inline void bmj_col_shallow_top(
    int i0, int j0, int IM, int JM, int LM, int lmh, int newswap,
    float psh, float pshu, float rsfcp, float efimn, float stefi,
    const float* __restrict__ psfc, const float* __restrict__ sm,
    const int* __restrict__ deep_ok, const int* __restrict__ bypass,
    const int* __restrict__ lbot_in, int* __restrict__ ltop_io,
    const float* __restrict__ pk, const float* __restrict__ dtv, const float* __restrict__ efi,
    float* __restrict__ cldefi, int* __restrict__ deep, int* __restrict__ shallow,
    int* __restrict__ iswap, float* __restrict__ ptop)
{
    const size_t c2 = IDX2(i0, j0, IM);

    // ---- 953-955: shallow-only column, goto 600. Nothing else in the failed-deep branch runs. ----
    if (bypass[c2]) {
        shallow[c2] = 1;
        deep[c2] = 0;
        return;
    }

    // ---- 1223-1226: deep convection. 555 shallow=.false. and 953 deep=.true. (depth >= depmin). ----
    if (deep_ok[c2]) {
        shallow[c2] = 0;
        deep[c2] = 1;
        iswap[c2] = 0;                      // 1225
        cldefi[c2] = efi[c2];               // 1226
        return;
    }

    // ---- 1316: failed-deep branch ----
    deep[c2] = 0;                           // 555 deep=.false.; 1404 deep=.false. below
    const int lbot = lbot_in[c2];           // Fortran lbot (1-based)
    const float psfc_c = psfc[c2];
    const float sm_c = sm[c2];
    cldefi[c2] = efimn * sm_c + stefi * (1.0f - sm_c);

    // depmin = psh*psfc*rsfcp (Fortran 551, same operation order)
    const float depmin = psh * psfc_c * rsfcp;

    // ---- 1324: ptpk = max(pshu, pk(lbot) - depmin) ----
    // Fortran pk(lbot) is pk[lbot-1]. lbot < 1 would read pk(0), out of bounds in the Fortran. Refused here.
    // This is an addition: the Fortran has no counterpart; the cloud-base unit gives lbot >= 1.
    if (lbot < 1) {
        shallow[c2] = 0;
        return;
    }
    const float ptpk = fmaxf(pshu, pk[IDX3(i0, j0, lbot - 1, IM, JM)] - depmin);

    // Fortran cloudtop (885) gives ltop = max(min(ltp1, lbot), 1). The clamp here is the same guarantee
    // on the input, so the ptop read below (pk(ltop), Fortran 1405) is always pk[ltop-1] with ltop >= 1.
    int ltop = ltop_io[c2];
    if (ltop < 1) ltop = 1;

    // ---- 1328-1332: cloud top = level just below ptpk (old swap, skipped when newswap) ----
    if (!newswap) {
        for (int l = 1; l <= lmh; ++l) {    // Fortran do l=1,lmh (lmh = lm, module_CU_BMJ.F90:303)
            if (pk[IDX3(i0, j0, l - 1, IM, JM)] <= ptpk) ltop = l + 1;
        }
    }

    // ---- 1384-1394: buoyancy check on dtv, from lbot-1 down to ltop (old swap only) ----
    if (!newswap) {
        int ltp1 = lbot;
        for (int l = lbot - 1; l >= ltop; --l) {   // Fortran do l=lbot-1,ltop,-1
            if (dtv[IDX3(i0, j0, l - 1, IM, JM)] > 0.0f) {
                ltp1 = l;
            } else {
                break;                      // Fortran exit
            }
        }
        ltop = min(ltp1, lbot);
    }
    ltop_io[c2] = ltop;

    // ---- 1403-1405: failed deep convection -> shallow swap point ----
    iswap[c2] = 1;
    ptop[c2] = pk[IDX3(i0, j0, ltop - 1, IM, JM)];
    shallow[c2] = 1;
    deep[c2] = 0;
    // 1436-1440 print_diag: debug write, dropped.
    // 1444 if(.not.shallow) return: shallow is 1 here, so the column passes the gather.
}

// Batch entry (one thread per column, grid_cols). Arguments are device pointers or scalars.
extern "C" __global__ void phys_bmj_col_shallow_top_columns(
    const int IM, const int JM, const int LM, const int lmh, const int newswap,
    const float psh, const float pshu, const float rsfcp,
    const float efimn, const float stefi,
    const float* __restrict__ psfc,         // [JM, IM]
    const float* __restrict__ sm,           // [JM, IM]
    const int* __restrict__ deep_ok,        // [JM, IM] 0/1, result of the test at 1223 (only where bypass = 0)
    const int* __restrict__ bypass,         // [JM, IM] 0/1, depth < depmin (goto 600 columns)
    const int* __restrict__ lbot,           // [JM, IM] cloud base (1-based)
    int* __restrict__ ltop,                 // [JM, IM] in/out cloud top (1-based)
    const float* __restrict__ pk,           // [LM, JM, IM] prsmid copy
    const float* __restrict__ dtv,          // [LM, JM, IM] virtual-temperature excess
    const float* __restrict__ efi,          // [JM, IM] Fortran efi (read on deep columns only)
    float* __restrict__ cldefi,             // [JM, IM] in/out
    int* __restrict__ deep,                 // [JM, IM] in/out 0/1
    int* __restrict__ shallow,              // [JM, IM] in/out 0/1 (gather mask for phys-bmj-col-shallow-adjust)
    int* __restrict__ iswap,                // [JM, IM] out (0 deep, 1 failed-deep)
    float* __restrict__ ptop)               // [JM, IM] out
{
    COLUMN_IJ(IM, JM)
    bmj_col_shallow_top(i0, j0, IM, JM, LM, lmh, newswap, psh, pshu, rsfcp, efimn, stefi,
                        psfc, sm, deep_ok, bypass, lbot, ltop, pk, dtv, efi, cldefi, deep, shallow,
                        iswap, ptop);
}
