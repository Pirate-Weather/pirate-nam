// phys_rrtm_getozn.cu -- getozn: climatological ozone mass mixing ratio profile for the NMMB RRTM
// radiation call (ntoz == 0 branch of grrad_nmmb).
//
// NOTICE: NAM source file phys/radiation_gases_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence
// file). Routine header kept as the notice (module_radiation_gases_nmmb, getozn):
//   getozn sets up climatological ozone profile for radiation calculation!
//   this code is originally written By Shrinivas Moorthi
//   modified to make output o3mmr has same vertical index order as given by control flag 'iflip'
//   by Apr. 03, y-t hou
// Ported to CUDA for WOOF-NAM from phys/radiation_gases_nmmb.f:984-1094 (nam.v4.2.11, 83df4e13c1).
//
// Line-by-line counterpart of the routine's statements in that range:
//   elte = blte + (JMR-1)*dlte                                    -> elte in this kernel
//   do i: deglat = xlat(i)*raddeg; three-way latitude branch       -> per-column, tem1/j1/j2
//   do j=1,LOZ: tem2 / tem3 / tem4 / o3i(i,j) (time interpolation k1oz/k2oz, facoz)
//   do l=1,LM: ll = l, or LM-l+1 when ivflip==1                    -> level loop in the thread
//      wk1(i)=prslk(i,ll); do k=1,LOZ-1: pressure-bracket interpolation of o3i -> o3mmr(i,ll)
//      do i: wk1 > pkstr(LOZ) -> o3i(i,LOZ); wk1 < pkstr(1) -> o3i(i,1)
// Elements the Fortran leaves untouched are left untouched here: when wk1 equals pkstr(1) exactly,
// or when no bracket matches, o3mmr(i,ll) keeps its previous value (the Fortran intent(out) array
// is not written there). For xlat NaN all latitude branches are false and the Fortran j1/j2/tem1
// are undefined; this kernel returns for that column and writes nothing.
//
// Constants: raddeg = 180/con_pi in single precision (radiation_gases_nmmb.f parameter, con_pi from
// physcons.f). blte, dlte, JMR (=latsozc), LOZ (=levozc) are module ozne_def values set in
// module_SOLVER_GRID_COMP.F90:10380-10382 (dphiozc: module_SOLVER_GRID_COMP.f90:9419) for the NMM/NAM configuration: levozc=17, latsozc=18,
// blatc=-85.0, dphiozc=10.0.
//
// Layout (CONTRACT.md section 1): prslk and o3mmr are [nlev, JM, IM] with index 0 the model top
// (ivflip == 0). o3r (Fortran o3r(JMR,LOZ,12)) is [12, LOZ, JMR] with the month index first.
// pkstr is [LOZ]. xlat is [JM, IM] in radians. The grrad caller passes LM = LMK (LM+1, LTP = 1), so
// the level count is an argument, nlev.
// One thread per column (i, j), the level loop runs inside the thread (grid_cols). Single precision
// throughout (kfpt = single). No host sync, no allocation, no printf.

#include "nmmb_common.cuh"

#define GETOZN_RADDEG   57.29578f      /* 180 / con_pi, single precision */
#define GETOZN_MAXLOZ   64             /* local o3i storage; the wrapper raises above 64 (no silent return) */

#define OZR_IDX(j0, k0, m0, JMR_, LOZ_) (((size_t)(m0) * (LOZ_) + (k0)) * (JMR_) + (j0))

extern "C" __global__ void phys_rrtm_getozn(
    const float* __restrict__ prslk,   // [nlev, JM, IM] exner function (grrad prslk1)
    const float* __restrict__ xlat,    // [JM, IM]       latitude, radians
    const float* __restrict__ o3r,     // [12, LOZ, JMR] climatology o3r(JMR,LOZ,12)
    const float* __restrict__ pkstr,   // [LOZ]          climatology pressure (pkap)
    float* __restrict__ o3mmr,         // [nlev, JM, IM] output ozone mass mixing ratio
    int im, int jm, int nlev,
    int jmr, int loz,
    int k1oz, int k2oz,                // 1-based month indices (module k1oz, k2oz)
    float facoz,                       // module facoz
    float blte, float dlte,            // module ozne_def blte, dlte
    int ivflip)
{
    COLUMN_IJ(im, jm)

    const float xl = xlat[IDX2(i0, j0, im)];
    const float deglat = xl * GETOZN_RADDEG;
    const float elte = blte + (float)(jmr - 1) * dlte;

    int j1, j2;
    float tem1;
    if (deglat > blte && deglat < elte) {
        tem1 = (deglat - blte) / dlte + 1.0f;
        j1 = (int)tem1;
        j2 = j1 + 1;
        tem1 = tem1 - (float)j1;
    } else if (deglat <= blte) {
        j1 = 1;
        j2 = 1;
        tem1 = 1.0f;
    } else if (deglat >= elte) {
        j1 = jmr;
        j2 = jmr;
        tem1 = 1.0f;
    } else {
        return;                        // NaN latitude: Fortran branch undefined, write nothing
    }

    // o3i(i, j) for j = 1..LOZ, time-interpolated between k1oz and k2oz (Fortran 1-based indices).
    float o3i[GETOZN_MAXLOZ];
    const float tem2 = 1.0f - tem1;
    for (int j = 0; j < loz; ++j) {
        const float tem3 = tem2 * o3r[OZR_IDX(j1 - 1, j, k1oz - 1, jmr, loz)]
                         + tem1 * o3r[OZR_IDX(j2 - 1, j, k1oz - 1, jmr, loz)];
        const float tem4 = tem2 * o3r[OZR_IDX(j1 - 1, j, k2oz - 1, jmr, loz)]
                         + tem1 * o3r[OZR_IDX(j2 - 1, j, k2oz - 1, jmr, loz)];
        o3i[j] = tem4 * facoz + tem3 * (1.0f - facoz);
    }

    // Level loop: ll is the Fortran level index (1-based); ivflip == 1 reverses the vertical order.
    for (int l = 1; l <= nlev; ++l) {
        const int ll = (ivflip == 1) ? (nlev - l + 1) : l;
        const size_t oidx = IDX3(i0, j0, ll - 1, im, jm);
        const float wk1 = prslk[oidx];

        // do k = 1, LOZ-1: pressure-bracket interpolation (bracket is (pkstr(k), pkstr(k+1)]).
        for (int k = 0; k < loz - 1; ++k) {
            const float temp = 1.0f / (pkstr[k + 1] - pkstr[k]);
            if (wk1 > pkstr[k] && wk1 <= pkstr[k + 1]) {
                const float tem = (pkstr[k + 1] - wk1) * temp;
                o3mmr[oidx] = tem * o3i[k] + (1.0f - tem) * o3i[k + 1];
            }
        }

        // Above the top or below the bottom of the climatology the end values are used.
        if (wk1 > pkstr[loz - 1]) o3mmr[oidx] = o3i[loz - 1];
        if (wk1 < pkstr[0])       o3mmr[oidx] = o3i[0];
    }
}
