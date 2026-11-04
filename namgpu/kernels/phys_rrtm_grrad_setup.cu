// phys_rrtm_grrad_setup.cu -- nam-gpu unit phys-rrtm-grrad-setup: the argument, layer-array and
// unit-conversion setup of NMMB grrad_nmmb (one thread per column, every level in a loop).
//
// NOTICE: NAM source file phys/grrad_nmmb.f (NOAA/NCEP/EMC, nam.v4.2.11, 83df4e13c1), subroutine
// grrad_nmmb, module module_radiation_driver_nmmb: "this program is the driver of radiation
// calculation subroutines. It sets up profile variables for radiation input, including clouds,
// surface albedos, atmospheric aerosols, ozone, etc." NOAA/NCEP US Government work, no licence file
// in the NAM clone; the NOAA author headers stay in the Fortran source and are not reproduced here.
// Functions from phys/funcphys.f (fpvs, fpvsx, fpkapx) and constants from phys/physcons.f are
// reproduced below with their line references.
//
// Ported to CUDA for nam-gpu from phys/grrad_nmmb.f:660-1396 (nam.v4.2.11, 83df4e13c1).
// Scope: from the subroutine header (660) through the local-array setup, ending before the
// climatological ozone call (getozn, line 1397 onward, unit phys-rrtm-grrad-gases). olyr is written
// here only for ntoz > 0 (Fortran lines 1386-1392); the ntoz == 0 branch belongs to that unit.
//
// Layer convention: LTP = 1 extra top layer (lextop = .true.). Local 0-based index l = k0 + kd for
// input level k0 (0-based, top first), kd = 1 for ivflip == 0 (toa to sfc, the NAM setting) and
// kd = 0 for ivflip == 1. Local arrays have LMK = LM + 1 layers and LMP = LM + 2 levels.
//
// Units: the Fortran radiation takes prsi and prsl in cb (kPa); the NAM caller passes Pa (CONTRACT
// state table), so the kernel converts Pa -> cb with 1e-3. Local plvl and plyr are mb (cb * 10,
// the Fortran "10.0 * prsi" lines). The Fortran es = min(prsl, 0.001 * fpvs) is cb, with fpvs in Pa.
#include "nmmb_common.cuh"

// physcons.f constants (single precision)
#define RAD_CON_RD      2.8705e+2f
#define RAD_CON_RV      4.6150e+2f
#define RAD_CON_CP      1.0046e+3f
#define RAD_CON_ROCP    (RAD_CON_RD / RAD_CON_CP)            // con_rocp = con_rd/con_cp
#define RAD_CON_EPS     (RAD_CON_RD / RAD_CON_RV)            // con_eps
#define RAD_CON_EPSM1   (RAD_CON_RD / RAD_CON_RV - 1.0f)     // con_epsm1
#define RAD_CON_TTP     2.7316e+2f
#define RAD_CON_PSAT    6.1078e+2f
#define RAD_CON_CVAP    1.8460e+3f
#define RAD_CON_CLIQ    4.1855e+3f
#define RAD_CON_CSOL    2.1060e+3f
#define RAD_CON_HVAP    2.5000e+6f
#define RAD_CON_HFUS    3.3358e+5f
#define RAD_CON_FVIRT   (RAD_CON_RV / RAD_CON_RD - 1.0f)    // con_fvirt = con_rv/con_rd-1 (physcons.f:79)

// grrad_nmmb parameters (grrad_nmmb.f:197 QMIN, :199 prsmin, :210 LTP)
#define RAD_QMIN        1.0e-10f
#define RAD_QME5        1.0e-7f     // grrad_nmmb.f:197 QME5 (active parameter line)
#define RAD_QME6        1.0e-7f     // grrad_nmmb.f:197 QME6 (active parameter line)
#define RAD_PRSMIN      1.0e-6f
#define RAD_LTP         1

// funcphys.f gpvs table: NXPVS = 7501 nodes on [180, 330] K, xinc = 150/7500 = 0.02 K
#define FPVS_NX     7501
#define FPVS_XMIN   180.0f
#define FPVS_XINC   0.02f

// funcphys.f fpvsx(t) (lines 794-829): exact saturation vapour pressure in Pa; liquid at or above
// tliq = ttp, ice below tice = ttp - 20, linear blend in between.
__device__ inline float rad_fpvsx(float t)
{
    const float tliq = RAD_CON_TTP;
    const float tice = RAD_CON_TTP - 20.0f;
    const float dldtl = RAD_CON_CVAP - RAD_CON_CLIQ;
    const float heatl = RAD_CON_HVAP;
    const float xponal = -dldtl / RAD_CON_RV;
    const float xponbl = -dldtl / RAD_CON_RV + heatl / (RAD_CON_RV * RAD_CON_TTP);
    const float dldti = RAD_CON_CVAP - RAD_CON_CSOL;
    const float heati = RAD_CON_HVAP + RAD_CON_HFUS;
    const float xponai = -dldti / RAD_CON_RV;
    const float xponbi = -dldti / RAD_CON_RV + heati / (RAD_CON_RV * RAD_CON_TTP);
    const float tr = RAD_CON_TTP / t;
    if (t >= tliq) {
        return RAD_CON_PSAT * powf(tr, xponal) * expf(xponbl * (1.0f - tr));
    } else if (t < tice) {
        return RAD_CON_PSAT * powf(tr, xponai) * expf(xponbi * (1.0f - tr));
    } else {
        const float w = (t - tice) / (tliq - tice);
        const float pvl = RAD_CON_PSAT * powf(tr, xponal) * expf(xponbl * (1.0f - tr));
        const float pvi = RAD_CON_PSAT * powf(tr, xponai) * expf(xponbi * (1.0f - tr));
        return w * pvl + (1.0f - w) * pvi;
    }
}

// funcphys.f fpvs(t) (lines 702-714): linear interpolation in the table tbpvs(j) = fpvsx(xmin +
// (j-1)*xinc), with c2xpvs = 1/xinc and c1xpvs = 1 - xmin*c2xpvs (gpvs, lines 657-669). The table
// nodes are recomputed from fpvsx instead of being stored, which gives the same node values; the
// interpolation is the Fortran expression. Output in Pa.
__device__ inline float rad_fpvs(float t)
{
    const float c2 = 1.0f / FPVS_XINC;
    const float c1 = 1.0f - FPVS_XMIN * c2;
    const float xj = fminf(fmaxf(c1 + c2 * t, 1.0f), (float)FPVS_NX);
    const int jx = (int)fminf(xj, (float)(FPVS_NX - 1));          // Fortran JX = MIN(XJ, NXPVS-1)
    const float f0 = rad_fpvsx(FPVS_XMIN + (float)(jx - 1) * FPVS_XINC);   // tbpvs(jx)
    const float f1 = rad_fpvsx(FPVS_XMIN + (float)jx * FPVS_XINC);         // tbpvs(jx+1)
    return f0 + (xj - (float)jx) * (f1 - f0);
}

// funcphys.f fpkapx(p) (lines 2377-2385): (p/1e5)^kappa, p in Pa.
__device__ inline float rad_fpkapx(float p)
{
    return powf(p / 1.0e5f, RAD_CON_ROCP);
}

extern "C" __global__ void rrtm_grrad_setup(
    const float* __restrict__ prsi,      // PINT      [LM+1, JM, IM], Pa   (Fortran prsi, cb after conversion)
    const float* __restrict__ prsl,      // P_PHY     [LM,   JM, IM], Pa   (Fortran prsl)
    const float* __restrict__ prslk,     // PRSLK     [LM,   JM, IM]       (Fortran prslk)
    const float* __restrict__ tgrs,      // T         [LM,   JM, IM], K
    const float* __restrict__ qgrs,      // Q         [LM,   JM, IM], kg/kg
    const float* __restrict__ tracer,    // RAD_GR1   [NTRAC=9, LM, JM, IM] (Fortran GR1, slot order of module_RA_RRTM)
    const float* __restrict__ cldf,      // CLDFRA    [LM,   JM, IM]
    const float* __restrict__ fcice,     // F_ICE     [LM,   JM, IM]  (np3d == 3)
    const float* __restrict__ frain,     // F_RAIN    [LM,   JM, IM]  (np3d == 3)
    const float* __restrict__ rrime,     // F_RIMEF   [LM,   JM, IM]  (np3d == 3)
    const float* __restrict__ tsfc,      // TSFC      [JM, IM], K
    float* __restrict__ plvl,            // RAD_PLVL   [LM+2, JM, IM], mb
    float* __restrict__ plyr,            // RAD_PLYR   [LM+1, JM, IM], mb
    float* __restrict__ tlyr,            // RAD_TLYR   [LM+1, JM, IM], K
    float* __restrict__ prslk1,          // RAD_PRSLK1 [LM+1, JM, IM]
    float* __restrict__ rhly,            // RAD_RHLY   [LM+1, JM, IM]
    float* __restrict__ qstl,            // RAD_QSTL   [LM+1, JM, IM]
    float* __restrict__ cldfk1,          // RAD_CLDFK1 [LM+1, JM, IM]
    float* __restrict__ tracer1,         // RAD_TRACER1 [NTRAC, LM+1, JM, IM]
    float* __restrict__ gcice,           // RAD_GCICE  [LM+1, JM, IM]
    float* __restrict__ grain,           // RAD_GRAIN  [LM+1, JM, IM]
    float* __restrict__ grime,           // RAD_GRIME  [LM+1, JM, IM]
    float* __restrict__ olyr,            // RAD_OLYR   [LM+1, JM, IM]  (written only when ntoz > 0)
    float* __restrict__ tskn,            // RAD_TSKN   [JM, IM], K
    float* __restrict__ tsfg,            // RAD_TSFG   [JM, IM], K
    float* __restrict__ qlyr,            // RAD_QLYR   [LM+1, JM, IM], kg/kg (Fortran qlyr)
    float* __restrict__ tvly,            // RAD_TVLY   [LM+1, JM, IM], K (Fortran tvly)
    const int IM, const int JM, const int LM, const int NTRAC,
    const int ivflip, const int np3d, const int ntoz)
{
    COLUMN_IJ(IM, JM)                    // i0, j0 from one thread per column
    const int LMK = LM + RAD_LTP;        // number of local layers (LMK)
    const int LMP = LMK + 1;             // number of local levels (LMP)

    // kd and the extra-top-layer indices, 0-based (Fortran lines 1166-1195, lextop branch)
    int kd, lla, llb, lya, lyb;
    if (ivflip == 0) {                   // input from toa to sfc
        kd = 1;  lla = 1;  llb = 0;  lya = 1;  lyb = 0;
    } else {                             // input from sfc upward
        kd = 0;  lla = LMK - 1;  llb = LMP - 1;  lya = LM - 1;  lyb = LM;
    }

    // ---- model layers, lines 1199-1262 (k loop) ---------------------------------------------
    for (int k0 = 0; k0 < LM; ++k0) {
        const int l = k0 + kd;
        const float pi_cb = prsi[IDX3(i0, j0, k0, IM, JM)] * 1.0e-3f;   // Pa -> cb
        const float pl_cb = prsl[IDX3(i0, j0, k0, IM, JM)] * 1.0e-3f;   // Pa -> cb
        const float tg = tgrs[IDX3(i0, j0, k0, IM, JM)];
        const float qg = qgrs[IDX3(i0, j0, k0, IM, JM)];
        plvl[IDX3(i0, j0, l, IM, JM)] = 10.0f * pi_cb;                  // cb -> mb
        plyr[IDX3(i0, j0, l, IM, JM)] = 10.0f * pl_cb;                  // cb -> mb
        tlyr[IDX3(i0, j0, l, IM, JM)] = tg;
        prslk1[IDX3(i0, j0, l, IM, JM)] = prslk[IDX3(i0, j0, k0, IM, JM)];

        // es = min(prsl, 0.001 * fpvs(tgrs)); qs = max(QMIN, eps*es/(prsl + epsm1*es))
        const float es = fminf(pl_cb, 0.001f * rad_fpvs(tg));
        const float qs = fmaxf(RAD_QMIN, RAD_CON_EPS * es / (pl_cb + RAD_CON_EPSM1 * es));
        // rhly = max(0, min(1, max(QMIN, qgrs)/qs))
        rhly[IDX3(i0, j0, l, IM, JM)] = fmaxf(0.0f, fminf(1.0f, fmaxf(RAD_QMIN, qg) / qs));
        qstl[IDX3(i0, j0, l, IM, JM)] = qs;

        for (int n = 0; n < NTRAC; ++n) {
            tracer1[IDX4(i0, j0, l, n, IM, JM, LMK)] = tracer[IDX4(i0, j0, k0, n, IM, JM, LM)];
        }
        cldfk1[IDX3(i0, j0, l, IM, JM)] = cldf[IDX3(i0, j0, k0, IM, JM)];
        if (np3d == 3) {                                                // lines 1343-1358
            gcice[IDX3(i0, j0, l, IM, JM)] = fcice[IDX3(i0, j0, k0, IM, JM)];
            grain[IDX3(i0, j0, l, IM, JM)] = frain[IDX3(i0, j0, k0, IM, JM)];
            grime[IDX3(i0, j0, l, IM, JM)] = rrime[IDX3(i0, j0, k0, IM, JM)];
        }
    }
    // model-bottom interface: Fortran plvl(i,LP1+kd) = 10*prsi(i,LP1)
    plvl[IDX3(i0, j0, LM + kd, IM, JM)] = 10.0f * (prsi[IDX3(i0, j0, LM, IM, JM)] * 1.0e-3f);

    // ---- qlyr and tvly, Fortran 1437-1460 (ivflip == 0, toa first) and 1497-1520 (ivflip == 1,
    // sfc first, looped toa to sfc). Both are one top-to-surface recurrence on the moisture floor.
    {
        float tem1d = RAD_QME6;
        for (int s = 0; s < LM; ++s) {
            const int k0 = (ivflip == 0) ? s : (LM - 1 - s);
            const int l = k0 + kd;
            const float ql = fmaxf(tem1d, qgrs[IDX3(i0, j0, k0, IM, JM)]);
            tem1d = fminf(RAD_QME5, ql);
            qlyr[IDX3(i0, j0, l, IM, JM)] = ql;
            tvly[IDX3(i0, j0, l, IM, JM)] = tgrs[IDX3(i0, j0, k0, IM, JM)] * (1.0f + RAD_CON_FVIRT * ql);
        }
    }

    // ---- extra top layer, lines 1262-1300 (lextop) -------------------------------------------
    plvl[IDX3(i0, j0, llb, IM, JM)] = RAD_PRSMIN;
    if (plvl[IDX3(i0, j0, lla, IM, JM)] <= RAD_PRSMIN)
        plvl[IDX3(i0, j0, lla, IM, JM)] = 2.0f * RAD_PRSMIN;
    plyr[IDX3(i0, j0, lyb, IM, JM)] = 0.5f * plvl[IDX3(i0, j0, lla, IM, JM)];
    tlyr[IDX3(i0, j0, lyb, IM, JM)] = tlyr[IDX3(i0, j0, lya, IM, JM)];
    rhly[IDX3(i0, j0, lyb, IM, JM)] = rhly[IDX3(i0, j0, lya, IM, JM)];
    qstl[IDX3(i0, j0, lyb, IM, JM)] = qstl[IDX3(i0, j0, lya, IM, JM)];
    prslk1[IDX3(i0, j0, lyb, IM, JM)] = rad_fpkapx(plyr[IDX3(i0, j0, lyb, IM, JM)] * 100.0f);
    // tracer 1 (ozone, Fortran tracer index 1) copied from the layer below; the rest zeroed
    tracer1[IDX4(i0, j0, lyb, 0, IM, JM, LMK)] = tracer1[IDX4(i0, j0, lya, 0, IM, JM, LMK)];
    for (int n = 1; n < NTRAC; ++n)
        tracer1[IDX4(i0, j0, lyb, n, IM, JM, LMK)] = 0.0f;
    // Fortran cldfk1(i,lyb) = cldf(i,lya): the INPUT cldf array read at the local index lya
    // (grrad_nmmb.f:1347). lya is 0-based and names the input layer directly, so ivflip == 0
    // reads input layer 1 (Fortran cldf(i,2)) and ivflip == 1 reads input layer LM-1 (Fortran LM).
    cldfk1[IDX3(i0, j0, lyb, IM, JM)] = cldf[IDX3(i0, j0, lya, IM, JM)];
    qlyr[IDX3(i0, j0, lyb, IM, JM)] = qlyr[IDX3(i0, j0, lya, IM, JM)];   // lines 1297-1330 lextop
    tvly[IDX3(i0, j0, lyb, IM, JM)] = tvly[IDX3(i0, j0, lya, IM, JM)];
    if (np3d == 3) {
        gcice[IDX3(i0, j0, lyb, IM, JM)] = 1.0f;
        grain[IDX3(i0, j0, lyb, IM, JM)] = 0.0f;
        grime[IDX3(i0, j0, lyb, IM, JM)] = 1.0f;
    }

    // ---- interactive ozone (ntoz > 0), lines 1385-1392; the climatological branch is getozn ---
    if (ntoz > 0) {
        for (int l = 0; l < LMK; ++l)
            olyr[IDX3(i0, j0, l, IM, JM)] = fmaxf(RAD_QMIN, tracer1[IDX4(i0, j0, l, ntoz - 1, IM, JM, LMK)]);
    }

    // ---- surface temperatures, lines 1221-1232: skin and ground both = tsfc ------------------
    tskn[IDX2(i0, j0, IM)] = tsfc[IDX2(i0, j0, IM)];
    tsfg[IDX2(i0, j0, IM)] = tsfc[IDX2(i0, j0, IM)];
}
