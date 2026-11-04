// phys_jsfc_sfcdif.cu -- SFCDIF, Janjic similarity surface layer (NAM JSFC), one thread per column.
//
// Ported to CUDA for nam-gpu from phys/module_SF_JSFC.F90:357-1011 (nam.v4.2.11, 83df4e13c1).
// Source header (kept as the notice; NOAA/NCEP work, no licence file in the NAM clone):
//   SUBROUTINE SFCDIF -- SURFACE LAYER, Janjic (module_SF_JSFC.F90, MODULE MODULE_SF_JSFC).
//
// Layout: one column per thread, i.e. one (i,j) of the 954 x 835 B grid (kernels.grid_cols).
// Every per-column input/output is a flat float32 array indexed by col = j0*IM + i0 (IDX2 in
// nmmb_common.cuh). Module-level PSIM1/PSIH1/PSIM2/PSIH2 (KZTM = 10001) and the scalars
// ZTMIN1, ZTMAX1, ZTMIN2, ZTMAX2, DZETA1, DZETA2, FH01, FH02 are set by JSFC_INIT
// (phys-jsfc-init) and passed in, so this kernel does not own them.
// Constants: module_SF_JSFC PARAMETERs (below), NMMB module_CONSTANTS values (nmmb_common.cuh).
//
// Fortran arguments not used by the body and therefore not bound here: ZSFC (only in the
// commented-out topo term), VEGF and SNOC (never read), I and J (only the write(0,*) diagnostic
// in the sea branch, dropped). NTSD is passed as ntsd. The "ZILFC1, SNOWZO, Zom_ztmax" block is
// commented out in the source and is not ported. CT is identically zero (countergradient block
// commented out in the source).
//
// Faithful quirks kept on purpose: the land branch uses ZSLT = ZSL + ZU ("u,v and t are at the
// same level"); ZT in the land iteration is rebuilt from Z0BASE; the land 10 m AKMS10/AKHS02/AKHS10
// are floored by AKMS/AKHS (not CXCHL-maxed as over sea); the first-step test is NTSD==0.

#include "nmmb_common.cuh"

// ---- module_SF_JSFC PARAMETERs ----
#define JSFC_VKARMAN   0.4f
#define JSFC_ELOCP     (2.72e6f / NMMB_CP)
#define JSFC_A2S       17.2693882f
#define JSFC_A3S       273.16f
#define JSFC_A4S       35.86f
#define JSFC_GLKBR     10.0f
#define JSFC_GLKBS     30.0f
#define JSFC_QVISC     2.1e-5f
#define JSFC_RIC       0.505f
#define JSFC_SMALL     0.35f
#define JSFC_SQPR      0.84f
#define JSFC_SQSC      0.84f
#define JSFC_SQVISC    258.2f
#define JSFC_TVISC     2.1e-5f
#define JSFC_USTC      0.7f
#define JSFC_USTR      0.225f
#define JSFC_VISC      1.5e-5f
#define JSFC_WWST      1.2f
#define JSFC_ZTFC      1.0f
#define JSFC_SEAFC     0.98f
#define JSFC_PQ0SEA    (NMMB_PQ0 * JSFC_SEAFC)
#define JSFC_CZIV      (JSFC_SMALL * JSFC_GLKBS)
#define JSFC_GRRS      (JSFC_GLKBR / JSFC_GLKBS)
#define JSFC_RTVISC    (1.0f / JSFC_TVISC)
#define JSFC_RVISC     (1.0f / JSFC_VISC)
#define JSFC_ZQRZT     (JSFC_SQSC / JSFC_SQPR)
#define JSFC_USTFC     (0.018f / NMMB_G)
#define JSFC_FZQ1      (JSFC_RTVISC * JSFC_QVISC * JSFC_ZQRZT)
#define JSFC_FZQ2      (JSFC_RTVISC * JSFC_QVISC * JSFC_ZQRZT)
#define JSFC_FZT1      (JSFC_RVISC * JSFC_TVISC * JSFC_SQPR)
#define JSFC_FZT2      (JSFC_CZIV * JSFC_GRRS * JSFC_TVISC * JSFC_SQPR)
#define JSFC_FZU1      (JSFC_CZIV * JSFC_VISC)
#define JSFC_WWST2     (JSFC_WWST * JSFC_WWST)
#define JSFC_RQVISC    (1.0f / JSFC_QVISC)
#define JSFC_EPSU2     1.0e-6f
#define JSFC_EPSUST    1.0e-9f
#define JSFC_EPSZT     1.0e-28f
#define JSFC_EXCML     0.0001f
#define JSFC_EXCMS     0.0001f
#define JSFC_CZIL      0.1f
#define JSFC_ZILFC     (-JSFC_CZIL * JSFC_VKARMAN * JSFC_SQVISC)
#define JSFC_CZETMAX   10.0f
#define JSFC_ITRMX     5
#define JSFC_KZTM      10001
#define JSFC_KZTM2     (JSFC_KZTM - 2)

// Fortran table lookup, e.g. PSMZ=(PSIM1(K+2)-PSIM1(K+1))*RDZT+PSIM1(K+1) with K = INT(RZ) clamped
// to [0, KZTM-2]. Fortran PSI(K+1) (1-based) is psi[K] here; RDZT is taken before the clamp.
__device__ __forceinline__ float jsfc_psi(const float* __restrict__ psi,
                                          float zeta, float ztmin, float dzeta)
{
    const float rz = (zeta - ztmin) / dzeta;
    int k = (int)rz;
    const float rdzt = rz - (float)k;
    k = min(k, JSFC_KZTM2);
    k = max(k, 0);
    return (psi[k + 1] - psi[k]) * rdzt + psi[k];
}

extern "C" __global__ void phys_jsfc_sfcdif_column(
    // per-column inputs
    const float* __restrict__ seamask, const float* __restrict__ ths,
    const float* __restrict__ psfc, const float* __restrict__ tz0,
    const float* __restrict__ zsl, const float* __restrict__ pblh,
    const float* __restrict__ wetm, const float* __restrict__ z0base,
    const float* __restrict__ ulow, const float* __restrict__ vlow,
    const float* __restrict__ tlow, const float* __restrict__ thlow,
    const float* __restrict__ thelow, const float* __restrict__ qlow,
    const float* __restrict__ cwmlow, const float* __restrict__ plow,
    // INOUT state (Fortran INTENT(INOUT))
    float* __restrict__ qs, float* __restrict__ uz0, float* __restrict__ vz0,
    float* __restrict__ thz0, float* __restrict__ qz0, float* __restrict__ ustar,
    float* __restrict__ z0, float* __restrict__ akms, float* __restrict__ akhs,
    // outputs (Fortran INTENT(OUT))
    float* __restrict__ ct, float* __restrict__ rlmo, float* __restrict__ rib,
    float* __restrict__ chs, float* __restrict__ chs2, float* __restrict__ cqs2,
    float* __restrict__ hfx, float* __restrict__ qfx, float* __restrict__ flx_lh,
    float* __restrict__ flhc, float* __restrict__ flqc, float* __restrict__ qgh,
    float* __restrict__ cpm, float* __restrict__ u10, float* __restrict__ v10,
    float* __restrict__ th02, float* __restrict__ th10, float* __restrict__ q02,
    float* __restrict__ q10, float* __restrict__ pshltr,
    // JSFC_INIT tables and scalars (module_SF_JSFC.F90:65-69)
    const float* __restrict__ psim1, const float* __restrict__ psih1,
    const float* __restrict__ psim2, const float* __restrict__ psih2,
    float ztmin1, float ztmax1, float ztmin2, float ztmax2,
    float dzeta1, float dzeta2, float fh01, float fh02,
    int ntsd, int n, int im, const int* __restrict__ lim)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    // Tile guard: SFCDIF runs only over the JSFC B1 tile (1-based ITS_B1..ITE_B1, JTS_B1..JTE_B1 from
    // the drv-loop-limits table), as the Fortran JSFC call (module_TURBULENCE.F90:798-802) passes it.
    {
        const int I = col % im + 1;
        const int J = col / im + 1;
        if (I < lim[DL_ITS_B1] || I > lim[DL_ITE_B1] || J < lim[DL_JTS_B1] || J > lim[DL_JTE_B1]) return;
    }

    const float sea = seamask[col];
    const float ths_c = ths[col];
    const float psfc_c = psfc[col];
    const float tz0_c = tz0[col];
    const float zsl_c = zsl[col];
    const float pblh_c = pblh[col];
    const float wetm_c = wetm[col];
    const float z0base_c = z0base[col];
    const float ulow_c = ulow[col];
    const float vlow_c = vlow[col];
    const float tlow_c = tlow[col];
    const float thlow_c = thlow[col];
    const float thelow_c = thelow[col];
    const float qlow_c = qlow[col];
    const float cwmlow_c = cwmlow[col];
    const float plow_c = plow[col];

    float ust = ustar[col];
    float zz0 = z0[col];
    float akmsv = akms[col];
    float akhsv = akhs[col];
    float qsv = qs[col];
    float uz0v = uz0[col];
    float vz0v = vz0[col];
    float thz0v = thz0[col];
    float qz0v = qz0[col];

    // setup (module SFCDIF preamble)
    const float rdz = 1.0f / zsl_c;
    const float cxchl = JSFC_EXCML * rdz;
    const float cxchs = JSFC_EXCMS * rdz;
    const float btgx = NMMB_G / thlow_c;
    const float elfc = JSFC_VKARMAN * btgx;
    const float btgh = (pblh_c > 1000.0f) ? btgx * pblh_c : btgx * 1000.0f;

    // Locals carried across the branches and into the diagnostic block.
    float zu = zz0, zt = zz0;
    float dthv = 0.0f, du2 = JSFC_EPSU2, rib_c = 0.0f, rlmo_c = 0.0f;
    float wstar2 = 0.0f;
    float ustark = ust * JSFC_VKARMAN;
    float psmz = 0.0f, pshz = 0.0f;

    if (sea > 0.5f) {
        // ---- SEA POINTS ----
        for (int itr = 0; itr < JSFC_ITRMX; ++itr) {
            zz0 = fmaxf(JSFC_USTFC * ust * ust, 1.59E-5f);

            if (ust < JSFC_USTC) {
                if (ust < JSFC_USTR) {
                    // viscous sublayer, first step only: seed AKMS/AKHS/QS
                    if (ntsd == 0) {
                        akmsv = cxchs;
                        akhsv = cxchs;
                        qsv = qlow_c;
                    }
                    zu = JSFC_FZU1 * sqrtf(sqrtf(zz0 * ust * JSFC_RVISC)) / ust;
                    const float wght = akmsv * zu * JSFC_RVISC;
                    const float rwgh = wght / (wght + 1.0f);
                    uz0v = (ulow_c * rwgh + uz0v) * 0.5f;
                    vz0v = (vlow_c * rwgh + vz0v) * 0.5f;
                    zt = JSFC_FZT1 * zu;
                    const float zq = JSFC_FZQ1 * zt;
                    const float wghtt = akhsv * zt * JSFC_RTVISC;
                    const float wghtq = akhsv * zq * JSFC_RQVISC;
                    if (ntsd > 0) {
                        thz0v = ((wghtt * thlow_c + ths_c) / (wghtt + 1.0f) + thz0v) * 0.5f;
                        qz0v = ((wghtq * qlow_c + qsv) / (wghtq + 1.0f) + qz0v) * 0.5f;
                    } else {
                        thz0v = (wghtt * thlow_c + ths_c) / (wghtt + 1.0f);
                        qz0v = (wghtq * qlow_c + qsv) / (wghtq + 1.0f);
                    }
                }
                if (ust >= JSFC_USTR && ust < JSFC_USTC) {
                    zu = zz0;
                    uz0v = 0.0f;
                    vz0v = 0.0f;
                    zt = JSFC_FZT2 * sqrtf(sqrtf(zz0 * ust * JSFC_RVISC)) / ust;
                    const float zq = JSFC_FZQ2 * zt;
                    const float wghtt = akhsv * zt * JSFC_RTVISC;
                    const float wghtq = akhsv * zq * JSFC_RQVISC;
                    if (ntsd > 0) {
                        thz0v = ((wghtt * thlow_c + ths_c) / (wghtt + 1.0f) + thz0v) * 0.5f;
                        qz0v = ((wghtq * qlow_c + qsv) / (wghtq + 1.0f) + qz0v) * 0.5f;
                    } else {
                        thz0v = (wghtt * thlow_c + ths_c) / (wghtt + 1.0f);
                        qz0v = (wghtq * qlow_c + qsv) / (wghtq + 1.0f);
                    }
                }
            } else {
                zu = zz0;
                uz0v = 0.0f;
                vz0v = 0.0f;
                zt = zz0;
                thz0v = ths_c;
                qz0v = qsv;
            }

            const float tem = (tlow_c + tz0_c) * 0.5f;
            const float thm = (thelow_c + thz0v) * 0.5f;
            const float a = thm * NMMB_P608;
            const float b = (JSFC_ELOCP / tem - 1.0f - NMMB_P608) * thm;
            dthv = ((thelow_c - thz0v) * ((qlow_c + qz0v + cwmlow_c) * (0.5f * NMMB_P608) + 1.0f)
                    + (qlow_c - qz0v + cwmlow_c) * a + cwmlow_c * b);
            du2 = fmaxf((ulow_c - uz0v) * (ulow_c - uz0v)
                        + (vlow_c - vz0v) * (vlow_c - vz0v), JSFC_EPSU2);
            rib_c = btgx * dthv * zsl_c / du2;

            const float zslu = zsl_c + zu;
            const float zslt = zsl_c + zt;
            const float rzsu = zslu / zu;
            const float rzst = zslt / zt;
            const float rlogu = logf(rzsu);
            const float rlogt = logf(rzst);

            rlmo_c = elfc * akhsv * dthv / (ust * ust * ust);
            float zetalu = zslu * rlmo_c;
            float zetalt = zslt * rlmo_c;
            float zetau = zu * rlmo_c;
            float zetat = zt * rlmo_c;
            zetalu = fminf(fmaxf(zetalu, ztmin1), ztmax1);
            zetalt = fminf(fmaxf(zetalt, ztmin1), ztmax1);
            zetau = fminf(fmaxf(zetau, ztmin1 / rzsu), ztmax1 / rzsu);
            zetat = fminf(fmaxf(zetat, ztmin1 / rzst), ztmax1 / rzst);

            psmz = jsfc_psi(psim1, zetau, ztmin1, dzeta1);
            const float psmzl = jsfc_psi(psim1, zetalu, ztmin1, dzeta1);
            const float simm = psmzl - psmz + rlogu;
            pshz = jsfc_psi(psih1, zetat, ztmin1, dzeta1);
            const float pshzl = jsfc_psi(psih1, zetalt, ztmin1, dzeta1);
            const float simh = (pshzl - pshz + rlogt) * fh01;

            ustark = ust * JSFC_VKARMAN;
            akmsv = fmaxf(ustark / simm, cxchs);
            akhsv = fmaxf(ustark / simh, cxchs);

            if (dthv <= 0.0f) {
                wstar2 = JSFC_WWST2 * powf(fabsf(btgh * akhsv * dthv), 2.0f / 3.0f);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmaxf(sqrtf(akmsv * sqrtf(du2 + wstar2)), JSFC_EPSUST);
        }
    } else {
        // ---- LAND POINTS ----
        if (ntsd == 0) qsv = qlow_c;
        zu = zz0;
        uz0v = 0.0f;
        vz0v = 0.0f;
        zt = zu * JSFC_ZTFC;
        thz0v = ths_c;
        qz0v = qsv;

        const float tem = (tlow_c + tz0_c) * 0.5f;
        const float thm = (thelow_c + thz0v) * 0.5f;
        const float a = thm * NMMB_P608;
        const float b = (JSFC_ELOCP / tem - 1.0f - NMMB_P608) * thm;
        dthv = ((thelow_c - thz0v) * ((qlow_c + qz0v + cwmlow_c) * (0.5f * NMMB_P608) + 1.0f)
                + (qlow_c - qz0v + cwmlow_c) * a + cwmlow_c * b);
        du2 = fmaxf((ulow_c - uz0v) * (ulow_c - uz0v)
                    + (vlow_c - vz0v) * (vlow_c - vz0v), JSFC_EPSU2);
        rib_c = btgx * dthv * zsl_c / du2;

        const float zslu = zsl_c + zu;
        const float rzsu = zslu / zu;
        const float rlogu = logf(rzsu);
        const float zslt = zsl_c + zu;          // u, v and t are at the same level

        // RIB modification of the Zilitinkevitch term
        float zzil;
        if (dthv > 0.0f) {
            if (rib_c < JSFC_RIC) {
                zzil = JSFC_ZILFC * (1.0f + (rib_c / JSFC_RIC) * (rib_c / JSFC_RIC) * JSFC_CZETMAX);
            } else {
                zzil = JSFC_ZILFC * (JSFC_CZETMAX + 1.0f);
            }
        } else {
            zzil = JSFC_ZILFC;
        }

        for (int itr = 0; itr < JSFC_ITRMX; ++itr) {
            // Zilitinkevitch fix for ZT, from Z0BASE
            zt = fmaxf(expf(zzil * sqrtf(ust * z0base_c)) * z0base_c, JSFC_EPSZT);
            const float rzst = zslt / zt;
            const float rlogt = logf(rzst);

            rlmo_c = elfc * akhsv * dthv / (ust * ust * ust);
            float zetalu = zslu * rlmo_c;
            float zetalt = zslt * rlmo_c;
            float zetau = zu * rlmo_c;
            float zetat = zt * rlmo_c;
            zetalu = fminf(fmaxf(zetalu, ztmin2), ztmax2);
            zetalt = fminf(fmaxf(zetalt, ztmin2), ztmax2);
            zetau = fminf(fmaxf(zetau, ztmin2 / rzsu), ztmax2 / rzsu);
            zetat = fminf(fmaxf(zetat, ztmin2 / rzst), ztmax2 / rzst);

            psmz = jsfc_psi(psim2, zetau, ztmin2, dzeta2);
            const float psmzl = jsfc_psi(psim2, zetalu, ztmin2, dzeta2);
            const float simm = psmzl - psmz + rlogu;
            pshz = jsfc_psi(psih2, zetat, ztmin2, dzeta2);
            const float pshzl = jsfc_psi(psih2, zetalt, ztmin2, dzeta2);
            const float simh = (pshzl - pshz + rlogt) * fh02;

            ustark = ust * JSFC_VKARMAN;
            akmsv = fmaxf(ustark / simm, cxchl);
            akhsv = fmaxf(ustark / simh, cxchl);

            if (dthv <= 0.0f) {
                wstar2 = JSFC_WWST2 * powf(fabsf(btgh * akhsv * dthv), 2.0f / 3.0f);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmaxf(sqrtf(akmsv * sqrtf(du2 + wstar2)), JSFC_EPSUST);
        }
    }
    // countergradient term: commented out in the source, CT = 0
    const float ct_c = 0.0f;

    // ---- 2 m / 10 m diagnostics ----
    const float umflx = akmsv * (ulow_c - uz0v);
    const float vmflx = akmsv * (vlow_c - vz0v);
    const float hsflx = akhsv * (thlow_c - thz0v);
    const float hlflx = akhsv * (qlow_c - qz0v);

    const float zu10 = zu + 10.0f;
    const float zt02 = zt + 2.0f;
    const float zt10 = zt + 10.0f;
    const float rlnu10 = logf(zu10 / zu);
    const float rlnt02 = logf(zt02 / zt);
    const float rlnt10 = logf(zt10 / zt);

    float ztau10 = zu10 * rlmo_c;
    float ztat02 = zt02 * rlmo_c;
    float ztat10 = zt10 * rlmo_c;

    float akms10, akhs02, akhs10;
    if (sea > 0.5f) {
        ztau10 = fminf(fmaxf(ztau10, ztmin1), ztmax1);
        ztat02 = fminf(fmaxf(ztat02, ztmin1), ztmax1);
        ztat10 = fminf(fmaxf(ztat10, ztmin1), ztmax1);
        const float psm10 = jsfc_psi(psim1, ztau10, ztmin1, dzeta1);
        const float simm10 = psm10 - psmz + rlnu10;
        const float psh02 = jsfc_psi(psih1, ztat02, ztmin1, dzeta1);
        const float simh02 = (psh02 - pshz + rlnt02) * fh01;
        const float psh10 = jsfc_psi(psih1, ztat10, ztmin1, dzeta1);
        const float simh10 = (psh10 - pshz + rlnt10) * fh01;
        akms10 = fmaxf(ustark / simm10, cxchs);
        akhs02 = fmaxf(ustark / simh02, cxchs);
        akhs10 = fmaxf(ustark / simh10, cxchs);
    } else {
        ztau10 = fminf(fmaxf(ztau10, ztmin2), ztmax2);
        ztat02 = fminf(fmaxf(ztat02, ztmin2), ztmax2);
        ztat10 = fminf(fmaxf(ztat10, ztmin2), ztmax2);
        const float psm10 = jsfc_psi(psim2, ztau10, ztmin2, dzeta2);
        const float simm10 = psm10 - psmz + rlnu10;
        const float psh02 = jsfc_psi(psih2, ztat02, ztmin2, dzeta2);
        const float simh02 = (psh02 - pshz + rlnt02) * fh02;
        const float psh10 = jsfc_psi(psih2, ztat10, ztmin2, dzeta2);
        const float simh10 = (psh10 - pshz + rlnt10) * fh02;
        akms10 = ustark / simm10;
        akhs02 = ustark / simh02;
        akhs10 = ustark / simh10;
        if (akms10 <= cxchl) akms10 = akmsv;
        if (akhs02 <= cxchl) akhs02 = akhsv;
        if (akhs10 <= cxchl) akhs10 = akhsv;
    }

    const float u10_c0 = umflx / akms10 + uz0v;
    const float v10_c0 = vmflx / akms10 + vz0v;
    const float th02_c = hsflx / akhs02 + thz0v;
    const float th10_c = hsflx / akhs10 + thz0v;
    const float q02_c = hlflx / akhs02 + qz0v;
    const float q10_c = hlflx / akhs10 + qz0v;
    const float term1 = -0.068283f / tlow_c;
    const float pshltr_c = psfc_c * expf(term1);

    // ---- equivalent Z0 for the shelter winds (land only; overrides U10/V10) ----
    float u10e = u10_c0;
    float v10e = v10_c0;
    if (sea < 0.5f) {
        const float zuuz = fminf(zu * 0.50f, 0.18f);
        const float zu_e = fmaxf(zu * 0.35f, zuuz);
        const float zu10e = zu_e + 10.0f;
        const float rzsu_e = zu10e / zu_e;
        const float rlnu10e = logf(rzsu_e);
        float zetau_e = zu_e * rlmo_c;
        float ztau10e = zu10e * rlmo_c;
        ztau10e = fminf(fmaxf(ztau10e, ztmin2), ztmax2);
        zetau_e = fminf(fmaxf(zetau_e, ztmin2 / rzsu_e), ztmax2 / rzsu_e);
        const float psm10e = jsfc_psi(psim2, ztau10e, ztmin2, dzeta2);
        const float simm10e = psm10e - psmz + rlnu10e;
        const float ekms10 = fmaxf(ustark / simm10e, cxchl);
        u10e = umflx / ekms10 + uz0v;
        v10e = vmflx / ekms10 + vz0v;
    }

    // ---- WRF-style driver arrays ----
    const float rlow = plow_c / (NMMB_R_D * tlow_c);
    const float hfx_c = -rlow * NMMB_CP * hsflx;
    const float qfx_c = -rlow * hlflx * wetm_c;
    const float flx_lh_c = NMMB_ELWV * qfx_c;
    const float flhc_c = rlow * NMMB_CP * akhsv;
    const float flqc_c = rlow * akhsv * wetm_c;
    const float qgh_c = ((1.0f - sea) * NMMB_PQ0 + sea * JSFC_PQ0SEA) / plow_c
                        * expf(JSFC_A2S * (tlow_c - JSFC_A3S) / (tlow_c - JSFC_A4S));
    const float cpm_c = NMMB_CP * (1.0f + 0.8f * qlow_c);

    // QS over water is diagnosed from the surface flux; over land QS is prognostic.
    if (sea > 0.5f) {
        qsv = qlow_c + qfx_c / (rlow * akhsv);
        qsv = qsv / (1.0f - qsv);
    }

    // ---- write back ----
    ustar[col] = ust;
    z0[col] = zz0;
    akms[col] = akmsv;
    akhs[col] = akhsv;
    qs[col] = qsv;
    uz0[col] = uz0v;
    vz0[col] = vz0v;
    thz0[col] = thz0v;
    qz0[col] = qz0v;

    ct[col] = ct_c;
    rlmo[col] = rlmo_c;
    rib[col] = rib_c;
    chs[col] = akhsv;
    chs2[col] = akhs02;
    cqs2[col] = akhs02;
    hfx[col] = hfx_c;
    qfx[col] = qfx_c;
    flx_lh[col] = flx_lh_c;
    flhc[col] = flhc_c;
    flqc[col] = flqc_c;
    qgh[col] = qgh_c;
    cpm[col] = cpm_c;
    u10[col] = u10e;
    v10[col] = v10e;
    th02[col] = th02_c;
    th10[col] = th10_c;
    q02[col] = q02_c;
    q10[col] = q10_c;
    pshltr[col] = pshltr_c;
}
