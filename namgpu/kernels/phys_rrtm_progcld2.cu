// phys_rrtm_progcld2.cu -- nam-gpu unit phys-rrtm-progcld2: PROGCLD2, Ferrier prognostic cloud
// optical properties for the RRTM radiation (np3d = 3, Ferrier microphysics).
//
// NOTICE: NOAA/NCEP NMMB source, US government work (no licence file in the NAM clone). Header of
// phys/radiation_clouds_nmmb.f, module_radiation_clouds (NMMB version; authors Brad Ferrier,
// Yu-Tai Hou, Hsin-Mu Lin; progcld2 and rsipath2_tmp are the NMMB progcld2 branch):
//   "computes cloud related quantities using Ferrier's prognostic cloud microphysics scheme"
// Ported to CUDA for nam-gpu from phys/radiation_clouds_nmmb.f:787-1323 (nam.v4.2.11, 83df4e13c1)
// with the two routines it calls, rsipath2_tmp (:2779-2964) and gethml (:2167-2494), inlined as
// device code so the unit is one launch.
//
// Units: plyr and plvl are in hPa (mb) exactly as the Fortran uses them (rho = 100*plyr/...,
// cpath uses 100000/g). The kernel converts nothing; the caller supplies hPa.
//
// Mapping (Fortran local -> here):
//   progcld2 cldtot/cwp/cip/crp/csp/rew/rei/rer  -> per level scalars, written to clouds[] at once
//   cldcnv                                       -> 0 (never set in progcld2)
//   tem2d (1st: tlyr-t0c, used for ice class)    -> tem2d_t
//   tem2d (2nd: layer density, used for rei)     -> tem2d
//   clwf, clw2, qcice, qrain, qcwat, fcice, frain -> per level scalars (lcrick uses clw neighbours)
//   res, rsden, rrime, tc, qsnow, snow path      -> computed by rsipath2_tmp but never read by
//       progcld2 (clouds 8, 9 are zero; snden = 0 always); not kept
//   ptop1(IX,4)                                  -> ptop1[4] per column
//   cl1/cl2/idom/kbt/kth/clds/mtop/mbot in gethml -> per column scalars and small local arrays
//   cldtot array (needed by gethml across levels) -> local float[PGC_MAXLAY] per column thread
//   xlon, slmsk, flgmin, iflip/lprnt             -> not read by progcld2 or rsipath2_tmp; not passed
//   IX loops                                     -> one thread per column (COLUMN_IJ)
// MPI/halo/OpenMP/print: none in these routines. NMMB 'CHK' chunking is a CPU detail; dropped.
//
// Fortran 1-based level k is used in the gethml section with CT1()/PL1() helpers so the code reads
// like the source. ivflip selects the Fortran branch (0: index 1 = model top; 1: index 1 = surface).
// llyr is the Fortran 1-based module value (phys-rrtm-cld-init sets it; default 2 in the module).
// iovr is max(iovrsw, iovrlw) (module value; default 1).
#include "nmmb_common.cuh"

#define PGC_NF_CLDS    9
#define PGC_MAXLAY     128

// physcons.f (kind_phys = 4): NOTE these differ from module_CONSTANTS in nmmb_common.cuh.
#define PGC_CON_PI     3.1415926535897931f
#define PGC_CON_G      9.80665f
#define PGC_CON_RD     287.05f
#define PGC_CON_RV     461.5f
#define PGC_CON_FVIRT  (PGC_CON_RV / PGC_CON_RD - 1.0f)
#define PGC_CON_T0C    273.15f
#define PGC_CON_TTP    273.16f

// radiation_clouds_nmmb.f module parameters
#define PGC_CCLIMIT     0.001f
#define PGC_CCLIMIT2    0.05f
#define PGC_CLIMIT      0.001f
#define PGC_OVCST       (1.0f - 1.0e-8f)
#define PGC_RECWAT_DEF  5.0f
#define PGC_RECICE_DEF  10.0f
#define PGC_RERAIN_DEF  100.0f
#define PGC_RE_50C      (1250.0f / 9.917f)
#define PGC_RE_40C      (1250.0f / 9.337f)
#define PGC_RE_30C      (1250.0f / 9.208f)
#define PGC_RE_20C      (1250.0f / 9.387f)

// rsipath2_tmp parameters
#define PGC_CEXP        (1.0f / 3.0f)
#define PGC_EPSQ        1.0e-12f
#define PGC_CN0R0       2511.54f
#define PGC_TNW         200.0f

// Cloud fraction, power form: the Fortran "(onemrh*qstl)**0.49" branch (sashal, and ivflip == 0
// with lsashal). clwmin is 0 in both Fortran branches that call this.
__device__ inline float pgc_cldtot_pow(float clw2, float pl, float qstl, float rhly)
{
    const float clwm = 0.0f / fmaxf(0.01f, pl * 0.001f);      // clwmin / max(0.01, plyr*0.001)
    const float onemrh = fmaxf(1.0e-10f, 1.0f - rhly);
    float tem1 = fminf(fmaxf(powf(onemrh * qstl, 0.49f), 0.0001f), 1.0f);
    tem1 = 100.0f / tem1;
    const float value = fmaxf(fminf(tem1 * (clw2 - clwm), 50.0f), 0.0f);
    const float tem2 = sqrtf(sqrtf(rhly));
    return fmaxf(tem2 * (1.0f - expf(-value)), 0.0f);
}

// Cloud fraction, sqrt(sqrt()) form with 2000 / tem1: ivflip /= 0 without lsashal.
__device__ inline float pgc_cldtot_sqrt(float clw2, float pl, float qstl, float rhly)
{
    const float clwm = 0.0f / fmaxf(0.01f, pl * 0.001f);
    const float onemrh = fmaxf(1.0e-10f, 1.0f - rhly);
    float tem1 = fminf(fmaxf(sqrtf(sqrtf(onemrh * qstl)), 0.0001f), 1.0f);
    tem1 = 2000.0f / tem1;
    const float value = fmaxf(fminf(tem1 * (clw2 - clwm), 50.0f), 0.0f);
    const float tem2 = sqrtf(sqrtf(rhly));
    return fmaxf(tem2 * (1.0f - expf(-value)), 0.0f);
}

extern "C" __global__ void phys_rrtm_progcld2_column(
    const int im, const int jm, const int nlay,
    const int ivflip, const int llyr, const int iovr,
    const int lcrick, const int lcnorm, const int lnoprec, const int lsashal,
    const int cld_fraction,
    const float* __restrict__ plyr,     // [NLAY, JM, IM] hPa
    const float* __restrict__ plvl,     // [NLAY+1, JM, IM] hPa
    const float* __restrict__ tlyr,     // [NLAY, JM, IM] K
    const float* __restrict__ tvly,     // [NLAY, JM, IM] K
    const float* __restrict__ qlyr,     // [NLAY, JM, IM] kg/kg
    const float* __restrict__ qstl,     // [NLAY, JM, IM] kg/kg
    const float* __restrict__ rhly,     // [NLAY, JM, IM]
    const float* __restrict__ clw,      // [NLAY, JM, IM] total condensate (CW)
    const float* __restrict__ f_ice,    // [NLAY, JM, IM]
    const float* __restrict__ f_rain,   // [NLAY, JM, IM]
    const float* __restrict__ r_rime,   // [NLAY, JM, IM] (unused by rsipath2_tmp; kept for the call)
    const float* __restrict__ cldfk1,   // [NLAY, JM, IM] Thompson cloud fraction (cld_fraction /= 0)
    const float* __restrict__ xlat,     // [JM, IM] radians
    float* __restrict__ clouds,         // [NF_CLDS, NLAY, JM, IM]
    float* __restrict__ clds,           // [5, JM, IM]
    int* __restrict__ mtop,             // [3, JM, IM]
    int* __restrict__ mbot)             // [3, JM, IM]
{
    COLUMN_IJ(im, jm)

    // Fortran 1-based helpers for the gethml section.
    #define CT1(k)  cldtot[(k) - 1]
    #define PL1(k)  plyr[IDX3(i0, j0, (k) - 1, im, jm)]

    // rsipath2_tmp: recw1 = 620.35/TNW**CEXP (single precision, computed once per column).
    const float recw1 = 620.35f / powf(PGC_TNW, PGC_CEXP);

    // ---- ptop1 for this column (progcld2 lines 969-978) --------------------------------------
    float ptop1[4];
    {
        const float ptopc1[4] = {1050.0f, 650.0f, 400.0f, 0.0f};   // data ptopc(:,1)
        const float ptopc2[4] = {1050.0f, 750.0f, 500.0f, 0.0f};   // data ptopc(:,2)
        const float xl = xlat[IDX2(i0, j0, im)];
        const float tem2 = fmaxf(0.0f, 4.0f * fabsf(xl) / PGC_CON_PI - 1.0f);
        for (int id = 0; id < 4; ++id) {
            const float tem1 = ptopc2[id] - ptopc1[id];
            ptop1[id] = ptopc1[id] + tem1 * tem2;
        }
    }

    float cldtot[PGC_MAXLAY];

    // ---- per level: progcld2 lines 979-1290 and rsipath2_tmp; levels are independent ----------
    for (int k0 = 0; k0 < nlay; ++k0) {
        const size_t p = IDX3(i0, j0, k0, im, jm);
        const float pl  = plyr[p];                                  // plyr(i,k)
        const float plo = plvl[p];                                  // plvl(i,k)
        const float plp = plvl[IDX3(i0, j0, k0 + 1, im, jm)];       // plvl(i,k+1)

        // clwf: lcrick smoothing of clw (progcld2 lines 979-1000)
        float clwf;
        if (lcrick) {
            if (k0 == 0) {
                clwf = 0.75f * clw[p] + 0.25f * clw[IDX3(i0, j0, 1, im, jm)];
            } else if (k0 == nlay - 1) {
                clwf = 0.75f * clw[p] + 0.25f * clw[IDX3(i0, j0, nlay - 2, im, jm)];
            } else {
                clwf = 0.25f * clw[IDX3(i0, j0, k0 - 1, im, jm)] + 0.5f * clw[p]
                     + 0.25f * clw[IDX3(i0, j0, k0 + 1, im, jm)];
            }
        } else {
            clwf = clw[p];
        }

        // fcice, frain clipped to [0,1] (progcld2 init loop).
        const float fcice = fmaxf(0.0f, fminf(1.0f, f_ice[p]));
        const float frain = fmaxf(0.0f, fminf(1.0f, f_rain[p]));

        // Ice / liquid / rain split on tem2d = tlyr - con_t0c (progcld2 classification loop).
        const float tem2d_t = tlyr[p] - PGC_CON_T0C;
        float qcice, qrain, qcwat, clw2;
        if (tem2d_t > -40.0f) {
            qcice = clwf * fcice;
            const float tem1 = clwf - qcice;
            qrain = tem1 * frain;
            qcwat = tem1 - qrain;
            clw2 = qcwat + qcice;
        } else {
            qcice = clwf;
            qrain = 0.0f;
            qcwat = 0.0f;
            clw2 = clwf;
        }

        // rsipath2_tmp (per level): hydrometeor paths and effective radii.
        float cwp = 0.0f, cip = 0.0f, crp = 0.0f, csp = 0.0f;
        float rew = PGC_RECWAT_DEF;     // progcld2 default before rsipath2
        float rei = PGC_RECICE_DEF;     // progcld2 default before the ice radius block
        float rer = PGC_RERAIN_DEF;     // progcld2 default before rsipath2
        {
            const float totcnd = qcwat + qcice + qrain;
            if (totcnd > PGC_EPSQ) {
                const float rho = 100.0f * pl / (PGC_CON_RD * tlyr[p] * (1.0f + PGC_CON_FVIRT * qlyr[p]));
                const float cpath = fabsf(plp - plo) * (100000.0f / PGC_CON_G);
                if (qcwat > 0.0f) {
                    const float tem = recw1 * powf(rho * qcwat, PGC_CEXP);
                    rew = fmaxf(PGC_RECWAT_DEF, tem);
                    cwp = cpath * qcwat;
                }
                if (qrain > 0.0f) {
                    const float tem = PGC_CN0R0 * sqrtf(sqrtf(rho * qrain));
                    rer = 1.5f * fmaxf(50.0f, fminf(1000.0f, tem));
                    crp = cpath * qrain;
                }
                cip = cpath * qcice;     // all ice is cloud ice (Oct 2013); no snow path
            }
        }

        // layer density: the second tem2d assignment (progcld2 lines 1012-1026).
        float tem2d;
        if (ivflip == 0) {
            tem2d = (PGC_CON_G * pl) / (PGC_CON_RD * (plp - plo));
        } else {
            tem2d = (PGC_CON_G * pl) / (PGC_CON_RD * (plo - plp));
        }

        // layer cloud fraction (progcld2 cldfracs block).
        float ct = 0.0f;
        if (cld_fraction == 0) {
            if (ivflip == 0) {
                if (!lsashal) {
                    const float clwt = 1.0e-8f * (pl * 0.001f);
                    if (clw2 > clwt) {
                        const float tem1 = 1.0e5f * clw2 * pl * 0.001f;
                        ct = fminf(1.0f, tem1);
                    }
                } else {
                    const float clwt = 2.0e-6f * (pl * 0.001f);
                    if (clw2 > clwt) ct = pgc_cldtot_pow(clw2, pl, qstl[p], rhly[p]);
                }
            } else {
                const float clwt = 2.0e-6f * (pl * 0.001f);
                if (clw2 > clwt) {
                    if (!lsashal) ct = pgc_cldtot_sqrt(clw2, pl, qstl[p], rhly[p]);
                    else          ct = pgc_cldtot_pow(clw2, pl, qstl[p], rhly[p]);
                }
            }
        } else {
            ct = cldfk1[p];         // Thompson cloud fraction branch
        }

        // cclimit zeroing.
        if (ct < PGC_CCLIMIT) {
            ct = 0.0f;
            cwp = 0.0f;
            cip = 0.0f;
            crp = 0.0f;
            csp = 0.0f;
        }
        // lnoprec: precipitation has no radiative effect.
        if (lnoprec) {
            crp = 0.0f;
            csp = 0.0f;
        }
        // lcnorm: in-cloud condensate normalisation.
        if (lcnorm) {
            if (ct >= PGC_CCLIMIT) {
                const float tem1 = 1.0f / fmaxf(PGC_CCLIMIT2, ct);
                cwp = cwp * tem1;
                cip = cip * tem1;
                crp = crp * tem1;
                csp = csp * tem1;
            }
        }

        // effective ice radius.
        if (cip > 0.0f) {
            const float tem3 = tem2d * cip / tvly[p];      // IWC, g m^-3
            const float tem1 = tlyr[p] - PGC_CON_TTP;
            float tem2;
            if (tem1 < -50.0f) {
                tem2 = PGC_RE_50C * powf(tem3, 0.109f);
            } else if (tem1 < -40.0f) {
                tem2 = PGC_RE_40C * powf(tem3, 0.08f);
            } else if (tem1 < -30.0f) {
                tem2 = PGC_RE_30C * powf(tem3, 0.055f);
            } else {
                tem2 = PGC_RE_20C * powf(tem3, 0.031f);
            }
            rei = fmaxf(PGC_RECICE_DEF, tem2 * 0.6495f);
        }

        cldtot[k0] = ct;

        // clouds(:,:,1..7); 8 and 9 stay zero (progcld2 zeroes all NF_CLDS first).
        clouds[IDX4(i0, j0, k0, 0, im, jm, nlay)] = ct;
        clouds[IDX4(i0, j0, k0, 1, im, jm, nlay)] = cwp;
        clouds[IDX4(i0, j0, k0, 2, im, jm, nlay)] = rew;
        clouds[IDX4(i0, j0, k0, 3, im, jm, nlay)] = cip;
        clouds[IDX4(i0, j0, k0, 4, im, jm, nlay)] = rei;
        clouds[IDX4(i0, j0, k0, 5, im, jm, nlay)] = crp;
        clouds[IDX4(i0, j0, k0, 6, im, jm, nlay)] = rer;
        clouds[IDX4(i0, j0, k0, 7, im, jm, nlay)] = 0.0f;
        clouds[IDX4(i0, j0, k0, 8, im, jm, nlay)] = 0.0f;
        (void)tem2d;   // used above for rei
    }

    // ---- gethml (radiation_clouds_nmmb.f:2167-2494): cloud layer fractions and indices --------
    float cl[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};    // clds(i,1..5)
    int mt[4] = {0, 0, 0, 0};                         // mtop(i,1..3); slot 4 is a guard only
    int mb[4] = {0, 0, 0, 0};

    // total and bl clouds (gethml: random or max/ran overlap, layers from the top or surface).
    {
        float cl1 = 1.0f, cl2 = 1.0f;
        for (int kk = 0; kk < nlay; ++kk) {
            const int k = (ivflip == 0) ? (nlay - kk) : (kk + 1);   // Fortran 1-based
            const float ccur = fminf(PGC_OVCST, fmaxf(CT1(k), 0.0f));  // cldcnv = 0
            if (iovr == 0) {
                if (ccur >= PGC_CLIMIT) cl1 = cl1 * (1.0f - ccur);
                if (k == llyr) cl[4] = 1.0f - cl1;
            } else {
                if (ccur >= PGC_CLIMIT) {
                    cl2 = fminf(cl2, 1.0f - ccur);
                } else {
                    cl1 = cl1 * cl2;
                    cl2 = 1.0f;
                }
                if (k == llyr) cl[4] = 1.0f - cl1 * cl2;
            }
        }
        if (iovr == 0) {
            cl[3] = 1.0f - cl1;
        } else {
            cl[3] = 1.0f - cl1 * cl2;
        }
    }

    // high, mid, low clouds and their top and base layer indices.
    {
        float cl1 = 0.0f, cl2 = 0.0f;
        int kbt1, kbt2, kth1 = 0, kth2 = 0, idom = 1;
        if (ivflip == 0) {
            kbt1 = nlay; kbt2 = nlay;
            mb[0] = nlay; mt[0] = nlay;
            for (int k = nlay; k >= 1; --k) {
                int id = idom;
                int id1 = id + 1;
                const float pcur = PL1(k);
                const float ccur = fminf(PGC_OVCST, fmaxf(CT1(k), 0.0f));
                float pnxt, cnxt;
                if (k > 1) {
                    pnxt = PL1(k - 1);
                    cnxt = fminf(PGC_OVCST, fmaxf(CT1(k - 1), 0.0f));
                } else {
                    pnxt = -1.0f;
                    cnxt = 0.0f;
                }
                if (pcur < ptop1[id1 - 1]) {
                    id += 1;
                    id1 += 1;
                    idom = id;
                }
                if (ccur >= PGC_CLIMIT) {
                    if (kth2 == 0) kbt2 = k;
                    kth2 += 1;
                    if (iovr == 0) {
                        cl2 = cl2 + ccur - cl2 * ccur;
                    } else {
                        cl2 = fmaxf(cl2, ccur);
                    }
                    if (cnxt < PGC_CLIMIT || pnxt < ptop1[id1 - 1]) {
                        kbt1 = (int)roundf((cl1 * (float)kbt1 + cl2 * (float)kbt2) / (cl1 + cl2));
                        kth1 = (int)roundf((cl1 * (float)kth1 + cl2 * (float)kth2) / (cl1 + cl2));
                        cl1 = cl1 + cl2 - cl1 * cl2;
                        kbt2 = k - 1;
                        kth2 = 0;
                        cl2 = 0.0f;
                    }
                }
                if (pnxt < ptop1[id1 - 1]) {
                    cl[id - 1] = cl1;
                    mt[id - 1] = min(kbt1, kbt1 - kth1 + 1);
                    mb[id - 1] = kbt1;
                    cl1 = 0.0f;
                    kbt1 = k - 1;
                    kth1 = 0;
                    if (id1 <= 3) {
                        mb[id1 - 1] = kbt1;
                        mt[id1 - 1] = kbt1;
                    }
                }
            }
        } else {
            kbt1 = 1; kbt2 = 1;
            mb[0] = 1; mt[0] = 1;
            for (int k = 1; k <= nlay; ++k) {
                int id = idom;
                int id1 = id + 1;
                const float pcur = PL1(k);
                const float ccur = fminf(PGC_OVCST, fmaxf(CT1(k), 0.0f));
                float pnxt, cnxt;
                if (k < nlay) {
                    pnxt = PL1(k + 1);
                    cnxt = fminf(PGC_OVCST, fmaxf(CT1(k + 1), 0.0f));
                } else {
                    pnxt = -1.0f;
                    cnxt = 0.0f;
                }
                if (pcur < ptop1[id1 - 1]) {
                    id += 1;
                    id1 += 1;
                    idom = id;
                }
                if (ccur >= PGC_CLIMIT) {
                    if (kth2 == 0) kbt2 = k;
                    kth2 += 1;
                    if (iovr == 0) {
                        cl2 = cl2 + ccur - cl2 * ccur;
                    } else {
                        cl2 = fmaxf(cl2, ccur);
                    }
                    if (cnxt < PGC_CLIMIT || pnxt < ptop1[id1 - 1]) {
                        kbt1 = (int)roundf((cl1 * (float)kbt1 + cl2 * (float)kbt2) / (cl1 + cl2));
                        kth1 = (int)roundf((cl1 * (float)kth1 + cl2 * (float)kth2) / (cl1 + cl2));
                        cl1 = cl1 + cl2 - cl1 * cl2;
                        kbt2 = k + 1;
                        kth2 = 0;
                        cl2 = 0.0f;
                    }
                }
                if (pnxt < ptop1[id1 - 1]) {
                    cl[id - 1] = cl1;
                    mt[id - 1] = max(kbt1, kbt1 + kth1 - 1);
                    mb[id - 1] = kbt1;
                    cl1 = 0.0f;
                    kbt1 = k + 1;
                    kth1 = 0;
                    if (id1 <= 3) {
                        mb[id1 - 1] = kbt1;
                        mt[id1 - 1] = kbt1;
                    }
                }
            }
        }
    }

    for (int c = 0; c < 5; ++c) {
        clds[IDX3(i0, j0, c, im, jm)] = cl[c];
    }
    for (int c = 0; c < 3; ++c) {
        mtop[IDX3(i0, j0, c, im, jm)] = mt[c];
        mbot[IDX3(i0, j0, c, im, jm)] = mb[c];
    }

    #undef CT1
    #undef PL1
}
