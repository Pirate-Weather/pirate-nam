// !!!!!  ==============================================================  !!!!!
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
// !!!!!               lw-rrtm3 radiation package description             !!!!!
// !!!!!  ==============================================================  !!!!!
// !                                                                          !
// !   this package includes ncep's modifications of the rrtm-lw radiation    !
// !   code from aer inc.                                                     !
// !                                                                          !
// !    the lw-rrtm3 package includes these parts:                            !
// !                                                                          !
// !       'radlw_rrtm3_param.f'                                              !
// !       'radlw_rrtm3_datatb.f'                                             !
// !       'radlw_rrtm3_main.f'                                               !
// !                                                                          !
// !    the 'radlw_rrtm3_param.f' contains:                                   !
// !                                                                          !
// !       'module_radlw_parameters'  -- band parameters set up               !
// !                                                                          !
// !    the 'radlw_rrtm3_datatb.f' contains:                                  !
// !                                                                          !
// !       'module_radlw_avplank'     -- plank flux data                      !
// !       'module_radlw_ref'         -- reference temperature and pressure   !
// !       'module_radlw_cldprlw'     -- cloud property coefficients          !
// !       'module_radlw_kgbnn'       -- absorption coeffients for 16         !
// !                                     bands, where nn = 01-16              !
// !                                                                          !
// !    the 'radlw_rrtm3_main.f' contains:                                    !
// !                                                                          !
// !       'module_radlw_main'        -- main lw radiation transfer           !
// !                                                                          !
// !    in the main module 'module_radlw_main' there are only two             !
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:1344-1697 (nam.v4.2.11, 83df4e13c1).
//
// CLDPROP: LW cloud optical depth per band and layer (taucld), and for the McICA path (isubclw>0)
// the layer cloud fraction cldf that the sub-column generator (phys-rrtm-lw-mcica-subcol) consumes.
// One thread per column (i, j); the level loop runs inside the thread (CONTRACT section 3).
//
// INPUT (CONTRACT section 1, model state): clouds = RAD_CLOUDS [NF_CLDS=9, nlay, JM, IM] written by
// phys-rrtm-grrad-clouds. State layer index l0 = 0 is the MODEL TOP. Channels (lwrad_prep, lwrad
// radlw_main_nmmb.f:1066-1096): 0 cfrac, 1 cliqp (clwp), 2 reliq, 3 cicep (ciwp), 4 reice, 5 cda1,
// 6 cda2 (not read by cldprop), 7 cda3, 8 cda4. With ilwcliq == 0 the input optical depth is channel 1.
//
// RADLW INTERNAL ORDER: radlw works surface first (ivflip = 0 on NAM: lwrad_prep uses kin = nlay-1-k0).
// Internal layer k0 = 0 is the surface (Fortran k = 1). This kernel reads state layer kin = nlay-1-k0
// for internal layer k0. The outputs taucld and cldf are written in INTERNAL order (k0 = 0 surface),
// because rtrn and mcica_subcol consume them in that order. They are scratch, not state arrays, so the
// CONTRACT top-first rule does not apply to them. Each layer's result depends only on its own inputs.
//
// GATE: Fortran calls cldprop only when some layer has cldfrc(k) > eps (eps = 1.0e-6, radlw_main_nmmb.f
// :267, and the lcf1 test at 924-945). Otherwise lwrad sets cldfmc = taucld = 0 for the whole column and
// runs no cldprop statement. This kernel applies that gate per column: if no layer has cfrac > eps, taucld
// and cldf are 0 for the column.
//
// Not in this kernel: cldfmc. Fortran zeroes cldfmc(ngptlw,nlay) at radlw_main_nmmb.f:1477-1481, calls
// mcica_subcol at 1676 and writes the 1/0 cldfmc at 1683-1691, all on the isubclw > 0 path. Those lines
// belong to phys-rrtm-lw-mcica-subcol (named gap, owner: that unit). With isubclw = 0 (NAM default) rtrn
// reads taucld only and never cldfmc. Its isubclw = 0 claim is verified: rtrn (lines 2127-2484) never
// reads cldfmc.
//
// Table binding: NAM radlw_main_nmmb.f:1444 `use module_radlw_cldprlw` resolves to radlw_datatb.f:910
// (module_radlw_cldprlw). radlw_datatb_nmmb.f:733 defines module_radlw_cldprlw_nmmb, which NAM never uses.
// The tables are absliq1 [16, 58], absice1 [5, 2], absice2 [16, 43], absice3 [16, 46], ipat [16]. absice0
// is in the module but cldprop never reads it.
//
// Unsupported ilwcliq and ilwcice values are rejected by the wrapper (Fortran leaves tauliq / tauice
// undefined for them), so this kernel never sees them.

#include "nmmb_common.cuh"

#define NBANDS_LW 16
#define NG_TAB_LIQ 58
#define NG_TAB_ICE2 43
#define NG_TAB_ICE3 46
#define LW_EPS 1.0e-6f          /* radlw_main_nmmb.f:267 eps */

extern "C" __global__ void phys_rrtm_lw_cldprop(
    const float* __restrict__ clouds,   /* [NF_CLDS=9, nlay, JM, IM] RAD_CLOUDS, state order (model top first) */
    const float* __restrict__ absliq1,  /* [16, 58] */
    const float* __restrict__ absice1,  /* [5, 2] */
    const float* __restrict__ absice2,  /* [16, 43] */
    const float* __restrict__ absice3,  /* [16, 46] */
    const int*   __restrict__ ipat,     /* [16] */
    float* __restrict__ taucld,         /* [16, nlay, JM, IM] internal order (k0 = 0 surface) */
    float* __restrict__ cldf,           /* [nlay, JM, IM] internal order; written only when isubclw > 0 */
    int nlay, int ilwcliq, int ilwcice, int isubclw,
    int IM, int JM)
{
    COLUMN_IJ(IM, JM)
    const float f_zero = 0.0f;
    const float cldmin = 0.0f;              /* NMMB cldmin = 1.0e-80, which underflows to 0 in single (kind_phys=4) */
    const float absrain = 0.33e-3f;         /* radlw_datatb: absrain (ncar coeff) */
    const float abssnow0 = 1.5f;            /* radlw_datatb: abssnow0 (fu coeff) */

    const size_t plane = (size_t)JM * IM;
    const size_t colbase = (size_t)j0 * IM + i0;
    /* internal layer k0 (0 = surface) -> state layer nlay-1-k0 (model top first), channel n */
#define CL(n, k0) clouds[((size_t)(n) * nlay + (size_t)(nlay - 1 - (k0))) * plane + colbase]
#define TAU(ib, k0) taucld[((size_t)(ib) * nlay + (size_t)(k0)) * plane + colbase]

    for (int k0 = 0; k0 < nlay; ++k0)
        for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0)
            TAU(ib0, k0) = f_zero;

    /* lcf1 gate (radlw_main_nmmb.f:924-945): cldprop runs only if some layer has cldfrc > eps */
    int lcf1 = 0;
    for (int k0 = 0; k0 < nlay && !lcf1; ++k0)
        if (CL(0, k0) > LW_EPS) lcf1 = 1;
    if (!lcf1) {
        if (isubclw > 0)
            for (int k0 = 0; k0 < nlay; ++k0) cldf[(size_t)k0 * plane + colbase] = f_zero;
        return;
    }

    if (ilwcliq > 0) {
        for (int k0 = 0; k0 < nlay; ++k0) {
            const float cf = CL(0, k0);                                   /* Fortran cfrac(k) */
            if (!(cf > cldmin)) continue;

            float tauliq[NBANDS_LW], tauice[NBANDS_LW], tausnow[NBANDS_LW];
            float tauran, tausnw, csno = f_zero, radsno = f_zero;

            if (ilwcice < 4) {
                tauran = absrain * CL(5, k0);                              /* ncar formula, cda1 */
                if (CL(7, k0) > f_zero && CL(8, k0) > 10.0f) {             /* fu formula, cda3 / cda4 */
                    tausnw = abssnow0 * 1.05756f * CL(7, k0) / CL(8, k0);
                } else {
                    tausnw = f_zero;
                }
            } else {
                csno = fmaxf(f_zero, CL(7, k0));
                radsno = fmaxf(25.0f, fminf(130.0f, CL(8, k0)));
                tausnw = 0.0f;
                tauran = 0.0f;
            }

            const float cldliq = CL(1, k0);                                /* clwp */
            const float cldice = CL(3, k0);                                /* ciwp */
            const float refliq = CL(2, k0);                                /* relw */
            float refice = CL(4, k0);                                      /* reiw */

            /* water droplet absorption */
            for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) tauliq[ib0] = f_zero;
            if (cldliq > f_zero && ilwcliq == 1) {
                const float factor = refliq - 1.5f;
                const int index = max(1, min(57, (int)factor));
                const float fint = factor - (float)index;
                for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) {
                    const float a1 = absliq1[ib0 * NG_TAB_LIQ + (index - 1)];
                    const float a2 = absliq1[ib0 * NG_TAB_LIQ + index];
                    tauliq[ib0] = fmaxf(f_zero, cldliq * (a1 + fint * (a2 - a1)));
                }
            }

            /* ice absorption */
            for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) tauice[ib0] = f_zero;
            if (cldice > f_zero) {
                if (ilwcice == 1) {
                    refice = fminf(130.0f, fmaxf(13.0f, refice));
                    for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) {
                        const int ia0 = ipat[ib0] - 1;           /* ebert & curry band index */
                        tauice[ib0] = fmaxf(f_zero,
                            cldice * (absice1[ia0 * 2 + 0] + absice1[ia0 * 2 + 1] / refice));
                    }
                } else if (ilwcice == 2) {
                    const float factor = (refice - 2.0f) / 3.0f;
                    const int index = max(1, min(42, (int)factor));
                    const float fint = factor - (float)index;
                    for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) {
                        const float a1 = absice2[ib0 * NG_TAB_ICE2 + (index - 1)];
                        const float a2 = absice2[ib0 * NG_TAB_ICE2 + index];
                        tauice[ib0] = fmaxf(f_zero, cldice * (a1 + fint * (a2 - a1)));
                    }
                } else if (ilwcice == 3) {
                    const float dgeice = fmaxf(5.0f, 1.0315f * refice);
                    const float factor = (dgeice - 2.0f) / 3.0f;
                    const int index = max(1, min(45, (int)factor));
                    const float fint = factor - (float)index;
                    for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) {
                        const float a1 = absice3[ib0 * NG_TAB_ICE3 + (index - 1)];
                        const float a2 = absice3[ib0 * NG_TAB_ICE3 + index];
                        tauice[ib0] = fmaxf(f_zero, cldice * (a1 + fint * (a2 - a1)));
                    }
                } else if (ilwcice == 4) {
                    const float factor = (refice - 2.0f) / 3.0f;
                    const int index = max(1, min(42, (int)factor));
                    const float fint = factor - (float)index;
                    for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) {
                        const float a1 = absice2[ib0 * NG_TAB_ICE2 + (index - 1)];
                        const float a2 = absice2[ib0 * NG_TAB_ICE2 + index];
                        tauice[ib0] = fmaxf(f_zero, cldice * (a1 + fint * (a2 - a1)));
                        tauice[ib0] = fminf(35.0f, tauice[ib0]);
                    }
                }
            }

            /* snow absorption (ilwcice == 4) */
            for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) tausnow[ib0] = f_zero;
            if (ilwcice == 4 && csno > f_zero) {
                const float factor = (radsno - 2.0f) / 3.0f;
                const int index = max(1, min(42, (int)factor));
                const float fint = factor - (float)index;
                for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) {
                    const float a1 = absice2[ib0 * NG_TAB_ICE2 + (index - 1)];
                    const float a2 = absice2[ib0 * NG_TAB_ICE2 + index];
                    tausnow[ib0] = fmaxf(f_zero, csno * (a1 + fint * (a2 - a1)));
                    tausnow[ib0] = fminf(35.0f, tausnow[ib0]);
                }
            }

            if (ilwcice >= 4) {
                for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0) {
                    tauliq[ib0] = fminf(35.0f, tauliq[ib0]);
                    const float dummy_tau = fminf(tauliq[ib0] + tauice[ib0] + tausnow[ib0], 35.0f);
                    TAU(ib0, k0) = fmaxf(1.0e-3f, dummy_tau);
                }
            } else {
                for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0)
                    TAU(ib0, k0) = tauice[ib0] + tauliq[ib0] + tauran + tausnw;
            }
        }
    } else {
        /* ilwcliq == 0: input optical depth (cda1, channel 1) where cfrac > cldmin */
        for (int k0 = 0; k0 < nlay; ++k0) {
            if (CL(0, k0) > cldmin) {
                const float v = CL(1, k0);
                for (int ib0 = 0; ib0 < NBANDS_LW; ++ib0)
                    TAU(ib0, k0) = v;
            }
        }
    }

    if (isubclw > 0) {
        for (int k0 = 0; k0 < nlay; ++k0) {
            const float cf = CL(0, k0);
            cldf[(size_t)k0 * plane + colbase] = (cf < cldmin) ? f_zero : cf;
        }
    }
#undef CL
#undef TAU
}
