// init_gfs_vinterp.cu -- GFS pressure-level to NMMB hybrid-level vertical interpolation
// (unit init-gfs-vinterp, CONTRACT.md sections 3 and 6).
//
// NOAA NAM / NMMB nemsinterp (public domain, US government work); NOAA author headers are kept in the
// Fortran source. Source tree: $NMMB_INIT = sorc/nam_nps.fd/sorc/NMMB_init/NPS/nemsinterp/src.
// Ported to CUDA for WOOF-NAM from module_vinterp_routines.F90:1363-1955 (compute_nmm_surfacep),
// 1956-2038 (compute_3d_pressure), 2039-2251 (interp_press2press_lin), 2433-2524 (wind_adjust),
// 2525-2666 (interp_press2press_log), 2667-2802 (rh_to_mxrat), with the caller at 423-1010
// (nam.v4.2.11, 83df4e13c1).
//
// Kernels (one launch each, all device pointers, no host sync):
//   init_gfs_vinterp_surfp   per column  compute_nmm_surfacep, non-spectral branch -> PSFC_OUT
//   init_gfs_vinterp_pd      per column  compute_3d_pressure -> PD, P3D (and PINT when requested)
//   init_gfs_vinterp_vpoints per column  V-point PSFC_OUTV, PDVP and P3DV_IN (B-grid averaging)
//   init_gfs_vinterp_lin     per column  interp_press2press_lin (T, U, V, FRIMEF)
//   init_gfs_vinterp_log     per column  interp_press2press_log (Q, SNMR, CLWMR, RWMR, CICE)
//   init_gfs_vinterp_wind    per column  wind_adjust (hyb_coor) and eastern-boundary zeroing (B grid)
//   init_gfs_vinterp_rhq     per cell    rh_to_mxrat (liquid branch) -> specific humidity
//   init_gfs_vinterp_cwm_acc per cell  condensate thresholds; CWM = SNMR+CLWMR+RWMR+CICE (one species at a time)
//
// Index convention: Fortran (i,j,l) with 1-based i,j,l maps to zero-based (i0,j0,l0). Vertical level
// index 0 of a GFS input array is the lowest (surface) metgrid level; output level 0 is model top.
// Inputs are [nlev, jm, im] (metgrid order); model outputs are [lm, jm, im] and [lm+1, jm, im] (PINT).
#include "nmmb_common.cuh"

// 1-based level accessor for the Fortran-ordered routines.
__device__ __forceinline__ float vi_lv(const float* __restrict__ a, int i0, int j0, int l1,
                                       int im, int jm)
{
    return a[IDX3(i0, j0, l1 - 1, im, jm)];
}

// ---------------------------------------------------------------------------
// compute_nmm_surfacep (non-spectral). One thread per column (i0, j0).
// The Fortran II_loop smoothing of PSFC_IN starts with CYCLE II_loop, so its body never runs; the
// smoothing is dead code in the source and is not ported. Cells where the Fortran CYCLE I_loop skips
// the first pass keep DEFINED_PSFC = false (exactly as the source).
// psfc_out is left unchanged in a cell where neither half of the estimate is defined.
// ---------------------------------------------------------------------------
extern "C" __global__ void init_gfs_vinterp_surfp(
    const float* __restrict__ terrain,   // HGT_M [jm, im]  (TERRAIN_HGT_T)
    const float* __restrict__ pres,      // [nlev, jm, im]  (PRESS3D_IN)
    const float* __restrict__ ght,       // [nlev, jm, im]  (Z3D_IN)
    const float* __restrict__ tt,        // [nlev, jm, im]  (T3D_IN)
    float* __restrict__ psfc_out,        // [jm, im]        (PSFC_OUT, in/out)
    int im, int jm, int nlev)
{
    COLUMN_IJ(im, jm)
#define SF_P(l) vi_lv(pres, i0, j0, (l), im, jm)
#define SF_Z(l) vi_lv(ght,  i0, j0, (l), im, jm)
#define SF_T(l) vi_lv(tt,   i0, j0, (l), im, jm)
    const float terr = terrain[IDX2(i0, j0, im)];

    float psfc_in, topo_in;
    if (SF_P(1) != 200100.0f) { psfc_in = SF_P(1); topo_in = SF_Z(1); }
    else                      { psfc_in = SF_P(2); topo_in = SF_Z(2); }

    bool defined_psfc = false, defined_psfcb = false;
    bool skip = false;                       // Fortran CYCLE I_loop
    float dum2d = -9.0f, dum2db = -9.0f;
    float bot_press = psfc_in, bot_hgt = topo_in;

    // short_loop: first input layer that brackets the terrain height
    for (int L = 1; L <= nlev - 1; ++L) {
        if (SF_P(L) > psfc_in && SF_Z(L) < terr && SF_Z(L + 1) > terr) {
            bot_press = SF_P(L);
            bot_hgt = SF_Z(L);
            break;
        }
    }

    // non-spectral consistency tests on the lowest levels
    if (SF_P(1) != 200100.0f && (psfc_in > SF_P(2) || topo_in < SF_Z(2))) {
        if (SF_Z(2) != topo_in) {
            const float dpdz = (SF_P(2) - psfc_in) / (SF_Z(2) - topo_in);
            const float rhs = -9.81f * ((SF_P(2) + psfc_in) / 2.0f) / (287.04f * SF_T(2));
            if (fabsf(SF_P(2) - psfc_in) > 290.0f) {
                if (dpdz < 1.05f * rhs || dpdz > 0.95f * rhs) skip = true;
            }
        } else {
            if (SF_P(2) != psfc_in) skip = true;
        }
        if (!skip && fabsf(SF_P(2) - psfc_in) > 290.0f) {
            if (SF_P(2) < psfc_in && SF_Z(2) < topo_in) skip = true;
            else if (SF_P(2) > psfc_in && SF_Z(2) > topo_in) skip = true;
        }
    }
    for (int L = 3; L <= 6 && !skip; ++L) {
        if (SF_P(1) != 200100.0f &&
            ((psfc_in - SF_P(L)) < 400.0f || topo_in < SF_Z(L))) {
            if (SF_Z(L) != topo_in) {
                const float dpdz = (SF_P(L) - psfc_in) / (SF_Z(L) - topo_in);
                const float rhs = -9.81f * ((SF_P(L) + psfc_in) / 2.0f) / (287.04f * SF_T(L));
                if (fabsf(SF_P(L) - psfc_in) > 290.0f) {
                    if (dpdz < 1.05f * rhs || dpdz > 0.95f * rhs) skip = true;
                }
            } else {
                if (SF_P(2) != psfc_in) skip = true;
            }
        }
        if (!skip && fabsf(SF_P(L) - psfc_in) > 290.0f) {
            if (SF_P(L) < psfc_in && SF_Z(L) < topo_in) skip = true;
            else if (SF_P(L) > psfc_in && SF_Z(L) > topo_in) skip = true;
        }
    }

    if (!skip) {
        if (terr == bot_hgt) {
            dum2d = bot_press;
            defined_psfc = true;
        } else if (terr < bot_hgt) {
            float dlnpdz = 0.0f;   // the Fortran FIND_THICK loop leaves this undefined if it finds nothing
            if (bot_press - SF_P(2) > 500.0f) {
                dlnpdz = (logf(bot_press) - logf(SF_P(2))) / (bot_hgt - SF_Z(2));
            } else if (fabsf(bot_press - SF_P(3)) > 290.0f) {
                dlnpdz = (logf(bot_press) - logf(SF_P(3))) / (bot_hgt - SF_Z(3));
            } else {
                for (int LL = 4; LL <= 7; ++LL) {
                    if (fabsf(bot_press - SF_P(LL)) > 290.0f) {
                        dlnpdz = (logf(bot_press) - logf(SF_P(LL))) / (bot_hgt - SF_Z(LL));
                        break;
                    }
                }
            }
            dum2d = expf(logf(bot_press) + dlnpdz * (terr - bot_hgt));
            defined_psfc = true;
        } else {
            // target level bounded by input levels
            for (int L = 2; L <= nlev - 1; ++L) {
                if (terr > SF_Z(L) && terr < SF_Z(L + 1)) {
                    const float dlnpdz = (logf(SF_P(L)) - logf(SF_P(L + 1))) / (SF_Z(L) - SF_Z(L + 1));
                    dum2d = logf(SF_P(L)) + dlnpdz * (terr - SF_Z(L));
                    dum2d = expf(dum2d);
                    defined_psfc = true;
                    break;
                }
            }
            if (dum2d == -9.0f && bot_hgt < terr && terr < SF_Z(2)) {
                const float dlnpdz = (logf(psfc_in) - logf(SF_P(2))) / (topo_in - SF_Z(2));
                dum2d = logf(psfc_in) + dlnpdz * (terr - topo_in);
                dum2d = expf(dum2d);
                defined_psfc = true;
            }
            if (dum2d == -9.0f) {
                for (int L = 1; L <= nlev - 1; ++L) {
                    if (terr == SF_Z(L)) {
                        dum2d = SF_P(L);
                        defined_psfc = true;
                        break;
                    }
                }
                if (terr == topo_in) {
                    dum2d = psfc_in;
                    defined_psfc = true;
                }
            }
        }
        // Fortran: if (.not. DEFINED_PSFC) DEFINED_PSFC = .TRUE.  (every non-skipped cell)
        defined_psfc = true;
    }

    // second estimate: terrain-relative extrapolation from the lowest input levels (all cells)
    if (terr < SF_Z(2)) {
        float dlnpdz;
        if (fabsf(SF_P(2) - SF_P(3)) > 290.0f) {
            dlnpdz = (logf(SF_P(2)) - logf(SF_P(3))) / (SF_Z(2) - SF_Z(3));
        } else {
            dlnpdz = (logf(SF_P(2)) - logf(SF_P(4))) / (SF_Z(2) - SF_Z(4));
        }
        dum2db = expf(logf(SF_P(2)) + dlnpdz * (terr - SF_Z(2)));
        defined_psfcb = true;
    } else if (terr > SF_Z(2)) {
        for (int L = 2; L <= nlev - 1; ++L) {
            if (terr > SF_Z(L) && terr < SF_Z(L + 1)) {
                const float dlnpdz = (logf(SF_P(L)) - logf(SF_P(L + 1))) / (SF_Z(L) - SF_Z(L + 1));
                dum2db = logf(SF_P(L)) + dlnpdz * (terr - SF_Z(L));
                dum2db = expf(dum2db);
                defined_psfcb = true;
                break;
            }
        }
    } else if (terr == SF_Z(2)) {
        dum2db = SF_P(2);
        defined_psfcb = true;
    }
    if (dum2db == -9.0f) {
        for (int L = 1; L <= nlev - 1; ++L) {
            if (terr == SF_Z(L)) {
                dum2db = SF_P(L);
                defined_psfcb = true;
                break;
            }
        }
    }

    // combine: mean when both halves are defined (the >400 Pa disagreement is a print only in the source)
    const size_t n2 = IDX2(i0, j0, im);
    if (defined_psfcb && defined_psfc) {
        psfc_out[n2] = 0.5f * (dum2d + dum2db);
    } else if (defined_psfc) {
        psfc_out[n2] = dum2d;
    } else if (defined_psfcb) {
        psfc_out[n2] = dum2db;
    }
#undef SF_P
#undef SF_Z
#undef SF_T
}

// ---------------------------------------------------------------------------
// compute_3d_pressure: PD, mid-level pressure P3D [lm] and (optionally) interface pressure PINT [lm+1].
// vcoord == 0 uses PD = PSFC - PDTOP - PT, otherwise PD = PSFC - PT (Fortran lines 1956-2038).
// The Fortran STOP for out-of-range pressures is a diagnostic and is not ported.
// ---------------------------------------------------------------------------
extern "C" __global__ void init_gfs_vinterp_pd(
    const float* __restrict__ psfc,      // [jm, im]
    const float* __restrict__ sgml1,     // [lm]
    const float* __restrict__ sgml2,     // [lm]
    const float* __restrict__ sg1,       // [lm+1]
    const float* __restrict__ sg2,       // [lm+1]
    float* __restrict__ pd_out,          // [jm, im]
    float* __restrict__ p3d_out,         // [lm, jm, im]
    float* __restrict__ pint_out,        // [lm+1, jm, im] (written when do_pint)
    int im, int jm, int lm, float pdtop, float pt, int vcoord, int do_pint)
{
    COLUMN_IJ(im, jm)
    const size_t n2 = IDX2(i0, j0, im);
    const float ps = psfc[n2];
    const float pd = (vcoord == 0) ? (ps - pdtop - pt) : (ps - pt);
    pd_out[n2] = pd;
    for (int k = 0; k < lm; ++k) {
        p3d_out[IDX3(i0, j0, k, im, jm)] = pd * sgml2[k] + pdtop * sgml1[k] + pt;
    }
    if (do_pint) {
        for (int k = 0; k <= lm; ++k) {
            pint_out[IDX3(i0, j0, k, im, jm)] = pd * sg2[k] + pdtop * sg1[k] + pt;
        }
    }
}

// ---------------------------------------------------------------------------
// V-point (B-grid) averaging of surface pressure, PD and the input pressure levels. V(i,j) sits NE of
// H(i,j); the three-branch rule of the Fortran AGRID block (caller lines 428-470):
//   J < JDE-1 and I < IDE-1 : mean of the four H neighbours
//   J == JDE-1 and I != IDE-1 : mean of H(i,j) and H(i+1,j)
//   I == IDE-1 : copy of H(i,j)
// With IDE = IM+1 and JDE = JM+1 (1-based) the tests become i0 < IM-1, j0 < JM-1, and so on.
// ---------------------------------------------------------------------------
extern "C" __global__ void init_gfs_vinterp_vpoints(
    const float* __restrict__ psfc,      // [jm, im]   PSFC_OUT (H)
    const float* __restrict__ pd,        // [jm, im]   PD (H)
    const float* __restrict__ pres,      // [nlev, jm, im]  PRES (H)
    float* __restrict__ psfc_outv,       // [jm, im]   PSFC_OUTV
    float* __restrict__ pdvp,            // [jm, im]   PDVP
    float* __restrict__ p3dv_in,         // [nlev, jm, im]  P3DV_IN
    int im, int jm, int nlev)
{
    COLUMN_IJ(im, jm)
    const bool avg4 = (j0 < jm - 1) && (i0 < im - 1);
    const bool avg2 = (j0 == jm - 1) && (i0 != im - 1);
    const size_t n2 = IDX2(i0, j0, im);
    const size_t ie = IDX2(i0 + 1, j0, im);
    const size_t jn = IDX2(i0, j0 + 1, im);
    const size_t ijn = IDX2(i0 + 1, j0 + 1, im);

    if (avg4) {
        psfc_outv[n2] = 0.25f * (psfc[n2] + psfc[ie] + psfc[jn] + psfc[ijn]);
        pdvp[n2] = 0.25f * (pd[n2] + pd[ie] + pd[jn] + pd[ijn]);
    } else if (avg2) {
        psfc_outv[n2] = 0.5f * (psfc[n2] + psfc[ie]);
        pdvp[n2] = 0.5f * (pd[n2] + pd[ie]);
    } else {
        psfc_outv[n2] = psfc[n2];
        pdvp[n2] = pd[n2];
    }
    for (int k = 0; k < nlev; ++k) {
        const size_t n = IDX3(i0, j0, k, im, jm);
        const size_t ne = IDX3(i0 + 1, j0, k, im, jm);
        const size_t nj = IDX3(i0, j0 + 1, k, im, jm);
        const size_t nij = IDX3(i0 + 1, j0 + 1, k, im, jm);
        if (avg4) {
            p3dv_in[n] = 0.25f * (pres[n] + pres[ne] + pres[nj] + pres[nij]);
        } else if (avg2) {
            p3dv_in[n] = 0.5f * (pres[n] + pres[ne]);
        } else {
            p3dv_in[n] = pres[n];
        }
    }
}

// ---------------------------------------------------------------------------
// interp_press2press_lin: linear in pressure, log-pressure interpolation of one column.
//   press_in/data_in [nlev] (index 0 = surface), press_out/data_out [lm] (index 0 = top).
//   extrapolate, ignore_lowest, tfield are the Fortran logical flags (0/1).
//   skip_interior = (.not. first): only the boundary strips of width lnsh are computed; the interior
//   keeps the -99999.9 fill that the Fortran writes first.
// Source quirk kept: the column smoother needs MAX_SMOOTH, which the Fortran never initialises. It is
// reached only when K=1 is more than 3000 Pa below the surface, which cannot occur with the 200 Pa
// model top. Here an unset MAX_SMOOTH skips the smoothing.
// ---------------------------------------------------------------------------
extern "C" __global__ void init_gfs_vinterp_lin(
    const float* __restrict__ press_in,  // [nlev, jm, im]
    const float* __restrict__ press_out, // [lm, jm, im]
    const float* __restrict__ data_in,   // [nlev, jm, im]
    float* __restrict__ data_out,        // [lm, jm, im]
    int im, int jm, int nlev, int lm,
    int extrapolate, int ignore_lowest, int tfield,
    int skip_interior, int lnsh, int ide, int jde)
{
    COLUMN_IJ(im, jm)
    const int I = i0 + 1, J = j0 + 1;
#define LIN_PIN(k) press_in[IDX3(i0, j0, (k), im, jm)]
#define LIN_DIN(k) data_in[IDX3(i0, j0, (k), im, jm)]
#define LIN_POUT(k) press_out[IDX3(i0, j0, (k), im, jm)]
#define LIN_DOUT(k) data_out[IDX3(i0, j0, (k), im, jm)]

    for (int k = 0; k < lm; ++k) LIN_DOUT(k) = -99999.9f;
    if (skip_interior && J >= lnsh + 2 && J <= jde - 2 - lnsh &&
        I >= lnsh + 2 && I <= ide - 2 - lnsh) {
        return;
    }

    const int lmin = ignore_lowest ? 1 : 0;   // zero-based LMIN
    const int gen = nlev - 1;                 // zero-based index of the top-most input level
    bool col_smooth = false;
    int max_smooth = -1;                      // 1-based K, unset until found

    for (int k = 0; k < lm; ++k) {
        const float desired = LIN_POUT(k);
        if (col_smooth && tfield && k >= 1) {
            if (desired < LIN_PIN(lmin) && LIN_POUT(k - 1) > LIN_PIN(lmin)) max_smooth = k + 1;
        }
        if (desired > LIN_PIN(lmin)) {
            if (tfield && k == 0 && (desired - LIN_PIN(lmin)) > 3000.0f) col_smooth = true;
            if ((desired - LIN_PIN(lmin)) < 50.0f) {
                LIN_DOUT(k) = LIN_DIN(lmin);
            } else if (extrapolate) {
                float tiso = 0.0f;
                if (tfield) tiso = 0.5f * (LIN_DIN(0) + LIN_DIN(1));
                float dvaldlnp, dlnp;
                if ((LIN_PIN(lmin) - LIN_PIN(lmin + 1)) > 500.0f) {
                    dlnp = logf(LIN_PIN(lmin)) - logf(LIN_PIN(lmin + 1));
                    dvaldlnp = (LIN_DIN(lmin) - LIN_DIN(lmin + 1)) / dlnp;
                } else {
                    dlnp = logf(LIN_PIN(lmin)) - logf(LIN_PIN(lmin + 5));
                    dvaldlnp = (LIN_DIN(lmin) - LIN_DIN(lmin + 5)) / dlnp;
                }
                float v = LIN_DIN(lmin) + dvaldlnp * (logf(desired) - logf(LIN_PIN(lmin)));
                if (tfield && v < tiso - 0.2f) {
                    dvaldlnp = fmaxf(dvaldlnp, -1.0f / logf(LIN_PIN(lmin) / (LIN_PIN(lmin) - 1000.0f)));
                    v = LIN_DIN(lmin) + dvaldlnp * (logf(desired) - logf(LIN_PIN(lmin)));
                } else if (tfield && v > tiso + 0.2f) {
                    dvaldlnp = fminf(dvaldlnp, 0.8f / logf(LIN_PIN(lmin) / (LIN_PIN(lmin) - 1000.0f)));
                    v = LIN_DIN(lmin) + dvaldlnp * (logf(desired) - logf(LIN_PIN(lmin)));
                }
                LIN_DOUT(k) = v;
            } else {
                LIN_DOUT(k) = LIN_DIN(lmin);
            }
        } else if (desired < LIN_PIN(gen)) {
            if ((LIN_PIN(gen) - desired) < 10.0f) {
                LIN_DOUT(k) = LIN_DIN(gen);
            } else if (extrapolate) {
                float dlnp, dvaldlnp;
                if ((LIN_PIN(gen - 1) - LIN_PIN(gen)) > 50.0f) {
                    dlnp = logf(LIN_PIN(gen)) - logf(LIN_PIN(gen - 1));
                    dvaldlnp = (LIN_DIN(gen) - LIN_DIN(gen - 1)) / dlnp;
                } else {
                    dlnp = logf(LIN_PIN(gen)) - logf(LIN_PIN(gen - 2));
                    dvaldlnp = (LIN_DIN(gen) - LIN_DIN(gen - 2)) / dlnp;
                }
                LIN_DOUT(k) = LIN_DIN(gen) + dvaldlnp * (logf(desired) - logf(LIN_PIN(gen)));
            } else {
                LIN_DOUT(k) = LIN_DIN(gen);
            }
        } else {
            for (int kk = lmin; kk <= gen - 1; ++kk) {
                if (desired == LIN_PIN(kk)) {
                    LIN_DOUT(k) = LIN_DIN(kk);
                    break;
                } else if (desired < LIN_PIN(kk) && desired > LIN_PIN(kk + 1)) {
                    const float dlnp = logf(LIN_PIN(kk)) - logf(LIN_PIN(kk + 1));
                    const float dvaldlnp = (LIN_DIN(kk) - LIN_DIN(kk + 1)) / dlnp;
                    LIN_DOUT(k) = LIN_DIN(kk + 1) + dvaldlnp * (logf(desired) - logf(LIN_PIN(kk + 1)));
                    break;
                }
            }
        }
    }

    if (col_smooth && max_smooth > 0) {
        for (int K = (max_smooth - 4 > 1 ? max_smooth - 4 : 1); K <= max_smooth + 4; ++K) {
            if (K >= lm) break;               // the source would read data_out(K+1) past the column top
            LIN_DOUT(K - 1) = 0.5f * (LIN_DOUT(K - 1) + LIN_DOUT(K));
        }
    }
#undef LIN_PIN
#undef LIN_DIN
#undef LIN_POUT
#undef LIN_DOUT
}

// ---------------------------------------------------------------------------
// interp_press2press_log: as lin, but interpolates ln(data) against ln(p). data_in is clamped to 1e-13
// (in place, as the source does). Skip rule uses lnsh without the +1 offset (source lines 2525-2666).
// ---------------------------------------------------------------------------
extern "C" __global__ void init_gfs_vinterp_log(
    const float* __restrict__ press_in,  // [nlev, jm, im]
    const float* __restrict__ press_out, // [lm, jm, im]
    float* __restrict__ data_in,         // [nlev, jm, im]  INOUT (clamped to >= 1e-13)
    float* __restrict__ data_out,        // [lm, jm, im]
    int im, int jm, int nlev, int lm,
    int extrapolate, int ignore_lowest,
    int skip_interior, int lnsh, int ide, int jde)
{
    COLUMN_IJ(im, jm)
    const int I = i0 + 1, J = j0 + 1;
#define LOG_PIN(k) press_in[IDX3(i0, j0, (k), im, jm)]
#define LOG_DIN(k) data_in[IDX3(i0, j0, (k), im, jm)]
#define LOG_POUT(k) press_out[IDX3(i0, j0, (k), im, jm)]
#define LOG_DOUT(k) data_out[IDX3(i0, j0, (k), im, jm)]

    for (int k = 0; k < lm; ++k) LOG_DOUT(k) = -99999.9f;
    for (int k = 0; k < nlev; ++k) LOG_DIN(k) = fmaxf(LOG_DIN(k), 1.0e-13f);
    if (skip_interior && J >= 1 + lnsh && J <= jde - 1 - lnsh &&
        I >= 1 + lnsh && I <= ide - 1 - lnsh) {
        return;
    }

    const int lmin = ignore_lowest ? 1 : 0;
    const int gen = nlev - 1;
    for (int k = 0; k < lm; ++k) {
        const float desired = LOG_POUT(k);
        if (desired > LOG_PIN(lmin)) {
            if ((desired - LOG_PIN(lmin)) < 10.0f) {
                LOG_DOUT(k) = LOG_DIN(lmin);
            } else if (extrapolate) {
                float dlnp, dlnvaldlnp;
                if ((LOG_PIN(lmin) - LOG_PIN(lmin + 1)) > 100.0f) {
                    dlnp = logf(LOG_PIN(lmin)) - logf(LOG_PIN(lmin + 1));
                    dlnvaldlnp = (logf(LOG_DIN(lmin)) - logf(LOG_DIN(lmin + 1))) / dlnp;
                } else {
                    dlnp = logf(LOG_PIN(lmin)) - logf(LOG_PIN(lmin + 2));
                    dlnvaldlnp = (logf(LOG_DIN(lmin)) - logf(LOG_DIN(lmin + 2))) / dlnp;
                }
                LOG_DOUT(k) = expf(logf(LOG_DIN(lmin)) +
                                   dlnvaldlnp * (logf(desired) - logf(LOG_PIN(lmin))));
            } else {
                LOG_DOUT(k) = LOG_DIN(lmin);
            }
        } else if (desired < LOG_PIN(gen)) {
            if ((LOG_PIN(gen) - desired) < 10.0f) {
                LOG_DOUT(k) = LOG_DIN(gen);
            } else if (extrapolate) {
                float dlnp, dlnvaldlnp;
                if ((LOG_PIN(gen - 1) - LOG_PIN(gen)) > 50.0f) {
                    dlnp = logf(LOG_PIN(gen)) - logf(LOG_PIN(gen - 1));
                    dlnvaldlnp = (logf(LOG_DIN(gen)) - logf(LOG_DIN(gen - 1))) / dlnp;
                } else {
                    dlnp = logf(LOG_PIN(gen)) - logf(LOG_PIN(gen - 2));
                    dlnvaldlnp = (logf(LOG_DIN(gen)) - logf(LOG_DIN(gen - 2))) / dlnp;
                }
                LOG_DOUT(k) = expf(logf(LOG_DIN(gen)) +
                                   dlnvaldlnp * (logf(desired) - logf(LOG_PIN(gen))));
            } else {
                LOG_DOUT(k) = LOG_DIN(gen);
            }
        } else {
            for (int kk = lmin; kk <= gen - 1; ++kk) {
                if (desired == LOG_PIN(kk)) {
                    LOG_DOUT(k) = LOG_DIN(kk);
                    break;
                } else if (desired < LOG_PIN(kk) && desired > LOG_PIN(kk + 1)) {
                    const float dlnp = logf(LOG_PIN(kk)) - logf(LOG_PIN(kk + 1));
                    const float dlnvaldlnp = (logf(LOG_DIN(kk)) - logf(LOG_DIN(kk + 1))) / dlnp;
                    LOG_DOUT(k) = expf(logf(LOG_DIN(kk + 1)) +
                                       dlnvaldlnp * (logf(desired) - logf(LOG_PIN(kk + 1))));
                    break;
                }
            }
        }
    }
#undef LOG_PIN
#undef LOG_DIN
#undef LOG_POUT
#undef LOG_DOUT
}

// ---------------------------------------------------------------------------
// wind_adjust (hyb_coor branch, depth_replace = 5000 Pa) and the B-grid eastern-boundary zeroing
// (source lines 2433-2524; caller lines 760-800 and 880-900). One thread per column.
// INLOOP and OUTLOOP build thicknesses above the lowest input level (press_in(2)) and above the top
// output level (press_out(1)). The source reads p1d_in(MAXLIN+1), one entry past the last thickness it
// wrote, which is a stale value there; here that entry is computed exactly, or treated as +inf past the
// last input level.
// ---------------------------------------------------------------------------
extern "C" __global__ void init_gfs_vinterp_wind(
    const float* __restrict__ press_in,  // [nlev, jm, im]   P3DV_IN
    const float* __restrict__ press_out, // [lm, jm, im]     P3DV_OUT
    const float* __restrict__ u_in,      // [nlev, jm, im]
    const float* __restrict__ v_in,      // [nlev, jm, im]
    float* __restrict__ u_out,           // [lm, jm, im]     INOUT
    float* __restrict__ v_out,           // [lm, jm, im]     INOUT
    int im, int jm, int nlev, int lm, float depth_replace, int do_adjust, int zero_east)
{
    COLUMN_IJ(im, jm)
#define WI_PI(k) press_in[IDX3(i0, j0, (k), im, jm)]
#define WI_PO(k) press_out[IDX3(i0, j0, (k), im, jm)]
#define WI_UI(k) u_in[IDX3(i0, j0, (k), im, jm)]
#define WI_VI(k) v_in[IDX3(i0, j0, (k), im, jm)]
#define WI_UO(k) u_out[IDX3(i0, j0, (k), im, jm)]
#define WI_VO(k) v_out[IDX3(i0, j0, (k), im, jm)]

    const float p2 = WI_PI(1);
    // Fortran wind_adjust is called only when hyb_coor (caller line 767). Inside it the whole
    // adjustment sits under IF ((press_in(2)-press_out(1)) .gt. 200) (source 2463-2515); outside that
    // block U/V keep the interp_press2press_lin result (no top copy, no level loops).
    if (do_adjust && p2 - WI_PO(0) > 200.0f) {
        WI_UO(0) = WI_UI(1);
        WI_VO(0) = WI_VI(1);

        // p1d_in(L), 1-based L = 2..nlev: thickness above the lowest input level
        int maxlin = 1;
        for (int L = 2; L <= nlev; ++L) {
            if (p2 - WI_PI(L - 1) < depth_replace) maxlin = L;
            else break;
        }
        int maxlout = 1;
        for (int L = 1; L <= lm; ++L) {
            if (WI_PO(0) - WI_PO(L - 1) < depth_replace) maxlout = L;
            else break;
        }

        for (int L = 1; L <= maxlout; ++L) {
            const float ptarg = WI_PO(0) - WI_PO(L - 1);
            for (int LL = 2; LL <= maxlin; ++LL) {
                const float lo = p2 - WI_PI(LL - 1);
                const float hi = (LL + 1 <= nlev) ? (p2 - WI_PI(LL)) : __int_as_float(0x7f800000);
                if (lo < ptarg && hi > ptarg) {
                    const float dlnp = logf(lo) - logf(hi);
                    const float dudlnp = (WI_UI(LL - 1) - WI_UI(LL)) / dlnp;
                    const float dvdlnp = (WI_VI(LL - 1) - WI_VI(LL)) / dlnp;
                    WI_UO(L - 1) = WI_UI(LL - 1) + dudlnp * (logf(ptarg) - logf(lo));
                    WI_VO(L - 1) = WI_VI(LL - 1) + dvdlnp * (logf(ptarg) - logf(lo));
                    break;
                }
            }
        }
    }

    // B-grid eastern boundary: U(IDE-1,J,K) = V(IDE-1,J,K) = 0 on every row
    if (zero_east && i0 == im - 1) {
        for (int k = 0; k < lm; ++k) {
            WI_UO(k) = 0.0f;
            WI_VO(k) = 0.0f;
        }
    }
#undef WI_PI
#undef WI_PO
#undef WI_UI
#undef WI_VI
#undef WI_UO
#undef WI_VO
}

// ---------------------------------------------------------------------------
// rh_to_mxrat (liquid branch, wrt_liquid = .true., as nemsinterp calls it) followed by the caller's
// QTMP2 = QTMP/(1+QTMP): specific humidity on the input levels. One thread per cell.
// rh is clamped in place (INOUT), as the source does.
// ---------------------------------------------------------------------------
extern "C" __global__ void init_gfs_vinterp_rhq(
    float* __restrict__ rh,              // [nlev, jm, im]  INOUT
    const float* __restrict__ t,         // [nlev, jm, im]
    const float* __restrict__ p,         // [nlev, jm, im]  (Pa)
    float* __restrict__ qtmp2,           // [nlev, jm, im]  specific humidity out
    int im, int jm, int nlev)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= nlev) return;
    const size_t n = IDX3(i0, j0, l0, im, jm);
    const float r = fminf(fmaxf(rh[n], 0.001f), 100.0f);
    rh[n] = r;
    const float tk = t[n];
    const float pk = p[n];
    const float svp1 = 0.6112f, svp2 = 17.67f, svp3 = 29.65f, svpt0 = 273.15f, eps = 0.622f;
    const float es = svp1 * 10.0f * expf(svp2 * (tk - svpt0) / (tk - svp3));
    const float qs = eps * es / (pk / 100.0f - es);
    const float q = fmaxf(0.01f * r * qs, 0.0f);
    qtmp2[n] = q / (1.0f + q);
}

// ---------------------------------------------------------------------------
// CWM (condensate) accumulation on the model mid levels, one species at a time (source lines 940-1010).
// The source thresholds each interpolated species below 1e-11 to zero, then sums
// CWM = SNMR + CLWMR + RWMR + CICE. Here each species is interpolated into one scratch buffer and added
// in that same order (reset = 1 for SNMR: 0 + SNMR = SNMR exactly). With zero inputs the log
// interpolation returns its 1e-13 floor and the threshold returns 0, so running it unconditionally
// matches the source's maxval gate.
// ---------------------------------------------------------------------------
extern "C" __global__ void init_gfs_vinterp_cwm_acc(
    float* __restrict__ species,         // [lm, jm, im]  INOUT (thresholded in place)
    float* __restrict__ cwm,             // [lm, jm, im]  accumulated CWM
    int reset, int im, int jm, int lm)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t n = IDX3(i0, j0, l0, im, jm);
    if (species[n] < 1.0e-11f) species[n] = 0.0f;
    cwm[n] = (reset ? 0.0f : cwm[n]) + species[n];
}
