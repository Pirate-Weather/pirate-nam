// phys_rrtm_setaer.cu -- SETAER (aerosol optical properties per layer), RRTM group, WOOF-NAM.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1). NOAA US government work; the NOAA header of
// phys/radiation_aerosols_nmmb.f (module_radiation_aerosols_nmmb) is kept as the notice:
//   NOAA / NCEP, module_radiation_aerosols_nmmb ("climatological atmospheric aerosol schemes for
//   radiation computations"; subroutine setaer, NCEP radiation_aerosols v5.2 Jan 2013 tag).
//
// Ported to CUDA for WOOF-NAM from phys/radiation_aerosols_nmmb.f:1845-2371 (nam.v4.2.11, 83df4e13c1).
//
// Unit split (CONTRACT section 7, UNITS.json): this file holds the setaer routine's own code.
// The call to aer_property (Fortran line 2045, unit phys-rrtm-aer-property, kernel file
// phys_rrtm_aer_property.cu) sits between the prep and volc launches and is made by the wrapper
// (namgpu/physics/rrtm/phys_rrtm_setaer.py), not here.
//
// Kernels (one launch each, all device pointers, no host sync):
//   phys_rrtm_setaer_zero  : Fortran 1871-1895 (zero aerosw/aerolw)
//   phys_rrtm_setaer_prep  : Fortran 1956-1964 (alon/alat) and 1994-2051 (dz/hz per column)
//   phys_rrtm_setaer_volc  : Fortran 2092-2352 (stratospheric volcanic aerosol, lavoflg branch)
//
// Layout (CONTRACT section 1). Fortran arrays (IMAX, NLAY|NLP1) are flattened over columns:
// column c = j*IM + i (the [JM, IM] grid), level k (1-based Fortran, zero-based k0 = k-1) at
//   X[k0 * imax + c]        for per-level fields (prsi, prsl, tlay, qlay, dz, hz)
// Outputs (NF_AESW=3 or NF_AELW=3 fields, NBDSW=14 or NBDLW=16 bands, NLAY levels, IMAX columns):
//   aerosw[((n * NBDSW + b) * NLAY + k0) * imax + c]    n: 0=tau, 1=ssa, 2=asy; b zero-based band
//   aerolw[((n * NBDLW + b) * NLAY + k0) * imax + c]
// The wrapper passes imax = IM*JM and nlay = LM+1, so the [NF, NBDSW, LMK, JM, IM] CuPy arrays
// used by phys_rrtm_grrad_aerosols.py are exactly this layout.
//
// Precision: kfpt=single (machine.f kind_phys = 4), so float everywhere; logf/powf/expf are the
// NVRTC single-precision functions; FMA contraction is on (--fmad=true, kernels/__init__.py).

#include "nmmb_common.cuh"

#define SA_NF        3      // NF_AESW = NF_AELW (module_radiation_aerosols_nmmb)
#define SA_NBDSW     14     // NBDSW  = NBANDS of radsw_param_nmmb.f
#define SA_NBDLW     16     // NBDLW  = NBANDS of radlw_param_nmmb.f
#define SA_NSWSTR    1      // radsw_param_nmmb.f: NSWSTR = 1
#define SA_PSRFH     5.0f   // setaer local psrfh (mb) for the upper bound
#define SA_CON_PI    3.1415926535897931f  // physcons.f con_pi
#define SA_CON_RD    2.8705e+2f           // physcons.f con_rd
#define SA_CON_G     9.80665e+0f          // physcons.f con_g
#define SA_CON_RV    4.6150e+2f           // physcons.f con_rv
#define SA_CON_FVIRT (SA_CON_RV / SA_CON_RD - 1.0f)  // physcons.f con_fvirt

// radsw_param_nmmb.f data wvnum1 / wvnum2 (SW band wavenumber limits, cm-1), bands 1..14
__constant__ float SA_WVNSW1[SA_NBDSW] = {
    2600.0f, 3251.0f, 4001.0f, 4651.0f, 5151.0f, 6151.0f, 7701.0f,
    8051.0f, 12851.0f, 16001.0f, 22651.0f, 29001.0f, 38001.0f, 820.0f };
__constant__ float SA_WVNSW2[SA_NBDSW] = {
    3250.0f, 4000.0f, 4650.0f, 5150.0f, 6150.0f, 7700.0f, 8050.0f,
    12850.0f, 16000.0f, 22650.0f, 29000.0f, 38000.0f, 50000.0f, 2600.0f };
// radlw_param_nmmb.f data wvnlw1 / wvnlw2 (LW band wavenumber limits, cm-1), bands 1..16
__constant__ float SA_WVNLW1[SA_NBDLW] = {
    10.0f, 251.0f, 501.0f, 631.0f, 701.0f, 821.0f, 981.0f, 1081.0f,
    1181.0f, 1391.0f, 1481.0f, 1801.0f, 2081.0f, 2251.0f, 2381.0f, 2601.0f };
__constant__ float SA_WVNLW2[SA_NBDLW] = {
    250.0f, 500.0f, 630.0f, 700.0f, 820.0f, 980.0f, 1080.0f, 1180.0f,
    1390.0f, 1480.0f, 1800.0f, 2080.0f, 2250.0f, 2380.0f, 2600.0f, 3000.0f };

// ---------------------------------------------------------------------------
// Zero the outputs. Fortran 1871-1895: aerosw(IMAX, NLAY, NBDSW, NF) and aerolw likewise.
// Grid-stride over the flat arrays; n_sw = NF*NBDSW*NLAY*IMAX, n_lw = NF*NBDLW*NLAY*IMAX.
// ---------------------------------------------------------------------------
extern "C" __global__ void phys_rrtm_setaer_zero(float* __restrict__ aerosw,
                                                 float* __restrict__ aerolw,
                                                 size_t n_sw, size_t n_lw)
{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t e = t; e < n_sw; e += stride) aerosw[e] = 0.0f;
    for (size_t e = t; e < n_lw; e += stride) aerolw[e] = 0.0f;
}

// ---------------------------------------------------------------------------
// Prep, one thread per column.
//   alon/alat: Fortran 1956-1964. rdg = 180/con_pi; alon wraps negatives by +360;
//              alat clamped to [-90, 90].
//   dz/hz    : Fortran 1994-2051, run only when do_dz (laswflg .or. lalwflg, host-side).
//              ivflip=1: input surface->top (prsln(k)=log(prsi(k)), prsln(NLP1)=log(prsl(NLAY)),
//                        dz(NLAY) doubled, hz built upward from the surface);
//              ivflip=0: input top->surface (prsln(1)=log(prsl(1)), prsln(k)=log(prsi(k)),
//                        dz(1) doubled, hz built downward from hz(NLP1)=0).
// radiation_aerosols_nmmb.f 1845-2371 has no stop, abort or print statement (checked), so there
// is no abort path to reproduce.
// ---------------------------------------------------------------------------
extern "C" __global__ void phys_rrtm_setaer_prep(
    const float* __restrict__ prsi,   // [nlp1, imax]
    const float* __restrict__ prsl,   // [nlay, imax]
    const float* __restrict__ tlay,   // [nlay, imax]
    const float* __restrict__ qlay,   // [nlay, imax]
    const float* __restrict__ xlon,   // [imax] radians
    const float* __restrict__ xlat,   // [imax] radians
    float* __restrict__ alon,         // [imax] out, degrees
    float* __restrict__ alat,         // [imax] out, degrees
    float* __restrict__ dz,           // [nlay, imax] out (scratch for aer_property)
    float* __restrict__ hz,           // [nlp1, imax] out (scratch for aer_property)
    int imax, int nlay, int nlp1, int ivflip, int do_dz)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= imax) return;

    // Fortran rdg = 180.0 / con_pi ; rovg = 0.001 * con_rd / con_g ; con_fvirt from physcons.
    const float rdg  = 180.0f / SA_CON_PI;
    const float rovg = 0.001f * SA_CON_RD / SA_CON_G;
    const float fvirt = SA_CON_FVIRT;

    // Fortran 1956-1964.
    float a_lon = xlon[col] * rdg;
    if (a_lon < 0.0f) a_lon = a_lon + 360.0f;
    alon[col] = a_lon;
    float a_lat = xlat[col] * rdg;
    // NaN-propagating clamp: Fortran MIN/MAX propagate NaN, fminf/fmaxf would replace it.
    if (a_lat < -90.0f) a_lat = -90.0f;
    if (a_lat > 90.0f)  a_lat = 90.0f;
    alat[col] = a_lat;

    if (!do_dz) return;

#define PRSI(k)  prsi[((k) - 1) * imax + col]   // 1-based Fortran level
#define PRSL(k)  prsl[((k) - 1) * imax + col]
#define TLAY(k)  tlay[((k) - 1) * imax + col]
#define QLAY(k)  qlay[((k) - 1) * imax + col]
#define DZ(k)    dz[((k) - 1) * imax + col]
#define HZ(k)    hz[((k) - 1) * imax + col]

    if (ivflip == 1) {
        // Fortran 1998-2018 (input from sfc to toa). prsln(k) for k=1..NLAY is log(prsi(k));
        // prsln(NLP1) = log(prsl(NLAY)).
        for (int k = nlay; k >= 1; --k) {
            const float lnk  = logf(PRSI(k));
            const float lnk1 = (k + 1 <= nlay) ? logf(PRSI(k + 1)) : logf(PRSL(nlay));
            DZ(k) = rovg * (lnk - lnk1) * TLAY(k) * (1.0f + fvirt * QLAY(k));
        }
        DZ(nlay) = 2.0f * DZ(nlay);

        HZ(1) = 0.0f;
        for (int k = 1; k <= nlay; ++k) {
            HZ(k + 1) = HZ(k) + DZ(k);
        }
    } else {
        // Fortran 2019-2031 (input from toa to sfc). prsln(1) = log(prsl(1)); prsln(k)=log(prsi(k)), k>=2.
        for (int k = 1; k <= nlay; ++k) {
            const float lnk  = (k == 1) ? logf(PRSL(1)) : logf(PRSI(k));
            const float lnk1 = logf(PRSI(k + 1));
            DZ(k) = rovg * (lnk1 - lnk) * TLAY(k) * (1.0f + fvirt * QLAY(k));
        }
        DZ(1) = 2.0f * DZ(1);

        HZ(nlp1) = 0.0f;
        for (int k = nlay; k >= 1; --k) {
            HZ(k) = HZ(k + 1) + DZ(k);
        }
    }

#undef PRSI
#undef PRSL
#undef TLAY
#undef QLAY
#undef DZ
#undef HZ
}

// ---------------------------------------------------------------------------
// Volcanic stratospheric aerosol, one thread per column. Fortran 2092-2352 (lavoflg branch).
// Runs AFTER the aer_property stage (its assignments would otherwise be overwritten by this
// routine's accumulations, and the Fortran order is aer_property then volcanic).
//   ivolae : [10, 4, 12] int32, C order, i.e. Fortran ivolae(month, zone, year) at
//            ivolae[((year-1)*4 + (zone-1))*12 + (month-1)], with year = i1 = mod(kyrsav,10)+1
//            and month = kmonsav (set by the host from the aer_update unit).
//   do_sw / do_lw : laddsw / laddlw (host-side logic from iaerflg==100 and laswflg/lalwflg).
//   nlwbnd1 : NLWBND == 1 (lalw1bd); the wrapper passes 0 (lalw1bd = .false. in physparam).
// ---------------------------------------------------------------------------
__device__ inline int setaer_ivol(const int* __restrict__ ivolae, int year1, int zone, int month)
{
    return ivolae[((year1 - 1) * 4 + (zone - 1)) * 12 + (month - 1)];
}

extern "C" __global__ void phys_rrtm_setaer_volc(
    float* __restrict__ aerosw,        // [NF, NBDSW, nlay, imax] in/out
    float* __restrict__ aerolw,        // [NF, NBDLW, nlay, imax] in/out
    const float* __restrict__ prsi,    // [nlp1, imax]
    const float* __restrict__ alat,    // [imax] degrees
    const int* __restrict__ ivolae,    // [10, 4, 12]
    int imax, int nlay, int ivflip, int year1, int month,
    int do_sw, int do_lw, int nlwbnd1)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= imax) return;

#define PRSI(k)  prsi[((k) - 1) * imax + col]
#define SWA(n, b, k) aerosw[((((size_t)(n)) * SA_NBDSW + (b)) * nlay + ((k) - 1)) * imax + col]
#define LWA(n, b, k) aerolw[((((size_t)(n)) * SA_NBDLW + (b)) * nlay + ((k) - 1)) * imax + col]

    // Fortran 2104-2127: volcae(i) from the 45-degree latitude zones.
    const float lat = alat[col];
    float volcae;
    if (lat > 46.0f) {
        volcae = 1.0e-4f * (float)setaer_ivol(ivolae, year1, 1, month);
    } else if (lat > 44.0f) {
        volcae = 5.0e-5f * (float)(setaer_ivol(ivolae, year1, 1, month) + setaer_ivol(ivolae, year1, 2, month));
    } else if (lat > 1.0f) {
        volcae = 1.0e-4f * (float)setaer_ivol(ivolae, year1, 2, month);
    } else if (lat > -1.0f) {
        volcae = 5.0e-5f * (float)(setaer_ivol(ivolae, year1, 2, month) + setaer_ivol(ivolae, year1, 3, month));
    } else if (lat > -44.0f) {
        volcae = 1.0e-4f * (float)setaer_ivol(ivolae, year1, 3, month);
    } else if (lat > -46.0f) {
        volcae = 5.0e-5f * (float)(setaer_ivol(ivolae, year1, 3, month) + setaer_ivol(ivolae, year1, 4, month));
    } else {
        volcae = 1.0e-4f * (float)setaer_ivol(ivolae, year1, 4, month);
    }

    // Fortran 2134-2141 / 2225-2232: psrfl from |alat| (polar 250 mb, tropics 150 mb, else linear).
    const float tmp1a = fabsf(lat);
    float psrfl;
    if (tmp1a > 70.0f)      psrfl = 250.0f;
    else if (tmp1a < 20.0f) psrfl = 150.0f;
    else                    psrfl = 110.0f + 2.0f * tmp1a;

    int kcuth, kcutl;
    float rdelp;

    if (ivflip == 0) {
        // Fortran 2143-2165 (input data from toa to sfc).
        kcuth = nlay - 1;
        kcutl = 2;
        rdelp = 1.0f / PRSI(2);
        for (int k = 2; k <= nlay - 2; ++k) {
            if (PRSI(k) >= SA_PSRFH) { kcuth = k - 1; break; }
        }
        for (int k = 2; k <= nlay - 2; ++k) {
            if (PRSI(k) >= psrfl) {
                kcutl = k - 1;
                rdelp = 1.0f / (PRSI(k) - PRSI(kcuth));
                break;
            }
        }
    } else {
        // Fortran 2235-2257 (input data from sfc to toa).
        kcuth = 2;
        kcutl = nlay - 1;
        rdelp = 1.0f / PRSI(nlay - 1);
        for (int k = nlay - 1; k >= 2; --k) {
            if (PRSI(k) >= SA_PSRFH) { kcuth = k; break; }
        }
        for (int k = nlay; k >= 2; --k) {
            if (PRSI(k) >= psrfl) {
                kcutl = k;
                rdelp = 1.0f / (PRSI(k) - PRSI(kcuth + 1));
                break;
            }
        }
    }

    const int kh = kcuth;
    const int kl = kcutl;

    if (do_sw) {
        // Fortran 2166-2200 (ivflip=0) and 2284-2318 (ivflip=1). Bands are independent.
        for (int m = 1; m <= SA_NBDSW; ++m) {
            const int mb = SA_NSWSTR + m - 1;       // 1-based band index into wvnum tables
            const float w1 = SA_WVNSW1[mb - 1];
            const float w2 = SA_WVNSW2[mb - 1];
            float tmp2e;
            if (w1 > 20000.0f)      tmp2e = 0.74f;   // range of wvlth < 0.5 mu
            else if (w2 < 20000.0f) tmp2e = 1.14f;   // range of wvlth > 0.5 mu
            else                    tmp2e = 0.94f;   // range in between
            const float tmp1 = powf(0.275e-4f * (w2 + w1), tmp2e);
            const int b = m - 1;

            if (ivflip == 0) {
                for (int k = kh; k <= kl; ++k) {
                    const float tmp2 = tmp1 * ((PRSI(k + 1) - PRSI(k)) * rdelp);
                    SWA(0, b, k) = SWA(0, b, k) + tmp2 * volcae;
                }
                if (SWA(0, b, kl) > 10.0f * SWA(0, b, kl + 1)) {
                    const float tmp2 = SWA(0, b, kl) + SWA(0, b, kl + 1);
                    SWA(0, b, kl)     = 0.8f * tmp2;
                    SWA(0, b, kl + 1) = 0.2f * tmp2;
                }
            } else {
                for (int k = kl; k <= kh; ++k) {
                    const float tmp2 = tmp1 * ((PRSI(k) - PRSI(k + 1)) * rdelp);
                    SWA(0, b, k) = SWA(0, b, k) + tmp2 * volcae;
                }
                if (SWA(0, b, kl) > 10.0f * SWA(0, b, kl - 1)) {
                    const float tmp2 = SWA(0, b, kl) + SWA(0, b, kl - 1);
                    SWA(0, b, kl)     = 0.8f * tmp2;
                    SWA(0, b, kl - 1) = 0.2f * tmp2;
                }
            }
        }
    }

    if (do_lw) {
        // Fortran 2201-2223 (ivflip=0) and 2319-2351 (ivflip=1).
        if (nlwbnd1) {
            // NLWBND == 1: one broad band, same tau added to every LW band (lalw1bd = .false.
            // in physparam, so the wrapper never takes this branch).
            const float tmp1 = powf(0.55f / 11.0f, 1.2f);
            if (ivflip == 0) {
                for (int k = kh; k <= kl; ++k) {
                    const float tmp2 = tmp1 * ((PRSI(k + 1) - PRSI(k)) * rdelp) * volcae;
                    for (int m = 1; m <= SA_NBDLW; ++m) LWA(0, m - 1, k) = LWA(0, m - 1, k) + tmp2;
                }
            } else {
                for (int k = kl; k <= kh; ++k) {
                    const float tmp2 = tmp1 * ((PRSI(k) - PRSI(k + 1)) * rdelp) * volcae;
                    for (int m = 1; m <= SA_NBDLW; ++m) LWA(0, m - 1, k) = LWA(0, m - 1, k) + tmp2;
                }
            }
        } else {
            for (int m = 1; m <= SA_NBDLW; ++m) {
                const float tmp1 = powf(0.275e-4f * (SA_WVNLW2[m - 1] + SA_WVNLW1[m - 1]), 1.2f);
                if (ivflip == 0) {
                    for (int k = kh; k <= kl; ++k) {
                        const float tmp2 = tmp1 * ((PRSI(k + 1) - PRSI(k)) * rdelp);
                        LWA(0, m - 1, k) = LWA(0, m - 1, k) + tmp2 * volcae;
                    }
                } else {
                    for (int k = kl; k <= kh; ++k) {
                        const float tmp2 = tmp1 * ((PRSI(k) - PRSI(k + 1)) * rdelp);
                        LWA(0, m - 1, k) = LWA(0, m - 1, k) + tmp2 * volcae;
                    }
                }
            }
        }
    }

#undef PRSI
#undef SWA
#undef LWA
}
