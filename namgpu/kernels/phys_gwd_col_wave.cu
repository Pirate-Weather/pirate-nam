// phys_gwd_col_wave.cu -- GWD_col part 2: low-level wind, base stress, Ri-limited stress profile
// and top de-acceleration limit (orographic gravity wave drag, one column per thread).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_GWD (module_GWD.F90), "Initially incorporated
// into the WRF NMM from the GFS by B. Ferrier in April/May 2007; Ratko added in NMM-B (July '08);
// Mountain blocking made more robust in July 2016 by BSF".
// Ported to CUDA for nam-gpu from nmm/module_GWD.F90:898-1132 (nam.v4.2.11, 83df4e13c1).
//
// Indexing. Every per-point array is full-domain and is read at p = ipt[c], the cell index of the
// c-th active point. This is the Fortran IPT compaction done by index, so no compacted copy is
// needed; phys-gwd-col-blocking writes full-domain arrays and this unit reads them as they are.
//   [JM,IM]      HPRIME OC DUsfc DVsfc UBAR VBAR ROLL BNV2bar OA CLX, KREF and IDXZB (int)
//   [LM,JM,IM]   U1 V1 DEL A B RO BNV2 RI_N DB (level stride np = JM*IM)
//   [LM+1,JM,IM] PRSI (level stride np)
// Scratch VELCO, TAUP and TAUD are compact [LM+1, npt] (c = 0..npt-1, level stride npt), allocated
// once per npt. Level index is 1-based inside the kernel (Fortran K), so every loop below reads
// like the Fortran it came from.
// GWD_col vertical order is bottom-up: K = 1 is the surface, PRSI(K=1) is the surface pressure.
// PRSI is in centibars, as phys-gwd-col-blocking and phys-gwd-driver hand it over.
//
// Executed statements of GWD_col lines 898-1132, every one kept (Fortran order):
//   ULOW = max(sqrt(UBAR^2+VBAR^2), HONE); ULOI = 1/ULOW; DTFAC = 1; ICRILV = .false.; XN=YN=TAUB=0
//   VELCO(K) = 0.5*ULOI*((U1(K)+U1(K+1))*UBAR + (V1(K)+V1(K+1))*VBAR),  K = 1..KMM1
//   KINT search (K > KREF, VELCO < VELEPS), then KINT = KREF (the source's own WARNING line)
//   BNV, FR, XN, YN, EFACT, COEFM, XLINV, GFOBNV, TAUB (base stress), SCOR (Scorer parameter)
//   TAUP(K) = TAUB for K <= KREF
//   Ri-limited profile, K = KMPS..KMM1: ICRILV update, Scorer rescale (OA > 0, KP1 < KINT),
//     Brunt-Vaisala, TEM1, HD/FRO, Shutts (1985) minimum Richardson number RIM, Lindzen (1981)
//     saturation vs. TAUP(KP1) = TAUP(K)*RSCOR, TAUP(KP1) = min(TAUP(KP1), TAUP(K))
//   LCAP = KM: TAUP(KM+1) = (PRSI(KM+1)/PRSI(KM)) * TAUP(KM)
//   TAUD = G*(TAUP(K+1)-TAUP(K))/DEL, K = 1..KM; TAUD(KM) *= FACTOP (top limit, LCAP = KM)
//   DTFAC = min(DTFAC, |VELCO/(DELTIM*TAUD)|) for K > KREF, PRSI(K) >= RLOLEV, TAUD /= 0
//   TAUD *= DTFAC; DBIM branch (K < IDXZB, IDXZB /= 0) or DTAUX/DTAUY branch, DUsfc/DVsfc sums
//   DUsfc = ROGN*DUsfc, DVsfc = ROGN*DVsfc
// Notes on what is NOT copied and why:
//   - The source's "dbg" write(0,...) line (DBIM < 0 print) is a host print: dropped (CONTRACT 3).
//   - KBPS, KMPS (row-wide min/max of KREF) only bound the K loops in the source; the per-column
//     tests K <= KREF / K >= KREF already exclude every level outside them, so the loops run over
//     all levels with the same tests (identical result, no row reduction on the device).
//   - KBPSP1, KBPSM1, KMPSP1, KMLL, KMDS, IDIR, NWD etc. belong to the blocking part, not here.
//   - DUsfc/DVsfc are zeroed by the blocking part (source lines before 898); this kernel adds to them.
// TAUP: the source sets TAUP(I,K)=0.0 for K=1..KM before any read (module_GWD.F90:816), and writes
// TAUP(I,KM+1) in the LCAP loop before it is read (1055-1060). No uninitialised read exists. The
// kernel zeroes levels 1..LM+1, which is the same set of values the source defines.
//
// Precision: double (kind_phys = real*8), as the Fortran declares for GWD_col locals and PARAMETERs
// (module_GWD.F90:18, 494-502) and as phys-gwd-col-blocking, which feeds this unit, already uses.
// The constants below are double.
//
// Fast numerics: built with --fmad=true (CONTRACT section 3). Real ** real is pow.

#pragma once
#include "nmmb_common.cuh"

#define GW_G       9.806
#define GW_ROGN    (-1000.0 * (1.0 / 9.806))    /* -1000.*ROG, ROG = 1./G */
#define GW_RIC     0.25
#define GW_EFMIN   0.0
#define GW_EFMAX   10.0
#define GW_HPMAX   2400.0                          /* source: "hpmax was 200.0" */
#define GW_FRMAX   100.0
#define GW_CEOFRC  (0.8 / 1.0)                    /* CE/FRC, CE = 0.8, FRC = 1.0 */
#define GW_CG      0.5
#define GW_GMAX    1.0
#define GW_VELEPS  1.0
#define GW_HZERO   0.0
#define GW_HONE    1.0
#define GW_HE_2    0.01
#define GW_HE_1    0.1

// full-domain [LM, NP] (and [LM+1, NP] for interfaces): Fortran X(J, K), 1-based K
#define GW_FL(arr, k, p, np)   arr[((size_t)((k) - 1)) * (np) + (p)]
// compact [LM, npt]: Fortran X(I, K), 1-based K
#define GW_CL(arr, k, c, npt)  arr[((size_t)((k) - 1)) * (npt) + (c)]

__device__ void gwd_col_wave_dev(
    const int c, const int p,        // [npt] full-domain cell index of each active point
    const int npt,
    const int np,                       // JM*IM
    const int lm,                       // LM (KM = LM)
    const double dtphs,                  // DELTIM (physics step, s)
    const double cleff,                  // CLEFF (from GWD_init)
    const double factop,                 // FACTOP (top limit on de-acceleration)
    const double rlolev,                 // RLOLEV (pressure below which no critical-level limit)
    // full-domain [JM,IM]
    double* __restrict__ dusfc,          // INOUT: accumulated surface-stress sums
    double* __restrict__ dvsfc,
    const double* __restrict__ hprime,
    const double* __restrict__ oc,
    // full-domain [LM,JM,IM] and interfaces [LM+1,JM,IM]
    const double* __restrict__ u1,
    const double* __restrict__ v1,
    const double* __restrict__ del,
    const double* __restrict__ prsi,
    double* __restrict__ a,              // INOUT tendency arrays
    double* __restrict__ b,
    // full-domain [JM,IM], read at p (blocking-part outputs)
    const double* __restrict__ ubar,
    const double* __restrict__ vbar,
    const double* __restrict__ roll,
    const double* __restrict__ bnv2bar,
    const double* __restrict__ oa,
    const double* __restrict__ clx,
    const int* __restrict__ kref,
    const int* __restrict__ idxzb,
    // full-domain [LM,JM,IM], read at p (blocking-part outputs)
    const double* __restrict__ ro,
    const double* __restrict__ bnv2,
    const double* __restrict__ ri_n,
    const double* __restrict__ db,
    // scratch [LM+1, npt]
    double* __restrict__ taup,
    double* __restrict__ taud,
    double* __restrict__ velco)
{
    const int km = lm;
    const int kmm1 = lm - 1;
    const int lcap = km;
    const int lcapp1 = lcap + 1;

    // ---- INITIALIZE SOME ARRAYS (909-916) and LOW-LEVEL WIND (918)
    const double ubar_c = ubar[p];
    const double vbar_c = vbar[p];
    const double ulow = fmax(sqrt(ubar_c * ubar_c + vbar_c * vbar_c), GW_HONE);
    const double uloi = 1.0 / ulow;
    double dtfac = 1.0;
    bool icrilv = false;
    double xn = 0.0;
    double yn = 0.0;
    double taub = 0.0;
    const int kr = kref[p];
    const double oa_c = oa[p];
    const double clx_c = clx[p];
    const double roll_c = roll[p];
    const double bnv2bar_c = bnv2bar[p];
    const double oc_p = oc[p];
    const double hprime_p = hprime[p];

    // ---- VELCO: projected wind on the interfaces, K = 1..KMM1 (922-930)
    for (int k = 1; k <= kmm1; ++k) {
        GW_CL(velco, k, c, npt) = 0.5 * uloi *
            ((GW_FL(u1, k, p, np) + GW_FL(u1, k + 1, p, np)) * ubar_c +
             (GW_FL(v1, k, p, np) + GW_FL(v1, k + 1, p, np)) * vbar_c);
    }

    // ---- KINT search (933-945); overwritten by KINT = KREF below, kept as in the source
    int kint = km;
    for (int k = 1; k <= kmm1; ++k) {
        if (k > kr) {
            if (GW_CL(velco, k, c, npt) < GW_VELEPS && kint == km) {
                kint = k + 1;
            }
        }
    }
    kint = kr;

    // ---- base stress, enhancement, Scorer parameter (948-985)
    double xlinv = 0.0;
    double scor = 0.0;
    {
        const double bnv = sqrt(bnv2bar_c);
        double fr = bnv * uloi * fmin(hprime_p, GW_HPMAX);
        fr = fmin(fr, GW_FRMAX);
        xn = ubar_c * uloi;
        yn = vbar_c * uloi;

        double efact = pow(oa_c + 2.0, GW_CEOFRC * fr);
        efact = fmin(fmax(efact, GW_EFMIN), GW_EFMAX);

        const double coefm = pow(1.0 + clx_c, oa_c + 1.0);
        xlinv = coefm * cleff;

        const double tem0 = fr * fr * oc_p;
        const double gfobnv = GW_GMAX * tem0 / ((tem0 + GW_CG) * bnv);   // G/N0

        taub = xlinv * roll_c * ulow * ulow * ulow * gfobnv * efact;     // base flux Tau0

        const int kk = max(1, kr - 1);
        const double vk = GW_CL(velco, kk, c, npt);
        const double tvel = fmax(vk * vk, GW_HE_1);
        // SCOR(I) = BNV2(I,K) / TEM: Scorer parameter below the reference level, carried below.
        scor = GW_FL(bnv2, kk, p, np) / tvel;
    }

    // ---- SET UP BOTTOM VALUES OF STRESS: TAUP(K) = TAUB for K <= KREF (ref. lines 985-988).
    // TAUP above KREF is zeroed above first (source line 816).
    for (int k = 1; k <= lm + 1; ++k) {
        GW_CL(taup, k, c, npt) = 0.0;
    }
    for (int k = 1; k <= km; ++k) {
        if (k <= kr) {
            GW_CL(taup, k, c, npt) = taub;
        }
    }

    // ---- vertical structure of the stress (994-1051)
    for (int k = 1; k <= kmm1; ++k) {
        const int kp1 = k + 1;
        if (k >= kr) {
            // UNSTABLE LAYER IF RI < RIC or upper-air wind along the surface wind <= 0
            icrilv = icrilv || (GW_FL(ri_n, k, p, np) < GW_RIC) ||
                     (GW_CL(velco, k, c, npt) <= 0.0);
        }
        if (k >= kr) {
            if (!icrilv && GW_CL(taup, k, c, npt) > 0.0) {
                const double temv = 1.0 / fmax(GW_CL(velco, k, c, npt), GW_HE_2);
                double rscor;
                if (oa_c > 0.0 && kp1 < kint) {
                    const double scork = GW_FL(bnv2, k, p, np) * temv * temv;
                    rscor = fmin(GW_HONE, scork / scor);
                    scor = scork;
                } else {
                    rscor = 1.0;
                }

                const double brvf = sqrt(GW_FL(bnv2, k, p, np));           // Brunt-Vaisala
                const double tem1 = xlinv * (GW_FL(ro, kp1, p, np) + GW_FL(ro, k, p, np)) *
                                   brvf * 0.5 * fmax(GW_CL(velco, k, c, npt), GW_HE_2);
                double hd = sqrt(GW_CL(taup, k, c, npt) / tem1);
                const double fro = brvf * hd * temv;

                // RIM: minimum Richardson number by Shutts (1985)
                const double tem2 = sqrt(GW_FL(ri_n, k, p, np));
                const double tem = 1.0 + tem2 * fro;
                const double rim = GW_FL(ri_n, k, p, np) * (1.0 - fro) / (tem * tem);

                // saturation hypothesis of Lindzen (1981) except tropospheric downstream regions
                if (rim <= GW_RIC && (oa_c <= 0.0 || kp1 >= kint)) {
                    const double temc = 2.0 + 1.0 / tem2;
                    hd = GW_CL(velco, k, c, npt) * (2.0 * sqrt(temc) - temc) / brvf;
                    GW_CL(taup, kp1, c, npt) = tem1 * hd * hd;
                } else {
                    GW_CL(taup, kp1, c, npt) = GW_CL(taup, k, c, npt) * rscor;
                }
                GW_CL(taup, kp1, c, npt) = fmin(GW_CL(taup, kp1, c, npt),
                                                 GW_CL(taup, k, c, npt));
            }
        }
    }

    // ---- LCAP <= KM: TAUP(KLCAP) = SIRA*TAUP(LCAP) for KLCAP = LCAPP1..KM+1 (1053-1060)
    if (lcap <= km) {
        for (int klcap = lcapp1; klcap <= km + 1; ++klcap) {
            const double sira = GW_FL(prsi, klcap, p, np) / GW_FL(prsi, lcap, p, np);
            GW_CL(taup, klcap, c, npt) = sira * GW_CL(taup, lcap, c, npt);
        }
    }

    // ---- -(g/p*) d(tau)/d(sigma) (1063-1068)
    for (int k = 1; k <= km; ++k) {
        GW_CL(taud, k, c, npt) = GW_G * (GW_CL(taup, k + 1, c, npt) - GW_CL(taup, k, c, npt)) /
                                 GW_FL(del, k, p, np);
    }

    // ---- top de-acceleration limit to FACTOP (1073-1078)
    for (int klcap = lcap; klcap <= km; ++klcap) {
        GW_CL(taud, klcap, c, npt) = GW_CL(taud, klcap, c, npt) * factop;
    }

    // ---- critical-line limit on the time step below RLOLEV (1084-1095)
    for (int k = 1; k <= kmm1; ++k) {
        if (k > kr && GW_FL(prsi, k, p, np) >= rlolev) {
            if (GW_CL(taud, k, c, npt) != 0.0) {
                const double tem = dtphs * GW_CL(taud, k, c, npt);
                dtfac = fmin(dtfac, fabs(GW_CL(velco, k, c, npt) / tem));
            }
        }
    }

    // ---- apply DTFAC; tendencies, or damping where IDXZB applies (1097-1122)
    const int idxzb_c = idxzb[p];
    for (int k = 1; k <= km; ++k) {
        const double taud_k = GW_CL(taud, k, c, npt) * dtfac;
        GW_CL(taud, k, c, npt) = taud_k;
        const double dtaux = taud_k * xn;
        const double dtauy = taud_k * yn;
        const double dk = GW_FL(del, k, p, np);
        if (k < idxzb_c && idxzb_c != 0) {
            const double dbk = GW_FL(db, k, p, np);
            double dbim = dbk / (1.0 + dbk * dtphs);
            dbim = fmax(dbim, GW_HZERO);
            GW_FL(a, k, p, np) = -dbim * GW_FL(v1, k, p, np) + GW_FL(a, k, p, np);
            GW_FL(b, k, p, np) = -dbim * GW_FL(u1, k, p, np) + GW_FL(b, k, p, np);
            dusfc[p] = dusfc[p] - dbim * GW_FL(v1, k, p, np) * dk;
            dvsfc[p] = dvsfc[p] - dbim * GW_FL(u1, k, p, np) * dk;
        } else {
            GW_FL(a, k, p, np) = dtauy + GW_FL(a, k, p, np);
            GW_FL(b, k, p, np) = dtaux + GW_FL(b, k, p, np);
            dusfc[p] = dusfc[p] + dtaux * dk;
            dvsfc[p] = dvsfc[p] + dtauy * dk;
        }
    }

    dusfc[p] = GW_ROGN * dusfc[p];
    dvsfc[p] = GW_ROGN * dvsfc[p];
}

// Launchable form over the npt active points ipt[]; the body above is also called column-locally
// by gwd_col() in phys_gwd_col.cuh (c = p = 0, npt = np = 1, local arrays).
extern "C" __global__ void phys_gwd_col_wave(
    const int* __restrict__ ipt,        // [npt] full-domain cell index of each active point
    const int npt,
    const int np,                       // JM*IM
    const int lm,                       // LM (KM = LM)
    const double dtphs,                  // DELTIM (physics step, s)
    const double cleff,                  // CLEFF (from GWD_init)
    const double factop,                 // FACTOP (top limit on de-acceleration)
    const double rlolev,                 // RLOLEV (pressure below which no critical-level limit)
    // full-domain [JM,IM]
    double* __restrict__ dusfc,          // INOUT: accumulated surface-stress sums
    double* __restrict__ dvsfc,
    const double* __restrict__ hprime,
    const double* __restrict__ oc,
    // full-domain [LM,JM,IM] and interfaces [LM+1,JM,IM]
    const double* __restrict__ u1,
    const double* __restrict__ v1,
    const double* __restrict__ del,
    const double* __restrict__ prsi,
    double* __restrict__ a,              // INOUT tendency arrays
    double* __restrict__ b,
    // full-domain [JM,IM], read at p (blocking-part outputs)
    const double* __restrict__ ubar,
    const double* __restrict__ vbar,
    const double* __restrict__ roll,
    const double* __restrict__ bnv2bar,
    const double* __restrict__ oa,
    const double* __restrict__ clx,
    const int* __restrict__ kref,
    const int* __restrict__ idxzb,
    // full-domain [LM,JM,IM], read at p (blocking-part outputs)
    const double* __restrict__ ro,
    const double* __restrict__ bnv2,
    const double* __restrict__ ri_n,
    const double* __restrict__ db,
    // scratch [LM+1, npt]
    double* __restrict__ taup,
    double* __restrict__ taud,
    double* __restrict__ velco)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= npt) return;
    const int p = ipt[c];
    gwd_col_wave_dev(c, p, npt, np, lm, dtphs, cleff, factop, rlolev, dusfc, dvsfc, hprime, oc, u1, v1, del, prsi, a, b, ubar, vbar, roll, bnv2bar, oa, clx, kref, idxzb, ro, bnv2, ri_n, db, taup, taud, velco);
}
