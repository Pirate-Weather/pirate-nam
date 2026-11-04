// phys_rrtm_sol_update.cu -- SOL_UPDATE_NMMB + SOLAR: solar declination, equation of time and
// sun-earth-distance-adjusted solar constant (module_radiation_astronomy_nmmb). Scalar, one thread.
//
// NOTICE: NAM source file phys/radiation_astronomy_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence
// file). The subroutine headers are kept here as the notice:
//   subroutine sol_update_nmmb (jdate, kyear, deltsw, deltim, lsol_chg, me -> slag, sdec, cdec, solcon):
//     computes solar parameters at forecast time. The solar-constant table read (lsol_chg branch) is
//     file I/O and runs on the host in phys_rrtm_sol_update.py (it changes solc0 once per year).
//   subroutine solar (jd, fjd -> r1, dlt, alp): radius vector, declination and right ascension of the sun,
//     and equation of time (module variables sollag, sindec, cosdec).
//   program history: mar-xx-1989 kenneth campana; fall-1988 hualu pan; dec-15-2003 yu-tai hou; mar 2012
//     yu-tai hou (sol_init / sol_update split); jul-08-2012 hsin-mu lin (astronomy_nmmb fix).
// Ported to CUDA for WOOF-NAM from phys/radiation_astronomy_nmmb.f:265-570 (sol_update_nmmb) and
// :575-739 (solar) (nam.v4.2.11, 83df4e13c1).
//
// Precision: kind_phys is 4 in the SINGLE_PREC build (phys/machine.f:20, the WOOF-NAM build), so every
// kind_phys quantity here is float. solc0 is read from the double table storage that phys-rrtm-sol-init
// writes, and rounded to float once, as the Fortran single build holds it.
//
// Fortran -> CUDA map (every assignment of sol_update_nmmb and solar):
//   iyear/imon/iday/ihr/imin/isec = jdate(1,2,3,5,6,7)  -> kernel arguments yr, mon, day, hr, mn, sec
//   jd1 = iw3jdn(iyear,imon,iday)                        -> sol_iw3jdn() below (NCEP W3 Gregorian JDN)
//   if ihr < 12: jd1 = jd1-1; fjd1 = 0.5+hr*hrday+min*minday+sec*secday; else fjd1 = (hr-12)*hrday+...
//   fjd1 = fjd1 + jd1; jd = int(fjd1); fjd = fjd1 - jd  -> same statements
//   call solar(jd, fjd, r1, dlt, alp)                    -> sol_solar() below
//   solcon = solc0/(r1*r1); slag = sollag; sdec = sindec; cdec = cosdec -> out[3], out[0], out[1], out[2]
//   nswr = nint(deltsw/deltim); dtswh = deltsw/3600      -> same
//   if deltsw >= 3600: nn = max(6,min(12,nint(3600/deltim))); nstp = nint(dtswh)*nn+1
//   else: nstp = max(2,min(20,nswr))+1                   -> same
//   anginc = pid12*dtswh/float(nstp-1)                   -> state[3], nstp_out[0] = nstp
//   solar: t1, year, tyear, ec, angin, ador, jdoe, deleqn, year+=365, sni, tini, er, qq,
//          Newton loop 1 (equinox, ccr, iter limit 10), eq, dat, date, em, Newton loop 2, w1, r1,
//          sindec, cosdec, dlt, alp, tst (alp fix), sun, sollag  -> sol_solar() below, every line.
//   The Fortran write(6,*) iteration-limit messages are diagnostics; the loops still exit at iter > 10
//   exactly as written. No printf in kernels (CONTRACT section 3).
//   The `me == 0` prints and the commented-out prtime call are host diagnostics; not ported.
//   The table-read branch (lsol_chg) is host work in the wrapper (file I/O), see its docstring.

#include "nmmb_common.cuh"

#define SOL_PI      3.1415926535897931f   /* con_pi, physcons.f:39 */
#define SOL_TPI     (2.0f * SOL_PI)       /* tpi = 2.0*con_pi */
#define SOL_DEGRAD  (180.0f / SOL_PI)     /* degrad = 180.0/con_pi */
#define SOL_HRDAY   (1.0f / 24.0f)        /* hrday */
#define SOL_MINDAY  (1.0f / 1440.0f)      /* minday */
#define SOL_SECDAY  (1.0f / 86400.0f)     /* secday */
#define SOL_PID12   ((2.0f * asinf(1.0f)) / 12.0f)   /* pid12 = (2.0*asin(1.0))/f12 */
#define SOL_CYEAR   365.25f
#define SOL_CCR     1.3e-6f
#define SOL_TPP     1.55f
#define SOL_SVT6    78.035f
#define SOL_JDOR    2415020               /* jd of epoch, January 0 1900 at 12 h UT */

// NCEP W3 iw3jdn(jyear, jmonth, jday): Julian day number of a Gregorian date. Integer division truncates
// toward zero in both Fortran and C. Check: (1900,1,1) -> 2415021 = SOL_JDOR + 1.
__device__ __forceinline__ int sol_iw3jdn(int jyear, int jmonth, int jday)
{
    return jday - 32075
         + 1461 * (jyear + 4800 + (jmonth - 14) / 12) / 4
         + 367 * (jmonth - 2 - (jmonth - 14) / 12 * 12) / 12
         - 3 * ((jyear + 4900 + (jmonth - 14) / 12) / 100) / 4;
}

// SOLAR (phys/radiation_astronomy_nmmb.f:575-739): inputs jd, fjd; outputs r1, dlt, alp and the module
// variables sollag (equation of time), sindec, cosdec.
__device__ void sol_solar(int jd, float fjd,
                          float* r1, float* dlt, float* alp,
                          float* sollag, float* sindec, float* cosdec)
{
    float t1, year, tyear, ec, angin, ador, deleqn, sni, tini, er, qq;
    int jdoe;
    float e1, ep, cd, eq, dat, date, em, cr, w1, tst, sun;
    int iter;

    // computes time in julian centuries after epoch
    t1 = (float)(jd - SOL_JDOR) / 36525.0f;

    // computes length of anomalistic and tropical years (minus 365 days)
    year  = 0.25964134f + 0.304e-5f * t1;
    tyear = 0.24219879f - 0.614e-5f * t1;

    // computes orbit eccentricity and angle of earth's inclination from t
    ec    = 0.01675104f - (0.418e-4f + 0.126e-6f * t1) * t1;
    angin = 23.452294f - (0.0130125f + 0.164e-5f * t1) * t1;

    ador = (float)SOL_JDOR;
    jdoe = (int)(ador + (SOL_SVT6 * SOL_CYEAR) / (year - tyear));

    // deleqn is updated svt6 for current date
    deleqn = (float)(jdoe - jd) * (year - tyear) / SOL_CYEAR;
    year   = year + 365.0f;
    sni    = sinf(angin / SOL_DEGRAD);
    tini   = 1.0f / tanf(angin / SOL_DEGRAD);
    er     = sqrtf((1.0f + ec) / (1.0f - ec));
    qq     = deleqn * SOL_TPI / year;

    // determine true anomaly at equinox
    e1   = 1.0f;
    cd   = 1.0f;
    iter = 0;
    while (cd > SOL_CCR) {
        ep   = e1 - (e1 - ec * sinf(e1) - qq) / (1.0f - ec * cosf(e1));
        cd   = fabsf(e1 - ep);
        e1   = ep;
        iter = iter + 1;
        if (iter > 10) break;          // Fortran: write iteration count, exit lab_do_1
    }

    eq = 2.0f * atanf(er * tanf(0.5f * e1));

    // date is days since last perihelion passage
    dat  = (float)(jd - SOL_JDOR) - SOL_TPP + fjd;
    date = fmodf(dat, year);

    // solve orbit equations by newton's method
    em   = SOL_TPI * date / year;
    e1   = 1.0f;
    cr   = 1.0f;
    iter = 0;
    while (cr > SOL_CCR) {
        ep   = e1 - (e1 - ec * sinf(e1) - em) / (1.0f - ec * cosf(e1));
        cr   = fabsf(e1 - ep);
        e1   = ep;
        iter = iter + 1;
        if (iter > 10) break;          // Fortran: exit lab_do_2
    }

    w1 = 2.0f * atanf(er * tanf(0.5f * e1));

    *r1 = 1.0f - ec * cosf(e1);

    *sindec = sni * sinf(w1 - eq);
    *cosdec = sqrtf(1.0f - (*sindec) * (*sindec));

    *dlt = asinf(*sindec);
    *alp = asinf(tanf(*dlt) * tini);

    tst = cosf(w1 - eq);
    if (tst < 0.0f) *alp = SOL_PI - *alp;
    if (*alp < 0.0f) *alp = *alp + SOL_TPI;

    sun = SOL_TPI * (date - deleqn) / year;
    if (sun < 0.0f) sun = sun + SOL_TPI;
    *sollag = sun - *alp - 0.03255f;
}

// SOL_UPDATE_NMMB (one thread, one launch, no host sync). Inputs are the jdate fields the routine reads,
// deltsw/deltim, and solc0 from the solar-constant table (device, double storage, rounded to float here).
// Outputs:
//   out[0..3]   = slag, sdec, cdec, solcon   (the four subroutine outputs)
//   state[0..3] = sollag, sindec, cosdec, anginc (module variables kept between calls)
//   nstp_out[0] = nstp (zenith-angle iterations per sw call, used by coszmn)
extern "C" __global__ void sol_update_nmmb(
    const int yr, const int mon, const int day, const int hr, const int mn, const int sec,
    const float deltsw, const float deltim,
    const double* __restrict__ solc0_tbl,
    float* __restrict__ out,
    float* __restrict__ state,
    int* __restrict__ nstp_out)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;

    const float solc0 = (float)solc0_tbl[0];

    // --- forecast time: julian day and fraction of julian day (day starts from noon)
    int jd1 = sol_iw3jdn(yr, mon, day);
    float fjd1;
    if (hr < 12) {
        jd1 = jd1 - 1;
        fjd1 = 0.5f + (float)hr * SOL_HRDAY + (float)mn * SOL_MINDAY + (float)sec * SOL_SECDAY;
    } else {
        fjd1 = (float)(hr - 12) * SOL_HRDAY + (float)mn * SOL_MINDAY + (float)sec * SOL_SECDAY;
    }
    fjd1 = fjd1 + (float)jd1;

    const int jd = (int)fjd1;
    const float fjd = fjd1 - (float)jd;

    float r1, dlt, alp, sollag, sindec, cosdec;
    sol_solar(jd, fjd, &r1, &dlt, &alp, &sollag, &sindec, &cosdec);

    // sun-earth distance adjustment factor appropriate for date
    const float solcon = solc0 / (r1 * r1);

    // --- setting up calculation parameters used by subr coszmn_nmmb
    const int nswr = (int)roundf(deltsw / deltim);     // number of model t-steps per sw call
    const float dtswh = deltsw / 3600.0f;               // time length in hours

    int nstp;
    if (deltsw >= 3600.0f) {                            // longer sw call interval
        int nn = (int)roundf(3600.0f / deltim);          // nint(f3600/deltim)
        nn = max(6, min(12, nn));                        // calc per hour
        nstp = (int)roundf(dtswh) * nn + 1;              // calc per sw call
    } else {                                             // shorter sw call interval
        nstp = max(2, min(20, nswr)) + 1;
    }

    const float anginc = SOL_PID12 * dtswh / (float)(nstp - 1);

    out[0] = sollag;     // slag = sollag
    out[1] = sindec;     // sdec = sindec
    out[2] = cosdec;     // cdec = cosdec
    out[3] = solcon;     // solcon

    state[0] = sollag;
    state[1] = sindec;
    state[2] = cosdec;
    state[3] = anginc;
    nstp_out[0] = nstp;
}
