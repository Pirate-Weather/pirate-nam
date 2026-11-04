// nam-gpu unit phys-fer-init: FERRIER_INIT_hr, the once-per-run build of the Ferrier-Aligo
// microphysics constants, growth table and saturation-vapour-pressure tables.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The routine's own documentation block (module_MP_FER_HIRES, Ferrier and Jin, WRF heritage)
// is the notice; NAM keeps it in the Fortran file.
// Ported to CUDA for nam-gpu from phys/module_MP_FER_HIRES.F90:2391-2744 (nam.v4.2.11, 83df4e13c1),
// with the routines it calls: MY_GROWTH_RATES_NMM_hr (2746-2785), GPVS_hr (2791-2836),
// FPVS0 (2894-2911), FPVSX (2915-2971) and FPVSX0 (2974-2989).
//
// Split across launches (all on ctx.stream, no host sync, init only):
//   1. phys_fer_init_gpvs     GPVS_hr: TBPVS(JX), TBPVS0(JX), JX = 1..NX        (NX threads)
//   2. (growth table)        MY_GROWTH_RATES_NMM_hr is phys-fer-growth-rates, run by the wrapper
//   3. phys_fer_init_sdens    SDENS(I), I = MDImin..MDImax                       (NTAB threads)
//   4. phys_fer_init_scalars  the SAVE scalars of FERRIER_INIT_hr, ESW0 (reads TBPVS0 written by
//                             step 1) and MP_RESTART_STATE(1:43)               (one thread)
//   5. phys_fer_init_fill     F_ICE_PHY = 0, F_RAIN_PHY = 0, F_RIMEF_PHY = 1, not-RESTART block
//                             (2481-2483), grid-stride over the whole field      (one thread/element)
// The twelve lookup tables (VENTR1, VENTR2, ACCRR, MASSR, VRAIN, RRATE, VENTI1, VENTI2, ACCRI,
// MASSI, VSNOWI, VEL_RF) are READ from ETAMPNEW_DATA.expanded_rain at Fortran 2972-2983. That is
// host file I/O, done once by the wrapper (namgpu/physics/fer_hires/phys_fer_init.py), which
// uploads the arrays; no kernel here computes them.
//
// Index conventions. Fortran MASSR(50:1000) -> float[k], k = M - 50 (951 words). Fortran
// VEL_RF(2:9, 0:40) -> float[n*8 + (i-2)] (column-major flattening, shape (41, 8) in C order).
// Fortran MP_RESTART_STATE(1:43) -> float[0..42] in the Fortran order: growth 0..34, then
// C1XPVS0, C2XPVS0, C1XPVS, C2XPVS, CIACW, CIACR, CRACW, BRAUT.
// Scalars go to sc[] in the enum order below; the wrapper mirrors the names.
//
// Arithmetic: single precision throughout (kfpt = single). Fortran real**real is powf; 1.1**Nrime
// and the (1.E-3)**3 style integer powers are written as products. Fast numerics, so there is
// no bitwise claim for this unit.
//
// Dropped on purpose: the MPI broadcast of the tables, the ESMF abort path on an unused unit
// number, and the DX/DY pair (2380-2418), which feeds only the commented RHgrd line and produces no
// output. The module's RHgrd depends on DX, but the RHgrd is set in the caller, not here.

#include "nmmb_common.cuh"

#define FER_NX     7501      // NX (module, line 148)
#define FER_NTAB   951       // MDRmin:MDRmax == MDImin:MDImax == 50:1000 (lines 116-140)
#define FER_MDMIN  50        // MDRmin = MDImin = 1.e6*0.05e-3, INTEGER -> 50
#define FER_MY_N   35        // MY_T1:MY_T2 (line 127)
#define FER_NRIME  40        // Nrime (line 145)
#define FER_NMP    43        // MP_RESTART_STATE words written by FERRIER_INIT_hr (2720-2729)

#define FER_XMIN   180.0f    // XMIN (line 149)
#define FER_XMAX   330.0f    // XMAX (line 149)
#define FER_PI     3.141592653589793f   // module_CONSTANTS pi, kfpt = single
#define FER_RHOL   1000.0f   // module_CONSTANTS rhowater
#define FER_T0C    273.15f   // module_CONSTANTS tiw (T0C)
#define FER_RV     461.6f    // module_CONSTANTS r_v
#define FER_N0R0   8.E6f     // N0r0 (line 196)
#define FER_NCW    300.E6f   // NCW (line 198)
#define FER_C1     (1.0f/3.0f)   // C1 (line 161)
#define FER_DMRMIN 0.05E-3f  // DMRmin (line 138)
#define FER_DMRMAX 1.0E-3f   // DMRmax (line 138)

// Scalar slots written by phys_fer_init_scalars. The wrapper SCALAR_SLOTS list has the same order.
enum {
    S_C1XPVS = 0, S_C2XPVS, S_C1XPVS0, S_C2XPVS0, S_PI, S_ABFR, S_CBFR, S_CIACW, S_CIACR,
    S_CRACW, S_BRAUT, S_RQR_DRMIN, S_RQR_DRMAX, S_RR_DRMIN, S_RR_DR1, S_RR_DR2, S_RR_DR3,
    S_RR_DR4, S_RR_DR5, S_RR_DRMAX, S_C_NR, S_CRAIN, S_C_N0R0, S_CN0R0, S_CN0R_DMRMIN,
    S_CN0R_DMRMAX, S_ESW0, S_RFMAX, S_RFMX1, S_BETA6, S_ARAUT, S_QAUT0, S_ARCW, S_RH_NGC,
    S_RQHAIL, S_BVHAIL, S_AVHAIL, S_RH_NGT, S_DTPH, S_NSC
};

// FPVSX (2915-2971): exact saturation vapour pressure (kPa). Its own PARAMETERs: TTP, HVAP, PSAT,
// CLIQ, CVAP, CICE, HSUB. RV is the module constant.
__device__ float fer_fpvsx(float T)
{
    const float TTP = 2.7316E+2f, HVAP = 2.5000E+6f, PSAT = 6.1078E+2f;
    const float CLIQ = 4.1855E+3f, CVAP = 1.8460E+3f, CICE = 2.1060E+3f, HSUB = 2.8340E+6f;
    const float PSATK = PSAT * 1.E-3f;
    const float DLDT = CVAP - CLIQ;
    const float XA = -DLDT / FER_RV;
    const float XB = XA + HVAP / (FER_RV * TTP);
    const float DLDTI = CVAP - CICE;
    const float XAI = -DLDTI / FER_RV;
    const float XBI = XAI + HSUB / (FER_RV * TTP);
    const float TR = TTP / T;
    if (T >= TTP) {
        return PSATK * powf(TR, XA) * expf(XB * (1.f - TR));
    }
    return PSATK * powf(TR, XAI) * expf(XBI * (1.f - TR));
}

// FPVSX0 (2974-2989): water only, all temperatures.
__device__ float fer_fpvsx0(float T)
{
    const float TTP = 2.7316E+2f, HVAP = 2.5000E+6f, PSAT = 6.1078E+2f;
    const float CLIQ = 4.1855E+3f, CVAP = 1.8460E+3f;
    const float PSATK = PSAT * 1.E-3f;
    const float DLDT = CVAP - CLIQ;
    const float XA = -DLDT / FER_RV;
    const float XB = XA + HVAP / (FER_RV * TTP);
    const float TR = TTP / T;
    return PSATK * powf(TR, XA) * expf(XB * (1.f - TR));
}

// GPVS_hr (2791-2836): TBPVS(JX) = FPVSX(X), TBPVS0(JX) = FPVSX0(X), X = XMIN + (JX-1)*XINC.
// Thread t holds JX = t+1.
extern "C" __global__
void phys_fer_init_gpvs(float* __restrict__ tbpvs, float* __restrict__ tbpvs0)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= FER_NX) return;
    const float XINC = (FER_XMAX - FER_XMIN) / (float)(FER_NX - 1);
    const float X = FER_XMIN + (float)t * XINC;
    tbpvs[t] = fer_fpvsx(X);
    tbpvs0[t] = fer_fpvsx0(X);
}

// SDENS(I) = PI*1.5E-15*FLOAT(I*I*I)/MASSI(I) (2714), for I = MDImin + k. massi is MASSI(MDImin:MDImax),
// index k = I - 50. Inside the ALLOWED_TO_READ block, so it runs for every cold start.
extern "C" __global__
void phys_fer_init_sdens(const float* __restrict__ massi, float* __restrict__ sdens)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= FER_NTAB) return;
    const int I = FER_MDMIN + k;
    sdens[k] = FER_PI * 1.5E-15f * (float)(I * I * I) / massi[k];
}

// The SAVE scalars of FERRIER_INIT_hr in Fortran order (2594-2690 region), ESW0 (2637) from the
// TBPVS0 table written by phys_fer_init_gpvs (stream order makes that safe), and the
// MP_RESTART_STATE(1:43) (2720-2729) from the growth table. One thread; the work is scalar.
//
// rrate, massr: index M - 50 (951 words each). dtph = GSMDT = NPRECIP*DT (set by the caller at
// module_SOLVER_GRID_COMP.F90:10993-10997).
// write_state: 1 on a cold start (Fortran IF(.NOT.RESTART) at 2719), 0 on a restart. On a restart
// the scalars are still computed (they are not guarded in Fortran, 2494-2733), but MP_RESTART_STATE
// is left untouched: it holds the values the restart file supplies.
extern "C" __global__
void phys_fer_init_scalars(const float* __restrict__ rrate, const float* __restrict__ massr,
                           const float* __restrict__ tbpvs0, const float* __restrict__ growth,
                           float* __restrict__ sc, float* __restrict__ mp_restart, float dtph,
                           int write_state)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const float PI = FER_PI;
    const float RHOL = FER_RHOL;
    const float N0r0 = FER_N0R0;
    const float NCW = FER_NCW;
    const float C1 = FER_C1;
    const float DTPH = dtph;

    // MDR table indices (Fortran PARAMETERs, INT(1.e6*DMR); checked: 50, 100, 200, 320, 450, 670, 1000).
    const int MDRmin = 50, MDR1 = 100, MDR2 = 200, MDR3 = 320, MDR4 = 450, MDR5 = 670, MDRmax = 1000;

    // GPVS_hr (2818-2822): C1XPVS = 1 - XMIN/XINC, C2XPVS = 1/XINC; C1XPVS0/C2XPVS0 are the same values.
    const float XINC = (FER_XMAX - FER_XMIN) / (float)(FER_NX - 1);
    const float C1XPVS = 1.f - FER_XMIN / XINC;
    const float C2XPVS = 1.f / XINC;
    sc[S_C1XPVS] = C1XPVS;
    sc[S_C2XPVS] = C2XPVS;
    sc[S_C1XPVS0] = C1XPVS;
    sc[S_C2XPVS0] = C2XPVS;

    sc[S_PI] = PI;
    const float ABFR = -0.66f;                            // 2594
    sc[S_ABFR] = ABFR;
    const float BBFR = 100.f;                             // 2595, used only in CBFR
    // CBFR = 20.*PI*PI*BBFR*RHOL*1.E-21*DTPH (2596), left to right.
    const float CBFR = 20.f * PI * PI * BBFR * RHOL * 1.E-21f * DTPH;
    sc[S_CBFR] = CBFR;
    // CIACW = 0.5*DTPH*0.25*PI; CIACR = PI*DTPH (the two CIA lines of the routine body).
    const float CIACW = 0.5f * DTPH * 0.25f * PI;
    const float CIACR = PI * DTPH;
    sc[S_CIACW] = CIACW;
    sc[S_CIACR] = CIACR;

    // RR_DR* and RQR_DR* (the drop-diameter block of the routine body).
    sc[S_RR_DRMIN] = N0r0 * rrate[MDRmin - FER_MDMIN];
    sc[S_RR_DR1] = N0r0 * rrate[MDR1 - FER_MDMIN];
    sc[S_RR_DR2] = N0r0 * rrate[MDR2 - FER_MDMIN];
    sc[S_RR_DR3] = N0r0 * rrate[MDR3 - FER_MDMIN];
    sc[S_RR_DR4] = N0r0 * rrate[MDR4 - FER_MDMIN];
    sc[S_RR_DR5] = N0r0 * rrate[MDR5 - FER_MDMIN];
    sc[S_RR_DRMAX] = N0r0 * rrate[MDRmax - FER_MDMIN];
    sc[S_RQR_DRMIN] = N0r0 * massr[MDRmin - FER_MDMIN];
    sc[S_RQR_DRMAX] = N0r0 * massr[MDRmax - FER_MDMIN];

    const float C_NR = 1.f / (PI * RHOL);
    sc[S_C_NR] = C_NR;
    sc[S_CRAIN] = 720.E18f * C_NR * C_NR;
    const float C_N0r0 = PI * RHOL * N0r0;
    sc[S_C_N0R0] = C_N0r0;
    sc[S_CN0R0] = 1.E6f / sqrtf(sqrtf(C_N0r0));
    // CN0r_DMRmin and CN0r_DMRmax: 1/(PI*RHOL*DMR*DMR*DMR*DMR) (2629-2630), the product taken
    // left to right exactly as the Fortran expression is written.
    sc[S_CN0R_DMRMIN] = 1.f / (PI * RHOL * FER_DMRMIN * FER_DMRMIN * FER_DMRMIN * FER_DMRMIN);
    sc[S_CN0R_DMRMAX] = 1.f / (PI * RHOL * FER_DMRMAX * FER_DMRMAX * FER_DMRMAX * FER_DMRMAX);

    // CRACW = DTPH*0.25*PI*1.0.
    const float CRACW = DTPH * 0.25f * PI * 1.0f;
    sc[S_CRACW] = CRACW;

    // ESW0 = 1000.*FPVS0(T0C) (2637). FPVS0 is the table lookup on TBPVS0 (2894-2911).
    {
        const float XJ1 = fminf(fmaxf(C1XPVS + C2XPVS * FER_T0C, 1.f), (float)FER_NX);
        const int JX1 = (int)fminf(XJ1, (float)(FER_NX - 1));     // INTEGER assignment truncates
        const float v = tbpvs0[JX1 - 1] + (XJ1 - (float)JX1) * (tbpvs0[JX1] - tbpvs0[JX1 - 1]);
        sc[S_ESW0] = 1000.f * v;
    }
    // RFmax = 1.1**Nrime (2638, real**integer). gfortran lowers an integer power to libgcc's
    // __powisf2, which squares by binary decomposition of the exponent (not a running product);
    // this reproduces that order. RFmx1 = PI*900.
    {
        float x = 1.1f;
        unsigned int n = (unsigned int)FER_NRIME;
        float y = (n % 2u) ? x : 1.f;
        while (n >>= 1) {
            x = x * x;
            if (n % 2u) y = y * x;
        }
        sc[S_RFMAX] = y;
    }
    sc[S_RFMX1] = PI * 900.f;

    // BETA6 (RDIS = 0.5): ((1+3RDIS^2)(1+4RDIS^2)(1+5RDIS^2)) / ((1+RDIS^2)(1+2RDIS^2)).
    const float RDIS = 0.5f;
    const float BETA6 = ((1.f + 3.f * RDIS * RDIS) * (1.f + 4.f * RDIS * RDIS) * (1.f + 5.f * RDIS * RDIS)
                        / ((1.f + RDIS * RDIS) * (1.f + 2.f * RDIS * RDIS)));
    sc[S_BETA6] = BETA6;
    // ARAUT = 1.03e19/(NCW*SQRT(NCW)); BRAUT = DTPH*1.1E10*BETA6/NCW.
    const float ARAUT = 1.03e19f / (NCW * sqrtf(NCW));
    const float BRAUT = DTPH * 1.1E10f * BETA6 / NCW;
    sc[S_ARAUT] = ARAUT;
    sc[S_BRAUT] = BRAUT;
    // QAUT0 = PI*RHOL*NCW*(20.E-6)**3/6. (legacy, passed to radiation).
    const float c20 = 20.E-6f;
    sc[S_QAUT0] = PI * RHOL * NCW * (c20 * c20 * c20) / 6.f;
    // ARcw = 1.E6*(0.75/(PI*NCW*RHOL))**C1.
    sc[S_ARCW] = 1.E6f * powf(0.75f / (PI * NCW * RHOL), C1);

    // Hail and graupel thresholds.
    float RH_NgC = PI * 500.f * 1.E3f;
    const float RQHAIL = RH_NgC * (1.E-3f * 1.E-3f * 1.E-3f);      // (1.E-3)**3
    const float BVHAIL = 0.82f * C1;
    const float AVHAIL = 1353.f * powf(1.f / RH_NgC, BVHAIL);      // uses RH_NgC before its reset
    RH_NgC = 1.E6f * powf(1.f / RH_NgC, C1);
    float RH_NgT = PI * 300.f * 5.E3f;
    RH_NgT = 1.E6f * powf(1.f / RH_NgT, C1);
    sc[S_RQHAIL] = RQHAIL;
    sc[S_BVHAIL] = BVHAIL;
    sc[S_AVHAIL] = AVHAIL;
    sc[S_RH_NGC] = RH_NgC;
    sc[S_RH_NGT] = RH_NgT;
    sc[S_DTPH] = DTPH;

    // MP_RESTART_STATE(1:43) (2720-2729, cold start only, IF(.NOT.RESTART)): growth into 1..35,
    // then the eight scalars.
    if (write_state) {
        for (int t = 0; t < FER_MY_N; ++t) mp_restart[t] = growth[t];
        mp_restart[FER_MY_N + 0] = C1XPVS;     // C1XPVS0
        mp_restart[FER_MY_N + 1] = C2XPVS;     // C2XPVS0
        mp_restart[FER_MY_N + 2] = C1XPVS;     // C1XPVS
        mp_restart[FER_MY_N + 3] = C2XPVS;     // C2XPVS
        mp_restart[FER_MY_N + 4] = CIACW;
        mp_restart[FER_MY_N + 5] = CIACR;
        mp_restart[FER_MY_N + 6] = CRACW;
        mp_restart[FER_MY_N + 7] = BRAUT;
    }
}

// The not-RESTART block of FERRIER_INIT_hr (2481-2483): F_ICE_PHY = 0, F_RAIN_PHY = 0,
// F_RIMEF_PHY = 1 over the tile ITS:ITE, JTS:JTE, KTS:KTE (1-based, inclusive, as the caller
// passes them). Fields are (k, j, i) with i fastest: index = ((k-1)*jm + (j-1))*im + (i-1).
// Grid-stride over the tile; one thread per element.
extern "C" __global__
void phys_fer_init_fill(float* __restrict__ f_ice, float* __restrict__ f_rain,
                        float* __restrict__ f_rimef, int im, int jm,
                        int its, int ite, int jts, int jte, int kts, int kte)
{
    const int ni = ite - its + 1, nj = jte - jts + 1, nk = kte - kts + 1;
    const long long n = (long long)ni * nj * nk;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (long long)gridDim.x * blockDim.x) {
        const int i = its + (int)(t % ni);
        const int j = jts + (int)((t / ni) % nj);
        const int k = kts + (int)(t / ((long long)ni * nj));
        const long long idx = ((long long)(k - 1) * jm + (j - 1)) * im + (i - 1);
        f_ice[idx] = 0.f;
        f_rain[idx] = 0.f;
        f_rimef[idx] = 1.f;
    }
}
