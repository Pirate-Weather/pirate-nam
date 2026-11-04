// phys_fer_col_melt_evap.cu -- WOOF-NAM unit phys-fer-col-melt-evap.
//
// NOTICE: NOAA/NCEP NAM source module_MP_FER_HIRES.F90 (NCEP/EMC, "Modified fer_hires microphysics";
// US Government work, no licence file in the NAM clone). Module changelog lines 1-82 are the notice,
// as kept in fer_hires.cuh.
//
// Ported to CUDA for WOOF-NAM from phys/module_MP_FER_HIRES.F90:1479-1587 (nam.v4.2.11, 83df4e13c1).
//
// Fortran line map (every statement has a counterpart below):
//   1479-1483  comments only                                   -> none
//   1484-1485  SFACTOR = sqrt(VEL_INC)*(RHO/(DIFFUS^2*DYNVIS))**C2   -> melt block
//   1486-1490  VENTIL, AIEVP, DIEVP (Xratio switch)            -> melt block
//   1492-1503  evaporation from melting snow / shedding (PIEVP, PICND), WV<QSW .AND. DUM<=EPSQ branch
//   1505       PIMLT = THERM_COND*TCC*VENTIL*RRHO*DTPH/XLF     -> melt block
//   1508-1510  melt limit by DUM1 (0C crossing)                -> melt block
//   1512-1518  loss limit of snow (PILOSS) for melt+evap        -> melt block
//   1520       ENDIF of the melt block (TC>0, TCC>0, ICE)       -> melt block
//   1521-1534  TOT_RAIN, PRLOSS, QTRAIN, RQR, STRAT/DRZL flip   -> rain block
//   1536-1587  rain size distribution (DSD1: N0r, INDEXR, VRAIN2), PIACR for TC<T_ICE,
//              and the head of the DWVr branch. The body of that branch starts at Fortran
//              line 1588 and belongs to unit phys-fer-col-rain-sedim (it also closes the rain block).
//
// Per-level kernel: one thread per (i, j, l). The level-to-level carries (ARAIN and VRAIN1, the rain
// falling in from above) are inputs here; the column kernel of phys-fer-fer-hires supplies them.
// No host sync, no printf, no device allocation. Fast numerics (fmad on).
//
// Constants: module_MP_FER_HIRES parameters (XLF, XLV1, XLF1, EPSQ, EP_2, PI, RHOL, N0r0, C2, Xratio,
// T_ICE, RHO0) live in fer_hires.cuh and this file; runtime scalars set by FERRIER_INIT_hr (ESW0,
// CN0r0, RQR_DRmin, RQR_DRmax, DTPH) are kernel arguments; the MASSR and VRAIN lookup tables are
// kernel pointers (index = m - FER_MDR_MIN; FER_MDI_MIN for VENTI1/VENTI2).
#include "nmmb_common.cuh"
#include "fer_hires.cuh"

#define FER_MELT_XRATIO   0.025f     // Fortran Xratio (PARAMETER, line 790)
#define FER_MELT_RHO0     1.194f     // Fortran RHO0 (PARAMETER, line 789)
#define FER_MELT_BLEND    1.0f       // Fortran BLEND (PARAMETER, line 802)
#define FER_MELT_C2       (1.0f / 6.0f)

// Per-cell state of one level, read from and written back to the column arrays.
struct FerMeltRainCell {
    // inputs
    float VEL_INC, RHO, DIFFUS, DYNVIS, NLICE, QW, PCOND, WV, QSW, TC, TCC, PP, PILOSS, THERM_COND;
    float THICK, QR, ARAIN, VRAIN1;
    int   INDEXS, ICE, RAIN, INDEXRmin;
    // in/out (set earlier in the level loop, updated here)
    int   STRAT, DRZL;
    float PIEVP, PICND, PIMLT, PIACR;
    int   INDEXR;
    float N0r, VRAIN2;
    // outputs
    float PRLOSS, QTRAIN, RQR;
};

// Melting and evaporation (Fortran 1484-1520) followed by the rain block head (1521-1587).
// ventI1/ventI2 are indexed by INDEXS - FER_MDI_MIN; massr/vrain by INDEXR - FER_MDR_MIN.
__device__ inline void nam_fer_melt_evap_rain(FerMeltRainCell& s, float DTPH, float ESW0,
                                              float CN0r0, float RQR_DRmin, float RQR_DRmax,
                                              const float* __restrict__ ventI1,
                                              const float* __restrict__ ventI2,
                                              const float* __restrict__ massr,
                                              const float* __restrict__ vrain)
{
    const float EPSQ = FER_EPSQ;
    const float EPS = FER_EP2;              // EPS => EP_2 (line 113)
    const float RRHO = 1.0f / s.RHO;        // line 968
    float DUM, DUM1;

    // ---- Melting and evaporation/condensation (1479-1520) ----
    // QSW0 and DWV0 are zeroed before this block (1475-1476); they are block-local here.
    if (s.TC > 0.0f && s.TCC > 0.0f && s.ICE) {
        // 1484-1485: SFACTOR (units s**.5/m)
        const float SFACTOR = sqrtf(s.VEL_INC) *
                              powf(s.RHO / (s.DIFFUS * s.DIFFUS * s.DYNVIS), FER_MELT_C2);
        // 1486: VENTIL (units m**-2)
        const int idx = s.INDEXS - FER_MDI_MIN;
        const float VENTIL = s.NLICE * (ventI1[idx] + SFACTOR * ventI2[idx]);
        // 1487: AIEVP
        const float AIEVP = VENTIL * s.DIFFUS * DTPH;
        // 1488-1491: DIEVP
        float DIEVP;
        if (AIEVP < FER_MELT_XRATIO) {
            DIEVP = AIEVP;
        } else {
            DIEVP = 1.0f - expf(-AIEVP);
        }
        DUM = s.QW + s.PCOND;                                  // 1492
        if (s.WV < s.QSW && DUM <= EPSQ) {                     // 1493
            // 1497-1498: limit ESW0 at low pressure
            DUM = fminf(ESW0, 0.99f * s.PP);
            // 1499: QSW0 >= EPSQ
            const float QSW0 = fmaxf(EPSQ, EPS * DUM / (s.PP - DUM));
            // 1500
            const float DWV0 = fminf(s.WV, s.QSW) - QSW0;
            // 1501
            DUM = DWV0 * DIEVP;
            // 1502-1503
            s.PIEVP = fmaxf(fminf(0.0f, DUM), s.PILOSS);
            s.PICND = fmaxf(0.0f, DUM);
        }
        // 1505: PIMLT (melting, kg/kg per step)
        s.PIMLT = s.THERM_COND * s.TCC * VENTIL * RRHO * DTPH / FER_XLF;
        // 1508-1510: limit melting to prevent temperature oscillations across 0C
        DUM1 = fmaxf(0.0f, (s.TCC + FER_XLV1 * s.PIEVP) / FER_XLF1);
        s.PIMLT = fminf(s.PIMLT, DUM1);
        // 1512-1518: limit loss of snow by melting and evaporation
        DUM = s.PIEVP - s.PIMLT;
        if (DUM < s.PILOSS) {
            DUM1 = s.PILOSS / DUM;
            s.PIMLT = s.PIMLT * DUM1;
            s.PIEVP = s.PIEVP * DUM1;
        }
    }

    // ---- Rain block head (1521-1587) ----
    // 1521-1524: TOT_RAIN, QTRAIN, PRLOSS, RQR are zeroed before the rain IF.
    float TOT_RAIN = 0.0f;
    s.QTRAIN = 0.0f;
    s.PRLOSS = 0.0f;
    s.RQR = 0.0f;

    if (s.RAIN) {                                              // do_rain
        // 1527: TOT_RAIN = THICK*QR + BLEND*ARAIN
        TOT_RAIN = s.THICK * s.QR + FER_MELT_BLEND * s.ARAIN;
        // 1528: PRLOSS
        s.PRLOSS = -TOT_RAIN / s.THICK;
        // 1530: BLDTRH = BLEND*DTRHO, DTRHO = DTPH*RHO (lines 969-970)
        const float DTRHO = DTPH * s.RHO;
        const float BLDTRH = FER_MELT_BLEND * DTRHO;
        // 1531: QTRAIN, the time-averaged rain mixing ratio
        s.QTRAIN = TOT_RAIN / (s.THICK + BLDTRH * s.VRAIN1);
        // 1532: RQR (rain content kg/m**3)
        s.RQR = s.RHO * s.QTRAIN;
        // 1533-1534: stratiform and drizzle flags
        if (s.STRAT && s.RQR > 1.0e-3f) s.STRAT = 0;
        if (s.DRZL && s.PIMLT > EPSQ) s.DRZL = 0;

        // 1536-1581: rain size distribution (DSD1)
        if (s.RQR <= RQR_DRmin) {
            // extremely light rain
            s.INDEXR = FER_MDR_MIN;
            s.N0r = s.RQR / massr[s.INDEXR - FER_MDR_MIN];
        } else if (s.DRZL) {
            // drizzle: gradually increase N0r for rain contents < 0.5 g m^-3
            DUM = fmaxf(1.0f, 0.5e-3f / s.RQR);
            s.N0r = fminf(1.0e9f, FER_N0R0 * DUM * sqrtf(DUM));   // 8.e6 <= N0r <= 1.e9
            DUM1 = s.RQR / (FER_PI * FER_RHOL * s.N0r);
            DUM = 1.0e6f * sqrtf(sqrtf(DUM1));
            s.INDEXR = (int)fmaxf(FER_XMR_MIN, fminf(DUM, FER_XMR_MAX));
            s.N0r = s.RQR / massr[s.INDEXR - FER_MDR_MIN];
        } else if (s.RQR >= RQR_DRmax) {
            // extremely heavy rain
            s.INDEXR = FER_MDR_MAX;
            s.N0r = s.RQR / massr[s.INDEXR - FER_MDR_MIN];
        } else {
            // light to heavy rain
            s.N0r = FER_N0R0;
            DUM = CN0r0 * sqrtf(sqrtf(s.RQR));
            s.INDEXR = (int)fmaxf(FER_XMR_MIN, fminf(DUM, FER_XMR_MAX));
            if (s.STRAT && s.INDEXR < s.INDEXRmin) {
                // stratiform rain
                s.INDEXR = s.INDEXRmin;
                s.N0r = s.RQR / massr[s.INDEXR - FER_MDR_MIN];
            }
        }
        // VRAIN2 = GAMMAR*VRAIN(INDEXR), GAMMAR = (RHO0/RHO)**.4 (line 1056 formula)
        const float GAMMAR = powf(FER_MELT_RHO0 / s.RHO, 0.4f);
        s.VRAIN2 = GAMMAR * vrain[s.INDEXR - FER_MDR_MIN];

        // 1583-1584: freezing of rain onto large ice (TC < T_ICE)
        // 1586 (TC >= T_ICE): the DWVr branch (rain evaporation PREVP, accretion PRACW, Fortran 1588-1608)
        // is owned by phys-fer-col-rain-sedim, which reads WV, PCOND, QSW, QW and the table lookups
        // (VENTR1/VENTR2/ACCRR) that this unit does not bind. Nothing here, by design, so no evaporation
        // is applied twice.
        if (s.TC < FER_T_ICE) {
            s.PIACR = -s.PRLOSS;
        }
        // End of do_rain: the Fortran block continues in phys-fer-col-rain-sedim.
    }
}

// One thread per (i, j, l). Arrays are [LM, JM, IM] float32 (int32 for flags, INDEXS, INDEXR).
extern "C" __global__ void phys_fer_col_melt_evap(
    const int IM, const int JM, const int LM,
    const float DTPH, const float ESW0, const float CN0r0,
    const float RQR_DRmin, const float RQR_DRmax,
    const float* __restrict__ ventI1, const float* __restrict__ ventI2,
    const float* __restrict__ massr, const float* __restrict__ vrain,
    const float* __restrict__ VEL_INC, const float* __restrict__ RHO,
    const float* __restrict__ DIFFUS, const float* __restrict__ DYNVIS,
    const float* __restrict__ NLICE, const float* __restrict__ QW,
    const float* __restrict__ PCOND, const float* __restrict__ WV,
    const float* __restrict__ QSW, const float* __restrict__ TC,
    const float* __restrict__ TCC, const float* __restrict__ PP,
    const float* __restrict__ PILOSS, const float* __restrict__ THERM_COND,
    const float* __restrict__ THICK, const float* __restrict__ QR,
    const float* __restrict__ ARAIN, const float* __restrict__ VRAIN1,
    const int* __restrict__ INDEXS, const int* __restrict__ ICE,
    const int* __restrict__ RAIN, const int* __restrict__ INDEXRmin,
    int* __restrict__ STRAT, int* __restrict__ DRZL,
    float* __restrict__ PIEVP, float* __restrict__ PICND,
    float* __restrict__ PIMLT, float* __restrict__ PIACR,
    int* __restrict__ INDEXR, float* __restrict__ N0r, float* __restrict__ VRAIN2,
    float* __restrict__ PRLOSS, float* __restrict__ QTRAIN, float* __restrict__ RQR)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t p = IDX3(i0, j0, l0, IM, JM);

    FerMeltRainCell s;
    s.VEL_INC = VEL_INC[p];   s.RHO = RHO[p];       s.DIFFUS = DIFFUS[p];   s.DYNVIS = DYNVIS[p];
    s.NLICE = NLICE[p];       s.QW = QW[p];         s.PCOND = PCOND[p];     s.WV = WV[p];
    s.QSW = QSW[p];           s.TC = TC[p];         s.TCC = TCC[p];         s.PP = PP[p];
    s.PILOSS = PILOSS[p];     s.THERM_COND = THERM_COND[p];
    s.THICK = THICK[p];       s.QR = QR[p];         s.ARAIN = ARAIN[p];     s.VRAIN1 = VRAIN1[p];
    s.INDEXS = INDEXS[p];     s.ICE = ICE[p];       s.RAIN = RAIN[p];       s.INDEXRmin = INDEXRmin[p];
    s.STRAT = STRAT[p];       s.DRZL = DRZL[p];
    s.PIEVP = PIEVP[p];       s.PICND = PICND[p];   s.PIMLT = PIMLT[p];     s.PIACR = PIACR[p];
    s.INDEXR = INDEXR[p];     s.N0r = N0r[p];       s.VRAIN2 = VRAIN2[p];
    s.PRLOSS = 0.0f;          s.QTRAIN = 0.0f;      s.RQR = 0.0f;

    nam_fer_melt_evap_rain(s, DTPH, ESW0, CN0r0, RQR_DRmin, RQR_DRmax, ventI1, ventI2, massr, vrain);

    STRAT[p] = s.STRAT;       DRZL[p] = s.DRZL;
    PIEVP[p] = s.PIEVP;       PICND[p] = s.PICND;   PIMLT[p] = s.PIMLT;     PIACR[p] = s.PIACR;
    INDEXR[p] = s.INDEXR;     N0r[p] = s.N0r;       VRAIN2[p] = s.VRAIN2;
    PRLOSS[p] = s.PRLOSS;     QTRAIN[p] = s.QTRAIN; RQR[p] = s.RQR;
}
