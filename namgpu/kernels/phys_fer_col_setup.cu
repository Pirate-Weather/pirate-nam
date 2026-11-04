// phys_fer_col_setup.cu -- nam-gpu unit phys-fer-col-setup: EGCP01COLUMN_hr, column setup and the
// top-down level setup (Fortran lines 650-910).
//
// NOTICE: NOAA/NCEP NAM source module_MP_FER_HIRES.F90 (US government work, no licence file in the
// NAM clone). Module header kept as the notice (comment-prefixed):
//   MODULE MODULE_MP_FER_HIRES (Ferrier-Aligo high-resolution microphysics, nam.v4.2.11)
//   "Modified" fer_hires microphysics - 11 July 2016 version
//   SUBPROGRAM EGCP01COLUMN_hr (Ferrier, Jin; WRF structure modification).
// Ported to CUDA for nam-gpu from phys/module_MP_FER_HIRES.F90:650-910 (nam.v4.2.11, 83df4e13c1).
//
// Scope of this unit: the column initialisation (lines 843-853) and the first part of the top-down
// loop DO L=1,LSFC (lines 856-910): zeroing of the per-level outputs, the rain-from-above and
// snow-from-above checks, and TK/TC. Everything from line 911 on (satcheck, ice properties,
// processes, melt, rain, sedimentation) is a later unit. The column driver (phys-fer-fer-hires)
// calls fer_col_setup_begin once per column and fer_col_setup_level per level L = 1..LSFC (top to
// bottom), and carries the column state (ARAIN, ASNOW, VRAIN1, VSNOW1, TIMLT, INDEXS0C, STRAT,
// DRZL, RHOX0C) in registers between levels. No host loop, no host sync.
//
// Outputs this unit hands on (all per level, [LM, JM, IM] float32, levels 1..LSFC are the column):
//   TK_col, TC_col  TK and TC of the level (K and C).  TK_col is T at that level.
//   CLEAR_col       CLEAR flag of the level (1.0 = clear, 0.0 = condensate or precipitation from above).
// Levels LSFC+1..LM are written with defined defaults on every call (zero, DBZ_col = DBZmin, CLEAR_col =
// 0), so no scratch level keeps a value from an earlier call.
//
// Layout (CONTRACT.md): every per-level column array of the Fortran (KTS:KTE) is a [LM, JM, IM]
// float32 array, so level l0 of column (i0, j0) is at IDX3(i0, j0, l0, IM, JM) =
// l0 * (JM*IM) + IDX2(i0, j0, IM). The kernel takes the column offset and uses stride = JM*IM.
//
// Fortran-to-CUDA (every statement of 650-910 that this unit owns):
//   ARAIN=0.; VRAIN1=0.; VSNOW1=0.; ASNOW=0.          -> fer_col_setup_begin: carry.arain/vrain1/vsnow1/asnow = 0
//   INDEXS0C=MDImin; RHOX0C=22.5; TIMLT=0.            -> carry.indexs0c, carry.rhox0c, carry.timlt
//   STRAT=.FALSE.; DRZL=.FALSE.                       -> carry.strat, carry.drzl
//   big_loop: DO L=1,LSFC                             -> kernel loop l0 = 0 .. LSFC-1
//   pcond1d(L)..RFlag1d(L) = 0 (22 per-level arrays, Fortran order)
//   DBZ_col(L)=DBZmin; NR_col(L)=0; NS_col(L)=0       -> o.DBZ_col, o.NR_col, o.NS_col
//   IF (ARAIN .GT. CLIMIT) ... ELSE ... ARAIN=0       -> clear, carry.vrain1, carry.arain
//   IF (ASNOW .GT. CLIMIT) ... ELSE ASNOW=0           -> clear, carry.vsnow1, carry.asnow
//   TK=T_col(L); TC=TK-T0C                            -> o.TK_col, o.TC_col (and the return values)
// Constants: CLIMIT = 10.*EPSQ, EPSQ = 1.e-12 (module_CONSTANTS NMMB_EPSQ) -> 1.0e-11f.
//            MDImin = XMImin = 1.e6*DMImin, DMImin = .05e-3 -> 50 (INTEGER parameter, truncated).
//            DBZmin = -20. T0C = TIW = 273.15 (NMMB_TIW).
// All REAL state is single precision (kfpt = single): float everywhere.
// Not in this unit's lines: BLEND, PRINT_diag and the WARN/DBG flags (diagnostic only); the
// arrays P_col, QI_col, QR_col, Q_col, QW_col, RimeF_col, THICK_col, WC_col are first touched
// beyond line 910 and are handled by the later units.
//
// Entry state of the carry: ARAIN and ASNOW are zeroed at Fortran 844 and 847 before the loop, so the
// rain and snow branches are entered with zero carry at level 1. Within lines 650-910 nothing raises
// them; the body after line 910 (later units) does. The branches are therefore exercised by
// phys_fer_col_setup_probe, which runs fer_col_setup_level on injected carry states.

#include "nmmb_common.cuh"

#define FER_CLIMIT    1.0e-11f
#define FER_MDIMIN    50
#define FER_DBZMIN    (-20.0f)
#define FER_RHOX0C    22.5f
#define FER_T0C       NMMB_TIW

// Column carry-state of EGCP01COLUMN_hr that survives from one level to the next.
struct FerColCarry {
    float arain;      // ARAIN    accumulated rain falling into the layer from above (kg m-2)
    float asnow;      // ASNOW    accumulated snow falling into the layer from above (kg m-2)
    float vrain1;     // VRAIN1   rain fall speed into the layer from above (m s-1)
    float vsnow1;     // VSNOW1   ice fall speed into the layer from above (m s-1)
    float timlt;      // TIMLT    total ice melting in a layer (drizzle detection)
    float rhox0c;     // RHOX0C   estimated ice density at 0 C (kg m-3)
    int   indexs0c;   // INDEXS0C mean snow/graupel diameter just above the 0 C level (microns)
    bool  strat;      // STRAT    stratiform rain DSD below the melting level
    bool  drzl;       // DRZL     drizzle DSD below the melting level
};

// Per-level output views for one column: the element of level l0 is p[l0 * stride].
struct FerColLevelOut {
    float *pcond1d, *pidep1d, *piacw1d, *piacwi1d, *piacwr1d, *piacr1d, *picnd1d, *pievp1d,
          *pimlt1d, *praut1d, *pracw1d, *prevp1d, *pisub1d, *pevap1d, *vsnow1d, *vrain11d,
          *vrain21d, *vci1d, *NSmICE1d, *INDEXR1d, *INDEXS1d, *RFlag1d;
    float *DBZ_col, *NR_col, *NS_col;
    float *TK_col, *TC_col, *CLEAR_col;
    size_t stride;
};

// Lines 843-853: column initialisation, before the level loop.
__device__ __forceinline__ void fer_col_setup_begin(FerColCarry& c)
{
    c.arain    = 0.0f;
    c.vrain1   = 0.0f;
    c.vsnow1   = 0.0f;
    c.asnow    = 0.0f;
    c.indexs0c = FER_MDIMIN;
    c.rhox0c   = FER_RHOX0C;
    c.timlt    = 0.0f;
    c.strat    = false;
    c.drzl     = false;
}

// Lines 857-886 (per-level zeroing, Fortran order), plus the defined initial values of DBZ_col and the
// TK/TC/CLEAR outputs. Used at the top of every active level and for every level above LSFC.
__device__ __forceinline__ void fer_col_zero_outputs(int l0, const FerColLevelOut& o)
{
    const size_t k = (size_t)l0 * o.stride;
    o.pcond1d[k]  = 0.0f;
    o.pidep1d[k]  = 0.0f;
    o.piacw1d[k]  = 0.0f;
    o.piacwi1d[k] = 0.0f;
    o.piacwr1d[k] = 0.0f;
    o.piacr1d[k]  = 0.0f;
    o.picnd1d[k]  = 0.0f;
    o.pievp1d[k]  = 0.0f;
    o.pimlt1d[k]  = 0.0f;
    o.praut1d[k]  = 0.0f;
    o.pracw1d[k]  = 0.0f;
    o.prevp1d[k]  = 0.0f;
    o.pisub1d[k]  = 0.0f;
    o.pevap1d[k]  = 0.0f;
    o.vsnow1d[k]  = 0.0f;
    o.vrain11d[k] = 0.0f;
    o.vrain21d[k] = 0.0f;
    o.vci1d[k]    = 0.0f;
    o.NSmICE1d[k] = 0.0f;
    o.DBZ_col[k]  = FER_DBZMIN;
    o.NR_col[k]   = 0.0f;
    o.NS_col[k]   = 0.0f;
    o.INDEXR1d[k] = 0.0f;
    o.INDEXS1d[k] = 0.0f;
    o.RFlag1d[k]  = 0.0f;
    o.TK_col[k]   = 0.0f;
    o.TC_col[k]   = 0.0f;
    o.CLEAR_col[k] = 0.0f;
}

// Lines 856-910: one level L (l0 = L-1) of the top-down loop, the part this unit owns.
// On return: clear is the CLEAR flag of the level; tk and tc are TK and TC of the level. The same values
// are written to o.CLEAR_col, o.TK_col and o.TC_col at this level.
__device__ __forceinline__ void fer_col_setup_level(int l0, const FerColLevelOut& o,
                                                    const float* __restrict__ T_col,
                                                    FerColCarry& c,
                                                    bool& clear, float& tk, float& tc)
{
    const size_t k = (size_t)l0 * o.stride;

    fer_col_zero_outputs(l0, o);

    // Lines 889-899: rain falling in from above.
    if (c.arain > FER_CLIMIT) {
        clear = false;
        c.vrain1 = 0.0f;
    } else {
        clear = true;
        c.arain = 0.0f;
    }

    // Lines 902-907: snow (large ice) falling in from above.
    if (c.asnow > FER_CLIMIT) {
        clear = false;
        c.vsnow1 = 0.0f;
    } else {
        c.asnow = 0.0f;
    }

    // Lines 909-910: temperature of the level.
    tk = T_col[k];
    tc = tk - FER_T0C;

    o.TK_col[k]    = tk;
    o.TC_col[k]    = tc;
    o.CLEAR_col[k] = clear ? 1.0f : 0.0f;
}

// One thread per column (i0, j0). lsfc[col] = LSFC of the column (Fortran LSFC = KTE - LOWLYR + 1).
// carry_out is [NCARRY=9, JM, IM]: rows 0..8 = arain, asnow, vrain1, vsnow1, timlt, indexs0c (as
// float), strat (0/1), drzl (0/1), rhox0c, written after levels 1..LSFC have been set up.
// lsfc_bad is one int32 counter: incremented for every column whose LSFC is outside [0, LM]. Such a
// column cannot be written in bounds, so it is set to the nearest valid count and the run fails the
// check (the counter is the loud error; a silent clamp would hide a broken column prep).
extern "C" __global__ void phys_fer_col_setup(
    const float* __restrict__ T, const int* __restrict__ lsfc,
    float* pcond1d, float* pidep1d, float* piacw1d, float* piacwi1d, float* piacwr1d,
    float* piacr1d, float* picnd1d, float* pievp1d, float* pimlt1d, float* praut1d,
    float* pracw1d, float* prevp1d, float* pisub1d, float* pevap1d, float* vsnow1d,
    float* vrain11d, float* vrain21d, float* vci1d, float* NSmICE1d, float* INDEXR1d,
    float* INDEXS1d, float* RFlag1d, float* DBZ_col, float* NR_col, float* NS_col,
    float* TK_col, float* TC_col, float* CLEAR_col,
    float* carry_out, int* lsfc_bad, const int IM, const int JM, const int LM)
{
    COLUMN_IJ(IM, JM)   // defines col_, i0, j0; returns for out-of-range threads

    const size_t off = IDX2(i0, j0, IM);
    const size_t stride = (size_t)JM * (size_t)IM;

    FerColLevelOut o;
    o.pcond1d  = pcond1d  + off;  o.pidep1d  = pidep1d  + off;  o.piacw1d  = piacw1d  + off;
    o.piacwi1d = piacwi1d + off;  o.piacwr1d = piacwr1d + off;  o.piacr1d  = piacr1d  + off;
    o.picnd1d  = picnd1d  + off;  o.pievp1d  = pievp1d  + off;  o.pimlt1d  = pimlt1d  + off;
    o.praut1d  = praut1d  + off;  o.pracw1d  = pracw1d  + off;  o.prevp1d  = prevp1d  + off;
    o.pisub1d  = pisub1d  + off;  o.pevap1d  = pevap1d  + off;  o.vsnow1d  = vsnow1d  + off;
    o.vrain11d = vrain11d + off;  o.vrain21d = vrain21d + off;  o.vci1d    = vci1d    + off;
    o.NSmICE1d = NSmICE1d + off;  o.INDEXR1d = INDEXR1d + off;  o.INDEXS1d = INDEXS1d + off;
    o.RFlag1d  = RFlag1d  + off;
    o.DBZ_col  = DBZ_col  + off;  o.NR_col   = NR_col   + off;  o.NS_col    = NS_col    + off;
    o.TK_col   = TK_col   + off;  o.TC_col   = TC_col   + off;  o.CLEAR_col = CLEAR_col + off;
    o.stride   = stride;

    const float* T_col = T + off;

    int n = lsfc[off];
    if (n < 0 || n > LM) {
        atomicAdd(lsfc_bad, 1);
        n = (n < 0) ? 0 : LM;
    }

    FerColCarry c;
    fer_col_setup_begin(c);

    for (int l0 = 0; l0 < n; ++l0) {     // DO L=1,LSFC
        bool clear;
        float tk, tc;
        fer_col_setup_level(l0, o, T_col, c, clear, tk, tc);
        (void)clear; (void)tk; (void)tc; // the values are also in CLEAR_col / TK_col / TC_col
    }
    for (int l0 = n; l0 < LM; ++l0) {    // levels above LSFC: defined defaults, never stale
        fer_col_zero_outputs(l0, o);
    }

    carry_out[0 * stride + off] = c.arain;
    carry_out[1 * stride + off] = c.asnow;
    carry_out[2 * stride + off] = c.vrain1;
    carry_out[3 * stride + off] = c.vsnow1;
    carry_out[4 * stride + off] = c.timlt;
    carry_out[5 * stride + off] = (float)c.indexs0c;
    carry_out[6 * stride + off] = c.strat ? 1.0f : 0.0f;
    carry_out[7 * stride + off] = c.drzl ? 1.0f : 0.0f;
    carry_out[8 * stride + off] = c.rhox0c;
}

// Branch probe (test only, one thread): runs fer_col_setup_level for each injected case so the rain
// and snow branches (Fortran 889-908) are exercised with non-zero carry. Case i: in[5i..5i+4] =
// ARAIN, ASNOW, VRAIN1, VSNOW1, TK. Output: out[6i..6i+5] = CLEAR, ARAIN, ASNOW, VRAIN1, VSNOW1, TC
// after the level. scratch needs 32 floats; scratch[31] holds the level temperature, the rest absorb the
// per-level writes (stride 0, so every level write lands at scratch[0]).
extern "C" __global__ void phys_fer_col_setup_probe(
    const float* __restrict__ in, float* __restrict__ out, float* scratch, const int ncase)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    FerColLevelOut o;
    o.pcond1d  = scratch; o.pidep1d  = scratch; o.piacw1d  = scratch; o.piacwi1d = scratch;
    o.piacwr1d = scratch; o.piacr1d  = scratch; o.picnd1d  = scratch; o.pievp1d  = scratch;
    o.pimlt1d  = scratch; o.praut1d  = scratch; o.pracw1d  = scratch; o.prevp1d  = scratch;
    o.pisub1d  = scratch; o.pevap1d  = scratch; o.vsnow1d  = scratch; o.vrain11d = scratch;
    o.vrain21d = scratch; o.vci1d    = scratch; o.NSmICE1d = scratch; o.INDEXR1d = scratch;
    o.INDEXS1d = scratch; o.RFlag1d  = scratch; o.DBZ_col  = scratch; o.NR_col   = scratch;
    o.NS_col   = scratch; o.TK_col   = scratch; o.TC_col   = scratch; o.CLEAR_col = scratch;
    o.stride   = 0;

    for (int i = 0; i < ncase; ++i) {
        FerColCarry c;
        fer_col_setup_begin(c);
        c.arain  = in[5 * i + 0];
        c.asnow  = in[5 * i + 1];
        c.vrain1 = in[5 * i + 2];
        c.vsnow1 = in[5 * i + 3];
        scratch[31] = in[5 * i + 4];

        bool clear;
        float tk, tc;
        fer_col_setup_level(0, o, scratch + 31, c, clear, tk, tc);

        out[6 * i + 0] = clear ? 1.0f : 0.0f;
        out[6 * i + 1] = c.arain;
        out[6 * i + 2] = c.asnow;
        out[6 * i + 3] = c.vrain1;
        out[6 * i + 4] = c.vsnow1;
        out[6 * i + 5] = tc;
    }
}
