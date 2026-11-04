// nam-gpu unit io-quasipost-3d: calculate_3D, the 3-D diagnostic fields of NAM's unified post
// (heights, temperature/Tv and RH-bounded extrapolation on the NCEP post pressure levels, and the
// 500/700/850 mb winds, vorticity and 10-m wind magnitude/vorticity).
//
// Ported to CUDA for nam-gpu from nmm/module_QUASIPOST.F90:257-686 (nam.v4.2.11, 83df4e13c1).
// The Fortran file has no NOAA header block; the notice is in NOTICE.
//
// Launch order (wrapper io/io_quasipost_3d.py), one launch per section of the routine:
//   1. io_quasipost_3d_column    Pmass, the T/Q/Z interpolation to the 33 post levels (both
//                                TQ loops and the mid-to-surface loop), ground_level/ground_mask,
//                                the below-ground Tv extrapolation. One thread per column, the
//                                ks/kd recurrence runs in order inside the thread (as in Fortran).
//   2. for k in (500 mb, 700 mb, 850 mb):
//      io_quasipost_3d_interior  zero the boundary rows/columns, then the interior winds
//                                (disjoint writes, so one launch).
//      io_quasipost_3d_edge_i    copy the i-boundary columns from the interior (reads only
//                                interior values written by step 2a).
//      io_quasipost_3d_edge_j    copy the j-boundary rows (reads the i-copied corners, so it
//                                must run after edge_i, same stream).
//   3. io_quasipost_3d_m10       10-m wind magnitude and vorticity (H points, from V-point U10/V10)
//                                and the NaN count.
//
// Domain: single process, the whole 954 x 835 domain (ids = jds = 1, ide = IM, jde = JM,
// kds = 1, kde = LM), so every halo exchange and MPI abort in the Fortran is dropped. The
// Fortran NaN abort (m10rv) and the weight-out-of-range abort are counted into nbad[0] (weight)
// and nbad[1] (NaN m10rv) instead, so no host sync happens inside the step; the wrapper leaves
// the counters for the caller to read at output time.
//
// Fortran indexing is kept 1-based inside the kernels with the F3/F2 macros below, so each
// line can be compared with the Fortran line it came from.
//
// Decisions (see the report in the unit's commit):
//  - Pmass(i,j,kds) is never assigned in the Fortran (Pmass starts at kds+1). It is read in the
//    wind section's "at and below the lowest mass level" branch. This port compares against
//    Pmass(kde) (the lowest mass level, the intent of the comment there) instead of the undefined value.
//  - DECLARED DIVERGENCE (review of io-quasipost-3d): the same branch reads its winds from model
//    level LM (the lowest layer, the level the Fortran comment means), not from level 1. Level 1 is
//    the model TOP (CONTRACT.md section 1); the literal Fortran `ks=1` would take the 500/700/850 mb
//    winds from the top of the model wherever the lowest mass level is above those pressures (high
//    terrain). The literal read is not kept.
//  - DECLARED DIVERGENCE: U10 and V10 are V points (CONTRACT.md section 1, the same convention
//    io-wind-rotate uses). The Fortran comment "u10 and v10 are already on H points"
//    (module_QUASIPOST.F90:569) is not followed. The 10 m section averages them to H points with
//    the four-point V-to-H stencil (edge H points average the V neighbours that exist) before the
//    H-point speed and vorticity.
//  - nbad counters are per call: run() zeroes QP_NBAD at its start, so they count one run's
//    out-of-range weights and NaN m10rv values. The Fortran aborts on both; the port counts them,
//    and the caller reads QP_NBAD at output time.
//  - ZL in the below-ground extrapolation is computed in the Fortran and never read (the only
//    use is a commented-out print), so it is not computed here.
//  - The Fortran wraps the wind section in an if on size(p700rv) and size(p700u) being above 1.
//    The port always runs it: the wind arrays are allocated by the wrapper.
//  - wantuv is a constant true in the Fortran, so the U/V branches are unconditional here.
//  - The 500-mb wind and vorticity go to the Fortran local dummy1/dummy2 arrays, which are
//    never read outside the routine. The port keeps them as QP_DUMMY1/2 scratch, as the Fortran.

#include "nmmb_common.cuh"

#define NPRES 33
#define QP_K850 27
#define QP_K700 21
#define QP_K500 13
#define QP_D608 0.608f
#define QP_G 9.81f                       /* post_G (post constant, not NMMB_G) */
#define QP_RGAMOG (287.04f * 6.5e-3f / 9.81f)   /* post_RD*post_GAMMA/post_G */
#define QP_PQ0 379.90516f
#define QP_A2 17.2693882f
#define QP_A3 273.16f
#define QP_A4 35.86f
#define QP_RHMIN 1.0e-6f
#define QP_BADHEIGHT (-9.0e9f)

// post_stdpres(1:33) from module_QUASIPOST.F90:15-19 (Pa).
__constant__ float QP_STDPRES[NPRES] = {
    20000.f, 22500.f, 25000.f, 27500.f, 30000.f, 32500.f, 35000.f, 37500.f, 40000.f,
    42500.f, 45000.f, 47500.f, 50000.f, 52500.f, 55000.f, 57500.f, 60000.f,
    62500.f, 65000.f, 67500.f, 70000.f, 72500.f, 75000.f, 77500.f, 80000.f,
    82500.f, 85000.f, 87500.f, 90000.f, 92500.f, 95000.f, 97500.f, 100000.f};

// Fortran 1-based access: F3(A, i, j, l) = A(i,j,l) for an [L,JM,IM] array; F2(A, i, j) = A(i,j).
#define F3(a, ii, jj, ll) a[IDX3((ii) - 1, (jj) - 1, (ll) - 1, IM, JM)]
#define F2(a, ii, jj)     a[IDX2((ii) - 1, (jj) - 1, IM)]

// ---------------------------------------------------------------------------------------------
// Section 1: TQ interpolation, ground level, below-ground extrapolation. One thread per column.
// Fortran: Pmass (module_QUASIPOST.F90:300-307), TQ loop (318-364), mid-to-surface loop
// (366-386), ground_level/ground_mask (~560-580), kloop2 extrapolation (621-676).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void io_quasipost_3d_column(
    const int IM, const int JM, const int LM,
    const float* __restrict__ PINT,      // [LM+1,JM,IM] interface pressure (Pa)
    const float* __restrict__ Z,         // [LM,JM,IM]   height at mid layers (m)
    const float* __restrict__ T,         // [LM,JM,IM]   temperature (K)
    const float* __restrict__ Q,         // [LM,JM,IM]   specific humidity (kg/kg)
    const float* __restrict__ FIS,       // [JM,IM]      surface geopotential (m2 s-2)
    float* __restrict__ presTv,          // [NPRES,JM,IM] out
    float* __restrict__ presZ,           // [NPRES,JM,IM] out
    int* __restrict__ ground_mask,       // [NPRES,JM,IM] out
    int* __restrict__ ground_level,      // [JM,IM]       out
    int* __restrict__ ks850,             // [JM,IM]       out (model level bracketing 850 mb)
    int* __restrict__ ks700,             // [JM,IM]       out
    int* __restrict__ ks500,             // [JM,IM]       out
    int* __restrict__ nbad)              // [2]: nbad[0] weight out of range (Fortran abort)
{
    COLUMN_IJ(IM, JM)                    // i0, j0 (zero-based), returns for padding threads
    const int i = i0 + 1;
    const int j = j0 + 1;

    int Tkdest = 1;
    int Zkdest = 1;
    int s850 = 0;
    int s700 = 0;
    int s500 = 0;

    // TQ loop (Fortran 318-364): ks = kds+1 .. kde-1, kd resumes from Tkdest.
    for (int ks = 2; ks <= LM - 1; ++ks) {
        int kd = Tkdest;
        if (kd <= NPRES) {
            while (kd <= NPRES) {
                const float pmlo = sqrtf(F3(PINT, i, j, ks) * F3(PINT, i, j, ks + 1));
                const float pmhi = sqrtf(F3(PINT, i, j, ks + 1) * F3(PINT, i, j, ks + 2));
                const float pk = QP_STDPRES[kd - 1];
                if (!(pk <= pmhi && pk >= pmlo)) break;            /* cycle iTQ */
                const float w = logf(pk / pmlo) / logf(pmhi / pmlo);
                const float presT = w * F3(T, i, j, ks + 1) + (1.0f - w) * F3(T, i, j, ks);
                const float presQ = w * F3(Q, i, j, ks + 1) + (1.0f - w) * F3(Q, i, j, ks);
                F3(presZ, i, j, kd) = w * F3(Z, i, j, ks + 1) + (1.0f - w) * F3(Z, i, j, ks);
                F3(presTv, i, j, kd) = presT * (1.0f + QP_D608 * presQ);
                if (kd == QP_K850) s850 = ks;
                else if (kd == QP_K700) s700 = ks;
                else if (kd == QP_K500) s500 = ks;
                if (w < 0.0f || w > 1.0f) atomicAdd(&nbad[0], 1);  /* Fortran: abort */
                kd = kd + 1;
                Tkdest = kd;
                Zkdest = kd;
            }
        }
    }

    // Mid-lowest-layer to surface (Fortran 370-386): uses Pmass(kde) and PINT(kde+1).
    const float pmkde = sqrtf(F3(PINT, i, j, LM) * F3(PINT, i, j, LM + 1));
    {
        int kd = Zkdest;
        if (kd <= NPRES) {
            while (kd <= NPRES) {
                const float pk = QP_STDPRES[kd - 1];
                const float psfc = F3(PINT, i, j, LM + 1);
                if (!(pk <= psfc && pk >= pmkde)) break;           /* cycle iTQ2 */
                const float presT = F3(T, i, j, LM);
                const float presQ = F3(Q, i, j, LM);
                F3(presTv, i, j, kd) = presT * (1.0f + QP_D608 * presQ);
                const float w = logf(pk / pmkde) / logf(psfc / pmkde);
                F3(presZ, i, j, kd) = (1.0f - w) * F3(Z, i, j, LM) + w * F2(FIS, i, j) / QP_G;
                kd = kd + 1;
                Tkdest = kd;
                Zkdest = kd;
            }
        }
    }

    F2(ks850, i, j) = s850;
    F2(ks700, i, j) = s700;
    F2(ks500, i, j) = s500;

    // ground_level / ground_mask (Fortran: min(Zkdest,Tkdest); mask = kd >= ground_level).
    const int gl = min(Zkdest, Tkdest);
    F2(ground_level, i, j) = gl;
    for (int kd = 1; kd <= NPRES; ++kd) {
        F3(ground_mask, i, j, kd) = (kd >= gl) ? 1 : 0;
    }

    // Below-ground Tv and Z (Fortran kloop2, 634-676). Above ground nothing is written.
    if (gl <= NPRES) {
        for (int kd = gl; kd <= NPRES; ++kd) {
            const float PL = F3(PINT, i, j, LM - 1);                      /* PINT(kde-1) */
            const float TL = 0.5f * (F3(T, i, j, LM - 1) + F3(T, i, j, LM));
            float QL = 0.5f * (F3(Q, i, j, LM - 1) + F3(Q, i, j, LM));
            const float QSAT = QP_PQ0 / PL * expf(QP_A2 * (TL - QP_A3) / (TL - QP_A4));
            float RHL = QL / QSAT;
            if (RHL > 1.0f) {
                RHL = 1.0f;
                QL = RHL * QSAT;
            }
            if (RHL < QP_RHMIN) {
                RHL = QP_RHMIN;
                QL = RHL * QSAT;
            }
            const float TVRL = TL * (1.0f + QP_D608 * QL);
            const float TVRBLO = TVRL * powf(QP_STDPRES[kd - 1] / PL, QP_RGAMOG);
            const float TBLO = TVRBLO / (1.0f + QP_D608 * QL);
            F3(presTv, i, j, kd) = TBLO;
            F3(presZ, i, j, kd) = QP_BADHEIGHT;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Section 2a: zero the boundary rows/columns (Fortran 'No wind on boundaries', ~line 430-470)
// and compute the interior winds (Fortran 'Interpolate winds', ~line 480-520). The two write
// disjoint point sets (boundary rows/columns vs. i in [2,IM-2], j in [2,JM-2]), so one launch.
// Fortran: ks>1 branch interpolates between mass levels; the else-branch uses the lowest model
// level (LM, not level 1; see the decisions block above) for pressures at or below the lowest
// mass level.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void io_quasipost_3d_interior(
    const int IM, const int JM, const int LM, const int kd,
    const float* __restrict__ PINT,      // [LM+1,JM,IM]
    const float* __restrict__ U,         // [LM,JM,IM] grid-relative u (V points)
    const float* __restrict__ V,         // [LM,JM,IM] grid-relative v (V points)
    const float* __restrict__ DXH,       // [JM]  delta x, H points (m)
    const float DYH,                     // delta y, H points (m)
    const int* __restrict__ ksx,         // [JM,IM] ks500 / ks700 / ks850 for this kd
    float* __restrict__ presu,           // [JM,IM]
    float* __restrict__ presv,           // [JM,IM]
    float* __restrict__ preswind,        // [JM,IM]
    float* __restrict__ presrv)          // [JM,IM]
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;
    const int i = i0 + 1;
    const int j = j0 + 1;

    // No wind on boundaries: row j=jds, row j=jde-1, column i=ids, column i=ide-1 (all zeroed).
    if (j == 1 || j == JM - 1 || i == 1 || i == IM - 1) {
        F2(presu, i, j) = 0.0f;
        F2(presv, i, j) = 0.0f;
        F2(preswind, i, j) = 0.0f;
        F2(presrv, i, j) = 0.0f;
    }

    // Interpolate winds, interior only: j in [2, JM-2], i in [2, IM-2].
    if (i >= 2 && i <= IM - 2 && j >= 2 && j <= JM - 2) {
        const int ks = F2(ksx, i, j);
        const float pk = QP_STDPRES[kd - 1];
        if (ks > 1) {
            const float pmlo = sqrtf(F3(PINT, i, j, ks) * F3(PINT, i, j, ks + 1));
            const float pmhi = sqrtf(F3(PINT, i, j, ks + 1) * F3(PINT, i, j, ks + 2));
            const float w = logf(pk / pmlo) / logf(pmhi / pmlo);

            const float U1 = 0.25f * (F3(U, i, j - 1, ks) + F3(U, i, j, ks)
                                      + F3(U, i - 1, j, ks) + F3(U, i - 1, j - 1, ks));
            const float V1 = 0.25f * (F3(V, i, j - 1, ks) + F3(V, i, j, ks)
                                      + F3(V, i - 1, j, ks) + F3(V, i - 1, j - 1, ks));
            const float U2 = 0.25f * (F3(U, i, j - 1, ks + 1) + F3(U, i, j, ks + 1)
                                      + F3(U, i - 1, j, ks + 1) + F3(U, i - 1, j - 1, ks + 1));
            const float V2 = 0.25f * (F3(V, i, j - 1, ks + 1) + F3(V, i, j, ks + 1)
                                      + F3(V, i - 1, j, ks + 1) + F3(V, i - 1, j - 1, ks + 1));

            const float dvdx1 = (F3(V, i + 1, j, ks) - F3(V, i - 1, j, ks)) / (2.0f * DXH[j0]);
            const float dudy1 = (F3(U, i, j + 1, ks) - F3(U, i, j - 1, ks)) / (2.0f * DYH);
            const float dvdx2 = (F3(V, i + 1, j, ks + 1) - F3(V, i - 1, j, ks + 1)) / (2.0f * DXH[j0]);
            const float dudy2 = (F3(U, i, j + 1, ks + 1) - F3(U, i, j - 1, ks + 1)) / (2.0f * DYH);

            F2(presu, i, j) = w * U2 + (1.0f - w) * U1;
            F2(presv, i, j) = w * V2 + (1.0f - w) * V1;
            F2(preswind, i, j) = w * sqrtf(U2 * U2 + V2 * V2) + (1.0f - w) * sqrtf(U1 * U1 + V1 * V1);
            F2(presrv, i, j) = (dvdx2 - dudy2) * w + (dvdx1 - dudy1) * (1.0f - w);
        } else {
            // Fortran: elseif(post_stdpres(kd) >= Pmass(i,j,kds)), then ks = 1. Pmass(kds) is
            // undefined in the Fortran; compared here against Pmass(kde), and the winds are read
            // from the lowest model level LM (level 1 is the model top).
            const int kl = LM;
            const float pmkde = sqrtf(F3(PINT, i, j, LM) * F3(PINT, i, j, LM + 1));
            if (pk >= pmkde) {
                const float U1 = 0.25f * (F3(U, i, j - 1, kl) + F3(U, i - 1, j, kl)
                                          + F3(U, i, j, kl) + F3(U, i - 1, j - 1, kl));
                const float V1 = 0.25f * (F3(V, i, j - 1, kl) + F3(V, i - 1, j, kl)
                                          + F3(V, i, j, kl) + F3(V, i - 1, j - 1, kl));
                const float dvdx1 = (F3(V, i + 1, j, kl) - F3(V, i - 1, j, kl)) / (2.0f * DXH[j0]);
                const float dudy1 = (F3(U, i, j + 1, kl) - F3(U, i, j - 1, kl)) / (2.0f * DYH);

                F2(preswind, i, j) = sqrtf(U1 * U1 + V1 * V1);
                F2(presrv, i, j) = dvdx1 - dudy1;
                F2(presu, i, j) = U1;
                F2(presv, i, j) = V1;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Section 2b: copy the i-boundary columns (Fortran 'Copy wind-related fields to boundary points',
// first loop over b). Thread (jj, b): b=0 -> i=1 from 2, b=1 -> i=IM-1 from IM-2, b=2 -> i=IM
// from IM-2; j in [2, JM-2]. Grid: x over j, y over b.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void io_quasipost_3d_edge_i(
    const int IM, const int JM,
    float* __restrict__ presu,
    float* __restrict__ presv,
    float* __restrict__ preswind,
    float* __restrict__ presrv)
{
    const int b = blockIdx.y;
    const int j = 2 + (blockIdx.x * blockDim.x + threadIdx.x);
    if (j > JM - 2) return;
    int i, iref;
    if (b == 0) { i = 1;      iref = 2; }
    else if (b == 1) { i = IM - 1; iref = IM - 2; }
    else { i = IM;     iref = IM - 2; }

    F2(presu, i, j) = F2(presu, iref, j);
    F2(presv, i, j) = F2(presv, iref, j);
    F2(presrv, i, j) = F2(presrv, iref, j);
    F2(preswind, i, j) = F2(preswind, iref, j);
}

// ---------------------------------------------------------------------------------------------
// Section 2c: copy the j-boundary rows (Fortran 'Copy wind-related fields to boundary points',
// second loop over b). Thread (i, b): b=0 -> j=1 from 2, b=1 -> j=JM-1 from JM-2, b=2 -> j=JM
// from JM-2; all i in [1, IM]. Must run after edge_i (reads its corner values).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void io_quasipost_3d_edge_j(
    const int IM, const int JM,
    float* __restrict__ presu,
    float* __restrict__ presv,
    float* __restrict__ preswind,
    float* __restrict__ presrv)
{
    const int b = blockIdx.y;
    const int i = 1 + (blockIdx.x * blockDim.x + threadIdx.x);
    if (i > IM) return;
    int j, jref;
    if (b == 0) { j = 1;      jref = 2; }
    else if (b == 1) { j = JM - 1; jref = JM - 2; }
    else { j = JM;     jref = JM - 2; }

    F2(presu, i, j) = F2(presu, i, jref);
    F2(presv, i, j) = F2(presv, i, jref);
    F2(presrv, i, j) = F2(presrv, i, jref);
    F2(preswind, i, j) = F2(preswind, i, jref);
}

// ---------------------------------------------------------------------------------------------
// Section 3: 10-m wind magnitude and vorticity on H points (Fortran 'Calculate 10m wind and
// vorticity', i in [2, IM-1], j in [2, JM-1]). m10rv == m10rv is false only for NaN; the Fortran
// aborts there, the port counts it in nbad[1].
// U10/V10 are V points (CONTRACT.md section 1), so each H value is the mean of the V neighbours
// H(i,j) has: V(i,j), V(i-1,j), V(i,j-1), V(i-1,j-1). Edge H points average the neighbours that
// exist (the io-wind-rotate rule). The Fortran comment "already on H points" is not followed.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float h_of_v(const float* __restrict__ A, const int IM, const int JM,
                                        const int i, const int j)
{
    float s = 0.0f;
    int n = 0;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            const int ii = i + di;
            const int jj = j + dj;
            if (ii >= 1 && ii <= IM && jj >= 1 && jj <= JM) {
                s += F2(A, ii, jj);
                ++n;
            }
        }
    }
    return s / (float)n;
}
extern "C" __global__ void io_quasipost_3d_m10(
    const int IM, const int JM,
    const float* __restrict__ U10,       // [JM,IM] (V points)
    const float* __restrict__ V10,       // [JM,IM] (V points)
    const float* __restrict__ DXH,       // [JM]
    const float DYH,
    float* __restrict__ m10wind,         // [JM,IM] out
    float* __restrict__ m10rv,           // [JM,IM] out
    int* __restrict__ nbad)              // nbad[1]: NaN m10rv count
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;
    const int i = i0 + 1;
    const int j = j0 + 1;
    if (i < 2 || i > IM - 1 || j < 2 || j > JM - 1) return;

    const float u10h = h_of_v(U10, IM, JM, i, j);
    const float v10h = h_of_v(V10, IM, JM, i, j);
    F2(m10wind, i, j) = sqrtf(u10h * u10h + v10h * v10h);
    const float dvdx1 = (h_of_v(V10, IM, JM, i + 1, j) - h_of_v(V10, IM, JM, i - 1, j)) / (2.0f * DXH[j0]);
    const float dudy1 = (h_of_v(U10, IM, JM, i, j + 1) - h_of_v(U10, IM, JM, i, j - 1)) / (2.0f * DYH);
    F2(m10rv, i, j) = dvdx1 - dudy1;
    if (!(F2(m10rv, i, j) == F2(m10rv, i, j))) atomicAdd(&nbad[1], 1);
}
