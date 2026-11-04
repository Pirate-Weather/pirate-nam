// WOOF-NAM unit io-uphlcy: updraft helicity over two layers (2-5 km and 0-3 km AGL), hourly max/min.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module header kept as the notice:
//   MODULE MODULE_DIAGNOSE (NMMB diagnostics, nam.v4.2.11), nmm/module_DIAGNOSE.F90.
// Ported to CUDA for WOOF-NAM from nmm/module_DIAGNOSE.F90:2114-2186 (nam.v4.2.11, 83df4e13c1).
//
// Fortran SUBROUTINE CALC_UPHLCY, regional single-card case (ITS=1, ITE=IDE=IM, JTS=1,
// JTE=JDE=JM, no halo). The NAM fer_hires path calls it twice from SUBROUTINE MAX_FIELDS_HR
// (module_DIAGNOSE.F90 lines 2472 and 2477, HLOWER/HUPPER = 2000/5000 and 0/3000); the same two
// calls also sit in SUBROUTINE MAX_FIELDS at lines 2066 and 2071. Both calls are done in ONE
// launch here: the two layers are independent accumulators with their own EXIT state, so each
// output equals what the two Fortran calls write.
//
// Section by section (Fortran line -> here):
//   2141-2145  UPHL = 0 everywhere                     -> per-thread accumulator, starts at 0
//   2147-2149  R2DY, RDY from DY                       -> R2DY, RDY (float, same expressions)
//   2150       J_LOOP J = MAX(JTS,2)..MIN(JTE,JDE-1)   -> j0 in [1, JDE-2]
//   2152-2154  IF (DX(J) .LT. 0.1) CYCLE J_LOOP        -> whole row returns, no MAX/MIN update
//   2156-2157  R2DX, RDX from DX(J)                    -> R2DX, RDX
//   2158       I = MAX(ITS,2)..MIN(ITE,IDE-1)          -> i0 in [1, IDE-2]
//   2159       L_LOOP L = 1..LM-1                      -> l0 in [0, LM-2]
//   2160-2162  ZMIDLOC = Z(I,J,L); band test on ZMIDLOC - ZINTSFC -> in1, in2 per band
//   2163       DZ = Z(L) - Z(L+1)                      -> dz
//   2168-2170  W(L) < 0 in band: UPHL = 0, EXIT L_LOOP  -> done flag, accumulator = 0
//   2172-2173  VM1, VM2 (V on the four V neighbours)   -> vm1, vm2 (same order)
//   2174-2175  DVDX = (VM1-VM2)*RDX                    -> dvdx
//   2176-2177  UM1, UM2 (U on the four U neighbours)   -> um1, um2
//   2178       DUDY = (UM1-UM2)*RDY                    -> dudy
//   2178       UPHL += (DVDX-DUDY)*W*DZ                -> uphl += ((dvdx-dudy)*w)*dz
//   2181-2182  UPHLMAX = MAX(UPHL,UPHLMAX), MIN likewise -> written at the end of the thread
// ZINTSFC = FIS/g is computed in the caller (MAX_FIELDS_HR line 2313) with g = G from
// MODULE_CONSTANTS, so it is evaluated here as FIS / NMMB_G (nmmb_common.cuh).
// The commented-out third call (Fortran 2075-2077) is not part of the model and is not ported.
//
// W is the Fortran W dummy. The caller passes int_state%W_TOT there (module_SOLVER_GRID_COMP.F90
// line 5049, fer_hires branch), so this is W_TOT = W + BARO (set by VSOUND), not the bare W.
// It is read for the W<0 kill test and for the term.
//
// Index mapping (1-based Fortran -> 0-based CUDA): fortran (i,j,l) = cuda (i-1, j-1, l-1).
// One thread per column (grid_cols); the level loop is inside the thread. No host sync, no
// allocation, no printf. OpenMP and halo code are not ported (one card, whole domain).

#include "nmmb_common.cuh"

extern "C" __global__ void io_uphlcy(
    const int IM, const int JM, const int LM, const int IDE, const int JDE,
    const float* __restrict__ U,        // [LM,JM,IM] wind, grid-relative (m s-1)
    const float* __restrict__ V,        // [LM,JM,IM]
    const float* __restrict__ W,        // [LM,JM,IM] W_TOT, the Fortran W dummy: W + BARO (m s-1)
    const float* __restrict__ Z,        // [LM,JM,IM] height at midlayers (m)
    const float* __restrict__ FIS,      // [JM,IM] surface geopotential (m2 s-2); ZINTSFC = FIS/g
    const float* __restrict__ DXH,      // [JM] H-row grid spacing (m), Fortran DX(J)
    const float DYH,                    // grid spacing in y (m), Fortran DY
    const float HLOWER1, const float HUPPER1,   // layer 1: 2000 .. 5000 m AGL
    const float HLOWER2, const float HUPPER2,   // layer 2: 0 .. 3000 m AGL
    float* __restrict__ UPHLMAX25, float* __restrict__ UPHLMIN25,   // [JM,IM] hourly max/min, layer 1
    float* __restrict__ UPHLMAX03, float* __restrict__ UPHLMIN03)   // [JM,IM] hourly max/min, layer 2
{
    COLUMN_IJ(IM, JM)

    // Fortran loop limits: I in [2, IDE-1], J in [2, JDE-1] (1-based).
    if (i0 < 1 || i0 > IDE - 2 || j0 < 1 || j0 > JDE - 2) return;

    // Fortran J_LOOP: IF (DX(J) .LT. 0.1) CYCLE J_LOOP. The whole row is skipped, including the
    // MAX/MIN update, so nothing is written for it.
    const float dxj = DXH[j0];
    if (dxj < 0.1f) return;

    const float R2DY = 1.0f / (2.0f * DYH);
    const float RDY  = 2.0f * R2DY;
    const float R2DX = 1.0f / (2.0f * dxj);
    const float RDX  = 2.0f * R2DX;

    const float zint = FIS[IDX2(i0, j0, IM)] / NMMB_G;   // ZINTSFC(I,J) = FIS(I,J)/g

    float uphl1 = 0.0f;   // layer 1 (2-5 km) accumulator
    float uphl2 = 0.0f;   // layer 2 (0-3 km) accumulator
    bool done1 = false;   // set when W < 0 inside layer 1: UPHL = 0 and EXIT for that layer
    bool done2 = false;

    for (int l0 = 0; l0 < LM - 1; ++l0) {
        const size_t c = IDX3(i0, j0, l0, IM, JM);
        const float zmid = Z[c];
        const float zagl = zmid - zint;
        const bool in1 = (zagl >= HLOWER1) && (zagl <= HUPPER1);
        const bool in2 = (zagl >= HLOWER2) && (zagl <= HUPPER2);
        const bool act1 = in1 && !done1;
        const bool act2 = in2 && !done2;
        if (!act1 && !act2) continue;

        const float wv = W[c];
        const float dz = Z[c] - Z[IDX3(i0, j0, l0 + 1, IM, JM)];

        if (act1 && wv < 0.0f) { uphl1 = 0.0f; done1 = true; }
        if (act2 && wv < 0.0f) { uphl2 = 0.0f; done2 = true; }
        if (!(act1 && !done1) && !(act2 && !done2)) continue;

        // Horizontal vorticity-like term, same expression order as the Fortran.
        const float vm1 = 0.5f * (V[c] + V[IDX3(i0, j0 - 1, l0, IM, JM)]);
        const float vm2 = 0.5f * (V[IDX3(i0 - 1, j0, l0, IM, JM)] + V[IDX3(i0 - 1, j0 - 1, l0, IM, JM)]);
        const float dvdx = (vm1 - vm2) * RDX;
        const float um1 = 0.5f * (U[IDX3(i0 - 1, j0, l0, IM, JM)] + U[c]);
        const float um2 = 0.5f * (U[IDX3(i0 - 1, j0 - 1, l0, IM, JM)] + U[IDX3(i0, j0 - 1, l0, IM, JM)]);
        const float dudy = (um1 - um2) * RDY;
        const float term = (dvdx - dudy) * wv * dz;

        if (act1 && !done1) uphl1 = uphl1 + term;
        if (act2 && !done2) uphl2 = uphl2 + term;
    }

    // Fortran: UPHLMAX(I,J)=MAX(UPHL(I,J),UPHLMAX(I,J)); UPHLMIN(I,J)=MIN(UPHL(I,J),UPHLMIN(I,J)).
    const size_t o = IDX2(i0, j0, IM);
    UPHLMAX25[o] = fmaxf(uphl1, UPHLMAX25[o]);
    UPHLMIN25[o] = fminf(uphl1, UPHLMIN25[o]);
    UPHLMAX03[o] = fmaxf(uphl2, UPHLMAX03[o]);
    UPHLMIN03[o] = fminf(uphl2, UPHLMIN03[o]);
}
