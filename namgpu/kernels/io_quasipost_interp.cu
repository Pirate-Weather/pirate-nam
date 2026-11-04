// io_quasipost_interp.cu -- unit io-quasipost-interp: calculate_interP, interpolation of the
// membrane-MSLP inputs (below-ground extrapolation and the sea-level pressure at Z = 0).
//
// Ported to CUDA for WOOF-NAM from nam.v4.2.11 (83df4e13c1)
//   sorc/nam_nems_nmmb.fd/src/atmos/nmm/module_QUASIPOST.F90:688-815  (subroutine calculate_interP)
// Constants: module_QUASIPOST.F90:9-60 (npres = 33, post_stdpres, post_RD, post_G, post_slope).
// NOAA source is US government work (see NOTICE); no licence file is carried.
//
// Layout (CONTRACT.md section 1): Fortran level k (1-based) of a [npres] field -> index k-1 here
// (PRESTV/PRESZ are [npres, JM, IM]); interface PINT(i,j,kde+1) with kde = LM is index LM in the
// [LM+1, JM, IM] array (0-based, model top = 0). FIS and the first-guess ground index are [JM, IM].
//
// Fortran-to-CUDA review (every statement of 688-815):
//   STEP 1 loop j,i over the whole tile (ips..ipe, jps..jpe = 1..IM, 1..JM): one thread per column.
//   ground_level(i,j) < npres+1 branch: kp = ground_level-1; kp < 1 is the MPI_Abort in NAM; here it
//     sets err[0] = 1 (see the abort note below).
//     newZ = fis/post_G, newP = PINT(kde+1) (= PINT index LM), then the ip loop over kp..npres-1
//     with midTv = presTv(ip+1), newP = post_stdpres(ip+1), dZ = RD*midTv*ln(P/newP)/g (with the
//     Fortran evaluation order), dZ >= 0 is the MPI_Abort (err[0] = 2), presZ(ip+1) = newZ, and the
//     newZ <= 0 crossing sets Pmsl = (Z*newP - newZ*P)/(-dZ) and leaves the column (Fortran cycle iloop).
//   The two MPI_Abort paths set ERR (atomicMax: 1 kp<1, 2 dZ>=0) and PMSL = NaN for that column, a defined
//   value. The host reads ERR at the output stage (io_quasipost_interp.check_err) and raises; nothing is silent.
//   fall-through (ground at/below the top standard level or no crossing): TLYR = presTv(npres)
//     - presZ(npres)*slope*g*0.5, Pmsl = post_stdpres(npres)/EXP(-presZ(npres)*g/(RD*TLYR)).
//   The unused dummy arguments of NAM (T1, Q1, modelZ, ground_mask, IDS..KME) are read only in
//   commented-out code in NAM and are not passed here. ground_mask is set by calculate_3D, not here.
//   Float constants are single precision (kfpt=single); ln and exp use logf and expf.
//
// One thread per column (grid_cols); no host sync, no device malloc. Fast numerics (--fmad=true).

#include "nmmb_common.cuh"

#define QPI_NPRES 33

__constant__ float QPI_POST_STDPRES[QPI_NPRES] = {
    20000.f, 22500.f, 25000.f, 27500.f, 30000.f, 32500.f, 35000.f, 37500.f, 40000.f,
    42500.f, 45000.f, 47500.f, 50000.f, 52500.f, 55000.f, 57500.f, 60000.f,
    62500.f, 65000.f, 67500.f, 70000.f, 72500.f, 75000.f, 77500.f, 80000.f,
    82500.f, 85000.f, 87500.f, 90000.f, 92500.f, 95000.f, 97500.f, 100000.f};

#define QPI_RD     287.04f
#define QPI_G      9.81f
#define QPI_SLOPE  (-6.6e-4f)
#define QPI_NAN    __uint_as_float(0x7fc00000u)   // quiet NaN: marks a column that hit a NAM abort

extern "C" __global__ void io_quasipost_interp(
    const int IM, const int JM, const int LM,      // LM = KDE; PINT has LM+1 interface levels
    const float* __restrict__ PINT,                // [LM+1, JM, IM] interface pressure (Pa), PINT[0] = top
    const float* __restrict__ FIS,                 // [JM, IM] surface geopotential (m2 s-2)
    const int*   __restrict__ GROUND_LEVEL,        // [JM, IM] first-guess ground index, 1-based, 1..npres+1
    const float* __restrict__ PRESTV,              // [npres, JM, IM] input: Tv on standard levels (in)
    float*       PRESZ,                            // [npres, JM, IM] in/out: Z on standard levels
    float*       PMSL,                             // [JM, IM] out: sea-level pressure (Pa)
    int*         ERR)                              // [1] out: 0 ok, 1 kp<1 (NAM abort), 2 dZ>=0 (NAM abort)
{
    COLUMN_IJ(IM, JM)

    const int cij = IDX2(i0, j0, IM);
    const int gl  = GROUND_LEVEL[cij];
    const float ps_top = PINT[IDX3(i0, j0, LM, IM, JM)];   // PINT(i,j,kde+1)

    if (gl < QPI_NPRES + 1) {
        const int kp = gl - 1;                              // 1-based kp, as in NAM
        if (kp < 1) { atomicMax(ERR, 1); PMSL[cij] = QPI_NAN; return; }

        float Z = 0.0f;
        float P = 0.0f;
        float newZ = FIS[cij] / QPI_G;                      // newZ = fis/post_G
        float newP = ps_top;                                // newP = pint(i,j,kde+1)
        for (int ip = kp; ip <= QPI_NPRES - 1; ++ip) {      // Fortran ip = kp..npres-1
            P = newP;
            Z = newZ;
            const float midTv = PRESTV[IDX3(i0, j0, ip, IM, JM)];        // presTv(ip+1)
            newP = QPI_POST_STDPRES[ip];                                 // post_stdpres(ip+1)
            const float dZ = QPI_RD * midTv * logf(P / newP) / QPI_G;
            if (dZ >= 0.0f) { atomicMax(ERR, 2); PMSL[cij] = QPI_NAN; return; }
            newZ = Z + dZ;
            PRESZ[IDX3(i0, j0, ip, IM, JM)] = newZ;                      // presZ(ip+1)
            if (newZ <= 0.0f) {
                PMSL[cij] = (Z * newP - newZ * P) / (-dZ);
                return;                                                  // Fortran: cycle iloop
            }
        }
    }

    // Z = 0 is below the lowest standard level (or the ground is at/below it): extrapolate from npres.
    const float pz_n = PRESZ[IDX3(i0, j0, QPI_NPRES - 1, IM, JM)];
    const float tv_n = PRESTV[IDX3(i0, j0, QPI_NPRES - 1, IM, JM)];
    const float tlyr = tv_n - pz_n * QPI_SLOPE * QPI_G * 0.5f;
    PMSL[cij] = QPI_POST_STDPRES[QPI_NPRES - 1] / expf(-pz_n * QPI_G / (QPI_RD * tlyr));
}
