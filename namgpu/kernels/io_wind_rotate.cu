// NOAA/NCEP header of sorc/nam_post0.fd/PROF_NMMB.f (lines 1-28), kept as notice (NOTICE):
//       SUBROUTINE PROF_NMMB(filename,ITAG,INCR)
// C
// C$$$  SUBPROGRAM DOCUMENTATION BLOCK
// C                .      .    .
// C   SUBROUTINE:  PROF        PROFILE SOUNDINGS
// C   PRGRMMR: BLACK           ORG: W/NP22     DATE: 99-04-22
// C
// C ABSTRACT:  THIS ROUTINE GENERATES THE RAW PROFILE SOUNDING
// C            OUTPUT FILES FROM THE FORECAST RESTRT FILE AND
// C            AUXILIARY FILES
// C
// C PROGRAM HISTORY LOG:
// C   99-04-22  T BLACK - ORIGINATOR
// C   02-07-01  G MANIKIN - FIXED PROBLEM WITH DHCNVC AND DHRAIN
// C                          COMPUTATIONS - SEE COMMENTS BELOW
// C   03-04-01  M PYLE - BEGAN CONVERTING FOR WRF
// C   04-05-26  M PYLE - MADE CHANGES FOR WRF-NMM
// C   08-09-11  M PYLE - NMMB VERSION READING NEMSIO
// C   10-04-26  ROGERS - CHANGED TIMESTAMP FROM HOURS to MINUTES
// C
// C USAGE:  CALL PROF FROM PROGRAM POST0
// C
// C   INPUT ARGUMENT LIST:
// C     FILENAME - HISTORY FILE NAME
// C     LRSTRT - THE UNIT NUMBER FOR READING THE RESTRT FILE
// C     ITAG   - ELAPSED FORECAST TIME IN MINUTES OF THE HISTORY FILE WE ARE DEALING WITH
// C     INCR   - FORECAST INTERVAL BETWEEN HISTORY FILES IN MINUTES
// C
//
// Ported to CUDA for WOOF-NAM from sorc/nam_post0.fd/PROF_NMMB.f:2115-2150 and 2232-2237 (nam.v4.2.11, 83df4e13c1).
//
// WOOF-NAM unit io-wind-rotate: grid-relative winds at V points to earth-relative (east, north) winds
// at H points.
//
// The post (PROF_NMMB.f 2115-2150, 2232-2237) rotates with UE = U*COSALP + V*SINALP and
// VN = V*COSALP - U*SINALP, one ALPHA per station. This kernel does not take the post's order (see 2):
//
// 1. Each in-domain V neighbour of H(i0,j0) is rotated with its own exact 2x2 inverse of the forward map of
//    init-gfs-wind-rotate, u = UE*ce_e + VE*ce_n, v = UE*cn_e + VE*cn_n:
//      UE = (cn_n*u - ce_n*v) / det,   VE = (-cn_e*u + ce_e*v) / det,   det = ce_e*cn_n - ce_n*cn_e.
//    V(i,j) sits NE of H(i,j) (CONTRACT.md section 1), so H(i0,j0) has V(i0,j0), V(i0-1,j0), V(i0,j0-1),
//    V(i0-1,j0-1). The last V row and column (vi > IM-2, vj > JM-2) are unused and excluded.
// 2. Decision, labelled (not a NAM term): the earth-relative vectors of the valid neighbours are averaged
//    (rotate-then-average), not the grid-relative winds (average-then-rotate, the post's order). The reason is
//    measured: the geographic pole lies inside this domain (max lat 89.98 N; 77 V cells with a > 0.2 rad jump
//    of the grid-i direction at lat 89.5-89.98; tools/check-io-wind-rotate.py). Averaging grid-relative winds
//    across that jump mixes opposite directions (round-trip error 5.75 m/s on a 10 m/s uniform wind). Away from
//    the pole the two orders differ by the angle change across one cell (~1e-3 rad).
//
// Edge H points average the neighbours that exist. The same kernel rotates the 10 m winds (LM = 1).
// One thread per (i, j) with the level in blockIdx.z (grid_2d). The full [JM, IM] H domain is written;
// no halo and no host sync.

#include "nmmb_common.cuh"

extern "C" __global__ void io_wind_rotate(
    const int IM, const int JM, const int LM,
    const float* __restrict__ ce_e,     // [JM,IM] east component of grid-i unit vector (V points)
    const float* __restrict__ ce_n,     // [JM,IM] north component of grid-i unit vector
    const float* __restrict__ cn_e,     // [JM,IM] east component of grid-j unit vector
    const float* __restrict__ cn_n,     // [JM,IM] north component of grid-j unit vector
    const float* __restrict__ U,        // [LM,JM,IM] grid-relative wind, grid-i (V points)
    const float* __restrict__ V,        // [LM,JM,IM] grid-relative wind, grid-j (V points)
    float* __restrict__ UE,             // [LM,JM,IM] out: earth-relative eastward wind (H points)
    float* __restrict__ VE)             // [LM,JM,IM] out: earth-relative northward wind (H points)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;

    float sue = 0.0f;
    float sve = 0.0f;
    int n = 0;

    for (int dj = 0; dj < 2; ++dj) {
        for (int di = 0; di < 2; ++di) {
            const int vi = i0 - di;
            const int vj = j0 - dj;
            if (vi < 0 || vj < 0 || vi > IM - 2 || vj > JM - 2) continue;

            const size_t c2 = IDX2(vi, vj, IM);
            const size_t c3 = IDX3(vi, vj, l0, IM, JM);
            const float uu = U[c3];
            const float vv = V[c3];
            const float a = ce_e[c2];
            const float b = ce_n[c2];
            const float c = cn_e[c2];
            const float d = cn_n[c2];
            const float det = a * d - b * c;

            sue += (d * uu - b * vv) / det;
            sve += (-c * uu + a * vv) / det;
            ++n;
        }
    }

    const size_t c3h = IDX3(i0, j0, l0, IM, JM);
    if (n > 0) {
        const float inv = 1.0f / (float)n;
        UE[c3h] = sue * inv;
        VE[c3h] = sve * inv;
    } else {
        UE[c3h] = 0.0f;
        VE[c3h] = 0.0f;
    }
}
