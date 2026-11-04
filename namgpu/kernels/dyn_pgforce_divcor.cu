// dyn_pgforce_divcor.cu -- divergence correction (Janjic 1974 MWR, 1979 Beitrage) of the
// pressure-gradient force, with the row weights wpdar (wcor, dyh) set by grid_consts.
//
// NOTICE: NAM source file nmm/module_DYNAMICS_ROUTINES.F90 (NOAA/NCEP/EMC, US Government work,
// no licence file). The file carries no author/history block of its own; see NOTICE at the repo
// root. The section is the divergence-correction part of subroutine PGFORCE.
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:290-451 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (one level l; every level is independent, so all levels run in one launch):
//   global branch (global = .true.), jw = 1 (module_DYNAMICS_ROUTINES.F90:72):
//     div(i,j,l) = 0 for j in [jds, jds-1+jw] and [jde-jw+1, jde], i in [its, ite]
//     row jcl = jds+jw:   i in [its_b1, ite_b1]:  (pgx(i+1,j)-pgx(i,j) + pgy(i,j+1)
//                         - (pgne(i+1,j+1) + pgnw(i,j+1))*0.5) * wpdar(j)
//     row jch = jde-jw:   i in [its_b1, ite_b1]:  (pgx(i+1,j)-pgx(i,j) - pgy(i,j)
//                         - (-pgne(i,j) - pgnw(i+1,j))*0.5) * wpdar(j)
//     rows jcl+1..jch-1 (clipped to the tile), i in [its_b1, ite_b1]: interior formula below
//   regional branch (global = .false., the NAM 12 km CONUS parent):
//     s_bdy, j = jds+1,  i in [its_b2, ite_b2]:  row-jcl formula
//     n_bdy, j = jde-1,  i in [its_b2, ite_b2]:  row-jch formula
//     w_bdy, i = ids+1,  j in [jts_b2, jte_b2]:  (pgx(ip+1,j) + pgy(ip,j+1)-pgy(ip,j)
//                         - (pgne(ip+1,j+1)-pgnw(ip+1,j))*0.5) * wpdar(j)
//     e_bdy, i = ide-1,  j in [jts_b2, jte_b2]:  (-pgx(ip,j) + pgy(ip,j+1)-pgy(ip,j)
//                         - (-pgne(ip,j)+pgnw(ip,j+1))*0.5) * wpdar(j)
//     corners at (ip, jp) with ip = ids+1 or ide-1, jp = jds+1 or jde-1, each with its own
//       formula (s&w, s&e, n&w, n&e), written only when the tile holds the point
//     interior: i in [its_b2, ite_b2], j in [jts_b2, jte_b2]: interior formula
//   Cells outside these sets are not written by the routine, so DIV keeps its previous value
//   there, as in NMMB (the kernel does not write them either).
//
// Tile limits come at run time from the drv-loop-limits table (ctx.tables["NMMB_LIMITS"],
// module_DM_PARALLEL looplimits and tile limits). The boundary flags are the Fortran tile flags
// of module_CONTROL.f90:555 and its neighbours:
//   s_bdy = (jts == jds), n_bdy = (jte == jde), w_bdy = (its == ids), e_bdy = (ite == ide).
// Nothing here assumes one process: on one card the table holds the full domain, and on a
// multi-task layout each task's kernel writes only its own tile.
//
// Inputs: pgx, pgy, pgne, pgnw are the pgf scratch of dyn-pgforce-pgf (PGX_3D, PGY_3D, PGNE_3D,
// PGNW_3D, [LM, JM, IM]); wpdar is the [JM] row vector from init-grid-consts (wpdar(2) and
// wpdar(JM-1) are the pole rows; producer: module_CONTROL.F90:610, 640, 665, 685, 704).
// Output: DIV [LM, JM, IM].
//
// Index arithmetic is Fortran 1-based (i, j, ii, jj); the accessors convert to zero-based
// storage (IDX3(i-1, j-1, l0, IM, JM)). Each (i, j) is written by at most one branch.
//
// Fortran-to-CUDA review (every term of lines 290-451 is present):
//   - the global zeroing rows, the global row formulas, the regional s/n/w/e rows, the four
//     corner statements and the regional interior are each one branch below;
//   - wprp = wpdar(j) is read per row exactly where the Fortran reads it;
//   - the "*0.5" factors and every sign are kept; the operation order follows the Fortran.
//   - `first`, `restart` and the first-pass switch are outside lines 290-451 (uvupdate).
//   - no halo exchange, no OpenMP, no print: nothing skipped.
//
// Launch: kernels.grid_2d(IM, JM, LM), BLOCK_2D; blockIdx.z = level (l0 = l-1, model top = 0).

#include "nmmb_common.cuh"

#define DIVCOR_JW 1   // module_DYNAMICS_ROUTINES.F90:72 (jw = 01): rows w/o correction at the poles

// Indices into the drv-loop-limits table (namgpu/kernels/drv_loop_limits.cu, DL_*).
#define DLC_IDS    0
#define DLC_IDE    1
#define DLC_JDS    2
#define DLC_JDE    3
#define DLC_ITS    4
#define DLC_ITE    5
#define DLC_JTS    6
#define DLC_JTE    7
#define DLC_ITS_B1 8
#define DLC_ITE_B1 9
#define DLC_ITS_B2 10
#define DLC_ITE_B2 11
#define DLC_JTS_B2 14
#define DLC_JTE_B2 15

extern "C" __global__ void dyn_pgforce_divcor(
    const int* __restrict__ LIM,         // [NLIM] tile limits, 1-based Fortran values (DL_*)
    const float* __restrict__ PGX_3D,    // [LM, JM, IM]  pgf x component (pgx), from dyn-pgforce-pgf
    const float* __restrict__ PGY_3D,    // [LM, JM, IM]  pgf y component (pgy)
    const float* __restrict__ PGNE_3D,   // [LM, JM, IM]  pgf ne component (pgne)
    const float* __restrict__ PGNW_3D,   // [LM, JM, IM]  pgf nw component (pgnw)
    const float* __restrict__ WPDAR,     // [JM]          divergence correction weight per row (wpdar)
    float* __restrict__ DIV,             // [LM, JM, IM]  divergence, corrected in place
    int IM, int JM, int LM, int GLOBAL,  // GLOBAL = 1 for the global branch, 0 for regional
    int JOFF, int JMG)                   // row slab (nmmb_common.cuh): j below is GLOBAL, memory is local
{
    const int i  = blockIdx.x * blockDim.x + threadIdx.x + 1;   // Fortran i, 1..IM
    const int jl = blockIdx.y * blockDim.y + threadIdx.y + 1;   // local memory row, 1..JM
    const int j  = jl + JOFF;                                   // Fortran (global) j, 1..JMG
    const int l0 = blockIdx.z;                                  // Fortran l-1 (0 = top)
    if (i > IM || jl > JM || l0 >= LM) return;
    if (NMMB_SLAB_EDGE(jl - 1, JM, j - 1, JMG, 1)) return;     // reads j-1..j+1

    const int ids = LIM[DLC_IDS],  ide = LIM[DLC_IDE];
    const int jds = LIM[DLC_JDS],  jde = LIM[DLC_JDE];
    const int its = LIM[DLC_ITS],  ite = LIM[DLC_ITE];
    const int jts = LIM[DLC_JTS],  jte = LIM[DLC_JTE];
    const int its_b1 = LIM[DLC_ITS_B1], ite_b1 = LIM[DLC_ITE_B1];
    const int its_b2 = LIM[DLC_ITS_B2], ite_b2 = LIM[DLC_ITE_B2];
    const int jts_b2 = LIM[DLC_JTS_B2], jte_b2 = LIM[DLC_JTE_B2];
    const bool s_bdy = (jts == jds), n_bdy = (jte == jde);
    const bool w_bdy = (its == ids), e_bdy = (ite == ide);

#define PX(ii, jj)  PGX_3D[IDX3((ii) - 1, (jj) - 1 - JOFF, l0, IM, JM)]
#define PY(ii, jj)  PGY_3D[IDX3((ii) - 1, (jj) - 1 - JOFF, l0, IM, JM)]
#define PNE(ii, jj) PGNE_3D[IDX3((ii) - 1, (jj) - 1 - JOFF, l0, IM, JM)]
#define PNW(ii, jj) PGNW_3D[IDX3((ii) - 1, (jj) - 1 - JOFF, l0, IM, JM)]
#define WP(jj)      WPDAR[(jj) - 1 - JOFF]   /* wpdar is sliced to the local rows on the host */
#define DV(ii, jj)  DIV[IDX3((ii) - 1, (jj) - 1 - JOFF, l0, IM, JM)]

    const bool in_tile = (i >= its && i <= ite && j >= jts && j <= jte);

    if (GLOBAL) {
        // ---- global branch -----------------------------------------------------------------
        // zeroed rows: jds..jds-1+jw and jde-jw+1..jde, over the tile's i range
        const bool zero_lo = (j >= jds && j <= jds - 1 + DIVCOR_JW);
        const bool zero_hi = (j >= jde - DIVCOR_JW + 1 && j <= jde);
        if (in_tile && (zero_lo || zero_hi)) {
            DV(i, j) = 0.0f;
        }
        const bool i_ok = (i >= its_b1 && i <= ite_b1);
        const int jcl = jds + DIVCOR_JW;          // first corrected row
        const int jch = jde - DIVCOR_JW;          // last corrected row
        if (i_ok && j == jcl && jts <= jcl && jcl <= jte) {
            const float wprp = WP(j);
            DV(i, j) = (PX(i + 1, j) - PX(i, j)
                      + PY(i, j + 1)
                      - (PNE(i + 1, j + 1) + PNW(i, j + 1)) * 0.5f) * wprp;
        } else if (i_ok && j == jch && jts <= jch && jch <= jte) {
            const float wprp = WP(j);
            DV(i, j) = (PX(i + 1, j) - PX(i, j)
                      - PY(i, j)
                      - (-PNE(i, j) - PNW(i + 1, j)) * 0.5f) * wprp;
        } else if (i_ok && j >= jcl + 1 && j <= jch - 1 && jts <= j && j <= jte) {
            const float wprp = WP(j);
            DV(i, j) = (PX(i + 1, j) - PX(i, j)
                      + PY(i, j + 1) - PY(i, j)
                      - (PNE(i + 1, j + 1) - PNE(i, j)
                       + PNW(i, j + 1) - PNW(i + 1, j)) * 0.5f) * wprp;
        }
    } else {
        // ---- regional branch -----------------------------------------------------------------
        const int ip_w = ids + 1, ip_e = ide - 1;     // w_bdy column, e_bdy column
        const int jp_s = jds + 1, jp_n = jde - 1;     // s_bdy row, n_bdy row

        // s_bdy: row jp_s, i in its_b2..ite_b2
        if (s_bdy && j == jp_s && jts <= jp_s && jp_s <= jte && i >= its_b2 && i <= ite_b2) {
            const float wprp = WP(jp_s);
            DV(i, jp_s) = (PX(i + 1, jp_s) - PX(i, jp_s)
                         + PY(i, jp_s + 1)
                         - (PNE(i + 1, jp_s + 1) + PNW(i, jp_s + 1)) * 0.5f) * wprp;
        }
        // n_bdy: row jp_n, i in its_b2..ite_b2
        if (n_bdy && j == jp_n && jts <= jp_n && jp_n <= jte && i >= its_b2 && i <= ite_b2) {
            const float wprp = WP(jp_n);
            DV(i, jp_n) = (PX(i + 1, jp_n) - PX(i, jp_n)
                         - PY(i, jp_n)
                         - (-PNE(i, jp_n) - PNW(i + 1, jp_n)) * 0.5f) * wprp;
        }
        // w_bdy: column ip_w, j in jts_b2..jte_b2
        if (w_bdy && i == ip_w && its <= ip_w && ip_w <= ite && j >= jts_b2 && j <= jte_b2) {
            const float wprp = WP(j);
            DV(ip_w, j) = (PX(ip_w + 1, j)
                         + PY(ip_w, j + 1) - PY(ip_w, j)
                         - (PNE(ip_w + 1, j + 1) - PNW(ip_w + 1, j)) * 0.5f) * wprp;
        }
        // e_bdy: column ip_e, j in jts_b2..jte_b2
        if (e_bdy && i == ip_e && its <= ip_e && ip_e <= ite && j >= jts_b2 && j <= jte_b2) {
            const float wprp = WP(j);
            DV(ip_e, j) = (-PX(ip_e, j)
                         + PY(ip_e, j + 1) - PY(ip_e, j)
                         - (-PNE(ip_e, j) + PNW(ip_e, j + 1)) * 0.5f) * wprp;
        }
        // corner s&w: (ip_w, jp_s)
        if (s_bdy && w_bdy && i == ip_w && j == jp_s && in_tile) {
            const float wprp = WP(jp_s);
            DV(ip_w, jp_s) = (PX(ip_w + 1, jp_s)
                            + PY(ip_w, jp_s + 1)
                            - (PNE(ip_w + 1, jp_s + 1)) * 0.5f) * wprp;
        }
        // corner s&e: (ip_e, jp_s)
        if (s_bdy && e_bdy && i == ip_e && j == jp_s && in_tile) {
            const float wprp = WP(jp_s);
            DV(ip_e, jp_s) = (-PX(ip_e, jp_s)
                            + PY(ip_e, jp_s + 1)
                            - (PNW(ip_e, jp_s + 1)) * 0.5f) * wprp;
        }
        // corner n&w: (ip_w, jp_n)
        if (n_bdy && w_bdy && i == ip_w && j == jp_n && in_tile) {
            const float wprp = WP(jp_n);
            DV(ip_w, jp_n) = (PX(ip_w + 1, jp_n)
                            - PY(ip_w, jp_n)
                            - (-PNW(ip_w + 1, jp_n)) * 0.5f) * wprp;
        }
        // corner n&e: (ip_e, jp_n)
        if (n_bdy && e_bdy && i == ip_e && j == jp_n && in_tile) {
            const float wprp = WP(jp_n);
            DV(ip_e, jp_n) = (-PX(ip_e, jp_n)
                            - PY(ip_e, jp_n)
                            - (-PNE(ip_e, jp_n)) * 0.5f) * wprp;
        }
        // interior: i in its_b2..ite_b2, j in jts_b2..jte_b2
        if (i >= its_b2 && i <= ite_b2 && j >= jts_b2 && j <= jte_b2) {
            const float wprp = WP(j);
            DV(i, j) = (PX(i + 1, j) - PX(i, j)
                      + PY(i, j + 1) - PY(i, j)
                      - (PNE(i + 1, j + 1) - PNE(i, j)
                       + PNW(i, j + 1) - PNW(i + 1, j)) * 0.5f) * wprp;
        }
    }

#undef PX
#undef PY
#undef PNE
#undef PNW
#undef WP
#undef DV
}
