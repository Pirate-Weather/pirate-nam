// nam-gpu unit bnd-read-bc: the array-placement half of NMMB READ_BC (boundary input at the start
// of each 3-h interval). The file read and the header decode are host I/O in
// namgpu/boundary/bnd_read_bc.py; this kernel moves the staged record into the boundary arrays.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_fltbnds  (NMMB boundary I/O, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_FLTBNDS.F90:3579-3925 (nam.v4.2.11, 83df4e13c1).
//
// Fortran read_bc, line by line (single process, halo-free, one card: i_lo..i_hi = 1..IM and
// j_lo..j_hi = 1..JM because ims=ids, ime=ide, jms=jds, jme=jde):
//   3730-3739  open boco.NNNN (unit 51..59), read the header runbc, idatbc, ihrstbc, tboco
//              (I/O side).
//   3745-3777  2-D H vars  (nv = 1..nvars_bc_2d_h; record at 3754): one record = south(IM,lnsh,2),
//              north(IM,lnsh,2), west(lnsh,JM,2), east(lnsh,JM,2), column-major. Copied to
//              bnd_vars_h%var_2d(nv)%south(n1,n2,n3) and the others, n3 = time level 1..2.
//              south/north keep n1 in i_lo..i_hi, n2 in 1..lnsh; west/east keep n1 in 1..lnsh,
//              n2 in j_lo..j_hi.
//   3778-3817  3-D H vars  (nv = 1..nvars_bc_3d_h, storage slot nvx = n_bc_3d_h(nv); record at
//              3793): south(IM,lnsh,LM,2), north(...), west(lnsh,JM,LM,2), east(...). Destination
//              (n1,n2,n3=level,n4=time).
//   3818-3855  4-D H vars  (nv, nl = lbound..ubound of 4th index; record at 3829): one record per
//              nl, the same 3-D layout, destination (n1,n2,n3,n4,nl). The wrapper passes one entry
//              per (nv, nl).
//   3856-3886  2-D V vars  (record at 3864): lnsv instead of lnsh; same loop limits.
//   3887-3920  3-D V vars  (record at 3895): lnsv; same as 3-D H.
//   3921       close the unit.
//
// Staged record layout (what the I/O side writes into `rec`, per entry, float32, length PAYLOAD):
//   [0, S)            south  in Fortran column-major order: n1 fastest over IM, then n2 over LNSH,
//                     then level over NL, then time over NT=2
//   [S, 2S)           north  (same order)
//   [2S, 2S+W)        west   in column-major order: n1 over LNSH fastest, then j over JM,
//                     then level over NL, then time over NT=2
//   [2S+W, 2S+2W)     east   (same order)
//   S = IM*LNSH*NL*NT,  W = LNSH*JM*NL*NT.  For 2-D vars NL = 1.
//
// Destination arrays (device, C order, last index fastest, one per entry and side):
//   south/north: [NT][NL][LNSH][IM]   (Fortran (n1,n2,n3,n4) -> C [n4][n3][n2][n1])
//   west/east:   [NT][NL][JM][LNSH]   (CONTRACT 1a, fixed 2026-11-04; was [LNSH][JM]. n1 = line, j = along; the bnd-write-bc layout
//                                      [2, nspec, LM, lines=lnsh, along=JM] that init-boco-tendencies reads)
//
// Grid: blockIdx.z = entry, blockIdx.y = side (0 south, 1 north, 2 west, 3 east),
//       blockIdx.x * blockDim.x + threadIdx.x = linear index k within the side.
// One pure copy per element: no arithmetic, so no numerics to match; no host sync, no allocation.

#include "nmmb_common.cuh"

extern "C" __global__ void bnd_read_bc_scatter(
    const float* __restrict__ rec,                 // [nent, PAYLOAD] staged records (layout above)
    const unsigned long long* __restrict__ dst_s, // [nent] device address of south  [NT,NL,LNSH,IM]
    const unsigned long long* __restrict__ dst_n, // [nent] device address of north  [NT,NL,LNSH,IM]
    const unsigned long long* __restrict__ dst_w, // [nent] device address of west   [NT,NL,JM,LNSH]
    const unsigned long long* __restrict__ dst_e, // [nent] device address of east   [NT,NL,JM,LNSH]
    int im, int jm, int lnsh, int nl, int nt, long long payload)
{
    const int ent  = blockIdx.z;
    const int side = blockIdx.y;
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long S = (long long)im * lnsh * nl * nt;   // south and north element count
    const long long W = (long long)lnsh * jm * nl * nt;   // west and east element count
    const float* r = rec + (size_t)ent * (size_t)payload;

    if (side < 2) {
        if (k >= S) return;
        // Fortran column-major decode: n1 fastest, then n2, level, time.
        const int n1  = (int)(k % im);
        const int n2  = (int)((k / im) % lnsh);
        const int lev = (int)((k / ((long long)im * lnsh)) % nl);
        const int t   = (int)(k / ((long long)im * lnsh * nl));
        float* d = (float*)(side == 0 ? dst_s[ent] : dst_n[ent]);
        const size_t ci = (((size_t)t * nl + lev) * lnsh + n2) * im + n1;
        d[ci] = r[(side == 0 ? 0 : S) + k];
    } else {
        if (k >= W) return;
        const int n1  = (int)(k % lnsh);
        const int j   = (int)((k / lnsh) % jm);
        const int lev = (int)((k / ((long long)lnsh * jm)) % nl);
        const int t   = (int)(k / ((long long)lnsh * jm * nl));
        float* d = (float*)(side == 2 ? dst_w[ent] : dst_e[ent]);
        const size_t ci = (((size_t)t * nl + lev) * jm + j) * lnsh + n1;   // [NT][NL][JM][LNSH], CONTRACT 1a
        d[ci] = r[2 * S + (side == 2 ? 0 : W) + k];
    }
}
