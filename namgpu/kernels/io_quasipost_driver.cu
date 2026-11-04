// io_quasipost_driver.cu -- QUASIPOST driver: in-model post fields for history.
//
// Ported to CUDA for nam-gpu from nmm/module_QUASIPOST.F90:68-255 (nam.v4.2.11, 83df4e13c1).
// The NAM source file has no header block of its own; the NOAA/NCEP/EMC notice is in NOTICE at the
// repo root (US Government work, no licence file).
//
// This unit is the driver only. quasipost() and quasipost_impl() call three routines that are
// separate units and are NOT in this file:
//   calculate_3D      (module_QUASIPOST.F90:257-686)  -> unit io-quasipost-3d
//   calculate_interP  (module_QUASIPOST.F90:688-815)  -> unit io-quasipost-interp
//   smoothMSLP        (module_QUASIPOST.F90:817-852)  -> unit io-quasipost-mslp
// The wrapper io/io_quasipost_driver.py runs them in Fortran order through their run(ctx).
// relax4e (module_RELAX4E.F90) is called inside the level loop; no unit in UNITS.json owns it, so
// its sweep is written here as qp_relax_sweep (see the map below).
//
// Fortran-to-CUDA map, every statement of 68-255:
//   68-83   quasipost: bounds ids=1 ide=IM jds=1 jde=JM kde=LM, one tile covering the domain.
//           The wrapper takes IM, JM, LM from ctx.grid. No MPI.
//   85-100  quasipost_impl. The three post_stdpres checks (k850=27 is 85000 Pa, k700=21 is
//           70000 Pa, k500=13 is 50000 Pa) are constant-table asserts; post_stdpres(k) =
//           20000 + 2500*(k-1) Pa makes them true by construction, so they are not
//           re-evaluated (the Fortran abort is unreachable). mpi_abort dropped.
//   ~108    HALO_EXCH(V10, U10, 3, 3) for winds, MPI halo (single card): dropped. Verified: the
//           consumers read only interior points (windloop i in ids+1..ide-2, the 10 m section
//           i in ids+1..ide-1, j likewise), so no read reaches past the domain edge. The
//           exchange has nothing to fill on one tile.
//   ~113    reduce(PINT, REDUCE_MIN) -> pmin: the result is never read (dead; the Fortran
//           FIXME says so). Dropped.
//   ~127    relaxmask = 1 over the full domain: qp_init.
//   ~130    ipres loop, descending 33 .. 1 (0-based 32 .. 0), with EXIT:
//             interior (1-based j in 2..JM-1, i in 2..IM-1) relaxmask = ground_mask(ipres):
//                 qp_level_set (skipped once the loop has exited)
//             need_to_relax = any interior ground_mask(ipres) /= 0, global max (one tile, so
//                 the max_integer call is the identity): atomicOr into anyflag
//             if need_to_relax == 0, EXIT the whole loop: qp_level_decide clears running (the
//                 exit is permanent, as in Fortran; relaxmask keeps the zero interior of this level)
//             relaxwork = presTv(ipres) over the full tile: qp_copy_work
//             relax4e(relaxwork, relaxmask, 0.7, 100): 100 Jacobi sweeps, ping-pong buffers
//                 (100 is even, so the result lands in relaxwork): qp_relax_sweep x 100
//             ground_mask(ipres) = relaxmask (full tile), presTv(ipres) = relaxwork:
//                 qp_finalize_level
//   ~250    calculate_interP -> io-quasipost-interp (Python, between kernels)
//           grid%membrane_MSLP = Pmsl over the tile: qp_copy_2d
//           smoothMSLP -> io-quasipost-mslp (Python; smooths MEMBRANE_MSLP with relaxmask = 1 and its
//                 own scratch; it does not read QP_RELAXMASK or QP_RELAXWORK)
//   ~250    p850z = presZ(k850), p700z = presZ(k700): qp_copy_level_2d. Fortran gates this on
//           size(grid%p850z)>1. That test is true whenever the field is allocated, and this port
//           always allocates P850Z and P700Z (st.add), so the copy is unconditional here.
//           presZ is the post-calculate_interP array (interP writes presZ in place, as Fortran does).
//
// relax4e (module_RELAX4E.F90), as one Jacobi sweep. The Fortran copies work into next, sweeps
// next from work, then copies next back to work. Here src is the work at the start of the sweep and
// dst is next; every point of dst is written in each sweep (points with mask==0 copy src), which is
// what the Fortran's initial next=work copy plus its masked updates produce. The relaxation mask is
// intent(in) in relax4e, so the sweep does not change it. With r = relax coefficient, r1 = 1 - r:
//   next = r1*w + r*(W + S + E + N)/4          (1-based i in 2..IM-1, j in 2..JM-1, masked)
// Edges (single tile): south j=1 (W+E+N)/3; north j=JM (W+E+S)/3; west i=1 (E+S+N)/3;
// east i=IM (W+S+N)/3; corners SW (E+N)/2, SE (W+N)/2, NW (E+S)/2, NE (W+S)/2, each masked and
// each r1*w + r*(sum)/n. W = (i-1,j), E = (i+1,j), S = (i,j-1), N = (i,j+1).
//
// Domain: ide = IM and jde = JM. Verified against module_DM_PARALLEL.F90 (non-global branch sets
// ide = im, jde = jm; the NAM run is regional, global false). The CUDA index bounds rely on this.
//
// Layout (CONTRACT.md): 2-D [JM,IM] float or int; level arrays [npres,JM,IM] with level 0 the first
// pressure level (20000 Pa). IDX2/IDX3 come from nmmb_common.cuh. One thread per (i,j) with
// grid_2d; no host sync, no device malloc, no printf. running[0] and anyflag[0] are device scalars,
// so the exit decision never reaches the host.

#include "nmmb_common.cuh"

// ---------------------------------------------------------------------------
// relaxmask = 1 over the full tile; running = 1; anyflag = 0 (the exit flags start clear).
extern "C" __global__ void qp_init(
    const int IM, const int JM,
    int* __restrict__ relaxmask,      // [JM,IM]
    int* __restrict__ running,        // [1]
    int* __restrict__ anyflag)        // [1]
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    relaxmask[IDX2(i0, j0, IM)] = 1;
    if (i0 == 0 && j0 == 0) {
        running[0] = 1;
        anyflag[0] = 0;
    }
}

// ---------------------------------------------------------------------------
// Fortran 'do j=max(jps,jds+1),min(jde-1,jpe)' / 'do i=...': the interior of the tile
// (i0 in 1..IM-2, j0 in 1..JM-2) takes relaxmask = ground_mask(ipres). A nonzero value sets
// anyflag (the need_to_relax global max). Skipped once the loop has exited.
extern "C" __global__ void qp_level_set(
    const int IM, const int JM,
    const int ipres0,                     // 0-based level (Fortran ipres - 1)
    const int* __restrict__ ground_mask,  // [npres,JM,IM]
    int* __restrict__ relaxmask,          // [JM,IM]
    const int* __restrict__ running,      // [1]
    int* __restrict__ anyflag)            // [1]
{
    if (running[0] == 0) return;
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;
    const int gm = ground_mask[IDX3(i0, j0, ipres0, IM, JM)];
    relaxmask[IDX2(i0, j0, IM)] = gm;
    if (gm != 0) atomicOr(anyflag, 1);
}

// ---------------------------------------------------------------------------
// 'if(need_to_relax==0) exit': one thread. The exit is permanent.
extern "C" __global__ void qp_level_decide(
    int* __restrict__ running,            // [1]
    const int* __restrict__ anyflag)      // [1]
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (running[0] != 0 && anyflag[0] == 0) running[0] = 0;
    }
}

// ---------------------------------------------------------------------------
// relaxwork = presTv(:,:,ipres) over the tile.
extern "C" __global__ void qp_copy_work(
    const int IM, const int JM,
    const int ipres0,
    const float* __restrict__ presTv,     // [npres,JM,IM]
    float* __restrict__ relaxwork,        // [JM,IM]
    const int* __restrict__ running)      // [1]
{
    if (running[0] == 0) return;
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    relaxwork[IDX2(i0, j0, IM)] = presTv[IDX3(i0, j0, ipres0, IM, JM)];
}

// ---------------------------------------------------------------------------
// One Jacobi sweep of relax4e (see the header for the stencil and the edge cases).
extern "C" __global__ void qp_relax_sweep(
    const int IM, const int JM,
    const float r,                        // relax coefficient (0.7 in the driver)
    const int* __restrict__ mask,         // [JM,IM] relaxmask (intent(in) in relax4e)
    const float* __restrict__ src,        // [JM,IM] work at sweep start
    float* __restrict__ dst,              // [JM,IM] next
    const int* __restrict__ running)      // [1]
{
    if (running[0] == 0) return;
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    const float r1 = 1.0f - r;
    const size_t p = IDX2(i0, j0, IM);
    const float w = src[p];
    float out = w;
    if (mask[p] != 0) {
        const bool iin = (i0 >= 1 && i0 <= IM - 2);
        const bool jin = (j0 >= 1 && j0 <= JM - 2);
        // Neighbour reads are guarded: each one is only evaluated on the branch that uses it.
        if (iin && jin) {
            const float W = src[IDX2(i0 - 1, j0, IM)];
            const float S = src[IDX2(i0, j0 - 1, IM)];
            const float E = src[IDX2(i0 + 1, j0, IM)];
            const float N = src[IDX2(i0, j0 + 1, IM)];
            out = r1 * w + r * (W + S + E + N) / 4.0f;
        } else if (iin && j0 == 0) {                 // SOUTH
            const float W = src[IDX2(i0 - 1, j0, IM)];
            const float E = src[IDX2(i0 + 1, j0, IM)];
            const float N = src[IDX2(i0, j0 + 1, IM)];
            out = r1 * w + r * (W + E + N) / 3.0f;
        } else if (iin && j0 == JM - 1) {            // NORTH
            const float W = src[IDX2(i0 - 1, j0, IM)];
            const float E = src[IDX2(i0 + 1, j0, IM)];
            const float S = src[IDX2(i0, j0 - 1, IM)];
            out = r1 * w + r * (W + E + S) / 3.0f;
        } else if (jin && i0 == 0) {                 // WEST
            const float E = src[IDX2(i0 + 1, j0, IM)];
            const float S = src[IDX2(i0, j0 - 1, IM)];
            const float N = src[IDX2(i0, j0 + 1, IM)];
            out = r1 * w + r * (E + S + N) / 3.0f;
        } else if (jin && i0 == IM - 1) {            // EAST
            const float W = src[IDX2(i0 - 1, j0, IM)];
            const float S = src[IDX2(i0, j0 - 1, IM)];
            const float N = src[IDX2(i0, j0 + 1, IM)];
            out = r1 * w + r * (W + S + N) / 3.0f;
        } else if (i0 == 0 && j0 == 0) {             // SOUTHWEST
            const float E = src[IDX2(i0 + 1, j0, IM)];
            const float N = src[IDX2(i0, j0 + 1, IM)];
            out = r1 * w + r * (E + N) / 2.0f;
        } else if (i0 == IM - 1 && j0 == 0) {        // SOUTHEAST
            const float W = src[IDX2(i0 - 1, j0, IM)];
            const float N = src[IDX2(i0, j0 + 1, IM)];
            out = r1 * w + r * (W + N) / 2.0f;
        } else if (i0 == 0 && j0 == JM - 1) {        // NORTHWEST
            const float E = src[IDX2(i0 + 1, j0, IM)];
            const float S = src[IDX2(i0, j0 - 1, IM)];
            out = r1 * w + r * (E + S) / 2.0f;
        } else if (i0 == IM - 1 && j0 == JM - 1) {   // NORTHEAST
            const float W = src[IDX2(i0 - 1, j0, IM)];
            const float S = src[IDX2(i0, j0 - 1, IM)];
            out = r1 * w + r * (W + S) / 2.0f;
        }
    }
    dst[p] = out;
}

// ---------------------------------------------------------------------------
// ground_mask(:,:,ipres) = relaxmask and presTv(:,:,ipres) = relaxwork over the tile.
extern "C" __global__ void qp_finalize_level(
    const int IM, const int JM,
    const int ipres0,
    const int* __restrict__ relaxmask,    // [JM,IM]
    int* __restrict__ ground_mask,        // [npres,JM,IM]
    const float* __restrict__ relaxwork,  // [JM,IM]
    float* __restrict__ presTv,           // [npres,JM,IM]
    const int* __restrict__ running)      // [1]
{
    if (running[0] == 0) return;
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    const size_t p = IDX2(i0, j0, IM);
    ground_mask[IDX3(i0, j0, ipres0, IM, JM)] = relaxmask[p];
    presTv[IDX3(i0, j0, ipres0, IM, JM)] = relaxwork[p];
}

// ---------------------------------------------------------------------------
// grid%membrane_MSLP = Pmsl over the tile (any [JM,IM] float copy).
extern "C" __global__ void qp_copy_2d(
    const int IM, const int JM,
    const float* __restrict__ src,        // [JM,IM]
    float* __restrict__ dst)              // [JM,IM]
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    dst[IDX2(i0, j0, IM)] = src[IDX2(i0, j0, IM)];
}

// ---------------------------------------------------------------------------
// p850z = presZ(:,:,k850) and p700z = presZ(:,:,k700); k0 is the 0-based level index
// (k850 = 27 -> 26, k700 = 21 -> 20).
extern "C" __global__ void qp_copy_level_2d(
    const int IM, const int JM,
    const int k0,
    const float* __restrict__ src3,       // [npres,JM,IM]
    float* __restrict__ dst2)             // [JM,IM]
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    dst2[IDX2(i0, j0, IM)] = src3[IDX3(i0, j0, k0, IM, JM)];
}
