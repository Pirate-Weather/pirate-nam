// nam-gpu unit io-quasipost-mslp: smoothMSLP (Shuell/NAM membrane MSLP smoothing).
//
// Ported to CUDA for nam-gpu from nmm/module_QUASIPOST.F90:817-852 (nam.v4.2.11, 83df4e13c1).
// module_QUASIPOST.F90 carries no author header block (it starts at "module module_quasipost"),
// so no NOAA header is carried here; the source is in the NOTICE.
//
// Fortran semantics reproduced (smoothMSLP + module_RELAX4E.F90 relax4e, one relaxation pass):
//   smoothMSLP: relaxmask(i,j)=1 everywhere, relaxwork(i,j)=membrane_mslp(i,j), then
//               call relax4e(relaxwork, relaxmask, 0.5, iterations=1, ...), then
//               membrane_mslp(i,j)=relaxwork(i,j).
//   relax4e  : next = work (copy); one Jacobi pass over the interior and the four edges and
//              four corners of the domain, reading the OLD work values; then work = next.
//
// Single process, whole domain on one card (CONTRACT.md section 1): ids=1, ide=IM, jds=1, jde=JM,
// its=ids, ite=ide, jts=jds, jte=jde, so the halo exchange HALO_EXCH is dropped, and every
// "if(its<=ids)", "if(ite>=ide)", "if(jts<=jds)", "if(jte>=jde)" test is true. The loop
// limits i1=max(its,ids+1)=2, i2=min(ite,ide-1)=IM-1 and j1=2, j2=JM-1 (Fortran 1-based) are
// zero-based i0 in [1, IM-2], j0 in [1, JM-2]. relaxmask is 1 at every point by construction in
// smoothMSLP, so every "if(mask(i,j)/=0)" test is true and the mask array is not read.
//
// Arithmetic order is the Fortran order: r*(sum)/4, r*(sum)/3, r*(sum)/2 with r = 0.5 and
// r1 = 1 - r. Single precision (kfpt single). FMA is left on (fast numerics).
//
// Two kernels, one launch each, no host sync:
//   io_quasipost_mslp_relax  reads work (MEMBRANE_MSLP) and writes next (scratch), every point.
//   io_quasipost_mslp_copy   copies next back into work (MEMBRANE_MSLP), every point.
//
// Scratch "next" is the relax4e local array (its:ite,jts:jte), which is the full [JM, IM] domain
// here. The wrapper owns it (namgpu/io/io_quasipost_mslp.py).

#include "nmmb_common.cuh"

#define QPM_R      0.5f
#define QPM_R1     (1.0f - QPM_R)

extern "C" __global__ void io_quasipost_mslp_relax(
    const int IM, const int JM,
    const float* __restrict__ work,     // [JM,IM] in: membrane_mslp (Pa), relaxwork
    float* __restrict__ next)           // [JM,IM] out: one relaxation pass of work
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;

    const size_t c = IDX2(i0, j0, IM);
    const float wc = work[c];
    float out = wc;   // relax4e initj copy: points not updated below keep the old value

    // Fortran (i,j) = (i0+1, j0+1); neighbours: W = i0-1, E = i0+1, S = j0-1, N = j0+1.
    const bool interior_i = (i0 >= 1 && i0 <= IM - 2);
    const bool interior_j = (j0 >= 1 && j0 <= JM - 2);

    if (interior_i && interior_j) {
        // bigj/bigi loop: r1*work + r*(W + S + E + N)/4
        const float s = work[IDX2(i0 - 1, j0, IM)] + work[IDX2(i0, j0 - 1, IM)]
                      + work[IDX2(i0 + 1, j0, IM)] + work[IDX2(i0, j0 + 1, IM)];
        out = QPM_R1 * wc + QPM_R * s / 4.0f;
    } else if (j0 == 0 && interior_i) {
        // SOUTH (j=1): r1*work + r*(W + E + N)/3
        const float s = work[IDX2(i0 - 1, j0, IM)] + work[IDX2(i0 + 1, j0, IM)]
                      + work[IDX2(i0, j0 + 1, IM)];
        out = QPM_R1 * wc + QPM_R * s / 3.0f;
    } else if (j0 == JM - 1 && interior_i) {
        // NORTH (j=jde): r1*work + r*(W + E + S)/3
        const float s = work[IDX2(i0 - 1, j0, IM)] + work[IDX2(i0 + 1, j0, IM)]
                      + work[IDX2(i0, j0 - 1, IM)];
        out = QPM_R1 * wc + QPM_R * s / 3.0f;
    } else if (i0 == 0 && interior_j) {
        // WEST (i=1): r1*work + r*(E + S + N)/3
        const float s = work[IDX2(i0 + 1, j0, IM)] + work[IDX2(i0, j0 - 1, IM)]
                      + work[IDX2(i0, j0 + 1, IM)];
        out = QPM_R1 * wc + QPM_R * s / 3.0f;
    } else if (i0 == IM - 1 && interior_j) {
        // EAST (i=ide): r1*work + r*(W + S + N)/3
        const float s = work[IDX2(i0 - 1, j0, IM)] + work[IDX2(i0, j0 - 1, IM)]
                      + work[IDX2(i0, j0 + 1, IM)];
        out = QPM_R1 * wc + QPM_R * s / 3.0f;
    } else if (i0 == 0 && j0 == 0) {
        // SOUTHWEST (ids,jds): r1*work + r*(E + N)/2
        const float s = work[IDX2(i0 + 1, j0, IM)] + work[IDX2(i0, j0 + 1, IM)];
        out = QPM_R1 * wc + QPM_R * s / 2.0f;
    } else if (i0 == IM - 1 && j0 == 0) {
        // SOUTHEAST (ide,jds): r1*work + r*(W + N)/2
        const float s = work[IDX2(i0 - 1, j0, IM)] + work[IDX2(i0, j0 + 1, IM)];
        out = QPM_R1 * wc + QPM_R * s / 2.0f;
    } else if (i0 == 0 && j0 == JM - 1) {
        // NORTHWEST (ids,jde): r1*work + r*(E + S)/2
        const float s = work[IDX2(i0 + 1, j0, IM)] + work[IDX2(i0, j0 - 1, IM)];
        out = QPM_R1 * wc + QPM_R * s / 2.0f;
    } else if (i0 == IM - 1 && j0 == JM - 1) {
        // NORTHEAST (ide,jde): r1*work + r*(W + S)/2
        const float s = work[IDX2(i0 - 1, j0, IM)] + work[IDX2(i0, j0 - 1, IM)];
        out = QPM_R1 * wc + QPM_R * s / 2.0f;
    }

    next[c] = out;
}

extern "C" __global__ void io_quasipost_mslp_copy(
    const int IM, const int JM,
    const float* __restrict__ next,     // [JM,IM] in
    float* __restrict__ work)           // [JM,IM] out: membrane_mslp (relax4e back copy)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;
    const size_t c = IDX2(i0, j0, IM);
    work[c] = next[c];
}
