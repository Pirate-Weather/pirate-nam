// nam-gpu unit init-gfs-wind-rotate: GFS earth-relative (east, north) winds to the rotated
// B-grid components (grid-i, grid-j) at V points, on the boundary strips only.
//
// Mode "new": there is no NAM Fortran routine for this step in the NMMB solver tree or in the NPS
// preprocessor sources (nam_nps.fd/sorc/NMMB_init has no earth-to-grid rotation). Written from
// CONTRACT.md section 0 and the rotated-pole geometry of namgpu/grid.py (rotated_to_geographic).
// No NOAA header is carried; the notice is in NOTICE.
//
// Math (per V point, level-independent; the map is linear, so rotating the strip values equals
// interpolating the rotated winds):
//   u_grid = UE * ce_e + VE * ce_n        (ce = unit vector of increasing grid i, in (east, north))
//   v_grid = UE * cn_e + VE * cn_n        (cn = unit vector of increasing grid j)
// The four coefficients are built once on the host from the grid geometry (wrapper
// init_gfs_wind_rotate.py) and are [JM, IM] arrays.
//
// Input: the GFS UE/VE boundary strips on the V-point grid, produced by init-gfs-hinterp's
// interpolation (interp_strips) with the same strip layout:
//   south/north [LM][W][IM]   (j = r for south, j = JM-W+r for north)
//   west/east   [LM][JM][W]   (i = c for west,  i = IM-W+c for east)
// Output: st.U and st.V, written ONLY on the strip rows/columns (CONTRACT section 1: boundary units
// own the edge rows; interior U/V belong to other units and are not touched). The last row and last
// column of V are unused by the contract and are not written.
//
// Grid: blockIdx.x over the points of one side (ceil(max(W*IM, JM*W)/256)), blockIdx.y = side
// (0 south, 1 north, 2 west, 3 east), blockIdx.z = level l0 (LM). One thread per strip point.
// Corner points belong to two strips; both write the same value (same interpolated coordinates).

#include "nmmb_common.cuh"

extern "C" __global__ void init_gfs_wind_rotate_strips(
    const int IM, const int JM, const int LM, const int W,
    const float* __restrict__ ce_e,     // [JM,IM] east component of grid-i unit vector
    const float* __restrict__ ce_n,     // [JM,IM] north component of grid-i unit vector
    const float* __restrict__ cn_e,     // [JM,IM] east component of grid-j unit vector
    const float* __restrict__ cn_n,     // [JM,IM] north component of grid-j unit vector
    const float* __restrict__ ue_s, const float* __restrict__ ue_n,   // [LM,W,IM]
    const float* __restrict__ ue_w, const float* __restrict__ ue_e,   // [LM,JM,W]
    const float* __restrict__ ve_s, const float* __restrict__ ve_n,   // [LM,W,IM]
    const float* __restrict__ ve_w, const float* __restrict__ ve_e,   // [LM,JM,W]
    float* __restrict__ u,              // [LM,JM,IM] out: grid-relative wind, grid-i (strips only)
    float* __restrict__ v)              // [LM,JM,IM] out: grid-relative wind, grid-j (strips only)
{
    const int side = blockIdx.y;
    const int l0 = blockIdx.z;
    if (l0 >= LM) return;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;

    int i0, j0;
    size_t sidx;
    float uev, vnv;
    if (side < 2) {
        if (p >= W * IM) return;
        const int r = p / IM;
        i0 = p % IM;
        j0 = (side == 0) ? r : (JM - W + r);
        sidx = ((size_t)l0 * W + r) * IM + i0;
        const float* uu = (side == 0) ? ue_s : ue_n;
        const float* vv = (side == 0) ? ve_s : ve_n;
        uev = uu[sidx];
        vnv = vv[sidx];
    } else {
        if (p >= JM * W) return;
        j0 = p / W;
        const int c = p % W;
        i0 = (side == 2) ? c : (IM - W + c);
        sidx = ((size_t)l0 * JM + j0) * W + c;
        const float* uu = (side == 2) ? ue_w : ue_e;
        const float* vv = (side == 2) ? ve_w : ve_e;
        uev = uu[sidx];
        vnv = vv[sidx];
    }
    if (j0 >= JM - 1 || i0 >= IM - 1) return;   // last row / column: unused (CONTRACT section 1)

    const size_t c2 = IDX2(i0, j0, IM);
    const size_t c3 = IDX3(i0, j0, l0, IM, JM);
    u[c3] = uev * ce_e[c2] + vnv * ce_n[c2];
    v[c3] = uev * cn_e[c2] + vnv * cn_n[c2];
}
