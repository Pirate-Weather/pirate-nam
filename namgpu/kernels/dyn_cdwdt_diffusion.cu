// dyn_cdwdt_diffusion.cu -- lateral diffusion of w, added to DWDT (NMMB CDWDT, diffw branch).
//
// Notice: module_DYNAMICS_ROUTINES.F90 in NAM nam.v4.2.11 (MatthewPyle-NOAA/NAM @ 83df4e13c1,
// NMMB dynamics, NOAA/NCEP/EMC) has no author or history header block of its own; the NOAA
// source notice in NOTICE applies. NAM is US Government work with no licence file.
//
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:3553-3592 (nam.v4.2.11, 83df4e13c1).
//
// Fortran source covered (CDWDT, lateral diffusion of w, after the local derivative of w):
//   3556       if(diffw)   diffw = (dt .gt. 0), set at lines 3421-3424; the backward digital-filter
//              step (dt < 0) skips the whole block, which the kernel reproduces with an early return.
//   3558       do l=1,lm
//   w fluxes, on h points (wx, wy as 2-D scratch per level), over j=jts_b1..jte_h1, i=its_b1..ite_h1:
//     wx(i,j) = (w(i,j,l)-w(i-1,j,l))*(def(i-1,j,l)+def(i,j,l))
//     wy(i,j) = (w(i,j,l)-w(i,j-1,l))*(def(i,j-1,l)+def(i,j,l))
//   diffusion of w, over j=jts_b1..jte_b1, i=its_b1..ite_b1:
//     dwdt(i,j,l) = -((wx(i+1,j)-wx(i,j))*hdacx(i,j) + (wy(i,j+1)-wy(i,j))*hdacy(i,j))*rdt + dwdt(i,j,l)
//   rdt = 1./dt (line 3536, the same rdt as the local derivative).
//
// The wx/wy scratch is not stored: each thread evaluates the four face fluxes it needs
// (east/west wx, north/south wy) directly from W and DEF. Those fluxes depend only on W and DEF,
// never on DWDT, so the inline form is the same arithmetic as the two-pass Fortran.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM, jds=1 jde=JM):
//   its_b1 = 2, ite_b1 = IM-1, jts_b1 = 2, jte_b1 = JM-1 (1-based)
//   -> zero-based i0 in [1, IM-2], j0 in [1, JM-2]; the flux faces read i0-1..i0+1, j0-1..j0+1,
//      which lie inside the domain for every thread that passes the range test.
//   (its_h1 = IM, jte_h1 = JM in the wx/wy loops are covered by the same face expressions.)
//
// Layout (CONTRACT section 1): 3-D [l,j,i] via IDX3; 2-D hdacx/hdacy [j,i] via IDX2.
// One thread per (i, j, l) point (grid_2d, POINT_IJL). Each thread reads W, DEF and HDACX/HDACY
// and read-modify-writes only its own DWDT element, so there is no race.
// Fast numerics: fmad contraction is allowed.

#include "nmmb_common.cuh"

extern "C" __global__ void cdwdt_diffusion(
    const int IM, const int JM, const int LM,
    const float dt,                       // DT of the DYNAMICS call (diffw only when dt > 0)
    const float* __restrict__ w,          // W     [LM,JM,IM]
    const float* __restrict__ def,        // DEF   [LM,JM,IM] deformation (def3d from dyn-hdiff-coef)
    const float* __restrict__ hdacx,      // HDACX [JM,IM]
    const float* __restrict__ hdacy,      // HDACY [JM,IM]
    float* __restrict__ dwdt)             // DWDT  [LM,JM,IM] inout (local derivative in, diffused out)
{
    POINT_IJL   // i0, j0, l0 (l0 = blockIdx.z)

    // Fortran diffw = (dt .gt. 0): the backward digital-filter step does not diffuse w.
    if (!(dt > 0.0f)) return;

    // Fortran its_b1..ite_b1 and jts_b1..jte_b1 (1-based 2..IM-1, 2..JM-1)
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    const float rdt = 1.0f / dt;                       // Fortran rdt = 1./dt

    const size_t c  = IDX3(i0,     j0,     l0, IM, JM);
    const size_t ce = IDX3(i0 + 1, j0,     l0, IM, JM);
    const size_t cw = IDX3(i0 - 1, j0,     l0, IM, JM);
    const size_t cn = IDX3(i0,     j0 + 1, l0, IM, JM);
    const size_t cs = IDX3(i0,     j0 - 1, l0, IM, JM);
    const size_t h  = IDX2(i0, j0, IM);

    const float wc = w[c];

    // wx(i+1,j): east face of this cell; wx(i,j): west face
    const float wx_e = (w[ce] - wc) * (def[ce] + def[c]);
    const float wx_w = (wc - w[cw]) * (def[cw] + def[c]);
    // wy(i,j+1): north face of this cell; wy(i,j): south face
    const float wy_n = (w[cn] - wc) * (def[cn] + def[c]);
    const float wy_s = (wc - w[cs]) * (def[cs] + def[c]);

    dwdt[c] = -((wx_e - wx_w) * hdacx[h] + (wy_n - wy_s) * hdacy[h]) * rdt + dwdt[c];
}
