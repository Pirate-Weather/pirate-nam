// bnd_write_bc.cu -- nam-gpu unit bnd-write-bc (CONTRACT.md sections 3 and 6).
//
// Ported to CUDA for nam-gpu from nmm/module_FLTBNDS.F90:3928-5059 (write_bc, nam.v4.2.11, 83df4e13c1).
// NOAA/NCEP NMMB source: US government work, the NOAA author headers are kept as notices (NOTICE).
//
// write_bc, per boundary side (N, S, W, E) and per boundary variable:
//   if recomp_tend:  targ = bnd(slot 1) + tboco * bnd(slot 2)            (Fortran: targ_*_n etc.)
//   always:          bnd(slot 1) = state edge strip                       (Fortran: bnd_vars_*%north(...,1))
//   if recomp_tend:  bnd(slot 2) = (targ - bnd(slot 1)) / tboco          (Fortran: ...%north(...,2))
//
// The Fortran runs these as separate loops over a temporary targ array. Each buffer element is
// read and written by exactly one thread here, so the three steps are fused per element: the old
// slot 1 and slot 2 are read before either is written, and the new slot 1 is the state value f.
// That gives exactly the Fortran values, with no temporary.
//
// Buffer layout (one per side and variable, see namgpu/boundary/bnd_write_bc.py):
//   buf[t][s][l][line][a], t = 0,1 (slot), s = species (1 for 2-D/3-D), l = level (LM),
//   line = strip line (lnsh for H, lnsv for V), a = along-strip index (IM for N/S, JM for W/E).
// Field layout: field[s][l][j][i] (CONTRACT section 1), i0 = i-1, j0 = j-1, l0 = l-1.
//
// Strip line mapping, from write_bc (1-based Fortran -> zero-based line0 passed in):
//   side 0 (N): j = line0 + line, i = a         side 1 (S): j = line0 + line, i = a
//   side 2 (W): i = line0 + line, j = a         side 3 (E): i = line0 + line, j = a
#include "nmmb_common.cuh"

extern "C" __global__ void bcw_strip(
    const float* __restrict__ field,   // state field [nspec, LM, JM, IM] (2-D field: LM = 1, nspec = 1)
    float* __restrict__ buf,           // boundary buffer for this side and variable
    int IM, int JM, int LM, int nspec,
    int nlines, int along, int line0, int side,
    int recomp, float tboco)
{
    const int a    = blockIdx.x * blockDim.x + threadIdx.x;   // along-strip index
    const int line = blockIdx.y;                              // strip line
    const int sl   = blockIdx.z;                              // species * LM + level
    if (a >= along || line >= nlines) return;

    const int s = sl / LM;
    const int l = sl - s * LM;

    int i0, j0;
    if (side < 2) {                       // N, S: strip is a row of the domain
        i0 = a;
        j0 = line0 + line;
    } else {                              // W, E: strip is a column of the domain
        i0 = line0 + line;
        j0 = a;
    }

    const float f = field[IDX4(i0, j0, l, s, IM, JM, LM)];

    const size_t ofs = (((size_t)s * LM + l) * nlines + line) * along + a;     // slot 1
    const size_t tst = (size_t)nspec * LM * nlines * along;                    // slot stride (slot 2 = ofs + tst)

    if (recomp) {
        const float targ = buf[ofs] + tboco * buf[ofs + tst];
        buf[ofs] = f;
        buf[ofs + tst] = (targ - f) / tboco;
    } else {
        buf[ofs] = f;
    }
}
