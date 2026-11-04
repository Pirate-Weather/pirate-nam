// phys_myj_init.cu -- unit phys-myj-init: MYJPBL_INIT (zero the MYJ exchange coefficient at init).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone), module
//   module_BL_MYJPBL (Janjic Mellor-Yamada-Janjic PBL), nam.v4.2.11.
// Ported to CUDA for WOOF-NAM from phys/module_BL_MYJPBL.F90:1612-1657 (nam.v4.2.11, 83df4e13c1).
//
// Term-by-term review of MYJPBL_INIT (1612-1657):
//   args EXCH_H (INTENT OUT, IMS:IME x JMS:JME x 1:LM), RESTART, IDS..LM, IMS..JME, ITS..JTE.
//   JTF = MIN(JTE, JDE-1), KTF = LM, ITF = MIN(ITE, IDE-1)   -> full domain: JTF = JM-1, ITF = IM-1.
//   IF (.NOT. RESTART): EXCH_H(I,J,K) = 0 for K = 1..KTF, J = JTS..JTF, I = ITS..ITF.
//     -> kernel below: one thread per (i0, j0) with the level in blockIdx.z; the guard
//        i0 < IM-1 and j0 < JM-1 is ITF/JTF (the last row and column are not written, as in Fortran).
//     -> the RESTART test is the wrapper's launch decision (no launch on restart).
//   CALL HERFTBL, TABLEPT, TABLETT (unconditional in Fortran) -> phys-myj-tables, launched by the
//     wrapper (phys_myj_init.run) right after this kernel, on the same stream. Not in this file.
//
// Parallelism: one thread per (i, j) column-point, level in grid.z (kernels.grid_2d). No host sync.
// Layout: Fortran EXCH_H(i,j,l) -> CuPy [l, j, i] (IDX3, nmmb_common.cuh). The CuPy array is
// [LM+1, JM, IM] (state.py LEVELS_LMP1); IDX3 strides use only IM and JM, so the extra level is inert.
// Levels 0..LM-1 are the Fortran K = 1..LM range this kernel writes; level LM is not written.

#include "nmmb_common.cuh"

extern "C" __global__ void phys_myj_init_exch_h(float* __restrict__ exch_h,   // [LM+1, JM, IM]
                                                const int im,
                                                const int jm,
                                                const int lm)
{
    POINT_IJL                                   // i0, j0, l0 = blockIdx.z
    if (i0 >= im - 1 || j0 >= jm - 1 || l0 >= lm) return;   // ITF = IM-1, JTF = JM-1, KTF = LM
    exch_h[IDX3(i0, j0, l0, im, jm)] = 0.0f;
}
