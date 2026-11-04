// phys_fer_module_data.cu -- MP_FER_HIRES module constants, published to device vectors.
//
// NOTICE: NAM source file phys/module_MP_FER_HIRES.F90:1-207 (NCEP/EMC, US Government work, no
// licence block; changelog kept as the notice in fer_hires.cuh).
// Ported to CUDA for WOOF-NAM from phys/module_MP_FER_HIRES.F90:1-207 (nam.v4.2.11, 83df4e13c1).
//
// Lines 1-207 contain no executable statements: they are the module header, the USE list, PUBLIC /
// PRIVATE statements, PARAMETER declarations and SAVE declarations. Every PARAMETER maps to a macro in
// fer_hires.cuh (the statement map there names each line range).
//
// Two kinds of output:
//   1. Constants (PARAMETERs). fer_module_data_publish writes the REAL PARAMETER values to a float32
//      vector (FER_REAL_CONSTANT_LIST order) and the INTEGER PARAMETER values to an int32 vector
//      (FER_INT_CONSTANT_LIST order). Both are immutable.
//   2. Mutable SAVE state. Pwarn and QTwarn (Fortran line 172) are rewritten by the WARN4 path, so they
//      are not constants. fer_warn_state_init writes their initial values into a float32[2] state vector,
//      which the WARN4 path (phys-fer-fer-hires) updates in place.
//
// Both kernels are scalar: one thread, one launch, no host sync, no allocation.

#include "nmmb_common.cuh"
#include "fer_hires.cuh"

extern "C" __global__ void fer_module_data_publish(float* __restrict__ out, int* __restrict__ out_i)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int k = 0;
#define FER_PUBLISH(name) out[k++] = FER_##name;
    FER_REAL_CONSTANT_LIST(FER_PUBLISH)
#undef FER_PUBLISH
    int m = 0;
#define FER_PUBLISH_I(name) out_i[m++] = FER_##name;
    FER_INT_CONSTANT_LIST(FER_PUBLISH_I)
#undef FER_PUBLISH_I
}

// Initial values of the mutable SAVE state. Launched once, when the state vector is created.
extern "C" __global__ void fer_warn_state_init(float* __restrict__ warn_state)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    warn_state[0] = FER_PWARN_INIT;   // Pwarn
    warn_state[1] = FER_QTWARN_INIT;  // QTwarn
}
