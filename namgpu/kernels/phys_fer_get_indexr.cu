// phys_fer_get_indexr.cu -- WOOF-NAM unit phys-fer-get-indexr: GET_INDEXR (rain-size table index).
//
// NOTICE: NOAA/NCEP NAM source module_MP_FER_HIRES.F90 (nam.v4.2.11, 83df4e13c1), a US government work
// with no licence file in the NAM clone. The module header block is reproduced verbatim below
// (Fortran lines 1-30, "!" comments written as "//"). GET_INDEXR itself is Fortran lines 2297-2380.
//
// ---- begin NOAA module header (module_MP_FER_HIRES.F90 lines 1-30) ----
//
//-- "Modified" fer_hires microphysics - 11 July 2016 version
//
// (1) Ice nucleation: Fletcher (1962) replaces Meyers et al. (1992)
// (2) Cloud ice is a simple function of the number concentration from (1), and it
//     is no longer a fractional function of the large ice.  Thus, the FLARGE &
//     FSMALL parameters are no longer used.
// (3) T_ICE_init=-12 deg C provides a slight delay in the initial onset of ice.
// (4) NLImax is a function of rime factor (RF) and temperature.  
//    a) For RF>10, NLImax=1.e3.  Mean ice diameters can exceed the 1 mm maximum
//       size in the tables so that NLICE=NLImax=1.e3.
//    b) Otherwise, NLImax is 10 L-1 at 0C and decreasing to 5 L-1 at <=-40C.  
//       NLICE>NLImax at the maximum ice diameter of 1 mm.
// (5) Can turn off ice processes by setting T_ICE & T_ICE_init to be < -100 deg C
// (6) Modified the homogeneous freezing of cloud water when T<T_ICE.
// (7) Reduce the fall speeds of rimed ice by making VEL_INC a function of 
//     VrimeF**1.5 and not VrimeF**2.
// (8) RHgrd=0.98 (98% RH) for the onset of condensation, to match what's been
//     tested for many months in the NAMX.  Made obsolete with change in (13).
// (9) Rime factor is *never* adjusted when NLICE>NLImax.  
// (10) Ice deposition does not change the rime factor (RF) when RF>=10 & T>T_ICE.
// (11) Limit GAMMAS to <=1.5 (air resistance impact on ice fall speeds)
// (12) NSImax is maximum # conc of ice crsytals.  At cold temperature NSImax is
//      calculated based on assuming 10% of total ice content is due to cloud ice.
//
//-- Further modifications starting on 23 July 2015
// (13) RHgrd is passed in as an input argument so that it can vary for different
//      domains (RHgrd=0.98 for 12-km parent, 1.0 for 3-km nests)
// (14) Use the old "PRAUT" cloud water autoconversion *threshold* (QAUT0)

// ---- end NOAA module header ----
// and later revisions).
//
// Ported to CUDA for WOOF-NAM from phys/module_MP_FER_HIRES.F90:2297-2380 (nam.v4.2.11, 83df4e13c1).
//
// Fortran: INTEGER FUNCTION GET_INDEXR(RR) returns the mean rain drop diameter in microns for a rain
// rate RR (kg m-2 s-1). It is a piecewise power-law fit of the rain-table diameter against RR, clamped
// to the drop-size bin of each segment. The CUDA form is elementwise: one thread per element of RR,
// result written to INDEXR. The seven rain-rate breakpoints RR_DRmin, RR_DR1..RR_DR5, RR_DRmax are
// module variables that FER_HIRES_INIT computes from N0r0*RRATE(...) (Fortran line 2615-2621); they
// are passed in as thr[0..6] in that order, so this kernel does not recompute the rain table.
//
// Constants (module_MP_FER_HIRES.F90 lines 136-168, all REAL-to-INTEGER parameters, truncated as in
// Fortran): MDRmin = XMRmin = 50, MDR1 = 100, MDR2 = 200, MDR3 = 320, MDR4 = 450, MDR5 = 670,
// MDRmax = XMRmax = 1000 (microns). Values are checked by tools/check-phys-fer-get-indexr.py (constants asserted; GET_INDEXR replayed against the Fortran referee in tools/fer-get-indexr-ref/).
//
// Built with --fmad=true (fast numerics, namgpu/kernels/__init__.py): a*b+0.5f may contract to FMA, which can
// move an INT(x+.5) boundary by one count versus gfortran. Measured by the replay, not claimed bitwise.
//
// Fortran INT(x + .5) truncates toward zero, as does (int) in C; the arguments are positive in every
// segment that reaches INT, so this is floor(x + .5). Fortran IF/ELSE IF chain is kept in order, so a
// NaN RR falls through to the last ELSE (MDRmax), as in the source.
#include "nmmb_common.cuh"

#define NR_MDRMIN 50
#define NR_MDR1   100
#define NR_MDR2   200
#define NR_MDR3   320
#define NR_MDR4   450
#define NR_MDR5   670
#define NR_MDRMAX 1000

extern "C" __global__ void fer_get_indexr(const float* __restrict__ RR,
                                          const float* __restrict__ thr,
                                          int* __restrict__ INDEXR,
                                          int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    const float rr = RR[k];
    const float RR_DRmin = thr[0];
    const float RR_DR1   = thr[1];
    const float RR_DR2   = thr[2];
    const float RR_DR3   = thr[3];
    const float RR_DR4   = thr[4];
    const float RR_DR5   = thr[5];
    const float RR_DRmax = thr[6];
    int idx;

    if (rr <= RR_DRmin) {
        /* Fixed mean diameter of rain (0.05 mm) for low rain rates. */
        idx = NR_MDRMIN;
    } else if (rr <= RR_DR1) {
        /* 0.05-0.10 mm: Dr(microns) = 1.123e3*RR**.1947 */
        idx = (int)(1.123E3f * powf(rr, 0.1947f) + 0.5f);
        idx = max(NR_MDRMIN, min(idx, NR_MDR1));
    } else if (rr <= RR_DR2) {
        /* 0.10-0.20 mm: Dr(microns) = 1.225e3*RR**.2017 */
        idx = (int)(1.225E3f * powf(rr, 0.2017f) + 0.5f);
        idx = max(NR_MDR1, min(idx, NR_MDR2));
    } else if (rr <= RR_DR3) {
        /* 0.20-0.32 mm: Dr(microns) = 1.3006e3*RR**.2083 */
        idx = (int)(1.3006E3f * powf(rr, 0.2083f) + 0.5f);
        idx = max(NR_MDR2, min(idx, NR_MDR3));
    } else if (rr <= RR_DR4) {
        /* 0.32-0.45 mm: Dr(microns) = 1.354e3*RR**.2143 */
        idx = (int)(1.354E3f * powf(rr, 0.2143f) + 0.5f);
        idx = max(NR_MDR3, min(idx, NR_MDR4));
    } else if (rr <= RR_DR5) {
        /* 0.45-0.675 mm: Dr(microns) = 1.404e3*RR**.2213 */
        idx = (int)(1.404E3f * powf(rr, 0.2213f) + 0.5f);
        idx = max(NR_MDR4, min(idx, NR_MDR5));
    } else if (rr <= RR_DRmax) {
        /* 0.675-1.0 mm: Dr(microns) = 1.4457e3*RR**.2303 */
        idx = (int)(1.4457E3f * powf(rr, 0.2303f) + 0.5f);
        idx = max(NR_MDR5, min(idx, NR_MDRMAX));
    } else {
        /* Fixed mean diameter of rain (1.0 mm) for high rain rates. */
        idx = NR_MDRMAX;
    }
    INDEXR[k] = idx;
}
