// phys_rrtm_coszmn.cu -- COSZMN_NMMB: mean cosine of solar zenith over the SW call interval (RRTM group).
//
// NOAA notice: header block of phys/radiation_astronomy_nmmb.f:744-800 (nam.v4.2.11, 83df4e13c1), kept verbatim:
//...................................
//  ---  inputs:
//  ---  outputs:
//  ===================================================================  !
//                                                                       !
//  coszmn_nmmb computes mean cos solar zenith angle over 'dtswav' hours.!
//                                                                       !
//  inputs:                                                              !
//    xlon          - longitude in radians                               !
//    sinlat,coslat - sin and cos of latitude                            !
//    dtswav        - sw radiation calling interval in hour              !
//    solhr         - time after 00z in hours                            !
//    IM            - dimension for longitude direction                  !
//    me            - print message control flag                         !
//    NLON2,LATD    - dimensions for longitude/latitude directions       !
//                                                                       !
//  outputs:                                                             !
//    coszen        - average of cosz for daytime only in sw call interval
//    coszdg        - average of cosz over entire sw call interval       !
//                                                                       !
//  module variables:                                                    !
//    sollag        - equation of time                                   !
//    sindec        - sine of the solar declination angle                !
//    cosdec        - cosine of the solar declination angle              !
//    anginc        - solar angle increment per iteration for cosz calc  !
//    nstp          - total number of zenith angle iterations            !
//                                                                       !
//  usage:    call comzmn                                                !
//                                                                       !
//  external subroutines called: none                                    !
//                                                                       !
//  program history log:                                                 !
//     05-28-2004   yu-tai hou     - modified for gfs hybrid model       !
//                                                                       !
//  ===================================================================  !
//
//  ---  inputs:
//  ---  outputs:
//  ---  locals:
// Ported to CUDA for WOOF-NAM from phys/radiation_astronomy_nmmb.f:744-848 (nam.v4.2.11, 83df4e13c1).
//
// Fortran -> CUDA map (every statement of coszmn_nmmb, lines 744-848):
//   solang = pid12*(solhr - f12)                           -> solang (float, once per cell)
//   rnstp  = dtswav/float(nrads)                           -> rnstp
//   coszen(i)=0, coszdg(i)=0, istsun(i)=0                  -> locals
//   xlon1(i) = xlon(i)  (the 0..2pi branch is commented out in the Fortran)  -> xl
//   do it=1,nrads: cns = solang + float(it-1)*pid12*rnstp + sollag            -> cns
//       ss = sinlat*sindec; cc = coslat*cosdec                                 -> ss, cc
//       coszn = ss + cc*cos(cns + xlon1(i))                                    -> coszn
//       coszen(i) += max(0, coszn)                                             -> cz_sum
//       if (coszn > czlimt) istsun(i) += 1                                     -> istsun
//   coszdg(i) = coszen(i)*rnstp        -> coszdg output (whole SW interval; grrad COSZDG_V)
//   if istsun(i) > 0: coszen(i) = coszen(i)/istsun(i)  -> coszen output (daytime; grrad COSZEN_V)
//
// Precision: the NAM build is single precision. phys/machine.f sets kind_phys = 4 under SINGLE_PREC, and
// the Wcoss configure defines -DSINGLE_PREC, so every local of coszmn_nmmb is float here, as in the Fortran.
// czlimt is the 0.0001 kind_phys literal. sinlat/coslat are sinf/cosf of GLAT (radians); xlon is GLON.
// The module variables sollag, sindec, cosdec are read on the device from SOL_STATE (written by
// phys-rrtm-sol-update), so there is no host copy. One thread per (i, j), one launch, no host sync, no printf.

#include "nmmb_common.cuh"

#define COSZMN_F12    12.0f
#define COSZMN_CZLIMT 1.0e-4f     // czlimt, kind_phys literal

extern "C" __global__ void phys_rrtm_coszmn(
    const float* __restrict__ glon,     // GLON [JM, IM] radians (xlon)
    const float* __restrict__ glat,     // GLAT [JM, IM] radians (sinlat = sin, coslat = cos)
    float* __restrict__ coszen,         // out [JM, IM] daytime-average cos zenith (grrad COSZEN_V)
    float* __restrict__ coszdg,         // out [JM, IM] cos zenith averaged over the whole SW interval (grrad COSZDG_V)
    const float solhr,                  // time after 00z, hours (solhr = mod(fhr, 24), fhr = ihr + imin/60)
    const float dtswav,                 // SW calling interval, hours (FHSWR = NRADS*DT/3600)
    const int nrads,                    // zenith iterations per SW interval (NRADS)
    const float* __restrict__ sol,      // SOL_STATE float32[4] = sollag, sindec, cosdec, anginc
    const int im,
    const int jm)
{
    COLUMN_IJ(im, jm)                   // one thread per (i, j); returns out of range

    const size_t idx = IDX2(i0, j0, im);

    const float sollag = sol[0];
    const float sindec = sol[1];
    const float cosdec = sol[2];
    const float pid12 = (2.0f * asinf(1.0f)) / COSZMN_F12;    // pid12 = (2*asin(1))/f12 in single
    const float solang = pid12 * (solhr - COSZMN_F12);        // solar angle at present time
    const float rnstp = dtswav / (float)nrads;

    const float xl = glon[idx];                               // xlon1(i) = xlon(i)
    const float sinlat = sinf(glat[idx]);
    const float coslat = cosf(glat[idx]);

    float cz_sum = 0.0f;
    int istsun = 0;

    for (int it = 1; it <= nrads; ++it) {
        const float cns = solang + (float)(it - 1) * pid12 * rnstp + sollag;
        const float ss = sinlat * sindec;
        const float cc = coslat * cosdec;
        const float coszn = ss + cc * cosf(cns + xl);
        cz_sum += fmaxf(0.0f, coszn);
        if (coszn > COSZMN_CZLIMT) istsun += 1;
    }

    const float coszdg_v = cz_sum * rnstp;
    float coszen_v = cz_sum;
    if (istsun > 0) coszen_v = cz_sum / (float)istsun;

    coszen[idx] = coszen_v;
    coszdg[idx] = coszdg_v;
}
