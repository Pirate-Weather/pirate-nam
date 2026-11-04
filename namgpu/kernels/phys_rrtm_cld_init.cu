// phys_rrtm_cld_init.cu -- unit phys-rrtm-cld-init: cld_init, cloud scheme flags, WOOF-NAM.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone), module
//   MODULE_RADIATION_CLOUDS_NMMB (phys/radiation_clouds_nmmb.f, nam.v4.2.11). Author header kept in
//   the NAM clone; VTAGCLD 'NCEP-Radiation_clouds v5.1 Nov 2012'.
// Ported to CUDA for WOOF-NAM from phys/radiation_clouds_nmmb.f:225-349 (nam.v4.2.11, 83df4e13c1).
//
// Fortran SUBROUTINE cld_init(si, NLAY, me), every statement accounted for:
//  - 290-296  if (icldflg == 0): call rhtable(me, ier) and stop on ier < 0. The rhtable routine is
//             its own unit (phys-rrtm-rhtable, radiation_clouds_nmmb.f:2499-2769, mode skip). The
//             wrapper refuses icldflg == 0 because the RH table would be unset; the default NAM
//             configuration uses icldflg = 1 (physparam.f:141), so this branch is not reached.
//  - 297-321  icldflg /= 0: print of the cloud method (me == 0 diagnostic). The icmphys validity
//             check (icmphys not in {3,4,5,8} -> stop) is done on the host in the wrapper, once, at
//             init. No print inside the kernel.
//  - 327-344  llyr: the top of the boundary-layer cloud, from the sigma interfaces si.
//             The NAM run sets ivflip = iflip = 0 (module_SOLVER_GRID_COMP.F90:10716; rad_initialize_nmmb.f:183),
//             so the toa->sfc branch is the one that runs. The wrapper passes ivflip from the rad_initialize args.
//             ivflip == 0 (toa -> sfc):
//                 do k = NLAY, 2, -1 { kl = k; if (si(k) < 0.9) exit }   ; llyr = kl
//             ivflip /= 0 (sfc -> toa; physparam.f:164 default 1, not the NAM run value):
//                 do k = 2, NLAY    { kl = k; if (si(k) < 0.9) exit }   ; llyr = kl - 1
//             When the Fortran loop runs to completion without exit, kl keeps the last k assigned
//             (2 for the downward loop, NLAY for the upward loop); the loops below reproduce that.
//             The Fortran si(k) is 1-based; here si[k-1].
//  - 280      iovr = max(iovrsw, iovrlw) (module variable used for the diagnostic 3-domain overlap
//             output; physparam.f:143-144, both 1 in the default configuration).
//  - 0.9e0 is the single-precision literal (kind_phys = single, NAM kfpt single).
//
// Outputs (device int32[2]): out[0] = llyr (1-based level index, as the Fortran), out[1] = iovr.
// Single thread, one launch, no host sync, no allocation, no print. This runs once at forecast
// start (rad_initialize_nmmb, off the step path). The Fortran's module variable `llyr` also seeds the
// cloud-optical consumers in other units through ctx.tables["rrtm_cld"].
//
// Launch: <<<1, 1>>>.

extern "C" __global__ void phys_rrtm_cld_init(const float* __restrict__ si,
                                              const int nlay,
                                              const int ivflip,
                                              const int iovrsw,
                                              const int iovrlw,
                                              int* __restrict__ out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int kl = 1;     // Fortran kl is undefined when NLAY < 2 (loop does not run); not reachable at NLAY = 60
    int llyr;

    if (ivflip == 0) {
        // data from toa to sfc: lab_do_k0 : do k = NLAY, 2, -1
        for (int k = nlay; k >= 2; --k) {
            kl = k;
            if (si[k - 1] < 0.9f) break;
        }
        llyr = kl;
    } else {
        // data from sfc to toa: lab_do_k1 : do k = 2, NLAY
        for (int k = 2; k <= nlay; ++k) {
            kl = k;
            if (si[k - 1] < 0.9f) break;
        }
        llyr = kl - 1;
    }

    out[0] = llyr;
    out[1] = (iovrsw > iovrlw) ? iovrsw : iovrlw;
}
