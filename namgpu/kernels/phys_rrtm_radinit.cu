// phys_rrtm_radinit.cu -- radinit_nmmb control flags for the NMMB RRTM radiation package.
//
// NOTICE: NAM source file phys/grrad_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence file).
// Routine header kept as the notice:
//   subprogram: radinit  initialization of radiation calculations
//   program history: 08-14-2003 yu-tai hou created
// Ported to CUDA for WOOF-NAM from phys/grrad_nmmb.f:226-451 (nam.v4.2.11, 83df4e13c1).
//
// Line-by-line counterpart of the routine's own computations. Inputs are the physparam state that
// rad_initialize_nmmb set (rip = ctx.tables['rad_physparam'], slots RI_* of phys_rrtm_rad_initialize.py),
// so IEMSFLG, IOZNFLG, ICLDFLG, ICMPHYS come from the configured NAM values (ntoz, ntcw, iems, np3d),
// not from constants:
//   340   itsfc  = iemsflg / 10             -> out[0]   (integer division, truncation toward zero, as Fortran)
//   341   loz1st = (ioznflg == 0)           -> out[1]   (1 = .true.)
//   424-428  if (icmphys==5): icldflg_org = icldflg; icldflg = ricwp (ricwp = 0)
//            -> out[2] = icldflg as seen by the rlwinit and rswinit calls (cld_init runs before this block
//               and sees out[3])
//   444-446  if (icmphys==5): icldflg = icldflg_org
//            -> out[3] = icldflg after the routine returns
//   out[4] = iaerflg, out[5] = iaermdl   (gate for clim_aerinit, radiation_aerosols_nmmb.f:539-551)
//   out[6] = ioznflg                     (gas_init climatology branch, radiation_gases_nmmb.f:557 path)
//
// Single thread, one launch, no allocation. Output: int32 out[7].

extern "C" __global__ void radinit_ctl(const int* __restrict__ rip, int* __restrict__ out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int RI_IOZNFLG = 3, RI_IAERFLG = 4, RI_IAERMDL = 8, RI_ICLDFLG = 9, RI_ICMPHYS = 10,
              RI_IEMSFLG = 24;
    const int ricwp = 0;
    const int ioznflg = rip[RI_IOZNFLG];
    const int icldflg = rip[RI_ICLDFLG];
    const int icmphys = rip[RI_ICMPHYS];

    out[0] = rip[RI_IEMSFLG] / 10;
    out[1] = (ioznflg == 0) ? 1 : 0;
    out[2] = (icmphys == 5) ? ricwp : icldflg;
    out[3] = icldflg;
    out[4] = rip[RI_IAERFLG];
    out[5] = rip[RI_IAERMDL];
    out[6] = ioznflg;
}
