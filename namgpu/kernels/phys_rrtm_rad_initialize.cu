// phys_rrtm_rad_initialize.cu -- physparam flag setup of NMMB rad_initialize (scalars into ctx.tables).
//
// NOTICE: NAM source file phys/rad_initialize_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence file).
// The Fortran subroutine header is kept here as the notice:
//   subprogram: rad_initialize - a subprogram to initialize radiation
//   mar 2012 - yu-tai hou  create the program to initialize fixed control variables for radiaion processes.
// Ported to CUDA for WOOF-NAM from phys/rad_initialize_nmmb.f:2-224 (nam.v4.2.11, 83df4e13c1).
//
// Fortran subroutine rad_initialize_nmmb(si, levr, ictm, isol, ico2, iaer, iaer_mdl, ialb, iems, ntcw,
//     num_p3d, ntoz, iovr_sw, iovr_lw, isubc_sw, isubc_lw, icliq_sw, icice_sw, icliq_lw, icice_lw,
//     sashal, crick_proof, ccnorm, norad_precip, iflip, me).
//
// Every assignment of the routine (lines 135-183) and its slot in `out` (indices RI_* in
// namgpu/physics/rrtm/phys_rrtm_rad_initialize.py):
//   isolar  = isol                                        -> RI_ISOLAR
//   ictmflg = ictm                                        -> RI_ICTMFLG
//   ico2flg = ico2                                        -> RI_ICO2FLG
//   ioznflg = ntoz                                        -> RI_IOZNFLG
//   iaerflg = mod(iaer,100) if ictm==0 or ictm==-2,
//             else iaer                                   -> RI_IAERFLG
//   laswflg = (mod(iaerflg,10) > 0)                       -> RI_LASWFLG
//   lalwflg = (mod(iaerflg/10,10) > 0)                    -> RI_LALWFLG
//   lavoflg = (iaerflg >= 100)                            -> RI_LAVOFLG
//   iaermdl = iaer_mdl                                    -> RI_IAERMDL
//   icldflg = 1 if ntcw > 0 else 0                        -> RI_ICLDFLG
//   icmphys = num_p3d (the zhao/ferrier if-block at 157-161 is commented out in the Fortran)
//                                                         -> RI_ICMPHYS
//   iovrsw  = iovr_sw                                     -> RI_IOVRSW
//   iovrlw  = iovr_lw                                     -> RI_IOVRLW
//   lsashal = sashal                                      -> RI_LSASHAL
//   lcrick  = crick_proof                                 -> RI_LCRICK
//   lcnorm  = ccnorm                                      -> RI_LCNORM
//   lnoprec = norad_precip                                -> RI_LNOPREC
//   isubcsw = isubc_sw                                    -> RI_ISUBCSW
//   isubclw = isubc_lw                                    -> RI_ISUBCLW
//   iswcliq = icliq_sw                                    -> RI_ISWCLIQ
//   iswcice = icice_sw                                    -> RI_ISWCICE
//   ilwcliq = icliq_lw                                    -> RI_ILWCLIQ
//   ilwcice = icice_lw                                    -> RI_ILWCICE
//   ialbflg = ialb                                        -> RI_IALBFLG
//   iemsflg = iems                                        -> RI_IEMSFLG
//   ivflip  = iflip                                       -> RI_IVFLIP
//
// Not ported here, with reasons:
//   - `si` and `levr`: consumed only by radinit_nmmb, which is unit phys-rrtm-radinit (its own wrapper).
//   - `me` print block (191-208): host diagnostic prints for MPI rank 0, no state written.
//   - ipsd0 seed block (185-189): commented out in the Fortran.
//   - The routine has no per-cell loops: it is a single-thread scalar routine by construction.
//
// Integer semantics match Fortran: MOD truncates toward zero exactly as C `%`; integer division
// truncates as C `/`. Logicals are written as 0/1.
// One thread, one block, one launch, no host sync.

extern "C" __global__ void rad_initialize_flags(
    const int ictm,        // ictm: data time/date control flag
    const int isol,        // isol: solar constant flag
    const int ico2,        // ico2: co2 source flag
    const int iaer,        // iaer: 3-digit aerosol flag
    const int iaer_mdl,    // iaer_mdl: topospheric aerosol model
    const int ialb,        // ialb: surface albedo flag
    const int iems,        // iems: surface emissivity flag
    const int ntcw,        // ntcw: cloud condensate index (>0 prognostic)
    const int num_p3d,     // num_p3d: microphysics cloud scheme
    const int ntoz,        // ntoz: ozone data flag
    const int iovr_sw,     // iovr_sw: sw cloud overlap
    const int iovr_lw,     // iovr_lw: lw cloud overlap
    const int isubc_sw,    // isubc_sw: sw sub-column approximation
    const int isubc_lw,    // isubc_lw: lw sub-column approximation
    const int icliq_sw,    // icliq_sw: sw liquid optical property
    const int icice_sw,    // icice_sw: sw ice optical property
    const int icliq_lw,    // icliq_lw: lw liquid optical property
    const int icice_lw,    // icice_lw: lw ice optical property
    const int sashal,      // sashal (logical as 0/1): shallow convection
    const int crick_proof, // crick_proof (logical as 0/1)
    const int ccnorm,      // ccnorm (logical as 0/1): in-cloud condensate
    const int norad_precip,// norad_precip (logical as 0/1)
    const int iflip,       // iflip: vertical index direction
    int* __restrict__ out) // out[26], indices RI_* (see the wrapper)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int isolar = isol;
    const int ictmflg = ictm;
    const int ico2flg = ico2;
    const int ioznflg = ntoz;

    int iaerflg;
    if (ictm == 0 || ictm == -2) {
        iaerflg = iaer % 100;           // no volcanic aerosols for clim hindcast
    } else {
        iaerflg = iaer;
    }
    const int laswflg = ((iaerflg % 10) > 0) ? 1 : 0;
    const int lalwflg = (((iaerflg / 10) % 10) > 0) ? 1 : 0;
    const int lavoflg = (iaerflg >= 100) ? 1 : 0;
    const int iaermdl = iaer_mdl;

    const int icldflg = (ntcw > 0) ? 1 : 0;
    const int icmphys = num_p3d;

    out[0]  = isolar;
    out[1]  = ictmflg;
    out[2]  = ico2flg;
    out[3]  = ioznflg;
    out[4]  = iaerflg;
    out[5]  = laswflg;
    out[6]  = lalwflg;
    out[7]  = lavoflg;
    out[8]  = iaermdl;
    out[9]  = icldflg;
    out[10] = icmphys;
    out[11] = iovr_sw;
    out[12] = iovr_lw;
    out[13] = (sashal != 0) ? 1 : 0;
    out[14] = (crick_proof != 0) ? 1 : 0;
    out[15] = (ccnorm != 0) ? 1 : 0;
    out[16] = (norad_precip != 0) ? 1 : 0;
    out[17] = isubc_sw;
    out[18] = isubc_lw;
    out[19] = icliq_sw;
    out[20] = icice_sw;
    out[21] = icliq_lw;
    out[22] = icice_lw;
    out[23] = ialb;
    out[24] = iems;
    out[25] = iflip;
}
