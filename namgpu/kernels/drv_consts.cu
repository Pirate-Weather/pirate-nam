// drv_consts.cu -- derived dynamics constants of NMMB CONSTS (scalars into ctx.tables['consts']).
//
// NOTICE: NAM source file nmm/module_CONTROL.F90 (NOAA/NCEP/EMC, US Government work, no licence file).
// The Fortran module carries no licence block; the module header is kept here as the notice:
//   module module_control  (NMMB control, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_CONTROL.F90:123-345 (nam.v4.2.11, 83df4e13c1).
//
// Fortran routine CONSTS(global, dt, smag2, codamp, wcor, pt, tph0d, tlm0d, sbd, wbd, dphd, dlmd,
//                         ..., lnsh, lnsad, adv_standard, adv_upstream, nboco, tboco,
//                         my_domain_id, mype, its, ite, jts, jte, ims, ime, jms, jme, ids, ide, jds, jde).
//
// Every statement of CONSTS (lines 123-345) and where it lands:
//   lnsbc = lnsh                                                 -> outi[0]
//   bofac = 0. (global) / 4. (regional)                          -> outf[3]
//   lnsad = 0 (global) / lnsh + 2 (regional)                     -> outi[1]
//   adv_upstream  = jts<jds+1+lnsad .or. jte>=jde-1-lnsad .or.
//                   its<ids+1+lnsad .or. ite>=ide-1-lnsad          -> outi[3]
//   adv_standard  = jte>=jds+1+lnsad .and. jts<=jde-1-lnsad .and.
//                   ite>=ids+1+lnsad .and. its<=ide-1-lnsad        -> outi[2]
//   tboco  = read from the boco.0000 record (module_CONTROL:283)  -> wrapper (host file read, argument here)
//   nboco  = nint(tboco/dt) when regional and my_domain_id==1 (290-293) -> outi[4]
//            The Fortran leaves nboco unset in every other case; this kernel writes 0 and the
//            wrapper reports that case.
//   ef4t   = 0.5*dt/cp  (cp = NMMB_CP, module_CONSTANTS)         -> outf[1]
//   pl     = max(pt, 1000.)                                       -> outf[2]  (input to TABLEP)
//   dt, smag2, codamp, wcor, pt pass through                      -> outf[0], outf[4..7]
//
// Calls made by CONSTS and where each is ported (NOT in this kernel, by design):
//   grid_consts (module_CONTROL:351-1139) -> unit init-grid-consts (namgpu/kernels/init_grid_consts.cu).
//       init_case_driver.CHAIN runs init-grid-consts before drv-consts, so ctx.tables['grid'] is complete
//       when CONSTS runs. It is not ported twice.
//   exptbl, tablep, tablet, psitbl -> units drv-exptbl, drv-tablep, drv-tablet, drv-psitbl. They follow
//       drv-consts in init_case_driver.CHAIN and are status todo (no kernel yet). drv-tablep must read
//       pl from ctx.tables['consts']['f'][CF_PL] (slot 2) when it is ported.
//
// lnsh: the model reads lnsh from parm/nam_configfile (lnsh: 5). parm/nam_namelist.nps has lnsh = 1
// inside &nemsinterp, the NPS interpolation namelist (preprocessing), not an input to CONSTS.
//
// Index ranges come in as arguments from the caller. Single process, one domain (CONTRACT section 1):
// ids=jds=1, ide=IM, jde=JM, its=ids, ite=ide, jts=jds, jte=jde. The MPI looplimits split is dropped
// (one card, one domain).
//
// This is scalar work: one thread, one block, one launch, no host sync. Output slots are the
// CF_* and CI_* indices in namgpu/steps/drv_consts.py.

#include "nmmb_common.cuh"

extern "C" __global__ void consts_scalars(
    const int global_,          // Fortran `global` logical: 1 = global branch
    const int lnsh,             // configfile lnsh (5)
    const int my_domain_id,     // Fortran my_domain_id (single domain: 1)
    const int ids, const int ide, const int jds, const int jde,
    const int its, const int ite, const int jts, const int jte,
    const float dt,             // DT (25 s)
    const float smag2,          // SMAG2 (0.4)
    const float codamp,         // CODAMP (9.0)
    const float wcor,           // WCOR (0.18)
    const float pt,             // PT (200 Pa)
    const float tboco,          // TBOCO from boco.0000 (s), set by the wrapper
    float* __restrict__ outf,   // [8] float32: dt, ef4t, pl, bofac, smag2, codamp, wcor, pt
    int* __restrict__ outi)     // [8] int32:   lnsbc, lnsad, adv_standard, adv_upstream, nboco
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // [CONSTS] lnsbc / bofac / lnsad, global and regional branches
    int lnsbc, lnsad;
    float bofac;
    if (global_) {
        lnsbc = lnsh;
        bofac = 0.0f;
        lnsad = 0;
    } else {
        lnsbc = lnsh;
        bofac = 4.0f;
        lnsad = lnsh + 2;
    }

    // [CONSTS] advection regions (adv_upstream, adv_standard)
    int adv_upstream = 0;
    if (jts < jds + 1 + lnsad || jte >= jde - 1 - lnsad ||
        its < ids + 1 + lnsad || ite >= ide - 1 - lnsad) {
        adv_upstream = 1;
    }
    int adv_standard = 0;
    if (jte >= jds + 1 + lnsad && jts <= jde - 1 - lnsad &&
        ite >= ids + 1 + lnsad && its <= ide - 1 - lnsad) {
        adv_standard = 1;
    }

    // [CONSTS] nboco = nint(tboco/dt): regional and domain 1 only (Fortran 290-293).
    // Fortran NINT rounds half away from zero: roundf.
    int nboco = 0;
    if (!global_ && my_domain_id == 1) {
        nboco = (int)roundf(tboco / dt);
    }

    // [CONSTS] ef4t = 0.5*dt/cp   (cp = NMMB_CP, module_CONSTANTS)
    const float ef4t = 0.5f * dt / NMMB_CP;

    // [CONSTS] pl = max(pt, 1000.)
    const float pl = fmaxf(pt, 1000.0f);

    outf[0] = dt;
    outf[1] = ef4t;
    outf[2] = pl;
    outf[3] = bofac;
    outf[4] = smag2;
    outf[5] = codamp;
    outf[6] = wcor;
    outf[7] = pt;

    outi[0] = lnsbc;
    outi[1] = lnsad;
    outi[2] = adv_standard;
    outi[3] = adv_upstream;
    outi[4] = nboco;
}
