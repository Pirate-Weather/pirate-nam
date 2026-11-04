// NOAA notice: header block of phys/radiation_aerosols_nmmb.f (nam.v4.2.11, 83df4e13c1), kept verbatim.
// !!!!!  ==========================================================  !!!!!
// !!!!!            'module_radiation_aerosols' description           !!!!!
// !!!!!  ==========================================================  !!!!!
// !                                                                      !
// !   this module contains climatological atmospheric aerosol schemes for!
// !   radiation computations.                                            !
// !                                                                      !
// !   in the module, the externally callable subroutines are :           !
// !                                                                      !
// !      'aer_init'   -- initialization                                  !
// !         inputs:                                                      !
// !           ( NLAY, me )                                               !
// !         outputs:                                                     !
// !           ( none )                                                   !
// !                                                                      !
// !      'aer_update' -- updating aerosol data                           !
// !         inputs:                                                      !
// !           ( iyear, imon, me )                                        !
// !         outputs:                                                     !
// !           ( none )                                                   !
// !                                                                      !
// !      'setaer'     -- mapping aeros profile, compute aeros opticals   !
// !         inputs:                                                      !
// !           (prsi,prsl,prslk,tvly,rhlay,slmsk,tracer,xlon,xlat,        !
// !            IMAX,NLAY,NLP1, lsswr,lslwr,                              !
// !         outputs:                                                     !
// !           (aerosw,aerolw)                                            !
// !!          (aerosw,aerolw,aerodp)                                     !
// !                                                                      !
// !                                                                      !
// !   external modules referenced:                                       !
// !       'module physpara'                in 'physpara.f'               !
// !       'module physcons'                in 'physcons.f'               !
// !       'module module_radsw_parameters' in 'radsw_xxxx#_param.f'      !
// !       'module module_radlw_parameters' in 'radlw_xxxx#_param.f'      !
// !       'module module_radlw_cntr_para'  in 'radsw_xxxx#_param.f'      !
// !                                                                      !
// !   output variable definitions:                                       !
// !       aerosw(IMAX,NLAY,NBDSW,1) - aerosols optical depth for sw      !
// !       aerosw(IMAX,NLAY,NBDSW,2) - aerosols single scat albedo for sw !
// !       aerosw(IMAX,NLAY,NBDSW,3) - aerosols asymmetry parameter for sw!
// !                                                                      !
// !       aerolw(IMAX,NLAY,NBDLW,1) - aerosols optical depth for lw      !
// !       aerolw(IMAX,NLAY,NBDLW,2) - aerosols single scattering albedo  !
// !       aerolw(IMAX,NLAY,NBDLW,3) - aerosols asymetry parameter        !
// !                                                                      !
// !                                                                      !
// !   program history:                                                   !
// !     apr     2003  ---  y.-t. hou     created                         !
// !     nov 04, 2003  ---  y.-t. hou     modified version                !
// !     apr 15, 2005  ---  y.-t. hou     modified module structure       !
// !     jul     2006  ---  y.-t. hou     add volcanic forcing            !
// !     feb     2007  ---  y.-t. hou     add generalized spectral band   !
// !                   interpolation for sw aerosol optical properties    !
// !     mar     2007  ---  y.-t. hou     add generalized spectral band   !
// !                   interpolation for lw aerosol optical properties    !
// !     aug     2007  ---  y.-t. hou     change clim-aer vert domain     !
// !                   from pressure reference to sigma reference         !
// !     sep     2007  ---  y.-t. hou     moving temporary allocatable    !
// !                   module variable arrays to subroutine dynamically   !
// !                   allocated arrays (eliminate deallocate calls)      !
// !     jan     2010  ---  sarah lu      add gocart option               !
// !     may     2010  ---  sarah lu      add geos4-gocart climo          !
// !     jul     2010  --   s. moorthi - merged NEMS version with new GFS !
// !                        version                                       !
// !     oct 23, 2010  ---  Hsin-mu lin   modified subr setclimaer to     !
// !        interpolate the 5 degree aerosol data to small domain based on!
// !        the nearby 4 points instead of previous nearby assignment by  !
// !        using the 5 degree data. this process will eliminate the dsw  !
// !        jagged edges in the east conus where aerosol effect are lagre.!
// !     dec     2010  ---  y.-t. hou     modified and optimized bi-linear!
// !        horizontal interpolation in subr setclimaer. added safe guard !
// !        measures in lat/lon indexing and added sea/land mask variable !
// !        slmsk as input field to help aerosol profile selection.       !
// !     jan     2011  ---  y.-t. hou     divided the program into two    !
// !        separated interchangeable modules: a climatology aerosol      !
// !        module, and a gocart aerosol scheme module. the stratospheric !
// !        volcanic aerosol part is still within the two driver modules, !
// !        and may also become a separate one in the further development.!
// !        unified in/out argument list for both clim and gocart types of!
// !        schemes and added vertically integrated aer-opt-dep, aerodp,  !
// !        to replace tau_gocart as optional output for various species. !
// !     aug     2012  ---  y.-t. hou     changed the initialization subr !
// !        'aerinit' into two parts: 'aer_init' is called at the start   !
// !        of run to set up module parameters; and 'aer_update' is       !
// !        called within the time loop to check and update data sets.    !
// !     nov     2012  ---  y.-t. hou     modified control parameters thru!
// !        module 'physpara'.                                            !
// !     jan     2013  ---  sarah lu and y.-t. hou   reintegrate both     !
// !        opac-clim and gocart schemes into one module to make the      !
// !        program best utilize common components. added aerosol model   !
// !        scheme selection control variable iaer_mdl to the namelist.   !
// !                                                                      !
// !   references for opac climatological aerosols:                       !
// !     hou et al. 2002  (ncep office note 441)                          !
// !     hess et al. 1998 - bams v79 831-844                              !
// !                                                                      !
// !   references for gocart interactive aerosols:                        !
// !     chin et al., 2000 - jgr, v105, 24671-24687                       !
// !                                                                      !
// !   references for stratosperic volcanical aerosols:                   !
// !     sato et al. 1993 - jgr, v98, d12, 22987-22994                    !
// !                                                                      !
// !                                                                      !
// !!!!!  ==========================================================  !!!!!
// !!!!!                       end descriptions                       !!!!!
// !!!!!  ==========================================================  !!!!!
// 
// 
// 
// !=============================================!

// phys_rrtm_aer_update.cu -- AER_UPDATE / TROP_UPDATE fill (module_radiation_aerosols_nmmb).
// Ported to CUDA for nam-gpu from phys/radiation_aerosols_nmmb.f:1504-1839 (nam.v4.2.11, 83df4e13c1).
//
// Split of the Fortran routine:
//   - The 12 monthly records are text on disk. Host file I/O cannot run on the device, so the wrapper
//     (phys_rrtm_aer_update.py) reads the selected month's 72 x 37 records on the host, once per monthly
//     update (not per step) and uploads them as raw arrays. That is the only host work.
//   - trop_update's per-cell fill loop (idxcg, cmixg, denng, kprfg) is the device work: this kernel.
//   - volc_update is host file I/O plus a 10 x 4 x 12 integer table; the wrapper uploads that table.
//     It has no grid loop, so it has no kernel.
//   - The me == 0 print statements are host diagnostics; they write no state and are not ported.
//
// Fortran trop_update, every assignment and its counterpart here (per cell, one thread):
//   idxcg = 0, cmixg = 0, denng = 0 (zeroing)      -> every entry is overwritten below, so no zero survives
//   idxcg(k,i,j) = idxc(k), k = 1..NXC              -> out_idxcg[cell*NXC+k] = rec_idx (all k)
//   cmixg(k,i,j) = cmix(k), k = 1..NXC-1            -> out_cmixg[cell*NXC+k] = rec_cmix (k < NXC-1)
//   tem = 1; tem = tem - cmix(k), k = 1..NXC-1      -> double sum in order
//   cmixg(NXC,i,j) = tem (closes the sum to one)    -> out_cmixg[cell*NXC+NXC-1] = tem
//   denng(1,i,j) = denn                             -> out_denng[cell*2+0]
//   denng(2,i,j) = cmix(NXC) if kprf >= 6, else 0   -> out_denng[cell*2+1]
//   kprfg(i,j) = kprf                               -> out_kprfg[cell]
// Precision: kind_phys = selected_real_kind(13,60) (physparam.f:34), double. Every real here is double.
// Layout: CONTRACT section 1 reverses the Fortran index order, so Fortran idxcg(NXC,IMXAE,JMXAE) is
// int[JMXAE][IMXAE][NXC] (cell*NXC + m), denng(2,IMXAE,JMXAE) is double[JMXAE][IMXAE][2], kprfg is
// int[JMXAE][IMXAE]. cell = j0*IMXAE + i0 (zero-based), matching the host record order (j outer, i inner).

#define IMXAE 72   // physparam: num of lon-points in glb aeros data set (radiation_aerosols_nmmb.f:240)
#define JMXAE 37   // physparam: num of lat-points in glb aeros data set (radiation_aerosols_nmmb.f:241)
#define NXC    5   // physparam: num of max components in a profile (radiation_aerosols_nmmb.f:237)

extern "C" __global__ void aer_trop_fill(
    const int*    __restrict__ rec_idx,   // [JMXAE][IMXAE][NXC] raw component index of the month
    const double* __restrict__ rec_cmix,  // [JMXAE][IMXAE][NXC] raw component mixing ratio of the month
    const int*    __restrict__ rec_kprf,  // [JMXAE][IMXAE] profile index
    const double* __restrict__ rec_denn,  // [JMXAE][IMXAE] number density of the first layer
    int*    __restrict__ out_idxcg,       // [JMXAE][IMXAE][NXC]
    double* __restrict__ out_cmixg,       // [JMXAE][IMXAE][NXC]
    double* __restrict__ out_denng,       // [JMXAE][IMXAE][2]
    int*    __restrict__ out_kprfg)       // [JMXAE][IMXAE]
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= IMXAE * JMXAE) return;

    const size_t base = (size_t)cell * NXC;
    const int kprf = rec_kprf[cell];
    out_kprfg[cell] = kprf;
    out_denng[2 * cell + 0] = rec_denn[cell];

    double tem = 1.0;                                  // f_one
    for (int k = 0; k < NXC - 1; ++k) {
        out_idxcg[base + k] = rec_idx[base + k];
        out_cmixg[base + k] = rec_cmix[base + k];
        tem = tem - rec_cmix[base + k];
    }
    out_idxcg[base + NXC - 1] = rec_idx[base + NXC - 1];
    out_cmixg[base + NXC - 1] = tem;                   // make sure all components add to 1
    out_denng[2 * cell + 1] = (kprf >= 6) ? rec_cmix[base + NXC - 1] : 0.0;  // f_zero otherwise
}
