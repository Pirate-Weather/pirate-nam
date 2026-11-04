// phys_fer_growth_rates.cu -- WOOF-NAM unit phys-fer-growth-rates: MY_GROWTH_RATES_NMM_hr.
//
// NOTICE: NOAA/NCEP NAM source module_MP_FER_HIRES.F90 (US government work, no licence file in the
// NAM clone). The module header block is kept here verbatim as the notice (comment-prefixed):
// !
// !-- "Modified" fer_hires microphysics - 11 July 2016 version
// !
// ! (1) Ice nucleation: Fletcher (1962) replaces Meyers et al. (1992)
// ! (2) Cloud ice is a simple function of the number concentration from (1), and it
// !     is no longer a fractional function of the large ice.  Thus, the FLARGE &
// !     FSMALL parameters are no longer used.
// ! (3) T_ICE_init=-12 deg C provides a slight delay in the initial onset of ice.
// ! (4) NLImax is a function of rime factor (RF) and temperature.  
// !    a) For RF>10, NLImax=1.e3.  Mean ice diameters can exceed the 1 mm maximum
// !       size in the tables so that NLICE=NLImax=1.e3.
// !    b) Otherwise, NLImax is 10 L-1 at 0C and decreasing to 5 L-1 at <=-40C.  
// !       NLICE>NLImax at the maximum ice diameter of 1 mm.
// ! (5) Can turn off ice processes by setting T_ICE & T_ICE_init to be < -100 deg C
// ! (6) Modified the homogeneous freezing of cloud water when T<T_ICE.
// ! (7) Reduce the fall speeds of rimed ice by making VEL_INC a function of 
// !     VrimeF**1.5 and not VrimeF**2.
// ! (8) RHgrd=0.98 (98% RH) for the onset of condensation, to match what's been
// !     tested for many months in the NAMX.  Made obsolete with change in (13).
// ! (9) Rime factor is *never* adjusted when NLICE>NLImax.  
// ! (10) Ice deposition does not change the rime factor (RF) when RF>=10 & T>T_ICE.
// ! (11) Limit GAMMAS to <=1.5 (air resistance impact on ice fall speeds)
// ! (12) NSImax is maximum # conc of ice crsytals.  At cold temperature NSImax is
// !      calculated based on assuming 10% of total ice content is due to cloud ice.
// !
// !-- Further modifications starting on 23 July 2015
// ! (13) RHgrd is passed in as an input argument so that it can vary for different
// !      domains (RHgrd=0.98 for 12-km parent, 1.0 for 3-km nests)
// ! (14) Use the old "PRAUT" cloud water autoconversion *threshold* (QAUT0)
// 
// !-- Further modifications starting on 28 July 2015
// ! (15) Added calculations for radar reflectivity and number concentrations of
// !      rain (Nrain) and precipitating ice (Nsnow).
// ! (16) Removed double counting of air resistance term for riming onto ice (PIACW)
// ! (17) The maximum rime factor (RFmx) is now a function of MASSI(INDEXS), accounting
// !      for the increase in unrimed ice particle densities as values of INDEXS
// !      decrease from the maximum upper limit of 1000 microns to the lower limit of
// !      50 microns, coinciding with the assumed size of cloud ice; see lines 1128-1134.
// ! (18) A new closure is used for updating the rime factor, which is described in
// !      detail near lines 1643-1682.  The revised code is near lines 1683-1718.
// ! (19) Restructured the two-pass algorithm to be more robust, removed the HAIL
// !      & LARGE_RF logical variables so that NLICE>NLImax can occur.
// ! (20) Increased nsimax (see !aug27 below)
// ! (21) Modified the rain sedimentation (see two !aug27 blocks below)
// ! (22) NInuclei is the lower of Fletcher (1962), Cooper (1986), or NSImax. 
// ! (23) NLImax is no longer used or enforced. Instead, INDEXS=MDImax when RF>20,
// !      else INDEXS is a function of temperature. Look for !sep10 comment.
// ! (24) An override was inserted for (18), such that the rime density is not diluted
// !      diluted when RF>20. Look for !sep10 comment.
// ! (25) Radar reflectivity calculations were changes to reduce radar bright bands,
// !      limit enhanced, mixed-phase reflectivity to RF>=20. Look for !sep10 comments.
// ! (26) NLICE is not to exceed NSI_max (250 L^-1) when RF<20.  Look for !sep16 comments.
// ! Commented out! (28) Increase hail fall speeds using Thompson et al. (2008).  Look for !sep22 comments.
// ! (29) Modify NLImax, INDEXS for RF>=20. Look for !sep22 comments.
// ! (30) Check on NSmICE, Vci based on whether FLIMASS<1.  Look for !sep22a comments.
// ! Revised in (34)! (31) Introduced RFlag logical, which if =T enforces a lower limit of drop sizes not 
// !      to go below INDEXRmin and N0r is adjusted.  Look for !nov25 comments (corrections,
// !      refinements to sep25 & nov18 versions, includes an additional fix in nov25-fix).
// !      Also set INDEXRmin=500 rather than 250 microns.
// !-----------------------------------------------------------------------------
// !--- The following changes now refer to dates when those were made in 2016.
// !-----------------------------------------------------------------------------
// ! (32) Convective (RF>=20, Ng~10 L^-1, RHOg~500 kg m^-3), transition (RF=10, Ng~25 L^-1,
// !      RHOg~300 kg m^-3), & stratiform (RF<2) profiles are blended based on RF. !mar08
// ! (33) Fixed bug in Biggs' freezing, put back in collisional drop freezing.  !mar03
// ! (34) Changes in (31) are revised so that INDEXRmin at and below 0C level is 
// !      based on a rain rate equal to the snowfall rate above the 0C level.  !mar03
// ! (35) Increase radar reflectivity when RF>10 and RQSnew > 2.5 g m^-3. !mar12
// ! (36) !mar10 combines all elements of (32)-(35) together.
// ! (37) Bug fixes for the changes in (34) and the RFLAG variable  !apr18
// ! (38) Revised Schumann-Ludlam limit.  !apr18
// ! (39) Simplified PCOND (cloud cond/evap) calculation   !apr21
// ! (40) Slight change in calculating RF.   !apr22
// ! (41) Reduce RF values for calculating mean sizes of snow, graupel, sleet/hail  !apr22a
// ! (42) Increase reflectivity from large, wet, high rime factor ice (graupel) by
// !      assuming |Kw|**2/|Ki|**2 = 0.224 (Smith, 1984, JCAM).
// ! (43) Major restructuring of code to allow N0r to vary from N0r0   !may11
// ! (44) More major restructuring of code to use fixed XLS, XLV, XLF   !may12
// ! (45) Increased VEL_INC ~ VrimeF**2, put the enhanced graupel/hail fall speeds
// !      from Thompson into the code but only in limited circumstances, restructured
// !      and streamlined the INDEXS calculation, removed the upper limit for
// !      for the vapor mixing ratio is at water saturation when calculating ice
// !      deposition, and N0r is gradually increased for conditions supporting
// !      drizzle when rain contents decrease below 0.25 g/m**3.   !may17
// ! (46) The may11 code changes that increase N0r0 when rain contents exceed 1 g m^-3
// !      have been removed, limit the number of iterations calculating final rain
// !      parameters, remove the revised N0r calculation for reflectivity. All of
// !      the changes following those made in the may10 code.  !may20
// ! (47) Reduce the assumed # concentration of hail/sleet when RF>10 from 5 L^-1 to
// !      1 L^-1, and also reduce it for graupel when RF>5 from 10 L^-1 to 5 L^-1.
// !      This is being done to try and make greater use of the Thompson graupel/hail 
// !      fallspeeds by having INDEXS==MDImax.
// ! (48) Increased NCW from 200e6 to 300e6 for a more delayed onset of drizzle, 
// !      simplified drizzle algorithm to reduce/eliminate N0r bulls eyes and to allow
// !      for supercooled drizzle, and set limits for 8.e6 <= N0r (m^-4) <= 1.e9  !may31
// ! (49) Further restructuring of code to better define STRAT, DRZL logicals,
// !      add these rain flags to mprates arrays   !jun01
// ! (50) Increase in reflectivity due to wet ice was commented out.
// ! (51) Fixed minor bug to update INDEXR2 in the "rain_pass: do" loop.   !jun13
// ! (52) Final changes to Nsnow for boosting reflectivities from ice for 
// !      mass contents exceeding 5 g m^-3.  !jun16
// ! (53) Cosmetic changes only that do not affect the calculations. Removed old, unused 
// !      diagnostic arrays. Updated comments.
// !      
// !-----------------------------------------------------------------------------
// !
//
// Ported to CUDA for WOOF-NAM from phys/module_MP_FER_HIRES.F90:2746-2785 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (SUBROUTINE MY_GROWTH_RATES_NMM_hr(DTPH)):
//   REAL DT_ICE; REAL MY_600(35) (DATA, 35 values, g);
//   DT_ICE = (DTPH/600.)**1.5
//   MY_GROWTH_NMM(MY_T1:MY_T2) = DT_ICE*MY_600*1.E-3       (MY_T1=1, MY_T2=35, module array,
//                                                          SAVE, PRIVATE; kg)
// Every term is kept: the DATA table (below, same 35 values, same order), the 600 s reference
// (dtph/600), the 1.5 power (powf, single precision), and the g -> kg factor 1e-3 applied after the
// DT_ICE*MY_600 product, in the Fortran's left-to-right order.
//
// Layout: MY_GROWTH_NMM is 1-D, length 35; the CuPy array is float32[35], element k0 = MY_T1-1 + k
// (k0 = 0 .. 34), so Fortran MY_GROWTH_NMM(k) == out[k-1].
//
// Parallelism: one thread per table entry (35 threads), one launch, no host sync.
// Init-only: the wrapper computes it once for a given DTPH and reuses the device array.
#include "nmmb_common.cuh"

#define FER_MY_NTAB 35

__constant__ float FER_MY_600[FER_MY_NTAB] = {
    5.5e-8f, 1.4e-7f, 2.8e-7f, 6.e-7f, 3.3e-6f,
    2.e-6f, 9.e-7f, 8.8e-7f, 8.2e-7f, 9.4e-7f,
    1.2e-6f, 1.85e-6f, 5.5e-6f, 1.5e-5f, 1.7e-5f,
    1.5e-5f, 1.e-5f, 3.4e-6f, 1.85e-6f, 1.35e-6f,
    1.05e-6f, 1.e-6f, 9.5e-7f, 9.0e-7f, 9.5e-7f,
    9.5e-7f, 9.e-7f, 9.e-7f, 9.e-7f, 9.e-7f,
    9.e-7f, 9.e-7f, 9.e-7f, 9.e-7f, 9.e-7f
};   /* -31 to -35 deg C in the Fortran comment; values held invariant colder than -27 C */

// out: float32[35] (MY_GROWTH_NMM); dtph: physics step GSMDT (s), Fortran DTPH.
extern "C" __global__ void fer_growth_rates(float* __restrict__ out, const float dtph)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= FER_MY_NTAB) return;
    const float dt_ice = powf(dtph / 600.0f, 1.5f);     /* DT_ICE=(DTPH/600.)**1.5 */
    out[k] = dt_ice * FER_MY_600[k] * 1.0e-3f;          /* DT_ICE*MY_600*1.E-3 (g -> kg) */
}
