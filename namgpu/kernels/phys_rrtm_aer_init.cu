// phys_rrtm_aer_init.cu -- aer_init of the NMMB radiation aerosol module: physparam flag logic,
// set_spectrum (one-wavenumber solar and IR fluxes) and set_volcaer (allocation of the IVOLAE table).
//
// NOTICE: NAM source file phys/radiation_aerosols_nmmb.f (NOAA/NCEP/EMC, US Government work, no
// licence file). The module header of that file is kept here as the notice:
//   'module_radiation_aerosols' -- climatological atmospheric aerosol schemes for radiation.
//   program history: apr 2003 y.-t. hou created; ... NCEP-Radiation_aerosols v5.2 Jan 2013 (VTAGAER).
// Ported to CUDA for WOOF-NAM from phys/radiation_aerosols_nmmb.f:445-816 (nam.v4.2.11, 83df4e13c1).
// Subprograms covered: aer_init (445-572), wrt_aerlog (host print, not ported, writes no state),
// set_spectrum (the wavenumber loops), set_volcaer (allocation only).
//
// Inputs (device int32, written by phys-rrtm-rad-initialize into ctx.tables['rad_physparam']):
//   RI_IAERFLG, RI_LASWFLG, RI_LALWFLG, RI_LAVOFLG, RI_IAERMDL (slots 4, 5, 6, 7, 8 of
//   namgpu/physics/rrtm/phys_rrtm_rad_initialize.py).
// Physparam constant lalw1bd = .false. (physparam.f:87), so NLWBND = NBDLW when lalwflg.
//
// Outputs:
//   aflags[AF_*]   module scalars (NSWBND, NLWBND, NSWLWBD, KYRSTR, ...), the set_spectrum guard
//                  do_spectrum (iaerflg /= 0 and /= 100), do_volcaer (iaerflg /= 0 and lavoflg),
//                  and status (0 ok, 1 wrt_aerlog stop, 2 aer_init scheme-selection stop). The
//                  wrapper reads status only after a stream sync, not inside the launch.
//   solfwv[NWVTOT], eirfwv[NWVTIR], nwvns0[NWVSOL], s0intv[NWVSOL]   (set_spectrum).
//
// Every assignment of aer_init and its contained subprograms, with its Fortran line:
//   kyrstr = kyrend = kyrsav = kmonsav = 1                                    (:500-503)
//   wrt_aerlog: iaermdl outside 0..2 is a stop (me == 0)                      -> status 1
//   if (iaerflg == 0) return                                                  (:515)
//   nswbnd = nbdsw if laswflg else 0                                          (:520-525)
//   nlwbnd = (lalw1bd ? 1 : nbdlw) if lalwflg else 0                          (:527-535)
//   nswlwbd = nswbnd + nlwbnd                                                 (:537)
//   if (iaerflg /= 100): set_spectrum; if iaermdl == 0 clim_aerinit (another  (:539-566)
//       unit), else stop                                                      -> status 2
//   if (lavoflg) set_volcaer                                                  (:568-572)
//   The module SAVE initial values NSWBND = NBDSW, NLWBND = NBDLW and
//   NSWLWBD = NBDSW + NBDLW (radiation_aerosols_nmmb.f:146-150) are kept for the
//   iaerflg == 0 early return, which never assigns them.
//   set_spectrum:
//     do nb = 1, NWVSOL: nw1 = 1 (nb == 1) else nw1 = nw1 + nwvns0(nb-1);
//                        nw2 = nw1 + nwvns0(nb) - 1; solfwv(nw1:nw2) = s0intv(nb)   (:830-842)
//     tmp1 = 2.0 * con_pi * con_plnk * con_c**2;
//     tmp2 = con_plnk * con_c / (con_boltz * con_t0c)                              (:851-852)
//     do nw = 1, NWVTIR: tmp3 = 100.0 * nw;
//                        eirfwv(nw) = tmp1 * tmp3**3 / (exp(tmp2*tmp3) - 1.0)      (:855-858)
//   set_volcaer: allocate ivolae(12,4,10). Allocation only; the table is zeroed by the
//     wrapper because the Fortran leaves it undefined until the volcanic init fills it.
//
// Constants: physcons.f values (con_pi, con_c, con_plnk, con_boltz, con_t0c), single precision.
// Precision (open, project lead to rule): the kernel is float. The pinned strict build (nmmb configure.nems.Linux.gnu_strict,
// CPPFLAGS = -DCHNK_RRTM=8, no -DSINGLE_PREC) gives machine.f kind_phys = 8, and solfwv/eirfwv/s0intv are
// real(kind=kind_phys) (radiation_aerosols_nmmb.f:180,495,496). So this unit is single precision by program choice,
// NOT bitwise to the pinned strict build's physics kind. Exact-in-double would need kind_phys=8 here.
// NBDSW = 14 (radsw_param_nmmb.f:147-149, NBANDS = 29 - 16 + 1), NBDLW = 16
// (radlw_param_nmmb.f:140), NWVSOL = 151, NWVTOT = 57600, NWVTIR = 4000 (:158-161).
//
// Data tables: AER_NWVNS0 and AER_S0INTV are the DATA statements of radiation_aerosols_nmmb.f
// (:166-215), extracted from that source file by script (151 values each, sum(nwvns0) = NWVTOT).
//
// Launches (CONTRACT section 3): aer_init_flags (1 thread), aer_init_tables (151 threads),
// aer_init_solfwv and aer_init_eirfwv (one thread per wavenumber). All run once at init, on
// ctx.stream, no host sync, no per-step cost.

#include "nmmb_common.cuh"

#define NWVSOL 151
#define NWVTOT 57600
#define NWVTIR 4000
#define NBDSW 14
#define NBDLW 16

// rad_physparam slots (phys_rrtm_rad_initialize.py)
#define RI_IAERFLG 4
#define RI_LASWFLG 5
#define RI_LALWFLG 6
#define RI_LAVOFLG 7
#define RI_IAERMDL 8

// aflags slots (read by the wrapper)
#define AF_NSWBND 0
#define AF_NLWBND 1
#define AF_NSWLWBD 2
#define AF_DO_SPECTRUM 3
#define AF_DO_VOLCAER 4
#define AF_STATUS 5
#define AF_KYRSTR 6
#define AF_KYREND 7
#define AF_KYRSAV 8
#define AF_KMONSAV 9

#define AER_PI      3.1415926535897931f
#define AER_C       2.99792458e+8f
#define AER_PLNK    6.6260693e-34f
#define AER_BOLTZ   1.3806505e-23f
#define AER_T0C     2.7315e+2f

__constant__ int AER_NWVNS0[NWVSOL] = {100,11,14,18,24,33,50,83,12,12,13,15,15,17,18,20,21,24,26,30,32,37,42,47,55,64,76,91,111,139,179,238,333,41,42,45,46,48,51,53,55,58,61,64,68,71,75,79,84,89,95,101,107,115,123,133,142,154,167,181,197,217,238,263,293,326,368,417,476,549,641,758,909,101,103,105,108,109,112,115,117,119,122,125,128,130,134,137,140,143,147,151,154,158,163,166,171,175,181,185,190,196,201,207,213,219,227,233,240,248,256,264,274,282,292,303,313,325,337,349,363,377,392,408,425,444,462,483,505,529,554,580,610,641,675,711,751,793,841,891,947,1008,1075,1150,1231,1323,1425,1538,1667,1633,14300};

__constant__ float AER_S0INTV[NWVSOL] = {1.60000e-6f,2.88000e-5f,3.60000e-5f,4.59200e-5f,6.13200e-5f,8.55000e-5f,1.28600e-4f,2.16000e-4f,2.90580e-4f,3.10184e-4f,3.34152e-4f,3.58722e-4f,3.88050e-4f,4.20000e-4f,4.57056e-4f,4.96892e-4f,5.45160e-4f,6.00600e-4f,6.53600e-4f,7.25040e-4f,7.98660e-4f,9.11200e-4f,1.03680e-3f,1.18440e-3f,1.36682e-3f,1.57560e-3f,1.87440e-3f,2.25500e-3f,2.74500e-3f,3.39840e-3f,4.34000e-3f,5.75400e-3f,7.74000e-3f,9.53050e-3f,9.90192e-3f,1.02874e-2f,1.06803e-2f,1.11366e-2f,1.15830e-2f,1.21088e-2f,1.26420e-2f,1.32250e-2f,1.38088e-2f,1.44612e-2f,1.51164e-2f,1.58878e-2f,1.66500e-2f,1.75140e-2f,1.84450e-2f,1.94106e-2f,2.04864e-2f,2.17248e-2f,2.30640e-2f,2.44470e-2f,2.59840e-2f,2.75940e-2f,2.94138e-2f,3.13950e-2f,3.34800e-2f,3.57696e-2f,3.84054e-2f,4.13490e-2f,4.46880e-2f,4.82220e-2f,5.22918e-2f,5.70078e-2f,6.19888e-2f,6.54720e-2f,6.69060e-2f,6.81226e-2f,6.97788e-2f,7.12668e-2f,7.27100e-2f,7.31610e-2f,7.33471e-2f,7.34814e-2f,7.34717e-2f,7.35072e-2f,7.34939e-2f,7.35202e-2f,7.33249e-2f,7.31713e-2f,7.35462e-2f,7.36920e-2f,7.23677e-2f,7.25023e-2f,7.24258e-2f,7.20766e-2f,7.18284e-2f,7.32757e-2f,7.31645e-2f,7.33277e-2f,7.36128e-2f,7.33752e-2f,7.28965e-2f,7.24924e-2f,7.23307e-2f,7.21050e-2f,7.12620e-2f,7.10903e-2f,7.12714e-2f,7.08012e-2f,7.03752e-2f,7.00350e-2f,6.98639e-2f,6.90690e-2f,6.87621e-2f,6.52080e-2f,6.65184e-2f,6.60038e-2f,6.47615e-2f,6.44831e-2f,6.37206e-2f,6.24102e-2f,6.18698e-2f,6.06320e-2f,5.83498e-2f,5.67028e-2f,5.51232e-2f,5.48645e-2f,5.12340e-2f,4.85581e-2f,4.85010e-2f,4.79220e-2f,4.44058e-2f,4.48718e-2f,4.29373e-2f,4.15242e-2f,3.81744e-2f,3.16342e-2f,2.99615e-2f,2.92740e-2f,2.67484e-2f,1.76904e-2f,1.40049e-2f,1.46224e-2f,1.39993e-2f,1.19574e-2f,1.06386e-2f,1.00980e-2f,8.63808e-3f,6.52736e-3f,4.99410e-3f,4.39350e-3f,2.21676e-3f,1.33812e-3f,1.12320e-3f,5.59000e-4f,3.60000e-4f,2.98080e-4f,7.46294e-5f};

// aer_init control logic (single thread; O(1) work, once at init).
extern "C" __global__ void aer_init_flags(const int* __restrict__ rip, int* __restrict__ aflags)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int iaerflg = rip[RI_IAERFLG];
    const int laswflg = rip[RI_LASWFLG] != 0;
    const int lalwflg = rip[RI_LALWFLG] != 0;
    const int lavoflg = rip[RI_LAVOFLG] != 0;
    const int iaermdl = rip[RI_IAERMDL];
    const int lalw1bd = 0;                       /* physparam lalw1bd = .false. */

    int status = 0;
    if (iaermdl < 0 || iaermdl > 2) status = 1;  /* wrt_aerlog stop */

    int nswbnd = NBDSW, nlwbnd = NBDLW, nswlwbd = NBDSW + NBDLW;  /* SAVE initial values */
    int do_spectrum = 0, do_volc = 0;
    if (iaerflg != 0) {                         /* if (iaerflg == 0) return  (:515) */
        nswbnd = laswflg ? NBDSW : 0;
        if (lalwflg) nlwbnd = lalw1bd ? 1 : NBDLW;
        else         nlwbnd = 0;
        nswlwbd = nswbnd + nlwbnd;
        if (iaerflg != 100) {                   /* set_spectrum block (:539) */
            do_spectrum = 1;
            if (iaermdl != 0 && status == 0) status = 2;  /* scheme-selection stop (:563-569); wrt_aerlog stop (status 1) comes first */
        }
        if (lavoflg) do_volc = 1;                /* set_volcaer (:568) */
    }

    aflags[AF_NSWBND] = nswbnd;
    aflags[AF_NLWBND] = nlwbnd;
    aflags[AF_NSWLWBD] = nswlwbd;
    aflags[AF_DO_SPECTRUM] = do_spectrum;
    aflags[AF_DO_VOLCAER] = do_volc;
    aflags[AF_STATUS] = status;
    aflags[AF_KYRSTR] = 1;                       /* kyrstr = kyrend = kyrsav = kmonsav = 1 (:500-503) */
    aflags[AF_KYREND] = 1;
    aflags[AF_KYRSAV] = 1;
    aflags[AF_KMONSAV] = 1;
}

// nwvns0 and s0intv module tables (DATA statements, :166-215), one thread per region.
extern "C" __global__ void aer_init_tables(int* __restrict__ nwvns0_out, float* __restrict__ s0intv_out)
{
    const int nb = blockIdx.x * blockDim.x + threadIdx.x;
    if (nb >= NWVSOL) return;
    nwvns0_out[nb] = AER_NWVNS0[nb];
    s0intv_out[nb] = AER_S0INTV[nb];
}

// set_spectrum, solar part (:830-842): solfwv(nw) = s0intv(nb) for nw in [nw1(nb), nw2(nb)].
extern "C" __global__ void aer_init_solfwv(float* __restrict__ solfwv, const int* __restrict__ aflags)
{
    const int nw = blockIdx.x * blockDim.x + threadIdx.x + 1;   /* 1-based, as the Fortran */
    if (nw > NWVTOT || aflags[AF_DO_SPECTRUM] == 0) return;
    int nw1 = 1;
    for (int nb = 1; nb <= NWVSOL; ++nb) {
        if (nb > 1) nw1 = nw1 + AER_NWVNS0[nb - 2];   /* nw1 = nw1 + nwvns0(nb-1) */
        const int nw2 = nw1 + AER_NWVNS0[nb - 1] - 1;
        if (nw >= nw1 && nw <= nw2) {
            solfwv[nw - 1] = AER_S0INTV[nb - 1];
            return;
        }
    }
}

// set_spectrum, IR part (:851-858): black-body one-wavenumber flux at T0C.
extern "C" __global__ void aer_init_eirfwv(float* __restrict__ eirfwv, const int* __restrict__ aflags)
{
    const int nw = blockIdx.x * blockDim.x + threadIdx.x + 1;   /* 1-based */
    if (nw > NWVTIR || aflags[AF_DO_SPECTRUM] == 0) return;
    const float tmp1 = 2.0f * AER_PI * AER_PLNK * (AER_C * AER_C);
    const float tmp2 = AER_PLNK * AER_C / (AER_BOLTZ * AER_T0C);
    const float tmp3 = 100.0f * (float)nw;
    eirfwv[nw - 1] = (tmp1 * (tmp3 * tmp3 * tmp3)) / (expf(tmp2 * tmp3) - 1.0f);
}
