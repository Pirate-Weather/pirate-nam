// phys_rrtm_rsipath2.cu -- nam-gpu unit phys-rrtm-rsipath2.
//
// NOAA notice: phys/radiation_clouds_nmmb.f (NCEP/EMC NAM nam.v4.2.11, 83df4e13c1), subroutine
// rsipath2_tmp. NOAA/NCEP US government work; the NAM clone carries no licence file. The routine
// history block is kept as the notice of the source:
//   xx-xx-2001 b. ferrier   original program (rsipath)
//   xx-xx-2004 s. moorthi   modified for use in gfs model
//   05-20-2004 y. hou       added vertical index flag, rearranged to match radiation programs
//   02-24-2012 b. ferrier   simple temporary fix to separate cloud ice and snow
//
// Ported to CUDA for nam-gpu from phys/radiation_clouds_nmmb.f:2779-2964 (nam.v4.2.11, 83df4e13c1).
// Constants: phys/physcons.f (con_g, con_rd, con_rv, con_fvirt, con_t0c) and the routine's own
// PARAMETERs (CEXP, EPSQ, CN0r0, TNW), plus recwat_def and resnow_def (radiation_clouds_nmmb.f:180,183).
//
// One kernel, one launch, one thread per column (i,j); the l-loop runs inside the thread, the
// column recurrence shape of CONTRACT section 3 (kernels.grid_cols).
//
// Fortran-to-CUDA review (CONTRACT section 6, part 3). Every assignment of 2779-2964 has a term here:
//   resnow = resnow_def, cwatp/cicep/rainp/snowp/snden = 0       -> initial block (all cells)
//   rho = 100*plyr / (con_rd*tlyr*(1+con_fvirt*qlyr))            -> rho
//   cpath = |plvl(k+1)-plvl(k)| * (1e5/con_g)                    -> cpath
//   tc = tlyr - con_t0c              -> computed in Fortran, never read afterwards (dead). Not ported.
//   qsnow = 0.0                      -> dead in Fortran (never read). Not ported.
//   recwat = max(recwat_def, recw1*(rho*qcwat)^CEXP), cwatp = cpath*qcwat      (qcwat > 0)
//   rerain = 1.5*max(50, min(1000, CN0r0*sqrt(sqrt(rho*qrain)))), rainp = cpath*qrain (qrain > 0)
//   cicep = cpath*qcice                                          -> inside the totcnd > EPSQ block
// Unused dummy arguments (not ported, no term reads them): rrime, flgmin, lprnt (commented out in
// the Fortran). iflip only selects ksfc/k1, which the body never reads, so the kernel is the same
// for both orders. Its level index is the NMMB one: l=0 is the top, and plvl(k+1) is the interface
// below layer k, so the layout matches the Fortran with no flip.
//
// Fortran semantics kept on purpose: recwat and rerain are NOT zeroed at the start of the routine
// (only the cells that pass the qcwat > 0 or qrain > 0 test write them), so they keep their previous
// values elsewhere, exactly as the intent(out) arrays do in the Fortran.
//
// Index convention (CONTRACT section 1): plyr, tlyr, qlyr, qcwat, qcice, qrain and every output are
// [LM, JM, IM] float32; plvl is [LM+1, JM, IM] float32 (plvl[0] = top interface).
#include "nmmb_common.cuh"

#define RSI_CEXP        (1.0f / 3.0f)
#define RSI_EPSQ        1.0e-12f
#define RSI_CN0R0       2511.54f     // 1e6/(pi*1000*8e6)^0.25, N0r = 8e6 m^-4, RHOL = 1000 kg m^-3
#define RSI_TNW         200.0f       // droplet number, cm^-3
#define RSI_RECW_NUM    620.35f      // monodisperse Reff = 620.35*(Qcw/TNW)^(1/3)
#define RSI_RECWAT_DEF  5.0f         // recwat_def (radiation_clouds_nmmb.f:180), microns
#define RSI_RESNOW_DEF  50.0f        // resnow_def (radiation_clouds_nmmb.f:183), microns
#define RSI_CON_G       9.80665f     // physcons.f con_g
#define RSI_CON_RD      287.05f      // physcons.f con_rd (2.8705e+2)
#define RSI_CON_RV      461.5f       // physcons.f con_rv (4.6150e+2)
#define RSI_CON_FVIRT   (RSI_CON_RV / RSI_CON_RD - 1.0f)   // physcons.f con_fvirt

extern "C" __global__ void rsipath2_tmp(
    const int IM, const int JM, const int LM,
    const float* __restrict__ plyr,      // [LM, JM, IM] layer mean pressure, mb
    const float* __restrict__ plvl,      // [LM+1, JM, IM] interface pressure, mb
    const float* __restrict__ tlyr,      // [LM, JM, IM] layer temperature, K
    const float* __restrict__ qlyr,      // [LM, JM, IM] specific humidity, gm/gm
    const float* __restrict__ qcwat,     // [LM, JM, IM] cloud liquid condensate
    const float* __restrict__ qcice,     // [LM, JM, IM] cloud ice condensate
    const float* __restrict__ qrain,     // [LM, JM, IM] rain water
    float* cwatp,                        // [LM, JM, IM] cloud liquid path, g/m^2
    float* cicep,                        // [LM, JM, IM] cloud ice path, g/m^2
    float* rainp,                        // [LM, JM, IM] rain path, g/m^2
    float* snowp,                        // [LM, JM, IM] snow path, g/m^2
    float* recwat,                       // [LM, JM, IM] cloud liquid eff. radius, microns
    float* rerain,                       // [LM, JM, IM] rain eff. radius, microns
    float* resnow,                       // [LM, JM, IM] snow eff. radius, microns
    float* snden)                        // [LM, JM, IM] 1 / snow density
{
    COLUMN_IJ(IM, JM)

    // recw1 = 620.35/TNW**CEXP (monodisperse liquid droplet constant), once per column.
    const float recw1 = RSI_RECW_NUM / powf(RSI_TNW, RSI_CEXP);

    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t p  = IDX3(i0, j0, l0,     IM, JM);   // layer k
        const size_t pb = IDX3(i0, j0, l0 + 1, IM, JM);   // interface k+1

        // initial block: hydrometeor paths zeroed, snow radius at its default
        cwatp[p]  = 0.0f;
        cicep[p]  = 0.0f;
        rainp[p]  = 0.0f;
        snowp[p]  = 0.0f;
        snden[p]  = 0.0f;
        resnow[p] = RSI_RESNOW_DEF;

        const float qcw = qcwat[p];
        const float qci = qcice[p];
        const float qr  = qrain[p];
        const float totcnd = qcw + qci + qr;

        if (totcnd > RSI_EPSQ) {
            // air density (kg/m^3) and model mass thickness (g/m^2).
            // mb -> Pa is x100 in rho; the path uses x1e5/g (mb -> Pa and kg/s^2 -> g/s^2).
            const float rho = 100.0f * plyr[p]
                            / (RSI_CON_RD * tlyr[p] * (1.0f + RSI_CON_FVIRT * qlyr[p]));
            const float cpath = fabsf(plvl[pb] - plvl[p]) * (100000.0f / RSI_CON_G);

            // cloud water: effective radius and water path
            if (qcw > 0.0f) {
                const float tem = recw1 * powf(rho * qcw, RSI_CEXP);
                recwat[p] = fmaxf(RSI_RECWAT_DEF, tem);
                cwatp[p]  = cpath * qcw;
            }

            // rain: effective radius (factor 1.5 for exponential drops) and water path
            if (qr > 0.0f) {
                const float tem = RSI_CN0R0 * sqrtf(sqrtf(rho * qr));
                rerain[p] = 1.5f * fmaxf(50.0f, fminf(1000.0f, tem));
                rainp[p]  = cpath * qr;
            }

            // all ice is cloud ice; no snow (Oct 2013 change in the Fortran)
            cicep[p] = cpath * qci;
        }
    }
}
