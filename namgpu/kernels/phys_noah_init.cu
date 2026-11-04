// phys_noah_init.cu -- unit phys-noah-init: NOAH_LSM_INIT (soil liquid water, canopy water, ISLTYP check).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone), module
//   MODULE_LS_NOAHLSM (Noah land surface model), nam.v4.2.11.
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:1164-1293 (nam.v4.2.11, 83df4e13c1).
// FRH2O (phys/module_LS_NOAHLSM.F90:2644-2782) is ported here too: NOAH_LSM_INIT is its only caller.
//
// Term-by-term review of NOAH_LSM_INIT (1164-1293):
//   Arguments: CANWAT (= NMMB CMC), ISLTYP, TSLB (= NMMB STC), SMOIS (= NMMB SMC), SH2O, IVEGSRC,
//     num_soil_layers, RESTART, ALLOWED_TO_READ, ids..ite, MYPE, MPI_COMM.
//     MYPE / MPI_COMM_COMP: MPI bookkeeping, dropped (single process, CONTRACT section 1).
//     RESTART: the whole body sits inside IF(.NOT.RESTART); the wrapper does not launch on restart.
//     ALLOWED_TO_READ -> CALL LSM_PARM_INIT(IVEGSRC): builds BB, MAXSMC, SATPSI.  That is unit
//       phys-noah-parm-tables, NOT in this file; its tables arrive as the bb/maxsmc/satpsi arguments.
//   ITF = MIN(ITE, IDE-1) -> IM-1, JTF = MIN(JTE, JDE-1) -> JM-1 on the full domain.
//     Fortran loops i = 1..ITF, j = 1..JTF map to i0 in [0, IM-2], j0 in [0, JM-2].
//   Lines 1219-1230, ISLTYP range check: IVEGSRC == 0 and ISLTYP < 1, or IVEGSRC == 1 and ISLTYP < 0,
//     sets ERRFLAG = 1 and WRITE(0,*) the offending point.  The kernel cannot print, so it records the
//     diagnostic in bad_isltyp (layout below).  The wrapper reads it back ONCE at init (not on the step
//     path) and writes the Fortran message to stderr.  The Fortran never reads ERRFLAG afterwards, so
//     there is no state effect.
//   Lines 1182 and 1234-1287: IF(.NOT.FNDSOILW) with FNDSOILW = .true. (PARAMETER, line 1183).
//     In NAM this block is DEAD: the soil-liquid-water initialisation and the CANWAT = 0 loop are
//     skipped, so CANWAT and SH2O keep whatever the caller set.  The block is ported here behind
//     the runtime flag fndsoilw: the wrapper passes 1 (the Fortran value), and fndsoilw = 0 runs it.
//     Inside it:
//       BX = BB(ISLTYP); SMCMAX = MAXSMC(ISLTYP); PSISAT = SATPSI(ISLTYP)  (lines 1239-1241)
//       if (BX > 0 and SMCMAX > 0 and PSISAT > 0): for each soil layer
//         TSLB < 273.149: explicit first guess FK (1251-1260), SH2O = MIN(FK, SMOIS),
//           then the iterative FRH2O (1262-1264), SH2O = FREE.
//         else SH2O = SMOIS.
//       else SH2O = SMOIS for every layer.
//       CANWAT = 0 on every point (1275-1284).
//   Index mapping: soil fields are [NSOIL, JM, IM] (layer 1 at top): IDX3(i0, j0, ns0, IM, JM).
//     Table index: Fortran BB(ISLTYP) -> bb[ISLTYP - 1].  Out-of-table ISLTYP (0 or > NSLTYPE) makes
//     Fortran read outside BB(1:NSLTYPE), which is undefined.  For such a column the Fortran test
//     "BX > 0 .and. SMCMAX > 0 .and. PSISAT > 0" cannot be evaluated, so the kernel takes the Fortran
//     ELSE path (SH2O = SMOIS on every layer: the defined value Fortran also takes whenever that test
//     fails).  The column is flagged (bad_isltyp bit 1) and its first index is recorded.  This is a
//     stated choice, not a match to Fortran's undefined read.  CANWAT = 0 as for every column.
//
// Parallelism: one thread per column (i, j), an ns-loop inside it (CONTRACT section 3: column
// recurrence). No host sync, no device malloc, no printf.

#include "nmmb_common.cuh"

#define NOAHINIT_HLICE 3.335E5f
#define NOAHINIT_GRAV  9.81f
#define NOAHINIT_T0    273.15f
#define NOAHINIT_BLIM  5.5f
// FRH2O PARAMETERs (module_LS_NOAHLSM.F90:2656-2658).
#define NOAHFRH_CK     8.0f
#define NOAHFRH_ERROR  0.005f
#define NOAHFRH_GS     9.81f
#define NOAHFRH_HLICE  3.335E5f
#define NOAHFRH_T0     273.15f

// FRH2O (module_LS_NOAHLSM.F90:2644-2782): supercooled liquid soil water, Koren et al. 1999 eqn 17.
// Input TKELV, SMC, SH2O, SMCMAX, BEXP (= B), PSIS.  Returns FREE.
// Option 1 is the iterated Newton solution (CK = 8 /= 0, so it always runs).  Option 2 is the
// explicit Flerchinger form, used only when option 1 did not converge in 10 iterations (KCOUNT == 0).
__device__ __forceinline__ float frh2o_dev(float tkelv, float smc, float sh2o,
                                           float smcmax, float bexp, float psis)
{
    float bx = bexp;
    float swl, df, denom, swlk, dswl, fk;
    int nlog = 0;
    int kcount = 0;

    if (bexp > NOAHINIT_BLIM) bx = NOAHINIT_BLIM;      // BX = BEXP, BLIM = 5.5

    // Temperature not significantly below freezing (T0 - 1e-3): FREE = SMC.
    if (tkelv > (NOAHFRH_T0 - 1.0E-3f)) return smc;

    // OPTION 1: iterated solution (CK /= 0).
    swl = smc - sh2o;                                  // initial guess for frozen content
    if (swl > (smc - 0.02f)) swl = smc - 0.02f;        // keep within bounds
    if (swl < 0.0f) swl = 0.0f;
    while ((nlog < 10) && (kcount == 0)) {
        nlog = nlog + 1;
        df = logf((psis * NOAHFRH_GS / NOAHFRH_HLICE) * powf(1.0f + NOAHFRH_CK * swl, 2.0f)
                  * powf(smcmax / (smc - swl), bx))
             - logf(-(tkelv - NOAHFRH_T0) / tkelv);
        denom = 2.0f * NOAHFRH_CK / (1.0f + NOAHFRH_CK * swl) + bx / (smc - swl);
        swlk = swl - df / denom;
        if (swlk > (smc - 0.02f)) swlk = smc - 0.02f;  // bounds for the mathematical solution
        if (swlk < 0.0f) swlk = 0.0f;
        dswl = fabsf(swlk - swl);
        swl = swlk;
        if (dswl <= NOAHFRH_ERROR) kcount = kcount + 1;
    }
    float free_val = smc - swl;

    // OPTION 2: explicit Flerchinger solution, only when option 1 did not converge.
    if (kcount == 0) {
        fk = powf((NOAHFRH_HLICE / (NOAHFRH_GS * (-psis))) * ((tkelv - NOAHFRH_T0) / tkelv),
                  -1.0f / bx) * smcmax;
        if (fk < 0.02f) fk = 0.02f;
        free_val = fminf(fk, smc);
    }
    return free_val;
}

extern "C" __global__ void phys_noah_init_soil(
    float* __restrict__ sh2o,               // [NSOIL, JM, IM]  INOUT (NMMB SH2O)
    float* __restrict__ canwat,             // [JM, IM]         INOUT (NMMB CMC)
    const float* __restrict__ smois,        // [NSOIL, JM, IM]  (NMMB SMC)
    const float* __restrict__ tslb,         // [NSOIL, JM, IM]  (NMMB STC)
    const int*   __restrict__ isltyp,       // [JM, IM]
    const float* __restrict__ bb,           // [NSLTYPE]  BB(1:NSLTYPE)     from phys-noah-parm-tables
    const float* __restrict__ maxsmc,       // [NSLTYPE]  MAXSMC(1:NSLTYPE) from phys-noah-parm-tables
    const float* __restrict__ satpsi,       // [NSLTYPE]  SATPSI(1:NSLTYPE) from phys-noah-parm-tables
    int*         __restrict__ bad_isltyp,   // [4] layout below
    const int im, const int jm, const int nsoil, const int nsltype,
    const int ivegsrc, const int fndsoilw)  // fndsoilw: 1 = Fortran .true. (block skipped), 0 = run block
// bad_isltyp layout (int[4]; the wrapper sets {0, INT_MAX, 0, INT_MAX} before launch):
//   [0] flags: bit 0 = ISLTYP range check failed (ERRFLAG); bit 1 = ISLTYP outside table 1..NSLTYPE
//   [1] smallest linear column index (j0*IM + i0) failing the range check
//   [2] number of columns failing the range check
//   [3] smallest linear column index outside the table (dead block only)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= im * jm) return;
    const int i0 = col % im;
    const int j0 = col / im;
    if (i0 >= im - 1 || j0 >= jm - 1) return;          // ITF = IM-1, JTF = JM-1 (1-based interior)
    const size_t c2 = IDX2(i0, j0, im);
    const int isl = isltyp[c2];

    // ISLTYP range check (lines 1219-1230).  The Fortran WRITE is dropped; the flag is kept.
    if ((ivegsrc == 0 && isl < 1) || (ivegsrc == 1 && isl < 0)) {
        atomicOr(bad_isltyp, 1);
        atomicMin(bad_isltyp + 1, (int)c2);
        atomicAdd(bad_isltyp + 2, 1);
    }

    // Dead block with FNDSOILW = .true. (lines 1234-1287).  Runs only when fndsoilw == 0.
    if (fndsoilw == 0) {
        const bool table_ok = (isl >= 1 && isl <= nsltype);
        if (!table_ok) {
            // Fortran reads BB(0 or >30): undefined.  Take the ELSE path (SH2O = SMOIS), see header.
            atomicOr(bad_isltyp, 2);
            atomicMin(bad_isltyp + 3, (int)c2);
            for (int ns = 0; ns < nsoil; ++ns) {
                const size_t c3 = IDX3(i0, j0, ns, im, jm);
                sh2o[c3] = smois[c3];
            }
        } else {
            const int k = isl - 1;                      // BB(ISLTYP) as a 0-based index
            float bx = bb[k];
            float smcmax = maxsmc[k];
            float psisat = satpsi[k];
            if (bx > 0.0f && smcmax > 0.0f && psisat > 0.0f) {
                for (int ns = 0; ns < nsoil; ++ns) {
                    const size_t c3 = IDX3(i0, j0, ns, im, jm);
                    if (tslb[c3] < 273.149f) {
                        // Explicit first guess (Koren et al. 1999 eqn 17, KCOUNT = 0 form).
                        bx = bb[k];
                        smcmax = maxsmc[k];
                        psisat = satpsi[k];
                        if (bx > NOAHINIT_BLIM) bx = NOAHINIT_BLIM;
                        float fk = powf((NOAHINIT_HLICE / (NOAHINIT_GRAV * (-psisat)))
                                        * ((tslb[c3] - NOAHINIT_T0) / tslb[c3]),
                                        -1.0f / bx) * smcmax;
                        if (fk < 0.02f) fk = 0.02f;
                        sh2o[c3] = fminf(fk, smois[c3]);
                        // Iterative solution with FRH2O from that first guess.
                        const float free_val = frh2o_dev(tslb[c3], smois[c3], sh2o[c3],
                                                         smcmax, bx, psisat);
                        sh2o[c3] = free_val;
                    } else {
                        sh2o[c3] = smois[c3];           // SH2O = SMOIS for T >= 273.149 K
                    }
                }
            } else {
                for (int ns = 0; ns < nsoil; ++ns) {
                    const size_t c3 = IDX3(i0, j0, ns, im, jm);
                    sh2o[c3] = smois[c3];
                }
            }
        }
        // Canopy water starts at zero (lines 1275-1284).
        canwat[c2] = 0.0f;
    }
}
