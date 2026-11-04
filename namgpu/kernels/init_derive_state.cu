// init_derive_state.cu -- derived prognostics after the native NAM init read (unit init-derive-state).
//
// Notice: NMMB sources in NAM nam.v4.2.11 (MatthewPyle-NOAA/NAM @ 83df4e13c1, NOAA/NCEP/EMC) have no
// author or history header block of their own; the NOAA source notice in NOTICE applies. NAM is US
// Government work with no licence file.
//
// Ported to CUDA for WOOF-NAM from (nam.v4.2.11, 83df4e13c1). Every kernel below is cold-start unless
// it says otherwise. On a restart, native reads these fields from the restart file (the restart branch of
// nmm/module_INIT_READ_BIN.F90, lines 639-2763), so the kernels leave them alone.
//
//   nmm/module_INIT_READ_BIN.F90:463-466   PINT(1) = PT; PINT(l+1) = PINT(l) + DSG2(l)*PD + PDSG1(l),
//                                          PDSG1(l) = DSG1(l)*PDTOP (:764, :2755). Restart reads PINT (:1900).
//   nmm/module_INIT_READ_BIN.F90:430-455   UP/VP/TP = U/V/T; cold start only (restart reads them, :2427/:2570/:2649)
//   nmm/module_INIT_READ_BIN.F90:444-451   cold start: Q2 = 0.02, O3 = 0 with 10 in the box below, DWDT = 1, W = 0.
//                                          Restart reads Q2 (:1962), O3 (:1752), DWDT (:1518), W (:2677-2684).
//   nmm/module_INIT_READ_BIN.F90:448       O3 box (the active line is the global one): i in [IDE/2+1-6, IDE/2+1+6],
//                                          j in [JDE*3/4+1-6, JDE*3/4+1+6]; IDE = IM, JDE = JM on the single-card
//                                          whole-domain tile (CONTRACT section 1).
//   nmm/module_INIT_READ_BIN.F90:476-478   TRACERS_PREV(indx_q, cw, q2) = sqrt(max(x, 0)); restart reads them (:2456).
//   nmm/module_INIT_READ_BIN.F90:2542-2555 TOLD = 0 on a cold start (restart reads it)
//   nmm/module_SOLVER_GRID_COMP.F90:942    OMGALF = 0 (cold start; restart reads it, :1768)
//   nmm/module_SOLVER_GRID_COMP.F90:10240  THS = TSKIN*(100000/(SG2(LM+1)*PD + PSG1(LM+1)))**CAPPA (not on restart)
//   nmm/module_DYNAMICS_ROUTINES.F90:213   RTOP = DFDP = (Q*0.608 + (1-CW))*T*R_D/APELP. The literal 0.608 is
//                                          the value native uses in its dynamics (here and at :3069); the P608
//                                          constant (R_V/R_D - 1, module_CONSTANTS.F90:45) is not used.
//   nmm/module_DYNAMICS_ROUTINES.F90:3032-3069  Z: rg = 1/G, trog = 2*R_D/G, zlo = FIS*rg, then the upward
//                                          recursion dz = (Q*0.608+(1-CW))*T*trog*(DSG2*PD+PDSG1)/(PINT(l)+PINT(l+1)),
//                                          Z(l) = zlo + dz/2, zlo += dz. This is the recursion the dynamics runs
//                                          every step (it writes Z). The storm-search copy at
//                                          module_PARENT_CHILD_CPL_COMP.F90:23853-23864 is inside
//                                          COMPUTE_STORM_MOTION, whose call is commented out in this clone, so it
//                                          is not the init path. Restart reads Z (:2721).
//   nmm/module_SOLVER_GRID_COMP.F90 SOLVER_INITIALIZE (allocation init, before the read, on both paths):
//                                          REFL_10CM = DBZmin (-20), TADJ = 0, QSH = 0. Restart reads all three,
//                                          so these are set on a cold start only.
//   TRACERS(indx_q, indx_cw, indx_q2) = Q, CW, Q2: NMMB keeps Q/CW/Q2 as aliases of TRACERS slots
//          (module_PARENT_CHILD_CPL_COMP.F90:3112); WOOF-NAM keeps separate arrays, so the slots are copied.
//
// Not derived here: FIS is a read-only input (read from the file at module_INIT_READ_BIN.F90:1549-1553; the
// presence check is in the Python wrapper). F_ICE/F_RAIN/F_RIMEF cold fills are owned by phys-fer-init. T/U/V
// pre-read sentinels (-1E6, SOLVER_INITIALIZE) are overwritten by the init read and are not state on this path.
//
// Layout (CONTRACT.md 1): zero-based i0 = i-1, j0 = j-1, l0 = l-1; l0 = 0 is the model top.
// PINT, Z: LM+1 interfaces; everything else LM layers. TRACERS[n, l, j, i]; slot indices arrive
// zero-based (Fortran INDX_x minus 1).
//
// Fast numerics: fmad contraction on, IEEE division. Single precision. No host sync, no malloc, no printf.

#include "nmmb_common.cuh"

#define NMMB_DYN_608  0.608f                        // dynamics literal (module_DYNAMICS_ROUTINES.F90:213, :3069)
#define NMMB_TROG     (2.0f * NMMB_R_D / NMMB_G)    // trog = 2.*r/g (module_DYNAMICS_ROUTINES.F90:3034)
#define NMMB_DBZMIN   (-20.0f)                      // module_CONSTANTS dbzmin
#define NMMB_Q2_COLD  0.02f                         // module_INIT_READ_BIN.F90:448 (cold start)
#define NMMB_O3_BOX   10.0f                         // module_INIT_READ_BIN.F90:448
#define NMMB_O3_HALF  6                             // module_INIT_READ_BIN.F90:448 (half-width 6)

// PINT at all LM+1 interfaces, by the native recursion (module_INIT_READ_BIN.F90:463-466). One thread per column.
// Restart: PINT is read from the restart file, so nothing is written.
extern "C" __global__ void init_derive_pint(
    const float* __restrict__ dsg1,    // [LM]
    const float* __restrict__ dsg2,    // [LM]
    const float* __restrict__ pd,      // [JM, IM]
    float* __restrict__ pint,          // [LM+1, JM, IM]  out
    const float pt,
    const float pdtop,
    const int restart,
    const int im, const int jm, const int lm)
{
    COLUMN_IJ(im, jm)
    if (restart) return;
    const float pdij = pd[IDX2(i0, j0, im)];
    pint[IDX3(i0, j0, 0, im, jm)] = pt;                       // PINT(1) = PT
    for (int l0 = 0; l0 < lm; ++l0) {
        const float pdsg1 = dsg1[l0] * pdtop;                 // PDSG1(L) = DSG1(L)*PDTOP
        pint[IDX3(i0, j0, l0 + 1, im, jm)] =
            pint[IDX3(i0, j0, l0, im, jm)] + dsg2[l0] * pdij + pdsg1;
    }
}

// Per-layer point fields. RTOP is derived on both paths. Everything else is cold-start only.
extern "C" __global__ void init_derive_levels(
    const float* __restrict__ t,       // [LM, JM, IM]
    const float* __restrict__ q,       // [LM, JM, IM]
    const float* __restrict__ cw,      // [LM, JM, IM]
    const float* __restrict__ u,       // [LM, JM, IM]
    const float* __restrict__ v,       // [LM, JM, IM]
    const float* __restrict__ pint,    // [LM+1, JM, IM]
    float* __restrict__ rtop,          // [LM, JM, IM]  out
    float* __restrict__ dwdt,          // [LM, JM, IM]  out (cold)
    float* __restrict__ omgalf,        // [LM, JM, IM]  out (cold)
    float* __restrict__ up,            // [LM, JM, IM]  out (cold)
    float* __restrict__ vp,            // [LM, JM, IM]  out (cold)
    float* __restrict__ tp,            // [LM, JM, IM]  out (cold)
    float* __restrict__ told,          // [LM, JM, IM]  out (cold)
    float* __restrict__ w,             // [LM, JM, IM]  out (cold)
    float* __restrict__ o3,            // [LM, JM, IM]  out (cold)
    float* __restrict__ q2,            // [LM, JM, IM]  out (cold)
    const int restart,
    const int ide, const int jde,
    const int im, const int jm, const int lm)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t k = IDX3(i0, j0, l0, im, jm);

    // APELP = (PINT(l) + PINT(l+1)) * 0.5 ; DFDP = (Q*0.608 + (1-CW)) * T * R_D / APELP
    const float apelp = (pint[k] + pint[IDX3(i0, j0, l0 + 1, im, jm)]) * 0.5f;
    rtop[k] = (q[k] * NMMB_DYN_608 + (1.0f - cw[k])) * t[k] * NMMB_R_D / apelp;

    if (restart) return;                  // restart reads DWDT, OMGALF, UP/VP/TP, TOLD, W, Q2, O3 from file

    dwdt[k]   = 1.0f;                     // module_INIT_READ_BIN.F90:448 (cold)
    omgalf[k] = 0.0f;                     // module_SOLVER_GRID_COMP.F90:942
    up[k] = u[k];
    vp[k] = v[k];
    tp[k] = t[k];
    told[k] = 0.0f;                       // module_INIT_READ_BIN.F90:2542-2555
    w[k] = 0.0f;                          // module_INIT_READ_BIN.F90:451 (cold)
    q2[k] = NMMB_Q2_COLD;                 // module_INIT_READ_BIN.F90:444-448 (cold)

    // O3 (cold): 0 everywhere, 10 in the box. Fortran i,j are 1-based; the box is the active global line.
    const int i = i0 + 1, j = j0 + 1;
    const int ci = ide / 2 + 1;           // ide/2+1
    const int cj = (jde * 3) / 4 + 1;     // jde*3/4+1
    const bool in_box = (i >= ci - NMMB_O3_HALF && i <= ci + NMMB_O3_HALF &&
                         j >= cj - NMMB_O3_HALF && j <= cj + NMMB_O3_HALF);
    o3[k] = in_box ? NMMB_O3_BOX : 0.0f;
}

// TRACERS(n, l) = Q, CW, Q2 at their slots (both paths), and TRACERS_PREV = sqrt(max(x, 0)) (cold only).
// Runs after init_derive_levels on the same stream, so on a cold start Q2 is already 0.02 here.
extern "C" __global__ void init_derive_tracers(
    const float* __restrict__ q,       // [LM, JM, IM]
    const float* __restrict__ cw,      // [LM, JM, IM]
    const float* __restrict__ q2,      // [LM, JM, IM]
    float* __restrict__ tracers,       // [NTR, LM, JM, IM]  out
    float* __restrict__ tracers_prev,  // [NTR, LM, JM, IM]  out (cold)
    const int n_q, const int n_cw, const int n_q2,   // zero-based slots
    const int restart,
    const int im, const int jm, const int lm)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t k = IDX3(i0, j0, l0, im, jm);

    const float vq  = q[k];
    const float vcw = cw[k];
    const float vq2 = q2[k];

    const size_t kq  = IDX4(i0, j0, l0, n_q,  im, jm, lm);
    const size_t kcw = IDX4(i0, j0, l0, n_cw, im, jm, lm);
    const size_t kq2 = IDX4(i0, j0, l0, n_q2, im, jm, lm);

    tracers[kq]  = vq;
    tracers[kcw] = vcw;
    tracers[kq2] = vq2;
    if (restart) return;                  // restart reads TRACERS_PREV (module_INIT_READ_BIN.F90:2456)
    tracers_prev[kq]  = sqrtf(fmaxf(vq,  0.0f));
    tracers_prev[kcw] = sqrtf(fmaxf(vcw, 0.0f));
    tracers_prev[kq2] = sqrtf(fmaxf(vq2, 0.0f));
}

// Midlayer geopotential height Z, by the recursion the dynamics runs each step (module_DYNAMICS_ROUTINES.F90
// :3032-3069): zlo = FIS*rg; dz = (Q*0.608+(1-CW))*T*trog*(DSG2*PD+PDSG1)/(PINT(l)+PINT(l+1)); Z(l) = zlo + dz/2;
// zlo += dz. Z[LM] = FIS*GI is the surface interface. Restart reads Z (module_INIT_READ_BIN.F90:2721).
// One thread per column; upward recursion from the surface.
extern "C" __global__ void init_derive_z(
    const float* __restrict__ fis,     // [JM, IM]
    const float* __restrict__ pd,      // [JM, IM]
    const float* __restrict__ t,       // [LM, JM, IM]
    const float* __restrict__ q,       // [LM, JM, IM]
    const float* __restrict__ cw,      // [LM, JM, IM]
    const float* __restrict__ pint,    // [LM+1, JM, IM]
    const float* __restrict__ dsg1,    // [LM]
    const float* __restrict__ dsg2,    // [LM]
    float* __restrict__ z,             // [LM+1, JM, IM]  out
    const float pdtop,
    const float gi,
    const int restart,
    const int im, const int jm, const int lm)
{
    COLUMN_IJ(im, jm)
    if (restart) return;
    const float pdij = pd[IDX2(i0, j0, im)];
    float zlo = fis[IDX2(i0, j0, im)] * gi;
    z[IDX3(i0, j0, lm, im, jm)] = zlo;

    for (int l0 = lm - 1; l0 >= 0; --l0) {
        const size_t k = IDX3(i0, j0, l0, im, jm);
        const float pdsg1 = dsg1[l0] * pdtop;
        const float sum = pint[k] + pint[IDX3(i0, j0, l0 + 1, im, jm)];
        const float dz = (q[k] * NMMB_DYN_608 + (1.0f - cw[k])) * t[k] * NMMB_TROG *
                         (dsg2[l0] * pdij + pdsg1) / sum;
        z[k] = dz * 0.5f + zlo;
        zlo = zlo + dz;
    }
}

// THS (potential skin temperature) from TSKIN; cold start only. One thread per (i, j).
extern "C" __global__ void init_derive_ths(
    const float* __restrict__ tskin,   // [JM, IM]
    const float* __restrict__ pd,      // [JM, IM]
    const float* __restrict__ sg1,     // [LM+1]
    const float* __restrict__ sg2,     // [LM+1]
    float* __restrict__ ths,           // [JM, IM]  out
    const float pdtop,
    const float pt,
    const int restart,
    const int im, const int jm, const int lm)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || restart) return;
    const size_t k2 = IDX2(i0, j0, im);
    const float psg1_top = sg1[lm] * pdtop + pt;          // PSG1(LM+1)
    const float denom = sg2[lm] * pd[k2] + psg1_top;      // SG2(LM+1)*PD + PSG1(LM+1)
    ths[k2] = tskin[k2] * powf(100000.0f / denom, NMMB_CAPPA);
}

// Cold-start allocation fills that SOLVER_INITIALIZE sets before the read (REFL_10CM = DBZmin, TADJ = 0,
// QSH = 0 are the values native keeps on a cold start). Flat 1-D launch: n3 = LM*JM*IM, n2 = JM*IM.
extern "C" __global__ void init_derive_cold_fill(
    float* __restrict__ refl_10cm,     // [LM, JM, IM]  out
    float* __restrict__ tadj,          // [LM, JM, IM]  out
    float* __restrict__ qsh,           // [JM, IM]      out
    const int n3, const int n2)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n3) {
        refl_10cm[idx] = NMMB_DBZMIN;
        tadj[idx] = 0.0f;
    }
    if (idx < n2) qsh[idx] = 0.0f;
}
