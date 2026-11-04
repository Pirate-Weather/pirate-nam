// WOOF-NAM unit dyn-adv1-vert-uv: Crank-Nicolson vertical advection of U and V (ADV1, vertical section).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:1706-1837 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine adv1, "crank-nicholson vertical advection" section; w1 = 0.9, w2 = 2 - w1,
// dts = dt*(0.25*0.25)). Per column (i,j), with 1-based l:
//   [1] l=1:    pdop = (pd+pd(i+1)+pd(j+1)+pd(i+1,j+1)+pdo...)*0.125; pvvlo = (psgdt(l=1) 4-point sum)*dts;
//               vvlo = pvvlo/(dsg2(1)*pdop+pdsg1(1)); cmw = -vvlo*w2+1; rcmw = 1/cmw; crw = vvlo*w2;
//               rstu = (-vvlo*w1)*(u(2)-u(1))+u(1); rstv likewise with v.
//   [2] l=2..lm-1: rdp = 1/(dsg2(l)*pdop+pdsg1(l)); pvvup = pvvlo; pvvlo = psgdt(l) 4-point sum * dts;
//               vvup = pvvup*rdp; vvlo = pvvlo*rdp; cf = -vvup*w2*rcmw(l-1);
//               cmw = -crw(l-1)*cf+((vvup-vvlo)*w2+1); rcmw = 1/cmw; crw = vvlo*w2;
//               rstu = -rstu(l-1)*cf+u(l)-((u(l)-u(l-1))*vvup+(u(l+1)-u(l))*vvlo)*w1; rstv likewise.
//   [3] l=lm:   pvvup = pvvlo; vvup = pvvup/(dsg2(lm)*pdop+pdsg1(lm)); cf = -vvup*w2*rcmw(lm-1);
//               cmw = -crw(lm-1)*cf+(vvup*w2+1); rcmw = 1/cmw; crw(lm) = vvlo (the l=lm-1 value);
//               rstu = -rstu(lm-1)*cf+u(lm)-(u(lm)-u(lm-1))*vvup*w1; rstv likewise;
//               tcu(lm) = rstu(lm)*rcmw(lm)-u(lm); tcv(lm) = rstv(lm)*rcmw(lm)-v(lm).
//   [4] l=lm-1..1 (backward): tcu(l) = (-crw(l)*(u(l+1)+tcu(l+1))+rstu(l))*rcmw(l)-u(l); tcv likewise.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   its_b1 = 2 .. ite_b2 = IM-2 (1-based), jts_b1 = 2 .. jte_b2 = JM-2  -> i0 in [1, IM-3], j0 in [1, JM-3].
//   The b2 limit is the one the Fortran uses for the loops (ite_b2 = min(ite, ide-2)); the pdop and psgdt
//   4-point sums read i+1, j+1 which stay inside the domain. Columns outside the range keep their TCU/TCV.
//   The OpenMP looplimits split (ENABLE_SMP) is dropped; one thread per column.
//
// pdop(i,j) is also read after this unit: the UV horizontal section (Fortran line 1992) uses the 8-point value,
// so it is written out to PDOP_UV [JM,IM] (pdop_out). The Fortran temporaries pvvlo, vvlo and the forward-sweep
// previous level (rcmw, crw, rstu, rstv at
// l-1) are per-column scalars held in registers here. The four level arrays needed by the backward sweep
// (rcmw, crw, rstu, rstv at every level) are full-domain scratch arrays [LM,JM,IM] (RCMW, CRW, RSTU, RSTV,
// added by the wrapper with st.add), the same storage the Fortran uses for its (its_b1:ite_b2,...,lm) temporaries.
// The Fortran U, V, PD, PDO, PSGDT, DSG2 and PDSG1 are read only; TCU and TCV are outputs.
// Not ported: the commented-out "if(abs(cmw).lt.epscm)" floor (the Fortran comments it out), the OpenMP block,
// and the ENABLE_SMP looplimits call. No line of the routine's assignments is skipped.

#include "nmmb_common.cuh"

#define U3(l0) u[IDX3(i0, j0, (l0), IM, JM)]
#define V3(l0) v[IDX3(i0, j0, (l0), IM, JM)]
// psgdt 4-point sum at level l0: psgdt(i,j,l)+psgdt(i+1,j,l)+psgdt(i,j+1,l)+psgdt(i+1,j+1,l)
#define PSG4(l0) (psgdt[IDX3(i0, j0, (l0), IM, JM)] + psgdt[IDX3(i0 + 1, j0, (l0), IM, JM)] \
                  + psgdt[IDX3(i0, j0 + 1, (l0), IM, JM)] + psgdt[IDX3(i0 + 1, j0 + 1, (l0), IM, JM)])

extern "C" __global__ void dyn_adv1_vert_uv(
    const int IM, const int JM, const int LM,
    const float dt,
    const float* __restrict__ dsg2,    // DSG2  [LM]          Fortran dsg2(1:lm)
    const float* __restrict__ pdsg1,   // PDSG1 [LM]          Fortran pdsg1(1:lm)
    const float* __restrict__ pd,      // PD    [JM,IM]       in
    const float* __restrict__ pdo,     // PDO   [JM,IM]       in
    const float* __restrict__ psgdt,   // PSGDT [LM+1,JM,IM]  in, levels 0..LM-2 read
    const float* __restrict__ u,       // U     [LM,JM,IM]    in
    const float* __restrict__ v,       // V     [LM,JM,IM]    in
    float* __restrict__ tcu,           // TCU   [LM,JM,IM]    out
    float* __restrict__ tcv,           // TCV   [LM,JM,IM]    out
    float* __restrict__ rcmw,          // RCMW  [LM,JM,IM]    scratch
    float* __restrict__ crw,           // CRW   [LM,JM,IM]    scratch
    float* __restrict__ rstu,          // RSTU  [LM,JM,IM]    scratch
    float* __restrict__ rstv,          // RSTV  [LM,JM,IM]    scratch
    float* __restrict__ pdop_out)      // PDOP_UV [JM,IM]     out (read by dyn-adv1-horiz-uv)
{
    COLUMN_IJ(IM, JM)

    // b2 range: i in [its_b1, ite_b2] = [2, IM-2], j in [jts_b1, jte_b2] = [2, JM-2] (1-based)
    if (i0 < 1 || i0 > IM - 3 || j0 < 1 || j0 > JM - 3) return;

    const float dts = dt * (0.25f * 0.25f);
    const float w1 = 0.9f;
    const float w2 = 2.0f - w1;

    const size_t c2 = IDX2(i0, j0, IM);
    // pdop = (pd(i,j)+pd(i+1,j)+pd(i,j+1)+pd(i+1,j+1)+pdo(i,j)+pdo(i+1,j)+pdo(i,j+1)+pdo(i+1,j+1))*0.125
    const float pdop = (pd[c2] + pd[IDX2(i0 + 1, j0, IM)] + pd[IDX2(i0, j0 + 1, IM)] + pd[IDX2(i0 + 1, j0 + 1, IM)]
                        + pdo[c2] + pdo[IDX2(i0 + 1, j0, IM)] + pdo[IDX2(i0, j0 + 1, IM)] + pdo[IDX2(i0 + 1, j0 + 1, IM)])
                       * 0.125f;
    pdop_out[c2] = pdop;   // Fortran pdop(i,j) at line 1731-1732, consumed at line 1992

    // ---- [1] Fortran l=1 (l0 = 0)
    float pvvlo = PSG4(0) * dts;
    float vvlo = pvvlo / (dsg2[0] * pdop + pdsg1[0]);
    float cmw = -vvlo * w2 + 1.0f;
    float rcmw_p = 1.0f / cmw;             // level l-1 values carried in registers
    float crw_p = vvlo * w2;
    float rstu_p = (-vvlo * w1) * (U3(1) - U3(0)) + U3(0);
    float rstv_p = (-vvlo * w1) * (V3(1) - V3(0)) + V3(0);
    rcmw[IDX3(i0, j0, 0, IM, JM)] = rcmw_p;
    crw[IDX3(i0, j0, 0, IM, JM)] = crw_p;
    rstu[IDX3(i0, j0, 0, IM, JM)] = rstu_p;
    rstv[IDX3(i0, j0, 0, IM, JM)] = rstv_p;

    // ---- [2] Fortran l=2..lm-1 (l0 = 1..LM-2)
    for (int l0 = 1; l0 <= LM - 2; ++l0) {
        const float rdp = 1.0f / (dsg2[l0] * pdop + pdsg1[l0]);
        const float pvvup = pvvlo;
        pvvlo = PSG4(l0) * dts;
        const float vvup = pvvup * rdp;
        vvlo = pvvlo * rdp;
        const float cf = -vvup * w2 * rcmw_p;
        cmw = -crw_p * cf + ((vvup - vvlo) * w2 + 1.0f);
        const float rcmw_l = 1.0f / cmw;
        const float crw_l = vvlo * w2;
        const float rstu_l = -rstu_p * cf + U3(l0) - ((U3(l0) - U3(l0 - 1)) * vvup + (U3(l0 + 1) - U3(l0)) * vvlo) * w1;
        const float rstv_l = -rstv_p * cf + V3(l0) - ((V3(l0) - V3(l0 - 1)) * vvup + (V3(l0 + 1) - V3(l0)) * vvlo) * w1;
        rcmw[IDX3(i0, j0, l0, IM, JM)] = rcmw_l;
        crw[IDX3(i0, j0, l0, IM, JM)] = crw_l;
        rstu[IDX3(i0, j0, l0, IM, JM)] = rstu_l;
        rstv[IDX3(i0, j0, l0, IM, JM)] = rstv_l;
        rcmw_p = rcmw_l;
        crw_p = crw_l;
        rstu_p = rstu_l;
        rstv_p = rstv_l;
    }

    // ---- [3] Fortran l=lm (l0 = LM-1): the last level, and the start of the backward sweep
    const int lt = LM - 1;
    const float pvvup_t = pvvlo;
    const float vvup_t = pvvup_t / (dsg2[lt] * pdop + pdsg1[lt]);
    const float cf_t = -vvup_t * w2 * rcmw_p;
    cmw = -crw_p * cf_t + (vvup_t * w2 + 1.0f);
    const float rcmw_t = 1.0f / cmw;
    const float crw_t = vvlo;               // Fortran crw(i,j,lm)=vvlo, the l=lm-1 value
    const float rstu_t = -rstu_p * cf_t + U3(lt) - (U3(lt) - U3(lt - 1)) * vvup_t * w1;
    const float rstv_t = -rstv_p * cf_t + V3(lt) - (V3(lt) - V3(lt - 1)) * vvup_t * w1;
    rcmw[IDX3(i0, j0, lt, IM, JM)] = rcmw_t;
    crw[IDX3(i0, j0, lt, IM, JM)] = crw_t;
    rstu[IDX3(i0, j0, lt, IM, JM)] = rstu_t;
    rstv[IDX3(i0, j0, lt, IM, JM)] = rstv_t;
    float tcu_next = rstu_t * rcmw_t - U3(lt);
    float tcv_next = rstv_t * rcmw_t - V3(lt);
    tcu[IDX3(i0, j0, lt, IM, JM)] = tcu_next;
    tcv[IDX3(i0, j0, lt, IM, JM)] = tcv_next;

    // ---- [4] Fortran l=lm-1..1 (l0 = LM-2..0), backward sweep; tcu(l+1), tcv(l+1) carried in registers
    for (int l0 = LM - 2; l0 >= 0; --l0) {
        const size_t k = IDX3(i0, j0, l0, IM, JM);
        const float tcu_l = (-crw[k] * (U3(l0 + 1) + tcu_next) + rstu[k]) * rcmw[k] - U3(l0);
        const float tcv_l = (-crw[k] * (V3(l0 + 1) + tcv_next) + rstv[k]) * rcmw[k] - V3(l0);
        tcu[k] = tcu_l;
        tcv[k] = tcv_l;
        tcu_next = tcu_l;
        tcv_next = tcv_l;
    }
}

#undef U3
#undef V3
#undef PSG4
