// WOOF-NAM unit phys-bmj-col-deep-iter: one pass of the cloud-efficiency iteration body of BMJ deep
// convection (cloud efficiency, enthalpy conservation, lqm, temperature and moisture corrections).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module and routine header kept as the notice:
//   module module_cu_bmj  (NMMB, nam.v4.2.11)
//   subroutine bmj  (deep convection; BMJ scheme)
//   programer: z. janjic, yugoslav fed. hydromet. inst., beograd
// Ported to CUDA for WOOF-NAM from phys/module_CU_BMJ.F90:1085-1179 (nam.v4.2.11, 83df4e13c1).
//
// Scope: lines 1085-1179 are the body of `cloud_efficiency : do itrefi=1,itrefi_max`
// (opened at 1088, closed at 1216). Lines 1180-1216 (heating/moistening, precipitation, the
// efficiency update and the loop close) belong to other units. The itrefi loop is driven by the
// caller; this kernel is ONE pass of the body.
//
// Fortran -> CUDA, one thread per column (i, j). Levels stay 1-based Fortran in the code: macro
// P3(a,l) reads a[IDX3(i0,j0,l-1)]. Column scalars (ltop, lb, l0, sm, sm1, pbotfc, efi, depwl, depth,
// pk0, pkb, pkt, the slop*/dsp* column arrays) are per (i,j) arrays. Constants are scalars.
//
// Mapping:
//   1091-1097  dspbk, dsp0k, dsptk from efi, sm, sm1, pbotfc and the slope/dsp column arrays
//   1099-1113  dsp per level from the depwl>=depth and l<l0 branches; humidity profile for pk(l)>pqm
//              (psk, rxnersk, thsk, qrefk); else qrefk = qk
//   1136-1177  enthalpy_conservation do iter=1,2: sumde, sumdp over ltop..lb; hcorr; lcor=ltop+1;
//              lqm = largest l in 1..lb with pk(l)<=pqm (1 if none); above lqm trefk += hcorr*rcp;
//              lcor..lb: tskl, dhdt, trefk, thskl, qrefk
// Writes: trefk, qrefk (in place), psk, rxnersk, thsk (scratch, [LM,JM,IM]).
// Reads: pk, qk, tk, dprs, el, rxnerk. Scalars sumde, sumdp, hcorr, lcor, lqm are per thread.
// Every assignment in 1085-1179 has a counterpart below; no term is dropped.
//
// Precision: kfpt = single, float everywhere. Fast numerics (fmad on, IEEE division).
// Fortran "**" with real exponent -> powf; "**2" -> x*x (gfortran).
// No host sync, no device malloc, no printf. One launch per pass.

#include "nmmb_common.cuh"

#define P3(a, l) a[IDX3(i0, j0, (l) - 1, IM, JM)]
#define P2(a)    a[IDX2(i0, j0, IM)]

extern "C" __global__ void phys_bmj_col_deep_iter(
    const int IM, const int JM, const int LM,
    const float efimn, const float pq0, const float pqm,
    const float a2, const float a3, const float a4,
    const float cappa, const float cp, const float rcp, const float a23m4l,
    const float* __restrict__ sm, const float* __restrict__ sm1, const float* __restrict__ pbotfc,
    const float* __restrict__ efi, const float* __restrict__ depwl, const float* __restrict__ depth,
    const float* __restrict__ pk0c, const float* __restrict__ pkbc, const float* __restrict__ pktc,
    const float* __restrict__ slopbs, const float* __restrict__ slop0s, const float* __restrict__ slopts,
    const float* __restrict__ slopbl, const float* __restrict__ slop0l, const float* __restrict__ sloptl,
    const float* __restrict__ dspbss, const float* __restrict__ dsp0ss, const float* __restrict__ dsptss,
    const float* __restrict__ dspbsl, const float* __restrict__ dsp0sl, const float* __restrict__ dsptsl,
    const int* __restrict__ ltopc, const int* __restrict__ lbc, const int* __restrict__ l0c,
    const float* __restrict__ pk, const float* __restrict__ qk, const float* __restrict__ tk,
    const float* __restrict__ dprs, const float* __restrict__ el, const float* __restrict__ rxnerk,
    float* __restrict__ trefk, float* __restrict__ qrefk,
    float* __restrict__ psk, float* __restrict__ rxnersk, float* __restrict__ thsk)
{
    COLUMN_IJ(IM, JM)
    (void)LM;

    const float smv    = P2(sm);
    const float sm1v   = P2(sm1);
    const float pbot   = P2(pbotfc);
    const float efiv   = P2(efi);
    const float depwlv = P2(depwl);
    const float depthv = P2(depth);
    const float pk0    = P2(pk0c);
    const float pkb    = P2(pkbc);
    const float pkt    = P2(pktc);
    const int   ltop   = ltopc[IDX2(i0, j0, IM)];
    const int   lb     = lbc[IDX2(i0, j0, IM)];
    const int   l0     = l0c[IDX2(i0, j0, IM)];

    /* 1091-1097: deficit saturation pressures for this column */
    const float dspbk = ((efiv - efimn) * P2(slopbs) + P2(dspbss) * pbot) * smv
                      + ((efiv - efimn) * P2(slopbl) + P2(dspbsl) * pbot) * sm1v;
    const float dsp0k = ((efiv - efimn) * P2(slop0s) + P2(dsp0ss) * pbot) * smv
                      + ((efiv - efimn) * P2(slop0l) + P2(dsp0sl) * pbot) * sm1v;
    const float dsptk = ((efiv - efimn) * P2(slopts) + P2(dsptss) * pbot) * smv
                      + ((efiv - efimn) * P2(sloptl) + P2(dsptsl) * pbot) * sm1v;

    /* 1099-1113: saturation pressure difference and humidity profile on ltop..lb */
    for (int l = ltop; l <= lb; ++l) {
        float dsp;
        if (depwlv >= depthv) {
            if (l < l0) {
                dsp = ((pk0 - P3(pk, l)) * dsptk + (P3(pk, l) - pkt) * dsp0k) / (pk0 - pkt);
            } else {
                dsp = ((pkb - P3(pk, l)) * dsp0k + (P3(pk, l) - pk0) * dspbk) / (pkb - pk0);
            }
        } else {
            dsp = dsp0k;
            if (l < l0) {
                dsp = ((pk0 - P3(pk, l)) * dsptk + (P3(pk, l) - pkt) * dsp0k) / (pk0 - pkt);
            }
        }
        if (P3(pk, l) > pqm) {
            P3(psk, l) = P3(pk, l) + dsp;
            P3(rxnersk, l) = powf(1.e5f / P3(psk, l), cappa);
            P3(thsk, l) = P3(trefk, l) * P3(rxnerk, l);
            P3(qrefk, l) = pq0 / P3(psk, l)
                         * expf(a2 * (P3(thsk, l) - a3 * P3(rxnersk, l))
                                / (P3(thsk, l) - a4 * P3(rxnersk, l)));
        } else {
            P3(qrefk, l) = P3(qk, l);
        }
    }

    /* 1136-1177: enthalpy conservation, two passes */
    for (int iter = 1; iter <= 2; ++iter) {
        float sumde = 0.f;
        float sumdp = 0.f;
        for (int l = ltop; l <= lb; ++l) {
            sumde = ((P3(tk, l) - P3(trefk, l)) * cp + (P3(qk, l) - P3(qrefk, l)) * P3(el, l))
                  * P3(dprs, l) + sumde;
            sumdp = sumdp + P3(dprs, l);
        }
        const float hcorr = sumde / (sumdp - P3(dprs, ltop));
        int lcor = ltop + 1;

        /* find lqm: the largest l in 1..lb with pk(l) <= pqm (1 when none) */
        int lqm = 1;
        for (int l = 1; l <= lb; ++l) {
            if (P3(pk, l) <= pqm) lqm = l;
        }

        /* above lqm correct temperature only */
        if (lcor <= lqm) {
            for (int l = lcor; l <= lqm; ++l) {
                P3(trefk, l) = P3(trefk, l) + hcorr * rcp;
            }
            lcor = lqm + 1;
        }

        /* below lqm correct both temperature and moisture */
        for (int l = lcor; l <= lb; ++l) {
            const float tskl = P3(trefk, l) * P3(rxnerk, l) / P3(rxnersk, l);
            const float tq = tskl - a4;
            const float dhdt = P3(qrefk, l) * a23m4l / (tq * tq) + cp;
            P3(trefk, l) = hcorr / dhdt + P3(trefk, l);
            const float thskl = P3(trefk, l) * P3(rxnerk, l);
            P3(qrefk, l) = pq0 / P3(psk, l)
                         * expf(a2 * (thskl - a3 * P3(rxnersk, l))
                                / (thskl - a4 * P3(rxnersk, l)));
        }
    }
}
