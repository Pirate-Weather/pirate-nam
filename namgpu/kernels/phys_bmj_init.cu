// nam-gpu unit phys-bmj-init: BMJ_INIT look-up tables for the Betts-Miller-Janjic convection
// scheme (module-private tables of module_CU_BMJ, plus cldefi and the spline they call).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module header kept as the notice:
//   module module_cu_bmj  (NMMB, nam.v4.2.11): "the convection drivers and packages",
//   "for bmj convection". Subroutines bmj_init (2158-2425) and spline (after bmj_init).
// Ported to CUDA for nam-gpu from phys/module_CU_BMJ.F90:2158-2425 (nam.v4.2.11, 83df4e13c1),
// with the SPLINE routine of the same module (after bmj_init, end of file section).
//
// Kernels (one launch each, CONTRACT section 2 "per routine section"):
//   phys_bmj_init_sat    coarse saturation-point table, loop 100 (one thread per kth = 1..jtb):
//                        qs0, sqs, ptbl and the _exp module-share copies
//   phys_bmj_init_ttbl   coarse t(p) table, loop 200 (one thread per kp = 1..itb): the0, sthe, ttbl
//   phys_bmj_init_ttblq  fine t(p) table, loop 300 (one thread per kp = 1..itbq): the0q, stheq, ttblq
//   phys_bmj_init_cldefi cldefi = avgefi over the interior it/jt range (skipped when restart)
//
// Each table build is a sequence of Fortran steps per column/row (th, p recurrences, the eps
// monotonicity fix, the spline solve). The recurrences that Fortran runs as running sums
// (th = th + dth, p = p + dp, qsnew(kp) = qsnew(kp-1) + dqs) are replayed term by term in the
// same order, so each table entry sees the same float32 operations as the Fortran.
//
// Index mapping (CONTRACT section 1): Fortran ptbl(itb,jtb) -> CuPy [jtb, itb] with
// IDX2(kp-1, kth-1, ITB); ttbl(jtb,itb) -> [itb, jtb] with IDX2(kth-1, kp-1, JTB);
// ttblq(jtbq,itbq) -> [itbq, jtbq] with IDX2(kth-1, kp-1, JTBQ). 1-D tables are plain arrays.
// Fortran 1-based k is stored at [k-1]. Scalars arrive from the wrapper (nmmb_common constants).
//
// Numerics: float32 throughout (kfpt = single). Literals are the Fortran single literals
// (1.e5, 0.5, .1666667, 6.); FMA contraction on (fast numerics, CONTRACT section 3).
// No host sync, no printf, no device allocation. Local arrays are per-thread scratch.

#include "nmmb_common.cuh"

#define BMJ_ITB   76     // itb   (module parameter)
#define BMJ_JTB   134    // jtb
#define BMJ_ITBQ  152    // itbq
#define BMJ_JTBQ  440    // jtbq

// module_CU_BMJ.F90:29-63 (single precision parameters)
#define BMJ_PL     2500.0f
#define BMJ_PLQ    70000.0f
#define BMJ_PH     105000.0f
#define BMJ_THL    210.0f
#define BMJ_THH    365.0f
#define BMJ_THHQ   325.0f
#define BMJ_ELIWV  2.683e6f       // eliwv (bmj_init parameter list)
#define BMJ_EPS    1.0e-9f        // eps (bmj_init parameter list)
#define BMJ_EFIMN  0.20f          // efimn
#define BMJ_AVGEFI ((BMJ_EFIMN + 1.0f) * 0.5f)   // avgefi = (efimn+1.)*0.5

// ---------------------------------------------------------------------------------------------
// SPLINE (module_CU_BMJ.F90, subroutine spline(jtbx,nold,xold,yold,y2,nnew,xnew,ynew,p,q)).
// Natural cubic spline, Janjic's scalar routine, Fortran statement order kept:
//   forward sweep k = 3..nold-1 (label 100), back substitution k = nold-1..2 (label 200),
//   evaluation k1 = 1..nnew (label 300): search kold, coefficients ak/bk/ck, ynew.
// Fortran label 450/550 reuse of ak/bk/ck for the same kold is not needed: the reused values
// are the ones recomputed from the same k, so the result is identical.
// Out of range (xk >= xold(nold)): ynew = yold(nold), label 600.
// Fortran 1-based index k is xold[k-1], so xold(k+1) is xold[k], p(k-1) is p[k-2], q(k-2) is q[k-3].
// ---------------------------------------------------------------------------------------------
__device__ static void bmj_spline(const int nold,
                                  const float* xold, const float* yold, float* y2,
                                  const int nnew,
                                  const float* xnew, float* ynew,
                                  float* p, float* q)
{
    const int noldm1 = nold - 1;

    float dxl = xold[1] - xold[0];                       // xold(2)-xold(1)
    float dxr = xold[2] - xold[1];                       // xold(3)-xold(2)
    float dydxl = (yold[1] - yold[0]) / dxl;
    float dydxr = (yold[2] - yold[1]) / dxr;
    const float rtdxc = 0.5f / (dxl + dxr);

    p[0] = rtdxc * (6.0f * (dydxr - dydxl) - dxl * y2[0]);   // p(1)
    q[0] = -rtdxc * dxr;                                      // q(1)

    // label 100: k = 3 .. nold-1 (no iterations when nold == 3, label 150)
    for (int k = 3; k < nold; ++k) {
        dxl = dxr;
        dydxl = dydxr;
        dxr = xold[k] - xold[k - 1];                     // xold(k+1)-xold(k)
        dydxr = (yold[k] - yold[k - 1]) / dxr;
        const float dxc = dxl + dxr;
        const float den = 1.0f / (dxl * q[k - 3] + dxc + dxc);   // q(k-2)
        p[k - 2] = den * (6.0f * (dydxr - dydxl) - dxl * p[k - 3]);  // p(k-1) = ... p(k-2)
        q[k - 2] = -den * dxr;                                       // q(k-1)
    }

    // label 200: y2(k) = p(k-1) + q(k-1)*y2(k+1), k = nold-1 down to 2
    for (int k = noldm1; k > 1; --k) {
        y2[k - 1] = p[k - 2] + q[k - 2] * y2[k];
    }

    // label 300: evaluation
    for (int k1 = 1; k1 <= nnew; ++k1) {
        const float xk = xnew[k1 - 1];
        int kold = -1;
        for (int k2 = 2; k2 <= nold; ++k2) {             // first xold(k2) > xk
            if (xold[k2 - 1] > xk) { kold = k2 - 1; break; }
        }
        if (kold < 0) {                                  // label 600 with ynew(k1) = yold(nold)
            ynew[k1 - 1] = yold[nold - 1];
            continue;
        }
        const int k = kold;                              // label 500
        const float y2k = y2[k - 1];                     // y2(k)
        const float y2kp1 = y2[k];                       // y2(k+1)
        const float dx = xold[k] - xold[k - 1];          // xold(k+1)-xold(k)
        const float rdx = 1.0f / dx;
        const float ak = 0.1666667f * rdx * (y2kp1 - y2k);
        const float bk = 0.5f * y2k;
        const float ck = rdx * (yold[k] - yold[k - 1]) - 0.1666667f * dx * (y2kp1 + y2k + y2k);
        // label 550
        const float x = xk - xold[k - 1];                // xk - xold(k)
        const float xsq = x * x;
        ynew[k1 - 1] = ak * xsq * x + bk * xsq + ck * x + yold[k - 1];
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel 1: coarse saturation-point table (bmj_init loop 100).
// One thread per kth (Fortran kth = 1..jtb). Outputs: qs0(kth), sqs(kth) and their _exp copies
// (qs0_exp, sqs_exp: module-share arrays for module_bl_myjpbl), ptbl(kp,kth) and ptbl_exp.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_bmj_init_sat(
    float* __restrict__ qs0,
    float* __restrict__ sqs,
    float* __restrict__ qs0_exp,
    float* __restrict__ sqs_exp,
    float* __restrict__ ptbl,        // [JTB, ITB]  Fortran ptbl(itb,jtb)
    float* __restrict__ ptbl_exp,    // [JTB, ITB]
    const float a2, const float a3, const float a4, const float cappa, const float pq0)
{
    const int kth = blockIdx.x * blockDim.x + threadIdx.x + 1;   // Fortran kth
    if (kth > BMJ_JTB) return;

    const float dth = (BMJ_THH - BMJ_THL) / (float)(BMJ_JTB - 1);   // (thh-thl)/float(kthm-1)
    const float dp = (BMJ_PH - BMJ_PL) / (float)(BMJ_ITB - 1);      // (ph-pl)/float(kpm-1)

    // th = thl - dth; then th = th + dth for each kth
    float th = BMJ_THL - dth;
    for (int s = 0; s < kth; ++s) th = th + dth;

    float qsold[BMJ_ITB], pold[BMJ_ITB], y2p[BMJ_ITB];
    float qsnew[BMJ_ITB], pnew[BMJ_ITB], app[BMJ_ITB], aqp[BMJ_ITB];

    float p = BMJ_PL - dp;                                   // p=pl-dp
    for (int kp = 1; kp <= BMJ_ITB; ++kp) {
        p = p + dp;
        const float rxner = powf(100000.0f / p, cappa);      // (100000./p)**cappa
        const float denom = th - a4 * rxner;
        if (denom > BMJ_EPS) {
            qsold[kp - 1] = (pq0 / p) * expf(a2 * (th - a3 * rxner) / denom);
        } else {
            qsold[kp - 1] = 0.0f;
        }
        pold[kp - 1] = p;
    }

    const float qs0k = qsold[0];                             // qs0k=qsold(1)
    const float sqsk = qsold[BMJ_ITB - 1] - qsold[0];        // sqsk=qsold(kpm)-qsold(1)
    qsold[0] = 0.0f;
    qsold[BMJ_ITB - 1] = 1.0f;

    for (int kp = 2; kp <= BMJ_ITB - 1; ++kp) {
        qsold[kp - 1] = (qsold[kp - 1] - qs0k) / sqsk;
        if ((qsold[kp - 1] - qsold[kp - 2]) < BMJ_EPS) qsold[kp - 1] = qsold[kp - 2] + BMJ_EPS;
    }

    qs0[kth - 1] = qs0k;
    qs0_exp[kth - 1] = qs0k;
    sqs[kth - 1] = sqsk;
    sqs_exp[kth - 1] = sqsk;

    // qsnew: uniform grid in qs
    qsnew[0] = 0.0f;
    qsnew[BMJ_ITB - 1] = 1.0f;
    const float dqs = 1.0f / (float)(BMJ_ITB - 1);          // dqs=1./float(kpm-1)
    for (int kp = 2; kp <= BMJ_ITB - 1; ++kp) {
        qsnew[kp - 1] = qsnew[kp - 2] + dqs;
    }

    y2p[0] = 0.0f;                                           // y2p(1) = 0
    y2p[BMJ_ITB - 1] = 0.0f;                                 // y2p(kpm) = 0

    // call spline(jtb,kpm,qsold,pold,y2p,kpm,qsnew,pnew,app,aqp)
    bmj_spline(BMJ_ITB, qsold, pold, y2p, BMJ_ITB, qsnew, pnew, app, aqp);

    for (int kp = 1; kp <= BMJ_ITB; ++kp) {
        ptbl[IDX2(kp - 1, kth - 1, BMJ_ITB)] = pnew[kp - 1];      // ptbl(kp,kth)=pnew(kp)
        ptbl_exp[IDX2(kp - 1, kth - 1, BMJ_ITB)] = pnew[kp - 1];  // ptbl_exp(kp,kth)=pnew(kp)
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel 2: coarse t(p) table from constant the (bmj_init loop 200).
// One thread per kp (Fortran kp = 1..itb). Outputs: the0(kp), sthe(kp), ttbl(kth,kp).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_bmj_init_ttbl(
    float* __restrict__ ttbl,        // [ITB, JTB]  Fortran ttbl(jtb,itb)
    float* __restrict__ the0,        // [ITB]
    float* __restrict__ sthe,        // [ITB]
    const float a2, const float a3, const float a4, const float cappa, const float cp,
    const float pq0)
{
    const int kp = blockIdx.x * blockDim.x + threadIdx.x + 1;    // Fortran kp
    if (kp > BMJ_ITB) return;

    const float dth = (BMJ_THH - BMJ_THL) / (float)(BMJ_JTB - 1);
    const float dp = (BMJ_PH - BMJ_PL) / (float)(BMJ_ITB - 1);

    // p = pl - dp; then p = p + dp for each kp
    float p = BMJ_PL - dp;
    for (int s = 0; s < kp; ++s) p = p + dp;

    float told[BMJ_JTB], theold[BMJ_JTB], thenew[BMJ_JTB];
    float tnew[BMJ_JTB], y2t[BMJ_JTB], apt[BMJ_JTB], aqt[BMJ_JTB];

    float th = BMJ_THL - dth;                                // th=thl-dth
    for (int kth = 1; kth <= BMJ_JTB; ++kth) {
        th = th + dth;
        const float rxner = powf(1.0e5f / p, cappa);         // (1.e5/p)**cappa
        const float denom = th - a4 * rxner;
        float qs;
        if (denom > BMJ_EPS) {
            qs = (pq0 / p) * expf(a2 * (th - a3 * rxner) / denom);
        } else {
            qs = 0.0f;
        }
        told[kth - 1] = th / rxner;
        theold[kth - 1] = th * expf(BMJ_ELIWV * qs / (cp * told[kth - 1]));
    }

    const float the0k = theold[0];
    const float sthek = theold[BMJ_JTB - 1] - theold[0];
    theold[0] = 0.0f;
    theold[BMJ_JTB - 1] = 1.0f;

    for (int kth = 2; kth <= BMJ_JTB - 1; ++kth) {
        theold[kth - 1] = (theold[kth - 1] - the0k) / sthek;
        if ((theold[kth - 1] - theold[kth - 2]) < BMJ_EPS) theold[kth - 1] = theold[kth - 2] + BMJ_EPS;
    }

    the0[kp - 1] = the0k;                                    // the0(kp)=the0k
    sthe[kp - 1] = sthek;                                    // sthe(kp)=sthek

    thenew[0] = 0.0f;
    thenew[BMJ_JTB - 1] = 1.0f;
    const float dthe = 1.0f / (float)(BMJ_JTB - 1);         // dthe=1./float(kthm-1)
    for (int kth = 2; kth <= BMJ_JTB - 1; ++kth) {
        thenew[kth - 1] = thenew[kth - 2] + dthe;
    }

    y2t[0] = 0.0f;
    y2t[BMJ_JTB - 1] = 0.0f;

    // call spline(jtb,kthm,theold,told,y2t,kthm,thenew,tnew,apt,aqt)
    bmj_spline(BMJ_JTB, theold, told, y2t, BMJ_JTB, thenew, tnew, apt, aqt);

    for (int kth = 1; kth <= BMJ_JTB; ++kth) {
        ttbl[IDX2(kth - 1, kp - 1, BMJ_JTB)] = tnew[kth - 1];   // ttbl(kth,kp)=tnew(kth)
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel 3: fine t(p) table from constant the (bmj_init loop 300).
// One thread per kp (Fortran kp = 1..itbq). Outputs: the0q(kp), stheq(kp), ttblq(kth,kp).
// Same arithmetic as kernel 2 with the fine grids (plq, thhq, itbq, jtbq).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_bmj_init_ttblq(
    float* __restrict__ ttblq,       // [ITBQ, JTBQ]  Fortran ttblq(jtbq,itbq)
    float* __restrict__ the0q,       // [ITBQ]
    float* __restrict__ stheq,       // [ITBQ]
    const float a2, const float a3, const float a4, const float cappa, const float cp,
    const float pq0)
{
    const int kp = blockIdx.x * blockDim.x + threadIdx.x + 1;    // Fortran kp
    if (kp > BMJ_ITBQ) return;

    const float dth = (BMJ_THHQ - BMJ_THL) / (float)(BMJ_JTBQ - 1);   // dth=(thhq-thl)/float(kthm-1)
    const float dp = (BMJ_PH - BMJ_PLQ) / (float)(BMJ_ITBQ - 1);     // dp=(ph-plq)/float(kpm-1)

    // p = plq - dp; then p = p + dp for each kp
    float p = BMJ_PLQ - dp;
    for (int s = 0; s < kp; ++s) p = p + dp;

    float toldq[BMJ_JTBQ], theoldq[BMJ_JTBQ], thenewq[BMJ_JTBQ];
    float tnewq[BMJ_JTBQ], y2tq[BMJ_JTBQ], aptq[BMJ_JTBQ], aqtq[BMJ_JTBQ];

    float th = BMJ_THL - dth;                                // th=thl-dth
    for (int kth = 1; kth <= BMJ_JTBQ; ++kth) {
        th = th + dth;
        const float rxner = powf(1.0e5f / p, cappa);
        const float denom = th - a4 * rxner;
        float qs;
        if (denom > BMJ_EPS) {
            qs = (pq0 / p) * expf(a2 * (th - a3 * rxner) / denom);
        } else {
            qs = 0.0f;
        }
        toldq[kth - 1] = th / rxner;
        theoldq[kth - 1] = th * expf(BMJ_ELIWV * qs / (cp * toldq[kth - 1]));
    }

    const float the0k = theoldq[0];
    const float sthek = theoldq[BMJ_JTBQ - 1] - theoldq[0];
    theoldq[0] = 0.0f;
    theoldq[BMJ_JTBQ - 1] = 1.0f;

    for (int kth = 2; kth <= BMJ_JTBQ - 1; ++kth) {
        theoldq[kth - 1] = (theoldq[kth - 1] - the0k) / sthek;
        if ((theoldq[kth - 1] - theoldq[kth - 2]) < BMJ_EPS) theoldq[kth - 1] = theoldq[kth - 2] + BMJ_EPS;
    }

    the0q[kp - 1] = the0k;
    stheq[kp - 1] = sthek;

    thenewq[0] = 0.0f;
    thenewq[BMJ_JTBQ - 1] = 1.0f;
    const float dthe = 1.0f / (float)(BMJ_JTBQ - 1);
    for (int kth = 2; kth <= BMJ_JTBQ - 1; ++kth) {
        thenewq[kth - 1] = thenewq[kth - 2] + dthe;
    }

    y2tq[0] = 0.0f;
    y2tq[BMJ_JTBQ - 1] = 0.0f;

    // call spline(jtbq,kthm,theoldq,toldq,y2tq,kthm,thenewq,tnewq,aptq,aqtq)
    bmj_spline(BMJ_JTBQ, theoldq, toldq, y2tq, BMJ_JTBQ, thenewq, tnewq, aptq, aqtq);

    for (int kth = 1; kth <= BMJ_JTBQ; ++kth) {
        ttblq[IDX2(kth - 1, kp - 1, BMJ_JTBQ)] = tnewq[kth - 1];   // ttblq(kth,kp)=tnewq(kth)
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel 4: cldefi (bmj_init, non-restart branch). Fortran loops i = its..itf, j = jts..jtf with
// itf = min(ite, ide-1), jtf = min(jte, jde-1): for the single regional card its = jts = 1,
// so the 0-based range is i0 in [0, IM-2], j0 in [0, JM-2]. Skipped when restart is true.
// cldefi is Fortran cldefi(ims:ime,jms:jme) -> CuPy [JM, IM], IDX2(i0, j0, IM).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_bmj_init_cldefi(
    float* __restrict__ cldefi,
    const int IM, const int JM, const int restart)
{
    if (restart) return;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= IM * JM) return;
    const int i0 = idx % IM;
    const int j0 = idx / IM;
    if (i0 > IM - 2 || j0 > JM - 2) return;                  // itf = IM-1, jtf = JM-1 (1-based)
    cldefi[IDX2(i0, j0, IM)] = BMJ_AVGEFI;
}
