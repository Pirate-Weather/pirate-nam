// nam-gpu unit phys-bmj-col-shallow-adjust: BMJ shallow convection adjustment for every shallow column.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module header kept as the notice: MODULE MODULE_CU_BMJ (Betts-Miller-Janjic convection, NMMB, nam.v4.2.11).
// Ported to CUDA for nam-gpu from phys/module_CU_BMJ.F90:1450-2067 (nam.v4.2.11, 83df4e13c1).
//
// Scope: the shallow part of subroutine bmj, from the per-level setup at 1450 to the relaxation at 820 and
// the end of the routine at 2067. It runs only for columns the shallow-top unit marked shallow (Fortran
// 1444: if(.not.shallow) return). The cloud-top raise (1475-1518), the shallow cloud top (1520-1545),
// the old-cloud checks, the new-cloud branch (810), the momentum relaxation and the 820 relaxation are all
// here. Inputs written by earlier BMJ units are named in the wrapper (phys_bmj_col_shallow_adjust.py).
//
// Fortran -> kernel map (bmj line numbers, kernel comments carry the same numbers):
//   1450-1459  pk, tk, qk, trefk, qrefk, qsatk, rxnerk, thvref, el                     -> setup loop
//   1475-1518  cloud-top raise (rhshmax, overshoot)                                    -> raise block
//   1520-1547  shallow cloud top: ltp1, ptop/pbot test, newall-plume jump              -> cloud-top block
//              (depth, 1530, feeds only the debug print and the commented depmin test: not kept)
//   1562-1600  top scaled theta and table index, saturation point (ptbl)               -> top block
//   1600-1700  reference profile, upward transport and humidity-profile sums           -> profile block
//   1704-1741  newupup/plume and dst jumps, old-cloud exits                            -> jump block
//   1748-1804  dqref, humidity profile, too dry/moist tests, dtdp slope test           -> humidity block
//   1811-1828  mmntshal1 relaxation of u and v (old cloud)                             -> mmntshal1 block
//   810-1972   new cloud: ttblex (tref), mean rh and slopes, first guess, enthalpy     -> L_NEW block
//   1974-2030  mmntshal2 mean-momentum relaxation (the u/v profile branch is unreachable: goto 8888)
//   820-2030   dtdt, dqdt relaxation towards the reference profiles                    -> L_RELAX block
// Debug writes (print_diag) are not ported: bmjdrv passes print_diag = .false. for every column.
//
// Index conventions: Fortran level l (1-based, 1 = model top) is kept as a 1-based index in the local
// arrays (index 0 unused) so every Fortran subscript reads the same in this file. Global [LM, JM, IM]
// arrays are read and written through GL(p, l) = p[(l-1)*sl] with sl = JM*IM and the column base p2.
// Tables: PTBL(kp, kth) = ptbl[(kth-1)*ITB + (kp-1)] (bmj_init stores ptbl(kp,kth) at [kth-1, kp-1]);
// ttbl and ttblq are read by bmj_ttblex (phys_bmj_ttblex.cu) with the layouts bmj_init writes.
//
// Fortran reads that are undefined (outside the arrays), guarded rather than read:
//   thvref(l-1) at l = 1, t(ltp1) and q(ltp1) at ltop = 1, trefk(lbot+1) and pk(lbot+1) at lbot = lmh,
//   and lbot < 1 on entry. The kernel leaves the column without a shallow cloud (lbot=0, ltop=LM,
//   ptop=pbot) for ltop < 2 or lbot >= lmh, and writes nothing for lbot < 1. See the wrapper docstring.
//
// Numerics: float32 (kfpt = single). Literals follow the Fortran single-precision constants. Fortran
// int(), aint() and integer assignment truncate toward zero (C cast and truncf). FMA on (CONTRACT section 3).
// No host sync, no printf, no device allocation. Per-column arrays live in thread-local memory.

#include "nmmb_common.cuh"
#include "phys_bmj_ttblex.cu"   // bmj_ttblex, the TTBLEX body (phys-bmj-ttblex), __device__

#define BMJ_LM        60
#define BMJ_ITB       76
#define BMJ_JTB       134
#define BMJ_ITBQ      152
#define BMJ_JTBQ      440

// module_CU_BMJ.F90 parameters used by the shallow adjustment (lines 29-106).
#define BMJ_PONE      2500.0f     // pone
#define BMJ_PNO       1000.0f     // pno
#define BMJ_PSHU      45000.0f    // pshu
#define BMJ_PL        2500.0f     // pl
#define BMJ_PLQ       70000.0f    // plq
#define BMJ_PH        105000.0f   // ph
#define BMJ_THL       210.0f      // thl
#define BMJ_THH       365.0f      // thh
#define BMJ_RHLSC     0.00f       // rhlsc
#define BMJ_RHHSC     1.00f       // rhhsc
#define BMJ_EPSDN     1.05f       // epsdn
#define BMJ_EPSDT     0.0f        // epsdt
#define BMJ_STABS     1.0f        // stabs
#define BMJ_DTSHAL    -1.0f       // dtshal
#define BMJ_WDRY      0.75f       // wdry (the 0.75 line; 0.50 and 0.90 are commented in the source)
#define BMJ_DEFTOP    0.95f       // deftop
// rdp = (itb-1)/(ph-pl), rdpq = (itbq-1)/(ph-plq), rdq = itb-1, rdth = (jtb-1)/(thh-thl),
// rdthe = jtb-1, rdtheq = jtbq-1 (module_CU_BMJ.F90:81-85).
#define BMJ_RDP       (75.0f / (BMJ_PH - BMJ_PL))
#define BMJ_RDPQ      (151.0f / (BMJ_PH - BMJ_PLQ))
#define BMJ_RDQ       75.0f
#define BMJ_RDTH      (133.0f / (BMJ_THH - BMJ_THL))
#define BMJ_RDTHE     133.0f
#define BMJ_RDTHEQ    439.0f

#define PTBL(kp, kth) ptbl[((kth) - 1) * BMJ_ITB + ((kp) - 1)]
#define GL(p, l)      ((p)[(size_t)((l) - 1) * sl])

// Shallow adjustment of one column (Fortran subroutine bmj, lines 1450-2067). Levels 1..lmh.
// pbot, psp, thbt: the column scalars the Fortran holds at line 1450 (written upstream).
// On entry lbot, ltop, ptop hold the shallow-top values; on exit they hold the adjusted values.
// dtdt, dqdt, dudt, dvdt are written for ltop..lbot only (the Fortran leaves the rest untouched).
__device__ void bmj_col_shallow_adjust(
    const int lm, const int lmh,
    const int newall, const int newswap, const int newupup, const int nodeep,
    const int mmntshal1, const int mmntshal2, const int plume,
    const float a2, const float a3, const float a4, const float cappa, const float cp,
    const float p608, const float pq0, const float tiw, const float elwv, const float eliv,
    const float rdtcnvc, const float tauksc, const float uvscals1, const float uvscals2,
    const float pbot, const float psp, const float thbt,
    const float* prsmid, const float* dprs, const float* tin, const float* qin,
    const float* rxner, const float* thescnv, const float* cpe, const float* dtv,
    const float* uin, const float* vin,
    float* dtdt, float* dqdt, float* dudt, float* dvdt, const size_t sl,
    const float* ptbl, const float* qs0, const float* sqs,
    const float* sthe, const float* the0, const float* ttbl,
    const float* stheq, const float* the0q, const float* ttblq,
    int* lbot_io, int* ltop_io, float* ptop_io)
{
    // Per-column arrays, 1-based Fortran level l at index l.
    float pk[BMJ_LM + 2], tk[BMJ_LM + 2], qk[BMJ_LM + 2], trefk[BMJ_LM + 2], qrefk[BMJ_LM + 2];
    float qsatk[BMJ_LM + 2], rxnerk[BMJ_LM + 2], thvref[BMJ_LM + 2], el[BMJ_LM + 2], fpk[BMJ_LM + 2];
    float wcld[BMJ_LM + 2], tref[BMJ_LM + 2], rhk[BMJ_LM + 2], thmak[BMJ_LM + 2], thvmk[BMJ_LM + 2];

    int l;
    int lbot = *lbot_io;
    int ltop = *ltop_io;
    float ptop = *ptop_io;

    // Scalars (Fortran locals of bmj).
    int overshoot = 0;
    int ltsh, ltp1, it, iq, lmid;
    float rhavg, sumdp, tlev2, qsat1, qsat2, rhshmax;
    float dpl, thtpk, tthk, qqk, bqs00k, sqs00k, bqs10k, sqs10k, bqk, sqk, tqk, ppk;
    float part1, part2, part3, ptpk, dpmix, smix, treflo, pklo, pkhi, rxnerlo, rxnerhi;
    float sumdt, rdpsum, tcorr, trfkl, fptk, dpkl, psum, qsum, rtbar, otsum, potsum, qotsum;
    float dst, dsq, rotsum, dstq, den, dqref, qrftp, qrfkl, tnew, qnew, dtdp, facuv;
    float ubar, vbar, sumdq, qcorr, tkl, diftl, difql, dpot, dentpy;
    float sumrh, a11, a12, b1qsat, b1thvm, b1thma, b1rh, b2qsat, b2thvm, b2thma, b2rh;
    float rhmean, rden, a21, a22, aqs, avm, ama, arh, bqs, bvm, bma, brh, adef, fk, rhref;

    // lbot < 1 is not a shallow column (shallow-top); nothing is read or written.
    if (lbot < 1) return;

    // ---- 1450-1459: per-level setup ----
    for (l = 1; l <= lmh; ++l) {
        pk[l] = GL(prsmid, l);
        tk[l] = GL(tin, l);
        qk[l] = GL(qin, l);
        trefk[l] = tk[l];
        qrefk[l] = qk[l];
        qsatk[l] = pq0 / pk[l] * expf(a2 * (tk[l] - a3) / (tk[l] - a4));
        rxnerk[l] = GL(rxner, l);
        thvref[l] = tk[l] * rxnerk[l] * (qk[l] * p608 + 1.0f);
        if (tk[l] >= tiw) el[l] = elwv; else el[l] = eliv;
    }

    // ---- 1475-1518: raise cloud top if avg rh > rhshmax (dtv > 0 if newswap) ----
    tlev2 = tk[lbot] * powf((pk[lbot] - BMJ_PONE) / pk[lbot], cappa);
    qsat1 = pq0 / pk[lbot] * expf(a2 * (tk[lbot] - a3) / (tk[lbot] - a4));
    qsat2 = pq0 / (pk[lbot] - BMJ_PONE) * expf(a2 * (tlev2 - a3) / (tlev2 - a4));
    rhshmax = qsat2 / qsat1;
    sumdp = 0.0f;
    rhavg = 0.0f;
    for (l = lbot; l >= ltop; --l) {
        dpl = GL(dprs, l);
        rhavg = rhavg + dpl * qk[l] / qsatk[l];
        sumdp = sumdp + dpl;
    }

    if (rhavg / sumdp > rhshmax) {
        ltsh = ltop;
        // overshoot: allow cloud top to be 1 level above neutral/positive buoyancy (apr21, jun04)
        if (newswap) overshoot = 1;
        for (l = ltop - 1; l >= 1; --l) {
            dpl = GL(dprs, l);
            rhavg = rhavg + dpl * qk[l] / qsatk[l];
            sumdp = sumdp + dpl;
            if (!newswap) {
                if (GL(cpe, l) > 0.0f) {
                    ltsh = l;
                } else {
                    break;
                }
            } else {
                if (GL(dtv, l) <= 0.0f) {                 // More strict positive buoyancy criterion
                    if (GL(cpe, l) <= 0.0f) break;
                    if (overshoot) {
                        overshoot = 0;
                    } else {
                        break;
                    }
                }
                ltsh = l;
            }
            if (rhavg / sumdp <= rhshmax) break;
            if (pk[l] <= BMJ_PSHU) break;
        }
        ltop = ltsh;
    }

    // ---- 1520-1547: shallow cloud top, quick exits ----
    ltp1 = ltop - 1;
    ptpk = ptop;
    if (ptop > pbot - BMJ_PNO || ltop > lbot - 2) {
        goto L_RET;
    }
    // 1558: new cloud at plume points
    if (newall && plume) goto L_NEW;

    // Guards for undefined Fortran reads: t(ltp1), q(ltp1) need ltp1 >= 1; trefk(lbot+1), pk(lbot+1),
    // rxnerk(lbot+1) need lbot+1 <= lmh (lbot = lmh would read past the top of the array).
    if (ltp1 < 1 || lbot + 1 > lmh) {
        goto L_RET;
    }

    // ---- 1562-1600: scaled potential temperature and table index at top ----
    thtpk = tk[ltp1] * rxnerk[ltp1];
    tthk = (thtpk - BMJ_THL) * BMJ_RDTH;
    qqk = tthk - truncf(tthk);
    it = (int)tthk + 1;
    if (it < 1) {
        it = 1;
        qqk = 0.0f;
    }
    if (it >= BMJ_JTB) {
        it = BMJ_JTB - 1;
        qqk = 0.0f;
    }

    // base and scaling factor for spec. humidity at top
    bqs00k = qs0[it - 1];
    sqs00k = sqs[it - 1];
    bqs10k = qs0[it];
    sqs10k = sqs[it];
    bqk = (bqs10k - bqs00k) * qqk + bqs00k;
    sqk = (sqs10k - sqs00k) * qqk + sqs00k;

    // scaling spec. humidity & table index at top (Fortran tqk uses q(ltp1))
    tqk = (qk[ltp1] - bqk) / sqk * BMJ_RDQ;
    ppk = tqk - truncf(tqk);
    iq = (int)tqk + 1;
    if (iq < 1) {
        iq = 1;
        ppk = 0.0f;
    }
    if (iq >= BMJ_ITB) {
        iq = BMJ_ITB - 1;
        ppk = 0.0f;
    }

    // cloud top saturation point pressure
    part1 = (PTBL(iq + 1, it) - PTBL(iq, it)) * ppk;
    part2 = (PTBL(iq, it + 1) - PTBL(iq, it)) * qqk;
    part3 = (PTBL(iq, it) - PTBL(iq + 1, it)
             - PTBL(iq, it + 1) + PTBL(iq + 1, it + 1)) * ppk * qqk;
    ptpk = PTBL(iq, it) + part1 + part2 + part3;

    dpmix = ptpk - psp;
    if (fabsf(dpmix) < 3000.0f) dpmix = -3000.0f;

    // temperature profile slope
    smix = (thtpk - thbt) / dpmix * BMJ_STABS;

    treflo = trefk[lbot + 1];
    pklo = pk[lbot + 1];
    pkhi = pk[lbot];
    rxnerlo = rxnerk[lbot + 1];
    rxnerhi = rxnerk[lbot];

    lmid = (int)(0.5f * (float)(lbot + ltop));

    for (l = lbot; l >= ltop; --l) {
        treflo = ((pkhi - pklo) * smix + treflo * rxnerlo) / rxnerhi;
        trefk[l] = treflo;
        if (l <= lmid) trefk[l] = fmaxf(trefk[l], tk[l] + BMJ_DTSHAL);
        rxnerlo = rxnerhi;
        pklo = pkhi;
        rxnerhi = rxnerk[l - 1];
        pkhi = pk[l - 1];
    }

    // temperature reference profile correction
    sumdt = 0.0f;
    sumdp = 0.0f;
    for (l = ltop; l <= lbot; ++l) {
        dpl = GL(dprs, l);
        sumdt = (tk[l] - trefk[l]) * dpl + sumdt;
        sumdp = sumdp + dpl;
    }
    rdpsum = 1.0f / sumdp;
    fpk[lbot] = trefk[lbot];
    tcorr = sumdt * rdpsum;
    for (l = ltop; l <= lbot; ++l) {
        trfkl = trefk[l] + tcorr;
        trefk[l] = trfkl;
        fpk[l] = trfkl;
    }

    // humidity profile equations
    psum = 0.0f;
    qsum = 0.0f;
    potsum = 0.0f;
    qotsum = 0.0f;
    otsum = 0.0f;
    dst = 0.0f;
    fptk = fpk[ltop];
    for (l = ltop; l <= lbot; ++l) {
        dpl = GL(dprs, l);
        dpkl = fpk[l] - fptk;
        psum = dpkl * dpl + psum;
        qsum = qk[l] * dpl + qsum;
        rtbar = 2.0f / (trefk[l] + tk[l]);
        otsum = dpl * rtbar + otsum;
        potsum = dpkl * rtbar * dpl + potsum;
        qotsum = qk[l] * rtbar * dpl + qotsum;
        dst = (trefk[l] - tk[l]) * rtbar * dpl / el[l] + dst;
    }
    psum = psum * rdpsum;
    qsum = qsum * rdpsum;
    rotsum = 1.0f / otsum;
    potsum = potsum * rotsum;
    qotsum = qotsum * rotsum;
    dst = dst * rotsum * cp;

    // if upward transport of temperature go to new cloud (newupup and plume)
    if (newupup && dst > 0.0f && plume) goto L_NEW;

    // otherwise old cloud: dst > 0 is the upward-transport exit
    if (dst > 0.0f) {
        if (newswap) goto L_NEW;
        goto L_RET;
    }

    // ensure positive entropy change
    dstq = dst * BMJ_EPSDN;

    // check for isothermal atmosphere
    den = potsum - psum;
    if (-den / psum < 5.0e-5f) {
        if (newswap) goto L_NEW;
        goto L_RET;
    } else {
        dqref = (qotsum - dstq - qsum) / den;
    }

    // humidity does not increase with height
    if (dqref < 0.0f) {
        if (newswap) goto L_NEW;
        goto L_RET;
    }

    // humidity at the cloud top
    qrftp = qsum - dqref * psum;

    // humidity profile (too dry and too moist clouds not allowed)
    for (l = ltop; l <= lbot; ++l) {
        qrfkl = (fpk[l] - fptk) * dqref + qrftp;
        tnew = (trefk[l] - tk[l]) * tauksc + tk[l];
        qsatk[l] = pq0 / pk[l] * expf(a2 * (tnew - a3) / (tnew - a4));
        qnew = (qrfkl - qk[l]) * tauksc + qk[l];
        if (qnew < qsatk[l] * BMJ_RHLSC) {
            if (newswap) goto L_NEW;
            goto L_RET;
        }
        if (qnew > qsatk[l] * BMJ_RHHSC) {
            if (newswap) goto L_NEW;
            goto L_RET;
        }
        thvref[l] = trefk[l] * rxnerk[l] * (qrfkl * p608 + 1.0f);
        qrefk[l] = qrfkl;
    }

    // eliminate impossible slopes (betts, dtheta/dq); l-1 >= ltop-1 >= 1 by the ltp1 guard
    for (l = ltop; l <= lbot; ++l) {
        dtdp = (thvref[l - 1] - thvref[l]) / (GL(prsmid, l) - GL(prsmid, l - 1));
        if (dtdp < BMJ_EPSDT) {
            if (newswap) goto L_NEW;
            goto L_RET;
        }
    }

    // relaxation to reference profiles, old cloud: mmntshal1 (1811-1828)
    if (mmntshal1) {
        facuv = tauksc * rdtcnvc * uvscals1;
        ubar = 0.0f;
        vbar = 0.0f;
        sumdp = 0.0f;
        for (l = ltop; l <= lbot; ++l) {
            dpl = GL(dprs, l);
            ubar = GL(uin, l) * dpl + ubar;
            vbar = GL(vin, l) * dpl + vbar;
            sumdp = dpl + sumdp;
        }
        rdpsum = 1.0f / sumdp;
        ubar = ubar * rdpsum;
        vbar = vbar * rdpsum;
        for (l = ltop; l <= lbot; ++l) {
            GL(dudt, l) = (ubar - GL(uin, l)) * facuv;
            GL(dvdt, l) = (vbar - GL(vin, l)) * facuv;
        }
    }
    goto L_RELAX;

    // ---- 810: new cloud starts here ----
L_NEW:
    for (l = 1; l <= lmh; ++l) {
        wcld[l] = 0.0f;
        rhk[l] = qk[l] / qsatk[l];
        thvmk[l] = tk[l] * rxnerk[l];
        // updraft temperature along the moist adiabat (ttblex, low or moisture table by pressure)
        if (pk[l] < BMJ_PLQ) {
            tref[l] = bmj_ttblex(BMJ_ITB, BMJ_JTB, BMJ_PL, pk[l], BMJ_RDP, BMJ_RDTHE,
                                 GL(thescnv, l), sthe, the0, ttbl);
        } else {
            tref[l] = bmj_ttblex(BMJ_ITBQ, BMJ_JTBQ, BMJ_PLQ, pk[l], BMJ_RDPQ, BMJ_RDTHEQ,
                                 GL(thescnv, l), stheq, the0q, ttblq);
        }
        thmak[l] = tref[l] * rxnerk[l];
    }

    // mean rh and slopes within cloud
    sumdp = 0.0f;
    sumrh = 0.0f;
    a11 = 0.0f;
    a12 = 0.0f;
    b1qsat = 0.0f;
    b2qsat = 0.0f;
    b1thvm = 0.0f;
    b1thma = 0.0f;
    b2thvm = 0.0f;
    b2thma = 0.0f;
    b1rh = 0.0f;
    b2rh = 0.0f;
    for (l = ltop; l <= lbot; ++l) {
        dpl = GL(dprs, l);
        sumdp = dpl + sumdp;
        sumrh = rhk[l] * dpl + sumrh;
        a11 = pk[l] * pk[l] * dpl + a11;
        a12 = pk[l] * dpl + a12;
        b1qsat = qsatk[l] * pk[l] * dpl + b1qsat;
        b1thvm = thvmk[l] * pk[l] * dpl + b1thvm;
        b1thma = thmak[l] * pk[l] * dpl + b1thma;
        b1rh = rhk[l] * pk[l] * dpl + b1rh;
        b2qsat = qsatk[l] * dpl + b2qsat;
        b2thvm = thvmk[l] * dpl + b2thvm;
        b2thma = thmak[l] * dpl + b2thma;
        b2rh = rhk[l] * dpl + b2rh;
    }

    rhmean = sumrh / sumdp;
    // no shallow convection if the cloud is saturated
    if (rhmean > 0.95f) goto L_RET;

    a21 = a12;
    a22 = sumdp;
    rden = 1.0f / (a11 * a22 - a12 * a21);

    aqs = (b1qsat * a22 - a12 * b2qsat) * rden;
    avm = (b1thvm * a22 - a12 * b2thvm) * rden;
    ama = (b1thma * a22 - a12 * b2thma) * rden;
    arh = (b1rh * a22 - a12 * b2rh) * rden;

    bqs = (a11 * b2qsat - b1qsat * a21) * rden;
    bvm = (a11 * b2thvm - b1thvm * a21) * rden;
    bma = (a11 * b2thma - b1thma * a21) * rden;
    brh = (a11 * b2rh - b1rh * a21) * rden;

    // no shallow convection if the cloud moister on top (soft2)
    if (arh < 0.0f) goto L_RET;

    // first guess t and q profiles (soft2 deftop)
    adef = (1.0f - BMJ_DEFTOP) * 2.0f / (pk[lbot] - pk[ltop]);
    for (l = ltop; l <= lbot; ++l) {
        fk = (pk[l] - pk[ltop]) * adef + BMJ_DEFTOP;
        rhref = rhmean * fk;
        wcld[l] = (1.0f - BMJ_WDRY) * rhref / (1.0f - BMJ_WDRY * rhref);
        trefk[l] = ((1.0f - wcld[l]) * (avm * pk[l] + bvm)
                    + wcld[l] * (ama * pk[l] + bma)) / rxnerk[l];
        if (nodeep && fabsf(trefk[l] - tk[l]) > 5.0f) goto L_RET;
        qrefk[l] = rhref * (aqs * pk[l] + bqs);
    }

    // enthalpy conservation
    sumdp = 0.0f;
    sumdt = 0.0f;
    sumdq = 0.0f;
    for (l = ltop; l <= lbot; ++l) {
        dpl = GL(dprs, l);
        sumdp = dpl + sumdp;
        sumdt = (tk[l] - trefk[l]) * dpl + sumdt;
        sumdq = (qk[l] - qrefk[l]) * dpl + sumdq;
    }
    rdpsum = 1.0f / sumdp;
    tcorr = sumdt * rdpsum;
    qcorr = sumdq * rdpsum;
    for (l = ltop; l <= lbot; ++l) {
        trefk[l] = trefk[l] + tcorr;
        qrefk[l] = qrefk[l] + qcorr;
    }

    dsq = 0.0f;
    dst = 0.0f;
    for (l = ltop; l <= lbot; ++l) {
        tkl = tk[l];
        diftl = (trefk[l] - tkl) * tauksc;
        difql = (qrefk[l] - qk[l]) * tauksc;
        dpot = GL(dprs, l) / (tkl + tkl + diftl);
        dst = diftl * dpot + dst;
        dsq = difql * el[l] * dpot + dsq;
    }
    dst = (dst + dst) * cp;
    dsq = dsq + dsq;
    dentpy = dst + dsq;
    if (dentpy < 0.0f) goto L_RET;

    // mean momentum (mmntshal2). The u/v profile branch that follows the goto 8888 is unreachable.
    if (mmntshal2) {
        ubar = 0.0f;
        vbar = 0.0f;
        sumdp = 0.0f;
        for (l = ltop; l <= lbot; ++l) {
            dpl = GL(dprs, l);
            ubar = GL(uin, l) * dpl + ubar;
            vbar = GL(vin, l) * dpl + vbar;
            sumdp = dpl + sumdp;
        }
        rdpsum = 1.0f / sumdp;
        ubar = ubar * rdpsum;
        vbar = vbar * rdpsum;
        facuv = tauksc * rdtcnvc * uvscals2;
        for (l = ltop; l <= lbot; ++l) {
            GL(dudt, l) = (ubar - GL(uin, l)) * facuv;
            GL(dvdt, l) = (vbar - GL(vin, l)) * facuv;
        }
    }

    // ---- 820: relaxation towards reference profiles ----
L_RELAX:
    for (l = ltop; l <= lbot; ++l) {
        GL(dtdt, l) = (trefk[l] - tk[l]) * tauksc * rdtcnvc;
        GL(dqdt, l) = (qrefk[l] - qk[l]) * tauksc * rdtcnvc;
    }
    goto L_DONE;

    // quick exits of the Fortran routine: lbot=0, ltop=lm, ptop=pbot
L_RET:
    lbot = 0;
    ltop = lm;
    ptop = pbot;

L_DONE:
    *lbot_io = lbot;
    *ltop_io = ltop;
    *ptop_io = ptop;
}

// One thread per column (CONTRACT section 3, grid_cols). The [LM, JM, IM] arrays use IDX3; the [JM, IM]
// arrays use IDX2. Only shallow columns (shallow != 0) are adjusted; the others are left as they are.
extern "C" __global__ void phys_bmj_col_shallow_adjust_columns(
    const int IM, const int JM, const int LM, const int lmh,
    const int newall, const int newswap, const int newupup, const int nodeep,
    const int mmntshal1, const int mmntshal2,
    const float a2, const float a3, const float a4, const float cappa, const float cp,
    const float p608, const float pq0, const float tiw, const float elwv, const float eliv,
    const float rdtcnvc, const float tauksc, const float uvscals1, const float uvscals2,
    const int* __restrict__ shallow,        // [JM, IM] (BMJ_FLAG[2])
    const int* __restrict__ plume,          // [JM, IM] (BMJ_FLAG[1])
    const float* __restrict__ pbot,         // [JM, IM]
    const float* __restrict__ psp,          // [JM, IM]
    const float* __restrict__ thbt,         // [JM, IM]
    const float* __restrict__ prsmid,       // [LM, JM, IM]
    const float* __restrict__ dprs,         // [LM, JM, IM]
    const float* __restrict__ tin,          // [LM, JM, IM] (BMJ_TK)
    const float* __restrict__ qin,          // [LM, JM, IM] (BMJ_QK)
    const float* __restrict__ rxner,        // [LM, JM, IM]
    const float* __restrict__ thescnv,      // [LM, JM, IM]
    const float* __restrict__ cpe,          // [LM, JM, IM]
    const float* __restrict__ dtv,          // [LM, JM, IM]
    const float* __restrict__ uin,          // [LM, JM, IM] (U_PHY)
    const float* __restrict__ vin,          // [LM, JM, IM] (V_PHY)
    float* __restrict__ dtdt,               // [LM, JM, IM] inout
    float* __restrict__ dqdt,               // [LM, JM, IM] inout
    float* __restrict__ dudt,               // [LM, JM, IM] inout
    float* __restrict__ dvdt,               // [LM, JM, IM] inout
    int* __restrict__ lbot,                 // [JM, IM] inout
    int* __restrict__ ltop,                 // [JM, IM] inout
    float* __restrict__ ptop,               // [JM, IM] inout
    const float* __restrict__ ptbl,         // [JTB, ITB]
    const float* __restrict__ qs0,          // [JTB]
    const float* __restrict__ sqs,          // [JTB]
    const float* __restrict__ sthe,         // [ITB]
    const float* __restrict__ the0,         // [ITB]
    const float* __restrict__ ttbl,         // [ITB, JTB]
    const float* __restrict__ stheq,        // [ITBQ]
    const float* __restrict__ the0q,        // [ITBQ]
    const float* __restrict__ ttblq)        // [ITBQ, JTBQ]
{
    COLUMN_IJ(IM, JM)
    const size_t p2 = IDX2(i0, j0, IM);
    if (shallow[p2] == 0) return;

    const size_t sl = (size_t)JM * (size_t)IM;     // stride between levels
    int lb = lbot[p2];
    int lt = ltop[p2];
    float pt = ptop[p2];

    bmj_col_shallow_adjust(
        LM, lmh, newall, newswap, newupup, nodeep, mmntshal1, mmntshal2, plume[p2],
        a2, a3, a4, cappa, cp, p608, pq0, tiw, elwv, eliv,
        rdtcnvc, tauksc, uvscals1, uvscals2,
        pbot[p2], psp[p2], thbt[p2],
        prsmid + p2, dprs + p2, tin + p2, qin + p2, rxner + p2, thescnv + p2, cpe + p2, dtv + p2,
        uin + p2, vin + p2,
        dtdt + p2, dqdt + p2, dudt + p2, dvdt + p2, sl,
        ptbl, qs0, sqs, sthe, the0, ttbl, stheq, the0q, ttblq,
        &lb, &lt, &pt);
    lbot[p2] = lb;
    ltop[p2] = lt;
    ptop[p2] = pt;
}
