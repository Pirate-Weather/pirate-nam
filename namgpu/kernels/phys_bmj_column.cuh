// phys_bmj_column.cuh -- SUBROUTINE BMJ for one column (Betts-Miller-Janjic convection, NMMB).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). Module header
// kept as the notice: MODULE MODULE_CU_BMJ (NMMB convection packages, nam.v4.2.11).
// Ported to CUDA for nam-gpu from phys/module_CU_BMJ.F90:399-2067 (nam.v4.2.11, 83df4e13c1), the BMJ
// subroutine called by bmjdrv (phys-bmj-bmjdrv). Integration (Opus integrator, 2026-11-04): one
// device function, a statement-for-statement transliteration of the Fortran, labels and GO TOs kept
// (C++ goto; every local is declared at the top, as in the Fortran, so no jump crosses an
// initialisation). The section units phys-bmj-col-prep ... phys-bmj-col-shallow-adjust were reviewed
// against the same line ranges; their launchable forms stay as they are. ttblex is the reviewed
// device function of unit phys-bmj-ttblex.
//
// Index: every local column array is 1-based (index 0 exists and is 0; the Fortran reads index 0 only
// where ltop == 1 and the value is then not used). Input/output arrays are the bmjdrv column arrays,
// 0-based: X(l) is x[l-1].
// Tables (BMJ_INIT, phys-bmj-init, ctx.tables["bmj_*"]): ptbl Fortran (itb,jtb) stored [JTB, ITB] C
// order, so ptbl(iq,it) = ptbl[(it-1)*ITB + iq-1]; ttbl/ttblq are read by bmj_ttblex.
// Dropped: the print_diag WRITE(6,...) blocks (print_diag is .false. in bmjdrv; kernels do not print).
#pragma once
#include "nmmb_common.cuh"
#include "phys_bmj_ttblex.cu"

#ifndef BMJ_LM
#define BMJ_LM 60
#endif

struct BmjTbl {
    const float *ptbl, *qs0, *sqs, *sthe, *the0, *ttbl, *stheq, *the0q, *ttblq;
};

// module_cu_bmj PARAMETERs (module_CU_BMJ.F90:1-116), kfpt = single.
#define BMJ_EFIFC   5.0f
#define BMJ_EFIMN_C 0.20f
#define BMJ_EFMNTL  0.70f
#define BMJ_EFMNTS  0.70f
#define BMJ_ELIWV   2.683e6f
#define BMJ_ENPLO   98500.f
#define BMJ_ENPUP   95000.f
#define BMJ_EPSDN   1.05f
#define BMJ_EPSDT   0.f
#define BMJ_EPSNTP  .0001f
#define BMJ_EPSPR   1.e-7f
#define BMJ_FUP     (1.f / 200000.f)
#define BMJ_PFRZ    15000.f
#define BMJ_PNO     1000.f
#define BMJ_PONE    2500.f
#define BMJ_PQM     20000.f
#define BMJ_PSH     20000.f
#define BMJ_PSHU    45000.f
#define BMJ_RENDP   (1.f / (BMJ_ENPLO - BMJ_ENPUP))
#define BMJ_RHLSC   0.00f
#define BMJ_RHHSC   1.00f
#define BMJ_ROW     1.e3f
#define BMJ_STABDF  0.90f
#define BMJ_STABDS  0.90f
#define BMJ_STABS   1.0f
#define BMJ_DTSHAL  (-1.0f)
#define BMJ_TREL    2400.f
#define BMJ_DTPTRIGR (-0.0f * BMJ_PONE)
#define BMJ_PL      2500.f
#define BMJ_PLQ     70000.f
#define BMJ_PH      105000.f
#define BMJ_THL     210.f
#define BMJ_THH     365.f
#define BMJ_ITB     76
#define BMJ_JTB     134
#define BMJ_ITBQ    152
#define BMJ_JTBQ    440
#define BMJ_ITREFI_MAX 3
#define BMJ_RDP     ((BMJ_ITB - 1.f) / (BMJ_PH - BMJ_PL))
#define BMJ_RDPQ    ((BMJ_ITBQ - 1.f) / (BMJ_PH - BMJ_PLQ))
#define BMJ_RDQ     ((float)(BMJ_ITB - 1))
#define BMJ_RDTH    ((BMJ_JTB - 1.f) / (BMJ_THH - BMJ_THL))
#define BMJ_RDTHE   (BMJ_JTB - 1.f)
#define BMJ_RDTHEQ  (BMJ_JTBQ - 1.f)
#define BMJ_RSFCP   (1.f / 101300.f)
#define BMJ_AVGEFI  ((BMJ_EFIMN_C + 1.f) * 0.5f)
#define BMJ_STEFI   1.f
#define BMJ_ELEVFC  0.6f
#define BMJ_SLOPST  ((BMJ_STABDF - BMJ_STABDS) / (1.f - BMJ_EFIMN_C))
#define BMJ_SLOPEL  ((1.f - BMJ_EFMNTL) / (1.f - BMJ_EFIMN_C))
#define BMJ_SLOPES  ((1.f - BMJ_EFMNTS) / (1.f - BMJ_EFIMN_C))
#define BMJ_WDRY    0.75f
#define BMJ_DEFTOP  .95f

#define BPTBL(iq, it) tb.ptbl[((it) - 1) * BMJ_ITB + ((iq) - 1)]

// call ttblex(...) with the pressure test of the Fortran (pl table below plq, plq table above).
__device__ __forceinline__ float bmj_ttblex2(const BmjTbl& tb, float p, float thesp)
{
    if (p < BMJ_PLQ)
        return bmj_ttblex(BMJ_ITB, BMJ_JTB, BMJ_PL, p, BMJ_RDP, BMJ_RDTHE, thesp, tb.sthe, tb.the0, tb.ttbl);
    return bmj_ttblex(BMJ_ITBQ, BMJ_JTBQ, BMJ_PLQ, p, BMJ_RDPQ, BMJ_RDTHEQ, thesp, tb.stheq, tb.the0q, tb.ttblq);
}

__device__ void bmj_column(
    int entrain, int newall, int newswap, int newupup, int nodeep, int print_diag,
    int i, int j, int lm, int lmh, int lpbl, int ntsd,
    int* lbot_out, int* ltop_out,
    float a2, float a3, float a4, float cappa, float* cldefi_io, float cp, float dtcnvc,
    float eliv, float elwv, float g, float p608, float pq0, float r_d, float tiw,
    float psfc, float pt, float sm,
    float dspbfl, float dsp0fl, float dsptfl, float dspbfs, float dsp0fs, float dsptfs,
    float dspbsl, float dsp0sl, float dsptsl, float dspbss, float dsp0ss, float dsptss,
    float slopbl, float slop0l, float sloptl, float slopbs, float slop0s, float slopts,
    const float* dprs_, const float* prsmid_, const float* q_, const float* t_,
    const float* exner_, const float* u_, const float* v_,
    float* dqdt_, float* dtdt_, float* dudt_, float* dvdt_, float* pcpcol_io,
    const BmjTbl& tb)
{
    (void)print_diag; (void)i; (void)j; (void)lpbl; (void)ntsd; (void)pt;
    (void)dspbfl; (void)dsp0fl; (void)dsptfl; (void)dspbfs; (void)dsp0fs; (void)dsptfs;
#define dprs(l)   dprs_[(l) - 1]
#define prsmid(l) prsmid_[(l) - 1]
#define q(l)      q_[(l) - 1]
#define t(l)      t_[(l) - 1]
#define exner(l)  exner_[(l) - 1]
#define u(l)      u_[(l) - 1]
#define v(l)      v_[(l) - 1]
#define dqdt(l)   dqdt_[(l) - 1]
#define dtdt(l)   dtdt_[(l) - 1]
#define dudt(l)   dudt_[(l) - 1]
#define dvdt(l)   dvdt_[(l) - 1]
    const int N = BMJ_LM + 2;
    bool deep, mmntdeep, mmntshal1, mmntshal2, plume, shallow, overshoot;
    float dum, dvm, facuv, uvscald, uvscals1, uvscals2, ubar, vbar;
    float rxnerk[N], rxnersk[N], el[N], fpk[N], pk[N], psk[N], qbte[N], qbtk[N], qk[N], qrefk[N],
          qsatk[N], therk[N], thesp[N], thsk[N], thvref[N], tk[N], trefk[N], urefk[N], vrefk[N],
          wcld[N];
    float rxner[N], difq[N], dift[N], thee[N], thes[N], tref[N];
    float cpe[N], cpecnv[N], dtv[N], dtvcnv[N], thescnv[N];
    float rhk[N], thmak[N], thvmk[N];
    int iq, iqtb, iswap, it, iter, itrefi, ittb, ittbk, kb, l, l0, l0m1, lb, lbm1, lcor, lqm, ltp1,
        ltsh, lbotcnv, ltopcnv, lmid;
    float avrgt, avrgtl, bq, bqk, bqs00k, bqs10k, cup, den, dentpy, depmin, depth, depwl, dhdt,
          difql, diftl, dp, dpkl, dplo, dpmix, dpot, dpup, dqref, drheat, dsp, dsp0, dsp0k, dspb,
          dspbk, dspt, dsptk, dsq, dst, dstq, dthem, dtdp, efi, fefi, ffup, fprs, fptk, hcorr, otsum,
          p, p00k, p01k, p10k, p11k, part1, part2, part3, pbot, pbotfc, pk0, pkb, pkl, pkt, pklo, pkhi,
          pelevfc, pbtmx, plo, potsum, pp1, ppk, preck, psp, psum, ptop, ptpk, pup, qbt, qkl, qnew,
          qotsum, qq1, qqk, qrfkl, qrftp, qsp, qsum, qup, rdp0t, rdpsum, rdtcnvc, rotsum, rtbar, rhavg,
          rxnerp, rxnerlo, rxnerhi, rxners, sm1, smix, sq, sqk, sqs00k, sqs10k, stabdl, sumde, sumdp,
          sumdt, tauk, tauksc, tcorr, thbt, therkx, therky, thskl, thtpk, tkl, tlo, tnew, tq, tqk,
          treflo, trfkl, trmlo, trmup, tskl, tsp, tth, tthk, tup, capecnv, pspcnv, thbtcnv, capetrigr,
          cape, tlev2, qsat1, qsat2, rhshmax, rh_deep;
    float a23m4l, cprlg, elocp, rcp, qwat, a11, a12, a21, a22, ama, aqs, arh, au, av, avm, b1qsat,
          b1rh, b1thma, b1thvm, b1u, b1v, b2qsat, b2rh, b2thma, b2thvm, b2u, b2v, bma, bqs, brh, bu,
          bvm, bv, qcorr, rden, rhmean, rhref, sumdq, sumdu, sumdv, sumrh, ucorr, vcorr, adef, fk;
    int lbot, ltop;
    float cldefi = *cldefi_io;
    float pcpcol = *pcpcol_io;

    for (int k = 0; k < N; ++k) {
        rxnerk[k] = 0.f; rxnersk[k] = 0.f; el[k] = 0.f; fpk[k] = 0.f; pk[k] = 0.f; psk[k] = 0.f;
        qbte[k] = 0.f; qbtk[k] = 0.f; qk[k] = 0.f; qrefk[k] = 0.f; qsatk[k] = 1.f; therk[k] = 0.f;
        thesp[k] = 0.f; thsk[k] = 0.f; thvref[k] = 0.f; tk[k] = 0.f; trefk[k] = 0.f; urefk[k] = 0.f;
        vrefk[k] = 0.f; wcld[k] = 0.f; rxner[k] = 0.f; difq[k] = 0.f; dift[k] = 0.f; thee[k] = 0.f;
        thes[k] = 0.f; tref[k] = 0.f; cpe[k] = 0.f; cpecnv[k] = 0.f; dtv[k] = 0.f; dtvcnv[k] = 0.f;
        thescnv[k] = 0.f; rhk[k] = 0.f; thmak[k] = 0.f; thvmk[k] = 0.f;
    }
    overshoot = false; rdp0t = 0.f; dentpy = 0.f; preck = 0.f; efi = cldefi; l0 = 1; pk0 = 0.f;
    pkt = 0.f; sumdp = 0.f; rden = 0.f; a12 = 0.f; a22 = 0.f; a11 = 0.f; a21 = 0.f; ltsh = 1;
    (void)dsp0; (void)dspb; (void)dspt; (void)lcor; (void)dum; (void)dvm;

    cprlg = cp / (BMJ_ROW * g * elwv);
    elocp = BMJ_ELIWV / cp;
    rcp = 1.f / cp;
    a23m4l = a2 * (a3 - a4) * elwv;
    uvscald = 0.01f;
    uvscals1 = 0.01f;
    uvscals2 = 0.01f;
    rdtcnvc = 1.f / dtcnvc;
    depmin = BMJ_PSH * psfc * BMJ_RSFCP;
    deep = false;
    plume = false;
    shallow = false;
    rh_deep = nodeep ? 0.90f : 0.75f;
    mmntdeep = false;
    mmntshal1 = true;
    mmntshal2 = true;
    tauk = dtcnvc / (BMJ_TREL * 01.0f);
    tauksc = dtcnvc / (BMJ_TREL * 01.0f);
    iswap = 0;
    cup = 0.f;
    dsp0 = 0.f;
    dspb = 0.f;
    dspt = 0.f;
    pcpcol = 0.f;
    tref[1] = t(1);
    for (l = 1; l <= lmh; ++l) {
        tk[l] = t(l);
        qk[l] = q(l);
        rxner[l] = 1.f / exner(l);
        cpecnv[l] = 0.f;
        dtvcnv[l] = 0.f;
        thes[l] = 0.f;
        thesp[l] = 0.f;
        thescnv[l] = 0.f;
        dudt(l) = 0.f;
        dvdt(l) = 0.f;
        qsatk[l] = pq0 / prsmid(l) * expf(a2 * (tk[l] - a3) / (tk[l] - a4));
        rhk[l] = qk[l] / qsatk[l];
    }
    pelevfc = prsmid(lmh) * BMJ_ELEVFC;
    pbtmx = prsmid(lmh) - BMJ_PONE;
    capecnv = 0.f;
    pspcnv = 0.f;
    thbtcnv = 0.f;
    lbot = lmh;
    ltop = lmh;
    lbotcnv = lbot;
    ltopcnv = lbot;

    // prep_loop
    for (kb = lmh; kb >= 1; --kb) {
        if (prsmid(kb) < pelevfc && !entrain) break;
        qbt = q(kb);
        thbt = t(kb) * rxner[kb];
        tth = (thbt - BMJ_THL) * BMJ_RDTH;
        qq1 = tth - truncf(tth);
        ittb = (int)tth + 1;
        if (ittb < 1) { ittb = 1; qq1 = 0.f; }
        else if (ittb >= BMJ_JTB) { ittb = BMJ_JTB - 1; qq1 = 0.f; }
        ittbk = ittb;
        bqs00k = tb.qs0[ittbk - 1];
        sqs00k = tb.sqs[ittbk - 1];
        bqs10k = tb.qs0[ittbk];
        sqs10k = tb.sqs[ittbk];
        bq = (bqs10k - bqs00k) * qq1 + bqs00k;
        sq = (sqs10k - sqs00k) * qq1 + sqs00k;
        tq = (qbt - bq) / sq * BMJ_RDQ;
        pp1 = tq - truncf(tq);
        iqtb = (int)tq + 1;
        if (iqtb < 1) { iqtb = 1; pp1 = 0.f; }
        else if (iqtb >= BMJ_ITB) { iqtb = BMJ_ITB - 1; pp1 = 0.f; }
        iq = iqtb;
        it = ittb;
        p00k = BPTBL(iq, it);
        p10k = BPTBL(iq + 1, it);
        p01k = BPTBL(iq, it + 1);
        p11k = BPTBL(iq + 1, it + 1);
        psp = p00k + (p10k - p00k) * pp1 + (p01k - p00k) * qq1
            + (p00k - p10k - p01k + p11k) * pp1 * qq1;
        rxners = powf(1.e5f / psp, cappa);
        thesp[kb] = thbt * expf(elocp * qbt * rxners / thbt);
        psk[kb] = psp;
        rxnersk[kb] = rxners;
    }

    // max_buoy_loop
    for (kb = lmh; kb >= 1; --kb) {
        if (prsmid(kb) < pelevfc) break;
        qbt = q(kb);
        thbt = t(kb) * rxner[kb];
        psp = psk[kb];
        rxners = rxnersk[kb];
        for (l = 1; l <= lmh - 1; ++l) {
            p = prsmid(l);
            if (p < psp && p >= BMJ_PQM) lbot = l + 1;
        }
        pbot = prsmid(lbot);
        if (pbot >= pbtmx || lbot >= lmh) {
            for (l = 1; l <= lmh - 1; ++l) {
                p = prsmid(l);
                if (p < pbtmx) lbot = l;
            }
            pbot = prsmid(lbot);
        }
        ltop = lbot;
        ptop = pbot;
        if (entrain) {
            for (l = lmh; l >= kb; --l) { thes[l] = thesp[kb]; qbtk[l] = qk[kb]; }
            for (l = kb; l >= 1; --l) { thes[l] = thesp[l]; qbtk[l] = qk[l]; }
            for (l = 1; l <= lmh; ++l) { thee[l] = thes[l]; qbte[l] = qbtk[l]; }
            if (sm > 0.5f) fefi = (cldefi - BMJ_EFIMN_C) * BMJ_SLOPES + BMJ_EFMNTS;
            else fefi = (cldefi - BMJ_EFIMN_C) * BMJ_SLOPEL + BMJ_EFMNTL;
            ffup = BMJ_FUP / (fefi * fefi);
            if (pbot > BMJ_ENPLO) fprs = 1.f;
            else if (pbot > BMJ_ENPUP) fprs = (pbot - BMJ_ENPUP) * BMJ_RENDP;
            else fprs = 0.f;
            ffup = ffup * fprs * fprs * 0.5f;
            dpup = dprs(kb);
            for (l = kb - 1; l >= 1; --l) {
                dplo = dpup;
                dpup = dprs(l);
                thes[l] = ((-ffup * dplo + 1.f) * thes[l + 1]
                           + (thee[l] * dpup + thee[l + 1] * dplo) * ffup) / (ffup * dpup + 1.f);
                qbtk[l] = ((-ffup * dplo + 1.f) * qbtk[l + 1]
                           + (qbte[l] * dpup + qbte[l + 1] * dplo) * ffup) / (ffup * dpup + 1.f);
            }
        } else {
            for (l = lmh; l >= 1; --l) { thes[l] = thesp[kb]; qbtk[l] = qk[kb]; }
        }
        for (l = 1; l <= lmh; ++l) { cpe[l] = 0.f; dtv[l] = 0.f; }
        if (kb < lbot) goto L170;
        l = kb;
        plo = prsmid(l);
        tlo = thbt * exner(l);
        trmlo = ((qbt * p608 + 1.f) * tlo - (q(l) * p608 + 1.f) * t(l)) * r_d / plo;
        capetrigr = BMJ_DTPTRIGR / t(lbot);
        if (kb - lbot >= 1) {
            for (l = kb - 1; l >= lbot; --l) {
                pup = prsmid(l);
                tup = thbt * exner(l);
                dp = plo - pup;
                trmup = ((qbt * p608 + 1.f) * tup - (q(l) * p608 + 1.f) * t(l)) * r_d / pup;
                dtv[l] = (trmlo + trmup) * dp * 0.5f;
                cpe[l] = dtv[l] + cpe[l + 1];
                if (cpe[l] < capetrigr) goto L170;
                plo = pup;
                trmlo = trmup;
            }
        }
        l = lbot - 1;
        pup = psp;
        tup = thbt / rxners;
        tsp = ((t(l + 1) - t(l)) / (plo - prsmid(l))) * (pup - pbot) + t(l);
        qsp = ((q(l + 1) - q(l)) / (plo - prsmid(l))) * (pup - pbot) + q(l);
        dp = plo - pup;
        trmup = ((qbt * p608 + 1.f) * tup - (qsp * p608 + 1.f) * tsp) * r_d / pup;
        dtv[l] = (trmlo + trmup) * dp * 0.5f;
        plo = pup;
        trmlo = trmup;
        pup = prsmid(l);
        tup = bmj_ttblex2(tb, pup, thes[l]);
        qup = pq0 / pup * expf(a2 * (tup - a3) / (tup - a4));
        qwat = qbt - qup;
        dp = plo - pup;
        trmup = ((qup * p608 + 1.f - qwat) * tup - (q(l) * p608 + 1.f) * t(l)) * r_d / pup;
        dtv[l] = (trmlo + trmup) * dp * 0.5f + dtv[l];
        cpe[l] = dtv[l] + cpe[l + 1];
        if (cpe[l] < capetrigr) goto L170;
        plo = pup;
        trmlo = trmup;
        for (l = lbot - 2; l >= 1; --l) {
            pup = prsmid(l);
            tup = bmj_ttblex2(tb, pup, thes[l]);
            qup = pq0 / pup * expf(a2 * (tup - a3) / (tup - a4));
            qwat = qbt - qup;
            dp = plo - pup;
            trmup = ((qup * p608 + 1.f - qwat) * tup - (q(l) * p608 + 1.f) * t(l)) * r_d / pup;
            dtv[l] = (trmlo + trmup) * dp * 0.5f;
            cpe[l] = dtv[l] + cpe[l + 1];
            if (cpe[l] < capetrigr) goto L170;
            plo = pup;
            trmlo = trmup;
        }
L170:
        ltp1 = kb;
        cape = 0.f;
        for (l = kb; l >= 1; --l) {
            if (cpe[l] < capetrigr) break;
            else if (cpe[l] > cape) { ltp1 = l; cape = cpe[l]; }
            if (newswap && rhk[l] < rh_deep) break;
        }
        ltop = max(min(ltp1, lbot), 1);
        if (cape > capecnv) {
            capecnv = cape;
            pspcnv = psp;
            thbtcnv = thbt;
            lbotcnv = lbot;
            ltopcnv = ltop;
            for (l = lmh; l >= 1; --l) { cpecnv[l] = cpe[l]; dtvcnv[l] = dtv[l]; thescnv[l] = thes[l]; }
        }
    }

    if (capecnv > 0.f) {
        psp = pspcnv;
        thbt = thbtcnv;
        lbot = lbotcnv;
        ltop = ltopcnv;
        pbot = prsmid(lbot);
        ptop = prsmid(ltop);
        for (l = lmh; l >= 1; --l) { cpe[l] = cpecnv[l]; dtv[l] = dtvcnv[l]; thes[l] = thescnv[l]; }
    }
    if (ptop > pbot - BMJ_PNO || ltop > lbot - 2 || capecnv <= 0.f) {
        lbot = 0;
        ltop = lm;
        pbot = prsmid(lmh);
        ptop = pbot;
        cldefi = BMJ_AVGEFI * sm + BMJ_STEFI * (1.f - sm);
        goto Lreturn;
    }
    depth = pbot - ptop;
    if (depth >= depmin * 0.75f) plume = true;
    if (depth >= depmin) {
        deep = true;
    } else {
        shallow = true;
        goto L600;
    }

    // ---- 300: deep convection
    lb = lbot;
    efi = cldefi;
    for (l = 1; l <= lmh; ++l) {
        dift[l] = 0.f;
        difq[l] = 0.f;
        tkl = t(l);
        tk[l] = tkl;
        trefk[l] = tkl;
        qkl = q(l);
        qk[l] = qkl;
        qrefk[l] = qkl;
        pkl = prsmid(l);
        pk[l] = pkl;
        psk[l] = pkl;
        rxnerp = rxner[l];
        rxnerk[l] = rxnerp;
        tref[l] = bmj_ttblex2(tb, pkl, thes[l]);
        therk[l] = tref[l] * rxnerp;
    }
    ltp1 = ltop + 1;
    lbm1 = lb - 1;
    pkb = pk[lb];
    pkt = pk[ltop];
    stabdl = (efi - BMJ_EFIMN_C) * BMJ_SLOPST + BMJ_STABDS;
    el[lb] = elwv;
    l0 = lb;
    pk0 = pk[lb];
    treflo = trefk[lb];
    therkx = therk[lb];
    rxnerlo = rxnerk[lb];
    therky = therk[lbm1];
    rxnerhi = rxnerk[lbm1];
    for (l = lbm1; l >= ltop; --l) {
        if (t(l + 1) < tiw) goto L430;
        treflo = ((therky - therkx) * stabdl + treflo * rxnerlo) / rxnerhi;
        trefk[l] = treflo;
        el[l] = elwv;
        rxnerlo = rxnerhi;
        therkx = therky;
        rxnerhi = rxnerk[l - 1];
        therky = therk[l - 1];
        l0 = l;
        pk0 = pk[l0];
    }
    goto L450;
L430:
    l0m1 = l0 - 1;
    rdp0t = 1.f / (pk0 - pkt);
    dthem = therk[l0] - trefk[l0] * rxnerk[l0];
    for (l = ltop; l <= l0m1; ++l) {
        trefk[l] = (therk[l] - (pk[l] - pkt) * dthem * rdp0t) / rxnerk[l];
        el[l] = eliv;
    }
L450:
    depwl = pkb - pk0;
    depth = BMJ_PFRZ * psfc * BMJ_RSFCP;
    sm1 = 1.f - sm;
    pbotfc = 1.f;
    for (itrefi = 1; itrefi <= BMJ_ITREFI_MAX; ++itrefi) {          // cloud_efficiency
        dspbk = ((efi - BMJ_EFIMN_C) * slopbs + dspbss * pbotfc) * sm
              + ((efi - BMJ_EFIMN_C) * slopbl + dspbsl * pbotfc) * sm1;
        dsp0k = ((efi - BMJ_EFIMN_C) * slop0s + dsp0ss * pbotfc) * sm
              + ((efi - BMJ_EFIMN_C) * slop0l + dsp0sl * pbotfc) * sm1;
        dsptk = ((efi - BMJ_EFIMN_C) * slopts + dsptss * pbotfc) * sm
              + ((efi - BMJ_EFIMN_C) * sloptl + dsptsl * pbotfc) * sm1;
        for (l = ltop; l <= lb; ++l) {
            if (depwl >= depth) {
                if (l < l0) dsp = ((pk0 - pk[l]) * dsptk + (pk[l] - pkt) * dsp0k) / (pk0 - pkt);
                else dsp = ((pkb - pk[l]) * dsp0k + (pk[l] - pk0) * dspbk) / (pkb - pk0);
            } else {
                dsp = dsp0k;
                if (l < l0) dsp = ((pk0 - pk[l]) * dsptk + (pk[l] - pkt) * dsp0k) / (pk0 - pkt);
            }
            if (pk[l] > BMJ_PQM) {
                psk[l] = pk[l] + dsp;
                rxnersk[l] = powf(1.e5f / psk[l], cappa);
                thsk[l] = trefk[l] * rxnerk[l];
                qrefk[l] = pq0 / psk[l] * expf(a2 * (thsk[l] - a3 * rxnersk[l])
                                                   / (thsk[l] - a4 * rxnersk[l]));
            } else {
                qrefk[l] = qk[l];
            }
        }
        for (iter = 1; iter <= 2; ++iter) {                          // enthalpy_conservation
            sumde = 0.f;
            sumdp = 0.f;
            for (l = ltop; l <= lb; ++l) {
                sumde = ((tk[l] - trefk[l]) * cp + (qk[l] - qrefk[l]) * el[l]) * dprs(l) + sumde;
                sumdp = sumdp + dprs(l);
            }
            hcorr = sumde / (sumdp - dprs(ltop));
            lcor = ltop + 1;
            lqm = 1;
            for (l = 1; l <= lb; ++l) if (pk[l] <= BMJ_PQM) lqm = l;
            if (lcor <= lqm) {
                for (l = lcor; l <= lqm; ++l) trefk[l] = trefk[l] + hcorr * rcp;
                lcor = lqm + 1;
            }
            for (l = lcor; l <= lb; ++l) {
                tskl = trefk[l] * rxnerk[l] / rxnersk[l];
                dhdt = qrefk[l] * a23m4l / ((tskl - a4) * (tskl - a4)) + cp;
                trefk[l] = hcorr / dhdt + trefk[l];
                thskl = trefk[l] * rxnerk[l];
                qrefk[l] = pq0 / psk[l] * expf(a2 * (thskl - a3 * rxnersk[l])
                                                   / (thskl - a4 * rxnersk[l]));
            }
        }
        avrgt = 0.f;
        preck = 0.f;
        dsq = 0.f;
        dst = 0.f;
        for (l = ltop; l <= lb; ++l) {
            tkl = tk[l];
            diftl = (trefk[l] - tkl) * tauk;
            difql = (qrefk[l] - qk[l]) * tauk;
            avrgtl = (tkl + tkl + diftl);
            dpot = dprs(l) / avrgtl;
            dst = diftl * dpot + dst;
            dsq = difql * el[l] * dpot + dsq;
            avrgt = avrgtl * dprs(l) + avrgt;
            preck = diftl * dprs(l) + preck;
            dift[l] = diftl;
            difq[l] = difql;
        }
        dst = (dst + dst) * cp;
        dsq = dsq + dsq;
        dentpy = dst + dsq;
        avrgt = avrgt / (sumdp + sumdp);
        drheat = (preck * sm + fmaxf(1.e-7f, preck) * (1.f - sm)) * cp / avrgt;
        drheat = fmaxf(drheat, 1.e-20f);
        efi = BMJ_EFIFC * dentpy / drheat;
        efi = fminf(efi, 1.f);
        efi = fmaxf(efi, BMJ_EFIMN_C);
    }

    if (dentpy >= BMJ_EPSNTP && preck > BMJ_EPSPR && !nodeep) {
        iswap = 0;
        cldefi = efi;
        if (sm > 0.5f) fefi = (cldefi - BMJ_EFIMN_C) * BMJ_SLOPES + BMJ_EFMNTS;
        else fefi = (cldefi - BMJ_EFIMN_C) * BMJ_SLOPEL + BMJ_EFMNTL;
        fefi = (dentpy - BMJ_EPSNTP) * fefi / dentpy;
        preck = preck * fefi;
        cup = preck * cprlg;
        pcpcol = cup;
        for (l = ltop; l <= lb; ++l) {
            dtdt(l) = dift[l] * fefi * rdtcnvc;
            dqdt(l) = difq[l] * fefi * rdtcnvc;
        }
        if (mmntdeep) {
            facuv = fefi * rdtcnvc * uvscald;
            if (l0 > ltop && l0 < lb) {
                ubar = 0.f; vbar = 0.f; sumdp = 0.f;
                for (l = l0; l <= lb; ++l) {
                    ubar = u(l) * dprs(l) + ubar;
                    vbar = v(l) * dprs(l) + vbar;
                    sumdp = dprs(l) + sumdp;
                }
                rdpsum = 1.f / sumdp;
                ubar = ubar * rdpsum;
                vbar = vbar * rdpsum;
                for (l = l0; l <= lb; ++l) {
                    dudt(l) = (ubar - u(l)) * facuv;
                    dvdt(l) = (vbar - v(l)) * facuv;
                }
                dum = ubar - u(l0);
                dvm = vbar - v(l0);
                for (l = ltop; l <= l0 - 1; ++l) {
                    dudt(l) = (pk[l] - pkt) * dum * rdp0t * facuv;
                    dvdt(l) = (pk[l] - pkt) * dvm * rdp0t * facuv;
                }
            } else {
                ubar = 0.f; vbar = 0.f; sumdp = 0.f;
                for (l = ltop; l <= lb; ++l) {
                    ubar = u(l) * dprs(l) + ubar;
                    vbar = v(l) * dprs(l) + vbar;
                    sumdp = dprs(l) + sumdp;
                }
                rdpsum = 1.f / sumdp;
                ubar = ubar * rdpsum;
                vbar = vbar * rdpsum;
                for (l = ltop; l <= lb; ++l) {
                    dudt(l) = (ubar - u(l)) * facuv;
                    dvdt(l) = (vbar - v(l)) * facuv;
                }
            }
        }
    } else {
        cldefi = BMJ_EFIMN_C * sm + BMJ_STEFI * (1.f - sm);
        ptpk = fmaxf(BMJ_PSHU, pk[lbot] - depmin);
        if (!newswap) {
            for (l = 1; l <= lmh; ++l) if (pk[l] <= ptpk) ltop = l + 1;
        }
        if (!newswap) {
            ltp1 = lbot;
            for (l = lbot - 1; l >= ltop; --l) {
                if (dtv[l] > 0.f) ltp1 = l;
                else break;
            }
            ltop = min(ltp1, lbot);
        }
        iswap = 1;
        ptop = pk[ltop];
        shallow = true;
        deep = false;
    }

L600:
    (void)iswap; (void)deep; (void)cup;
    if (!shallow) goto Lreturn;
    for (l = 1; l <= lmh; ++l) {
        pk[l] = prsmid(l);
        tk[l] = t(l);
        qk[l] = q(l);
        trefk[l] = t(l);
        qrefk[l] = q(l);
        qsatk[l] = pq0 / pk[l] * expf(a2 * (tk[l] - a3) / (tk[l] - a4));
        rxnerk[l] = rxner[l];
        thvref[l] = tk[l] * rxnerk[l] * (qk[l] * p608 + 1.f);
        if (tk[l] >= tiw) el[l] = elwv;
        else el[l] = eliv;
    }
    tlev2 = t(lbot) * powf((pk[lbot] - BMJ_PONE) / pk[lbot], cappa);
    qsat1 = pq0 / pk[lbot] * expf(a2 * (t(lbot) - a3) / (tk[lbot] - a4));
    qsat2 = pq0 / (pk[lbot] - BMJ_PONE) * expf(a2 * (tlev2 - a3) / (tlev2 - a4));
    rhshmax = qsat2 / qsat1;
    sumdp = 0.f;
    rhavg = 0.f;
    for (l = lbot; l >= ltop; --l) {
        rhavg = rhavg + dprs(l) * qk[l] / qsatk[l];
        sumdp = sumdp + dprs(l);
    }
    if (rhavg / sumdp > rhshmax) {
        ltsh = ltop;
        if (newswap) overshoot = true;
        for (l = ltop - 1; l >= 1; --l) {
            rhavg = rhavg + dprs(l) * qk[l] / qsatk[l];
            sumdp = sumdp + dprs(l);
            if (!newswap) {
                if (cpe[l] > 0.f) ltsh = l;
                else break;
            } else {
                if (dtv[l] <= 0.f) {
                    if (cpe[l] <= 0.f) break;
                    if (overshoot) overshoot = false;
                    else break;
                }
                ltsh = l;
            }
            if (rhavg / sumdp <= rhshmax) break;
            if (pk[l] <= BMJ_PSHU) break;
        }
        ltop = ltsh;
    }
    lbm1 = lbot - 1;
    ptpk = ptop;
    ltp1 = ltop - 1;
    depth = pbot - ptop;
    if (ptop > pbot - BMJ_PNO || ltop > lbot - 2) {
        lbot = 0;
        ltop = lm;
        ptop = pbot;
        goto Lreturn;
    }
    if (newall && plume) goto L810;
    thtpk = t(ltp1) * rxner[ltp1];
    tthk = (thtpk - BMJ_THL) * BMJ_RDTH;
    qqk = tthk - truncf(tthk);
    it = (int)tthk + 1;
    if (it < 1) { it = 1; qqk = 0.f; }
    if (it >= BMJ_JTB) { it = BMJ_JTB - 1; qqk = 0.f; }
    bqs00k = tb.qs0[it - 1];
    sqs00k = tb.sqs[it - 1];
    bqs10k = tb.qs0[it];
    sqs10k = tb.sqs[it];
    bqk = (bqs10k - bqs00k) * qqk + bqs00k;
    sqk = (sqs10k - sqs00k) * qqk + sqs00k;
    tqk = (q(ltp1) - bqk) / sqk * BMJ_RDQ;
    ppk = tqk - truncf(tqk);
    iq = (int)tqk + 1;
    if (iq < 1) { iq = 1; ppk = 0.f; }
    if (iq >= BMJ_ITB) { iq = BMJ_ITB - 1; ppk = 0.f; }
    part1 = (BPTBL(iq + 1, it) - BPTBL(iq, it)) * ppk;
    part2 = (BPTBL(iq, it + 1) - BPTBL(iq, it)) * qqk;
    part3 = (BPTBL(iq, it) - BPTBL(iq + 1, it) - BPTBL(iq, it + 1) + BPTBL(iq + 1, it + 1)) * ppk * qqk;
    ptpk = BPTBL(iq, it) + part1 + part2 + part3;
    dpmix = ptpk - psp;
    if (fabsf(dpmix) < 3000.f) dpmix = -3000.f;
    smix = (thtpk - thbt) / dpmix * BMJ_STABS;
    treflo = trefk[lbot + 1];
    pklo = pk[lbot + 1];
    pkhi = pk[lbot];
    rxnerlo = rxnerk[lbot + 1];
    rxnerhi = rxnerk[lbot];
    lmid = (int)(.5f * (float)(lbot + ltop));
    for (l = lbot; l >= ltop; --l) {
        treflo = ((pkhi - pklo) * smix + treflo * rxnerlo) / rxnerhi;
        trefk[l] = treflo;
        if (l <= lmid) trefk[l] = fmaxf(trefk[l], tk[l] + BMJ_DTSHAL);
        rxnerlo = rxnerhi;
        pklo = pkhi;
        rxnerhi = rxnerk[l - 1];
        pkhi = pk[l - 1];
    }
    sumdt = 0.f;
    sumdp = 0.f;
    for (l = ltop; l <= lbot; ++l) {
        sumdt = (tk[l] - trefk[l]) * dprs(l) + sumdt;
        sumdp = sumdp + dprs(l);
    }
    rdpsum = 1.f / sumdp;
    fpk[lbot] = trefk[lbot];
    tcorr = sumdt * rdpsum;
    for (l = ltop; l <= lbot; ++l) {
        trfkl = trefk[l] + tcorr;
        trefk[l] = trfkl;
        fpk[l] = trfkl;
    }
    psum = 0.f;
    qsum = 0.f;
    potsum = 0.f;
    qotsum = 0.f;
    otsum = 0.f;
    dst = 0.f;
    fptk = fpk[ltop];
    for (l = ltop; l <= lbot; ++l) {
        dpkl = fpk[l] - fptk;
        psum = dpkl * dprs(l) + psum;
        qsum = qk[l] * dprs(l) + qsum;
        rtbar = 2.f / (trefk[l] + tk[l]);
        otsum = dprs(l) * rtbar + otsum;
        potsum = dpkl * rtbar * dprs(l) + potsum;
        qotsum = qk[l] * rtbar * dprs(l) + qotsum;
        dst = (trefk[l] - tk[l]) * rtbar * dprs(l) / el[l] + dst;
    }
    psum = psum * rdpsum;
    qsum = qsum * rdpsum;
    rotsum = 1.f / otsum;
    potsum = potsum * rotsum;
    qotsum = qotsum * rotsum;
    dst = dst * rotsum * cp;
    if (newupup && dst > 0.f && plume) goto L810;
    if (dst > 0.f) {
        if (newswap) goto L810;
        lbot = 0; ltop = lm; ptop = pbot;
        goto Lreturn;
    }
    dstq = dst * BMJ_EPSDN;
    den = potsum - psum;
    if (-den / psum < 5.e-5f) {
        if (newswap) goto L810;
        lbot = 0; ltop = lm; ptop = pbot;
        goto Lreturn;
    } else {
        dqref = (qotsum - dstq - qsum) / den;
    }
    if (dqref < 0.f) {
        if (newswap) goto L810;
        lbot = 0; ltop = lm; ptop = pbot;
        goto Lreturn;
    }
    qrftp = qsum - dqref * psum;
    for (l = ltop; l <= lbot; ++l) {
        qrfkl = (fpk[l] - fptk) * dqref + qrftp;
        tnew = (trefk[l] - tk[l]) * tauksc + tk[l];
        qsatk[l] = pq0 / pk[l] * expf(a2 * (tnew - a3) / (tnew - a4));
        qnew = (qrfkl - qk[l]) * tauksc + qk[l];
        if (qnew < qsatk[l] * BMJ_RHLSC) {
            if (newswap) goto L810;
            lbot = 0; ltop = lm; ptop = pbot;
            goto Lreturn;
        }
        if (qnew > qsatk[l] * BMJ_RHHSC) {
            if (newswap) goto L810;
            lbot = 0; ltop = lm; ptop = pbot;
            goto Lreturn;
        }
        thvref[l] = trefk[l] * rxnerk[l] * (qrfkl * p608 + 1.f);
        qrefk[l] = qrfkl;
    }
    for (l = ltop; l <= lbot; ++l) {
        if (l - 1 < 1) continue;   // Fortran reads prsmid(0) (out of bounds) only if ltop == 1; undefined there
        dtdp = (thvref[l - 1] - thvref[l]) / (prsmid(l) - prsmid(l - 1));
        if (dtdp < BMJ_EPSDT) {
            if (newswap) goto L810;
            lbot = 0; ltop = lm; ptop = pbot;
            goto Lreturn;
        }
    }
    if (mmntshal1) {
        facuv = tauksc * rdtcnvc * uvscals1;
        ubar = 0.f; vbar = 0.f; sumdp = 0.f;
        for (l = ltop; l <= lbot; ++l) {
            ubar = u(l) * dprs(l) + ubar;
            vbar = v(l) * dprs(l) + vbar;
            sumdp = dprs(l) + sumdp;
        }
        rdpsum = 1.f / sumdp;
        ubar = ubar * rdpsum;
        vbar = vbar * rdpsum;
        for (l = ltop; l <= lbot; ++l) {
            dudt(l) = (ubar - u(l)) * facuv;
            dvdt(l) = (vbar - v(l)) * facuv;
        }
    }
    goto L820;

L810:
    for (l = 1; l <= lmh; ++l) {
        wcld[l] = 0.f;
        rhk[l] = qk[l] / qsatk[l];
        thvmk[l] = tk[l] * rxnerk[l];
        tref[l] = bmj_ttblex2(tb, pk[l], thescnv[l]);   // pk(l) = prsmid(l) here; table chosen on prsmid < plq
        thmak[l] = tref[l] * rxnerk[l];
    }
    sumdp = 0.f; sumrh = 0.f; a11 = 0.f; a12 = 0.f;
    b1qsat = 0.f; b2qsat = 0.f; b1thvm = 0.f; b1thma = 0.f; b2thvm = 0.f; b2thma = 0.f;
    b1rh = 0.f; b2rh = 0.f;
    for (l = ltop; l <= lbot; ++l) {
        sumdp = dprs(l) + sumdp;
        sumrh = rhk[l] * dprs(l) + sumrh;
        a11 = prsmid(l) * prsmid(l) * dprs(l) + a11;
        a12 = prsmid(l) * dprs(l) + a12;
        b1qsat = qsatk[l] * prsmid(l) * dprs(l) + b1qsat;
        b1thvm = thvmk[l] * prsmid(l) * dprs(l) + b1thvm;
        b1thma = thmak[l] * prsmid(l) * dprs(l) + b1thma;
        b1rh = rhk[l] * prsmid(l) * dprs(l) + b1rh;
        b2qsat = qsatk[l] * dprs(l) + b2qsat;
        b2thvm = thvmk[l] * dprs(l) + b2thvm;
        b2thma = thmak[l] * dprs(l) + b2thma;
        b2rh = rhk[l] * dprs(l) + b2rh;
    }
    rhmean = sumrh / sumdp;
    if (rhmean > 0.95f) {
        lbot = 0; ltop = lm; ptop = pbot;
        goto Lreturn;
    }
    a21 = a12;
    a22 = sumdp;
    rden = 1.f / (a11 * a22 - a12 * a21);
    aqs = (b1qsat * a22 - a12 * b2qsat) * rden;
    avm = (b1thvm * a22 - a12 * b2thvm) * rden;
    ama = (b1thma * a22 - a12 * b2thma) * rden;
    arh = (b1rh * a22 - a12 * b2rh) * rden;
    bqs = (a11 * b2qsat - b1qsat * a21) * rden;
    bvm = (a11 * b2thvm - b1thvm * a21) * rden;
    bma = (a11 * b2thma - b1thma * a21) * rden;
    brh = (a11 * b2rh - b1rh * a21) * rden;
    (void)brh;
    if (arh < 0.f) {
        lbot = 0; ltop = lm; ptop = pbot;
        goto Lreturn;
    }
    adef = (1.f - BMJ_DEFTOP) * 2.f / (pk[lbot] - pk[ltop]);
    for (l = ltop; l <= lbot; ++l) {
        fk = (pk[l] - pk[ltop]) * adef + BMJ_DEFTOP;
        rhref = rhmean * fk;
        wcld[l] = (1.f - BMJ_WDRY) * rhref / (1.f - BMJ_WDRY * rhref);
        trefk[l] = ((1.f - wcld[l]) * (avm * pk[l] + bvm) + wcld[l] * (ama * pk[l] + bma)) / rxnerk[l];
        if (nodeep && fabsf(trefk[l] - tk[l]) > 5.f) {
            lbot = 0; ltop = lm; ptop = pbot;
            goto Lreturn;
        }
        qrefk[l] = rhref * (aqs * prsmid(l) + bqs);
    }
    sumdp = 0.f; sumdt = 0.f; sumdq = 0.f;
    for (l = ltop; l <= lbot; ++l) {
        sumdp = dprs(l) + sumdp;
        sumdt = (tk[l] - trefk[l]) * dprs(l) + sumdt;
        sumdq = (qk[l] - qrefk[l]) * dprs(l) + sumdq;
    }
    rdpsum = 1.f / sumdp;
    tcorr = sumdt * rdpsum;
    qcorr = sumdq * rdpsum;
    for (l = ltop; l <= lbot; ++l) {
        trefk[l] = trefk[l] + tcorr;
        qrefk[l] = qrefk[l] + qcorr;
    }
    dsq = 0.f;
    dst = 0.f;
    for (l = ltop; l <= lbot; ++l) {
        tkl = tk[l];
        diftl = (trefk[l] - tkl) * tauksc;
        difql = (qrefk[l] - qk[l]) * tauksc;
        dpot = dprs(l) / (tkl + tkl + diftl);
        dst = diftl * dpot + dst;
        dsq = difql * el[l] * dpot + dsq;
    }
    dst = (dst + dst) * cp;
    dsq = dsq + dsq;
    dentpy = dst + dsq;
    if (dentpy < 0.f) {
        lbot = 0; ltop = lm; ptop = pbot;
        goto Lreturn;
    }
    if (mmntshal2) {
        // Fortran "go to 8888" skips the urefk/vrefk block (b1u ... dvdt), so only the 8888 part runs.
        (void)b1u; (void)b1v; (void)b2u; (void)b2v; (void)au; (void)av; (void)bu; (void)bv;
        (void)sumdu; (void)sumdv; (void)ucorr; (void)vcorr; (void)urefk; (void)vrefk;
        ubar = 0.f; vbar = 0.f; sumdp = 0.f;
        for (l = ltop; l <= lbot; ++l) {
            ubar = u(l) * dprs(l) + ubar;
            vbar = v(l) * dprs(l) + vbar;
            sumdp = dprs(l) + sumdp;
        }
        rdpsum = 1.f / sumdp;
        ubar = ubar * rdpsum;
        vbar = vbar * rdpsum;
        facuv = tauksc * rdtcnvc * uvscals2;
        for (l = ltop; l <= lbot; ++l) {
            dudt(l) = (ubar - u(l)) * facuv;
            dvdt(l) = (vbar - v(l)) * facuv;
        }
    }
L820:
    for (l = ltop; l <= lbot; ++l) {
        dtdt(l) = (trefk[l] - tk[l]) * tauksc * rdtcnvc;
        dqdt(l) = (qrefk[l] - qk[l]) * tauksc * rdtcnvc;
    }

Lreturn:
    (void)ptop; (void)pbot; (void)depth; (void)lbm1; (void)l0m1; (void)thsk; (void)qbt; (void)ama;
    *lbot_out = lbot;
    *ltop_out = ltop;
    *cldefi_io = cldefi;
    *pcpcol_io = pcpcol;
#undef dprs
#undef prsmid
#undef q
#undef t
#undef exner
#undef u
#undef v
#undef dqdt
#undef dtdt
#undef dudt
#undef dvdt
}
