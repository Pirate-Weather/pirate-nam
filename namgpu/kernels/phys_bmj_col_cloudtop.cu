// nam-gpu unit phys-bmj-col-cloudtop: BMJ cloud-top search, entrainment, buoyancy (CAPE) and the
// maximum-instability selection, with the quick exit, for the BMJ convection scheme.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module and routine header kept as the notice:
//   module module_cu_bmj  (NMMB, nam.v4.2.11)
//   subroutine bmj(...)  -- programer: z. janjic, yugoslav fed. hydromet. inst., beograd
// Ported to CUDA for nam-gpu from phys/module_CU_BMJ.F90:697-939 (nam.v4.2.11, 83df4e13c1).
//
// Fortran -> CUDA map (module_CU_BMJ.F90 line ranges, the sections of the routine in order):
//   697-705   ltop=lbot, ptop=pbot                                           bmjct_kb_body
//   705-770   entrainment (entrain branch: thes/qbtk, fefi, ffup, fprs, recursion over dprs)
//             and the no-entrain branch (thes=thesp(kb), qbtk=qk(kb))         bmjct_kb_body
//   first entropy check: cpe=0, dtv=0 for l=1..lmh; "if(kb.lt.lbot) go to 170"  bmjct_kb_body
//   buoyancy below cloud base (kb-1 .. lbot)                                  bmjct_kb_body
//   cloud base layer (l=lbot-1, pup=psp, tup=thbt/rxners, tsp, qsp, trmup)     bmjct_kb_body
//   above saturation (ttblex on the low or moisture table, qup, qwat, dtv, cpe) bmjct_kb_body
//   in cloud above cloud base (l=lbot-2 .. 1)                                 bmjct_kb_body
//   170: cloud-top search (ltp1, cape maximum, newswap rh_deep exit, ltop)     bmjct_kb_body
//   cape > capecnv: save psp, thbt, lbot, ltop and cpe, dtv, thes              bmjct_kb_body
//   906-939   maximum instability restore, quick exit (lbot=0, ltop=lm,
//             pbot=prsmid(lmh), cldefi), depth=pbot-ptop (the next unit's first line)  bmjct_post
// The per-kb cloud-base search (671-696) sits in the loop of the routine. It is carried here as
// bmjct_cloudbase_kb because the cloud-top body reads the loop-carried lbot of that search. The loop
// itself (bmjct_max_buoy) is the max_buoy_loop of 671-905. psk, rxnersk and thesp are the outputs
// of the prep_loop at 598-670 (the cloud-base unit phys-bmj-col-cloudbase), passed in per level.
//
// Layout (CONTRACT section 1): 3-D [LM, JM, IM] (level l, 1-based, at index (l-1)*JM*IM), 2-D
// [JM, IM]. One thread per column (kernels.grid_cols), with an l-loop. Thread-local per-column arrays
// are 1-based (index l), the same convention as bmj_column in phys_bmj_bmjdrv.cu. Fast numerics
// (fmad on), float everywhere (kfpt = single). No host sync, no device malloc, no printf.
//
// Notes (where the Fortran is not a plain transliteration):
//   1. FUP. The routine declares a local REAL fup (module_CU_BMJ.F90:495) that hides the module
//      parameter fup = 1/200000 (line 36) and is never assigned before its use at line 723
//      (ffup = fup/(fefi*fefi)). The Fortran value there is undefined. This port uses the module
//      parameter 1/200000, the value clearly intended. DECLARED DIVERGENCE, pending the project lead's ruling
//      (undefined-value rule): no witness run and no sign-off are recorded for this read.
//   2. capetrigr. It is set only after the "if(kb.lt.lbot) go to 170" test. On that path the cape
//      loop runs with cpe(l)=0 for every l, so its outcome does not depend on capetrigr (cape stays 0
//      and capecnv is not updated). This port sets capetrigr=0 on that path.
//   3. lbot=1. The cloud-base layer reads level lbot-1. With lbot=1 the Fortran reads index 0 (t, q,
//      prsmid), which is out of bounds. The below-cloud loop above it (defined levels) runs as in
//      the Fortran and writes its dtv and cpe. This port then takes goto 170 before the cloud-base
//      layer, so the cloud-base layer's dtv/cpe write, the above-saturation update, the in-cloud
//      loop and the cloud-top search for that column are skipped. DECLARED DIVERGENCE, pending
//      The undefined-value rule: no witness run and no sign-off are recorded.
//   4. The tables sthe, the0, ttbl (low pressure) and stheq, the0q, ttblq (moisture) are the BMJ_INIT
//      tables (phys-bmj-init): ttbl element (kth, kp) at [kp-1][kth-1] in C order.

#include "nmmb_common.cuh"

#define BMJCT_LMAX 64

// module_CU_BMJ.F90 parameters used by lines 697-939 (single precision).
#define BMJCT_PQM      20000.0f
#define BMJCT_PONE     2500.0f
#define BMJCT_PNO      1000.0f
#define BMJCT_PL       2500.0f
#define BMJCT_PLQ      70000.0f
#define BMJCT_PH       105000.0f
#define BMJCT_ITB      76
#define BMJCT_JTB      134
#define BMJCT_ITBQ     152
#define BMJCT_JTBQ     440
#define BMJCT_RDP      ((BMJCT_ITB - 1.0f) / (BMJCT_PH - BMJCT_PL))
#define BMJCT_RDPQ     ((BMJCT_ITBQ - 1.0f) / (BMJCT_PH - BMJCT_PLQ))
#define BMJCT_RDTHE    ((float)(BMJCT_JTB - 1))
#define BMJCT_RDTHEQ   ((float)(BMJCT_JTBQ - 1))
#define BMJCT_ELEVFC   0.6f
#define BMJCT_EFIMN    0.20f
#define BMJCT_EFMNTL   0.70f
#define BMJCT_EFMNTS   0.70f
#define BMJCT_ENPLO    98500.0f
#define BMJCT_ENPUP    95000.0f
#define BMJCT_RENDP    (1.0f / (BMJCT_ENPLO - BMJCT_ENPUP))
#define BMJCT_FUP      (1.0f / 200000.0f)     // see note 1
#define BMJCT_AVGEFI   ((BMJCT_EFIMN + 1.0f) * 0.5f)
#define BMJCT_STEFI    1.0f
#define BMJCT_SLOPES   ((1.0f - BMJCT_EFMNTS) / (1.0f - BMJCT_EFIMN))
#define BMJCT_SLOPEL   ((1.0f - BMJCT_EFMNTL) / (1.0f - BMJCT_EFIMN))
#define BMJCT_DTPTRIGR (-0.0f * BMJCT_PONE)   // dttrigr=-0.0 (line 42), dtptrigr=dttrigr*pone

// Column inputs. Level l of array p is p[(l-1)*ls]: global [LM, JM, IM] fields are passed as
// base+col with ls=JM*IM; thread-local arrays with ls=1.
struct BmjCtIn {
    const float* t;        // T (tk)
    const float* q;        // Q (qk)
    const float* prsmid;   // PRSMID (prsmid, pressure at mid levels)
    const float* dprs;     // DPRS (dprs, layer thickness in pressure)
    const float* exner;    // EXNER
    const float* thesp;    // THESP (prep_loop output, cloud-base unit)
    const float* rhk;      // RHK (rhk, from the BMJ prelude, phys-bmj-col-prep)
    const float* psk;      // PSK (psk, prep_loop output)
    const float* rxnersk;  // RXNERSK (prep_loop output)
    const float* sthe;
    const float* the0;
    const float* ttbl;
    const float* stheq;
    const float* the0q;
    const float* ttblq;
    size_t ls;
    int lm, lmh, entrain, newswap;
    float rh_deep, a2, a3, a4, p608, pq0, r_d;
};

// Thread-local column state. Arrays are 1-based (index l, 1..lmh).
struct BmjCtCol {
    float thes[BMJCT_LMAX + 2], thee[BMJCT_LMAX + 2], qbtk[BMJCT_LMAX + 2], qbte[BMJCT_LMAX + 2];
    float cpe[BMJCT_LMAX + 2], dtv[BMJCT_LMAX + 2];
    float cpecnv[BMJCT_LMAX + 2], dtvcnv[BMJCT_LMAX + 2], thescnv[BMJCT_LMAX + 2];
    int lbot, ltop, lbotcnv, ltopcnv;
    float pbot, ptop, psp, thbt, qbt, rxners, capecnv, pspcnv, thbtcnv;
    float pelevfc, pbtmx, cldefi, sm;
};

__device__ __forceinline__ float bmjct_ld(const BmjCtIn& in, const float* p, const int l)
{
    return p[(size_t)(l - 1) * in.ls];
}

// TTBLEX (module_CU_BMJ.F90:2071-2154), the exact body. Table ttbl(jtbx, itbx): element (kth, kp)
// at ttbl[(kp-1)*jtbx + (kth-1)].
__device__ __forceinline__ float bmjct_ttblex(
    const int itbx, const int jtbx, const float plx, const float prsmid, const float rdpx,
    const float rdthex, const float thesp,
    const float* __restrict__ sthe, const float* __restrict__ the0, const float* __restrict__ ttbl)
{
    const float pk = prsmid;
    const float tpk = (pk - plx) * rdpx;
    float qq = tpk - truncf(tpk);
    int iptb = (int)tpk + 1;
    if (iptb < 1) {
        iptb = 1;
        qq = 0.0f;
    }
    if (iptb >= itbx) {
        iptb = itbx - 1;
        qq = 0.0f;
    }
    const float bthe00k = the0[iptb - 1];
    const float sthe00k = sthe[iptb - 1];
    const float bthe10k = the0[iptb];
    const float sthe10k = sthe[iptb];
    const float bthk = (bthe10k - bthe00k) * qq + bthe00k;
    const float sthk = (sthe10k - sthe00k) * qq + sthe00k;
    const float tthk = (thesp - bthk) / sthk * rdthex;
    float pp = tthk - truncf(tthk);
    int ithtb = (int)tthk + 1;
    if (ithtb < 1) {
        ithtb = 1;
        pp = 0.0f;
    }
    if (ithtb >= jtbx) {
        ithtb = jtbx - 1;
        pp = 0.0f;
    }
    const float* col_lo = ttbl + (size_t)(iptb - 1) * jtbx;
    const float* col_hi = ttbl + (size_t)iptb * jtbx;
    const float t00k = col_lo[ithtb - 1];
    const float t10k = col_lo[ithtb];
    const float t01k = col_hi[ithtb - 1];
    const float t11k = col_hi[ithtb];
    return (t00k + (t10k - t00k) * pp + (t01k - t00k) * qq
            + (t00k - t10k - t01k + t11k) * pp * qq);
}

// The two calls of ttblex in the routine: pup < plq uses the low-pressure table, else the moisture table.
__device__ __forceinline__ float bmjct_tref(const BmjCtIn& in, const float prs, const float thesp)
{
    if (prs < BMJCT_PLQ) {
        return bmjct_ttblex(BMJCT_ITB, BMJCT_JTB, BMJCT_PL, prs, BMJCT_RDP, BMJCT_RDTHE, thesp,
                            in.sthe, in.the0, in.ttbl);
    }
    return bmjct_ttblex(BMJCT_ITBQ, BMJCT_JTBQ, BMJCT_PLQ, prs, BMJCT_RDPQ, BMJCT_RDTHEQ, thesp,
                        in.stheq, in.the0q, in.ttblq);
}

// Lines 697-905: one pass of the max_buoy_loop body for parcel level kb. Inputs in c: lbot, pbot,
// psp, thbt, qbt, rxners from the cloud-base search of this kb, and the running capecnv state.
__device__ void bmjct_kb_body(const BmjCtIn& in, BmjCtCol& c, const int kb)
{
    const int lmh = in.lmh;
    int l, ltp1;
    float fefi, ffup, fprs, dpup, dplo;
    float plo, tlo, trmlo, capetrigr, pup, tup, dp, trmup, tsp, qsp, qup, qwat, cape;

    // ---- cloud top computation ----
    c.ltop = c.lbot;
    c.ptop = c.pbot;

    // ---- entrainment during parcel ascent ----
    if (in.entrain) {
        for (l = lmh; l >= kb; --l) {
            c.thes[l] = bmjct_ld(in, in.thesp, kb);
            c.qbtk[l] = bmjct_ld(in, in.q, kb);
        }
        for (l = kb; l >= 1; --l) {
            c.thes[l] = bmjct_ld(in, in.thesp, l);
            c.qbtk[l] = bmjct_ld(in, in.q, l);
        }
        for (l = 1; l <= lmh; ++l) {
            c.thee[l] = c.thes[l];
            c.qbte[l] = c.qbtk[l];
        }
        if (c.sm > 0.5f) {
            fefi = (c.cldefi - BMJCT_EFIMN) * BMJCT_SLOPES + BMJCT_EFMNTS;
        } else {
            fefi = (c.cldefi - BMJCT_EFIMN) * BMJCT_SLOPEL + BMJCT_EFMNTL;
        }
        ffup = BMJCT_FUP / (fefi * fefi);
        if (c.pbot > BMJCT_ENPLO) {
            fprs = 1.0f;
        } else if (c.pbot > BMJCT_ENPUP) {
            fprs = (c.pbot - BMJCT_ENPUP) * BMJCT_RENDP;
        } else {
            fprs = 0.0f;
        }
        ffup = ffup * fprs * fprs * 0.5f;
        dpup = bmjct_ld(in, in.dprs, kb);
        for (l = kb - 1; l >= 1; --l) {
            dplo = dpup;
            dpup = bmjct_ld(in, in.dprs, l);
            c.thes[l] = ((-ffup * dplo + 1.0f) * c.thes[l + 1]
                         + (c.thee[l] * dpup + c.thee[l + 1] * dplo) * ffup) / (ffup * dpup + 1.0f);
            c.qbtk[l] = ((-ffup * dplo + 1.0f) * c.qbtk[l + 1]
                         + (c.qbte[l] * dpup + c.qbte[l + 1] * dplo) * ffup) / (ffup * dpup + 1.0f);
        }
    } else {
        // ---- no entrainment ----
        for (l = lmh; l >= 1; --l) {
            c.thes[l] = bmjct_ld(in, in.thesp, kb);
            c.qbtk[l] = bmjct_ld(in, in.q, kb);
        }
    }

    // ---- first entropy check ----
    for (l = 1; l <= lmh; ++l) {
        c.cpe[l] = 0.0f;
        c.dtv[l] = 0.0f;
    }

    // ---- buoyancy check including deep convection ----
    capetrigr = 0.0f;
    if (kb < c.lbot) goto L170;

    l = kb;
    plo = bmjct_ld(in, in.prsmid, l);
    tlo = c.thbt * bmjct_ld(in, in.exner, l);
    trmlo = ((c.qbt * in.p608 + 1.0f) * tlo
             - (bmjct_ld(in, in.q, l) * in.p608 + 1.0f) * bmjct_ld(in, in.t, l)) * in.r_d / plo;
    capetrigr = BMJCT_DTPTRIGR / bmjct_ld(in, in.t, c.lbot);

    // below cloud base
    if (kb - c.lbot >= 1) {
        for (l = kb - 1; l >= c.lbot; --l) {
            pup = bmjct_ld(in, in.prsmid, l);
            tup = c.thbt * bmjct_ld(in, in.exner, l);
            dp = plo - pup;
            trmup = ((c.qbt * in.p608 + 1.0f) * tup
                     - (bmjct_ld(in, in.q, l) * in.p608 + 1.0f) * bmjct_ld(in, in.t, l)) * in.r_d / pup;
            c.dtv[l] = (trmlo + trmup) * dp * 0.5f;
            c.cpe[l] = c.dtv[l] + c.cpe[l + 1];
            if (c.cpe[l] < capetrigr) goto L170;
            plo = pup;
            trmlo = trmup;
        }
    }

    // cloud base layer
    l = c.lbot - 1;
    if (l < 1) goto L170;                                   // note 3
    pup = c.psp;
    tup = c.thbt / c.rxners;
    tsp = ((bmjct_ld(in, in.t, l + 1) - bmjct_ld(in, in.t, l))
           / (plo - bmjct_ld(in, in.prsmid, l))) * (pup - c.pbot) + bmjct_ld(in, in.t, l);
    qsp = ((bmjct_ld(in, in.q, l + 1) - bmjct_ld(in, in.q, l))
           / (plo - bmjct_ld(in, in.prsmid, l))) * (pup - c.pbot) + bmjct_ld(in, in.q, l);
    dp = plo - pup;
    trmup = ((c.qbt * in.p608 + 1.0f) * tup - (qsp * in.p608 + 1.0f) * tsp) * in.r_d / pup;
    c.dtv[l] = (trmlo + trmup) * dp * 0.5f;
    plo = pup;
    trmlo = trmup;

    // above saturation: updraft t and q along the moist adiabat
    pup = bmjct_ld(in, in.prsmid, l);
    tup = bmjct_tref(in, pup, c.thes[l]);
    qup = in.pq0 / pup * expf(in.a2 * (tup - in.a3) / (tup - in.a4));
    qwat = c.qbt - qup;
    dp = plo - pup;
    trmup = ((qup * in.p608 + 1.0f - qwat) * tup
             - (bmjct_ld(in, in.q, l) * in.p608 + 1.0f) * bmjct_ld(in, in.t, l)) * in.r_d / pup;
    c.dtv[l] = (trmlo + trmup) * dp * 0.5f + c.dtv[l];
    c.cpe[l] = c.dtv[l] + c.cpe[l + 1];
    if (c.cpe[l] < capetrigr) goto L170;
    plo = pup;
    trmlo = trmup;

    // in cloud, above cloud base
    for (l = c.lbot - 2; l >= 1; --l) {
        pup = bmjct_ld(in, in.prsmid, l);
        tup = bmjct_tref(in, pup, c.thes[l]);
        qup = in.pq0 / pup * expf(in.a2 * (tup - in.a3) / (tup - in.a4));
        qwat = c.qbt - qup;
        dp = plo - pup;
        trmup = ((qup * in.p608 + 1.0f - qwat) * tup
                 - (bmjct_ld(in, in.q, l) * in.p608 + 1.0f) * bmjct_ld(in, in.t, l)) * in.r_d / pup;
        c.dtv[l] = (trmlo + trmup) * dp * 0.5f;
        c.cpe[l] = c.dtv[l] + c.cpe[l + 1];
        if (c.cpe[l] < capetrigr) goto L170;
        plo = pup;
        trmlo = trmup;
    }

    // ---- 170: cloud-top level is where cape is a maximum ----
L170:
    ltp1 = kb;
    cape = 0.0f;
    for (l = kb; l >= 1; --l) {
        if (c.cpe[l] < capetrigr) {
            break;
        } else if (c.cpe[l] > cape) {
            ltp1 = l;
            cape = c.cpe[l];
        }
        if (in.newswap && bmjct_ld(in, in.rhk, l) < in.rh_deep) break;
    }
    c.ltop = max(min(ltp1, c.lbot), 1);

    // ---- maximum instability ----
    if (cape > c.capecnv) {
        c.capecnv = cape;
        c.pspcnv = c.psp;
        c.thbtcnv = c.thbt;
        c.lbotcnv = c.lbot;
        c.ltopcnv = c.ltop;
        for (l = lmh; l >= 1; --l) {
            c.cpecnv[l] = c.cpe[l];
            c.dtvcnv[l] = c.dtv[l];
            c.thescnv[l] = c.thes[l];
        }
    }
}

// Lines 906-939: maximum instability restore and quick exit. Returns 1 when the Fortran returns
// (lbot=0, no deep or shallow convection for this column), 0 otherwise. Depth (line 939) is
// computed by the next unit from pbot and ptop.
__device__ int bmjct_post(const BmjCtIn& in, BmjCtCol& c)
{
    const int lmh = in.lmh;
    if (c.capecnv > 0.0f) {
        c.psp = c.pspcnv;
        c.thbt = c.thbtcnv;
        c.lbot = c.lbotcnv;
        c.ltop = c.ltopcnv;
        c.pbot = bmjct_ld(in, in.prsmid, c.lbot);
        c.ptop = bmjct_ld(in, in.prsmid, c.ltop);
        for (int l = lmh; l >= 1; --l) {
            c.cpe[l] = c.cpecnv[l];
            c.dtv[l] = c.dtvcnv[l];
            c.thes[l] = c.thescnv[l];
        }
    }
    if (c.ptop > c.pbot - BMJCT_PNO || c.ltop > c.lbot - 2 || c.capecnv <= 0.0f) {
        c.lbot = 0;
        c.ltop = in.lm;
        c.pbot = bmjct_ld(in, in.prsmid, lmh);
        c.ptop = c.pbot;
        c.cldefi = BMJCT_AVGEFI * c.sm + BMJCT_STEFI * (1.0f - c.sm);
        return 1;
    }
    return 0;
}

// Lines 671-696: the cloud-base search of one max_buoy_loop pass (carried here, see the header).
__device__ void bmjct_cloudbase_kb(const BmjCtIn& in, BmjCtCol& c, const int kb)
{
    const int lmh = in.lmh;
    c.qbt = bmjct_ld(in, in.q, kb);
    c.thbt = bmjct_ld(in, in.t, kb) * (1.0f / bmjct_ld(in, in.exner, kb));   // rxner(kb)
    c.psp = bmjct_ld(in, in.psk, kb);
    c.rxners = bmjct_ld(in, in.rxnersk, kb);
    for (int l = 1; l <= lmh - 1; ++l) {
        const float p = bmjct_ld(in, in.prsmid, l);
        if (p < c.psp && p >= BMJCT_PQM) c.lbot = l + 1;
    }
    c.pbot = bmjct_ld(in, in.prsmid, c.lbot);
    if (c.pbot >= c.pbtmx || c.lbot >= lmh) {
        for (int l = 1; l <= lmh - 1; ++l) {
            if (bmjct_ld(in, in.prsmid, l) < c.pbtmx) c.lbot = l;
        }
        c.pbot = bmjct_ld(in, in.prsmid, c.lbot);
    }
    c.ltop = c.lbot;
}

// The max_buoy_loop (671-905) and the post-loop section (906-939) for one column.
__device__ int bmjct_max_buoy(const BmjCtIn& in, BmjCtCol& c)
{
    const int lmh = in.lmh;
    c.pelevfc = bmjct_ld(in, in.prsmid, lmh) * BMJCT_ELEVFC;
    c.pbtmx = bmjct_ld(in, in.prsmid, lmh) - BMJCT_PONE;
    c.capecnv = 0.0f;
    c.pspcnv = 0.0f;
    c.thbtcnv = 0.0f;
    c.lbot = lmh;
    c.ltop = lmh;
    c.lbotcnv = c.lbot;
    c.ltopcnv = c.lbot;
    for (int kb = lmh; kb >= 1; --kb) {
        if (bmjct_ld(in, in.prsmid, kb) < c.pelevfc) break;
        bmjct_cloudbase_kb(in, c, kb);
        bmjct_kb_body(in, c, kb);
    }
    return bmjct_post(in, c);
}

// One thread per column. Outputs (per column unless noted, [JM, IM]):
//   LBOT, LTOP (int, Fortran 1-based), PBOT, PTOP, PSP, THBT, QBT, RXNERS, CAPE (= capecnv),
//   CLDEFI (in/out, the quick exit writes it);
//   CPE, DTV, THES, QBTK [LM, JM, IM] for levels 1..LMH (the arrays the shallow and deep units read).
extern "C" __global__ void phys_bmj_col_cloudtop(
    const int IM, const int JM, const int lm, const int lmh,
    const int entrain, const int newswap, const float rh_deep,
    const float a2, const float a3, const float a4, const float p608, const float pq0, const float r_d,
    const float* __restrict__ T, const float* __restrict__ Q, const float* __restrict__ PRSMID,
    const float* __restrict__ DPRS, const float* __restrict__ EXNER,
    const float* __restrict__ THESP, const float* __restrict__ RHK,
    const float* __restrict__ PSK, const float* __restrict__ RXNERSK,
    const float* __restrict__ SM, float* __restrict__ CLDEFI,
    const float* __restrict__ STHE, const float* __restrict__ THE0, const float* __restrict__ TTBL,
    const float* __restrict__ STHEQ, const float* __restrict__ THE0Q, const float* __restrict__ TTBLQ,
    int* __restrict__ LBOT, int* __restrict__ LTOP,
    float* __restrict__ PBOT, float* __restrict__ PTOP, float* __restrict__ PSP,
    float* __restrict__ THBT, float* __restrict__ QBT, float* __restrict__ RXNERS,
    float* __restrict__ CAPE,
    float* __restrict__ CPE, float* __restrict__ DTV, float* __restrict__ THES,
    float* __restrict__ QBTK)
{
    COLUMN_IJ(IM, JM)
    // The wrapper refuses lm > BMJCT_LMAX before launch (phys_bmj_col_cloudtop.py). This guard only
    // keeps the device arrays in bounds if the kernel is ever launched directly.
    if (lm > BMJCT_LMAX) return;
    const size_t LS = (size_t)IM * JM;
    const size_t col = (size_t)col_;

    BmjCtIn in;
    in.t = T + col;
    in.q = Q + col;
    in.prsmid = PRSMID + col;
    in.dprs = DPRS + col;
    in.exner = EXNER + col;
    in.thesp = THESP + col;
    in.rhk = RHK + col;
    in.psk = PSK + col;
    in.rxnersk = RXNERSK + col;
    in.sthe = STHE;
    in.the0 = THE0;
    in.ttbl = TTBL;
    in.stheq = STHEQ;
    in.the0q = THE0Q;
    in.ttblq = TTBLQ;
    in.ls = LS;
    in.lm = lm;
    in.lmh = lmh;
    in.entrain = entrain;
    in.newswap = newswap;
    in.rh_deep = rh_deep;
    in.a2 = a2;
    in.a3 = a3;
    in.a4 = a4;
    in.p608 = p608;
    in.pq0 = pq0;
    in.r_d = r_d;

    BmjCtCol c;
    c.cldefi = CLDEFI[col];
    c.sm = SM[col];
    (void)bmjct_max_buoy(in, c);

    CLDEFI[col] = c.cldefi;
    LBOT[col] = c.lbot;
    LTOP[col] = c.ltop;
    PBOT[col] = c.pbot;
    PTOP[col] = c.ptop;
    PSP[col] = c.psp;
    THBT[col] = c.thbt;
    QBT[col] = c.qbt;
    RXNERS[col] = c.rxners;
    CAPE[col] = c.capecnv;
    for (int l = 1; l <= lmh; ++l) {
        const size_t o = (size_t)(l - 1) * LS + col;
        CPE[o] = c.cpe[l];
        DTV[o] = c.dtv[l];
        THES[o] = c.thes[l];
        QBTK[o] = c.qbtk[l];
    }
}
