// nam-gpu unit phys-bmj-col-cloudbase: BMJ cloud-base search (max-buoyancy parcel, table
// indices, saturation point, cloud base) for one column.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module and routine header kept as the notice:
//   module module_cu_bmj  (NMMB, nam.v4.2.11)
//   subroutine bmj(...)   programer: z. janjic, yugoslav fed. hydromet. inst., beograd
// Ported to CUDA for nam-gpu from phys/module_CU_BMJ.F90:598-696 (nam.v4.2.11, 83df4e13c1).
//
// Fortran -> CUDA, one thread per column. Fortran line map:
//   590      thesp(l) = 0, l = 1..lmh (prep range; repeated here so the array is defined)
//   600-601  pelevfc = prsmid(lmh)*elevfc ; pbtmx = prsmid(lmh)-pone
//   602-609  capecnv, pspcnv, thbtcnv = 0 ; lbot = lmh ; ltop = lmh ; lbotcnv = lbot ; ltopcnv = lbot
//   612-664  prep_loop: kb = lmh..1 (exit at prsmid(kb) < pelevfc unless entrain). Table indices
//            ittb/iqtb with the Fortran clamps, saturation point psp (4-point interpolation on
//            ptbl), rxners = (1e5/psp)**cappa, thesp = thbt*exp(elocp*qbt*rxners/thbt), psk = psp.
//   671-696  max_buoy_loop head: kb = lmh..1 (exit at prsmid(kb) < pelevfc). Cloud base lbot: last
//            level with p < psp and p >= pqm, plus one; fallback to the last level with p < pbtmx
//            when pbot >= pbtmx or lbot >= lmh. Stored per kb as lbot_kb / pbot_kb, because lbot is
//            carried from one kb to the next and read by the cloud-top body (698 onward, units
//            phys-bmj-col-cloudtop and later). ltop = lbot and ptop = pbot at 698-699 are the same
//            values and are not stored separately.
// Not in this unit: 697 onward (entrainment, entropy check, cloud top, deep and shallow).
//
// Fortran level l (1-based, 1 = model top) is read at [(l-1)*stride]. A level-major field
// F(i,j,l) has its column base at IDX3(i0,j0,0,IM,JM) and stride JM*IM between levels.
//
// Not produced here, named: rxner(l) = 1/exner(l) is recomputed inline where the Fortran uses it
// (prep, 580-586). prsmid is the BMJ midlevel pressure passed to bmj from bmjdrv (pcol = phmid):
// the wrapper passes the PHMID state field that phys-bmj-col-prep stages (no separate array). The tables are the BMJ_INIT tables (phys-bmj-init): ptbl(itb,jtb), qs0(jtb), sqs(jtb),
// stored C-order as ptbl[jtb][itb] and qs0/sqs[jtb].
//
// Q clamp: bmjdrv passes qcol = max(epsq, q) (module_CU_BMJ.F90 line 294, epsq = 1e-12) to bmj, so
// bmj reads the clamped q at line 616. The kernel applies the same clamp to the raw Q it reads
// (fmaxf(NMMB_EPSQ, q)); it is idempotent, so a caller that already passes qcol gets the same values.
//
// Precision: kfpt = single, so float everywhere. Fast numerics (fmad on, default IEEE division).
// No host sync, no device malloc, no printf. The __device__ function is the column form for the
// bmjdrv column path (phys-bmj-bmjdrv); the standalone kernel is the one-launch form.

#include "nmmb_common.cuh"

// module_CU_BMJ.F90 parameters (module header lines 40-74; elevfc is local to bmj)
#define BMJ_ITB     76           // itb
#define BMJ_JTB     134          // jtb
#define BMJ_PQM     20000.0f     // pqm
#define BMJ_PONE    2500.0f      // pone
#define BMJ_ELEVFC  0.6f         // elevfc
#define BMJ_THL     210.0f       // thl
#define BMJ_THH     365.0f       // thh
#define BMJ_RDTH    ((BMJ_JTB - 1.0f) / (BMJ_THH - BMJ_THL))   // rdth = (jtb-1.)/(thh-thl)
#define BMJ_RDQ     ((float)(BMJ_ITB - 1))                     // rdq = itb-1
#define BMJ_ELIWV   2.683e6f     // eliwv (elocp = eliwv/cp inside bmj)

// Fortran A(l) at 1-based level l1, stride s between levels.
#define BMJ_L(a, l1, s)    a[((size_t)((l1) - 1)) * (s)]
// Fortran ptbl(iq1, it1), iq1 in 1..itb, it1 in 1..jtb, stored [jtb][itb].
#define BMJ_PTBL(iq1, it1) ptbl[((size_t)((it1) - 1)) * BMJ_ITB + ((iq1) - 1)]

struct BmjCbOut {
    float pelevfc, pbtmx;
    float capecnv, pspcnv, thbtcnv;
    int   lbotcnv, ltopcnv;
};

// One column of the cloud-base search. The column views t, q, exner, prsmid (inputs) and
// thesp, psk, rxnersk, lbot_kb, pbot_kb (outputs) point at level 1 of the column. Level l is at
// [(l-1)*s] for the inputs and [(l-1)*so] for the outputs.
__device__ inline BmjCbOut bmj_col_cloudbase(
    const int lmh, const int entrain, const float cp, const float cappa,
    const float* __restrict__ t, const float* __restrict__ q,
    const float* __restrict__ exner, const float* __restrict__ prsmid, const size_t s,
    const float* __restrict__ ptbl, const float* __restrict__ qs0, const float* __restrict__ sqs,
    float* thesp, float* psk, float* rxnersk, int* lbot_kb, float* pbot_kb, const size_t so)
{
    const float elocp = BMJ_ELIWV / cp;                       // elocp=eliwv/cp (bmj locals)

    // ---- 600-601 ----
    BmjCbOut o;
    o.pelevfc = BMJ_L(prsmid, lmh, s) * BMJ_ELEVFC;           // pelevfc=prsmid(lmh)*elevfc
    o.pbtmx   = BMJ_L(prsmid, lmh, s) - BMJ_PONE;             // pbtmx  =prsmid(lmh)-pone
    const float pelevfc = o.pelevfc;
    const float pbtmx   = o.pbtmx;

    // ---- 602-609 ----
    o.capecnv = 0.0f;
    o.pspcnv  = 0.0f;
    o.thbtcnv = 0.0f;
    int lbot = lmh;                                           // lbot=lmh
    o.lbotcnv = lbot;                                         // lbotcnv=lbot
    o.ltopcnv = lbot;                                         // ltopcnv=lbot

    // ---- 590: thesp(l)=0 for l=1..lmh (prep range) ----
    for (int l = 1; l <= lmh; ++l) BMJ_L(thesp, l, so) = 0.0f;

    // ---- 612-664: prep_loop ----
    for (int kb = lmh; kb >= 1; --kb) {
        if (BMJ_L(prsmid, kb, s) < pelevfc && !entrain) break;     // exit
        const float qbt = fmaxf(NMMB_EPSQ, BMJ_L(q, kb, s));     // qcol = max(epsq, q), line 294; read at 616
        const float rxner_kb = 1.0f / BMJ_L(exner, kb, s);        // rxner(kb)=1./exner(kb)
        const float thbt = BMJ_L(t, kb, s) * rxner_kb;            // thbt=t(kb)*rxner(kb)
        const float tth = (thbt - BMJ_THL) * BMJ_RDTH;            // tth=(thbt-thl)*rdth
        float qq1 = tth - truncf(tth);                            // qq1=tth-aint(tth)
        int ittb = (int)tth + 1;                                  // ittb=int(tth)+1
        if (ittb < 1) {
            ittb = 1;
            qq1 = 0.0f;
        } else if (ittb >= BMJ_JTB) {
            ittb = BMJ_JTB - 1;
            qq1 = 0.0f;
        }
        const float bqs00k = qs0[ittb - 1];                       // qs0(ittbk)
        const float sqs00k = sqs[ittb - 1];                       // sqs(ittbk)
        const float bqs10k = qs0[ittb];                           // qs0(ittbk+1)
        const float sqs10k = sqs[ittb];                           // sqs(ittbk+1)
        const float bq = (bqs10k - bqs00k) * qq1 + bqs00k;        // bq=(bqs10k-bqs00k)*qq1+bqs00k
        const float sq = (sqs10k - sqs00k) * qq1 + sqs00k;        // sq=(sqs10k-sqs00k)*qq1+sqs00k
        const float tq = (qbt - bq) / sq * BMJ_RDQ;               // tq=(qbt-bq)/sq*rdq
        float pp1 = tq - truncf(tq);                              // pp1=tq-aint(tq)
        int iqtb = (int)tq + 1;                                   // iqtb=int(tq)+1
        if (iqtb < 1) {
            iqtb = 1;
            pp1 = 0.0f;
        } else if (iqtb >= BMJ_ITB) {
            iqtb = BMJ_ITB - 1;
            pp1 = 0.0f;
        }
        const int iq = iqtb;
        const int it = ittb;
        const float p00k = BMJ_PTBL(iq,     it);                  // ptbl(iq  ,it  )
        const float p10k = BMJ_PTBL(iq + 1, it);                  // ptbl(iq+1,it  )
        const float p01k = BMJ_PTBL(iq,     it + 1);              // ptbl(iq  ,it+1)
        const float p11k = BMJ_PTBL(iq + 1, it + 1);              // ptbl(iq+1,it+1)
        const float psp = p00k + (p10k - p00k) * pp1 + (p01k - p00k) * qq1
                        + (p00k - p10k - p01k + p11k) * pp1 * qq1;
        const float rxners = powf(1.e5f / psp, cappa);            // rxners=(1.e5/psp)**cappa
        BMJ_L(thesp,   kb, so) = thbt * expf(elocp * qbt * rxners / thbt);
        BMJ_L(psk,     kb, so) = psp;
        BMJ_L(rxnersk, kb, so) = rxners;
    }

    // ---- 671-696: max_buoy_loop head (cloud base per kb) ----
    for (int kb = lmh; kb >= 1; --kb) {
        if (BMJ_L(prsmid, kb, s) < pelevfc) break;                // exit
        const float psp = BMJ_L(psk, kb, so);                     // psp=psk(kb)
        for (int l = 1; l <= lmh - 1; ++l) {
            const float p = BMJ_L(prsmid, l, s);
            if (p < psp && p >= BMJ_PQM) lbot = l + 1;           // lbot=l+1
        }
        float pbot = BMJ_L(prsmid, lbot, s);                      // pbot=prsmid(lbot)
        if (pbot >= pbtmx || lbot >= lmh) {
            for (int l = 1; l <= lmh - 1; ++l) {
                const float p = BMJ_L(prsmid, l, s);
                if (p < pbtmx) lbot = l;                          // lbot=l
            }
            pbot = BMJ_L(prsmid, lbot, s);                        // pbot=prsmid(lbot)
        }
        BMJ_L(lbot_kb, kb, so) = lbot;                            // carried to phys-bmj-col-cloudtop
        BMJ_L(pbot_kb, kb, so) = pbot;
    }
    return o;
}

// Standalone launch: one thread per column (grid_cols). Fields are NMMB [LM, JM, IM]; the
// per-column outputs are [JM, IM]; the per-level outputs are [LM, JM, IM].
extern "C" __global__ void phys_bmj_col_cloudbase_column(
    const int IM, const int JM, const int lmh, const int entrain,
    const float cp, const float cappa,
    const float* __restrict__ t, const float* __restrict__ q,
    const float* __restrict__ exner, const float* __restrict__ prsmid,
    const float* __restrict__ ptbl, const float* __restrict__ qs0, const float* __restrict__ sqs,
    float* thesp, float* psk, float* rxnersk, int* lbot_kb, float* pbot_kb,
    float* pelevfc, float* pbtmx, int* lbotcnv, int* ltopcnv,
    float* capecnv, float* pspcnv, float* thbtcnv)
{
    COLUMN_IJ(IM, JM)
    const size_t c3 = IDX3(i0, j0, 0, IM, JM);                    // level 1 of this column
    const size_t c2 = IDX2(i0, j0, IM);
    const size_t s3 = (size_t)JM * IM;                            // stride between levels

    const BmjCbOut o = bmj_col_cloudbase(
        lmh, entrain, cp, cappa,
        t + c3, q + c3, exner + c3, prsmid + c3, s3,
        ptbl, qs0, sqs,
        thesp + c3, psk + c3, rxnersk + c3, lbot_kb + c3, pbot_kb + c3, s3);

    pelevfc[c2] = o.pelevfc;
    pbtmx[c2]   = o.pbtmx;
    lbotcnv[c2] = o.lbotcnv;
    ltopcnv[c2] = o.ltopcnv;
    capecnv[c2] = o.capecnv;
    pspcnv[c2]  = o.pspcnv;
    thbtcnv[c2] = o.thbtcnv;
}
