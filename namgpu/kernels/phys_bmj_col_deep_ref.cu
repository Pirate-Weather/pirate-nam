// nam-gpu unit phys-bmj-col-deep-ref: BMJ deep-convection reference temperature and humidity
// profiles, freezing level and the setup of the first adjustment.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module and routine kept as the notice (same file as phys-bmj-spline):
//   module module_cu_bmj  (NMMB, nam.v4.2.11)
//   subroutine bmj, deep convection section (label 300 .. comment block ahead of the cloud
//   efficiency iteration loop)
// Ported to CUDA for nam-gpu from phys/module_CU_BMJ.F90:940-1084 (nam.v4.2.11, 83df4e13c1).
//
// Scope of this unit (Fortran line ranges):
//   940-958   depth = pbot - ptop, plume test, deep / shallow test (shallow: go to 600, the
//             shallow path is phys-bmj-col-shallow-top; this kernel only sets the flags there)
//   300-306   lb = lbot, efi = cldefi
//   init loop over l = 1..lmh: dift, difq, tk, trefk, qk, qrefk, pk, psk, rxnerk,
//             tref (moist adiabat through ttblex), therk
//   l = lbm1 .. ltop descending: trefk below the freezing level (goto 430 on t(l+1) < tiw),
//             el = elwv
//   430       trefk above the freezing level (el = eliv)
//   450       depwl, depth (= pfrz*psfc*rsfcp, overwrites the cloud depth as in the Fortran),
//             sm1, pbotfc
//   The first-adjustment block (sumdt/sumdp/tcorr) is commented out in the NAM source and is not
//   ported. Fortran ltp1 = ltop+1 is computed for the same names; downstream units recompute it.
//   Fortran dthem and rdp0t are locals of the freezing-level block only.
//
// Layout (CONTRACT.md sections 1 and 3): Fortran A(i,j,l) is A[l0][j0][i0] with
// IDX3(i0, j0, l0, IM, JM); the column stride between levels is S = JM*IM. Per-column scalars
// are [JM, IM] (IDX2). The Fortran 1-based level index is kept in the body through the LV()
// accessor, so every line reads like the Fortran.
// One thread per column (i, j); the l-loops run inside the thread (CONTRACT section 3,
// kernels.grid_cols). No host sync, no device allocation, no printf.
//
// Inputs written by the bmjdrv column units before this kernel (names in the wrapper):
//   T          [LM,JM,IM]  column temperature (bmjdrv tcol)
//   Q          [LM,JM,IM]  column humidity, already max(EPSQ, Q) (bmjdrv qcol)
//   prsmid     [LM,JM,IM]  mid-layer pressure (bmjdrv pcol = phmid)
//   rxner      [LM,JM,IM]  1/exner (bmj prep, Fortran rxner(l)=1./exner(l))
//   thes       [LM,JM,IM]  theta-e of the moist adiabat as left by cloud-base/cloud-top (Fortran thes(l))
//   lbot, ltop [JM,IM]     cloud base / cloud top levels (int32, 1-based as in Fortran)
//   pbot, ptop [JM,IM]     cloud base / cloud top pressures (Fortran pbot, ptop)
//   psfc       [JM,IM]     surface pressure (bmjdrv psfc = phint(lm+1))
//   seamask    [JM,IM]     xland - 1 (Fortran sm)
//   cldefi     [JM,IM]     convective cloud efficiency (Fortran cldefi)
//   tables: sthe, the0 [ITB]; ttbl [ITB,JTB] (CuPy, Fortran ttbl(jtb,itb)); stheq, the0q [ITBQ];
//           ttblq [ITBQ,JTBQ] (Fortran ttblq(jtbq,itbq)). Built by phys-bmj-init, uploaded once.
// Outputs (wrapper names BMJ_DR_*): see the wrapper docstring. For a column whose DEEP flag is 0
// the kernel writes only PLUME, DEEP and SHALLOW; the other outputs are stale there.

#include "nmmb_common.cuh"

// module_CU_BMJ parameters (nam.v4.2.11, module header)
#define BMJ_EFIMN     0.20f
#define BMJ_STABDS    0.90f
#define BMJ_STABDF    0.90f
#define BMJ_SLOPST    ((BMJ_STABDF - BMJ_STABDS) / (1.0f - BMJ_EFIMN))   // slopst
#define BMJ_PSH       20000.0f
#define BMJ_PFRZ      15000.0f
#define BMJ_PL        2500.0f
#define BMJ_PLQ       70000.0f
#define BMJ_PH        105000.0f
#define BMJ_RSFCP     (1.0f / 101300.0f)
#define BMJ_ITB       76
#define BMJ_JTB       134
#define BMJ_ITBQ      152
#define BMJ_JTBQ      440
#define BMJ_RDP       ((BMJ_ITB - 1.0f) / (BMJ_PH - BMJ_PL))     // rdp
#define BMJ_RDPQ      ((BMJ_ITBQ - 1.0f) / (BMJ_PH - BMJ_PLQ))   // rdpq
#define BMJ_RDTHE     ((float)(BMJ_JTB - 1))                     // rdthe = jtb-1.
#define BMJ_RDTHEQ    ((float)(BMJ_JTBQ - 1))                    // rdtheq = jtbq-1.
#define BMJ_PNO       1000.0f                                    // pno (thin-cloud depth test, Pa)
#define BMJ_AVGEFI    ((BMJ_EFIMN + 1.0f) * 0.5f)               // avgefi = (efimn+1.)*0.5
#define BMJ_STEFI     1.0f                                       // stefi

// 1-based level access for column c (c = IDX2 of the column), Fortran a(l).
#define LV(a, l)      a[c + (size_t)((l) - 1) * S]

// Fortran TTBLEX (module_CU_BMJ.F90:2071-2154), inlined per column. Table layout: ttbl is
// Fortran ttbl(jtbx, itbx), so ttbl(ithtb, iptb) is ttbl[(iptb-1)*jtbx + (ithtb-1)].
// Index clamps, int()/aint() truncation and the four-point blend follow the Fortran order.
__device__ __forceinline__ float bmj_ttblex_dr(
    const int itbx, const int jtbx,
    const float plx, const float prsmid, const float rdpx, const float rdthex, const float thesp,
    const float* __restrict__ sthe, const float* __restrict__ the0,
    const float* __restrict__ ttbl)
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
    const float t00k = ttbl[(size_t)(iptb - 1) * jtbx + (ithtb - 1)];
    const float t10k = ttbl[(size_t)(iptb - 1) * jtbx + (ithtb)];
    const float t01k = ttbl[(size_t)(iptb)     * jtbx + (ithtb - 1)];
    const float t11k = ttbl[(size_t)(iptb)     * jtbx + (ithtb)];
    return (t00k + (t10k - t00k) * pp + (t01k - t00k) * qq
            + (t00k - t10k - t01k + t11k) * pp * qq);
}

extern "C" __global__ void phys_bmj_col_deep_ref(
    const int IM, const int JM, const int LM,
    const float elwv, const float eliv, const float tiw,
    const float* __restrict__ T,
    const float* __restrict__ Q,
    const float* __restrict__ prsmid,
    const float* __restrict__ rxner,
    const float* __restrict__ thes,
    int*   __restrict__ lbot,          // in/out: quick exit rewrites lbot/ltop/pbot/ptop (Fortran)
    int*   __restrict__ ltop,
    float* __restrict__ pbot,
    float* __restrict__ ptop,
    const float* __restrict__ psfc,
    const float* __restrict__ seamask,
    float* __restrict__ cldefi,
    const float* __restrict__ capecnv,
    const float* __restrict__ sthe,
    const float* __restrict__ the0,
    const float* __restrict__ ttbl,
    const float* __restrict__ stheq,
    const float* __restrict__ the0q,
    const float* __restrict__ ttblq,
    // [LM,JM,IM] outputs
    float* __restrict__ trefk, float* __restrict__ qrefk, float* __restrict__ therk,
    float* __restrict__ rxnerk, float* __restrict__ el, float* __restrict__ pk,
    float* __restrict__ psk, float* __restrict__ qk, float* __restrict__ tk,
    float* __restrict__ dift, float* __restrict__ difq, float* __restrict__ tref,
    // [JM,IM] outputs
    int* __restrict__ lb_out, int* __restrict__ l0_out,
    int* __restrict__ deep_out, int* __restrict__ plume_out, int* __restrict__ shallow_out,
    float* __restrict__ pk0_out, float* __restrict__ pkb_out, float* __restrict__ pkt_out,
    float* __restrict__ stabdl_out, float* __restrict__ efi_out, float* __restrict__ depth_out,
    float* __restrict__ depwl_out, float* __restrict__ pbotfc_out, float* __restrict__ sm1_out)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= IM * JM) return;
    const int i0 = col % IM;
    const int j0 = col / IM;
    const size_t c = (size_t)j0 * IM + i0;          // IDX2(i0, j0, IM)
    const size_t S = (size_t)JM * IM;               // stride between levels

    // ---- lines 931-937: quick exit (thin cloud, no CAPE, or cloud top below 1) ----
    // Fortran: if(ptop>pbot-pno.or.ltop>lbot-2.or.capecnv<=0.) then lbot=0, ltop=lm,
    // pbot=prsmid(lmh), ptop=pbot, cldefi=avgefi*sm+stefi*(1-sm), return. plume, deep and shallow
    // are .false. from the bmj prologue (module_CU_BMJ.F90:553-555), so they stay 0 here. The
    // ltop<1 clause cannot come from cloud-top (ltop is 1-based); it keeps the lb-1 reads of
    // lines 222-223 at index >= 2, so lb >= 3 holds on every path past this guard.
    {
        const int lbot_q = lbot[c];
        const int ltop_q = ltop[c];
        const float pbot_q = pbot[c];
        const float ptop_q = ptop[c];
        if (ptop_q > pbot_q - BMJ_PNO || ltop_q > lbot_q - 2 || capecnv[c] <= 0.0f || ltop_q < 1) {
            const float sm_q = seamask[c];
            const float pbot_lmh = LV(prsmid, LM);
            lbot[c] = 0;
            ltop[c] = LM;
            pbot[c] = pbot_lmh;
            ptop[c] = pbot_lmh;
            cldefi[c] = BMJ_AVGEFI * sm_q + BMJ_STEFI * (1.0f - sm_q);
            plume_out[c] = 0;
            deep_out[c] = 0;
            shallow_out[c] = 0;
            return;
        }
    }

    // ---- lines 940-958: depth, plume, deep / shallow ----
    // depmin is the Fortran prologue value depmin = psh*psfc*rsfcp (module_CU_BMJ.F90:551).
    const float psfc_c = psfc[c];
    const float depmin = BMJ_PSH * psfc_c * BMJ_RSFCP;
    const float pbot_c = pbot[c];
    const float ptop_c = ptop[c];
    const float depth_cloud = pbot_c - ptop_c;
    int plume = 0;
    int deep = 0;
    if (depth_cloud >= depmin * 0.75f) plume = 1;
    if (depth_cloud >= depmin) deep = 1;
    deep_out[c] = deep;
    plume_out[c] = plume;
    shallow_out[c] = deep ? 0 : 1;
    if (!deep) return;                               // go to 600: shallow path, other unit

    // ---- label 300: lb = lbot, efi = cldefi ----
    const int lb = lbot[c];
    const float efi = cldefi[c];
    const int ltp = ltop[c];
    const int lbm1 = lb - 1;

    // ---- init loop, l = 1..lmh (lmh = LM) ----
    for (int l = 1; l <= LM; ++l) {
        LV(dift, l) = 0.0f;
        LV(difq, l) = 0.0f;
        const float tkl = LV(T, l);
        LV(tk, l) = tkl;
        LV(trefk, l) = tkl;
        const float qkl = LV(Q, l);
        LV(qk, l) = qkl;
        LV(qrefk, l) = qkl;
        const float pkl = LV(prsmid, l);
        LV(pk, l) = pkl;
        LV(psk, l) = pkl;
        const float rxnerp = LV(rxner, l);
        LV(rxnerk, l) = rxnerp;
        // moist-adiabat temperature (ttblex): low-pressure table set above plq, moisture set below
        float tref_l;
        if (pkl < BMJ_PLQ) {
            tref_l = bmj_ttblex_dr(BMJ_ITB, BMJ_JTB, BMJ_PL, pkl, BMJ_RDP, BMJ_RDTHE,
                                   LV(thes, l), sthe, the0, ttbl);
        } else {
            tref_l = bmj_ttblex_dr(BMJ_ITBQ, BMJ_JTBQ, BMJ_PLQ, pkl, BMJ_RDPQ, BMJ_RDTHEQ,
                                   LV(thes, l), stheq, the0q, ttblq);
        }
        LV(tref, l) = tref_l;
        LV(therk, l) = tref_l * rxnerp;
    }

    // ---- deep convection reference temperature profile ----
    const float pkb = LV(pk, lb);
    const float pkt = LV(pk, ltp);
    const float stabdl = (efi - BMJ_EFIMN) * BMJ_SLOPST + BMJ_STABDS;

    // temperature reference profile below the freezing level
    LV(el, lb) = elwv;
    int l0 = lb;
    float pk0 = LV(pk, lb);
    float treflo = LV(trefk, lb);
    float therkx = LV(therk, lb);
    float rxnerlo = LV(rxnerk, lb);
    float therky = LV(therk, lbm1);
    float rxnerhi = LV(rxnerk, lbm1);
    int above_freezing_path = 0;                     // 1 = fell through to label 430
    for (int l = lbm1; l >= ltp; --l) {
        if (LV(T, l + 1) < tiw) {                    // t(l+1) < tiw: go to 430
            above_freezing_path = 1;
            break;
        }
        treflo = ((therky - therkx) * stabdl + treflo * rxnerlo) / rxnerhi;
        LV(trefk, l) = treflo;
        LV(el, l) = elwv;
        rxnerlo = rxnerhi;
        therkx = therky;
        // rxnerk(l-1), therk(l-1) at l = 1 would read Fortran index 0, which does not exist. The
        // value is never used because the loop ends at l = 1, so the read is skipped there.
        if (l - 1 >= 1) {
            rxnerhi = LV(rxnerk, l - 1);
            therky = LV(therk, l - 1);
        }
        l0 = l;
        pk0 = LV(pk, l0);
    }

    if (above_freezing_path) {
        // ---- label 430: temperature reference profile above the freezing level ----
        const int l0m1 = l0 - 1;
        const float rdp0t = 1.0f / (pk0 - pkt);
        const float dthem = LV(therk, l0) - LV(trefk, l0) * LV(rxnerk, l0);
        for (int l = ltp; l <= l0m1; ++l) {
            LV(trefk, l) = (LV(therk, l) - (LV(pk, l) - pkt) * dthem * rdp0t) / LV(rxnerk, l);
            LV(el, l) = eliv;
        }
    }

    // ---- label 450: depwl, depth, sm1, pbotfc ----
    const float depwl = pkb - pk0;
    const float depth = BMJ_PFRZ * psfc_c * BMJ_RSFCP;      // depth=pfrz*psfc*rsfcp
    const float sm1 = 1.0f - seamask[c];
    const float pbotfc = 1.0f;
    // The first adjustment (sumdt/sumdp/tcorr, trefk shift) is commented out in the NAM source
    // and is not ported. The cloud-efficiency iteration (line 1088 on) is phys-bmj-col-deep-iter.

    lb_out[c] = lb;
    l0_out[c] = l0;
    pk0_out[c] = pk0;
    pkb_out[c] = pkb;
    pkt_out[c] = pkt;
    stabdl_out[c] = stabdl;
    efi_out[c] = efi;
    depth_out[c] = depth;
    depwl_out[c] = depwl;
    pbotfc_out[c] = pbotfc;
    sm1_out[c] = sm1;
}
