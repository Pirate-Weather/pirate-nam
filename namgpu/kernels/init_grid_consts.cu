// init_grid_consts.cu -- GRID_CONSTS of NMMB module_CONTROL (nam.v4.2.11): every row constant
// (dxh, dxv, rdxh, rdxv, dyh, dyv, rdyh, rdyv, ddv, rddv, ddmpu, ddmpv, wpdar, fcp, fdiv, fad, fah,
// dare, rare, curv, hdacxj..hdacvyj), the 2-D fields (f, glat, glon, vlat, vlon, hdacx, hdacy,
// hdacvx, hdacvy) and the SW corner (glat_sw, glon_sw), for the global and the regional branch.
//
// NOTICE: NAM source file nmm/module_CONTROL.F90 (NOAA/NCEP/EMC, US Government work, no licence file).
// The Fortran module carries no licence block; the module header is kept here as the notice:
//   module module_control  (NMMB control, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_CONTROL.F90:351-1139 (nam.v4.2.11, 83df4e13c1).
//
// Kernel sequence (the wrapper launches them in this order on one stream; each step reads only
// what earlier steps wrote, so every Fortran loop-carried dependence is kept):
//   1. gc_rows_interior   one thread per row J: zero the row outputs (Fortran zero loop), then the
//                         per-row formulas. Global: rows J = 3 .. JM-2 (jds+2 .. jde-2, tph = sb+(J-2)dph).
//                         Regional: every row J = 1 .. JM (tph = sb+(J-1)dph), and ddv/rddv
//                         (regional diagonal-distance loop; the same thread owns row J).
//   2. gc_poles_scalars   one thread: global pole rows and ghost lines (south pole J=2, ghost J=1,
//                         north pole J=JM-1, ghost J=JM); then dyh, dyv, rdyh, rdyv, ddmpv,
//                         glat_sw, glon_sw. O(1) work by design (about 40 assignments).
//   3. gc_pole_average    global only: one thread per row J: the height-latitude average of the
//                         H rows (J <-> JM+1-J, J <= JM/2) and of the wind rows (J <-> JM-J, J <= (JM-1)/2).
//   4. gc_global_ddv      global only: ddv/rddv from the averaged dxv (rows 3 .. JM-3) and the
//                         ghost/pole rows (s_bdy and n_bdy blocks).
//   5. gc_fields_2d       one thread per (I, J): f, glat, glon, vlat, vlon and the hdac* 2-D fields
//                         (regional: boundary enhancement with bofac over lnsbc rows/columns).
//
// Single process, one card (CONTRACT section 1): its = ids = 1, ite = ide = IM, jts = jds = 1,
// jte = jde = JM, so w_bdy = e_bdy = s_bdy = n_bdy = true and every ims/jms halo limit collapses
// to the full domain (i_lo = 1, i_hi = IM, j_lo = 1, j_hi = JM).
//
// Indexing: Fortran arrays are 1-based. Row arrays are [JM] float32; 2-D fields are [JM, IM]
// (CONTRACT section 1). GJ(a, J) is a(J) in Fortran, with I and J the 1-based Fortran indices.
//
// Numerics: float everywhere (kfpt = single). The Fortran pi, dtr, a, cp, twom are the single
// precision values of module_CONSTANTS. Operations keep the Fortran association order.
//
// Fortran not mapped (dead code only): the tph/tpv updates in the north-pole section, which no
// north-pole assignment reads, and the commented-out loops in the grid_consts body.

#include "nmmb_common.cuh"

#define GC_PI      3.14159274f           /* pi, kfpt=single (module_CONSTANTS pi) */
#define GC_DTR     (GC_PI / 180.0f)      /* dtr = pi/180 */
#define GC_FPOLE   4.0f                  /* global pole area factor (fpole=4.0) */
#define GJ(a, J)   a[(J) - 1]

#define GC_SCALAR_PARAMS                                                                   \
    const int JM, const int global_, const float dt, const float smag2,                    \
    const float codamp, const float wcor, const float tph0d, const float tlm0d,            \
    const float sbd, const float wbd, const float dphd, const float dlmd,                      const int ioff, const int joff
// sbd/wbd are the REFERENCE grid's origin; ioff/joff the moving-nest window's whole-cell offset from it (0, 0 for
// every grid without window keys), so every rotated coordinate is origin + (index + offset) x step and a ground
// point shared by two windows gets the same bits (namgpu/nest/move.py alignment check)

#define GC_ROW_PARAMS                                                                      \
    float* __restrict__ dxh, float* __restrict__ rdxh, float* __restrict__ dxv,            \
    float* __restrict__ rdxv, float* __restrict__ dare, float* __restrict__ rare,          \
    float* __restrict__ wpdar, float* __restrict__ curv, float* __restrict__ fad,          \
    float* __restrict__ fah, float* __restrict__ fcp, float* __restrict__ fdiv,            \
    float* __restrict__ ddmpu, float* __restrict__ hdacxj, float* __restrict__ hdacyj,     \
    float* __restrict__ hdacvxj, float* __restrict__ hdacvyj

// Derived scalars of GRID_CONSTS (its locals). Every kernel rebuilds them with the same
// expressions, so the values are identical in every kernel.
struct GCS {
    float dt, wcor, acdt, cddamp, tph0, wb, sb, dlm, dph, stph0, ctph0, dyh, dyv, tlm0;
};

__device__ __forceinline__ GCS gc_scalars(float dt, float smag2, float codamp, float wcor,
                                          float tph0d, float tlm0d, float sbd, float wbd,
                                          float dphd, float dlmd) {
    GCS s;
    const float coac = smag2 * smag2;                 // coac = smag2*smag2
    s.dt = dt;
    s.wcor = wcor;
    s.acdt = coac * dt;                               // acdt = coac*dt
    s.cddamp = codamp * dt;                           // cddamp = codamp*dt
    s.tph0 = tph0d * GC_DTR;                          // tph0 = tph0d*dtr
    s.wb = wbd * GC_DTR;                              // wb = wbd*dtr
    s.sb = sbd * GC_DTR;                              // sb = sbd*dtr
    s.dlm = dlmd * GC_DTR;                            // dlm = dlmd*dtr
    s.dph = dphd * GC_DTR;                            // dph = dphd*dtr
    s.tlm0 = tlm0d * GC_DTR;                          // tlm0d*dtr as used in glon/vlon
    s.stph0 = sinf(s.tph0);
    s.ctph0 = cosf(s.tph0);
    s.dyh = NMMB_A * s.dph;                           // dyh = a*dph
    s.dyv = NMMB_A * s.dph;                           // dyv = a*dph
    return s;
}

// ---------------------------------------------------------------------------------------------
// 1. Row kernel: zero the row outputs, then the per-row formulas of the between-the-poles loop
//    (global: 'do j=jds+2,jde-2'; regional: 'do j=jds,jde', same expressions).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void gc_rows_interior(GC_SCALAR_PARAMS, GC_ROW_PARAMS,
                                            float* __restrict__ ddv, float* __restrict__ rddv)
{
    const int J = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (J > JM) return;
    // zero loop (Fortran 'do j=jds,jde': dxh, rdxh, dare, rare, wpdar, fah, fcp, fdiv, dxv,
    // rdxv, curv, fad, ddmpu, hdacxj, hdacyj, hdacvxj, hdacvyj)
    GJ(dxh, J) = 0.f;  GJ(rdxh, J) = 0.f;  GJ(dare, J) = 0.f;  GJ(rare, J) = 0.f;
    GJ(wpdar, J) = 0.f; GJ(fah, J) = 0.f;  GJ(fcp, J) = 0.f;   GJ(fdiv, J) = 0.f;
    GJ(dxv, J) = 0.f;  GJ(rdxv, J) = 0.f;  GJ(curv, J) = 0.f;  GJ(fad, J) = 0.f;
    GJ(ddmpu, J) = 0.f; GJ(hdacxj, J) = 0.f; GJ(hdacyj, J) = 0.f;
    GJ(hdacvxj, J) = 0.f; GJ(hdacvyj, J) = 0.f;

    const GCS s = gc_scalars(dt, smag2, codamp, wcor, tph0d, tlm0d, sbd, wbd, dphd, dlmd);
    const float a = NMMB_A;
    const float dyh = s.dyh, dyv = s.dyv, acdt = s.acdt, cddamp = s.cddamp;

    float tph;
    if (global_) {
        if (J < 3 || J > JM - 2) return;              // jds+2 .. jde-2
        tph = s.sb + (float)(J - 2) * s.dph;          // tph=sb+(j-jds-1)*dph
    } else {
        tph = s.sb + (float)(J - 1 + joff) * s.dph;   // tph=sb+(j-jds)*dph (window: reference row)
    }
    const float tpv = tph + s.dph * 0.5f;             // tpv=tph+dph*0.5
    const float dlm = s.dlm, cp = NMMB_CP;

    const float dxh_ = a * dlm * cosf(tph);           // dxh(j)=a*dlm*cos(tph)
    const float dxv_ = a * dlm * cosf(tpv);           // dxv(j)=a*dlm*cos(tpv)
    GJ(dxh, J) = dxh_;
    GJ(dxv, J) = dxv_;
    GJ(rdxh, J) = 1.f / dxh_;
    GJ(rdxv, J) = 1.f / dxv_;
    GJ(dare, J) = dxh_ * dyh;                         // dare(j)=dxh(j)*dyh
    GJ(rare, J) = 1.f / (dxh_ * dyh);
    GJ(wpdar, J) = -wcor * (dyh * dyh) / (dt * dxh_ * dyh) / 100000.00f;
    GJ(curv, J) = tanf(tpv) / a;                      // curv(j)=tan(tpv)/a
    GJ(fad, J) = -dt / (3.f * dxv_ * dyv * 2.f * 2.f);
    GJ(fah, J) = -dt / (3.f * dxh_ * dyh);
    GJ(fcp, J) = dt / (3.f * dxh_ * dyh * cp);
    GJ(fdiv, J) = 2.f / (3.f * dxh_ * dyh);
    GJ(hdacxj, J) = acdt * dyh * fmaxf(dxh_, dyh) / (4.f * dxh_ * dyh);
    GJ(hdacyj, J) = acdt * dyh * fmaxf(dxh_, dyh) / (4.f * dxh_ * dyh);
    GJ(hdacvxj, J) = acdt * dyv * fmaxf(dxv_, dyv) / (4.f * dxv_ * dyv);
    GJ(hdacvyj, J) = acdt * dyv * fmaxf(dxv_, dyv) / (4.f * dxv_ * dyv);
    GJ(ddmpu, J) = cddamp * dyv / (2.f * dxv_);       // ddmpu(j)=cddamp*dyv/(2*dxv(j))

    if (!global_) {
        // regional diagonal-distance loop: ddv(j)=sqrt(dxv(j)**2+dyv**2)
        const float dd = sqrtf(dxv_ * dxv_ + dyv * dyv);
        GJ(ddv, J) = dd;
        GJ(rddv, J) = 1.f / dd;
    }
}

// ---------------------------------------------------------------------------------------------
// 2. Poles and scalars (single thread, O(1) work). Global branch: south pole, ghost line beyond
//    the south pole, north pole, ghost line beyond the north pole. Both branches: the scalars
//    dyh, dyv, rdyh, rdyv, ddmpv and glat_sw, glon_sw.
//    scal[8] layout: 0 dyh, 1 dyv, 2 rdyh, 3 rdyv, 4 ddmpv, 5 glat_sw, 6 glon_sw, 7 unused.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void gc_poles_scalars(GC_SCALAR_PARAMS, GC_ROW_PARAMS, float* __restrict__ scal)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const GCS s = gc_scalars(dt, smag2, codamp, wcor, tph0d, tlm0d, sbd, wbd, dphd, dlmd);
    const float a = NMMB_A;
    const float dt_ = s.dt, wcor_ = s.wcor, acdt = s.acdt, cddamp = s.cddamp;
    const float dyh = s.dyh, dyv = s.dyv, dph = s.dph, dlm = s.dlm, cp = NMMB_CP;
    // this grid's own SW corner: the reference origin plus the window offset (x + 0.0f = x without a window)
    const float sb = s.sb + (float)joff * s.dph, wb = s.wb + (float)ioff * s.dlm;

    const float ddmpv = cddamp * dyv / (2.f * dyv);   // ddmpv=cddamp*dyv/(2.*dyv)

    if (global_) {
        const float fpole = GC_FPOLE;
        // ---- south pole (jds+1 = 2) ----
        float tph = sb - dph;
        tph = tph + dph;
        const float tpv = tph + dph * 0.5f;
        GJ(dxh, 2) = 0.f;
        GJ(dxv, 2) = a * dlm * cosf(tpv);
        GJ(rdxh, 2) = 0.f;
        GJ(rdxv, 2) = 1.f / GJ(dxv, 2);
        GJ(dare, 2) = GJ(dxv, 2) * dyv * 0.5f * fpole;             // ghost area
        GJ(rare, 2) = 1.f / GJ(dare, 2);
        GJ(wpdar, 2) = -wcor_ * (dyh * dyh) / (dt_ * GJ(dxv, 2) * dyv * 0.5f * fpole) / 100000.00f;
        GJ(curv, 2) = tanf(tpv) / a;
        GJ(fad, 2) = -dt_ / (3.f * GJ(dxv, 2) * dyv * 2.f * 2.f);
        GJ(fah, 2) = -dt_ / (3.f * GJ(dxv, 2) * dyv * 0.5f * fpole);
        GJ(fcp, 2) = dt_ / (3.f * GJ(dxv, 2) * dyv * 0.5f * cp * fpole);
        GJ(fdiv, 2) = 2.f / (3.f * GJ(dxv, 2) * dyv * 0.5f * fpole);
        GJ(hdacxj, 2) = 0.f;
        GJ(hdacyj, 2) = acdt * dyh * dyh / (4.f * GJ(dxv, 2) * dyv * 0.5f * fpole);
        GJ(hdacvxj, 2) = acdt * dyv * dyv / (4.f * GJ(dxv, 2) * dyv);
        GJ(hdacvyj, 2) = acdt * dyv * dyv / (4.f * GJ(dxv, 2) * dyv);
        GJ(ddmpu, 2) = cddamp * dyv / (2.f * GJ(dxv, 2));

        // ---- ghost line beyond the south pole (jds = 1) ----
        GJ(dxh, 1) = GJ(dxh, 3);
        GJ(dxv, 1) = GJ(dxv, 2);
        GJ(rdxh, 1) = GJ(rdxh, 3);
        GJ(rdxv, 1) = GJ(rdxv, 2);
        GJ(dare, 1) = GJ(dare, 3);
        GJ(rare, 1) = GJ(rare, 3);
        GJ(wpdar, 1) = GJ(wpdar, 3);
        GJ(curv, 1) = GJ(curv, 2);
        GJ(fad, 1) = GJ(fad, 2);
        GJ(fah, 1) = GJ(fah, 3);
        GJ(fcp, 1) = GJ(fcp, 3);
        GJ(fdiv, 1) = GJ(fdiv, 3);
        GJ(hdacxj, 1) = GJ(hdacxj, 3);
        GJ(hdacyj, 1) = GJ(hdacyj, 3);
        GJ(hdacvxj, 1) = GJ(hdacvxj, 2);
        GJ(hdacvyj, 1) = GJ(hdacvyj, 2);
        GJ(ddmpu, 1) = GJ(ddmpu, 2);

        // ---- north pole (jde-1 = JM-1) ----
        // The tph/tpv update of the Fortran is not read by any north-pole assignment.
        GJ(dxh, JM - 1) = 0.f;
        GJ(rdxh, JM - 1) = 0.f;
        GJ(dare, JM - 1) = GJ(dxv, JM - 2) * dyv * 0.5f * fpole;
        GJ(rare, JM - 1) = 1.f / GJ(dare, JM - 1);
        GJ(wpdar, JM - 1) = -wcor_ * (dyh * dyh) / (dt_ * GJ(dxv, JM - 2) * dyv * 0.5f * fpole) / 100000.00f;
        GJ(fah, JM - 1) = -dt_ / (3.f * GJ(dxv, JM - 2) * dyv * 0.5f * fpole);
        GJ(fcp, JM - 1) = dt_ / (3.f * GJ(dxv, JM - 2) * dyv * 0.5f * fpole * cp);
        GJ(fdiv, JM - 1) = 2.f / (3.f * GJ(dxv, JM - 2) * dyv * 0.5f * fpole);
        GJ(hdacxj, JM - 1) = 0.f;
        GJ(hdacyj, JM - 1) = acdt * dyh * dyh / (4.f * GJ(dxv, JM - 2) * dyv * 0.5f * fpole);

        // ---- ghost line beyond the north pole (jde = JM) ----
        GJ(dxh, JM) = GJ(dxh, JM - 2);
        GJ(dxv, JM - 1) = GJ(dxv, JM - 2);
        GJ(dxv, JM) = GJ(dxv, JM - 2);
        GJ(rdxh, JM) = GJ(rdxh, JM - 2);
        GJ(rdxv, JM - 1) = GJ(rdxv, JM - 2);
        GJ(rdxv, JM) = GJ(rdxv, JM - 2);
        GJ(dare, JM) = GJ(dare, JM - 2);
        GJ(rare, JM) = GJ(rare, JM - 2);
        GJ(wpdar, JM) = GJ(wpdar, JM - 2);
        GJ(curv, JM - 1) = GJ(curv, JM - 2);
        GJ(curv, JM) = GJ(curv, JM - 2);
        GJ(fad, JM - 1) = GJ(fad, JM - 2);
        GJ(fad, JM) = GJ(fad, JM - 2);
        GJ(fah, JM) = GJ(fah, JM - 2);
        GJ(fcp, JM) = GJ(fcp, JM - 2);
        GJ(fdiv, JM) = GJ(fdiv, JM - 2);
        GJ(hdacxj, JM) = GJ(hdacxj, JM - 2);
        GJ(hdacyj, JM) = GJ(hdacyj, JM - 2);
        GJ(hdacvxj, JM - 1) = GJ(hdacvxj, JM - 2);
        GJ(hdacvyj, JM - 1) = GJ(hdacvyj, JM - 2);
        GJ(hdacvxj, JM) = GJ(hdacvxj, JM - 2);
        GJ(hdacvyj, JM) = GJ(hdacvyj, JM - 2);
        GJ(ddmpu, JM - 1) = GJ(ddmpu, JM - 2);
        GJ(ddmpu, JM) = GJ(ddmpu, JM - 2);
    }

    // ---- scalars (both branches) ----
    scal[0] = dyh;                  // dyh = a*dph
    scal[1] = dyv;                  // dyv = a*dph
    scal[2] = 1.f / dyh;            // rdyh
    scal[3] = 1.f / dyv;            // rdyv
    scal[4] = ddmpv;

    // SW corner, geographic lat/lon (radians)
    float arg = sinf(sb) * s.ctph0 + cosf(sb) * s.stph0 * cosf(wb);
    arg = fminf(arg, 1.f);
    arg = fmaxf(arg, -1.f);
    const float glat_sw = asinf(arg);
    arg = (cosf(sb) * cosf(wb)) / (cosf(glat_sw) * s.ctph0)
          - tanf(glat_sw) * tanf(tph0d * GC_DTR);
    arg = fminf(arg, 1.f);
    arg = fmaxf(arg, -1.f);
    scal[5] = glat_sw;
    scal[6] = tlm0d * GC_DTR + copysignf(1.f, wb) * acosf(arg);   // glon_sw
    scal[7] = 0.f;
}

// ---------------------------------------------------------------------------------------------
// 3. Global only: height-latitude average of the H rows and wind-latitude average of the V rows.
//    One thread per row J; each thread averages its own pair(s), so no two threads write the
//    same element. Fortran: do j=jds,jde/2 (H rows, mirror jde+1-j); do j=jds,(jde-1)/2
//    (V rows, mirror jde-j).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void gc_avg_pair(float* a, int J, int M) {
    const float ave = (GJ(a, J) + GJ(a, M)) * 0.5f;
    GJ(a, J) = ave;
    GJ(a, M) = ave;
}

extern "C" __global__ void gc_pole_average(const int JM, GC_ROW_PARAMS)
{
    const int J = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (J > JM) return;
    if (J <= JM / 2) {                                // do j=jds,jde/2
        const int M = JM + 1 - J;                     // jde+1-j
        gc_avg_pair(dxh, J, M);
        gc_avg_pair(rdxh, J, M);
        gc_avg_pair(dare, J, M);
        gc_avg_pair(rare, J, M);
        gc_avg_pair(wpdar, J, M);
        gc_avg_pair(fah, J, M);
        gc_avg_pair(fcp, J, M);
        gc_avg_pair(fdiv, J, M);
        gc_avg_pair(hdacxj, J, M);
        gc_avg_pair(hdacyj, J, M);
    }
    if (J <= (JM - 1) / 2) {                          // do j=jds,(jde-1)/2
        const int M = JM - J;                         // jde-j
        gc_avg_pair(dxv, J, M);
        gc_avg_pair(rdxv, J, M);
        gc_avg_pair(fad, J, M);
        gc_avg_pair(hdacvxj, J, M);
        gc_avg_pair(hdacvyj, J, M);
        gc_avg_pair(ddmpu, J, M);
    }
}

// ---------------------------------------------------------------------------------------------
// 4. Global only: diagonal distances at V points (s_bdy and n_bdy blocks, and the interior loop
//    do j=max(jms,jds+2),min(jme,jde-3)). One thread per row. Each thread writes only its own row
//    and reads only dxv of its own row, so the launch has no cross-thread ordering.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void gc_global_ddv(GC_SCALAR_PARAMS, GC_ROW_PARAMS,
                                         float* __restrict__ ddv, float* __restrict__ rddv)
{
    const int J = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (J > JM) return;
    const GCS s = gc_scalars(dt, smag2, codamp, wcor, tph0d, tlm0d, sbd, wbd, dphd, dlmd);
    const float dyv = s.dyv;
    // Every write below is the Fortran value computed in its own thread. The Fortran copies
    // (ddv(jds)=ddv(jds+1), ddv(jde-1)=ddv(jde-2)) are the same dyv, so each ghost/boundary
    // thread writes dyv and 1/dyv itself; no thread reads an element another thread writes in
    // this launch (the earlier copy form raced and could copy a stale 0).
    // s_bdy (single process: jts == jds): ddv(jds)=ddv(jds+1)=dyv; rddv = 1/dyv
    if (J == 1 || J == 2) { GJ(ddv, J) = dyv; GJ(rddv, J) = 1.f / dyv; }
    // interior: do j=jds+2, jde-3
    if (J >= 3 && J <= JM - 3) {
        const float dd = sqrtf(GJ(dxv, J) * GJ(dxv, J) + dyv * dyv);
        GJ(ddv, J) = dd;
        GJ(rddv, J) = 1.f / dd;
    }
    // n_bdy (single process: jte == jde): ddv(jde-2)=ddv(jde-1)=dyv; rddv = 1/dyv
    if (J == JM - 2 || J == JM - 1) { GJ(ddv, J) = dyv; GJ(rddv, J) = 1.f / dyv; }
    // Row JM (jde) is not written by the Fortran global branch (its diagonal-distance loop stops
    // at jde-1), so ddv(JM) and rddv(JM) keep the zero the wrapper allocates. No consumer of the
    // ddv/rddv row tables reads row JM (checked: dyn_ddamp_extmode uses its own [JM,IM] rddv
    // scratch, and no other kernel or unit reads grid["ddv"] or grid["rddv"]).
}

// ---------------------------------------------------------------------------------------------
// 5. Two-dimensional fields, one thread per (I, J). Global: f, glat, glon, vlat, vlon, and the
//    hdac* fields equal to the row values. Regional: the same kinds plus the boundary
//    enhancement (bofac) of the hdac* fields. Regional bofac = 4.0 and lnsbc = lnsh (CONSTS);
//    the wrapper passes both. The hdacx/hdacy S and N blocks are guarded by lnsbc >= 2 in the
//    Fortran; the V-point blocks and the W and E blocks are not. Every write in the boundary
//    blocks is hdac*(J)*bofac, so the final value is bofac-scaled exactly when any block covers
//    (I, J), whatever the write order.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void gc_fields_2d(GC_SCALAR_PARAMS, GC_ROW_PARAMS, const int IM,
                                        const int lnsbc, const float bofac,
                                        float* __restrict__ f, float* __restrict__ glat,
                                        float* __restrict__ glon, float* __restrict__ vlat,
                                        float* __restrict__ vlon, float* __restrict__ hdacx,
                                        float* __restrict__ hdacy, float* __restrict__ hdacvx,
                                        float* __restrict__ hdacvy)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;
    const int I = i0 + 1, J = j0 + 1;
    const size_t idx = IDX2(i0, j0, IM);
    // rotated-coordinate index on the reference grid (moving-nest window offset; I, J without a window). The
    // boundary blocks below stay on this grid's own I, J.
    const float Ir = (float)(I + ioff), Jr = (float)(J + joff);

    const GCS s = gc_scalars(dt, smag2, codamp, wcor, tph0d, tlm0d, sbd, wbd, dphd, dlmd);
    const float sb = s.sb, wb = s.wb, dph = s.dph, dlm = s.dlm;
    const float stph0 = s.stph0, ctph0 = s.ctph0, tlm0 = s.tlm0;
    const float twom = NMMB_TWOM;
    const float hx = GJ(hdacxj, J), hy = GJ(hdacyj, J);
    const float hvx = GJ(hdacvxj, J), hvy = GJ(hdacvyj, J);

    if (global_) {
        hdacx[idx] = hx;
        hdacy[idx] = hy;
        hdacvx[idx] = hvx;
        hdacvy[idx] = hvy;

        // coriolis parameter in tll system
        float tph_base = sb - dph - dph * 0.5f;
        float tph = tph_base + Jr * dph;
        float stph = sinf(tph), ctph = cosf(tph);
        float tlm_base = wb - dlm - dlm * 0.5f;
        float tlm = tlm_base + Ir * dlm;
        f[idx] = twom * (ctph0 * stph + stph0 * ctph * cosf(tlm));

        // latitudes and longitudes of h points (global: pole-aware wrap)
        tph_base = sb - dph - dph;
        float aph = tph_base + Jr * dph;
        tlm_base = wb - dlm - dlm;
        float alm = tlm_base + Ir * dlm;
        if (alm > GC_PI) {
            if (I <= IM - 2) alm = alm - GC_PI - GC_PI;
            else             alm = GC_PI + (float)(I - IM + 1) * dlm;
        }
        if (alm < -GC_PI) {
            if (I >= 3) alm = alm + GC_PI + GC_PI;
            else        alm = -GC_PI + (float)(I - 2) * dlm;      // (i-its-1)*dlm, its = 1
        }
        glat[idx] = aph;
        glon[idx] = alm;

        // V-point offset lat/lon
        tph_base = sb - dph - 0.5f * dph;
        aph = tph_base + Jr * dph;
        tlm_base = wb - dlm - 0.5f * dlm;
        alm = tlm_base + Ir * dlm;
        if (alm > GC_PI) alm = alm - GC_PI - GC_PI;
        if (alm < -GC_PI) alm = alm + GC_PI + GC_PI;
        vlat[idx] = aph;
        vlon[idx] = alm;
    } else {
        const int lb = lnsbc;

        // hdacx/hdacy: S and N blocks (lnsbc >= 2 guard), then W and E blocks
        bool bx = (lb >= 2) && (I >= 2 && I <= IM - 1) &&
                  ((J >= 2 && J <= lb) || (J >= JM + 1 - lb && J <= JM - 1));
        bx = bx || (J >= 1 + lb && J <= JM - lb && I >= 2 && I <= lb);                 // W
        bx = bx || (J >= 1 + lb && J <= JM - lb && I >= IM + 1 - lb && I <= IM - 1);   // E
        hdacx[idx] = bx ? hx * bofac : hx;
        hdacy[idx] = bx ? hy * bofac : hy;

        // hdacvx/hdacvy: S, N, W, E blocks (no lnsbc guard)
        const bool bv =
            (J >= 2 && J <= lb && I >= 2 && I <= IM - 2) ||                            // S
            (J >= JM - lb && J <= JM - 2 && I >= 2 && I <= IM - 2) ||                  // N
            (J >= 1 + lb && J <= JM - 1 - lb && I >= 2 && I <= lb) ||                  // W
            (J >= 1 + lb && J <= JM - 1 - lb && I >= IM - lb && I <= IM - 2);          // E
        hdacvx[idx] = bv ? hvx * bofac : hvx;
        hdacvy[idx] = bv ? hvy * bofac : hvy;

        // coriolis parameter in tll system (regional)
        float tph_base = sb - dph * 0.5f;
        float tph = tph_base + Jr * dph;
        float stph = sinf(tph), ctph = cosf(tph);
        float tlm_base = wb - dlm * 0.5f;
        float tlm = tlm_base + Ir * dlm;
        f[idx] = twom * (ctph0 * stph + stph0 * ctph * cosf(tlm));

        // latitudes and longitudes of h points (regional)
        tph_base = sb - dph;
        tph = tph_base + Jr * dph;
        stph = sinf(tph);
        ctph = cosf(tph);
        tlm_base = wb - dlm;
        tlm = tlm_base + Ir * dlm;
        float stlm = sinf(tlm), ctlm = cosf(tlm);
        float sph = ctph0 * stph + stph0 * ctph * ctlm;
        float aph = asinf(sph);
        glat[idx] = aph;
        float anum = ctph * stlm;
        float denom = (ctlm * ctph - stph0 * sph) / ctph0;
        float relm = atan2f(anum, denom);
        float alm = relm + tlm0;
        if (alm > GC_PI) alm = alm - GC_PI - GC_PI;
        if (alm < -GC_PI) alm = alm + GC_PI + GC_PI;
        glon[idx] = alm;

        // V-point offset lat/lon (regional)
        tph_base = sb - 0.5f * dph;
        tph = tph_base + Jr * dph;
        stph = sinf(tph);
        ctph = cosf(tph);
        tlm_base = wb - 0.5f * dlm;
        tlm = tlm_base + Ir * dlm;
        stlm = sinf(tlm);
        ctlm = cosf(tlm);
        sph = ctph0 * stph + stph0 * ctph * ctlm;
        aph = asinf(sph);
        vlat[idx] = aph;
        anum = ctph * stlm;
        denom = (ctlm * ctph - stph0 * sph) / ctph0;
        relm = atan2f(anum, denom);
        alm = relm + tlm0;
        if (alm > GC_PI) alm = alm - GC_PI - GC_PI;
        if (alm < -GC_PI) alm = alm + GC_PI + GC_PI;
        vlon[idx] = alm;
    }
}
