"""phys-bmj-col-cloudbase: BMJ cloud-base search for every column (max-buoyancy parcel, table indices,
saturation point, cloud base).

Ported from phys/module_CU_BMJ.F90:598-696 (NAM nam.v4.2.11, 83df4e13c1), inside SUBROUTINE bmj.
Kernel: namgpu/kernels/phys_bmj_col_cloudbase.cu, entry phys_bmj_col_cloudbase_column, one thread
per column (grid_cols), one launch, on ctx.stream. The same column code is the __device__ function
bmj_col_cloudbase, which phys-bmj-bmjdrv uses in its per-column path.

Reads (ctx.st, NMMB names, [LM, JM, IM] float32 unless noted):
    T, EXNER              model temperature and Exner function (state tables)
    Q                     specific humidity, raw. The kernel clamps it to max(NMMB_EPSQ, Q) (1e-12),
                          as the bmjdrv qcol does (module_CU_BMJ.F90 line 294), before bmj reads it.
    PHMID                 BMJ midlevel pressure, Fortran prsmid(l) = bmjdrv pcol = phmid(l)
                          (module_CU_BMJ.F90 lines 300-330). phys-bmj-col-prep stages it (or derives
                          it from PINT when the BMJ driver has not run). This wrapper refuses to run
                          without it.
Tables (ctx.tables, BMJ_INIT, phys-bmj-init):
    bmj_ptbl  float32 [JTB=134, ITB=76]   Fortran ptbl(itb,jtb), stored [jtb][itb]
    bmj_qs0   float32 [JTB=134]           Fortran qs0(jtb)
    bmj_sqs   float32 [JTB=134]           Fortran sqs(jtb)
Scalars: lmh = LM (bmj is called with lmh = lm, module_CU_BMJ.F90 line 303), entrain (configfile),
cp = 1004.6 and cappa = R_D/CP (nmmb_common.cuh, the values bmjdrv passes).
Writes (ctx.st, added once with st.add; zero on first use):
    BMJ_THESP, BMJ_PSK, BMJ_RXNERSK   float32 [LM, JM, IM]  per kb of the prep and max-buoy loops
    BMJ_LBOT_KB                       int32   [LM, JM, IM]  cloud base lbot per kb (carried in Fortran)
    BMJ_PBOT_KB                       float32 [LM, JM, IM]  prsmid(lbot) per kb
    BMJ_PELEVFC, BMJ_PBTMX            float32 [JM, IM]
    BMJ_LBOTCNV, BMJ_LTOPCNV          int32   [JM, IM]      both lmh at the end of this unit
    BMJ_CAPECNV, BMJ_PSPCNV, BMJ_THBTCNV  float32 [JM, IM]  zero here; set by the cloud-top unit
BMJ_THESP is zeroed on all LM levels (Fortran line 590), then written for the prep range only.
psk and rxnersk are written for the prep range only (the Fortran exits the loop at prsmid < pelevfc);
the rest keep their previous values, which the downstream reads never use.
BMJ_THESP is also an output name of phys-bmj-col-prep (LEVEL_OUT, zeroed at its lines 587-593). The
two units share one array through st.add, so phys-bmj-col-prep must run before this unit; this unit
then holds the final thesp.

run(ctx) is the done-rule entry. Without a ctx it builds a full-size plausible state (954 x 835 x 60,
random pressures, temperatures and tables) and runs the kernel once. Refuses nothing else silently.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_col_cloudbase"
KERNEL = "phys_bmj_col_cloudbase_column"
BMJ_LM = 60
ITB = 76     # module_CU_BMJ.F90 itb
JTB = 134    # jtb

# NMMB constants passed by bmjdrv (nmmb_common.cuh): CP = 1004.6, CAPPA = R_D/CP, R_D = 287.04.
R_D = np.float32(287.04)
CP = np.float32(1004.6)
CAPPA = np.float32(R_D / CP)

_OUT_3D_F32 = ("BMJ_THESP", "BMJ_PSK", "BMJ_RXNERSK", "BMJ_PBOT_KB")
_OUT_3D_I32 = ("BMJ_LBOT_KB",)
_OUT_2D_F32 = ("BMJ_PELEVFC", "BMJ_PBTMX", "BMJ_CAPECNV", "BMJ_PSPCNV", "BMJ_THBTCNV")
_OUT_2D_I32 = ("BMJ_LBOTCNV", "BMJ_LTOPCNV")


def _outputs(st, lm, jm, im):
    """Scratch outputs, added once with st.add (zero on first use)."""
    outs = {}
    for name in _OUT_3D_F32:
        outs[name] = st.add(name, (lm, jm, im), dtype="float32")
    for name in _OUT_3D_I32:
        outs[name] = st.add(name, (lm, jm, im), dtype="int32")
    for name in _OUT_2D_F32:
        outs[name] = st.add(name, (jm, im), dtype="float32")
    for name in _OUT_2D_I32:
        outs[name] = st.add(name, (jm, im), dtype="int32")
    return outs


def launch(im, jm, lm, lmh, entrain, t, q, exner, prsmid, ptbl, qs0, sqs, outs, stream=None):
    """One launch over every column. t, q, exner, prsmid: [LM, JM, IM] float32 device arrays.
    ptbl [JTB, ITB], qs0 [JTB], sqs [JTB]. outs: the dict from _outputs."""
    f32 = np.float32
    i32 = np.int32
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            i32(im), i32(jm), i32(lmh), i32(1 if entrain else 0),
            CP, CAPPA,
            t, q, exner, prsmid, ptbl, qs0, sqs,
            outs["BMJ_THESP"], outs["BMJ_PSK"], outs["BMJ_RXNERSK"],
            outs["BMJ_LBOT_KB"], outs["BMJ_PBOT_KB"],
            outs["BMJ_PELEVFC"], outs["BMJ_PBTMX"], outs["BMJ_LBOTCNV"], outs["BMJ_LTOPCNV"],
            outs["BMJ_CAPECNV"], outs["BMJ_PSPCNV"], outs["BMJ_THBTCNV"],
        ))
    return outs


def run(ctx=None, *, im=None, jm=None, lm=None, seed: int = 1):
    """Done-rule entry: one launch on a full-size state. Returns the dict of written arrays.

    With a ctx: reads T, Q, EXNER, PHMID from ctx.st, the BMJ tables from ctx.tables, and
    entrain from ctx.cfg.raw. Without one: builds a random full-size state (954 x 835 x 60)."""
    import cupy as cp

    if ctx is None:
        im = 954 if im is None else im
        jm = 835 if jm is None else jm
        lm = BMJ_LM if lm is None else lm
        rng = np.random.default_rng(seed)
        # prsmid increasing with level (level 1 = model top), per-column jitter, sorted in l
        base = np.geomspace(200.0, 1.0e5, lm).astype(np.float32)
        pm = base[:, None, None] * (1.0 + 0.01 * rng.standard_normal((lm, jm, im)).astype(np.float32))
        pm = np.sort(pm.astype(np.float32), axis=0)
        ex = (pm / np.float32(1.0e5)) ** CAPPA
        tt = (220.0 + 85.0 * rng.random((lm, jm, im))).astype(np.float32)
        qq = (1.0e-6 + 0.02 * rng.random((lm, jm, im)) ** 2).astype(np.float32)
        # saturation-pressure-like table: smooth, positive, increasing with the thbt index (jtb)
        jj = np.arange(JTB, dtype=np.float64)[:, None]
        ii = np.arange(ITB, dtype=np.float64)[None, :]
        ptbl = (100.0 * np.exp(0.045 * jj) * (1.0 + 0.3 * ii / ITB)
                * (1.0 + 0.01 * rng.standard_normal((JTB, ITB)))).astype(np.float32)
        qs0 = (0.02 * rng.random(JTB)).astype(np.float32)
        sqs = (1.0e-4 + 1.0e-3 * rng.random(JTB)).astype(np.float32)
        t, q, exner, prsmid = cp.asarray(tt), cp.asarray(qq), cp.asarray(ex.astype(np.float32)), cp.asarray(pm)
        ptbl, qs0, sqs = cp.asarray(ptbl), cp.asarray(qs0), cp.asarray(sqs)
        entrain = False
        class _St(dict):
            def add(self, name, shape, dtype="float32"):
                if name not in self:
                    self[name] = cp.zeros(shape, dtype=dtype)
                return self[name]
        st = _St()
    else:
        cfg, g, st = ctx.cfg, ctx.grid, ctx.st
        im, jm, lm = int(g.im), int(g.jm), int(g.lm)
        if lm != BMJ_LM:
            raise RuntimeError(f"phys-bmj-col-cloudbase: lm={lm}; the kernel is sized for {BMJ_LM} levels")
        if "PHMID" not in st.fields:
            raise RuntimeError("phys-bmj-col-cloudbase: PHMID is not in ctx.st (phys-bmj-col-prep stages it)")
        for key in ("bmj_ptbl", "bmj_qs0", "bmj_sqs"):
            if key not in ctx.tables:
                raise RuntimeError(f"phys-bmj-col-cloudbase: ctx.tables['{key}'] missing (phys-bmj-init)")
        t, q, exner, prsmid = st["T"], st["Q"], st["EXNER"], st["PHMID"]
        ptbl, qs0, sqs = ctx.tables["bmj_ptbl"], ctx.tables["bmj_qs0"], ctx.tables["bmj_sqs"]
        entrain = bool(cfg.raw["entrain"])

    outs = _outputs(st, lm, jm, im)
    stream = getattr(ctx, "stream", None) if ctx is not None else None
    launch(im, jm, lm, lm, entrain, t, q, exner, prsmid, ptbl, qs0, sqs, outs, stream=stream)
    return outs


if __name__ == "__main__":
    out = run()
    import cupy as cp
    bad = {k: int(cp.count_nonzero(~cp.isfinite(v))) for k, v in out.items() if v.dtype == cp.float32}
    print({k: v for k, v in bad.items() if v})
    print("non-finite float outputs:", sum(bad.values()))
