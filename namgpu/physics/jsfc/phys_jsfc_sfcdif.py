"""phys-jsfc-sfcdif: SFCDIF, Janjic similarity surface layer (sea and land branches, 2 m / 10 m diagnostics).

Ported from phys/module_SF_JSFC.F90:357-1011 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE SFCDIF.
Kernel: namgpu/kernels/phys_jsfc_sfcdif.cu, entry phys_jsfc_sfcdif_column, one thread per column
(kernels.grid_cols), one launch, on ctx.stream, no host sync, no allocation after the first call.

Tables and scalars come from JSFC_INIT (phys-jsfc-init): ctx.tables["jsfc"] holds psim1, psih1,
psim2, psih2 (float32 [KZTM=10001]) and scal = [ztmin1, ztmax1, ztmin2, ztmax2, dzeta1, dzeta2,
fh01, fh02]. The scalars are read to the host once and cached in the dict (scal_host), so the
step path has no per-call device sync.

Reads (ctx.st, [JM, IM] float32 per column, the JSFC driver arrays):
    SEAMASK, THS, PSFC, TZ0, ZSL, PBLH, WETM, Z0BASE, ULOW, VLOW, TLOW, THLOW, THELOW, QLOW,
    CWMLOW, PLOW                            per-column inputs
    QS, UZ0, VZ0, THZ0, QZ0, USTAR, Z0, AKMS, AKHS
                                           INOUT state (written back)
Writes (ctx.st, added once with st.add, zero on first use; [JM, IM] float32):
    CT, RLMO, RIB, CHS, CHS2, CQS2, HFX, QFX, FLX_LH, FLHC, FLQC, QGH, CPM, U10, V10, TH02, TH10,
    Q02, Q10, PSHLTR

Not bound (Fortran arguments the body never uses, see the kernel header): ZSFC, VEGF, SNOC, I, J.
NTSD is the step counter (int); the first-step branches test NTSD == 0.
Constants: module_SF_JSFC PARAMETERs and the NMMB module_CONSTANTS values in nmmb_common.cuh
(compiled into the kernel), so no constant is passed as an argument.

run(ctx) is the done-rule entry. Without a ctx it builds a full-size random state
(954 x 835) with plausible ranges, builds the JSFC tables with phys-jsfc-init, runs the kernel
once and returns the written arrays. A ctx run takes the same path as the step driver.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np

from namgpu import kernels

UNIT = "phys_jsfc_sfcdif"
KERNEL = "phys_jsfc_sfcdif_column"
KZTM = 10001            # module_SF_JSFC.F90:65
JSFC_LM = None          # column kernel: no level dependence

_IN_F32 = ("SEAMASK", "THS", "PSFC", "TZ0", "ZSL", "PBLH", "WETM", "Z0BASE",
           "ULOW", "VLOW", "TLOW", "THLOW", "THELOW", "QLOW", "CWMLOW", "PLOW")
_INOUT_F32 = ("QS", "UZ0", "VZ0", "THZ0", "QZ0", "USTAR", "Z0", "AKMS", "AKHS")
_OUT_F32 = ("CT", "RLMO", "RIB", "CHS", "CHS2", "CQS2", "HFX", "QFX", "FLX_LH", "FLHC",
            "FLQC", "QGH", "CPM", "U10", "V10", "TH02", "TH10", "Q02", "Q10", "PSHLTR")


def _outputs(st, jm, im):
    """Scratch outputs, added once with st.add (zero on first use)."""
    return {name: st.add(name, (jm, im), dtype="float32") for name in _OUT_F32}


def _scal_host(tbl):
    """Host copies of JSFC_INIT's scalars, read once and cached (no per-step sync)."""
    if "scal_host" not in tbl:
        import cupy as cp
        s = cp.asnumpy(tbl["scal"]).astype(np.float32)
        tbl["scal_host"] = tuple(np.float32(x) for x in s)
    return tbl["scal_host"]


def launch(im, jm, step, tbl, arrs, outs, lim, stream=None):
    """One launch over every column; columns outside the B1 tile are skipped in the kernel.

    arrs: dict with every name in _IN_F32 and _INOUT_F32 (device float32 [JM, IM]).
    outs: dict keyed by _OUT_F32 names, values are device float32 [JM, IM] arrays (the caller
          chooses where each output lands; phys-jsfc-jsfc binds them to its state names).
    tbl: ctx.tables['jsfc'] from phys-jsfc-init. lim: ctx.tables['NMMB_LIMITS'] (int32, 1-based).
    """
    f32 = np.float32
    kern = kernels.get(UNIT, KERNEL)
    ztmin1, ztmax1, ztmin2, ztmax2, dzeta1, dzeta2, fh01, fh02 = _scal_host(tbl)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            arrs["SEAMASK"], arrs["THS"], arrs["PSFC"], arrs["TZ0"],
            arrs["ZSL"], arrs["PBLH"], arrs["WETM"], arrs["Z0BASE"],
            arrs["ULOW"], arrs["VLOW"], arrs["TLOW"], arrs["THLOW"],
            arrs["THELOW"], arrs["QLOW"], arrs["CWMLOW"], arrs["PLOW"],
            arrs["QS"], arrs["UZ0"], arrs["VZ0"], arrs["THZ0"], arrs["QZ0"],
            arrs["USTAR"], arrs["Z0"], arrs["AKMS"], arrs["AKHS"],
            *(outs[name] for name in _OUT_F32),
            tbl["psim1"], tbl["psih1"], tbl["psim2"], tbl["psih2"],
            f32(ztmin1), f32(ztmax1), f32(ztmin2), f32(ztmax2),
            f32(dzeta1), f32(dzeta2), f32(fh01), f32(fh02),
            np.int32(step), np.int32(im * jm), np.int32(im), lim,
        ))
    return outs


def _random_state(im, jm, seed):
    """Full-size plausible surface state for the done-rule run (ranges, not a forecast)."""
    import cupy as cp
    rng = np.random.default_rng(seed)

    def u(lo, hi):
        return (lo + (hi - lo) * rng.random((jm, im))).astype(np.float32)

    tlow = u(260.0, 300.0)
    thlow = (tlow * np.float32(1.1)).astype(np.float32)
    qlow = u(1.0e-4, 0.02)
    arrs = {
        "SEAMASK": (rng.random((jm, im)) < 0.3).astype(np.float32),
        "THS": u(275.0, 310.0),
        "PSFC": u(8.0e4, 1.02e5),
        "TZ0": u(270.0, 305.0),
        "ZSL": u(10.0, 60.0),
        "PBLH": u(300.0, 1500.0),
        "WETM": u(0.0, 1.0),
        "Z0BASE": u(1.0e-3, 0.2),
        "ULOW": u(-12.0, 12.0),
        "VLOW": u(-12.0, 12.0),
        "TLOW": tlow,
        "THLOW": thlow,
        "THELOW": thlow.copy(),
        "QLOW": qlow,
        "CWMLOW": u(0.0, 1.0e-4),
        "PLOW": u(5.0e4, 1.0e5),
        "QS": qlow.copy(),
        "UZ0": np.zeros((jm, im), np.float32),
        "VZ0": np.zeros((jm, im), np.float32),
        "THZ0": u(275.0, 310.0),
        "QZ0": qlow.copy(),
        "USTAR": u(0.1, 0.5),
        "Z0": u(0.01, 0.1),
        "AKMS": u(0.005, 0.02),
        "AKHS": u(0.005, 0.02),
    }
    return {k: cp.asarray(v) for k, v in arrs.items()}


def _full_domain_limits(im, jm):
    """Standalone (no ctx) tile limits: the whole domain with the NMMB B1 rule (ids+1 .. ide-1),
    in the drv-loop-limits layout (module_DM_PARALLEL looplimits, 1-based)."""
    import cupy as cp
    from namgpu.steps import drv_loop_limits as dll

    host = np.zeros(dll.NLIM, dtype=np.int32)
    vals = {"ids": 1, "ide": im, "jds": 1, "jde": jm, "its": 1, "ite": im, "jts": 1, "jte": jm,
            "its_b1": 2, "ite_b1": im - 1, "jts_b1": 2, "jte_b1": jm - 1}
    for name, v in vals.items():
        host[dll.NAMES.index(name)] = v
    return cp.asarray(host)


def run(ctx=None, *, im=None, jm=None, step: int = 1, seed: int = 1):
    """Done-rule entry: one launch over every column. Returns the dict of written arrays.

    With a ctx: reads the inputs and INOUT state from ctx.st, the tables from ctx.tables['jsfc']
    (phys-jsfc-init must have run), and writes the INOUT state and the outputs back.
    Without one: a random full-size state, tables built with phys-jsfc-init, one launch."""
    if ctx is None:
        from namgpu.physics.jsfc import phys_jsfc_init

        im = 954 if im is None else im
        jm = 835 if jm is None else jm

        class _St(dict):
            def add(self, name, shape, dtype="float32"):
                import cupy as cp
                if name not in self:
                    self[name] = cp.zeros(shape, dtype=dtype)
                return self[name]

        import cupy as cp
        st = _St()
        arrs = _random_state(im, jm, seed)
        st["USTAR"] = arrs["USTAR"]          # phys-jsfc-init reads and sets USTAR on a cold start
        ictx = SimpleNamespace(tables={}, st=st, grid=SimpleNamespace(im=im, jm=jm), stream=None)
        tbl = phys_jsfc_init.run(ictx)
        arrs["USTAR"] = st["USTAR"]
        stream = None
        lim = _full_domain_limits(im, jm)
    else:
        st, g = ctx.st, ctx.grid
        im, jm = int(g.im), int(g.jm)
        if "jsfc" not in ctx.tables:
            raise RuntimeError("phys-jsfc-sfcdif: ctx.tables['jsfc'] missing (phys-jsfc-init)")
        if "NMMB_LIMITS" not in ctx.tables:
            raise RuntimeError("phys-jsfc-sfcdif: ctx.tables['NMMB_LIMITS'] missing (drv-loop-limits)")
        tbl = ctx.tables["jsfc"]
        lim = ctx.tables["NMMB_LIMITS"]
        arrs = {name: st[name] for name in _IN_F32 + _INOUT_F32}
        stream = getattr(ctx, "stream", None)

    outs = _outputs(st, jm, im)
    launch(im, jm, step, tbl, arrs, outs, lim, stream=stream)
    if ctx is None:
        # return every written array, INOUT state included
        result = dict(outs)
        for name in _INOUT_F32:
            result[name] = arrs[name]
        return result
    return outs


if __name__ == "__main__":
    import cupy as cp

    out = run()
    bad = {k: int(cp.count_nonzero(~cp.isfinite(v))) for k, v in out.items()}
    print("non-finite values by array:", {k: v for k, v in bad.items() if v} or "none")
    print("total non-finite:", sum(bad.values()))
