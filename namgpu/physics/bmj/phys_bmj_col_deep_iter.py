"""phys-bmj-col-deep-iter: one pass of the BMJ deep-convection cloud-efficiency iteration body.

Ported from phys/module_CU_BMJ.F90:1085-1179 (NAM nam.v4.2.11, 83df4e13c1). The body belongs to
`cloud_efficiency : do itrefi=1,itrefi_max` (1088-1216); the itrefi loop is driven by the caller, one
launch per pass. Lines 1180-1216 belong to other units.
Kernel: namgpu/kernels/phys_bmj_col_deep_iter.cu (phys_bmj_col_deep_iter: one thread per column,
one launch, no host sync, no per-step allocation; scratch psk, rxnersk, thsk cached per shape).

Inputs (device arrays, float32 unless noted; 3-D are [LM, JM, IM] with level 1 = model top):
  per column [JM, IM]:  sm, sm1, pbotfc, efi, depwl, depth, pk0, pkb, pkt,
                        slopbs, slop0s, slopts, slopbl, slop0l, sloptl,
                        dspbss, dsp0ss, dsptss, dspbsl, dsp0sl, dsptsl
  per column int32:     ltop, lb, l0 (Fortran levels, 1-based)
  per level [LM,JM,IM]: pk, qk, tk, dprs, el, rxnerk (read); trefk, qrefk (read and written in place)
Constants (module_CU_BMJ.F90 parameters and module_CONSTANTS): efimn 0.20, pqm 20000, pq0,
a2, a3, a4, elwv, cp, cappa = r_d/cp, rcp = 1/cp, a23m4l = a2*(a3-a4)*elwv.

run(ctx): the done-rule entry. Builds a full-size plausible state on the device (954 x 835 x 60 from
ctx.grid), runs one pass, checks the written arrays for NaN/Inf and returns them. Nothing is written
to ctx.st (the body's inputs come from the deep-ref unit, which is not in the state tables).
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_col_deep_iter"
KERNEL = "phys_bmj_col_deep_iter"

# module_CU_BMJ.F90 parameter efimn=0.20, pqm=20000.; module_CONSTANTS (share): a2, a3, a4, pq0, elwv, cp, r_d
EFIMN = 0.20
PQM = 20000.0
A2 = 17.2693882
A3 = 273.15
A4 = 35.86
PQ0 = 379.90516
ELWV = 2.501e6
CP = 1004.6
R_D = 287.04
CAPPA = R_D / CP
RCP = 1.0 / CP

COLS_2D = ("sm", "sm1", "pbotfc", "efi", "depwl", "depth", "pk0", "pkb", "pkt",
           "slopbs", "slop0s", "slopts", "slopbl", "slop0l", "sloptl",
           "dspbss", "dsp0ss", "dsptss", "dspbsl", "dsp0sl", "dsptsl")
COLS_INT = ("ltop", "lb", "l0")
LEVELS_IN = ("pk", "qk", "tk", "dprs", "el", "rxnerk")
LEVELS_OUT = ("trefk", "qrefk")

_scratch: dict = {}


def _scratch_3d(lm: int, jm: int, im: int):
    """psk, rxnersk, thsk scratch, allocated once per shape."""
    import cupy as cp

    key = (lm, jm, im)
    if key not in _scratch:
        _scratch[key] = tuple(cp.zeros((lm, jm, im), dtype=cp.float32) for _ in range(3))
    return _scratch[key]


def a23m4l() -> np.float32:
    """a23m4l = a2*(a3-a4)*elwv, evaluated in single precision as the Fortran (kfpt=single)."""
    return np.float32(np.float32(A2) * np.float32(np.float32(A3) - np.float32(A4)) * np.float32(ELWV))


def deep_iter(im, jm, lm, *, cols, levels, trefk, qrefk, stream=None):
    """One pass of the iteration body.

    cols:   dict of the per-column arrays named in COLS_2D and COLS_INT.
    levels: dict of the read-only per-level arrays named in LEVELS_IN.
    trefk, qrefk: per-level arrays [lm, jm, im], updated in place.
    Returns (trefk, qrefk, psk, rxnersk, thsk); the last three are scratch that the body also writes.
    """
    k = kernels.get(UNIT, KERNEL)
    psk, rxnersk, thsk = _scratch_3d(lm, jm, im)
    args = [np.int32(im), np.int32(jm), np.int32(lm),
            np.float32(EFIMN), np.float32(PQ0), np.float32(PQM),
            np.float32(A2), np.float32(A3), np.float32(A4),
            np.float32(CAPPA), np.float32(CP), np.float32(RCP), a23m4l()]
    args += [cols[n] for n in COLS_2D]
    args += [cols[n] for n in COLS_INT]
    args += [levels[n] for n in LEVELS_IN]
    args += [trefk, qrefk, psk, rxnersk, thsk]
    k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, tuple(args))
    return trefk, qrefk, psk, rxnersk, thsk


def _plausible(im, jm, lm):
    """Full-size plausible inputs, generated on the device."""
    import cupy as cp

    f32 = cp.float32
    lvl = cp.linspace(2000.0, 101000.0, lm, dtype=f32).reshape(lm, 1, 1)
    pk = (lvl * (1.0 + 0.002 * cp.random.uniform(-1.0, 1.0, (lm, jm, im), dtype=f32))).astype(f32)
    qk = cp.random.uniform(1e-5, 1e-2, (lm, jm, im), dtype=f32)
    tk = cp.random.uniform(220.0, 300.0, (lm, jm, im), dtype=f32)
    dprs = cp.random.uniform(500.0, 2000.0, (lm, jm, im), dtype=f32)
    el = cp.full((lm, jm, im), 2.5e6, dtype=f32)
    rxnerk = (1.0e5 / pk) ** np.float32(CAPPA)
    rxnerk = rxnerk.astype(f32)
    trefk = (tk + cp.random.uniform(-1.0, 1.0, (lm, jm, im), dtype=f32)).astype(f32)
    qrefk = qk.copy()

    J, I = cp.meshgrid(cp.arange(jm), cp.arange(im), indexing="ij")
    ltop = cp.random.randint(5, 26, (jm, im), dtype=cp.int32)
    lb = cp.random.randint(45, 56, (jm, im), dtype=cp.int32)
    l0 = (lb - 5).astype(cp.int32)
    pk0 = pk[l0 - 1, J, I]
    pkb = pk[lb - 1, J, I]
    pkt = pk[ltop - 1, J, I]
    sm = cp.random.uniform(0.0, 1.0, (jm, im), dtype=f32)
    cols = {
        "sm": sm, "sm1": (1.0 - sm).astype(f32),
        "pbotfc": cp.ones((jm, im), dtype=f32),
        "efi": cp.random.uniform(EFIMN, 1.0, (jm, im), dtype=f32),
        "depwl": (pkb - pk0).astype(f32),
        "depth": cp.random.uniform(3000.0, 20000.0, (jm, im), dtype=f32),
        "pk0": pk0.astype(f32), "pkb": pkb.astype(f32), "pkt": pkt.astype(f32),
    }
    for n in ("slopbs", "slop0s", "slopts", "slopbl", "slop0l", "sloptl"):
        cols[n] = cp.random.uniform(-500.0, 500.0, (jm, im), dtype=f32)
    for n in ("dspbss", "dsp0ss", "dsptss", "dspbsl", "dsp0sl", "dsptsl"):
        cols[n] = cp.random.uniform(0.0, 2000.0, (jm, im), dtype=f32)
    cols.update({"ltop": ltop, "lb": lb, "l0": l0})
    levels = {"pk": pk, "qk": qk, "tk": tk, "dprs": dprs, "el": el, "rxnerk": rxnerk}
    return cols, levels, trefk, qrefk


def run(ctx=None):
    """Done-rule entry: one pass on a full-size plausible column state (954 x 835 x 60 from ctx.grid)."""
    import cupy as cp

    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    cols, levels, trefk, qrefk = _plausible(im, jm, lm)
    outs = deep_iter(im, jm, lm, cols=cols, levels=levels, trefk=trefk, qrefk=qrefk)
    for name, a in zip(("trefk", "qrefk", "psk", "rxnersk", "thsk"), outs):
        if not bool(cp.isfinite(a).all()):
            raise FloatingPointError(f"phys-bmj-col-deep-iter: NaN/Inf in {name}")
    return dict(zip(("trefk", "qrefk", "psk", "rxnersk", "thsk"), outs))
