"""init-vcoord-recover: the NMMB hybrid vertical coordinate for the case.

Kernel: namgpu/kernels/init_vcoord_recover.cu (no NAM Fortran counterpart for the fit; mode "new").

Source of the coordinate, in order:
  1. header: if ctx.tables["vcoord_header"] is present (the NAM input header values PT, PDTOP,
     LPT2, SG1, SG2, DSG1, DSG2, SGML1, SGML2; NAM reads these in module_INIT_READ_BIN.F90:177-185
     and 720-751 and module_INIT_READ_NEMSIO.F90:190-198 and 914-944), the header is used exactly
     as NAM uses it. It is checked for its own consistency (SGML = midpoints of SG) and compared
     with the fit; the comparison is recorded in ctx.tables["vcoord_receipt"].
  2. fit: otherwise PRES(i,j,l) ~= PSGML1(l) + SGML2(l) * PD is fitted by least squares on the
     device over the b1 columns (PD = PSFC - PDTOP - PT), and the interfaces are solved with
     SG(1) = 0 and SG(LM+1) = 1 fixed (grid.vcoord_from_mid), then validated.
     The fit is recorded as the source; nothing in it is silently defaulted.

The GPU port's live path reads GRIB2 (init-read-nemsio is skipped), so no reader currently
supplies ctx.tables["vcoord_header"]; the consumer below is what a NAM-format reader must feed.

Reads : PRES [LM, JM, IM] analysis mid-layer pressure (ctx.st field "PRES", from init-native-atm),
        PSFC [JM, IM] from ctx.st.
Writes: ctx.grid vertical coefficients (pt, pdtop, sg1, sg2, dsg1, dsg2, sgml1, sgml2), ctx.st
        SG1/SG2/DSG1/DSG2/SGML1/SGML2/PSG1/PSGML1/PDSG1 (in place, so every consumer that already
        holds the device array sees the new values), scalar PDTOP, LPT2/EPSQ2 tables.
One device-to-host copy of 2 x LM floats at init (the only sync, before the step loop starts).
"""
from __future__ import annotations

import numpy as np

from namgpu import grid as gridmod
from namgpu import kernels

UNIT = "init_vcoord_recover"
FIT = "init_vcoord_fit_partial"
SOLVE = "init_vcoord_solve"
HEADER_KEY = "vcoord_header"
RECEIPT_KEY = "vcoord_receipt"

# Host tables that depend on the vertical coordinate and are cached in ctx.tables (dropped here so
# the next consumer rebuilds them from the new grid). bmj_vert caches SG2/PSG1/SGML2/PSGML1.
_STALE_TABLE_KEYS = ("pdsg1", "psg1", "dsg1", "dsg2", "sg1", "sg2", "sgml1", "sgml2", "psgml1",
                     "bmj_vert")


def _device_acc(ctx, lm: int):
    import cupy as cp
    key = "vcoord_acc"
    if key not in ctx.tables:
        ctx.tables[key] = cp.zeros((4, lm), dtype=cp.float64)
    return ctx.tables[key]


def _put(st, name: str, host: np.ndarray):
    """Write a host vector into the state field of that name (allocating it if the state does
    not carry it), on the device, in place. In-place matters: consumers such as dyn-adv2-horiz
    (_grid_const) cache the device array under the field name and never re-read it."""
    import cupy as cp
    dev = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    if name in st.fields:
        st.fields[name][...] = dev.reshape(st.fields[name].shape)
    else:
        st.add(name, dev.shape).__setitem__(Ellipsis, dev)


# A layer is pure pressure when its SGML2 is at or below this (fitted values carry float32 noise).
LPT2_TOL = 1e-4


def publish_vcoord_tables(ctx, header_lpt2=None):
    """Publish LPT2 and EPSQ2 for the vertical coordinate in ctx.grid.

    LPT2 (module_INIT_READ_BIN:668, module_INIT_READ_NEMSIO:191/877) is the number of pure-pressure
    layers from the top. NAM reads it from the input header; when a header supplies it, that value
    is used and must be a layer count in 1..LM. Otherwise it is derived from the coordinate: the
    pure-pressure layers must be the top LPT2 layers, anything else raises.
    EPSQ2 is the 2TKE floor per level. module_INIT_READ_BIN:237-240 sets EPSQ2(K-1)=0.02 for K=2..LM
    (the pressure-dependent cosine is commented out) and EPSQ2(LM)=EPSQ2(LM-1), so every level is
    NMMB_EPSQ2 = 0.02 (kernels/nmmb_common.cuh).
    """
    import cupy as cp

    g = ctx.grid
    lm = int(g.lm)
    pure = np.abs(np.asarray(g.sgml2, np.float64)) <= LPT2_TOL
    derived = int(np.argmin(pure)) if not pure.all() else lm
    if pure[derived:].any():
        raise RuntimeError(
            "init-vcoord-recover: pure-pressure layers are not the top LPT2 layers "
            f"(SGML2 pattern {pure.astype(int).tolist()})")
    if header_lpt2 is None:
        lpt2 = derived
    else:
        lpt2 = int(header_lpt2)
        if not 1 <= lpt2 <= lm:
            raise RuntimeError(
                f"init-vcoord-recover: header LPT2={lpt2} is outside 1..LM={lm}")
    ctx.tables["LPT2"] = lpt2
    ctx.tables["EPSQ2"] = cp.full((lm,), 0.02, dtype=cp.float32)
    return lpt2


def _header_arrays(hdr: dict, lm: int):
    """Validate a NAM header dict and return (pt, pdtop, sg1, sg2, sgml1, sgml2, lpt2)."""
    def vec(key: str, size: int) -> np.ndarray:
        if key not in hdr:
            raise RuntimeError(f"init-vcoord-recover: header is missing {key}")
        v = np.asarray(hdr[key], np.float64).ravel()
        if v.size != size:
            raise RuntimeError(f"init-vcoord-recover: header {key} has {v.size} values, expected {size}")
        return v
    pt = float(hdr["PT"])
    pdtop = float(hdr["PDTOP"])
    sg1 = vec("SG1", lm + 1)
    sg2 = vec("SG2", lm + 1)
    sgml1 = vec("SGML1", lm)
    sgml2 = vec("SGML2", lm)
    lpt2 = int(hdr["LPT2"])
    # A header must agree with its own interfaces: SGML is the midpoint of SG (NMMB definition).
    mid1 = 0.5 * (sg1[1:] + sg1[:-1])
    mid2 = 0.5 * (sg2[1:] + sg2[:-1])
    d = max(float(np.max(np.abs(mid1 - sgml1))), float(np.max(np.abs(mid2 - sgml2))))
    if d > 1e-5:
        raise RuntimeError(
            "init-vcoord-recover: header SGML1/SGML2 disagree with the midpoints of its own "
            f"SG1/SG2 (max |diff| {d:.3g}); refusing a corrupt header")
    return pt, pdtop, sg1, sg2, sgml1, sgml2, lpt2


def run(ctx):
    import cupy as cp

    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    pres = st.fields.get("PRES")
    if pres is None:
        raise RuntimeError(
            "init-vcoord-recover: analysis mid-layer pressure st.fields['PRES'] missing "
            "(init-native-atm must fill it from the bgrd3d00 file before this unit)")
    psfc = st["PSFC"]

    pt = float(ctx.cfg.pt)
    pdtop = float(ctx.cfg.ptsgm) - pt              # as grid.vcoord_from_config
    n_b1 = float((im - 2) * (jm - 2))              # b1 columns (CONTRACT.md section 0)

    acc = _device_acc(ctx, lm)
    acc[...] = 0.0
    psgml1_d = cp.zeros((lm,), dtype=cp.float32)
    sgml2_d = cp.zeros((lm,), dtype=cp.float32)

    # PRES and PSFC were written on the caller's (default) stream; ctx.stream is non-blocking and
    # does not wait for that work, so order it first.
    cp.cuda.get_current_stream().synchronize()
    with ctx.stream:
        fit = kernels.get(UNIT, FIT)
        fit(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D, (
            pres, psfc, acc, np.float32(pt), np.float32(pdtop),
            np.int32(im), np.int32(jm), np.int32(lm),
        ))
        solve = kernels.get(UNIT, SOLVE)
        solve(((lm + 127) // 128, 1, 1), (128, 1, 1), (
            acc, np.float64(n_b1), psgml1_d, sgml2_d, np.int32(lm),
        ))
    # The one init-time sync: the host reads below need both kernels finished. (Without it the
    # asnumpy copies raced the non-blocking stream and read partial sums.)
    ctx.stream.synchronize()

    a_fit = cp.asnumpy(psgml1_d).astype(np.float64)
    b_fit = cp.asnumpy(sgml2_d).astype(np.float64)
    if not (np.isfinite(a_fit).all() and np.isfinite(b_fit).all()):
        bad = np.nonzero(~np.isfinite(b_fit))[0]
        raise RuntimeError(
            "init-vcoord-recover: the fit is undetermined because PD is constant over the b1 "
            f"domain (no slope at levels {bad[:8].tolist()}). A constant-PD analysis has no "
            "recoverable hybrid coordinate; the unit refuses rather than writing SG2(LM+1) != 1.")

    hdr = ctx.tables.get(HEADER_KEY)
    if hdr is None and int(getattr(ctx.cfg, "vcoord", 1)) == 1:
        # The NAM coordinate is the one NPS generated from the namelist (vcgenerator); the GRIB2
        # analysis carries no header, so build it from the same namelist (integrator, 2026-11-04).
        from namgpu.init import vcoord_nps  # noqa: PLC0415
        hdr = vcoord_nps.header_from_config(ctx.cfg)
        ctx.tables[HEADER_KEY] = hdr
    receipt = {"lm": lm, "n_b1_columns": int(n_b1)}
    lpt2_hdr = None
    if hdr is None:
        receipt["source"] = "fit"
        # Init-only host step: 60-value interface solve in grid.vcoord_from_mid (validated there).
        gridmod.vcoord_from_mid(g, a_fit, b_fit, pdtop)
    else:
        receipt["source"] = "header"
        pt_h, pdtop_h, sg1_h, sg2_h, sgml1_h, sgml2_h, lpt2_hdr = _header_arrays(hdr, lm)
        # Comparison with the fit, recorded. The header is the coordinate NAM runs on.
        psgml1_h = sgml1_h * pdtop_h + pt_h
        receipt["fit_vs_header_max_abs_psgml1_pa"] = float(np.max(np.abs(psgml1_h - a_fit)))
        receipt["fit_vs_header_max_abs_sgml2"] = float(np.max(np.abs(sgml2_h - b_fit)))
        receipt["header_pdtop_minus_cfg_pa"] = float(pdtop_h - pdtop)
        pt, pdtop = pt_h, pdtop_h
        g.set_vcoord(pt, pdtop, sg1_h, sg2_h)
        gridmod.validate_vcoord(g)
    receipt["pdtop"] = float(pdtop)
    receipt["pt"] = float(pt)
    ctx.tables[RECEIPT_KEY] = receipt

    for name, arr in (("SG1", g.sg1), ("SG2", g.sg2), ("DSG1", g.dsg1), ("DSG2", g.dsg2),
                      ("SGML1", g.sgml1), ("SGML2", g.sgml2), ("PSG1", g.psg1),
                      ("PSGML1", g.psgml1), ("PDSG1", g.pdsg1)):
        _put(st, name, arr)
    st.scalars["PDTOP"] = pdtop
    for key in _STALE_TABLE_KEYS:
        ctx.tables.pop(key, None)
    publish_vcoord_tables(ctx, header_lpt2=lpt2_hdr)
