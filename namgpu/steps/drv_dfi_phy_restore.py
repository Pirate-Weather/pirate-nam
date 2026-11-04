"""drv-dfi-phy-restore: DIGITAL_FILTER_PHY_RESTORE_NMM, the physics-state restore after the digital filter.

Fortran: nmm/module_DIGITAL_FILTER_NMM.F90:829-921 (nam.v4.2.11, 83df4e13c1), called from the NMM DFI
driver (RUN_DIGITAL_FILTER_NMM, module_NMM_GRID_COMP.F90) after the filtered dynamics are written back.
Kernel: namgpu/kernels/drv_dfi_phy_restore.cu (dfi_phy_restore_2d, dfi_phy_restore_3d).

Which fields: the physics fields flagged R in parm/nam_filt_vars_norad.txt (restore, not filter), in
file order.  That file is the NAM's own table; the Fortran takes the fields from the ESMF bundle in the
order the bundle holds them, which the NAM driver builds from this table.  With restart=False (every
cold-start forecast, exnam_fcst.sh) the B-flagged fields are filtered, not restored, so they are left
out.  cfg.restart (default False) adds them.

Reads : DFI_PHY_SAVE_2D [N2, JM, IM] and DFI_PHY_SAVE_3D [N3, LM, JM, IM] (written by drv-dfi-phy-save
        with the same name list and order; the values are the pre-DFI copies).
Writes: the R-flagged physics fields in ctx.st (2-D [JM, IM]; 3-D [LDIM, JM, IM] with LDIM = the
        array's own first dimension, lm for CW and NSOIL=4 for STC/SMC/SH2O, as SIZE(HOLD_3D,3)).
Launches: two (all 2-D fields, then all 3-D fields).  Device pointer tables are built once and cached
        in ctx.tables; no host sync, no per-step allocation.
Missing save arrays are an error, never a silent zero restore.
After the restore, the run-total accumulators ACPREC_TOT and CUPREC_TOT are zeroed (the filter's rain is not
forecast rain; see the end of run()).
Dtype and levels are checked on the host before any launch: every target and save array must be
float32 (the kernel reads raw float32), and every 3-D field's LDIM must be 1..lm, because the save
stacks slots with an lm stride and an LDIM above lm would read into the next field's slab.
Halo: the Fortran HALO_EXCH calls after the restore (lines 54 and 86) are dropped. The port's physics
arrays hold only the interior 1..IM x 1..JM (and 1..LDIM levels), with no halo ring in memory, so
the halo values the exchange would fill have no storage to be read from; CONTRACT.md 1a. Multi-card
layouts must add the halo offset and re-document this.
"""
from __future__ import annotations

import contextlib
import re

import numpy as np

from namgpu import kernels
from namgpu.config import PARM

UNIT = "drv_dfi_phy_restore"
SAVE2D = "DFI_PHY_SAVE_2D"
SAVE3D = "DFI_PHY_SAVE_3D"
FILT_TABLE = "nam_filt_vars_norad.txt"
_ROW = re.compile(r"^'([A-Za-z0-9_]+)'\s+([FRB])\b")


def restore_names(restart: bool = False):
    """Names of the restored physics fields, in table order.  Shared with drv-dfi-phy-save."""
    keep = {"R", "B"} if restart else {"R"}
    rows = []
    for line in (PARM / FILT_TABLE).read_text().splitlines():
        m = _ROW.match(line.strip())
        if m and m.group(2) in keep:
            rows.append(m.group(1))
    return rows


def _split(st, names):
    n2, n3 = [], []
    for name in names:
        if name not in st.fields:
            raise RuntimeError(f"drv-dfi-phy-restore: physics field {name} is not in the state")
        ndim = st.fields[name].ndim
        if ndim == 2:
            n2.append(name)
        elif ndim == 3:
            n3.append(name)
        else:
            raise RuntimeError(f"drv-dfi-phy-restore: {name} has rank {ndim}, expected 2 or 3")
    return n2, n3


def run(ctx):
    import cupy as cp

    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    restart = bool(getattr(ctx.cfg, "restart", False))

    names2, names3 = _split(st, restore_names(restart))
    rec = ctx.tables.get("dfi_phy_init")
    if rec is not None and (list(rec["name_save_2d"]) != names2 or list(rec["name_save_3d"]) != names3):
        raise RuntimeError("drv-dfi-phy-restore: slot names differ from drv-dfi-phy-init's record; "
                           "init and restore must use the same restart flag")
    n2, n3 = len(names2), len(names3)
    if n2 == 0 and n3 == 0:
        return

    for name in names2 + names3:
        if st.fields[name].dtype != np.float32:
            raise TypeError(f"drv-dfi-phy-restore: {name} is {st.fields[name].dtype}, expected float32")
    for name in names3:
        ldim = int(st.fields[name].shape[0])
        if not 0 < ldim <= lm:
            raise RuntimeError(f"drv-dfi-phy-restore: {name} has LDIM {ldim}, must be in 1..{lm} (save slot stride)")

    for key, count in ((SAVE2D, n2), (SAVE3D, n3)):
        if count and key not in st.fields:
            raise RuntimeError(f"drv-dfi-phy-restore: {key} missing; drv-dfi-phy-save must run first")

    tables = ctx.tables
    key = (tuple(names2), tuple(names3), tuple(st.fields[x].data.ptr for x in names2 + names3))
    cached = tables.get("dfi_phy_restore")
    if cached is None or cached[0] != key:
        ptr2 = cp.asarray(np.array([st.fields[x].data.ptr for x in names2], dtype=np.uint64))
        ptr3 = cp.asarray(np.array([st.fields[x].data.ptr for x in names3], dtype=np.uint64))
        ldim3 = cp.asarray(np.array([st.fields[x].shape[0] for x in names3], dtype=np.int32))
        cached = (key, ptr2, ptr3, ldim3)
        tables["dfi_phy_restore"] = cached
    _, ptr2, ptr3, ldim3 = cached

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        if n2:
            save2 = st.fields[SAVE2D]
            if save2.dtype != np.float32:
                raise TypeError(f"drv-dfi-phy-restore: {SAVE2D} is {save2.dtype}, expected float32")
            if save2.shape != (n2, jm, im):
                raise RuntimeError(f"drv-dfi-phy-restore: {SAVE2D} shape {save2.shape}, want {(n2, jm, im)}")
            kern2 = kernels.get(UNIT, "dfi_phy_restore_2d")
            kern2(kernels.grid_2d(im, jm, n2), kernels.BLOCK_2D,
                  (save2, ptr2, np.int32(im), np.int32(jm), np.int32(n2)))
        if n3:
            save3 = st.fields[SAVE3D]
            if save3.dtype != np.float32:
                raise TypeError(f"drv-dfi-phy-restore: {SAVE3D} is {save3.dtype}, expected float32")
            if save3.shape != (n3, lm, jm, im):
                raise RuntimeError(f"drv-dfi-phy-restore: {SAVE3D} shape {save3.shape}, want {(n3, lm, jm, im)}")
            kern3 = kernels.get(UNIT, "dfi_phy_restore_3d")
            kern3(kernels.grid_2d(im, jm, n3 * lm), kernels.BLOCK_2D,
                  (save3, ptr3, ldim3, np.int32(im), np.int32(jm), np.int32(lm), np.int32(n3)))
        # Several cards: HALO_EXCH of the restored hold_2d / hold_3d fields (module_DIGITAL_FILTER_NMM.F90:882, 914).
        # One card: no-op.
        from namgpu.steps import drv_halo_exch as _hx
        _hx.exchange_names(ctx, list(names2) + list(names3), "DFI_PHY")
        # The filter's precipitation does not count toward the forecast. The table restores the Fortran's
        # buckets ACPREC and CUPREC to their pre-filter values; the run totals ACPREC_TOT and CUPREC_TOT,
        # which history RAINNC/RAINC are drawn from, are not in the table (CUPREC_TOT is a port addition and
        # the Fortran zeroes ACPREC_TOT with ACPREC at a timestep-0 start, module_SOLVER_GRID_COMP.F90:10307-10318),
        # so the filter's rain stayed in the 0 h totals (RAINNC up to 16 mm at f00, smoke24 11:25Z). Zero them
        # with the buckets.
        for name in ("ACPREC_TOT", "CUPREC_TOT"):
            if name in st.fields:
                st.fields[name].fill(0)
