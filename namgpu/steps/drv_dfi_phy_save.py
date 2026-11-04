"""drv-dfi-phy-save: copy the physics restore state into the packed save arrays before the digital filter.

Fortran: nmm/module_DIGITAL_FILTER_NMM.F90:741-824, DIGITAL_FILTER_PHY_SAVE_NMM (NAM nam.v4.2.11,
83df4e13c1). Called by module_DOMAIN_GRID_COMP.F90 at the start of the backward TDFI (line 5240)
and at the half-filter time (line 5176 for filter_method 1). The Fortran save runs over
FILT_BUNDLE_RESTORE (module_DOMAIN_GRID_COMP.F90:3331-3346), so its membership is the restore bundle:
the 'R' rows of parm/nam_filt_vars_norad.txt, plus the 'B' rows when RESTART is true
(FILT_TYP_INT==2, module_DOMAIN_GRID_COMP.F90:5893-5900). Cold starts (filter_method 3, the CONTRACT
case) send 'B' to the filter bundle, so they are not saved here.
Kernel: namgpu/kernels/drv_dfi_phy_save.cu (drv_dfi_phy_save, one launch for all fields).

Membership and order come from drv-dfi-phy-restore.restore_names(restart), the same list restore reads,
so the two sides cannot disagree.
Writes: DFI_PHY_SAVE_2D [N2, JM, IM] and DFI_PHY_SAVE_3D [N3, LM, JM, IM] in ctx.st (the names and
    layout drv-dfi-phy-restore reads), allocated once with st.add(). Within each group, field k is
    slot k in list order. A 3-D field with NL < LM (STC, SMC, SH2O: NL = 4) fills levels 0..NL-1 of its
    slot.
Tile: the copy covers i < IM, j < JM. That is the Fortran ITS..ITE, JTS..JTE because the layout is
    single process, halo-free, one card holds the whole domain (CONTRACT.md section 1; its = ids).
Tables: the device pointer tables (src, dst, nlev) are rebuilt only when a source or save array's
    device address changes. The key is the addresses themselves, so a reallocated array or a freed-and-
    reused state can never be copied through a stale pointer. Tables live in ctx.tables.
Launch: one kernel on ctx.stream, no host sync. grid z = nfields * lmax.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.steps.drv_dfi_phy_restore import SAVE2D, SAVE3D, restore_names

UNIT = "drv_dfi_phy_save"
KERNEL = "drv_dfi_phy_save"
TABLE_KEY = "dfi_phy_save"


def save_names(restart: bool = False) -> list[str]:
    """Fields saved before the filter: the restore bundle for this restart flag, in table order."""
    return restore_names(restart)


def _restart(ctx) -> bool:
    # Same accessor as drv-dfi-phy-restore, so the save and restore lists always match.
    return bool(getattr(ctx.cfg, "restart", False))


def _split(st, names):
    n2, n3 = [], []
    for name in names:
        src = st.fields.get(name)
        if src is None:
            raise KeyError(f"drv-dfi-phy-save: restore field {name} is not in the state")
        if src.dtype != np.float32:
            raise TypeError(f"drv-dfi-phy-save: {name} is {src.dtype}, expected float32")
        if src.ndim == 2:
            n2.append(name)
        elif src.ndim == 3:
            n3.append(name)
        else:
            raise ValueError(f"drv-dfi-phy-save: {name} has rank {src.ndim}, need 2 or 3")
    return n2, n3


def _build_tables(ctx, st, names, save2, save3, jm, im, lm, key):
    import cupy as cp

    src_ptr, dst_ptr, nlev = [], [], []
    k2 = k3 = 0
    for name in names:
        src = st.fields[name]
        if src.ndim == 2:
            if tuple(src.shape) != (jm, im):
                raise ValueError(f"drv-dfi-phy-save: {name} shape {src.shape}, want {(jm, im)}")
            dst = save2.data.ptr + k2 * jm * im * 4
            k2 += 1
            nl = 1
        else:
            nl = int(src.shape[0])
            if nl > lm or tuple(src.shape[1:]) != (jm, im):
                raise ValueError(f"drv-dfi-phy-save: {name} shape {src.shape} does not fit (NL<={lm}, {jm}, {im})")
            dst = save3.data.ptr + k3 * lm * jm * im * 4
            k3 += 1
        src_ptr.append(src.data.ptr)
        dst_ptr.append(dst)
        nlev.append(nl)
    hit = (
        key,
        cp.asarray(np.array(src_ptr, dtype=np.uint64)),
        cp.asarray(np.array(dst_ptr, dtype=np.uint64)),
        cp.asarray(np.array(nlev, dtype=np.int32)),
        max(nlev),
    )
    return hit


def run(ctx):
    names = save_names(_restart(ctx))
    if not names:
        return
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    n2, n3 = _split(st, names)

    save2 = save3 = None
    if n2:
        save2 = st.add(SAVE2D, (len(n2), jm, im))
        if tuple(save2.shape) != (len(n2), jm, im):
            raise RuntimeError(f"drv-dfi-phy-save: {SAVE2D} shape {save2.shape}, want {(len(n2), jm, im)}")
    if n3:
        save3 = st.add(SAVE3D, (len(n3), lm, jm, im))
        if tuple(save3.shape) != (len(n3), lm, jm, im):
            raise RuntimeError(f"drv-dfi-phy-save: {SAVE3D} shape {save3.shape}, want {(len(n3), lm, jm, im)}")

    # Address key: rebuilt only when any source or destination address changes.
    key = (tuple(names),
           tuple(st.fields[x].data.ptr for x in names),
           save2.data.ptr if save2 is not None else 0,
           save3.data.ptr if save3 is not None else 0)
    tables = ctx.tables
    hit = tables.get(TABLE_KEY)
    if hit is None or hit[0] != key:
        hit = _build_tables(ctx, st, names, save2, save3, jm, im, lm, key)
        tables[TABLE_KEY] = hit
    _, src_tab, dst_tab, nlev, lmax = hit
    nf = len(names)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, nf * lmax)
    block = kernels.BLOCK_2D
    args = (np.int32(im), np.int32(jm), np.int32(nf), np.int32(lmax),
            src_tab, dst_tab, nlev)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
