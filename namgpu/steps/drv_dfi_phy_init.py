"""drv-dfi-phy-init: DIGITAL_FILTER_PHY_INIT_NMM, the physics save arrays of the digital filter.

Fortran: nmm/module_DIGITAL_FILTER_NMM.F90:650-736 (NAM nam.v4.2.11, 83df4e13c1), called from
DOMAIN_GRID_COMP at module_DOMAIN_GRID_COMP.F90:5064 on the first filter call (FIRST_FILTER).
Kernel: namgpu/kernels/drv_dfi_phy_init.cu (zero_save_2d, zero_save_3d).

Creates, once, the arrays the filter needs to restore physics fields after filtering:
  DFI_PHY_SAVE_2D  [n2, JM, IM]       float32   (array_save_2d_phys, Fortran (ITS:ITE,JTS:JTE,n2))
  DFI_PHY_SAVE_3D  [n3, LM, JM, IM]   float32   (array_save_3d_phys, Fortran (ITS:ITE,JTS:JTE,LM,n3))
These are the names drv-dfi-phy-save writes and drv-dfi-phy-restore reads (CONTRACT section 1, unit-owned
arrays in ctx.st). One 2-D and one 3-D array exist, as in the Fortran.

Bundle membership (Fortran 650-710): the Fortran walks FILT_BUNDLE and counts rank-2 and rank-3 fields by
ESMF dimCount, recording each field name in name_save_2d / name_save_3d (slot N maps back to that name).
The restore bundle is the 'R' rows of parm/nam_filt_vars_norad.txt, plus the 'B' rows when cfg.restart
(FILT_BUNDLE_RESTORE, module_DOMAIN_GRID_COMP.F90:3331-3346 and 6032). This port takes that list from
drv-dfi-phy-restore.restore_names(restart), the same list save and restore use, and takes each field's rank
from the state (the dimCount equivalent). Nothing is hard-coded: a listed field missing from the state is an
error, as it would be an ESMF error in the Fortran.

The name lists and counts are recorded in ctx.tables["dfi_phy_init"] (name_save_2d, name_save_3d,
tot_rank_2d, tot_rank_3d). drv-dfi-phy-restore checks its own list against this record, so a restart flag that
changed between init and restore raises instead of mapping slots to the wrong fields.

Allocation is per array, as in the Fortran: each array is created only if it is absent from ctx.st and its
count is positive (IF (tot_rank_2d > 0 .and. .not. associated(...)) and the same for 3-D). A present array is
kept with its saved values; a present array of the wrong shape or dtype is an error. The 3-D array is sized LM
for every 3-D field (Fortran line 723). Zeroing is one launch per array on ctx.stream, no host sync.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.steps.drv_dfi_phy_restore import restore_names

UNIT = "drv_dfi_phy_init"
NAME_2D = "DFI_PHY_SAVE_2D"
NAME_3D = "DFI_PHY_SAVE_3D"
RECORD = "dfi_phy_init"


def _restart(ctx) -> bool:
    # Same accessor as drv-dfi-phy-save and drv-dfi-phy-restore, so the three lists always agree.
    return bool(getattr(ctx.cfg, "restart", False))


def bundle_fields(ctx):
    """(names2, names3): the restore bundle split by rank, in bundle (table) order."""
    st = ctx.st
    names2, names3 = [], []
    for name in restore_names(_restart(ctx)):
        if name not in st.fields:
            raise RuntimeError(f"drv-dfi-phy-init: bundle field {name} is not in the state")
        ndim = st.fields[name].ndim
        if ndim == 2:
            names2.append(name)
        elif ndim == 3:
            names3.append(name)
        else:
            raise RuntimeError(f"drv-dfi-phy-init: {name} has rank {ndim}, expected 2 or 3")
    return names2, names3


def _check_present(name, arr, shape):
    if arr.dtype != np.float32:
        raise TypeError(f"drv-dfi-phy-init: {name} is {arr.dtype}, expected float32")
    if tuple(arr.shape) != tuple(shape):
        raise RuntimeError(f"drv-dfi-phy-init: {name} shape {tuple(arr.shape)}, want {tuple(shape)}")


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    names2, names3 = bundle_fields(ctx)
    n2, n3 = len(names2), len(names3)

    ctx.tables[RECORD] = {
        "name_save_2d": names2,
        "name_save_3d": names3,
        "tot_rank_2d": n2,
        "tot_rank_3d": n3,
    }

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        if n2 > 0:
            if NAME_2D in st.fields:
                _check_present(NAME_2D, st.fields[NAME_2D], (n2, jm, im))
            else:
                a2 = st.add(NAME_2D, (n2, jm, im), "float32")
                kern = kernels.get(UNIT, "zero_save_2d")
                kern(kernels.grid_2d(im, jm, n2), kernels.BLOCK_2D,
                     (np.int32(im), np.int32(jm), np.int32(n2), a2))
        if n3 > 0:
            if NAME_3D in st.fields:
                _check_present(NAME_3D, st.fields[NAME_3D], (n3, lm, jm, im))
            else:
                a3 = st.add(NAME_3D, (n3, lm, jm, im), "float32")
                kern = kernels.get(UNIT, "zero_save_3d")
                kern(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
                     (np.int32(im), np.int32(jm), np.int32(lm), np.int32(n3), a3))
