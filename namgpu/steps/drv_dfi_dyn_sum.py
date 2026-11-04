"""drv-dfi-dyn-sum: DIGITAL_FILTER_DYN_SUM_NMM, add the weighted state each filter step.

Fortran: nmm/module_DIGITAL_FILTER_NMM.F90:208-383 (NAM nam.v4.2.11, 83df4e13c1).
Kernels: namgpu/kernels/drv_dfi_dyn_sum.cu
         dfi_dyn_sum_2d, dfi_dyn_sum_3d, dfi_dyn_sum_4d (one field per launch), dfi_dyn_sum_total,
         dfi_digfil_nondolph (the non-dolph digfil, device side, only when DOLPH is switched off).

Reads the state drv-dfi-dyn-init leaves in ctx.filt (the one copy of the Fortran state):
    nstep                    ndfistep (filter half window, filter steps)
    kstep                    Fortran KSTEP; this wrapper increments it in place (kstep = kstep + 1)
    dolph                    float32[2*nstep+1], dolph_wgts(k) at index k + nstep
    name_save_2d/3d/4d       field names in dyn-init order (the same order as the save arrays)
    tot_rank_2d/3d/4d        counts, checked against the name lists
    sum2d  [n2, JM, IM]      array_save_2d
    sum3d  [n3, LM+1, JM, IM] array_save_3d
    sum4d  [n4, SPEC_MAX, LM, JM, IM] array_save_4d (dyn-init refuses rank-4 fields, so n4 is 0)
    digfil_buf               float32[1], created on first use, only read when DOLPH is off
    totalsum                 float32[1]

Arguments follow the Fortran call (module_DOMAIN_GRID_COMP.F90:5152-5170):
    mean_on                  MEAN_ON
    num_water, num_tracers   NUM_TRACERS_MET, NUM_TRACERS_CHEM (3rd and 4th positional in the
                             Fortran; needed only when a rank-4 field exists)
    its, ite, jts, jte       solver ITS/ITE/JTS/JTE. The kernels cover the full domain, so these
                             must be 1..IM and 1..JM; anything else raises instead of being assumed.

A missing key is an error (no defaults are invented here). Each step: no host sync, no per-step
allocation. The digfil index is a host int; the weight is read on the device.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_dfi_dyn_sum"
K2D = "dfi_dyn_sum_2d"
K3D = "dfi_dyn_sum_3d"
K4D = "dfi_dyn_sum_4d"
KTOT = "dfi_dyn_sum_total"
KDIG = "dfi_digfil_nondolph"
# Fortran sets dolph = .true. (module_DIGITAL_FILTER_NMM.F90:270, "future task: make this a
# configure file item"). The non-dolph branch is kept, as device code, behind this switch.
DOLPH = True


def _state(ctx) -> dict:
    filt = ctx.filt
    if not isinstance(filt, dict) or "kstep" not in filt:
        raise RuntimeError("drv-dfi-dyn-sum: ctx.filt is not set; run drv-dfi-dyn-init first")
    return filt


def _check_domain(g, its, ite, jts, jte):
    im, jm = int(g.im), int(g.jm)
    got = (1 if its is None else int(its), im if ite is None else int(ite),
           1 if jts is None else int(jts), jm if jte is None else int(jte))
    if got != (1, im, 1, jm):
        raise NotImplementedError(
            f"drv-dfi-dyn-sum: loop bounds (its,ite,jts,jte)={got} are not the full domain "
            f"{(1, im, 1, jm)}; the kernels cover the full domain only")


def run(ctx, mean_on=0, num_water=None, num_tracers=None, its=None, ite=None, jts=None, jte=None):
    import cupy as cp

    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    _check_domain(g, its, ite, jts, jte)
    filt = _state(ctx)

    names2 = list(filt["name_save_2d"])
    names3 = list(filt["name_save_3d"])
    names4 = list(filt["name_save_4d"])
    if len(names2) != int(filt["tot_rank_2d"]) or len(names3) != int(filt["tot_rank_3d"]) \
            or len(names4) != int(filt["tot_rank_4d"]):
        raise RuntimeError("drv-dfi-dyn-sum: name lists disagree with the tot_rank counts in ctx.filt")
    if filt["sum2d"].shape[0] != len(names2) or filt["sum3d"].shape[0] != len(names3):
        raise RuntimeError("drv-dfi-dyn-sum: sum arrays do not match the name lists")
    if names4 and (num_water is None or num_tracers is None):
        raise RuntimeError("drv-dfi-dyn-sum: rank-4 filter fields need num_water and num_tracers")

    nstep = int(filt["nstep"])
    filt["kstep"] = int(filt["kstep"]) + 1          # kstep = kstep + 1 (Fortran KSTEP, host int)
    kstep = int(filt["kstep"])

    if DOLPH:
        tab = filt["dolph"]
        idx = kstep + nstep                         # dolph_wgts(kstep) on -nstep:nstep, zero-based
        if not (0 <= idx < tab.shape[0]):
            raise RuntimeError(f"drv-dfi-dyn-sum: kstep {kstep} outside the window (+-{nstep})")
    else:
        tab = filt.get("digfil_buf")
        if tab is None:
            tab = cp.zeros((1,), dtype=cp.float32)
            filt["digfil_buf"] = tab
        idx = 0

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    k2d = kernels.get(UNIT, K2D)
    k3d = kernels.get(UNIT, K3D)
    k4d = kernels.get(UNIT, K4D)
    ktot = kernels.get(UNIT, KTOT)
    with scope:
        if not DOLPH:
            kernels.get(UNIT, KDIG)((1, 1, 1), (1, 1, 1),
                                    (tab, np.int32(kstep), np.int32(nstep), np.int32(mean_on)))

        for n, name in enumerate(names2):
            src = st[name]
            if src.shape != (jm, im):
                raise RuntimeError(f"drv-dfi-dyn-sum: {name} is {src.shape}, want {(jm, im)}")
            k2d(kernels.grid_2d(im, jm, 1), kernels.BLOCK_2D,
                (src, filt["sum2d"][n], tab, np.int32(idx), np.int32(im), np.int32(jm)))

        for n, name in enumerate(names3):
            src = st[name]
            if src.ndim != 3 or src.shape[1:] != (jm, im):
                raise RuntimeError(f"drv-dfi-dyn-sum: {name} is {src.shape}, want [L,{jm},{im}]")
            ldim = int(src.shape[0])                # LDIM = size(hold_3d, dim=3)
            if ldim > filt["sum3d"].shape[1]:
                raise RuntimeError(f"drv-dfi-dyn-sum: {name} has {ldim} levels, sum holds "
                                   f"{filt[sum3d].shape[1]}")
            k3d(kernels.grid_2d(im, jm, ldim), kernels.BLOCK_2D,
                (src, filt["sum3d"][n], tab, np.int32(idx),
                 np.int32(im), np.int32(jm), np.int32(ldim)))

        for n, name in enumerate(names4):
            if name == "TRACERS":
                nspec = int(num_tracers)
            elif name == "WATER":
                nspec = int(num_water)
            else:
                raise RuntimeError(f"drv-dfi-dyn-sum: no NUM_SPEC for 4-D field {name} "
                                   "(Fortran leaves it undefined)")
            src = st[name]
            if src.ndim != 4 or src.shape[1:] != (lm, jm, im):
                raise RuntimeError(f"drv-dfi-dyn-sum: {name} is {src.shape}, want [N,{lm},{jm},{im}]")
            sum4d = filt.get("sum4d")
            if sum4d is None or nspec > src.shape[0] or nspec > sum4d.shape[1]:
                raise RuntimeError(f"drv-dfi-dyn-sum: num_spec {nspec} exceeds {name} storage")
            k4d(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
                (src, sum4d[n], tab, np.int32(idx),
                 np.int32(im), np.int32(jm), np.int32(lm), np.int32(nspec)))

        ktot(kernels.grid_2d(1, 1, 1), (1, 1, 1), (tab, np.int32(idx), filt["totalsum"]))
    return kstep
