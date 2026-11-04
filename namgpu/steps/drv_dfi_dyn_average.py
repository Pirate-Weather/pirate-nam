"""drv-dfi-dyn-average: DIGITAL_FILTER_DYN_AVERAGE, filtered state from the accumulated sums.

Fortran: nmm/module_DIGITAL_FILTER_NMM.F90:388-646 (NAM nam.v4.2.11, 83df4e13c1).
Caller: nmm/module_DOMAIN_GRID_COMP.F90:5195-5202 (backward leg) and 5289 (forward leg).
Kernel: namgpu/kernels/drv_dfi_dyn_average.cu (dfi_scale2d, dfi_scale3d, dfi_scale4d,
dfi_copy2d, dfi_copy3d, dfi_copy4d; one thread per (i, j), level or field index in grid.z).

run(ctx, dfi=None, bounds=None) does, in the Fortran order:
  1. TOTALSUMI = 1/TOTALSUM, single precision;
  2. scale every summed field by TOTALSUMI in place (2-D, then 3-D, then 4-D);
  3. copy each averaged sum into the state field of the same name (2-D, 3-D, 4-D);
  4. reset the bookkeeping the Fortran resets (TOT_RANK_*, KSTEP, NSTEP) so the next leg starts clean.
Halo exchanges are dropped (single card, CONTRACT section 1). See the NAMED GAP note below.

The input is the handoff drv-dfi-dyn-init builds and drv-dfi-dyn-sum fills: ctx.tables["DFI"], with
    names_2d, names_3d, names_4d : field names in the order dyn-init recorded them;
    sum_2d  [N2, JM, IM], sum_3d [N3, LM+1, JM, IM], sum_4d [N4, SPEC_MAX, LM, JM, IM] : the sums;
    totalsum : float32 [1], TOTALSUM accumulated by dyn-sum;
    num_water, num_tracers : NUM_WATER and NUM_TRACERS as the Fortran caller passes them.
There is no default. A missing handoff raises. The earlier random-scratch and totalsum=1.0 fallback
is removed: it made the averages meaningless and was never the production path.

NAMED GAP (owner: coordinator): HALO_EXCH is called after the 2-D and 3-D copies in the Fortran
(lines 527 and 573 of the upper-case copy). Its definition is not in the NAM clone (searched
nam-port/gate0/nmmb/src and nam-port/nam-gh/sorc, no subroutine found). The claim that it writes
nothing on one PE is therefore unwitnessed. Our arrays hold the interior only (no halo), so dropping
it cannot change any stored value, but that is an argument, not a measurement.

Launch: one kernel per routine section (one launch per field for 3-D and 4-D, since LDIM and
NUM_SPEC depend on the field name), all on ctx.stream, no host sync except one read of TOTALSUM per
leg, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_dfi_dyn_average"
K_SCALE2D, K_SCALE3D, K_SCALE4D = "dfi_scale2d", "dfi_scale3d", "dfi_scale4d"
K_COPY2D, K_COPY3D, K_COPY4D = "dfi_copy2d", "dfi_copy3d", "dfi_copy4d"
REQUIRED = ("names_2d", "names_3d", "names_4d", "sum_2d", "sum_3d", "sum_4d",
            "totalsum", "num_water", "num_tracers")


def _launch(ctx, kernel, grid, args):
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kernels.get(UNIT, kernel)(grid, kernels.BLOCK_2D, args)


def _species_count(name, num_tracers, num_water):
    # Fortran: NUM_SPEC is set only for TRACERS and WATER. Any other 4-D name leaves it undefined
    # in the source, so it is refused here instead of guessed.
    if name == "TRACERS":
        return int(num_tracers)
    if name == "WATER":
        return int(num_water)
    raise ValueError(f"drv-dfi-dyn-average: 4-D filter field {name!r} has no NUM_SPEC "
                     "(Fortran leaves it undefined; only TRACERS and WATER are defined)")


def _handoff(ctx, dfi):
    """The DFI handoff from dyn-init and dyn-sum. Explicit dfi wins; otherwise ctx.tables['DFI']."""
    if dfi is None:
        dfi = ctx.tables.get("DFI")
    if dfi is None:
        raise RuntimeError("drv-dfi-dyn-average: no DFI sums in ctx.tables['DFI']; run drv-dfi-dyn-init "
                           "and drv-dfi-dyn-sum first (no random or default sums are substituted)")
    missing = [k for k in REQUIRED if k not in dfi]
    if missing:
        raise RuntimeError(f"drv-dfi-dyn-average: DFI handoff lacks {missing}")
    return dfi


def run(ctx, dfi=None, bounds=None):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    # The Fortran loops run ITS..ITE, JTS..JTE, passed from solver_int_state (module_DOMAIN_GRID_COMP.F90
    # 5195-5202). The kernels cover the whole domain 1..IM, 1..JM, which is what one card holds. Any
    # other tile is refused, not silently treated as the whole domain.
    tile = (1, im, 1, jm) if bounds is None else tuple(int(v) for v in bounds)
    if tile != (1, im, 1, jm):
        raise ValueError(f"drv-dfi-dyn-average: caller tile {tile} is not the whole domain {(1, im, 1, jm)}")

    dfi = _handoff(ctx, dfi)
    names2, names3, names4 = list(dfi["names_2d"]), list(dfi["names_3d"]), list(dfi["names_4d"])
    num_water, num_tracers = int(dfi["num_water"]), int(dfi["num_tracers"])
    sv2, sv3, sv4 = dfi["sum_2d"], dfi["sum_3d"], dfi["sum_4d"]
    spec_max = max(num_tracers, num_water, 1)
    if tuple(sv2.shape) != (len(names2), jm, im):
        raise ValueError(f"drv-dfi-dyn-average: sum_2d {tuple(sv2.shape)} does not match {len(names2)} names")
    if tuple(sv3.shape) != (len(names3), lm + 1, jm, im):
        raise ValueError(f"drv-dfi-dyn-average: sum_3d {tuple(sv3.shape)} does not match {len(names3)} names")
    if tuple(sv4.shape) != (len(names4), spec_max, lm, jm, im):
        raise ValueError(f"drv-dfi-dyn-average: sum_4d {tuple(sv4.shape)} does not match {len(names4)} names")

    if names2 or names3 or names4:
        total = float(dfi["totalsum"][0])
        if total == 0.0:
            raise ValueError("drv-dfi-dyn-average: TOTALSUM is zero (Fortran 1/TOTALSUM would be Inf)")
        totalsumi = np.float32(np.float32(1.0) / np.float32(total))   # TOTALSUMI = 1.0 / TOTALSUM, REAL

        # Section 2: scale the sums in place (Fortran 441-489).
        if names2:
            _launch(ctx, K_SCALE2D, kernels.grid_2d(im, jm, len(names2)),
                    (np.int32(im), np.int32(jm), totalsumi, sv2))
        for n, name in enumerate(names3):
            ldim = lm + 1 if name == "PINT" else lm
            _launch(ctx, K_SCALE3D, kernels.grid_2d(im, jm, ldim),
                    (np.int32(im), np.int32(jm), np.int32(ldim), totalsumi, sv3[n]))
        for n, name in enumerate(names4):
            nspec = _species_count(name, num_tracers, num_water)
            _launch(ctx, K_SCALE4D, kernels.grid_2d(im, jm, lm),
                    (np.int32(im), np.int32(jm), np.int32(lm), np.int32(nspec), totalsumi, sv4[n]))

        # Section 3: copy the averages into the state fields by name (Fortran 491-625).
        for n, name in enumerate(names2):
            dst = st[name]
            if tuple(dst.shape) != (jm, im):
                raise ValueError(f"drv-dfi-dyn-average: {name} has shape {dst.shape}, want {(jm, im)}")
            _launch(ctx, K_COPY2D, kernels.grid_2d(im, jm, 1),
                    (np.int32(im), np.int32(jm), sv2[n], dst))
        for n, name in enumerate(names3):
            dst = st[name]
            ldim = int(dst.shape[0])            # size(HOLD_3D,dim=3)
            if ldim > lm + 1 or tuple(dst.shape[1:]) != (jm, im):
                raise ValueError(f"drv-dfi-dyn-average: {name} has shape {dst.shape}")
            _launch(ctx, K_COPY3D, kernels.grid_2d(im, jm, ldim),
                    (np.int32(im), np.int32(jm), np.int32(ldim), sv3[n], dst))
        for n, name in enumerate(names4):
            dst = st[name]
            nspec = _species_count(name, num_tracers, num_water)
            if int(dst.shape[0]) < nspec or tuple(dst.shape[1:]) != (lm, jm, im):
                raise ValueError(f"drv-dfi-dyn-average: {name} has shape {dst.shape}")
            _launch(ctx, K_COPY4D, kernels.grid_2d(im, jm, lm),
                    (np.int32(im), np.int32(jm), np.int32(lm), np.int32(nspec), sv4[n], dst))

    # Fortran 627-644: deallocate and reset TOT_RANK_*, KSTEP, NSTEP. The sum arrays are CONTRACT
    # scratch owned by dyn-init, so they are kept (not freed); the counts and names are state, so they
    # are reset here. The next leg's dyn-init sets everything again.
    dfi["names_2d"], dfi["names_3d"], dfi["names_4d"] = [], [], []
    dfi["kstep"] = 0
    filt = ctx.filt if isinstance(ctx.filt, dict) else None
    if filt is not None:
        filt.update(tot_rank_2d=0, tot_rank_3d=0, tot_rank_4d=0,
                    name_save_2d=[], name_save_3d=[], name_save_4d=[], kstep=0, nstep=0)
    return ctx
