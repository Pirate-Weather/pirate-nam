"""init-grid-consts: GRID_CONSTS of NMMB module_CONTROL.F90 (NAM nam.v4.2.11, 83df4e13c1), in full.

Ported from nmm/module_CONTROL.F90:351-1139, both branches (global and regional).
Kernels: namgpu/kernels/init_grid_consts.cu (gc_rows_interior, gc_poles_scalars,
gc_pole_average, gc_global_ddv, gc_fields_2d), launched in that order on ctx.stream.

In:   ctx.cfg: im, jm, dt, smag2, codamp, wcor, tph0d, tlm0d, sbd, wbd, dphd, dlmd (properties),
      lnsh, global_ (config, parm/nam_configfile via namgpu.config).
Out:  ctx.tables['grid'] = dict of device arrays and scalars:
        row [JM] float32: dxh, rdxh, dxv, rdxv, dare, rare, wpdar, curv, fad, fah, fcp, fdiv,
                          ddmpu, ddv, rddv, hdacxj, hdacyj, hdacvxj, hdacvyj (row constants)
        2-D [JM, IM] float32: f, glat, glon, vlat, vlon, hdacx, hdacy, hdacvx, hdacvy
        'scalars': float32[8] device array: 0 dyh, 1 dyv, 2 rdyh, 3 rdyv, 4 ddmpv,
                   5 glat_sw, 6 glon_sw, 7 unused (use the SC_* indices below)
        'flags': {'w_bdy': True, 'e_bdy': True, 's_bdy': True, 'n_bdy': True}
              (single process: its = ids and so on, CONTRACT section 1)
      Compatibility keys read by units that were written before this port existed:
        ctx.tables['CURV'] = curv row [JM]                 (dyn-adv1-coriolis)
        ctx.tables['F_V']  = f [JM, IM]                    (dyn-adv1-coriolis)
      The Fortran bofac (CONSTS) is 4.0 regional and 0.0 global; it is passed to the 2-D kernel.
Scratch: none beyond the output row arrays (init only, runs once).
Launch: five kernels on ctx.stream, no host sync, no Python loop over cells or rows.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "init_grid_consts"
K_ROWS = "gc_rows_interior"
K_POLES = "gc_poles_scalars"
K_AVG = "gc_pole_average"
K_DDV = "gc_global_ddv"
K_2D = "gc_fields_2d"

SC_DYH, SC_DYV, SC_RDYH, SC_RDYV, SC_DDMPV, SC_GLAT_SW, SC_GLON_SW = range(7)

ROW_OUT = ("dxh", "rdxh", "dxv", "rdxv", "dare", "rare", "wpdar", "curv", "fad", "fah",
           "fcp", "fdiv", "ddmpu", "ddv", "rddv")
ROW_HDAC = ("hdacxj", "hdacyj", "hdacvxj", "hdacvyj")
FIELDS_2D = ("f", "glat", "glon", "vlat", "vlon", "hdacx", "hdacy", "hdacvx", "hdacvy")
ROW_BLOCK = 128


def _row_grid(jm: int):
    return ((jm + ROW_BLOCK - 1) // ROW_BLOCK, 1, 1), (ROW_BLOCK, 1, 1)


def run(ctx):
    import cupy as cp

    cfg = ctx.cfg
    im, jm = int(cfg.im), int(cfg.jm)
    is_global = bool(cfg.global_)
    bofac = 0.0 if is_global else 4.0            # CONSTS bofac (module_CONTROL)
    lnsbc = int(cfg.lnsh)                        # CONSTS lnsbc = lnsh

    row = lambda: cp.zeros(jm, dtype=cp.float32)  # noqa: E731
    rows = {n: row() for n in ROW_OUT + ROW_HDAC}
    fields = {n: cp.zeros((jm, im), dtype=cp.float32) for n in FIELDS_2D}
    scal = cp.zeros(8, dtype=cp.float32)

    # common scalar argument list (GC_SCALAR_PARAMS in the .cu, same order)
    common = (
        np.int32(jm), np.int32(1 if is_global else 0),
        np.float32(cfg.dt), np.float32(cfg.smag2), np.float32(cfg.codamp), np.float32(cfg.wcor),
        np.float32(cfg.tph0d), np.float32(cfg.tlm0d), np.float32(cfg.ref_sbd), np.float32(cfg.ref_wbd),
        np.float32(cfg.dphd), np.float32(cfg.dlmd),
        np.int32(cfg.win_i0), np.int32(cfg.win_j0),     # moving-nest window offset from the reference origin (0, 0)
    )
    row_args = tuple(rows[n] for n in (
        "dxh", "rdxh", "dxv", "rdxv", "dare", "rare", "wpdar", "curv", "fad", "fah", "fcp",
        "fdiv", "ddmpu", "hdacxj", "hdacyj", "hdacvxj", "hdacvyj"))

    grid_r, block_r = _row_grid(jm)
    grid_2d = kernels.grid_2d(im, jm, 1)
    block_2d = kernels.BLOCK_2D

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        # 1. interior rows (global J = 3..JM-2; regional all J) and regional ddv/rddv
        kernels.get(UNIT, K_ROWS)(grid_r, block_r,
                                  common + row_args + (rows["ddv"], rows["rddv"]))
        # 2. global poles and ghost lines; scalars and SW corner (one thread)
        kernels.get(UNIT, K_POLES)((1, 1, 1), (1, 1, 1), common + row_args + (scal,))
        if is_global:
            # 3. height-latitude and wind-latitude averages of the row pairs
            kernels.get(UNIT, K_AVG)(grid_r, block_r, (np.int32(jm),) + row_args)
            # 4. diagonal distances at V points (s_bdy, interior, n_bdy)
            kernels.get(UNIT, K_DDV)(grid_r, block_r,
                                     common + row_args + (rows["ddv"], rows["rddv"]))
        # 5. two-dimensional fields and the hdac* boundary enhancement
        kernels.get(UNIT, K_2D)(grid_2d, block_2d,
                                common + row_args + (
                                    np.int32(im), np.int32(lnsbc), np.float32(bofac),
                                    fields["f"], fields["glat"], fields["glon"],
                                    fields["vlat"], fields["vlon"], fields["hdacx"],
                                    fields["hdacy"], fields["hdacvx"], fields["hdacvy"]))

    out = {n: rows[n] for n in ROW_OUT}
    out.update(fields)
    out["hdacxj"], out["hdacyj"] = rows["hdacxj"], rows["hdacyj"]
    out["hdacvxj"], out["hdacvyj"] = rows["hdacvxj"], rows["hdacvyj"]
    out["scalars"] = scal
    out["flags"] = {"w_bdy": True, "e_bdy": True, "s_bdy": True, "n_bdy": True}
    ctx.tables["grid"] = out
    ctx.tables["CURV"] = out["curv"]          # read by dyn-adv1-coriolis
    ctx.tables["F_V"] = out["f"]              # read by dyn-adv1-coriolis
    return out
