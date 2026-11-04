"""dyn-ddamp-extmode: DDAMP external-mode divergence (vertically integrated) and the U/V
divergence-damping update.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:801-999 (NAM nam.v4.2.11, 83df4e13c1).
Kernels: namgpu/kernels/dyn_ddamp_extmode.cu (ddamp_apd_dive, ddamp_rdd, ddamp_uv).

Gate:   SOLVER_RUN calls DDAMP only under IF(HDIFF_ON>0) (module_SOLVER_GRID_COMP.F90:4176-4177).
        HDIFF_ON is 1 for forward integration (module_NMM_GRID_COMP.F90:3441); the backward
        pass sets it to 0 (:4210). This driver is forward-only, so HDIFF_ON = 1 is a named
        constant here and the wrapper returns without launching when it is 0.
Reads : PD, PDO, DIV [LM,JM,IM], U, V (inout) from ctx.st; SG1, SG2 [LM+1], DSG2 [LM] from ctx.st;
        PDSG1 [LM] from ctx.grid; DDMPU [JM] from ctx.tables["grid"] (init-grid-consts output);
        DDMPV scalar from ctx.cfg codamp/dt and ctx.grid.dyv (module_CONTROL.F90:535, 604).
        FREERUN from ctx.cfg (parm/nam_configfile freerun).
Writes: U, V in ctx.st (updated in place). Scratch DDAMP_APD, DDAMP_DIVE, DDAMP_RDDU, DDAMP_RDDV
        [JM,IM] are added with st.add() on the first call and rewritten every step. The damp
        unit (dyn-ddamp-damp, skip mode) reads DDAMP_APD from here.
Launch: three kernels on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "dyn_ddamp_extmode"
K_APD_DIVE = "ddamp_apd_dive"
K_RDD = "ddamp_rdd"
K_UV = "ddamp_uv"
ASSIMFC = 5.0
HDIFF_ON = 1          # module_NMM_GRID_COMP.F90:3441 forward default; 0 only in the backward pass


def _freerun(cfg) -> bool:
    raw = cfg.raw.get("freerun") if hasattr(cfg, "raw") else None
    if raw is None:
        raise RuntimeError("dyn-ddamp-extmode: configfile has no freerun key (parm/nam_configfile)")
    s = str(raw).strip().lower()
    if s in ("true", ".true.", "t", "1", "free"):
        return True
    if s in ("false", ".false.", "f", "0"):
        return False
    raise ValueError(f"dyn-ddamp-extmode: unrecognised freerun value {raw!r}")


def ddamp_coefficients(ctx):
    """DDMPU [JM] device row and DDMPV host scalar, shared by both ddamp units.

    DDMPU is the init-grid-consts row (ctx.tables["grid"]["ddmpu"], module_CONTROL.F90:626-627).
    DDMPV = cddamp*dyv/(2*dyv) with cddamp = codamp*dt in single precision (module_CONTROL.F90:535,
    604). Cached in ctx.tables so the scalar is computed once and no per-step host sync occurs.
    """
    grid = ctx.tables.get("grid")
    if grid is None or "ddmpu" not in grid:
        raise RuntimeError(
            "dyn-ddamp: ctx.tables[grid][ddmpu] missing; produced by unit init-grid-consts")
    if "ddamp_dvd" not in ctx.tables:
        cddamp = np.float32(ctx.cfg.codamp) * np.float32(ctx.cfg.dt)
        dyv = np.float32(ctx.grid.dyv)
        ctx.tables["ddamp_dvd"] = np.float32(cddamp * dyv / (np.float32(2.0) * dyv))
    return grid["ddmpu"], ctx.tables["ddamp_dvd"]


def run(ctx):
    if HDIFF_ON <= 0:
        return                       # IF(HDIFF_ON>0) gate, module_SOLVER_GRID_COMP.F90:4177
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if g.sg1 is None or g.pdsg1 is None:
        raise RuntimeError(
            "dyn-ddamp-extmode: vertical coordinate not set (grid.set_vcoord); sg1/pdsg1 missing")

    ddmpu, dvd_host = ddamp_coefficients(ctx)
    freerun = _freerun(ctx.cfg)
    fcxm = np.float32(1.0 if freerun else ASSIMFC)   # Fortran: freerun -> 1.0, else assimfc = 5.0
    pdtop = np.float32(g.pdtop)
    dvd = np.float32(dvd_host)
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    apd = st.add("DDAMP_APD", (jm, im))
    dive = st.add("DDAMP_DIVE", (jm, im))
    rddu = st.add("DDAMP_RDDU", (jm, im))
    rddv = st.add("DDAMP_RDDV", (jm, im))

    k_apd = kernels.get(UNIT, K_APD_DIVE)
    k_rdd = kernels.get(UNIT, K_RDD)
    k_uv = kernels.get(UNIT, K_UV)

    i32 = np.int32
    with ctx.stream:
        k_apd(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            i32(im), i32(jm), i32(lm),
            st["PD"], st["PDO"], st["DIV"], apd, dive,
        ))
        k_rdd(kernels.grid_2d(im, jm, 1), kernels.BLOCK_2D, (
            i32(im), i32(jm), i32(lm),
            fcxm, pdtop, dvd,
            st["SG1"], st["SG2"], ddmpu, apd, dive, rddu, rddv,
        ))
        k_uv(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D, (
            i32(im), i32(jm), i32(lm),
            i32(1 if freerun else 0), pdtop, dvd,
            st["SG1"], st["SG2"], st["DSG2"], pdsg1, ddmpu, apd, st["DIV"], rddu, rddv,
            st["U"], st["V"],
        ))


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]
