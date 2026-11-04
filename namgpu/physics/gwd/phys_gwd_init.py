"""phys-gwd-init: GWD_init (nmm/module_GWD.F90:32-119, nam.v4.2.11), one launch of the rotation
kernel and one scalar launch for CLEFF.

Kernel: namgpu/kernels/phys_gwd_init.cu (phys_gwd_init_rot, phys_gwd_init_cleff).

Reads:  ctx.st["GLAT"], ctx.st["GLON"]  float32 [JM, IM] radians (H points, SOLVER state table).
        ctx.st["HANGL"]  float32 [JM, IM] (orographic angle from init-gwd-orog, degrees on output).
                         Required: st["HANGL"] must exist (KeyError otherwise). Never allocated here,
                         so a missing init-gwd-orog cannot leave silent zero angles.
        ctx.cfg.tph0d, tlm0d (CEN_LAT, CEN_LON), dphd (DPHD), cleffamp (CLEFFAMP), dt, nphs (DTPHS).
Writes:
  ctx.st["CROT"], ctx.st["SROT"]  float32 [JM, IM] (added to the state via st.add, not in the NAM tables)
  ctx.st["HANGL"]                 converted in place from radians to degrees
  ctx.tables["gwd"], a dict: "nmtvr" (14), "deltim" (DTPHS, float64), "rdeltim" (1/DTPHS, float64),
  both float64 because Fortran declares DELTIM/RDELTIM REAL(KIND=KIND_PHYS), KIND_PHYS=SELECTED_REAL_KIND(13,60)
  (module_GWD.F90:20). DELTIM = double(DTPHS as single); RDELTIM = double(1./DTPHS in single precision),
  i.e. the single-precision division happens before the widening, as in the Fortran.
  "cleff" (device float32 [1], the Fortran CLEFF scalar).

DTPHS is cfg.dt * cfg.nphs (50 s) unless passed explicitly (SOLVER_GRID_COMP passes DTPHY or DTPHS).
Launches: two, on ctx.stream, no host sync, no allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_gwd_init"
NMTVR = 14          # module_GWD.F90:79 (NMTVR=14), number of input orography fields


def run(ctx, dtphs: float | None = None):
    """Set the GWD rotation fields and the GWD scalars; returns ctx.tables["gwd"]."""
    import cupy as cp

    if ctx.tables is None:
        ctx.tables = {}
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    im, jm = int(g.im), int(g.jm)
    shape = (jm, im)

    if dtphs is None:
        dtphs = float(cfg.dt) * int(cfg.nphs)
    dtphs32 = np.float32(dtphs)

    crot = st.add("CROT", shape)
    srot = st.add("SROT", shape)
    hangl = st["HANGL"]
    glat = st["GLAT"]
    glon = st["GLON"]

    tbl = ctx.tables.get("gwd")
    if tbl is None:
        tbl = {"cleff": cp.zeros(1, dtype=np.float32)}
        ctx.tables["gwd"] = tbl
    tbl["nmtvr"] = NMTVR
    tbl["deltim"] = np.float64(dtphs32)
    tbl["rdeltim"] = np.float64(np.float32(1.0) / dtphs32)

    k_rot = kernels.get(UNIT, "phys_gwd_init_rot")
    k_cleff = kernels.get(UNIT, "phys_gwd_init_cleff")
    with scope:
        k_rot(kernels.grid_2d(im, jm, 1), kernels.BLOCK_2D, (
            glat, glon, crot, srot, hangl,
            np.float32(cfg.tph0d), np.float32(cfg.tlm0d), np.int32(im), np.int32(jm)))
        k_cleff((1, 1, 1), (1, 1, 1), (
            tbl["cleff"], np.float32(cfg.raw["cleffamp"]), np.float32(cfg.dphd)))
    return tbl
