"""Concurrent and two-way nesting: a 12 km parent and a grid-associated nest stepped together in one process on one card.

    python -m namgpu.nest.twoway --parent-config P --nest-config N --parent-init DIR --nest-init DIR \
        --case 2022-09-26_12z --case-dir CASE --hours 84 --out OUT [--two-way] [--wgt 1.0]
    (the door: woof-nam nest --two-way | --concurrent ..., namgpu/nest/cli.py)

What NMMB does and this port follows (nam.v4.2.11, 83df4e13c1; NMMB core, US government code):
  geometry   module_NESTING.F90:4263-4592 SET_NEST_GRIDS: the nest uses the parent's pole (TPH0D/TLM0D), dlmd/dphd =
             the parent's / ratio, its SW H point on a parent H point (I_PARENT_START, J_PARENT_START)
             (namgpu/nest/domain.py nest_grid_associated).
  step order module_NMM_INTEGRATE.F90:540-700, 774, 1017, 1035-1060 (coupler phases 1-5): per parent step,
             (3) the parent takes the nest's two-way data from the end of the previous parent step, (step) the parent
             integrates, (4) it gives the nest boundary values valid at the end of its step, (2) the nest takes them
             and runs RATIO steps, (5) the nest makes its two-way data.
  boundary   module_PARENT_CHILD_CPL_COMP.F90:15871-16761 PARENT_TO_CHILD_INTERP_FACTORS (bilinear factors, here
             once, on the host), 20489-21029 PARENT_UPDATE_CHILD_PSFC, 21030-21680 PARENT_UPDATE_CHILD_BNDRY,
             21681-22031 PRESSURE_ON_NEST_BNDRY_V, and module_SOLVER_GRID_COMP.F90:8155-9298 UPDATE_BC_TENDS:
             kernels/nest_bc.cu. Fields PD, T, Q, CW (H) and U, V (V) (parm/nam_nests.txt).
  two-way    module_NESTING.F90:12355-12670 CHILD_2WAY_BOOKKEEPING (update points, here once), 12671-12904
             GENERATE_2WAY_DATA, module_PARENT_CHILD_CPL_COMP.F90:22032-22458 PARENT_2WAY_UPDATE: kernels/nest_2way.cu.
             Fields T, Q, CW (H) and U, V (V); PD is not fed back. Weight 2way_wgt (NMMB's own 2-way test: 1.0).
Not ported: the NEMS/ESMF framework and the MPI exchange between task sets (one process needs no messages; the LGPL
NEMS tree was not read); moving nests (COMPUTE_STORM_MOTION, MOVING_NEST_BOOKKEEPING; the NAM nests did not move).
Differences from NAM, stated: each domain runs its own TDFI before the coupled loop (the nest's filter takes its own
start-source boundaries); the nest starts from its own source analysis at its own resolution (NAM's input_ready: true
shape, without NAM's cycled 3 km first guess and GSI). The operational NAM ran its nests concurrent one-way
(nest_mode: 1-way in all 7 NAM configfiles): --concurrent is that shape, --two-way adds NMMB's feedback.

Output: <out>/parent/wrfout_d01_* and <out>/nest/wrfout_d01_* (each folder is an ordinary WOOF-NAM run folder, so the
maps, column store and sheets work on it as on any run), <out>/twoway-receipt.json (ms per parent step with the
nest's steps, per stage: parent, nest, boundary, two-way).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import traceback
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

UNIT_BC, UNIT_2WAY = "nest_bc", "nest_2way"
BLOCK = 128
EPS = np.float32(1.0e-4)                       # module_NESTING.F90 EPS, PARENT_TO_CHILD_INTERP_FACTORS EPS
# module_NESTING.F90:107-114: (h->H, v->V, sfc->H, sfc->V)
STENCILS = {"even": (3, 2, 3, 3), "odd": (3, 3, 3, 2)}
H_BC, V_BC = ("PD", "T", "Q", "CW"), ("U", "V")
FEEDBACK = ("T", "Q", "CW", "U", "V")          # kernel buffer order (nest_2way.cu)


# ---------------------------------------------------------------------------------------------- host tables (once)

def interp_factors(i_ps: int, j_ps: int, ratio: int, ic, jc, v_points: bool):
    """PARENT_TO_CHILD_INTERP_FACTORS (CPL 15871-16761) for nest points (ic, jc) (0-based nest indices).
    Returns parent 0-based SW indices (iw, js) on the parent H grid (H points) or V grid (V points) and the
    bilinear weights [n, 4] (SW, SE, NW, NE), in single precision as the Fortran REAL."""
    f = np.float32
    rcp = f(1.0) / f(ratio)                                          # CHILD_PARENT_SPACE_RATIO
    if v_points:
        i0 = f(i_ps) - f(0.5) + rcp * f(0.5)                         # PARENT_I_CHILD_WBND, V_POINTS
        j0 = f(j_ps) - f(0.5) + rcp * f(0.5)
    else:
        i0, j0 = f(i_ps), f(j_ps)
    ri = (i0 + np.asarray(ic, np.float32) * rcp).astype(np.float32)  # REAL_I_PARENT, (I_CHILD-1) = ic
    rj = (j0 + np.asarray(jc, np.float32) * rcp).astype(np.float32)
    iw1 = np.floor(ri + EPS).astype(np.int32)                         # INT(REAL+EPS), positive: floor
    js1 = np.floor(rj + EPS).astype(np.int32)
    ie1, jn1 = iw1 + 1, js1 + 1
    w = np.empty((len(ri), 4), np.float32)
    w[:, 0] = (ie1.astype(np.float32) - ri) * (jn1.astype(np.float32) - rj)
    w[:, 1] = (ri - iw1.astype(np.float32)) * (jn1.astype(np.float32) - rj)
    w[:, 2] = (ie1.astype(np.float32) - ri) * (rj - js1.astype(np.float32))
    w[:, 3] = (ri - iw1.astype(np.float32)) * (rj - js1.astype(np.float32))
    s = w.sum(axis=1, dtype=np.float32)
    w = (w * (np.float32(1.0) / s)[:, None]).astype(np.float32)      # RECIP_SUM
    return iw1 - 1, js1 - 1, w


def strip_points(im: int, jm: int, width: int, v_points: bool):
    """The nest boundary strip points in the ctx.bc layout (bnd-boundary-init): side 0 S [w][IM], 1 N [w][IM],
    2 W [JM][w], 3 E [JM][w]. V strips stop one short of IM / JM (the padding column/row) and their N/E strips start
    at JM-1-w / IM-1-w. Returns ic, jc, side, off (offset in the side's level-0 plane), lstr[4]."""
    ic, jc, sd, off = [], [], [], []
    ni, nj = (im - 1, jm - 1) if v_points else (im, jm)
    n0 = (jm - 1 - width) if v_points else (jm - width)
    e0 = (im - 1 - width) if v_points else (im - width)
    for s in range(width):
        i = np.arange(ni)
        for side, j in ((0, s), (1, n0 + s)):
            ic.append(i); jc.append(np.full(ni, j)); sd.append(np.full(ni, side)); off.append(s * im + i)
        j = np.arange(nj)
        for side, ii in ((2, s), (3, e0 + s)):
            ic.append(np.full(nj, ii)); jc.append(j); sd.append(np.full(nj, side)); off.append(j * width + s)
    cat = lambda xs, t: np.concatenate(xs).astype(t)  # noqa: E731
    lstr = np.array([width * im, width * im, jm * width, jm * width], np.int64)
    return cat(ic, np.int32), cat(jc, np.int32), cat(sd, np.int32), cat(off, np.int64), lstr


def ring_points(im: int, jm: int, width: int):
    """Nest H points within width+1 rows/columns of the edge: every H point a boundary H strip or a V strip's
    4-point PD mean reads (PARENT_UPDATE_CHILD_PSFC computes N_BLEND+1 rows for the same reason)."""
    j, i = np.mgrid[0:jm, 0:im]
    m = (i <= width) | (i >= im - 1 - width) | (j <= width) | (j >= jm - 1 - width)
    return i[m].astype(np.int32), j[m].astype(np.int32)


def update_points(i_ps: int, j_ps: int, ratio: int, im_c: int, jm_c: int, n_blend: int, im_p: int, jm_p: int):
    """CHILD_2WAY_BOOKKEEPING (module_NESTING.F90:12355-12670), one task each: the parent H/V points (1-based I, J)
    every stencil of which lies inside the nest minus its N_BLEND boundary rows. J outer, I inner (KOUNT order)."""
    f = np.float32
    rr = f(1.0) / f(ratio)
    ids, jds = f(i_ps), f(j_ps)
    ide = ids + f(im_c - 1) * rr
    jde = jds + f(jm_c - 1) * rr
    i_s, i_e, j_s, j_e = ids, ide, jds, jde
    for n in STENCILS["even" if ratio % 2 == 0 else "odd"]:
        n0 = n // 2
        i_s = max(i_s, ids, ids + f(n_blend + n0) * rr)
        i_e = min(i_e, ide, ide - f(n_blend + n0) * rr)
        j_s = max(j_s, jds, jds + f(n_blend + n0) * rr)
        j_e = min(j_e, jde, jde - f(n_blend + n0) * rr)
    i1, i2 = max(1, int(i_s + f(1.0) - EPS)), min(im_p, int(i_e + EPS))
    j1, j2 = max(1, int(j_s + f(1.0) - EPS)), min(jm_p, int(j_e + EPS))
    jj, ii = np.mgrid[j1:j2 + 1, i1:i2 + 1]
    return ii.ravel().astype(np.int32), jj.ravel().astype(np.int32), (i1, i2, j1, j2)


# ---------------------------------------------------------------------------------------------- the coupler

class Coupler:
    """Device tables and launches for one parent/nest pair. Every launch is on the (shared) stream; nothing syncs."""

    def __init__(self, ctx_p, ctx_c, ratio: int, i_ps: int, j_ps: int, two_way: bool, wgt: float = 1.0):
        import cupy as cp
        from namgpu import kernels
        self.cp, self.kernels = cp, kernels
        self.p, self.c = ctx_p, ctx_c
        self.ratio, self.i_ps, self.j_ps = int(ratio), int(i_ps), int(j_ps)
        self.two_way, self.wgt = bool(two_way), float(wgt)
        gp, gc = ctx_p.grid, ctx_c.grid
        self.imp, self.jmp, self.lm = int(gp.im), int(gp.jm), int(gp.lm)
        self.imc, self.jmc = int(gc.im), int(gc.jm)
        if int(gc.lm) != self.lm:
            raise ValueError(f"two-way nest: parent LM {self.lm} != nest LM {gc.lm} (NMMB nests share the levels)")
        if self.lm > 64:
            raise ValueError(f"two-way nest: LM {self.lm} > NEST_LMAX 64 (kernels/nest_spline.cuh)")
        stp, stc = ctx_p.st, ctx_c.st
        for k in ("SG1", "SG2", "PSGML1", "SGML2"):
            a, b = cp.asnumpy(stp[k]), cp.asnumpy(stc[k])
            if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-6):
                raise ValueError(f"two-way nest: parent and nest {k} differ (NMMB nests share the vertical coordinate)")
        # PT and PDTOP from the grid's vertical coordinate (set by the init and restored by slab_init.load); the state
        # scalars PT/PDTOP are not filled on the GFS-start path (smoke 2026-11-05: 0, so the top interfaces read p = 0)
        if abs(float(gp.pt) - float(gc.pt)) > 1e-3 or abs(float(gp.pdtop) - float(gc.pdtop)) > 1e-2:
            raise ValueError(f"two-way nest: parent PT/PDTOP {gp.pt}/{gp.pdtop} != nest {gc.pt}/{gc.pdtop}")
        self.pdtop = np.float32(gp.pdtop)
        self.pt = np.float32(gp.pt)
        lnsh, lnsv = int(ctx_c.cfg.lnsh), int(ctx_c.cfg.lnsv)
        self.recip = np.float32(1.0) / np.float32(np.float32(ctx_c.cfg.dt) * np.float32(self.ratio))

        def dev(x):
            return cp.asarray(np.ascontiguousarray(x))
        # parent -> nest boundary tables
        ri, rj = ring_points(self.imc, self.jmc, max(lnsh, lnsv))
        iw, js, w = interp_factors(i_ps, j_ps, ratio, ri, rj, False)
        self._check_parent(iw, js, 1, "boundary H")
        self.ring = [dev(ri), dev(rj), dev(iw), dev(js), dev(w), len(ri)]
        hi, hj, hs, ho, hl = strip_points(self.imc, self.jmc, lnsh, False)
        iw, js, w = interp_factors(i_ps, j_ps, ratio, hi, hj, False)
        self.h = [dev(hi), dev(hj), dev(iw), dev(js), dev(w), dev(hs), dev(ho), dev(hl), len(hi)]
        vi, vj, vs, vo, vl = strip_points(self.imc, self.jmc, lnsv, True)
        iw, js, w = interp_factors(i_ps, j_ps, ratio, vi, vj, True)
        self._check_parent(iw, js, 2, "boundary V")
        self.v = [dev(vi), dev(vj), dev(iw), dev(js), dev(w), dev(vs), dev(vo), dev(vl), len(vi)]
        self.pdb = cp.zeros((self.jmc, self.imc), cp.float32)
        self._bind_strips()
        # two-way tables
        self.npts = 0
        self.region = None
        if self.two_way:
            st = STENCILS["even" if ratio % 2 == 0 else "odd"]
            self.nst_h, self.nst_v, self.nst_sfc = st[0], st[1], st[2]
            pi, pj, self.region = update_points(i_ps, j_ps, ratio, self.imc, self.jmc, lnsh, self.imp, self.jmp)
            ich = (pi - i_ps) * ratio                  # (I - I_SW)*RATIO + 1, 0-based
            jch = (pj - j_ps) * ratio
            self._check_stencils(ich, jch)
            self.npts = len(pi)
            self.up = [dev(pi - 1), dev(pj - 1), dev(ich), dev(jch)]
            self.buf = cp.zeros((len(FEEDBACK), self.lm, self.npts), cp.float32)
            self.sfc = cp.zeros((self.npts, 2), cp.float32)
        self.have_feedback = False

    # -- checks
    def _check_parent(self, iw, js, extra, what):
        lo = min(int(iw.min()), int(js.min()))
        if lo < 0 or int(iw.max()) + extra >= self.imp or int(js.max()) + extra >= self.jmp:
            raise ValueError(f"two-way nest: {what} interpolation leaves the parent ({iw.min()}..{iw.max()} of "
                             f"{self.imp}, {js.min()}..{js.max()} of {self.jmp})")

    def _check_stencils(self, ich, jch):
        half_v = self.ratio // 2
        lo_h, hi_h = self.nst_h // 2, self.nst_h - 1 - self.nst_h // 2
        lo_v = self.nst_v // 2
        bad = (ich.min() - lo_h < 0 or jch.min() - lo_h < 0 or ich.max() + hi_h >= self.imc or jch.max() + hi_h >= self.jmc
               or ich.min() + half_v - lo_v < 0 or ich.max() + half_v - lo_v + self.nst_v - 1 >= self.imc - 1
               or jch.max() + half_v - lo_v + self.nst_v - 1 >= self.jmc - 1)
        if bad:
            raise ValueError("two-way nest: a two-way stencil leaves the nest (CHILD_2WAY_BOOKKEEPING region)")

    def _bind_strips(self):
        """Slot-0 base pointers of the nest's boundary strips, [var*4 + side]. Rebound when ctx.bc is replaced."""
        cp, bc = self.cp, self.c.bc
        sides = "SNWE"
        for name in H_BC + V_BC:
            for s in sides:
                a = bc[f"{s}_{name}"]
                if a.dtype != cp.float32 or not a.flags.c_contiguous:
                    raise ValueError(f"two-way nest: ctx.bc {s}_{name} is not a dense float32 strip")
        self._bc_id = {k: id(v) for k, v in bc.items()}
        self.hptr = cp.asarray(np.array([bc[f"{s}_{n}"].data.ptr for n in H_BC for s in sides], np.uint64))
        self.vptr = cp.asarray(np.array([bc[f"{s}_{n}"].data.ptr for n in V_BC for s in sides], np.uint64))

    # -- launches
    def boundary(self):
        """Parent state at the end of its step -> nest boundary tendencies over the next RATIO nest steps."""
        cp, K = self.cp, self.kernels
        p, c = self.p.st, self.c.st
        if {k: id(v) for k, v in self.c.bc.items()} != self._bc_id:
            self._bind_strips()
        f32, i32 = np.float32, np.int32
        ic, jc, iw, js, w, n = self.ring
        K.get(UNIT_BC, "nest_bc_psfc")(((n + BLOCK - 1) // BLOCK,), (BLOCK,), (
            p["PD"], p["FIS"], p["T"], p["Q"], i32(self.imp), i32(self.jmp), i32(self.lm),
            p["SG1"], p["SG2"], f32(self.pdtop), f32(self.pt), c["FIS"], i32(self.imc),
            i32(n), ic, jc, iw, js, w, self.pdb))
        ic, jc, iw, js, w, sd, off, ls, n = self.h
        K.get(UNIT_BC, "nest_bc_h")(((n + BLOCK - 1) // BLOCK,), (BLOCK,), (
            p["PD"], p["T"], p["Q"], p["CW"], i32(self.imp), i32(self.jmp), i32(self.lm),
            p["SG1"], p["SG2"], c["PSGML1"], c["SGML2"], f32(self.pdtop), f32(self.pt),
            self.pdb, i32(self.imc), i32(n), ic, jc, iw, js, w, sd, off, self.hptr, ls, f32(self.recip)))
        ic, jc, iw, js, w, sd, off, ls, n = self.v
        K.get(UNIT_BC, "nest_bc_v")(((n + BLOCK - 1) // BLOCK,), (BLOCK,), (
            p["PD"], p["U"], p["V"], i32(self.imp), i32(self.jmp), i32(self.lm),
            p["SG1"], p["SG2"], c["PSGML1"], c["SGML2"], f32(self.pdtop), f32(self.pt),
            self.pdb, i32(self.imc), i32(n), ic, jc, iw, js, w, sd, off, self.vptr, ls, f32(self.recip)))

    def generate(self):
        """GENERATE_2WAY_DATA at the end of the nest's RATIO steps (coupler phase 5)."""
        if not self.two_way:
            return
        K, c, i32 = self.kernels, self.c.st, np.int32
        _pi, _pj, ich, jch = self.up
        gx = (self.npts + BLOCK - 1) // BLOCK
        K.get(UNIT_2WAY, "nest_2way_gen")((gx, self.lm, len(FEEDBACK)), (BLOCK,), (
            c["T"], c["Q"], c["CW"], c["U"], c["V"], i32(self.imc), i32(self.jmc), i32(self.lm),
            i32(self.npts), ich, jch, i32(self.ratio // 2), i32(self.nst_h), i32(self.nst_v), self.buf))
        K.get(UNIT_2WAY, "nest_2way_sfc")((gx,), (BLOCK,), (
            c["FIS"], c["PD"], i32(self.imc), i32(self.npts), ich, jch, i32(self.nst_sfc), self.sfc))
        self.have_feedback = True

    def update(self):
        """PARENT_2WAY_UPDATE at the start of the parent's step (coupler phase 3), from the last generate()."""
        if not (self.two_way and self.have_feedback):
            return
        K, p, i32, f32 = self.kernels, self.p.st, np.int32, np.float32
        pi, pj, _ich, _jch = self.up
        K.get(UNIT_2WAY, "nest_2way_update")(((self.npts + BLOCK - 1) // BLOCK,), (BLOCK,), (
            p["T"], p["Q"], p["CW"], p["U"], p["V"], p["PD"], p["FIS"], i32(self.imp), i32(self.jmp),
            i32(self.lm), p["SG1"], p["SG2"], f32(self.pdtop), f32(self.pt), i32(self.npts), pi, pj,
            self.buf, self.sfc, f32(self.wgt)))

    def receipt(self) -> dict:
        r = {"ratio": self.ratio, "i_parent_start": self.i_ps, "j_parent_start": self.j_ps, "two_way": self.two_way,
             "boundary_points": {"psfc_ring": self.ring[-1], "h": self.h[-1], "v": self.v[-1]}}
        if self.two_way:
            r.update(wgt_child=self.wgt, update_points=self.npts, update_region_parent_1based=self.region,
                     stencils={"h": self.nst_h, "v": self.nst_v, "sfc": self.nst_sfc})
        return r


# ---------------------------------------------------------------------------------------------- the run

def _ctx(cfgfile: Path, init_dir: Path, stream, out: Path, case_dir: Path, start: datetime, source: str, hours: float):
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import slab_init
    from namgpu.steps import drv_loop_limits
    os.environ[config.CONFIG_ENV] = str(cfgfile)      # State() and the loaders read the default configfile
    cfg = config.load(cfgfile)
    g = gridmod.build(cfg)
    st = statemod.State(g, names={k for k, v in statemod.all_specs().items() if v.rank == 0})
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream)
    load = slab_init.load(ctx, init_dir)
    ctx.slab = ctx.comm = ctx.decomp = None
    ctx.case_dir, ctx.start, ctx.start_time, ctx.init_source = case_dir, start, start, source
    ctx.out_dir = out
    out.mkdir(parents=True, exist_ok=True)
    ctx.forecast_hours = max(6, int(hours + 0.999))
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    drv.check_ready()
    with stream:
        drv_loop_limits.run(ctx)
    return ctx, drv, load


def _ratio(cfg_p, cfg_c) -> int:
    def dt(c):
        r = c.raw
        return Fraction(int(r["dt_int"])) + (Fraction(int(r["dt_num"]), int(r["dt_den"])) if int(r["dt_den"]) else 0)
    q = dt(cfg_p) / dt(cfg_c)
    if q.denominator != 1:
        raise SystemExit(f"two-way nest: parent dt {dt(cfg_p)} s is not a whole number of nest steps ({dt(cfg_c)} s)")
    return int(q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m namgpu.nest.twoway", description=__doc__.split("\n\n")[0])
    ap.add_argument("--parent-config", type=Path, required=True)
    ap.add_argument("--nest-config", type=Path, required=True)
    ap.add_argument("--parent-init", type=Path, required=True)
    ap.add_argument("--nest-init", type=Path, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--source", default="gfs")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--two-way", action="store_true", help="NMMB's two-way feedback on (else concurrent one-way)")
    ap.add_argument("--wgt", type=float, default=1.0, help="2way_wgt: weight of the nest in the parent (0-1)")
    ap.add_argument("--history-fields", default=None)
    ap.add_argument("--full-every", default=None)
    a = ap.parse_args(argv)
    # two saved inits are opened (memory-mapped) in one process: ~530 files each passed the usual 1024 soft limit
    # (smoke 2026-11-05: OSError Too many open files in slab_init.load of the nest)
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = hard if hard != resource.RLIM_INFINITY else 1 << 20
    if soft != resource.RLIM_INFINITY and soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))

    import cupy as cp
    from namgpu import config, restart as rstmod
    from namgpu.io import history_select, io_history
    from namgpu.steps import drv_dfi_tdfi_driver, drv_timing_receipt as timing
    from namgpu.steps.drv_physics_initialize import in_run_dir
    from namgpu.boundary import bnd_boundary_init

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = datetime.strptime(Path(a.case).name[:14], "%Y-%m-%d_%Hz")
    case_dir = a.case_dir.resolve()
    cfg_c0 = config.load(a.nest_config)
    i_ps, j_ps = int(cfg_c0.raw["i_parent_start"]), int(cfg_c0.raw["j_parent_start"])
    stream = cp.cuda.Stream(non_blocking=True)            # one stream: parent, coupling and nest run in order
    rec = {"door": "woof-nam nest --two-way" if a.two_way else "woof-nam nest --concurrent", "case": a.case,
           "source": a.source, "hours": a.hours, "two_way": a.two_way, "start": start.isoformat()}
    t0 = time.perf_counter()
    ctx_p, drv_p, load_p = _ctx(a.parent_config, a.parent_init, stream, out / "parent", case_dir, start, a.source, a.hours)
    ctx_c, drv_c, load_c = _ctx(a.nest_config, a.nest_init, stream, out / "nest", case_dir, start, a.source, a.hours)
    ratio = _ratio(ctx_p.cfg, ctx_c.cfg)
    stream.synchronize()
    rec["load_s"] = round(time.perf_counter() - t0, 2)
    hf = a.history_fields or history_select.DEFAULT_PRESET
    fe = history_select.DEFAULT_FULL_EVERY if a.full_every is None else a.full_every
    for ctx in (ctx_p, ctx_c):
        ctx.history_select = history_select.Selection(hf, rstmod.steps_for(rstmod.parse_hours(fe), ctx.cfg.dt))
        ctx.restart_writer = None
    rec["grids"] = {"parent": [int(ctx_p.grid.im), int(ctx_p.grid.jm), int(ctx_p.grid.lm)],
                    "nest": [int(ctx_c.grid.im), int(ctx_c.grid.jm), int(ctx_c.grid.lm)]}
    rec["dt"] = {"parent": float(ctx_p.cfg.dt), "nest": float(ctx_c.cfg.dt), "ratio": ratio}
    print(f"two-way: parent {rec['grids']['parent']} dt {ctx_p.cfg.dt:g} s, nest {rec['grids']['nest']} dt "
          f"{ctx_c.cfg.dt:g} s, ratio {ratio}, SW corner on parent H ({i_ps}, {j_ps}), feedback "
          f"{'on, w ' + str(a.wgt) if a.two_way else 'off (concurrent one-way)'}", flush=True)

    hours_log = []
    stage = {"parent_s": 0.0, "nest_s": 0.0}
    rec["error"] = None
    nsteps = int(round(a.hours * 3600.0 / float(ctx_p.cfg.dt)))
    sph = max(1, int(round(3600.0 / float(ctx_p.cfg.dt))))
    n = 0
    with in_run_dir(ctx_p), stream:
        try:
            # TDFI on each domain (filter_method 3), each on its own start-source boundaries
            for nm, ctx in (("parent", ctx_p), ("nest", ctx_c)):
                if int(ctx.cfg.filter_method) > 0:
                    os.environ[config.CONFIG_ENV] = str(a.parent_config if ctx is ctx_p else a.nest_config)
                    tf = time.perf_counter()
                    ctx.filter_active = True
                    try:
                        drv_dfi_tdfi_driver.run(ctx)
                    finally:
                        ctx.filter_active = False
                    ctx.ntimestep, ctx.first_step = 0, True
                    stream.synchronize()
                    rec[f"tdfi_{nm}_s"] = round(time.perf_counter() - tf, 2)
                    print(f"tdfi {nm} done {rec[f'tdfi_{nm}_s']} s", flush=True)
            for ctx in (ctx_p, ctx_c):
                ctx.dt = ctx.cfg.dt
                ctx.start_time = ctx.start
            # the nest's carried boundary values start from its own (filtered) edges (boundary_init, as NMMB's
            # nests at their start); from here its boundaries come from the live parent, never from files
            ctx_c.concurrent_parent = True
            ctx_c.i_am_a_nest = True
            ctx_c.parent_child_time_ratio = ratio
            bnd_boundary_init.run(ctx_c)
            cpl = Coupler(ctx_p, ctx_c, ratio, i_ps, j_ps, a.two_way, a.wgt)
            rec["coupler"] = cpl.receipt()
            drv_p._initial_history()
            drv_c._initial_history()
            ev = [cp.cuda.Event() for _ in range(5)]
            acc = {"parent_incl_feedback": 0.0, "boundary": 0.0, "nest_x%d" % ratio: 0.0, "twoway_data": 0.0}
            t1 = time.perf_counter()
            sampled, pending = 0, False
            for n in range(1, nsteps + 1):
                mid = n % sph == sph // 2                      # one mid-hour step per hour is timed (no history in it)
                if mid:
                    ev[0].record(stream)
                cpl.update()                                   # phase 3: feedback from the end of the last step
                drv_p.step()                                   # one parent step
                if drv_p._due("history"):
                    drv_p.write_history()
                if mid:
                    ev[1].record(stream)
                cpl.boundary()                                 # phases 4 and 2: parent -> nest boundary tendencies
                if mid:
                    ev[2].record(stream)
                for _ in range(ratio):
                    drv_c.step()
                    if drv_c._due("history"):
                        drv_c.write_history()
                if mid:
                    ev[3].record(stream)
                cpl.generate()                                 # phase 5: nest -> parent two-way data
                if mid:
                    ev[4].record(stream)
                    pending = True
                if n % sph == 0:
                    stream.synchronize()                       # once per forecast hour: the only host sync
                    timing.drain(ctx_p)
                    timing.drain(ctx_c)
                    if pending:
                        pending = False
                        sampled += 1
                        acc["parent_incl_feedback"] += cp.cuda.get_elapsed_time(ev[0], ev[1]) / 1e3
                        acc["boundary"] += cp.cuda.get_elapsed_time(ev[1], ev[2]) / 1e3
                        acc["nest_x%d" % ratio] += cp.cuda.get_elapsed_time(ev[2], ev[3]) / 1e3
                        acc["twoway_data"] += cp.cuda.get_elapsed_time(ev[3], ev[4]) / 1e3
                    wall = time.perf_counter() - t1
                    line = (f"fh {n * float(ctx_p.cfg.dt) / 3600.0:6.2f}  parent step {n}  nest step {ctx_c.ntimestep}"
                            f"  wall {wall:8.1f} s  {wall / n * 1e3:7.1f} ms per parent step (incl. {ratio} nest steps)")
                    hours_log.append(line)
                    print(line, flush=True)
            stream.synchronize()
            wall = time.perf_counter() - t1
            rec["run_s"] = round(wall, 2)
            rec["parent_steps"] = int(ctx_p.ntimestep)
            rec["nest_steps"] = int(ctx_c.ntimestep)
            rec["ms_per_parent_step"] = round(wall / max(1, nsteps) * 1e3, 1)
            # one mid-hour parent step per hour, timed by events on the stream (no added syncs): the split per stage
            rec["sampled_ms_per_parent_step"] = {k: round(v / max(1, sampled) * 1e3, 2) for k, v in acc.items()}
            rec["sampled_steps"] = sampled
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc()[-3000:]
            print("RUN ERROR", rec["error"], flush=True)
            print(rec["trace"], flush=True)
    stream.synchronize()
    for ctx in (ctx_p, ctx_c):
        io_history.drain(ctx)
        try:
            timing.write_receipt(ctx, Path(ctx.out_dir))
        except Exception as e:  # noqa: BLE001
            rec.setdefault("receipt_errors", []).append(f"{type(e).__name__}: {e}")
    rec["hour_lines"] = hours_log
    rec["nonfinite_end"] = {nm: {k: int((~cp.isfinite(ctx.st[k])).sum()) for k in ("T", "Q", "CW", "U", "V", "PD")}
                            for nm, ctx in (("parent", ctx_p), ("nest", ctx_c))}
    rec["pool_end_gib"] = round(cp.get_default_memory_pool().total_bytes() / 2**30, 2)
    rec["card_used_gib"] = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    rec["wall_s"] = round(time.perf_counter() - t0, 2)
    (out / "twoway-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"two-way receipt {out / 'twoway-receipt.json'}: {rec.get('ms_per_parent_step')} ms per parent step, "
          f"non-finite {rec['nonfinite_end']}", flush=True)
    return 1 if rec["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
