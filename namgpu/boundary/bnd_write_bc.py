"""bnd-write-bc: save the state's boundary strips into the boundary working arrays.

Ported from nmm/module_FLTBNDS.F90:3928-5059 (write_bc, NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/bnd_write_bc.cu (bcw_strip, one launch per side and boundary variable).

Boundary variables are the parent's hardwired list (module_SOLVER_GRID_COMP.F90, MY_DOMAIN_ID == 1):
  2-D H: PD;  3-D H: T, Q, CW;  4-D H: none;  2-D V: none;  3-D V: U, V.
Source binding: the Fortran FULL_VAR pointers are set at module_SOLVER_GRID_COMP.F90:7772-8091 from
ESMF fields looked up by name. The name list here is the parent's list as written in this unit;
it is NOT yet traced to the registry file that sets the boundary flags (open, see the unit note).

4-D H: the kernel's species axis. nspec = size / (LM*JM*IM) (state layout [nspec, LM, JM, IM]).
  Declared divergence from the Fortran as written: the active tendency line
  (module_FLTBNDS.F90:4775) reads targ_4d_h_n(nv)%base(n1,n2,n3,1,nv), using the variable index nv
  as the fifth (species) index. This port reads the species index it writes. The parent passes zero
  4-D variables, so the divergence is inert for NAM today.
2-D V: no parent variables; the offsets are reproduced from the Fortran as written.

Call sites. The caller names the Fortran WRITE_BC call it is making (keyword `site`, no default):
  "ntsd0"      module_SOLVER_GRID_COMP.F90:3929, inside IF(NTIMESTEP==0): TBOCO, recompute TRUE.
               Requires ctx.ntimestep == 0.
  "dfi_start"  module_SOLVER_GRID_COMP.F90:3291, inside IF(.NOT.GLOBAL): TBOCO + DFIHR_BOCO/2,
               recompute TRUE. No step gate (the DFI start is not step 0).
  "recomp_off" init-boco value slot 0 (namgpu/init/init_boco_tendencies.py): recompute FALSE.
The nest write at module_SOLVER_GRID_COMP.F90:3829 (TBOCO = PARENT_CHILD_TIME_RATIO*DT, recompute
FALSE) is a nest-domain call and is not a site of this unit; any other site name raises.

Reads : state edge strips of PD, T, Q, CW (H points, width lnsh) and U, V (V points, width lnsv).
Writes: ctx.bc[(cls, name, side)], float32, shape [2, nspec, LM, lines, along]:
        slot 0 = the state edge values now; slot 1 = the tendency (only when recomp_tend).
        side is 0 N, 1 S, 2 W, 3 E; along is IM for N/S and JM for W/E.
Recompute (recomp_tend) reads the previous slot 0 and slot 1 values first (Fortran targ = slot0 +
  tboco*slot1). Those values come from bnd-boundary-init, so a recompute write raises unless the
  buffer already exists. The key convention here is (cls, name, side); bnd-boundary-init's docstring
  keys its buffers "<SIDE>_<NAME>". Reconciling the two is open until bnd-boundary-init lands.
Single layout: every state field must be the full domain, ending in (JM, IM) with size
  nspec*LM*JM*IM. Halo or multi-card layouts raise (the kernel indexes 1..IM and 1..JM).
No host sync; one kernel launch per (variable, side).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "bnd_write_bc"
KERNEL = "bcw_strip"
SIDE_N, SIDE_S, SIDE_W, SIDE_E = 0, 1, 2, 3

# Parent boundary variables (module_SOLVER_GRID_COMP.F90, NVARS_BC_* for MY_DOMAIN_ID == 1).
PARENT_BC_VARS = {
    "h2d": ["PD"],
    "h3d": ["T", "Q", "CW"],
    "h4d": [],
    "v2d": [],
    "v3d": ["U", "V"],
}

# The Fortran WRITE_BC call sites this unit serves (see the module docstring).
SITES = {
    "ntsd0": {"recompute": True, "step0_only": True},
    "dfi_start": {"recompute": True, "step0_only": False},
    "recomp_off": {"recompute": False, "step0_only": False},
}


def _line0(cls: str, side: int, im: int, jm: int, lnsh: int, lnsv: int) -> int:
    """Zero-based first field line of the strip, as write_bc indexes it (1-based Fortran minus 1).

    H (mass) strips:  N j = JM-lnsh+1..JM, S j = 1..lnsh, W i = 1..lnsh, E i = IM-lnsh+1..IM.
    3-D V strips:     N j = JM-lnsv+1..JM, S/W as H with lnsv, E i = IM-lnsv..IM-1.
    2-D V strips:     N j = JM-lnsv..JM-1, E i = IM-lnsh+1..IM-lnsh+lnsv  (write_bc uses lnsh here).
    The 2-D V east offset and the 2-D/3-D V north offset differ from the 3-D V ones in the Fortran
    as written; they are reproduced as written (see the unit note)."""
    if cls.startswith("h"):
        return {SIDE_N: jm - lnsh, SIDE_S: 0, SIDE_W: 0, SIDE_E: im - lnsh}[side]
    if cls == "v3d":
        return {SIDE_N: jm - lnsv, SIDE_S: 0, SIDE_W: 0, SIDE_E: im - lnsv - 1}[side]
    return {SIDE_N: jm - lnsv - 1, SIDE_S: 0, SIDE_W: 0, SIDE_E: im - lnsh}[side]


def _bc_buffer(ctx, key, nspec, lm, nlines, along, create: bool):
    import cupy as cp
    if ctx.bc is None:
        ctx.bc = {}
    shape = (2, nspec, lm, nlines, along)
    if key not in ctx.bc:
        if not create:
            raise RuntimeError(
                f"bnd-write-bc: recompute write needs the existing slot buffer {key!r}; the Fortran "
                "reads the previous slot values (targ = slot0 + tboco*slot1) and bnd-boundary-init "
                "has not populated them")
        ctx.bc[key] = cp.zeros(shape, dtype=np.float32)
    buf = ctx.bc[key]
    if tuple(buf.shape) != shape:
        raise RuntimeError(f"bnd-write-bc: buffer {key!r} has shape {tuple(buf.shape)}, expected {shape}")
    return buf


def run(ctx, *, site, tboco, recomp_tend):
    """Write the parent boundary strips of the state for one Fortran WRITE_BC call site.

    site        one of SITES (names the Fortran call, see the module docstring). Required.
    tboco       boundary interval in seconds for this call (the Fortran TBOCO argument). Required,
                and positive when recomp_tend is True.
    recomp_tend the Fortran recomp_tend argument; must match the site. Required.
    """
    if site not in SITES:
        raise ValueError(f"bnd-write-bc: unknown call site {site!r}; expected one of {sorted(SITES)}")
    spec = SITES[site]
    recomp = bool(recomp_tend)
    if recomp != spec["recompute"]:
        raise ValueError(f"bnd-write-bc: site {site!r} needs recomp_tend={spec['recompute']}, got {recomp_tend!r}")
    if spec["step0_only"] and int(ctx.ntimestep) != 0:
        raise ValueError(
            "bnd-write-bc: site 'ntsd0' is the NTIMESTEP==0 call (module_SOLVER_GRID_COMP.F90:3929); "
            f"ctx.ntimestep is {ctx.ntimestep}")
    if tboco is None:
        raise ValueError("bnd-write-bc: tboco is required (the Fortran TBOCO argument has no default)")
    tboco = float(tboco)
    if recomp and not tboco > 0.0:
        raise ValueError(f"bnd-write-bc: tboco must be positive for a recompute write, got {tboco}")

    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsh = int(ctx.cfg.lnsh)
    lnsv = int(ctx.cfg.lnsv)

    kern = kernels.get(UNIT, KERNEL)
    block = (128, 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    for cls, names in PARENT_BC_VARS.items():
        is_v = cls.startswith("v")
        nlines = lnsv if is_v else lnsh
        for name in names:
            field = st[name]
            if cls.endswith("4d"):
                nspec = int(field.size) // (lm * jm * im)
                lev = lm
            elif cls.endswith("3d"):
                nspec, lev = 1, lm
            else:
                nspec, lev = 1, 1
            if int(field.size) != nspec * lev * jm * im or tuple(field.shape[-2:]) != (jm, im):
                raise ValueError(
                    f"bnd-write-bc: state field {name} has shape {tuple(field.shape)}; expected a "
                    f"full-domain layout ending in (JM, IM) = ({jm}, {im}) with {nspec * lev} levels "
                    "(halo and multi-card layouts are not supported by this unit)")
            for side in (SIDE_N, SIDE_S, SIDE_W, SIDE_E):
                along = im if side < 2 else jm
                buf = _bc_buffer(ctx, (cls, name, side), nspec, lev, nlines, along, create=not recomp)
                grid = ((along + block[0] - 1) // block[0], nlines, nspec * lev)
                args = (
                    field, buf,
                    np.int32(im), np.int32(jm), np.int32(lev), np.int32(nspec),
                    np.int32(nlines), np.int32(along),
                    np.int32(_line0(cls, side, im, jm, lnsh, lnsv)), np.int32(side),
                    np.int32(int(recomp)), np.float32(tboco),
                )
                with scope:
                    kern(grid, block, args)


# ---- dict layout (the boundary arrays the forecast actually uses) --------------------------------
# ctx.bc from bnd-boundary-init is a dict keyed "<SIDE>_<NAME>" (CONTRACT 6a): [2, (LM,) lines, IM] for
# S/N and [2, (LM,) JM, lines] for W/E, slot axis first. The (cls, name, side) buffers of run() above
# are a different layout that no forecast consumer reads, so the TDFI direction-change WRITE_BC goes
# through run_dict(): same Fortran strip offsets (_line0), same arithmetic, the dict arrays in place.
DICT_SIDES = (("N", SIDE_N), ("S", SIDE_S), ("W", SIDE_W), ("E", SIDE_E))


def run_dict(ctx, *, site, tboco, recomp_tend):
    """WRITE_BC onto the dict boundary arrays (module_FLTBNDS.F90:3928-5059), recompute form:
    targ = slot0 + tboco*slot1 (old values), slot0 = state strip, slot1 = (targ - strip)/tboco.
    CuPy slices on ctx.stream, no host sync; called once per run (TDFI direction change)."""
    if site not in SITES:
        raise ValueError(f"bnd-write-bc: unknown call site {site!r}; expected one of {sorted(SITES)}")
    if bool(recomp_tend) != SITES[site]["recompute"]:
        raise ValueError(f"bnd-write-bc: site {site!r} needs recomp_tend={SITES[site]['recompute']}")
    if not isinstance(ctx.bc, dict):
        raise RuntimeError("bnd-write-bc: run_dict needs the dict boundary arrays of bnd-boundary-init")
    tb = np.float32(tboco)
    if recomp_tend and not tb > 0:
        raise ValueError(f"bnd-write-bc: tboco must be positive for a recompute write, got {tboco}")
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsh, lnsv = int(ctx.cfg.lnsh), int(ctx.cfg.lnsv)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        for cls, names in PARENT_BC_VARS.items():
            if not names:
                continue
            if cls not in ("h2d", "h3d", "v3d"):
                raise RuntimeError(f"bnd-write-bc: run_dict has no layout for class {cls}")
            w = lnsv if cls.startswith("v") else lnsh
            nlev = 1 if cls.endswith("2d") else lm
            for name in names:
                f = ctx.st[name].reshape(nlev, jm, im)
                for sname, side in DICT_SIDES:
                    l0 = _line0(cls, side, im, jm, lnsh, lnsv)
                    strip = f[:, l0:l0 + w, :] if side < 2 else f[:, :, l0:l0 + w]
                    buf = ctx.bc[f"{sname}_{name}"]
                    b = buf.reshape((2,) + strip.shape)
                    if b.data.ptr != buf.data.ptr:
                        raise RuntimeError(f"bnd-write-bc: {sname}_{name} reshape copied")
                    if recomp_tend:
                        targ = b[0] + tb * b[1]
                        b[0][...] = strip
                        b[1][...] = (targ - b[0]) / tb
                    else:
                        b[0][...] = strip
    return {"site": site, "tboco": float(tb)}
