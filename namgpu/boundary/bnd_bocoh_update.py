"""bnd-bocoh-update: BOCOH working-object update on the H-point S/N and W/E strips.

Ported from nmm/module_FLTBNDS.F90:5064-5354 (NAM nam.v4.2.11, 83df4e13c1), regional case.
Kernels: namgpu/kernels/bnd_bocoh_update.cu (bocoh_strip_2d, bocoh_strip_3d, bocoh_strip_4d).

What this unit does: working += tendency * dt, for every 2-d, 3-d and 4-d H-point boundary
variable, on each edge that is a domain edge (Fortran s_bdy, n_bdy, w_bdy, e_bdy).

What this unit does NOT do (other units own it; Fortran line ranges in UNITS.json):
  - the blend of the strips into PD, T, Q, CW, Q2 and PINT, the four corners and the outer PINT
    rows (5355-5962): bnd-bocoh-blend (namgpu/boundary/bnd_bocoh_blend.py). The blend weights
    wh(k) are read there (bocoh_wh in bnd_bocoh_blend.cu). wq(k) is never read in BOCOH.
  - the optional smoothing of the H rows (5963-6387): bnd-bocoh-smooth. It is dead while nsmud = 0.

Reads / writes: ctx.bc (boundary group, bnd-read-bc). Expected shape of the object:
    ctx.bc.h2d, ctx.bc.h3d, ctx.bc.h4d : lists of variables, each with attributes
        .south, .north, .west, .east  -> CuPy float32 strips in the CONTRACT layout
                                         (CONTRACT.md section 1a): k = 0 working, k = 1 tendency
        .lbnd, .ubnd (4-d only)       -> the Fortran nl bounds of that variable
    ctx.bc.s_bdy, n_bdy, w_bdy, e_bdy : optional explicit flags. If present they must agree with
                                         the derived flags (see _edge_flags), else ValueError.
ctx.dt is the dynamics step (25 s); ctx.grid.im/jm/lm; ctx.cfg.lnsh (5).

Ranges: the strip loops are the full strip (its_h2 = 1..IM, jts_h2 = 1..JM, ib = 1..lnsh), since
a single process owns the whole domain (module_DM_PARALLEL looplimits).
Launches: one per (edge, rank class) that has variables, in the Fortran order S, N, W, E; no host
sync, no per-step allocation. The pointer tables are built once and rebuilt only if the bound
arrays change. Every strip is checked to be a dense C-contiguous float32 array of the expected
size before its address is used, so a sliced or padded view raises instead of being read at the
wrong addresses.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "bnd_bocoh_update"
EDGES = ("south", "north", "west", "east")
H_2D = ("PD",)              # parent H boundary variables (module_SOLVER_GRID_COMP.F90:7658-7664)
H_3D = ("T", "Q", "CW")
_CACHE: dict = {}           # pointer tables per bc object (a dict has no __dict__ to hang them on)


def h_view(bc):
    """The h2d/h3d/h4d variable lists of ctx.bc. ctx.bc is the flat dict of bnd-boundary-init
    (CONTRACT 6a, keys "<SIDE>_<VAR>", slot axis first); each variable's .south/.north/.west/.east
    are those same arrays, so the update lands in the strips that bnd-bocoh-blend reads.
    Integrator fix 2026-11-04: before this the dict had no .h2d/.h3d, so the update silently did
    nothing and the boundary tendency was never applied."""
    import types
    if not isinstance(bc, dict):
        return (list(getattr(bc, "h2d", None) or []), list(getattr(bc, "h3d", None) or []),
                list(getattr(bc, "h4d", None) or []))

    def var(name):
        miss = [f"{s}_{name}" for s in "SNWE" if f"{s}_{name}" not in bc]
        if miss:
            raise KeyError(f"bnd-bocoh-update: ctx.bc missing {miss} (bnd-boundary-init writes them)")
        return types.SimpleNamespace(name=name, south=bc[f"S_{name}"], north=bc[f"N_{name}"],
                                     west=bc[f"W_{name}"], east=bc[f"E_{name}"])
    return [var(n) for n in H_2D], [var(n) for n in H_3D], []


def _edge_flags(ctx) -> dict:
    """Edge flags derived from the tile limits, as NMMB derives them (module_NMM_INTEGRATE.F90:340:
    S_BDY = (JTS == JDS), and likewise N, W, E). One card owns the whole domain, so the tile is the
    domain and every edge is a domain edge. Explicit flags on ctx.bc must agree."""
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    ids, ide, jds, jde = 1, im, 1, jm
    its, ite, jts, jte = 1, im, 1, jm          # one card: tile == domain
    flags = {"south": jts == jds, "north": jte == jde, "west": its == ids, "east": ite == ide}
    for edge in EDGES:
        attr = f"{edge[0]}_bdy"
        if hasattr(ctx.bc, attr) and bool(getattr(ctx.bc, attr)) != flags[edge]:
            raise ValueError(
                f"bnd-bocoh-update: ctx.bc.{attr} = {getattr(ctx.bc, attr)} disagrees with the "
                f"derived edge flag {flags[edge]} (one-card tile == domain)")
    return flags


def _strip_size(edge: str, lnsh: int, im: int, jm: int, lm: int, nl: int, rank_extra: bool) -> int:
    """Dense element count of one variable's strip on one edge: 2 time slots x (LM) x width x span
    (x NL for the 4-d class). Matches CONTRACT.md section 1a."""
    span = im if edge in ("south", "north") else jm
    n = 2 * span * lnsh * (lm if rank_extra else 1)
    return n * nl


def _check_strip(name: str, arr, expected: int):
    if arr.dtype != np.float32:
        raise ValueError(f"bnd-bocoh-update: {name} dtype {arr.dtype}, want float32")
    if not arr.flags.c_contiguous:
        raise ValueError(f"bnd-bocoh-update: {name} is not a dense C-contiguous strip "
                         f"(shape {arr.shape}, strides {arr.strides}); a sliced or padded view "
                         "would be read at the wrong addresses")
    if int(arr.size) != expected:
        raise ValueError(f"bnd-bocoh-update: {name} has {int(arr.size)} elements, "
                         f"want {expected} (CONTRACT.md section 1a)")


def _table(bc, key, arrs, names, expected):
    """Device table of data pointers, cached on the bc object; rebuilt if the arrays changed."""
    import cupy as cp

    for a, nm, n in zip(arrs, names, expected):
        _check_strip(nm, a, n)
    cache = _CACHE.setdefault((id(bc), "tables"), {})
    ptrs = tuple(int(a.data.ptr) for a in arrs)
    hit = cache.get(key)
    if hit is not None and hit[0] == ptrs:
        return hit[1]
    dev = cp.asarray(np.array(ptrs, dtype=np.uint64))
    cache[key] = (ptrs, dev)
    return dev


def _nlc(bc, edge, vars4d):
    """Device table of NL = ubnd - lbnd + 1 per 4-d variable, cached on the bc object."""
    import cupy as cp

    counts = tuple(int(v.ubnd) - int(v.lbnd) + 1 for v in vars4d)
    cache = _CACHE.setdefault((id(bc), "nlc"), {})
    hit = cache.get(edge)
    if hit is not None and hit[0] == counts:
        return hit[1]
    dev = cp.asarray(np.array(counts, dtype=np.int32))
    cache[edge] = (counts, dev)
    return dev


def run(ctx):
    bc = ctx.bc
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsh = int(ctx.cfg.lnsh)
    dt = np.float32(ctx.dt)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    flags = _edge_flags(ctx)

    h2, h3, h4 = h_view(bc)

    with scope:
        for edge in EDGES:
            if not flags[edge]:
                continue
            # south/north strips: fast index ib (IM), slow jb (lnsh); west/east: ib (lnsh), jb (JM).
            ns = edge in ("south", "north")
            ni, nj = (im, lnsh) if ns else (lnsh, jm)

            if h2:
                arrs = [getattr(v, edge) for v in h2]
                names = [f"{edge} h2d[{k}]" for k in range(len(h2))]
                exp = [_strip_size(edge, lnsh, im, jm, lm, 1, False)] * len(h2)
                tbl = _table(bc, (edge, "2d"), arrs, names, exp)
                kernels.get(UNIT, "bocoh_strip_2d")(
                    kernels.grid_2d(ni, nj, len(h2)), kernels.BLOCK_2D,
                    (np.int32(len(h2)), np.int32(ni), np.int32(nj), tbl, dt))

            if h3:
                arrs = [getattr(v, edge) for v in h3]
                names = [f"{edge} h3d[{k}]" for k in range(len(h3))]
                exp = [_strip_size(edge, lnsh, im, jm, lm, 1, True)] * len(h3)
                tbl = _table(bc, (edge, "3d"), arrs, names, exp)
                kernels.get(UNIT, "bocoh_strip_3d")(
                    kernels.grid_2d(ni, nj, lm), kernels.BLOCK_2D,
                    (np.int32(len(h3)), np.int32(ni), np.int32(nj), np.int32(lm), tbl, dt))

            if h4:
                arrs = [getattr(v, edge) for v in h4]
                names = [f"{edge} h4d[{k}]" for k in range(len(h4))]
                exp = [_strip_size(edge, lnsh, im, jm, lm, int(v.ubnd) - int(v.lbnd) + 1, True)
                       for v in h4]
                tbl = _table(bc, (edge, "4d"), arrs, names, exp)
                nlc = _nlc(bc, edge, h4)
                kernels.get(UNIT, "bocoh_strip_4d")(
                    kernels.grid_2d(ni, nj, lm), kernels.BLOCK_2D,
                    (np.int32(len(h4)), np.int32(ni), np.int32(nj), np.int32(lm), tbl, nlc, dt))
