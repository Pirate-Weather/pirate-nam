"""Binding of the V-point boundary objects for bnd-bocov-update and bnd-bocov-blend.

The boundary producer bnd_boundary_init (and bnd-read-bc / bnd-bocoh-update) keep the parent strips in
ctx.bc as a flat dict keyed "<SIDE>_<VAR>" (SIDE in S, N, W, E; VAR is the NMMB name), each a device array
with the Fortran time slot as its FIRST axis (slot 0 = value, slot 1 = tendency). The BOCOV wrappers
read a per-variable object instead, so this module builds that view from the dict and the state:

    v.var_2d[n] / v.var_3d[n]  each with
        .full                       the domain field, a view of ctx.st[VAR] reshaped to [LM, JM, IM]
                                    (3-D) or [JM, IM] (2-D); writes land in ctx.st
        .south/.north/.west/.east   ctx.bc["<SIDE>_<VAR>"], the same arrays the bocoh units use

The V-point variables are U and V (3-D, nvars 2D_V = 0, 3D_V = 2, module_SOLVER_GRID_COMP.F90:7658-7664).
Nothing here allocates on the device: the view is rebuilt on each call from existing arrays.
"""
from __future__ import annotations

import types

V_2D: tuple = ()
V_3D: tuple = ("U", "V")


def _var(ctx, name: str, three_d: bool, lm: int, jm: int, im: int):
    full = ctx.st[name]
    shape = (lm, jm, im) if three_d else (jm, im)
    if full.size != _prod(shape):
        raise ValueError(f"bnd-bocov: state {name} has {full.size} elements, expected {shape}")
    full = full.reshape(shape)
    strips = {}
    for side in ("S", "N", "W", "E"):
        key = f"{side}_{name}"
        if key not in ctx.bc:
            raise KeyError(f"bnd-bocov: ctx.bc[{key!r}] missing (bnd-boundary-init owns it)")
        strips[side] = ctx.bc[key]
    return types.SimpleNamespace(
        name=name, full=full,
        south=strips["S"], north=strips["N"], west=strips["W"], east=strips["E"])


def _prod(shape):
    p = 1
    for s in shape:
        p *= int(s)
    return p


def v_view(ctx):
    """Return the object with .var_2d and .var_3d lists that the bocov wrappers read as ctx.bc.v."""
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    return types.SimpleNamespace(
        var_2d=[_var(ctx, n, False, lm, jm, im) for n in V_2D],
        var_3d=[_var(ctx, n, True, lm, jm, im) for n in V_3D])
