"""Row-chunk views of the NMMB context for the RRTM chain (rrtm-call).

Why: the grrad chain holds per-column spectral and optical scratch (faersw [3, 14, LMK], faerlw
[3, 16, LMK], gasvmr [10, LMK], clouds [9, LMK], the GR1 tracers, the RAD_* layer arrays). Over the
whole 954 x 835 domain that is more than 30 GB, which does not fit beside the 10.6 GiB State on a
32 GB card. The Fortran runs RRTM per row and per DAYPART chunk (module_RADIATION.F90:586-625,
module_RA_RRTM.F90:879-1223, CHNK_RRTM = 8), so a row chunk is the Fortran's own decomposition.

How: ChunkCtx is a ctx whose grid has jm = W rows and whose st/tables are views of the parent:
  - a parent state field with trailing dims (JM, IM) is copied into a persistent chunk buffer on
    first access in the chunk (rows w0..w1) and written back for the OWNED rows only at the end;
  - st.add of a name the parent does not hold is chunk-local scratch (persists across chunks,
    one shape, allocated once);
  - a parent table entry that is a device array (or a dict of them) with trailing dims (JM, IM) is
    read as a row slice (read only); other parent table entries (k-tables, init records) are shared;
    table writes are chunk-local.
Each chunk computes rows [w0, w1) and owns [a, b) with a >= w0+1 and b <= w1-1 unless the edge is
the domain edge, so the RRTM row gate (rows 2..JM-1, module_RADIATION.F90:592, grrad_lw guard on
j0 == 0 and j0 == jm-1) skips only halo rows or the true domain edge rows.
Every column is independent in RRTM, so the chunked result is the whole-domain result.
"""
from __future__ import annotations

import copy
import re
import weakref

import numpy as np

# Fields created by st.add inside the chain that are read after the radiation call (driver-post
# outputs and its old-cloud scratch): created full-domain in the parent, not chunk-local.
PERSIST = {"CSMID", "CCMID", "COSZDG", "RRTM_CLDCFR", "RRTM_MTOP", "RRTM_MBOT"}

# One flat float32 workspace per chunk context (32 GB fit path), shared in time by the two units whose
# g-point working sets are never live together: swrad's chunk scratch (zero-filled for every SW chunk
# before use, its values dead when swrad returns) and the LW seam's taug/fracs pair (every element written
# by the band units before it is read, NAMGPU_LW_FILLCHECK; dead after rtrnmr). SW runs before LW in each
# chunk. Sized max(SW scratch, LW pair) instead of their sum: the LW pair (68 KB per column) costs nothing
# beside the SW scratch (126 KB per column). rrtm-call allocates it; the units carve views from it.
WS_KEY = "rrtm_call:gpoint_ws"


def workspace(ctx):
    """The chunk context's shared g-point workspace, or None (whole-domain callers, the full path)."""
    t = getattr(ctx, "tables", None)
    if isinstance(t, ChunkTables) and dict.__contains__(t, WS_KEY):
        return dict.__getitem__(t, WS_KEY)
    return None


MIN_ROWS = 3


def windows(jm: int, w: int):
    """(w0, w1, a, b) per chunk: compute rows [w0, w1), owned rows [a, b).

    An inner chunk of w rows owns w - 2 of them (one halo row each side), so w must be at least MIN_ROWS = 3. With
    w = 2 an inner chunk owns no row and the loop never advanced: radiation hung at its first call with the host
    list growing without end (woof-nam 0.1.2, 8 GB card, NAMGPU_RRTM_ROWS 2)."""
    if w >= jm:
        return [(0, jm, 0, jm)]
    if w < MIN_ROWS:
        raise ValueError(f"row chunks of {w} rows: radiation and column bands need at least {MIN_ROWS} rows per chunk "
                         f"(NAMGPU_RRTM_ROWS {MIN_ROWS} or more; 4 or more is the door's floor)")
    out, a = [], 0
    while a < jm:
        w0 = 0 if a == 0 else min(a - 1, jm - w)
        w1 = w0 + w
        b = jm if w1 >= jm else w1 - 1
        out.append((w0, w1, a, b))
        a = b
    return out


class _Fields(dict):
    """ctx.st.fields of a chunk: parent rows copied on first access."""

    def __init__(self, cst):
        super().__init__()
        # weak: ChunkState.fields -> _Fields -> ChunkState was a reference cycle, so a dropped chunk or band
        # state (and every chunk copy and chunk-local array it holds) waited for Python's cyclic collector
        # instead of going back to the pool when its call ended (memory work 2026-11-05: 12.8 GiB of
        # dead band states live at the 333x333 forecast peak, process peak 8.1 -> 22.2 GB)
        self._c = weakref.proxy(cst)

    def __contains__(self, name):
        return self._c.has(name)

    def __getitem__(self, name):
        return self._c.get(name)

    def get(self, name, default=None):
        return self._c.get(name) if self._c.has(name) else default

    def __setitem__(self, name, value):
        self._c.local[name] = value

    def keys(self):
        return list(dict.fromkeys(list(self._c.parent.fields) + list(self._c.local)))

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]


class ChunkState:
    def __init__(self, parent, jm_full: int, im: int, w: int, xp):
        self.parent = parent
        self.jm_full, self.im, self.w = jm_full, im, w
        self.xp = xp
        self.specs = parent.specs
        self.scalars = parent.scalars
        self.local = {}          # chunk-local scratch, persistent across chunks
        self.bufs = {}           # chunk copy buffers of parent fields, persistent
        self.live = set()        # parent fields copied in the current chunk
        self.w0 = self.w1 = 0
        self.grid = None
        self.fields = _Fields(self)

    def _rowfield(self, a) -> bool:
        return getattr(a, "ndim", 0) >= 2 and tuple(a.shape[-2:]) == (self.jm_full, self.im)

    def begin(self, w0, w1, grid):
        self.w0, self.w1 = w0, w1
        self.grid = grid
        self.live = set()

    def has(self, name):
        return name in self.local or name in self.parent.fields

    def get(self, name):
        if name in self.local:
            return self.local[name]
        a = self.parent.fields[name]
        if not self._rowfield(a):
            return a                               # vertical 1-D and small arrays: shared
        buf = self.bufs.get(name)
        if buf is None:
            buf = self.xp.empty(a.shape[:-2] + (self.w, self.im), dtype=a.dtype)
            self.bufs[name] = buf
        if name not in self.live:
            buf[...] = a[..., self.w0:self.w1, :]
            self.live.add(name)
        return buf

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        return self.has(name)

    def __getattr__(self, name):
        if name.startswith("_") or name in ("parent", "local", "bufs", "live", "fields"):
            raise AttributeError(name)
        if self.has(name):
            return self.get(name)
        raise AttributeError(name)

    def add(self, name, shape, dtype="float32"):
        if self.has(name):
            return self.get(name)
        shape = tuple(int(s) for s in shape)
        if name in PERSIST and len(shape) >= 2 and shape[-2:] == (self.w, self.im):
            self.parent.add(name, shape[:-2] + (self.jm_full, self.im), dtype)
            return self.get(name)
        self.local[name] = self.xp.zeros(shape, dtype=dtype)
        return self.local[name]

    def writeback(self, a, b):
        """Owned rows [a, b) of every parent field touched in this chunk go back to the parent."""
        la, lb = a - self.w0, b - self.w0
        for name in self.live:
            self.parent.fields[name][..., a:b, :] = self.bufs[name][..., la:lb, :]

    def nbytes(self):
        return int(sum(x.nbytes for x in list(self.local.values()) + list(self.bufs.values())))

    def release(self):
        """The call is over: drop every chunk copy and chunk-local array now (values are dead after writeback)."""
        self.local.clear()
        self.bufs.clear()
        self.live = set()


# Constant lookup tables the RRTM units load on first use (k-tables, reference tables, the rtrnmr
# transmittance tables): input-independent, the same bits on every call. Written by a chunk they go to the
# PARENT ctx.tables, so they are loaded and uploaded once per process instead of once per radiation call
# (the 32 GB fit path drops the chunk context after every call; ~1.1 MB of device tables, ~40 npz reads).
SHARED_TABLES = re.compile(
    r"^(rrtm_(lw|sw)_kgb\w*|rrtm_lw_taugb\d\d_tables|rrtm_sw_taumol\d\d_tables|rrtm_(lw|sw)_setcoef_tables"
    r"|rrtm_lw_cldprlw|rrtm_sw_cldprtb|rrtm_sw_sflux_\w+|rrtm_lw_tables|rrtm_lw_totplnk"
    r"|rrtm_lw_taugb(04|15)_refrat|rrtm_lw_ngb|lw_core:ngb)$")


class ChunkTables(dict):
    """ctx.tables of a chunk: writes local (constant lookup tables, SHARED_TABLES, go to the parent);
    reads fall back to the parent, row-sliced when the entry is a (dict of) device array(s) with trailing
    dims (JM, IM)."""

    def __init__(self, parent: dict, jm_full: int, im: int, xp):
        super().__init__()
        self._p, self._jm, self._im, self._xp = parent, jm_full, im, xp
        self._w0 = self._w1 = 0
        self._slices = {}

    def begin(self, w0, w1):
        self._w0, self._w1 = w0, w1
        self._slices = {}

    def _isrow(self, v):
        return (isinstance(v, self._xp.ndarray) and v.ndim >= 2
                and tuple(v.shape[-2:]) == (self._jm, self._im))

    def _parent(self, key):
        if key in self._slices:
            return self._slices[key]
        v = self._p[key]
        if isinstance(v, dict) and any(self._isrow(x) for x in v.values()):
            v = {k: (self._xp.ascontiguousarray(x[..., self._w0:self._w1, :]) if self._isrow(x) else x)
                 for k, x in v.items()}
        elif self._isrow(v):
            v = self._xp.ascontiguousarray(v[..., self._w0:self._w1, :])
        self._slices[key] = v
        return v

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return self._parent(key)

    def __setitem__(self, key, value):
        if isinstance(key, str) and SHARED_TABLES.match(key):
            self._p[key] = value
            self._slices.pop(key, None)
            return
        dict.__setitem__(self, key, value)

    def __contains__(self, key):
        return dict.__contains__(self, key) or key in self._p

    def __bool__(self):
        # The chunk/band dict is empty until a unit writes a local entry, yet every parent table reads through it.
        # Without this an untouched view is falsy, and "ctx.tables.get(k) if ctx.tables else None" (phys-gwd-driver,
        # the rrtm init units) reads None: the 12 km fit path stopped at the first banded TURBL call with
        # "phys-gwd-driver needs phys-gwd-init first" (2026-11-05, a 96 GB card with 32 GB free).
        return dict.__len__(self) > 0 or bool(self._p)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def release(self):
        """Drop the chunk's own table entries and row slices (the parent's tables are untouched)."""
        dict.clear(self)
        self._slices = {}


def chunk_grid(grid, w0: int, w1: int):
    """A shallow copy of the grid with jm = w1 - w0 and every [JM]- or [JM, IM]-leading host array
    row-sliced."""
    g = copy.copy(grid)
    jm = int(grid.jm)
    for k, v in vars(grid).items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == jm:
            setattr(g, k, v[w0:w1])
    g.jm = w1 - w0
    return g


class ChunkCtx:
    """Attribute proxy: grid/st/tables are the chunk views, everything else is the parent ctx."""

    def __init__(self, ctx, w: int):
        import cupy as cp
        object.__setattr__(self, "_p", ctx)
        g = ctx.grid
        object.__setattr__(self, "st", ChunkState(ctx.st, int(g.jm), int(g.im), w, cp))
        object.__setattr__(self, "tables", ChunkTables(ctx.tables, int(g.jm), int(g.im), cp))
        object.__setattr__(self, "grid", g)
        object.__setattr__(self, "w", w)

    def begin(self, w0, w1):
        g = chunk_grid(self._p.grid, w0, w1)
        object.__setattr__(self, "grid", g)
        self.st.begin(w0, w1, g)
        self.tables.begin(w0, w1)

    def release(self):
        """Free the chunk working set when the context is dropped (rrtm_call fit path)."""
        self.st.release()
        self.tables.release()

    def __getattr__(self, name):
        return getattr(self._p, name)

    def __setattr__(self, name, value):
        setattr(self._p, name, value)
