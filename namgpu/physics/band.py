"""Row bands for the column physics units (TURBL with JSFC/NOAH/SFCDIAGS/MYJPBL/GWD, CUCNVC/BMJ, GSMDRIVE/FER_HIRES).

Why: these units keep their per-column work arrays as whole-domain arrays (TURBL: PHINT, TH, EXNER, RR, PHMID, DZ,
U_PHY, V_PHY, Z, the BL tendencies and the GWD tendencies; CUCNVC: its copies of the same plus RTHCUTEN, RQCUTEN;
GSMDRIVE: DZ8W, RHO_PHY, P_PHY, PI_PHY, TH_PHY). At 12 km that is 2.7-3.4 GiB for TURBL alone (VRAM audit
2026-11-04, unit transient of phys-turbl-driver), live only while the unit runs. Every column of these schemes is
independent, so a unit can run on a band of rows at a time, the way RRTM already runs in row chunks (rrtm_chunk,
module_RADIATION.F90 CHNK_RRTM), and its work arrays then exist only for the current band.

How: BandCtx is rrtm_chunk's ChunkCtx with these changes.
  - Band-local arrays: only names proven write-before-read inside the unit (state.PHYS_SCRATCH and the unit's
    state.UNIT_SCRATCH, both checked with the NAMGPU_SCRATCH_POISON A/B), minus the names an unbanded unit reads
    afterwards (keep_whole: DUDT/DVDT, read by drv-h-to-v-tend). Any other name a unit adds is created whole in the parent State, as
    without bands, so a value carried from step to step is never band-local.
  - Parent fields are copied per band into one buffer per device allocation (keyed by pointer, shape and strides),
    so two names for one array (TSFC/TSKIN, DUDT_PHY/DUDT, QC and its TRACERS slot) share one buffer.
  - The NMMB_LIMITS tile table is given per band with every j limit shifted to band rows; the tile limits are
    clamped to the rows the band may compute (its first and last rows are halo rows unless they are the domain
    edge). jds/jde/jms/jme are shifted only, so a test for the domain edge still meets only the true edge.
  - 1-D per-row device vectors (BMJ_DXH) are sliced to the band.
  - Bands overlap by one row (rrtm_chunk.windows): a band computes rows [w0, w1) and writes back only the rows it
    owns, [a, b), never its halo rows unless they are the domain edge. The units' own row gates (j0 in [1, JM-2])
    and one-row stencils (U_PHY from U at j-1) therefore see the inputs of the whole-domain call, and the result
    is bitwise the whole-domain result.
  - Table entries a unit caches during the call (host scalars, uploaded vectors) are kept in the parent tables
    after the call unless they hold a band-shaped array.

When to band: only when the unit's whole-domain call would not fit, i.e. its working set (bytes per row x JM) plus
WHOLE_HEADROOM_MIB is more than the free device memory plus the free pool. A band copies its rows in and out, which cost
5 % of the step at 12 km on a 32 GB card (TURBL +10.8 s, BMJ +10.1 s, Ferrier +3.9 s in 6 h) and bought no memory
there (the process peak is set by the radiation call: 32,550 MiB with bands off, the same as with them on); at 333 x 333
the bands raised the peak (6,538 vs 5,982 MiB). Bands stay for a card that cannot hold the whole call.
Band size when banding: budget = min(NAMGPU_PHYS_BAND_MIB (default 512), half of the free device memory plus the free
pool); rows = budget / bytes per row, bytes per row measured on the unit's first banded call and kept in ctx.tables.
NAMGPU_PHYS_BAND_MIB set explicitly bands whenever the whole domain is over that budget (the old rule).
NAMGPU_PHYS_BAND_ROWS=N forces N rows (tests); 0 turns bands off. Bands run only on the fit memory path and on one
card (the slab limits of several cards are not shifted here).
"""
from __future__ import annotations

import copy
import os

import numpy as np

from namgpu.physics.rrtm import rrtm_chunk

ENV_ROWS = "NAMGPU_PHYS_BAND_ROWS"
ENV_MIB = "NAMGPU_PHYS_BAND_MIB"
DEFAULT_MIB = 512
WHOLE_HEADROOM_MIB = 1024       # free memory kept beside a whole-domain call (fragments, the unit's small temporaries)
MIN_ROWS = 8
LIMITS_KEY = "NMMB_LIMITS"
ROW_VECTORS = frozenset({"BMJ_DXH"})
# Host grid arrays per row: [JM, IM] fields and these 1-D [JM] vectors (sg*/dsg* are vertical, never sliced).
GRID_ROW_VECTORS = ("dxh", "dxv")
# First-call guess of the band working set in [LM+1] row-planes per row (then the measured value is used).
PLANES_GUESS = {"phys-turbl-driver": 48, "phys-bmj-driver": 30, "phys-fer-driver": 24}
_LIMIT_SHIFT_ONLY = ("jds", "jde", "jms", "jme")
_BLOCK = 256


def _key(a):
    return (int(a.data.ptr), tuple(a.shape), tuple(a.strides), str(a.dtype))


def _copy_kernel(name):
    from namgpu import kernels
    return kernels.get("phys_band_copy", name)


class BandState(rrtm_chunk.ChunkState):
    """ctx.st of a band. Parent fields come in through one buffer per device allocation; every C-contiguous
    4-byte field the unit has used so far moves in and out with one batched launch per band (phys_band_copy)."""

    def __init__(self, parent, jm_full, im, w, xp, scratch):
        super().__init__(parent, jm_full, im, w, xp)
        self.scratch = scratch
        self.src = {}          # key -> parent array
        self.names = {}        # parent field name -> key (the next bands' prefetch list)
        self.batch = []        # keys moved by the batched copies, in order
        self._tab = None       # device segment table of self.batch: uint64 [2, S] (parent plane, band plane)

    def _buf(self, a):
        k = _key(a)
        if k not in self.bufs:
            self.bufs[k] = self.xp.empty(a.shape[:-2] + (self.w, self.im), dtype=a.dtype)
            self.src[k] = a
            if a.flags.c_contiguous and a.dtype.itemsize == 4:
                self.batch.append(k)
                self._tab = None
        return k

    def get(self, name):
        if name in self.local:
            return self.local[name]
        a = self.parent.fields[name]
        if name in ROW_VECTORS and getattr(a, "ndim", 0) == 1 and a.shape[0] == self.jm_full:
            return a[self.w0:self.w1]
        if not self._rowfield(a):
            return a
        k = self._buf(a)
        self.names[name] = k
        buf = self.bufs[k]
        if k not in self.live:
            buf[...] = a[..., self.w0:self.w1, :]
            self.live.add(k)
        return buf

    def _table(self):
        if self._tab is None:
            jm, im, w = self.jm_full, self.im, self.w
            pp, bp = [], []
            for k in self.batch:
                p, b = self.src[k], self.bufs[k]
                outer = np.arange(p.size // (jm * im), dtype=np.uint64)
                pp.append(np.uint64(p.data.ptr) + outer * np.uint64(jm * im * 4))
                bp.append(np.uint64(b.data.ptr) + outer * np.uint64(w * im * 4))
            self._tab = self.xp.asarray(np.stack([np.concatenate(pp), np.concatenate(bp)]))
        return self._tab

    def _launch(self, kern, n, *offs):
        tab = self._table()
        nseg = int(tab.shape[1])
        gx = max(1, min(32, (n + _BLOCK - 1) // _BLOCK))
        _copy_kernel(kern)((gx, min(nseg, 65535), 1), (_BLOCK, 1, 1),
                           (tab[0], tab[1], np.int32(nseg), *[np.int64(o) for o in offs], np.int64(n)))

    def prefetch(self, names):
        """Right after begin(): every field in names the parent holds, and every batched field used so far in
        this call, comes into the band with one launch."""
        for n in names:
            a = self.parent.fields.get(n)
            if a is not None and n not in self.local and self._rowfield(a):
                self.names[n] = self._buf(a)
        if self.batch:
            self._launch("band_copy_in", self.w * self.im, self.w0 * self.im)
            self.live.update(self.batch)

    def add(self, name, shape, dtype="float32"):
        if self.has(name):
            return self.get(name)
        shape = tuple(int(s) for s in shape)
        if name not in self.scratch:
            # not proven write-before-read in this unit: held whole in the parent, as without bands
            if len(shape) >= 2 and shape[-2:] == (self.w, self.im):
                shape = shape[:-2] + (self.jm_full, self.im)
            self.parent.add(name, shape, dtype)
            return self.get(name)
        self.local[name] = self.xp.zeros(shape, dtype=dtype)
        return self.local[name]

    def release(self):
        """The unit call is over: band buffers, band-local arrays and the copy table go back to the pool now."""
        super().release()
        self.src.clear()
        self.batch = []
        self._tab = None

    def writeback(self, a, b):
        """Owned rows [a, b) of every parent field this band used go back to the parent."""
        la, lb = a - self.w0, b - self.w0
        batched = set(self.batch)
        if self.batch:
            self._launch("band_copy_out", (b - a) * self.im, a * self.im, la * self.im)
        for k in self.live:
            if k not in batched:
                self.src[k][..., a:b, :] = self.bufs[k][..., la:lb, :]


class BandTables(rrtm_chunk.ChunkTables):
    def __init__(self, parent, jm_full, im, xp, lims):
        super().__init__(parent, jm_full, im, xp)
        self._lims = lims          # {w0: device int32 [NLIM]} or None

    def begin(self, w0, w1):
        super().begin(w0, w1)
        if self._lims is not None:
            self._slices[LIMITS_KEY] = self._lims[w0]


def _band_limits(ctx, wins, w):
    """{w0: device NMMB_LIMITS of the band}, one upload per band size, cached in ctx.tables."""
    if LIMITS_KEY not in ctx.tables:
        return None
    key = f"band:limits:{w}"
    hit = ctx.tables.get(key)
    if hit is not None:
        return hit
    import cupy as cp
    from namgpu.steps import drv_loop_limits as dl
    g = ctx.grid
    jm = int(g.jm)
    v = dl.host_limits(int(g.im), jm, bool(ctx.cfg.raw.get("global", False)))
    have = cp.asnumpy(ctx.tables[LIMITS_KEY])  # once per band size (syncs the current stream): one-card table only
    want = np.array([v[n] for n in dl.NAMES], dtype=np.int32)
    if not np.array_equal(have, want):
        raise RuntimeError("physics bands: ctx.tables['NMMB_LIMITS'] is not the one-card table "
                           f"({have.tolist()} vs {want.tolist()}); bands are one-card only")
    rows = []
    for (w0, w1, _a, _b) in wins:
        lo = 1 if w0 == 0 else 2
        hi = w if w1 == jm else w - 1
        r = []
        for n in dl.NAMES:
            x = v[n]
            if n.startswith("j") and n != "jhalo":
                x -= w0
                if n not in _LIMIT_SHIFT_ONLY:
                    x = min(max(x, lo), hi)
            r.append(x)
        rows.append(r)
    dev = cp.asarray(np.array(rows, dtype=np.int32))
    out = {wins[i][0]: dev[i] for i in range(len(wins))}
    ctx.tables[key] = out
    return out


def _band_grid(grid, w0: int, w1: int):
    """Shallow copy of the grid with jm = w1 - w0 and its per-row host arrays row-sliced. Unlike
    rrtm_chunk.chunk_grid it never slices a vertical vector, whatever JM is."""
    g = copy.copy(grid)
    jm, im = int(grid.jm), int(grid.im)
    for k, v in vars(grid).items():
        if not isinstance(v, np.ndarray):
            continue
        if (v.ndim == 2 and v.shape == (jm, im)) or (v.ndim == 1 and k in GRID_ROW_VECTORS and v.shape[0] == jm):
            setattr(g, k, v[w0:w1])
    g.jm = w1 - w0
    return g


class BandCtx:
    """Attribute proxy like rrtm_chunk.ChunkCtx: grid/st/tables are the band views, the rest is the parent."""

    def __init__(self, ctx, w, scratch, wins):
        import cupy as cp
        object.__setattr__(self, "_p", ctx)
        g = ctx.grid
        object.__setattr__(self, "st", BandState(ctx.st, int(g.jm), int(g.im), w, cp, scratch))
        object.__setattr__(self, "tables", BandTables(ctx.tables, int(g.jm), int(g.im), cp,
                                                      _band_limits(ctx, wins, w)))
        object.__setattr__(self, "grid", g)
        object.__setattr__(self, "w", w)

    def begin(self, w0, w1):
        g = _band_grid(self._p.grid, w0, w1)
        object.__setattr__(self, "grid", g)
        self.st.begin(w0, w1, g)
        self.tables.begin(w0, w1)

    def __getattr__(self, name):
        return getattr(self._p, name)

    def __setattr__(self, name, value):
        setattr(self._p, name, value)


def _banded(v, w, im) -> bool:
    import cupy as cp
    if isinstance(v, cp.ndarray):
        return v.ndim >= 2 and tuple(v.shape[-2:]) == (w, im)
    if isinstance(v, dict):
        return any(_banded(x, w, im) for x in v.values())
    if isinstance(v, (tuple, list)):
        return any(_banded(x, w, im) for x in v)
    return False


def rows_for(ctx, uid) -> int:
    """Band rows for unit uid: JM (no bands) or fewer."""
    jm = int(ctx.grid.jm)
    forced = os.environ.get(ENV_ROWS, "").strip()
    if forced:
        n = int(forced)
        return jm if n <= 0 else min(max(n, 3), jm)
    from namgpu.state import fit_on
    if not fit_on() or getattr(ctx, "slab", None) is not None:
        return jm
    import cupy as cp
    free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
    g = ctx.grid
    bpr = ctx.tables.get(f"band:bpr:{uid}") or PLANES_GUESS.get(uid, 48) * (int(g.lm) + 1) * int(g.im) * 4
    explicit = os.environ.get(ENV_MIB, "").strip()
    if not explicit and int(bpr) * jm + (WHOLE_HEADROOM_MIB << 20) <= free:
        return jm                    # the whole-domain call fits: no copies in and out
    budget = int(explicit or DEFAULT_MIB) << 20
    budget = min(budget, free // 2)
    rows = int(budget // max(1, int(bpr)))
    if rows >= jm:
        return jm
    return min(max(rows, MIN_ROWS), jm)


def run(ctx, uid, body, *, setup=None, scratch=frozenset(), keep_whole=(), drop=()):
    """Run body(c) over row bands of ctx, or once on ctx when one band covers the domain.

    setup(ctx): runs once on the whole-domain ctx first (first-call work that writes state or tables).
    scratch: names that may be band-local (proven write-before-read in this unit).
    keep_whole: [LM, JM, IM] names an unbanded unit reads after this one; created whole in the parent.
    drop: the unit's own locals (state.UNIT_SCRATCH, released right after it) that an earlier unit left in the
    parent; dropped before the bands so the unit writes them band by band instead of copying them in and out."""
    if setup is not None:
        setup(ctx)
    jm = int(ctx.grid.jm)
    w = rows_for(ctx, uid)
    if w >= jm:
        return body(ctx)
    st, g = ctx.st, ctx.grid
    for n in keep_whole:
        st.add(n, (int(g.lm), jm, int(g.im)))
    for n in drop:
        st.fields.pop(n, None)
    wins = rrtm_chunk.windows(jm, w)
    bc = BandCtx(ctx, w, frozenset(scratch) - frozenset(keep_whole), wins)
    known = ctx.tables.get(f"band:names:{uid}", ())
    for (w0, w1, a, b) in wins:
        bc.begin(w0, w1)
        bc.st.prefetch(known)
        body(bc)
        bc.st.writeback(a, b)
    ctx.tables[f"band:names:{uid}"] = tuple(sorted(set(known) | set(bc.st.names)))
    nb = bc.st.nbytes()
    if nb:
        first = f"band:bpr:{uid}" not in ctx.tables
        ctx.tables[f"band:bpr:{uid}"] = -(-nb // w)
        if first:
            print(f"physics bands: {uid} {w} rows x {len(wins)} bands of {jm}, band working set "
                  f"{nb / 2**20:.0f} MiB ({nb / w / 1024:.0f} KiB/row, {nb / (w * int(g.im)):.0f} B/col)", flush=True)
    im = int(g.im)
    for k in list(dict.keys(bc.tables)):
        v = dict.__getitem__(bc.tables, k)
        if k not in ctx.tables and not _banded(v, w, im):
            ctx.tables[k] = v
    bc.st.release()
    bc.tables.release()
    return None
