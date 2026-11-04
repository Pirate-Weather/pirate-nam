"""dyn-mono-fix: MONO setup and per-species monotonisation (allreduce branch, local limiter sums).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:4768-5075 (NAM nam.v4.2.11, 83df4e13c1), MONO.
Kernel: namgpu/kernels/dyn_mono_fix.cu (dyn_mono_fix, one thread per (i, j) with the level in grid.z,
        one launch, species loop inside the thread, warp shuffle then one atomicAdd per warp).

Reads:   TRACERS [NTR, LM, JM, IM] (s, read only), TRACERS_SQRT [NTR, LM, JM, IM] (the Fortran s1, read on
         the allreduce branch; the ADV2 setup writes it, dyn-adv2-q2-mid/horiz), PD [JM, IM], DSG2 [LM]
         (state), DARE [JM] (= DXH(j)*DYH, computed here from ctx.grid and cached in ctx.tables), PDSG1 [LM]
         (ctx.grid.pdsg1, cached in ctx.tables).
Writes:  TRACERS_TEND [NTR, LM, JM, IM] (tcs): interior points get tcs = s1p - s with s1p = (s1+tcs)**2
         (the Fortran's first assignment). Boundary rows are left unchanged.
         Allreduce branch: MONO_XSUMS [2*NTR] float64 in ctx.tables and returned: entry 2*(k-1) is the sum of
         the positive limited changes dsp for tracer slot k, entry 2*(k-1)+1 the sum of the non-positive ones.
         These are the local sums the dyn-mono-global-sum unit reduces (device reduction in place of
         MPI_Allreduce). The buffer is allocated once and zeroed on the device each call.
         Send/recv branch: TRACERS_SQRT is overwritten with dsp on the interior (Fortran send/recv branch, 4997-5072); the column
         sums are formed by dyn-mono-global-sum.
Not written here: the rescale and the tcs += dsp correction (dyn-mono-global-sum), the Q2
         back-interpolation (dyn-mono-q2-back).
Ranges:  kss, kse and idtadt are required (no defaults): the caller passes the 1-based tracer slots of its
         MONO call and IDTADTQ/IDTADT. The driver resolves them (driver.py plan rows, SOLVER_RUN 4370 and
         4519). A default would limit the wrong species or the wrong cadence silently.
Branch:  use_allreduce defaults to parm/nam_configfile (cfg.raw["use_allreduce"], true for the 12 km parent).
         Both branches are ported; the send/recv one is completed by dyn-mono-global-sum.
Launch:  one kernel on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_mono_fix"
KERNEL = "dyn_mono_fix"


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx, kss: int, kse: int, idtadt: int, use_allreduce: bool | None = None):
    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    if use_allreduce is None:
        use_allreduce = bool(cfg.raw["use_allreduce"])
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ntr = int(st.TRACERS.shape[0])
    kss = int(kss)
    kse = int(kse)
    if not (1 <= kss <= kse <= ntr):
        raise ValueError(f"dyn-mono-fix: tracer range {kss}..{kse} outside 1..{ntr}")
    idtadt = int(idtadt)
    if g.pdsg1 is None or g.dsg2 is None:
        raise RuntimeError(
            "dyn-mono-fix: vertical coordinate not set (grid.set_vcoord); dsg2/pdsg1 missing")

    s = st.TRACERS
    s1 = st.add("TRACERS_SQRT", (ntr, lm, jm, im))
    tcs = st.add("TRACERS_TEND", (ntr, lm, jm, im))
    dsg2 = st["DSG2"]
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    dxh = np.asarray(g.dxh, dtype=np.float32)
    if dxh.shape[0] != jm:
        # several cards: grid.dxh is still the global row table; take this slab's memory rows (namgpu/slab.py)
        from namgpu import slab as _slab
        dxh = np.asarray(_slab.take_rows(dxh, ctx, axis=0))
    dare = _device_vector(ctx, "dare", dxh * np.float32(g.dyh))

    import cupy as cp
    from namgpu import parallel as par
    from namgpu import slab as slabmod

    xsums = ctx.tables.get("MONO_XSUMS")
    if xsums is None or xsums.shape != (2 * ntr,):
        xsums = cp.zeros((2 * ntr,), dtype=cp.float64)
        ctx.tables["MONO_XSUMS"] = xsums
    joff, jmg = slabmod.rows(ctx)
    comm = par.get(ctx)
    if comm is not None and not use_allreduce:
        raise RuntimeError("dyn-mono-fix: use_allreduce=false on several cards is refused: its per-column sums "
                           "(dyn-mono-global-sum send/recv branch) are not decomposition-invariant yet; the NAM "
                           "configfile sets use_allreduce true")
    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    nwx = int(grid[0])
    nks = kse - kss + 1
    wkey = f"MONO_WPART_{nks}"
    wpart = ctx.tables.get(wkey)
    if wpart is None or wpart.size != nks * 2 * lm * jm * nwx:
        wpart = cp.empty((nks * 2 * lm * jm * nwx,), dtype=cp.float64)
        ctx.tables[wkey] = wpart
    args = (
        np.int32(im), np.int32(jm), np.int32(lm), np.int32(idtadt),
        np.int32(kss), np.int32(kse), np.int32(ntr), np.int32(1 if use_allreduce else 0),
        s, s1, tcs, st.PD, dare, dsg2, pdsg1, np.int32(joff), np.int32(jmg), np.int32(nwx), wpart,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
        if use_allreduce:
            _deterministic_sums(ctx, cp, comm, jm, lm, nwx, kss, kse, ntr, wpart, xsums)
    return xsums


def _deterministic_sums(ctx, cp, comm, jm, lm, nwx, kss, kse, ntr, wpart, xsums):
    """Per-row partials (one block per row, fixed tree), owned rows allgathered over the cards, then a fixed-order
    sum over the global rows. The same code on one card (no gather), so one card and N cards give the same bits."""
    rowp = ctx.tables.get("MONO_ROWP")
    if rowp is None or rowp.shape != (jm, 2 * ntr):
        rowp = ctx.tables["MONO_ROWP"] = cp.empty((jm, 2 * ntr), dtype=cp.float64)
    kernels.get(UNIT, "mono_row_partials")((jm, 1, 1), (256, 1, 1),
                                            (np.int32(jm), np.int32(lm), np.int32(nwx), np.int32(kss),
                                             np.int32(kse), np.int32(ntr), wpart, rowp))
    s = getattr(ctx, "slab", None)
    if comm is None:
        nranks, nmax, growp = 1, jm, rowp
        nown = ctx.tables.get("MONO_NOWN1")
        if nown is None or int(ctx.tables.get("MONO_NOWN1_JM", -1)) != jm:
            nown = ctx.tables["MONO_NOWN1"] = cp.asarray(np.array([jm], dtype=np.int32))
            ctx.tables["MONO_NOWN1_JM"] = jm
    else:
        from namgpu.slab import split_rows
        parts = split_rows(s.jmg, s.nranks)
        nranks = s.nranks
        nmax = max(b - a + 1 for a, b in parts)
        g = ctx.tables.get("MONO_GATHER")
        if g is None:
            g = ctx.tables["MONO_GATHER"] = {
                "send": cp.zeros((nmax, 2 * ntr), dtype=cp.float64),
                "recv": cp.empty((nranks, nmax, 2 * ntr), dtype=cp.float64),
                "nown": cp.asarray(np.array([b - a + 1 for a, b in parts], dtype=np.int32)),
            }
        g["send"][:s.nown] = rowp[s.own0:s.own0 + s.nown]
        comm.allgather(g["send"], g["recv"], ctx.stream)
        growp, nown = g["recv"], g["nown"]
    kernels.get(UNIT, "mono_rows_total")(((2 * ntr + 63) // 64, 1, 1), (64, 1, 1),
                                          (np.int32(nranks), np.int32(nmax), nown, np.int32(ntr), np.int32(kss),
                                           np.int32(kse), growp, xsums))
