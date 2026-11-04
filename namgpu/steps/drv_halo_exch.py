"""drv-halo-exch: the NMMB halo exchanges between row slabs (MULTICARD-PLAN.md sections 3 and 10).

Ported from the HALO_EXCH calls of nmm/module_SOLVER_GRID_COMP.F90 (NAM nam.v4.2.11, 83df4e13c1), regional and
non-global; the SWAP/POLE/FFT calls are global only and are not called. The plan rows X1..X21 (driver.py) sit after
the unit each Fortran call follows. Every field is exchanged JHALO = 3 rows deep (the Fortran sends 1-3; sending 3
everywhere costs a few hundred KB and removes the off-by-one class).

Kernel:  namgpu/kernels/drv_halo_exch.cu (halo_pack, halo_unpack, row_hash).
Reads/writes: the fields named in EXCH (their halo rows are overwritten with the neighbours' owned rows).
Comm:    ctx.comm (namgpu.parallel.Comm): one NCCL group per exchange, send/recv to rank r-1 and r+1, on ctx.stream.
         No host sync. All fields of one exchange travel in one packed buffer per neighbour and direction.
One card: ctx.comm is None, run() returns at once, so the one-card plan, arrays and output are unchanged.

Debug (env):
  NAMGPU_HALO_ALL=1   exchange every row-slab state field after every plan unit (slow, complete); used once, only if
                      the Fortran schedule fails the bitwise check, to find the exchange the port's unit split moved.
  NAMGPU_UNIT_HASH=1  after every plan unit, a per-row hash of the OWNED rows of every row-slab state field, for the
                      first NAMGPU_UNIT_HASH_STEPS steps (default 3). Written to <out>/unit_hash_r<rank>.npz;
                      tools/unit_hash_compare.py names the first unit and field where N cards differ from one card.
"""
from __future__ import annotations

import os

import numpy as np

from namgpu import kernels
from namgpu import parallel as par
from namgpu.slab import JHALO

UNIT = "drv_halo_exch"

# Exchange rows: (Fortran site, fields). Tracer names (Q, CW, O3, Q2, TRACERS) resolve to the TRACERS array whole
# (Q, CW and Q2 are views of its slots: SOLVER_INTERNAL_STATE 1382-1404), so one exchange carries every species.
EXCH = {
    "X1": ("3415", ("T",)),
    "X2": ("3444-3446", ("DIV", "U", "V")),
    "X3": ("3533", ("T", "U", "V")),
    "X4": ("3657-3662", ("T", "Q", "CW", "Q2", "U", "V")),
    "X5": ("3695-3706", ("T", "Q", "CW", "U", "V", "PD")),      # WRITE_BC only; no write-bc unit on the plan
    "X6": ("4048-4050", ("DIV", "U", "V")),
    "X7": ("4165", ("DIV", "OMGALF")),
    "X8": ("4221", ("U", "V")),
    "X9": ("4284", ("PD", "PSDT", "PSGDT")),
    "X10": ("4409", ("Q",)),
    "X11": ("4570-4580", ("CW", "O3", "Q2", "TRACERS")),
    "X12": ("4681-4684", ("OMGALF", "PINT", "T")),
    "X13": ("4740", ("W",)),
    "X14": ("4797", ("DWDT",)),
    "X15": ("4862-4865", ("DWDT", "T", "W", "PINT")),
    "X16": ("5576", ("DUDT", "DVDT")),
    "X17": ("5630", ("U", "V")),
    "X18": ("5633-5656", ("UZ0", "VZ0", "Q2", "ALBEDO", "EPSR", "QSH", "QWBS", "QZ0", "SOILTB", "THS", "THZ0",
                          "USTAR", "Z0", "TSKIN", "CMC", "SMC", "SH2O", "STC")),
    "X19": ("5819", ("DUDT", "DVDT")),
    "X20": ("5857", ("U", "V")),
    "X21": ("6048-6049", ("T", "Q", "CW")),
}
TRACER_NAMES = {"Q", "CW", "O3", "Q2", "TRACERS"}

_THREADS = 256


def _blocks(n: int) -> int:
    return int(max(1, min((n + _THREADS - 1) // _THREADS, 65535 * 4)))


def _slab_ok(a, jm: int, im: int) -> bool:
    return (a.ndim >= 2 and a.shape[-1] == im and a.shape[-2] == jm and a.dtype.itemsize == 4
            and a.flags.c_contiguous)


def _resolve(ctx, names) -> list:
    """Field arrays for an exchange, tracer names folded into TRACERS, aliases (same memory) once."""
    st, g = ctx.st, ctx.grid
    jm, im = int(g.jm), int(g.im)
    out, seen = [], set()
    for nm in names:
        key = "TRACERS" if nm in TRACER_NAMES and "TRACERS" in st.fields else nm
        a = st.fields.get(key)
        if a is None:
            # produced lazily by a unit that has not run yet (e.g. DUDT before the first TURBL): nothing to send
            ctx.tables.setdefault("HALO_MISSING", set()).add(key)
            continue
        if not _slab_ok(a, jm, im):
            raise RuntimeError(f"drv-halo-exch: {key} shape {a.shape} dtype {a.dtype} is not a contiguous "
                               f"[..., {jm}, {im}] 4-byte field")
        if a.data.ptr in seen:
            continue
        seen.add(a.data.ptr)
        out.append((key, a))
    return out


def _bufs(ctx, tag: str, words: int, peer_lo: bool, peer_hi: bool):
    import cupy as cp
    cache = ctx.tables.setdefault("HALO_BUFS", {})
    b = cache.get(tag)
    if b is None or b["words"] < words:
        mk = lambda ok: cp.empty((words,), dtype=cp.uint32) if ok else None  # noqa: E731
        b = {"words": words, "slo": mk(peer_lo), "rlo": mk(peer_lo), "shi": mk(peer_hi), "rhi": mk(peer_hi)}
        cache[tag] = b
    return b


def exchange(ctx, fields, tag: str):
    """Exchange the halo rows of `fields` [(name, array)] with ranks r-1 and r+1, on ctx.stream, no host sync."""
    comm = par.get(ctx)
    if comm is None or not fields:
        return
    import cupy as cp
    s = ctx.slab
    g = ctx.grid
    jm, im = int(g.jm), int(g.im)
    lo, hi = s.rank > 0, s.rank < s.nranks - 1
    planes = [int(a.size // (jm * im)) for _, a in fields]
    words = sum(planes) * JHALO * im
    b = _bufs(ctx, tag, words, lo, hi)
    pack, unpack = kernels.get(UNIT, "halo_pack"), kernels.get(UNIT, "halo_unpack")
    own0, nown = s.own0, s.nown
    # rows sent: my first JHALO owned rows go down, my last JHALO owned rows go up;
    # rows received: from below into local rows own0-JHALO..own0-1, from above into own0+nown..own0+nown+JHALO-1
    send_lo_r0, send_hi_r0 = own0, own0 + nown - JHALO
    recv_lo_r0, recv_hi_r0 = own0 - JHALO, own0 + nown
    stream = ctx.stream
    with stream:
        off = 0
        for (_, a), p in zip(fields, planes):
            n = p * JHALO * im
            src = a.view(cp.uint32)
            if lo:
                pack((_blocks(n),), (_THREADS,), (src, b["slo"][off:off + n], np.int64(p), np.int32(jm),
                                                 np.int32(im), np.int32(send_lo_r0), np.int32(JHALO)))
            if hi:
                pack((_blocks(n),), (_THREADS,), (src, b["shi"][off:off + n], np.int64(p), np.int32(jm),
                                                 np.int32(im), np.int32(send_hi_r0), np.int32(JHALO)))
            off += n
        pairs = []
        if lo:
            pairs.append((b["slo"][:words], b["rlo"][:words], s.rank - 1))
        if hi:
            pairs.append((b["shi"][:words], b["rhi"][:words], s.rank + 1))
        comm.sendrecv(pairs, stream)
        off = 0
        for (_, a), p in zip(fields, planes):
            n = p * JHALO * im
            dst = a.view(cp.uint32)
            if lo:
                unpack((_blocks(n),), (_THREADS,), (b["rlo"][off:off + n], dst, np.int64(p), np.int32(jm),
                                                   np.int32(im), np.int32(recv_lo_r0), np.int32(JHALO)))
            if hi:
                unpack((_blocks(n),), (_THREADS,), (b["rhi"][off:off + n], dst, np.int64(p), np.int32(jm),
                                                   np.int32(im), np.int32(recv_hi_r0), np.int32(JHALO)))
            off += n


def run(ctx, x: str):
    """Plan entry: exchange row `x` (X1..X21). No-op on one card."""
    if par.get(ctx) is None:
        return
    _site, names = EXCH[x]
    exchange(ctx, _resolve(ctx, names), x)


def exchange_names(ctx, names, tag: str):
    """Exchange named fields (DFI hold variables, quasipost fields). No-op on one card."""
    if par.get(ctx) is None:
        return
    exchange(ctx, _resolve(ctx, names), tag)


# ---- debug modes ------------------------------------------------------------------------------------------------
def slab_fields(ctx) -> list:
    """Every row-slab state field (sorted by name, aliases once)."""
    g = ctx.grid
    jm, im = int(g.jm), int(g.im)
    out, seen = [], set()
    for nm in sorted(ctx.st.fields):
        a = ctx.st.fields[nm]
        if not hasattr(a, "data") or not _slab_ok(a, jm, im) or a.data.ptr in seen:
            continue
        seen.add(a.data.ptr)
        out.append((nm, a))
    return out


HALO_ALL = os.environ.get("NAMGPU_HALO_ALL", "0") == "1"
UNIT_HASH = os.environ.get("NAMGPU_UNIT_HASH", "0") == "1"
HASH_STEPS = int(os.environ.get("NAMGPU_UNIT_HASH_STEPS", "3"))


def exchange_all(ctx):
    if par.get(ctx) is None:
        return
    exchange(ctx, slab_fields(ctx), "ALL")


def hash_unit(ctx, label: str):
    """Per-row hashes of the owned rows of every slab field after a unit (host sync: debug only)."""
    if not UNIT_HASH or ctx.ntimestep >= HASH_STEPS:
        return
    import cupy as cp
    g = ctx.grid
    jm, im = int(g.jm), int(g.im)
    s = getattr(ctx, "slab", None)
    own0, nown, jts = (s.own0, s.nown, s.jts) if s is not None else (0, jm, 1)
    flds = slab_fields(ctx)
    k = kernels.get(UNIT, "row_hash")
    with ctx.stream:
        h = cp.zeros((len(flds), nown), dtype=cp.uint64)
        for n, (_, a) in enumerate(flds):
            p = int(a.size // (jm * im))
            cnt = p * nown * im
            k((_blocks(cnt),), (_THREADS,), (a.view(cp.uint32), np.int64(p), np.int32(jm), np.int32(im),
                                             np.int32(own0), np.int32(nown), h[n]))
        hh = h.get(stream=ctx.stream)
    ctx.stream.synchronize()
    rec = ctx.tables.setdefault("UNIT_HASH", {"keys": [], "rows": [], "jts": jts})
    seq = len(rec["keys"])
    for n, (nm, _) in enumerate(flds):
        rec["keys"].append(f"{ctx.ntimestep:05d}|{seq:05d}|{label}|{nm}")
        rec["rows"].append(hh[n])


def write_hashes(ctx, out_dir):
    rec = ctx.tables.get("UNIT_HASH")
    if not rec:
        return None
    from pathlib import Path
    s = getattr(ctx, "slab", None)
    r = s.rank if s is not None else 0
    p = Path(out_dir) / f"unit_hash_r{r}.npz"
    np.savez(p, keys=np.array(rec["keys"]), rows=np.stack(rec["rows"]) if rec["rows"] else np.zeros((0, 0)),
             jts=np.int64(rec["jts"]))
    return p
