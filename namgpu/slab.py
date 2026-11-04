"""Row-slab decomposition of the B grid (multi-card, MULTICARD-PLAN.md sections 1 and 3).

Interface agreed with lane E2 (MULTICARD-PLAN.md 2026-11-04 17:35Z). Written by E1 at 17:45Z to that spec so the
launcher and the E1 units can run; E2 extends it (helpers only, the fields below stay).

Card r owns global 1-based rows jts..jte and the whole i range. Memory rows are jms..jme with
jms = max(jts - JHALO, 1), jme = min(jte + JHALO, JMG): the edge ranks hold no phantom rows past the domain edge, so
their boundary rows behave exactly as one card. JOFF = jms - 1 (0 on rank 0 and on one card). ctx.grid.jm is the LOCAL
memory row count (jme - jms + 1). No ctx.slab means one card (JOFF 0, JMG = grid.jm).
"""
from __future__ import annotations

from dataclasses import dataclass

JHALO = 3   # module_DM_PARALLEL.F90:24 (IHALO = JHALO = 3)


def split_rows(jmg: int, nranks: int) -> list[tuple[int, int]]:
    """Balanced 1-based (jts, jte) per rank: JMG // N rows each, the first JMG % N ranks one more."""
    base, extra = divmod(int(jmg), int(nranks))
    out, j = [], 1
    for r in range(nranks):
        n = base + (1 if r < extra else 0)
        out.append((j, j + n - 1))
        j += n
    return out


@dataclass(frozen=True)
class Slab:
    rank: int
    nranks: int
    jts: int      # first owned row, global 1-based
    jte: int      # last owned row, global 1-based
    jmg: int      # global JM

    @classmethod
    def for_rank(cls, rank: int, nranks: int, jmg: int) -> "Slab":
        jts, jte = split_rows(jmg, nranks)[rank]
        if nranks > 1 and jte - jts + 1 < JHALO:
            raise ValueError(f"slab: rank {rank} owns {jte - jts + 1} rows < JHALO={JHALO}; the halo would need a "
                             "second neighbour (use fewer cards)")
        return cls(rank, nranks, jts, jte, jmg)

    @property
    def jms(self) -> int:
        return max(self.jts - JHALO, 1)

    @property
    def jme(self) -> int:
        return min(self.jte + JHALO, self.jmg)

    @property
    def jm(self) -> int:
        """Local memory rows."""
        return self.jme - self.jms + 1

    @property
    def joff(self) -> int:
        return self.jms - 1

    @property
    def nown(self) -> int:
        return self.jte - self.jts + 1

    @property
    def own0(self) -> int:
        """First owned row, LOCAL 0-based."""
        return self.jts - self.jms

    @property
    def halo_lo(self) -> int:
        return self.jts - self.jms

    @property
    def halo_hi(self) -> int:
        return self.jme - self.jte


def get(ctx):
    return getattr(ctx, "slab", None)


def rows(ctx) -> tuple[int, int]:
    """(JOFF, JMG) for the kernels: (0, grid.jm) on one card."""
    s = get(ctx)
    if s is None:
        return 0, int(ctx.grid.jm)
    return s.joff, s.jmg


def take_rows(a, ctx, axis: int = -2):
    """Slice a GLOBAL host or device array to this rank's memory rows along `axis` (no copy on one card)."""
    s = get(ctx)
    if s is None:
        return a
    ax = axis % a.ndim
    sl = [slice(None)] * a.ndim
    sl[ax] = slice(s.jms - 1, s.jme)
    return a[tuple(sl)]


def owned(a, ctx, axis: int = -2):
    """This rank's OWNED rows of a LOCAL array (the whole array on one card)."""
    s = get(ctx)
    if s is None:
        return a
    ax = axis % a.ndim
    sl = [slice(None)] * a.ndim
    sl[ax] = slice(s.own0, s.own0 + s.nown)
    return a[tuple(sl)]


def local_lims(ctx, lo_g: int, hi_g: int, reach: int = 0) -> tuple[int, int]:
    """1-based GLOBAL row limits lo_g..hi_g -> 1-based LOCAL limits clipped to memory rows 1..jm.
    `reach` is the kernel's row stencil: at an inner halo edge (not a domain edge) the limit is also
    pulled in by `reach` rows so no read leaves memory (the NMMB_SLAB_EDGE rule on the host).
    Empty (lo > hi) when the range misses this rank. One card: unchanged."""
    s = of(ctx)
    if s is None:
        return int(lo_g), int(hi_g)
    lo_mem = 1 + (reach if s.jms > 1 else 0)
    hi_mem = s.jm - (reach if s.jme < s.jmg else 0)
    return max(int(lo_g) - s.joff, lo_mem), min(int(hi_g) - s.joff, hi_mem)



# ---- E2 helpers (kernels, boundaries, init) ----------------------------------------------------

def of(ctx):
    """The slab when more than one card runs, else None (a one-rank Slab counts as one card)."""
    s = get(ctx)
    return s if (s is not None and s.nranks > 1) else None

def kargs(ctx):
    """(np.int32(JOFF), np.int32(JMG)): the two trailing kernel arguments."""
    import numpy as np
    joff, jmg = rows(ctx)
    return np.int32(joff), np.int32(jmg)



def edges(ctx) -> tuple[bool, bool]:
    """(s_bdy, n_bdy): does this rank's memory start at the domain's south row / end at its north row?
    The Fortran tile flags (module_CONTROL.F90:555): on one card both are True."""
    s = of(ctx)
    if s is None:
        return True, True
    return s.jms == 1, s.jme == s.jmg



def local_grid(g, s):
    """The rank's host grid: a copy of the GLOBAL BGrid with jm = slab.jm and every per-row array
    (tlm_h, tph_h, glat_h, glon_h, glat_v, glon_v [jm, im]; dxh, dxv [jm]) sliced to the memory rows.
    Built from the global arrays, so every value is bitwise the one-card value. sb stays the GLOBAL
    south boundary (use slab.rows for any j -> latitude arithmetic). One card: `g` itself."""
    if s is None or s.nranks <= 1:
        return g
    import dataclasses
    rows = slice(s.jms - 1, s.jme)
    kw = {"jm": s.jm}
    for name in ("tlm_h", "tph_h", "glat_h", "glon_h", "glat_v", "glon_v", "dxh", "dxv"):
        a = getattr(g, name)
        if a.shape[0] != s.jmg:
            raise ValueError(f"slab.local_grid: grid.{name} has {a.shape[0]} rows, expected JMG={s.jmg}")
        kw[name] = a[rows].copy()
    out = dataclasses.replace(g, **kw)
    out.joff = s.joff        # global row of local row 0 (0-based), for j -> rotated-latitude arithmetic
    out.jmg = s.jmg
    return out



# DL_* indices (kernels/nmmb_common.cuh) that hold a j value: jds jde jts jte, jts_b1..jte_b2,
# jts_h1..jte_h2, jts_b1_h1 jte_b1_h1 jte_b1_h2, jds_p1 jde_m1 jde_m2, jms jme.
_J_ENTRIES = (2, 3, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 27, 28, 29, 33, 34, 35, 40, 41)


def local_limits(ctx):
    """The drv-loop-limits table with every j entry shifted to LOCAL 1-based memory rows (j - JOFF).
    For kernels that compare a local j0 with tile limits (every comparison is shift-invariant, so the
    kernel code is unchanged). Built once on the device from ctx.tables["NMMB_LIMITS"], no host sync.
    One card: the NMMB_LIMITS table itself."""
    from namgpu.steps import drv_loop_limits as dll
    lim = dll.table(ctx)
    s = of(ctx)
    if s is None:
        return lim
    key = "NMMB_LIMITS_LOCAL"
    if key not in ctx.tables:
        import cupy as cp
        shift = cp.zeros(lim.shape, dtype=cp.int32)
        shift[list(_J_ENTRIES)] = s.joff
        ctx.tables[key] = lim - shift
    return ctx.tables[key]


def halo_exchange(ctx, arr, tag: str) -> None:
    """Exchange the halo rows of one [.., JM, IM] device array with the neighbour ranks on ctx.stream (E1's
    drv-halo-exch). One card: nothing. For iterative units whose every sweep reads the previous sweep's
    neighbours (io-quasipost relax4e), where the Fortran exchanges each sweep."""
    if of(ctx) is None:
        return
    from namgpu.steps import drv_halo_exch as hx
    hx.exchange(ctx, [(tag, arr)], tag)


def allreduce(ctx, arr, op: str) -> None:
    """In-place device allreduce ('max', 'min', 'sum') of a small array across ranks on ctx.stream (E1's
    communicator). One card: nothing. A domain-wide flag that differed between ranks would make their loops
    diverge (and an exchange inside them hang)."""
    if of(ctx) is None:
        return
    from namgpu import parallel as par
    comm = par.get(ctx)
    if comm is None:
        raise RuntimeError("slab.allreduce: ctx.slab says multi-card but ctx.comm is missing")
    comm.allreduce(arr, arr, op, ctx.stream)
