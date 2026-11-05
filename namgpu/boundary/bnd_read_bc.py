"""bnd-read-bc: read one staged 3-h boundary interval (boco.NNNN) and place it into the boundary arrays
(NMMB READ_BC).

Ported from nmm/module_FLTBNDS.F90:3579-3925 (nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/bnd_read_bc.cu (bnd_read_bc_scatter, one launch per variable class per
interval; no host sync; pointer tables are cached per stream and grown only on demand).

Scope. READ_BC does no arithmetic: it opens boco.NNNN, reads a header and one Fortran unformatted
record per variable, and copies strips into bnd_vars. Each record carries both time slots (n3 = 1
value, n3 = 2 tendency, READ_BC lines 3600+), so on the uppermost domain (the NAM 12 km parent, not a
nest) the tendency comes from the file and this unit places it. It does not form tendencies. Only a
nest forms them, in UPDATE_BC_TENDS (module_SOLVER_GRID_COMP.F90:8155), which is init-boco-tendencies.
The nest path is selected by I_AM_A_NEST (SOLVER line 3820), not by the staging of a file.

Two halves:
  stage_interval()  host I/O. Opens boco.NNNN for the interval (name from
                    ihr = nint(ntsd*|dt|/3600) in single precision, as Fortran evaluates it), decodes
                    the header record (runbc, idatbc(3), ihrstbc, tboco), reads every variable record
                    in READ_BC order (2-D H, 3-D H, 4-D H, 2-D V, 3-D V; Fortran record markers
                    stripped), checks each payload's byte count, uploads the records to the device as
                    float32 and binds every record to its destination views. The 3-D H records are
                    bound in n_bc_3d_h order (Fortran nvx = n_bc_3d_h(nv)); the 4-D H records are bound
                    per (nv, nl) with nl running lbnd..ubnd. Sets ctx.bc.pending.
  run()             device placement. One kernel launch per class; checks the slot and tag order that
                    stage_interval() produced, so a caller cannot pass the entries out of order.

Staged record payload (per entry, float32, one Fortran record):
    south, north (IM wide) then west, east (JM tall), each in Fortran column-major order.
Destination views (device, C-contiguous float32), per entry (south, north, west, east):
    south/north: [NT][NL][LNSH][IM]   (Fortran (n1,n2,n3,n4) -> C [n4][n3][n2][n1])
    west/east:   [NT][NL][JM][LNSH]   (CONTRACT 1a/6a; was [NT][NL][LNSH][JM] before 2026-11-04, which
                                      transposed every W/E strip against bocoh/bocov; the old note said
                                      it matched the write-side layout of bnd-write-bc: [2, nspec, LM, lines, along]
                                      with lines = lnsh and along = JM for W/E; this is the same order
                                      init-boco-tendencies consumes)
The destination storage (bnd_vars_h / bnd_vars_v) is the boundary group's; it is passed to
stage_interval() as `bnd`, a mapping of class -> {slot: (south, north, west, east)} where slot is the
Fortran storage index (2-D: nv; 3-D H: nvx; 4-D H: (nv, nl); 3-D V: nv).

Each class is placed with one kernel launch. run() clears ctx.bc.pending when done and returns the
header, so the driver runs it once per interval.
"""
from __future__ import annotations

import contextlib
import math
import os
import struct
from dataclasses import dataclass, field

import numpy as np

from namgpu import kernels, slab

UNIT = "bnd_read_bc"
KERNEL = "bnd_read_bc_scatter"
NT = 2            # time levels at the two ends of the 3-h interval (Fortran n3 / n4 = 1, 2)
BLOCK = 256       # threads per block for the scatter (tuning knob, kept here)
CLASSES = ("h2d", "h3d", "h4d", "v2d", "v3d")
HEADER_FMT = "<i3iif"   # runbc (Fortran logical, 4 bytes), idatbc(1:3), ihrstbc, tboco (kfpt single)
HEADER_BYTES = struct.calcsize(HEADER_FMT)


@dataclass
class Strips:
    """One class of boundary variables staged for one interval."""
    records: object                       # cupy float32 [nent, payload(width, nl)]
    entries: list = field(default_factory=list)   # [(south, north, west, east), ...] record order
    nl: int = 1                           # 1 for 2-D, LM for 3-D and 4-D
    width: int = 5                        # lnsh (H) or lnsv (V)
    slots: list | None = None             # h3d, h2d, v2d, v3d: storage index of each entry
    tags: list | None = None              # h4d: (nv, nl) of each entry, record order
    lbnd: int | None = None               # h4d: lbound of the 4th index
    ubnd: int | None = None               # h4d: ubound of the 4th index


@dataclass
class PendingInterval:
    header: dict
    h2d: object = None
    h3d: object = None
    h4d: object = None
    v2d: object = None
    v3d: object = None
    n_bc_3d_h: tuple | None = None        # Fortran n_bc_3d_h(1:3): storage order of the 3-D H vars


def payload(im: int, jm: int, width: int, nl: int) -> int:
    """Float32 count of one staged record: south + north (IM wide) + west + east (JM tall)."""
    s = im * width * nl * NT
    w = width * jm * nl * NT
    return 2 * s + 2 * w


def _entry_shapes(im, jm, width, nl):
    return ((NT, nl, width, im), (NT, nl, jm, width))   # W/E: CONTRACT 1a, [.., JM, lnsh]


# ---------------------------------------------------------------- host I/O

def _fortran_nint(x: float) -> int:
    """Fortran NINT: round half away from zero."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def boco_file_name(ntsd: int, dt: float) -> str:
    """'boco.NNNN' for the interval containing step ntsd (Fortran: ihr = nint(ntsd*abs(dt)/3600.))."""
    ihr = _fortran_nint(float(np.float32(ntsd) * np.float32(abs(dt)) / np.float32(3600.0)))
    return f"boco.{ihr:04d}"


def _records(f):
    """Yield the payload of each Fortran sequential unformatted record (little-endian markers)."""
    while True:
        head = f.read(4)
        if not head:
            return
        if len(head) != 4:
            raise ValueError("bnd-read-bc: truncated record marker")
        (n,) = struct.unpack("<i", head)
        body = f.read(n)
        tail = f.read(4)
        if len(body) != n or len(tail) != 4:
            raise ValueError(f"bnd-read-bc: truncated record (want {n} bytes)")
        (m,) = struct.unpack("<i", tail)
        if m != n:
            raise ValueError(f"bnd-read-bc: record markers disagree ({n} != {m})")
        yield body


def _next_payload(recs, want_floats: int, what: str) -> np.ndarray:
    body = next(recs, None)
    if body is None:
        raise ValueError(f"bnd-read-bc: boco file ends before {what}")
    if len(body) != 4 * want_floats:
        raise ValueError(f"bnd-read-bc: {what} record is {len(body)} bytes, want {4 * want_floats}")
    return np.frombuffer(body, dtype="<f4")


def stage_interval(ctx, boco_dir: str, ntsd: int, dt: float, dims: dict, bnd: dict) -> None:
    """Host I/O for one 3-h interval: read boco.NNNN and set ctx.bc.pending for run().

    dims: im, jm, lm, lnsh, lnsv; nvars_bc_2d_h, nvars_bc_3d_h, nvars_bc_4d_h, nvars_bc_2d_v,
          nvars_bc_3d_v; n_bc_3d_h (the storage indices of domain #1's 3-D H vars, in record
          order); lbnd4, ubnd4 (bounds of the 4-D H fourth index, used when nvars_bc_4d_h > 0).
    bnd:  destination views, see the module docstring.
    """
    import cupy as cp

    im, jm, lm = int(dims["im"]), int(dims["jm"]), int(dims["lm"])
    lnsh, lnsv = int(dims["lnsh"]), int(dims["lnsv"])
    n2h, n3h, n4h = int(dims["nvars_bc_2d_h"]), int(dims["nvars_bc_3d_h"]), int(dims["nvars_bc_4d_h"])
    n2v, n3v = int(dims["nvars_bc_2d_v"]), int(dims["nvars_bc_3d_v"])
    n_bc_3d_h = tuple(int(x) for x in dims["n_bc_3d_h"])
    if n3h:
        if len(n_bc_3d_h) < n3h:
            raise ValueError(f"bnd-read-bc: n_bc_3d_h has {len(n_bc_3d_h)} entries, want {n3h}")
        if sorted(n_bc_3d_h[:n3h]) != list(range(1, n3h + 1)):
            raise ValueError(f"bnd-read-bc: n_bc_3d_h {n_bc_3d_h} is not a permutation of 1..{n3h}")
    lbnd4, ubnd4 = int(dims.get("lbnd4", 1)), int(dims.get("ubnd4", 0))
    n4_levels = ubnd4 - lbnd4 + 1

    path = os.path.join(boco_dir, boco_file_name(ntsd, dt))
    with open(path, "rb") as f:
        recs = _records(f)
        hdr = next(recs, None)
        if hdr is None or len(hdr) != HEADER_BYTES:
            raise ValueError(f"bnd-read-bc: {path} header record is not {HEADER_BYTES} bytes")
        runbc, d1, d2, d3, ihrstbc, tboco = struct.unpack(HEADER_FMT, hdr)
        header = dict(runbc=bool(runbc), idatbc=(d1, d2, d3), ihrstbc=ihrstbc, tboco=float(tboco))

        rows_2dh = [_next_payload(recs, payload(im, jm, lnsh, 1), "2-D H") for _ in range(n2h)]
        rows_3dh = [_next_payload(recs, payload(im, jm, lnsh, lm), "3-D H") for _ in range(n3h)]
        rows_4dh = [_next_payload(recs, payload(im, jm, lnsh, lm), "4-D H")
                    for _ in range(n4h * max(n4_levels, 0))]
        rows_2dv = [_next_payload(recs, payload(im, jm, lnsv, 1), "2-D V") for _ in range(n2v)]
        rows_3dv = [_next_payload(recs, payload(im, jm, lnsv, lm), "3-D V") for _ in range(n3v)]
        if next(recs, None) is not None:
            raise ValueError(f"bnd-read-bc: {path} holds more records than READ_BC reads")

    def upload(rows):
        return cp.asarray(np.stack(rows)) if rows else None

    h2d = h3d = h4d = v2d = v3d = None
    if n2h:
        slots = list(range(1, n2h + 1))                     # record nv -> storage nv
        h2d = Strips(records=upload(rows_2dh), entries=[bnd["h2d"][s] for s in slots],
                     nl=1, width=lnsh, slots=slots)
    if n3h:
        slots = list(n_bc_3d_h[:n3h])                       # record nv -> storage n_bc_3d_h(nv)
        h3d = Strips(records=upload(rows_3dh), entries=[bnd["h3d"][s] for s in slots],
                     nl=lm, width=lnsh, slots=slots)
    if n4h:
        tags = [(nv, nl) for nv in range(1, n4h + 1) for nl in range(lbnd4, ubnd4 + 1)]
        h4d = Strips(records=upload(rows_4dh), entries=[bnd["h4d"][t] for t in tags],
                     nl=lm, width=lnsh, tags=tags, lbnd=lbnd4, ubnd=ubnd4)
    if n2v:
        slots = list(range(1, n2v + 1))
        v2d = Strips(records=upload(rows_2dv), entries=[bnd["v2d"][s] for s in slots],
                     nl=1, width=lnsv, slots=slots)
    if n3v:
        slots = list(range(1, n3v + 1))
        v3d = Strips(records=upload(rows_3dv), entries=[bnd["v3d"][s] for s in slots],
                     nl=lm, width=lnsv, slots=slots)

    pend = PendingInterval(header=header, h2d=h2d, h3d=h3d, h4d=h4d, v2d=v2d, v3d=v3d,
                           n_bc_3d_h=n_bc_3d_h[:n3h] if n3h else None)
    if not isinstance(ctx.bc, dict):
        ctx.bc.pending = pend
    return pend


# ---------------------------------------------------------------- device placement

_TABLE_CACHE: dict = {}


def _table_buffers(stream_key, n: int, cur):
    """Four uint64 pointer-table buffers for one stream, grown (never shrunk) to hold n entries.

    Growth synchronises the stream first, so no kernel still queued on it reads the buffers being
    replaced. Steady state allocates nothing.
    """
    import cupy as cp

    bufs = _TABLE_CACHE.get(stream_key)
    if bufs is None or bufs[0].shape[0] < n:
        old = bufs[0].shape[0] if bufs is not None else 0
        if bufs is not None:
            cur.synchronize()
        cap = max(n, 2 * old, 16)
        bufs = [cp.empty(cap, dtype=cp.uint64) for _ in range(4)]
        _TABLE_CACHE[stream_key] = bufs
    return bufs


def _check_view(arr, shape, what):
    import cupy as cp
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"bnd-read-bc: {what} shape {tuple(arr.shape)}, want {tuple(shape)}")
    if arr.dtype != cp.float32:
        raise ValueError(f"bnd-read-bc: {what} dtype {arr.dtype}, want float32")
    if not arr.flags.c_contiguous:
        raise ValueError(f"bnd-read-bc: {what} is not C-contiguous")


def place(ctx, cls: Strips, im: int, jm: int, expect_slots=None) -> None:
    """Scatter every entry of one class in one kernel launch on ctx.stream.
    Row slab: jm is the LOCAL row count; the records hold the GLOBAL JMG rows and the W/E strips
    receive rows jms..jme (slab.rows). One card: JOFF = 0, JMG = jm."""
    import cupy as cp

    n = len(cls.entries)
    if n == 0:
        return
    joff, jmg = slab.rows(ctx)
    expect = payload(im, jmg, cls.width, cls.nl)
    _check_view(cls.records, (n, expect), "records")
    sn_shape, we_shape = _entry_shapes(im, jm, cls.width, cls.nl)
    for k, (s, nn, w, ee) in enumerate(cls.entries):
        _check_view(s, sn_shape, f"entry {k} south")
        _check_view(nn, sn_shape, f"entry {k} north")
        _check_view(w, we_shape, f"entry {k} west")
        _check_view(ee, we_shape, f"entry {k} east")

    if cls.tags is not None:
        span = cls.ubnd - cls.lbnd + 1
        want = [(nv, nl) for nv in range(1, n // span + 1) for nl in range(cls.lbnd, cls.ubnd + 1)]
        if list(cls.tags) != want:
            raise ValueError("bnd-read-bc: 4-D entries are not in record order "
                             f"(nv outer, nl = {cls.lbnd}..{cls.ubnd} inner)")
    if expect_slots is not None:
        # 3-D H: entry k must be storage n_bc_3d_h(k+1) (Fortran nvx = n_bc_3d_h(nv)).
        if cls.slots is None or list(cls.slots) != list(expect_slots)[:n]:
            raise ValueError(f"bnd-read-bc: entries bound to slots {cls.slots}, "
                             f"n_bc_3d_h says {list(expect_slots)[:n]}")

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    stream_key = id(ctx.stream) if ctx.stream is not None else 0
    host = [np.array([e[k].data.ptr for e in cls.entries], dtype=np.uint64) for k in range(4)]
    with scope:
        cur = cp.cuda.get_current_stream()
        bufs = _table_buffers(stream_key, n, cur)
        for buf, h in zip(bufs, host):
            buf[:n].set(h, stream=cur)                  # H2D on ctx.stream, same stream as the kernel
        strip = max(cls.nl * cls.width * im, cls.nl * cls.width * jm) * NT
        grid = ((strip + BLOCK - 1) // BLOCK, 4, n)     # blockIdx.y = side, blockIdx.z = entry
        kern = kernels.get(UNIT, KERNEL)
        kern(grid, (BLOCK, 1, 1), (
            cls.records, bufs[0], bufs[1], bufs[2], bufs[3],
            np.int32(im), np.int32(jm), np.int32(cls.width), np.int32(cls.nl),
            np.int32(NT), np.int64(expect), np.int32(joff), np.int32(jmg),
        ))


PARENT_DIMS = dict(nvars_bc_2d_h=1, nvars_bc_3d_h=3, nvars_bc_4d_h=0, nvars_bc_2d_v=0,
                   nvars_bc_3d_v=2, n_bc_3d_h=(1, 2, 3))   # module_SOLVER_GRID_COMP.F90:7658-7664
H2D, H3D, V3D = ("PD",), ("T", "Q", "CW"), ("U", "V")


def readbc_due(ctx, nboco: int) -> bool:
    """Parent READBC at the not_firststep call site (module_SOLVER_GRID_COMP.F90:3884-3896):
    NTIMESTEP_BC == 1 or MOD(NTIMESTEP_BC, NBOCO) == 0 with FILTER_METHOD == 0. Inside the TDFI legs
    (filter_active) only the first step of a leg reads: each leg is 600 s, inside one interval."""
    n = int(ctx.ntimestep)
    if getattr(ctx, "filter_active", False):
        # WRITE_BC_FLAG (module_SOLVER_GRID_COMP.F90:3679-3686): with FILTER_METHOD > 0, a first step after
        # the boundaries were already read (BDY_WAS_READ) does not READ_BC. So only the backward leg
        # reads boco.0000; the forward leg keeps the changedir WRITE_BC values (drv-dfi-tdfi-driver).
        return n == 1 and not ctx.tables.get("bdy_was_read", False)
    return n == 1 or n % nboco == 0


def dict_destinations(ctx) -> dict:
    """bnd mapping for stage_interval() onto the ctx.bc dict strips (CONTRACT 6a): reshaped views,
    no copy, so the scatter writes the very arrays bnd-bocoh-update/-blend and bnd-bocov-* read."""
    bc, g = ctx.bc, ctx.grid
    im, jm = int(g.im), int(g.jm)

    def views(name, nl, w):
        return (bc[f"S_{name}"].reshape(NT, nl, w, im), bc[f"N_{name}"].reshape(NT, nl, w, im),
                bc[f"W_{name}"].reshape(NT, nl, jm, w), bc[f"E_{name}"].reshape(NT, nl, jm, w))
    lm, lnsh, lnsv = int(g.lm), int(ctx.cfg.lnsh), int(ctx.cfg.lnsv)
    return {"h2d": {1: views("PD", 1, lnsh)},
            "h3d": {k + 1: views(n, lm, lnsh) for k, n in enumerate(H3D)},
            "h4d": {}, "v2d": {},
            "v3d": {k + 1: views(n, lm, lnsv) for k, n in enumerate(V3D)}}


def run(ctx):
    """Driver entry.
    Parent with boco files (ctx.boco_dir from init-gfs-bocos, ctx.bc the dict of bnd-boundary-init):
    READ_BC on its cadence: read boco.NNNN on the host, place both slots on the device.
    Otherwise: acts only when stage_interval() has staged an interval (ctx.bc.pending)."""
    bc = ctx.bc
    if getattr(ctx, "concurrent_parent", False):
        # a concurrent nest (namgpu/nest/twoway.py): its boundary tendencies come from the live parent every parent
        # step (PARENT_UPDATE_CHILD_BNDRY + UPDATE_BC_TENDS, kernels/nest_bc.cu), never from boco files
        return None
    if isinstance(bc, dict):
        boco_dir = getattr(ctx, "boco_dir", None)
        if boco_dir is None:
            raise RuntimeError("bnd-read-bc: no ctx.boco_dir: the parent needs its lateral boundary files "
                               "(init-gfs-bocos writes them); a forecast without them would hold the "
                               "boundaries at the analysis edges")
        tboco = float(ctx.tables.get("consts", {}).get("tboco", 10800.0))
        nboco = int(round(tboco / abs(float(ctx.dt))))
        if not readbc_due(ctx, nboco):
            return None
        g = ctx.grid
        dims = dict(PARENT_DIMS, im=g.im, jm=slab.rows(ctx)[1], lm=g.lm,   # the file holds GLOBAL rows
                    lnsh=ctx.cfg.lnsh, lnsv=ctx.cfg.lnsv)
        pend = stage_interval(ctx, boco_dir, int(ctx.ntimestep), float(ctx.dt), dims,
                              dict_destinations(ctx))
        for name in CLASSES:
            cls = getattr(pend, name)
            if cls is not None:
                place(ctx, cls, int(g.im), int(g.jm),
                      expect_slots=pend.n_bc_3d_h if name == "h3d" else None)
        ctx.bc_last_header = dict(pend.header)
        ctx.tables["bdy_was_read"] = True          # Fortran BDY_WAS_READ (3953-3955)
        return dict(pend.header)
    pending = getattr(bc, "pending", None) if bc is not None else None
    if pending is None:
        return None
    g = ctx.grid
    im, jm = int(g.im), int(g.jm)
    for name in CLASSES:
        cls = getattr(pending, name)
        if cls is not None:
            place(ctx, cls, im, jm, expect_slots=pending.n_bc_3d_h if name == "h3d" else None)
    bc.pending = None
    bc.last_header = dict(pending.header)
    return dict(pending.header)
