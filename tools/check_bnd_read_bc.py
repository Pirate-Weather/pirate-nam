"""Done-rule check for bnd-read-bc (CONTRACT.md section 6, parts 1 and 2), plus the host I/O path.

    source env.sh
    python tools/check_bnd_read_bc.py

Three parts:
  1. Full size: one random 3-h interval (954 x 835 x 60, lnsh = lnsv = 5) through run(ctx). Every
     destination is checked for NaN/Inf, and 4096 random destination elements per side and class are
     checked against the Fortran column-major record offsets. The W/E destination layout is the
     bnd-write-bc layout [NT][NL][LNSH][JM] (write side: [2, nspec, LM, lines=lnsh, along=JM]).
  2. Round trip through the host I/O: a small synthetic boco.0003 written with Fortran record markers
     (header record, one record per variable, every class and both 3-D orders) is read by
     stage_interval() and placed by run(). Every destination must equal the value it was written from.
     The 3-D H records are checked against the n_bc_3d_h permutation (2, 1).
  3. Negative: place() with the 3-D H slots in the wrong order must raise.
"""
from __future__ import annotations

import struct
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402

from namgpu.boundary import bnd_read_bc as b  # noqa: E402

IM, JM, LM, LNSH, LNSV = 954, 835, 60, 5, 5
NT = b.NT
rng = np.random.default_rng(1)


# ------------------------------------------------------------------ part 1: full size, random strips

def make_class(width, nl, nent):
    pay = b.payload(IM, JM, width, nl)
    rec = cp.asarray(rng.standard_normal((nent, pay), dtype=np.float32))
    ents = []
    for _ in range(nent):
        s = cp.zeros((NT, nl, width, IM), cp.float32)
        n = cp.zeros_like(s)
        w = cp.zeros((NT, nl, width, JM), cp.float32)
        e = cp.zeros_like(w)
        ents.append((s, n, w, e))
    return b.Strips(records=rec, entries=ents, nl=nl, width=width)


def check_class(name, cls):
    S = IM * cls.width * cls.nl * NT
    W = cls.width * JM * cls.nl * NT
    bad = []
    for ei, (s, n, w, e) in enumerate(cls.entries):
        for arr in (s, n, w, e):
            if not bool(cp.isfinite(arr).all()):
                bad.append(f"{name} entry {ei}: NaN/Inf")
        rec = cp.asnumpy(cls.records[ei])
        for side, arr in (("south", s), ("north", n), ("west", w), ("east", e)):
            t = rng.integers(0, NT, 4096)
            lev = rng.integers(0, cls.nl, 4096)
            if side in ("south", "north"):
                n2 = rng.integers(0, cls.width, 4096)
                n1 = rng.integers(0, IM, 4096)
                k = n1 + IM * (n2 + cls.width * (lev + cls.nl * t))      # Fortran (IM, lnsh, NL, NT)
                off = 0 if side == "south" else S
                idx = (t, lev, n2, n1)
            else:
                j = rng.integers(0, JM, 4096)
                n1 = rng.integers(0, cls.width, 4096)
                k = n1 + cls.width * (j + JM * (lev + cls.nl * t))       # Fortran (lnsh, JM, NL, NT)
                off = 2 * S + (0 if side == "west" else W)
                idx = (t, lev, n1, j)
            got = cp.asnumpy(arr[tuple(cp.asarray(x) for x in idx)])
            want = rec[off + k]
            if not np.array_equal(got, want):
                bad.append(f"{name} entry {ei} {side}: {(got != want).sum()} of 4096 misplaced")
    return bad


def part1():
    classes = {
        "h2d": make_class(LNSH, 1, 3),
        "h3d": make_class(LNSH, LM, 4),
        "h4d": make_class(LNSH, LM, 2),
        "v2d": make_class(LNSV, 1, 2),
        "v3d": make_class(LNSV, LM, 2),
    }
    pending = types.SimpleNamespace(header={"runbc": True, "idatbc": [1, 1, 2023], "ihrstbc": 0,
                                            "tboco": 3.0}, n_bc_3d_h=None, **classes)
    ctx = types.SimpleNamespace(grid=types.SimpleNamespace(im=IM, jm=JM, lm=LM), stream=None,
                                bc=types.SimpleNamespace(pending=pending))
    hdr = b.run(ctx)
    cp.cuda.Stream.null.synchronize()
    bad = []
    for name, cls in classes.items():
        bad += check_class(name, cls)
    print("part 1 header:", hdr)
    print("part 1 pending cleared:", ctx.bc.pending is None)
    return bad


# ------------------------------------------------------------------ part 2: Fortran boco round trip

def fortran_record(body: bytes) -> bytes:
    m = struct.pack("<i", len(body))
    return m + body + m


def truth(width, nl, im, jm):
    """C-ordered truth arrays in the destination layout: S/N [NT,NL,W,IM], W/E [NT,NL,W,JM]."""
    s = rng.standard_normal((NT, nl, width, im)).astype(np.float32)
    n = rng.standard_normal((NT, nl, width, im)).astype(np.float32)
    w = rng.standard_normal((NT, nl, width, jm)).astype(np.float32)
    e = rng.standard_normal((NT, nl, width, jm)).astype(np.float32)
    return s, n, w, e


def fortran_bytes(s, n, w, e):
    # Fortran column-major payload: (IM, lnsh, NL, NT) south, north; (lnsh, JM, NL, NT) west, east.
    parts = [s.transpose(3, 2, 1, 0).ravel(order="F"), n.transpose(3, 2, 1, 0).ravel(order="F"),
             w.transpose(2, 3, 1, 0).ravel(order="F"), e.transpose(2, 3, 1, 0).ravel(order="F")]
    return np.concatenate(parts).astype("<f4").tobytes()


def dest(width, nl, im, jm):
    return (cp.zeros((NT, nl, width, im), cp.float32), cp.zeros((NT, nl, width, im), cp.float32),
            cp.zeros((NT, nl, width, jm), cp.float32), cp.zeros((NT, nl, width, jm), cp.float32))


def part2():
    im, jm, lm, lnsh, lnsv = 8, 6, 3, 2, 2
    n_bc_3d_h = (2, 1, 3)
    dims = dict(im=im, jm=jm, lm=lm, lnsh=lnsh, lnsv=lnsv, nvars_bc_2d_h=1, nvars_bc_3d_h=2,
                nvars_bc_4d_h=1, nvars_bc_2d_v=1, nvars_bc_3d_v=2, n_bc_3d_h=n_bc_3d_h,
                lbnd4=1, ubnd4=2)
    # What the file holds, in READ_BC order, and the value each destination must end up with.
    recs = []
    expect = {}                                  # (cls, key) -> truth tuple

    def add(cls_name, key, width, nl):
        t = truth(width, nl, im, jm)
        recs.append(fortran_record(fortran_bytes(*t)))
        expect[(cls_name, key)] = t

    add("h2d", 1, lnsh, 1)
    add("h3d", 2, lnsh, lm)                      # record nv=1 -> storage n_bc_3d_h(1) = 2
    add("h3d", 1, lnsh, lm)                      # record nv=2 -> storage n_bc_3d_h(2) = 1
    for nl in (1, 2):
        add("h4d", (1, nl), lnsh, lm)
    add("v2d", 1, lnsv, 1)
    add("v3d", 1, lnsv, lm)
    add("v3d", 2, lnsv, lm)

    header = struct.pack("<i3iif", 1, 1, 1, 2023, 0, 3.0)
    body = fortran_record(header) + b"".join(recs)
    boco_dir = tempfile.mkdtemp(prefix="bnd_read_bc_")
    with open(f"{boco_dir}/boco.0003", "wb") as f:
        f.write(body)

    # ntsd=1080, dt=10 s: ihr = nint(1080*10/3600) = 3 -> boco.0003
    bnd = {
        "h2d": {1: dest(lnsh, 1, im, jm)},
        "h3d": {1: dest(lnsh, lm, im, jm), 2: dest(lnsh, lm, im, jm), 3: dest(lnsh, lm, im, jm)},
        "h4d": {(1, 1): dest(lnsh, lm, im, jm), (1, 2): dest(lnsh, lm, im, jm)},
        "v2d": {1: dest(lnsv, 1, im, jm)},
        "v3d": {1: dest(lnsv, lm, im, jm), 2: dest(lnsv, lm, im, jm)},
    }
    ctx = types.SimpleNamespace(grid=types.SimpleNamespace(im=im, jm=jm), stream=None,
                                bc=types.SimpleNamespace(pending=None))
    b.stage_interval(ctx, boco_dir, 1080, 10.0, dims, bnd)
    hdr = b.run(ctx)
    cp.cuda.Stream.null.synchronize()

    bad = []
    if hdr != {"runbc": True, "idatbc": (1, 1, 2023), "ihrstbc": 0, "tboco": 3.0}:
        bad.append(f"header {hdr}")
    if ctx.bc.pending is not None:
        bad.append("pending not cleared")
    checks = [("h2d", 1, bnd["h2d"][1]), ("h3d", 2, bnd["h3d"][2]), ("h3d", 1, bnd["h3d"][1]),
              ("h4d", (1, 1), bnd["h4d"][(1, 1)]), ("h4d", (1, 2), bnd["h4d"][(1, 2)]),
              ("v2d", 1, bnd["v2d"][1]), ("v3d", 1, bnd["v3d"][1]), ("v3d", 2, bnd["v3d"][2])]
    for cls_name, key, views in checks:
        want = expect[(cls_name, key)]
        for side, got_dev, want_arr in zip(("south", "north", "west", "east"), views, want):
            got = cp.asnumpy(got_dev)
            if not np.array_equal(got, want_arr):
                bad.append(f"{cls_name} {key} {side}: {(got != want_arr).sum()} values differ")

    # Part 3: a wrong 3-D H slot order must be refused by place().
    neg_ok = False
    try:
        bad_cls = b.Strips(records=cp.zeros((2, b.payload(im, jm, lnsh, lm)), cp.float32),
                           entries=[bnd["h3d"][1], bnd["h3d"][2]], nl=lm, width=lnsh,
                           slots=[1, 2])
        b.place(ctx, bad_cls, im, jm, expect_slots=n_bc_3d_h)
    except ValueError:
        neg_ok = True
    if not neg_ok:
        bad.append("place() accepted entries in the wrong n_bc_3d_h order")

    print("part 2 header:", hdr)
    print("part 2 records:", len(recs), "destinations checked:", len(checks))
    print("part 3 wrong-order refused:", neg_ok)
    return bad


def main():
    bad = part1()
    bad2 = part2()
    bad = bad + bad2
    print("result:", "ok" if not bad else "FAIL")
    for line in bad[:20]:
        print("  ", line)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
