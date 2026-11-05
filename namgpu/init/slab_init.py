"""Init once, slice per rank (multi-card NAM, MULTICARD-PLAN.md section 6; lane E2).

Why: the init chain is not reproducible run to run (init-gwd-orog and init-vcoord-recover sum with float
atomics), so two processes that each run init would start from different bits and no 1-card vs N-card
comparison could pass. And at 4 km the whole-domain init does not fit one 32 GB card. So the init chain
runs ONCE over the whole domain (on one card; --managed puts the CuPy pool in CUDA managed memory so a
4 km domain can spill to host RAM), the whole post-init context is written to a directory (normally on
/dev/shm), and every forecast process, the one-card run included, loads it and keeps only its rows.

What is saved: every device and host array reachable from ctx.st.fields, ctx.bc and ctx.tables (nested
dicts, lists, tuples, SimpleNamespace), the grid's vertical coordinate, and the plain ctx attributes init
sets (boco_dir, restart, ...). Aliases are kept: arrays that share one allocation (Q = TRACERS[k],
PINT[1:], FAH in st and tables, bc reshaped views) are stored once, by their owner, and rebuilt as views.

Row slicing on load (ctx.slab set): an owner array whose axis -2 has JMG entries ([.., JM, IM] state,
[.., JM, lnsh] W/E strips) is cut to rows jms..jme on that axis; a 1-D array of JMG entries (the per-row
grid constants FAH, DARE, DXH, WPDAR, HDAC*, F, ...) is cut the same way. Views are rebuilt on the cut
owner. The cut shapes are listed in <dir>/load-receipt-r<rank>.json. Per-rank tables are dropped and
rebuilt by their units: NMMB_LIMITS (drv-loop-limits), NMMB_LIMITS_LOCAL, DIVCOR_LIMITS_CHECKED.

    python -m namgpu.init.slab_init save --case 2025-03-14_12z --source gfs --out /dev/shm/nam-init [--managed]
    forecast side:  ctx.slab = ...; ctx.grid = slab.local_grid(g, ctx.slab); slab_init.load(ctx, dir)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PER_RANK_TABLES = ("NMMB_LIMITS", "NMMB_LIMITS_LOCAL", "NMMB_LIMITS_HOST", "DIVCOR_LIMITS_CHECKED",
                   "HALO_BUFS", "HALO_MISSING", "UNIT_HASH")
CTX_SKIP = {"cfg", "grid", "st", "stream", "driver", "bc", "tables", "timings", "slab", "halo",
            "out_dir", "forecast_hours", "comm", "decomp", "hist_geo_global",
            "restart_writer", "history_future", "history_select"}   # the forecast process keeps its own
FORMAT = 1


class _Ref:
    """Placeholder for an array in the pickled skeleton."""
    __slots__ = ("k",)

    def __init__(self, k):
        self.k = k

    def __getstate__(self):
        return self.k

    def __setstate__(self, k):
        self.k = k


class _Skipped:
    __slots__ = ("what",)

    def __init__(self, what):
        self.what = what

    def __getstate__(self):
        return self.what

    def __setstate__(self, w):
        self.what = w


def _is_cp(a):
    try:
        import cupy as cp
        return isinstance(a, cp.ndarray)
    except ImportError:
        return False


# ------------------------------------------------------------------------------------------ save

class _Saver:
    def __init__(self):
        self.arrays = {}          # k -> array (cupy or numpy)
        self.by_id = {}           # id(array) -> k
        self.skipped = []

    def ref(self, a):
        k = self.by_id.get(id(a))
        if k is None:
            k = len(self.arrays)
            self.by_id[id(a)] = k
            self.arrays[k] = a
        return _Ref(k)

    def walk(self, obj, path):
        if _is_cp(obj) or isinstance(obj, np.ndarray):
            if obj.dtype == object:
                self.skipped.append(f"{path}: object array")
                return _Skipped(path)
            return self.ref(obj)
        if isinstance(obj, dict):
            return {k: self.walk(v, f"{path}/{k}") for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            out = [self.walk(v, f"{path}[{i}]") for i, v in enumerate(obj)]
            return out if isinstance(obj, list) else tuple(out)
        if isinstance(obj, types.SimpleNamespace):
            return types.SimpleNamespace(**{k: self.walk(v, f"{path}.{k}") for k, v in vars(obj).items()})
        try:
            pickle.dumps(obj)
            return obj
        except Exception:  # noqa: BLE001
            self.skipped.append(f"{path}: {type(obj).__name__}")
            return _Skipped(f"{path}: {type(obj).__name__}")


def _alloc_key(a):
    """(kind, base pointer, base size) of the allocation holding a."""
    if _is_cp(a):
        mem = a.data.mem
        return ("cp", int(mem.ptr), int(mem.size))
    b = a
    while isinstance(b.base, np.ndarray):
        b = b.base
    return ("np", int(b.__array_interface__["data"][0]), int(b.nbytes))


def _ptr(a):
    return int(a.data.ptr) if _is_cp(a) else int(a.__array_interface__["data"][0])


def plan(ctx):
    """What save() writes, without writing it: (jobs, header, receipt). jobs = [(file name, array)], largest
    first, each array C-contiguous (device or host); header = the skeleton.pkl dict. Arrays are not copied
    except a non-contiguous one or a reinterpreting view (copied on the current stream). The restart writer
    (namgpu/restart.py) uses this to stage the same set into pinned host memory."""
    import cupy as cp
    sv = _Saver()
    attrs = {k: v for k, v in vars(ctx).items() if k not in CTX_SKIP}
    tables = {k: v for k, v in ctx.tables.items() if k not in PER_RANK_TABLES}
    skel = {
        "st": sv.walk(dict(ctx.st.fields), "st"),
        "bc": sv.walk(ctx.bc, "bc"),
        "tables": sv.walk(tables, "tables"),
        "attrs": sv.walk(attrs, "attrs"),
    }
    g = ctx.grid
    vgrid = {k: getattr(g, k) for k in ("pt", "pdtop", "sg1", "sg2", "dsg1", "dsg2", "sgml1", "sgml2")}
    # group by allocation; the owner is the array that starts lowest and holds the most bytes
    groups = {}
    for k, a in sv.arrays.items():
        groups.setdefault(_alloc_key(a), []).append(k)
    meta = {}
    nbytes = 0
    jobs = []                                  # (file name, array)
    def _solo(k):
        # stored on its own (a copy): non-contiguous, or not inside any other member's bytes
        a = sv.arrays[k]
        ac = cp.ascontiguousarray(a) if _is_cp(a) else np.ascontiguousarray(a)
        jobs.append((f"a{k}.npy", ac))
        meta[k] = {"owner": k, "dev": bool(_is_cp(a)), "dtype": str(a.dtype), "shape": list(a.shape),
                   "off": 0, "strides": [], "c": True}
        return ac.nbytes

    nsplit = 0
    for key, ks in groups.items():
        # owners in turn: the lowest-starting, largest contiguous member; every member inside its bytes is
        # a view of it; members outside (disjoint views of a base nobody references) start a new owner
        rest = sorted(ks, key=lambda k: (_ptr(sv.arrays[k]), -sv.arrays[k].nbytes))
        while rest:
            own = rest[0]
            oa = sv.arrays[own]
            if not oa.flags.c_contiguous:
                nbytes += _solo(own)
                rest = rest[1:]
                continue
            lo, hi = _ptr(oa), _ptr(oa) + oa.nbytes
            inside = [k for k in rest if _ptr(sv.arrays[k]) >= lo and _ptr(sv.arrays[k]) + sv.arrays[k].nbytes <= hi
                      and sv.arrays[k].dtype.itemsize and (_ptr(sv.arrays[k]) - lo) % sv.arrays[k].dtype.itemsize == 0]
            jobs.append((f"a{own}.npy", oa))
            nbytes += oa.nbytes
            for k in inside:
                a = sv.arrays[k]
                meta[k] = {"owner": own, "dev": bool(_is_cp(a)), "dtype": str(a.dtype), "shape": list(a.shape),
                           "off": (_ptr(a) - lo) // a.dtype.itemsize,
                           "strides": [x // a.dtype.itemsize for x in a.strides],
                           "c": bool(a.flags.c_contiguous)}
                if a.dtype != oa.dtype:
                    # a reinterpreting view (e.g. .view(uint32)): keep it exact by storing it on its own
                    nbytes += _solo(k)
            rest = [k for k in rest if k not in inside]
            nsplit += 1 if rest else 0
    jobs.sort(key=lambda j: -j[1].nbytes)
    header = {"format": FORMAT, "skel": skel, "vgrid": vgrid, "meta": meta,
              "jm": int(g.jm), "im": int(g.im), "lm": int(g.lm)}
    rec = {"arrays": len(meta), "owners": len({m["owner"] for m in meta.values()}), "split_groups": nsplit,
           "gib": round(nbytes / 2**30, 2),
           "skipped": sv.skipped, "jm": int(g.jm), "im": int(g.im)}
    return jobs, header, rec


def write_files(out, jobs, header, rec, workers: int = 16) -> None:
    """Write a plan() to the directory `out`: one .npy per owner (D2H copy + write on many threads, both release
    the GIL: no single-core stage for a 15-30 GB state), then skeleton.pkl and save-receipt.json."""
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor
    out = Path(out)

    def _write(job):
        name, a = job
        if _is_cp(a):
            with cp.cuda.Stream(non_blocking=True) as s_:
                h = cp.asnumpy(a, stream=s_)
                s_.synchronize()
        else:
            h = a
        np.save(out / name, h, allow_pickle=False)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_write, jobs))
    with open(out / "skeleton.pkl", "wb") as f:
        pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "save-receipt.json").write_text(json.dumps(rec, indent=1))


def save(ctx, out) -> dict:
    """Write the post-init context to `out` (a directory). Returns the receipt."""
    import cupy as cp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if ctx.stream is not None:
        ctx.stream.synchronize()
    cp.cuda.Device().synchronize()
    jobs, header, rec = plan(ctx)
    cp.cuda.Device().synchronize()             # copies of non-contiguous arrays are done
    write_files(out, jobs, header, rec)
    return rec


# ------------------------------------------------------------------------------------------ load

def _row_axis(shape, jmg):
    """Axis that runs over the rows: -2 for [.., JM, X], 0 for a 1-D [JM]. None if not row-shaped."""
    if len(shape) >= 2 and shape[-2] == jmg:
        return len(shape) - 2
    if len(shape) == 1 and shape[0] == jmg:
        return 0
    return None


def load(ctx, src) -> dict:
    """Fill ctx (st, bc, tables, attrs, grid vertical coordinate) from a saved init, keeping only this
    rank's rows when ctx.slab is set. ctx.grid must already be the rank's grid (slab.local_grid)."""
    import cupy as cp
    from namgpu import slab as slabmod
    src = Path(src)
    with open(src / "skeleton.pkl", "rb") as f:
        d = pickle.load(f)
    if d["format"] != FORMAT:
        raise RuntimeError(f"slab_init.load: {src} has format {d['format']}, this code reads {FORMAT}")
    s = slabmod.of(ctx)
    jmg = d["jm"]
    if s is not None and s.jmg != jmg:
        raise RuntimeError(f"slab_init.load: saved init has JM={jmg}, slab JMG={s.jmg}")
    if s is None and int(ctx.grid.jm) != jmg:
        raise RuntimeError(f"slab_init.load: saved init has JM={jmg}, grid JM={ctx.grid.jm}")
    rows = slice(s.jms - 1, s.jme) if s is not None else None
    meta = {int(k): v for k, v in d["meta"].items()}
    full_shape = {k: m["shape"] for k, m in meta.items() if m["owner"] == k}
    hosts = {}
    bases = {}
    built = {}
    cut = []

    def owner(o):
        if o not in hosts:
            h = np.load(src / f"a{o}.npy", mmap_mode="r")
            ax = _row_axis(h.shape, jmg) if rows is not None else None
            if ax is not None:
                idx = [slice(None)] * h.ndim
                idx[ax] = rows
                h = h[tuple(idx)]
                cut.append(o)
            hosts[o] = (h, ax)                    # mmap view; read by the upload pool
        return hosts[o]

    def base_of(o, dev):
        key = (o, dev)
        if key not in bases:
            host, _ = owner(o)
            h = np.array(host, copy=True, order="C")
            bases[key] = cp.asarray(h) if dev else h
        return bases[key]

    def build(k):
        if k in built:
            return built[k]
        m = meta[k]
        o = m["owner"]
        _, ax = owner(o)
        dev = m["dev"]
        base = base_of(o, dev)
        shape = list(m["shape"])
        if k == o:
            arr = base
        elif ax is None:
            # owner not cut: the exact saved view (offset, shape, strides) on the base
            st_b = [x * base.itemsize for x in m["strides"]]
            if dev:
                arr = cp.ndarray(shape, dtype=m["dtype"], memptr=base.data + m["off"] * base.itemsize,
                                 strides=st_b)
            else:
                arr = np.lib.stride_tricks.as_strided(base.reshape(-1)[m["off"]:], shape, st_b)
        else:
            # cut owner: the view must be a run of whole leading planes of the owner, shaped [.., JMG, X]
            plane_g = int(np.prod(full_shape[o][ax:]))
            size = int(np.prod(shape))
            if not m["c"] or m["off"] % plane_g or size % plane_g:
                raise RuntimeError(f"slab_init.load: view {k} of owner {o} is not whole row planes "
                                   f"(off {m['off']}, size {size}, plane {plane_g}); cannot cut it")
            p0, n = m["off"] // plane_g, size // plane_g
            plane_l = int(np.prod(base.shape[ax:]))
            block = base.reshape(-1)[p0 * plane_l:(p0 + n) * plane_l]
            if len(shape) >= 2 and shape[-2] == jmg:
                shape[-2] = s.jm
            elif len(shape) == 1 and shape[0] == jmg:
                shape[0] = s.jm
            else:
                raise RuntimeError(f"slab_init.load: view {k} shape {m['shape']} of a cut owner has no row axis")
            arr = block.reshape(shape)
        built[k] = arr
        return arr

    def rebuild(obj):
        if isinstance(obj, _Ref):
            return build(obj.k)
        if isinstance(obj, dict):
            return {k: rebuild(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [rebuild(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(rebuild(v) for v in obj)
        if isinstance(obj, types.SimpleNamespace):
            return types.SimpleNamespace(**{k: rebuild(v) for k, v in vars(obj).items()})
        return obj

    # read + cut + upload every owner on many threads (np.load, the copy and the H2D release the GIL)
    from concurrent.futures import ThreadPoolExecutor
    want = sorted({(m["owner"], m["dev"]) for m in meta.values()})
    for o, _ in want:
        owner(o)                                   # mmap + cut header work, cheap; fills hosts serially

    def _up(od):
        o, dev = od
        host, _ = hosts[o]
        h = np.array(host, copy=True, order="C")   # the read of the mmap (and the row cut)
        if dev:
            with cp.cuda.Stream(non_blocking=True) as s_:
                b = cp.asarray(h)
                s_.synchronize()
        else:
            b = h
        return od, b

    with ThreadPoolExecutor(max_workers=16) as ex:
        for od, b in ex.map(_up, want):
            bases[od] = b

    sk = d["skel"]
    ctx.st.fields.clear()
    ctx.st.fields.update(rebuild(sk["st"]))
    ctx.bc = rebuild(sk["bc"])
    ctx.tables.clear()
    ctx.tables.update(rebuild(sk["tables"]))
    for k, v in rebuild(sk["attrs"]).items():
        if k in CTX_SKIP or isinstance(v, _Skipped):
            continue                                   # older saves carried comm/decomp = None (18:37Z)
        if v is None and getattr(ctx, k, None) is not None:
            continue
        setattr(ctx, k, v)
    vg = d["vgrid"]
    ctx.grid.set_vcoord(vg["pt"], vg["pdtop"], vg["sg1"], vg["sg2"])
    for k in ("dsg1", "dsg2", "sgml1", "sgml2"):        # exactly the saved values, not recomputed
        setattr(ctx.grid, k, vg[k])
    # the per-rank tile limits were not saved: build this rank's (drv-loop-limits reads ctx.decomp / ctx.slab)
    from namgpu.steps import drv_loop_limits as dll
    dll.run(ctx)
    cp.cuda.Device().synchronize()
    rec = {"src": str(src), "rank": s.rank if s else 0, "nranks": s.nranks if s else 1,
           "rows": [s.jms, s.jme] if s else [1, jmg], "owners_cut": len(cut),
           "cut_shapes": sorted({str(full_shape[o]) for o in cut})}
    try:
        (src / f"load-receipt-r{rec['rank']}.json").write_text(json.dumps(rec, indent=1))
    except OSError:
        pass
    return rec


# ------------------------------------------------------------------------------------------ CLI

def _save_main(a) -> int:
    import cupy as cp
    if a.managed:
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
    sys.path.insert(0, str(ROOT))
    from datetime import datetime
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import init_case_driver
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    from namgpu import paths
    ctx.case_dir = paths.case_dir(a.case)
    ctx.start = datetime.strptime(ctx.case_dir.name[:14], "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.init_source = a.source
    ctx.forecast_hours = max(6, int(a.hours + 0.999))
    ctx.out_dir = Path(a.out)
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    skip = [x for x in a.skip_init.split(",") if x]
    chain = [u for u in init_case_driver.CHAIN if u not in skip]
    t0 = time.perf_counter()
    with ctx.stream:
        init_case_driver.run(ctx, only=chain)
    ctx.stream.synchronize()
    t1 = time.perf_counter()
    import importlib
    mod = importlib.import_module("namgpu.init.slab_init")   # pickle the classes under their module name
    rec = mod.save(ctx, a.out)
    rec.update(init_s=round(t1 - t0, 2), save_s=round(time.perf_counter() - t1, 2), case=a.case,
               source=a.source, chain=chain, managed=bool(a.managed))
    (Path(a.out) / "save-receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("arrays", "owners", "gib", "init_s", "save_s")}), flush=True)
    if rec["skipped"]:
        print("skipped (not picklable):", rec["skipped"], flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("--case", required=True)
    s.add_argument("--source", default="native")
    s.add_argument("--hours", type=float, default=6.0, help="forecast length the bocos must cover")
    s.add_argument("--out", required=True)
    s.add_argument("--skip-init", default="init-fetch-case")
    s.add_argument("--managed", action="store_true", help="CUDA managed memory: a domain bigger than the card")
    a = ap.parse_args(argv)
    return _save_main(a)


if __name__ == "__main__":
    sys.exit(main())
