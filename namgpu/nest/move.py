"""Moving nests: the window shift and the state hand-off at a move.

A moving nest is a series of windows of ONE rotated lat-lon grid: the pole stays at the first centre and a window
is the reference (first) grid shifted by whole H cells (i0 columns, j0 rows). Every geometry kernel takes the
reference origin plus the offset (config.window, grid.build, init-grid-geo, init-grid-consts, init-static-build),
so a ground point that two windows share has the same bits in both: the overlap of a move is an index copy, with no
interpolation and no wind rotation.

    window_configfile(base, out, i0, j0, name, static)   the window's configfile (port keys window_*)
    snap(cfg, lat, lon)                                   (i0, j0) of the window centred nearest a geographic point

A move (process per segment, the hex pattern):
  1. the running segment writes a restart set at the landing step (run_forecast --restart-at H, right after that
     hour's history) and ends there;
  2. a fresh start of the NEW window at the landing time is saved (namgpu.init.slab_init save on the window's
     configfile, from the source's state at that time; boundaries named in hours from the run start);
  3. the next segment loads the fresh start and lays the set over it shifted by (dj, di) (overlay below), then
     steps on from the set's step count (no TDFI: the overlap is the model's own state).

Overlay rule, per leaf array of ctx.st.fields / ctx.tables / ctx.bc (classify):
  grid      last two axes (JM, IM): the overlap comes from the set, the strip stays the fresh start's; except the
            WINDOW_CONSTANTS rows, which encode where the window's edges are and stay the fresh start's everywhere
            (checked equal to the shifted old values away from both windows' edge bands)
  row/col   1-D JM (per-row grid constants) or IM: the fresh start's, checked equal on the shared rows/columns
  strip     a JM or IM axis elsewhere (boundary strips [.., JM, lnsh], [.., lnsh, IM]): the fresh start's (the new
            window's boundary data)
  global    no JM/IM axis: the set's (time-dependent state); the fresh value is compared and the differences are
            listed in the receipt
  other     a JM*IM-sized axis: refused by name
  counters  ntimestep, ntimestep_rad, first_step, filter_active, the clock (start, start_time) ...: the set's;
            boco_dir / case_dir: the fresh start's (the new window's boundaries)
  vertical  the grid's sg1/sg2/dsg/sgml (recovered from data at init): the set's
The strip (new ground) is the fresh start's: the source's state at the landing time, accumulators 0 (the strip
mask is in the receipt and in move-mask.npz, so sheets can mask QPF windows that straddle a move).

Shock metrics (Monitor): mean |dPD/dt| over the window and max |W| in the strip, along the seam (the overlap's edge
band next to the strip) and in the interior, every step for MONITOR_STEPS steps after a move (and before a stop),
kept as device scalars and read at the run's existing hourly sync: no new host sync on the step path.
"""
from __future__ import annotations

import json
import pickle
import time
import types
from pathlib import Path

import numpy as np

UNIT = "nest-move"
MONITOR_STEPS = 576          # 1 h at the 3 km dt (6.25 s); the receipt reads +15 min and +60 min from it
EXTRA_FRAMES_MIN = (15, 30)  # extra history frames after a move (minutes after the landing), to look at by eye
SEAM_CELLS = 4               # seam band: overlap cells next to the strip
EDGE_BAND = 8                # alignment checks skip this many cells next to either window's edge (lnsh 5 + halo 3)
# (JM, IM) arrays that encode where THIS window's edges are: never carried, always the fresh start's. Each row
# names why. Filled from the inventory of a real set .
WINDOW_CONSTANTS = {
    **{k: "init-grid-consts gc_fields_2d: hdac*j x bofac in this window's boundary blocks (st aliases of "
          "tables/grid by drv-filter-consts.bind)" for k in ("HDACX", "HDACY", "HDACVX", "HDACVY",
                                                             "hdacx", "hdacy", "hdacvx", "hdacvy")},
}
# (JM, IM) geometry: a function of the ground point only, so with the offset kernels the fresh start's values must
# equal the shifted old ones on the WHOLE overlap (the alignment check); taken from the fresh start.
GEOMETRY = ("GLAT", "GLON", "VLAT", "VLON", "glat", "glon", "vlat", "vlon", "f", "F_V")
# static surface fields: carried from the set (the model's own terrain and surface where it has run), and compared
# with the fresh start's on the interior of the overlap (the static alignment check)
STATIC_CHECK = ("FIS", "HGT_M", "SOILHGT", "SM", "ISLTYP", "IVGTYP", "VEGFRC", "ALBASE", "MXSNAL", "STDH",
                "EPSR", "SFCR", "LAND", "VGTYP", "SOTYP")      # not TG/SICE: the start writes them from the source
# Entries only the set has were made lazily during the run (caches built from the grid on first use: DXH_ROW, rdxh,
# FAD aliases, quasipost masks, physics tables, one-time fills guarded by a table flag such as MAVAIL_SET). They are
# NOT carried: the units rebuild them from the new window exactly as on a cold start's first step. CARRY_LAZY lists
# the lazily made entries that are state (they carry from step to step), each with why; they are carried with the
# overlap rule (strip zero, as st.add gives a cold start).
CARRY_LAZY: dict = {}
FRESH_ATTRS = ("boco_dir",)
# global tables that describe the window itself (its SW corner, the source sub-window it reads): the fresh start's
FRESH_GLOBALS = ("grid_sw", "scalars", "gfs_window", "grid_source", "static_extra")      # the new window's boundary files; the case folder is the run's own (run_forecast)


# ------------------------------------------------------------------------------------------------ windows

def window_meta(cfg) -> dict:
    """What a restart set records about its window (restart meta "window")."""
    return {"i0": int(cfg.win_i0), "j0": int(cfg.win_j0), "wbd0": float(cfg.ref_wbd), "sbd0": float(cfg.ref_sbd),
            "im": int(cfg.im), "jm": int(cfg.jm), "dlmd": float(cfg.dlmd), "dphd": float(cfg.dphd),
            "tph0d": float(cfg.tph0d), "tlm0d": float(cfg.tlm0d), "domain_name": cfg.raw.get("domain_name")}


def window_configfile(base: Path, out: Path, i0: int, j0: int, name: str, static_npz: Path,
                      extra: dict | None = None) -> dict:
    """Write the configfile of the window (i0, j0) of the reference grid `base` (the first window's configfile)
    to `out`. dlmd/dphd/wbd0/sbd0 are written repr-exact, so config.load gives the reference spacing bit for bit."""
    from namgpu import config
    from namgpu.nest import domain
    ref = config.load(Path(base))
    if ref.win_i0 or ref.win_j0:
        raise ValueError(f"{UNIT}: {base} is itself a shifted window; give the reference (first) configfile")
    wbd0, sbd0, dlmd, dphd = float(ref.wbd), float(ref.sbd), float(ref.dlmd), float(ref.dphd)
    keys = {"wbd": repr(wbd0 + int(i0) * dlmd), "sbd": repr(sbd0 + int(j0) * dphd),
            "window_i0": str(int(i0)), "window_j0": str(int(j0)),
            "window_wbd0": repr(wbd0), "window_sbd0": repr(sbd0), "dlmd": repr(dlmd), "dphd": repr(dphd),
            "domain_name": name, "static_npz": str(static_npz)}
    keys.update({k: str(v) for k, v in (extra or {}).items()})     # e.g. follow_meta (namgpu/nest/window.py)
    Path(out).write_text(domain.render(Path(base).read_text(), keys, name))
    c = config.load(Path(out))             # config.window checks the keys and the envelope
    if c.dlmd != ref.dlmd or c.dphd != ref.dphd:
        raise AssertionError(f"{UNIT}: window spacing {c.dlmd!r}/{c.dphd!r} != reference {ref.dlmd!r}/{ref.dphd!r}")
    return {"configfile": str(out), "i0": int(i0), "j0": int(j0), "wbd": c.wbd, "sbd": c.sbd, "name": name,
            "centre_rotated": [c.wbd + 0.5 * (c.im - 1) * c.dlmd, c.sbd + 0.5 * (c.jm - 1) * c.dphd],
            "centre_geo": list(centre_geo(c))}


def centre_geo(cfg) -> tuple:
    """Geographic (lat, lon) of the window centre."""
    from namgpu.grid import rotated_to_geographic, DTR
    tlm = (cfg.wbd + 0.5 * (cfg.im - 1) * cfg.dlmd) * DTR
    tph = (cfg.sbd + 0.5 * (cfg.jm - 1) * cfg.dphd) * DTR
    lon, lat = rotated_to_geographic(np.float64(tlm), np.float64(tph), cfg.tlm0d * DTR, cfg.tph0d * DTR)
    return round(float(lat) / DTR, 4), round(float(lon) / DTR, 4)


def snap(cfg, lat: float, lon: float) -> tuple:
    """(i0, j0) of the window of cfg's reference grid whose centre is nearest (lat, lon), whole cells."""
    from namgpu.nest import domain
    tlm, tph = domain.geographic_to_rotated(lat, lon, cfg.tph0d, cfg.tlm0d)
    c_tlm0 = cfg.ref_wbd + 0.5 * (cfg.im - 1) * cfg.dlmd        # reference window centre (0 for a door grid)
    c_tph0 = cfg.ref_sbd + 0.5 * (cfg.jm - 1) * cfg.dphd
    return int(round((tlm - c_tlm0) / cfg.dlmd)), int(round((tph - c_tph0) / cfg.dphd))


def shift_between(old: dict, cfg) -> tuple:
    """(dj, di) from the set's window (restart meta "window") to cfg's; refused unless both are windows of one
    reference grid (same pole, spacing, size and reference origin)."""
    new = window_meta(cfg)
    for k in ("im", "jm", "dlmd", "dphd", "tph0d", "tlm0d", "wbd0", "sbd0"):
        if old.get(k) != new[k]:
            raise SystemExit(f"{UNIT}: the restart set's window and this configfile are not windows of one grid: "
                             f"{k} {old.get(k)!r} != {new[k]!r} (set {old}, configfile {new})")
    return new["j0"] - old["j0"], new["i0"] - old["i0"]


def overlap(jm: int, im: int, dj: int, di: int):
    """(old rows, old cols, new rows, new cols) slices of the cells both windows hold; the new window is the old
    one shifted by dj rows north and di columns east (new index = old index - shift)."""
    if abs(dj) >= jm or abs(di) >= im:
        raise SystemExit(f"{UNIT}: a move of ({dj}, {di}) cells leaves no overlap on a {jm} x {im} window")
    o_r = slice(max(0, dj), jm + min(0, dj))
    o_c = slice(max(0, di), im + min(0, di))
    n_r = slice(max(0, -dj), jm + min(0, -dj))
    n_c = slice(max(0, -di), im + min(0, -di))
    return o_r, o_c, n_r, n_c


def ring_lines(jm: int, im: int, dj: int, di: int) -> tuple:
    """(rows, cols) of the NEW window where the OLD window's outermost ring lands inside the new interior. The
    old ring is the boundary row/column (physics never steps it; the V grid's last row/column is unused), so its
    values are not the model's own state there: these lines take the fresh start's values, like the strip.
    Found by the one-cell move (2026-11-05): the old east column carried into new column IM-2 put NaN in
    phys-turbl-driver on the first step. A null move has none (the old ring is the new ring)."""
    rows = [r for r in (-dj, jm - 1 - dj) if 1 <= r <= jm - 2 and (dj != 0)]
    cols = [c for c in (-di, im - 1 - di) if 1 <= c <= im - 2 and (di != 0)]
    return rows, cols


def masks(jm: int, im: int, dj: int, di: int, seam: int = SEAM_CELLS, band: int = EDGE_BAND) -> dict:
    """Boolean (JM, IM) masks in the NEW window: overlap, strip (new ground), seam (overlap cells within `seam` of
    the strip; for a null move, the overlap's own edge band) and interior (overlap cells `band` or more cells from
    both windows' edges)."""
    _, _, n_r, n_c = overlap(jm, im, dj, di)
    ov = np.zeros((jm, im), bool)
    ov[n_r, n_c] = True
    rr, cc = ring_lines(jm, im, dj, di)
    ov[rr, :] = False                      # the old ring lines take the fresh start's values (ring_lines)
    ov[:, cc] = False
    jj, ii = np.mgrid[0:jm, 0:im]
    d = np.full((jm, im), 10 ** 6)
    if n_r.start > 0:
        d = np.minimum(d, jj - n_r.start + 1)
    if n_r.stop < jm:
        d = np.minimum(d, n_r.stop - jj)
    if n_c.start > 0:
        d = np.minimum(d, ii - n_c.start + 1)
    if n_c.stop < im:
        d = np.minimum(d, n_c.stop - ii)
    edge_new = np.minimum.reduce([jj, ii, jm - 1 - jj, im - 1 - ii])
    edge_old = np.minimum.reduce([jj - n_r.start, ii - n_c.start, n_r.stop - 1 - jj, n_c.stop - 1 - ii])
    seam_m = ov & (d <= seam) if (dj or di) else ov & (edge_new < seam)
    interior = ov & (edge_new >= band) & (edge_old >= band)
    return {"overlap": ov, "strip": ~ov, "seam": seam_m, "interior": interior}


# ------------------------------------------------------------------------------------------------ sets on the host

def host_tree(src: Path) -> dict:
    """A saved init or a restart set as host arrays (read-only memory maps and views of them), no device:
    {"skel": {st, bc, tables, attrs}, "vgrid", "jm", "im", "lm", "meta"}."""
    from namgpu.init import slab_init
    src = Path(src)
    with open(src / "skeleton.pkl", "rb") as f:
        d = pickle.load(f)
    if d["format"] != slab_init.FORMAT:
        raise SystemExit(f"{UNIT}: {src} has format {d['format']}, this code reads {slab_init.FORMAT}")
    meta = {int(k): v for k, v in d["meta"].items()}
    owners: dict = {}
    built: dict = {}

    def owner(o):
        if o not in owners:
            owners[o] = np.load(src / f"a{o}.npy")          # read whole: thousands of memory maps run out of fds
        return owners[o]

    def build(k):
        if k in built:
            return built[k]
        m = meta[k]
        base = owner(m["owner"])
        if k == m["owner"]:
            arr = base
        else:
            flat = base.reshape(-1)
            arr = np.lib.stride_tricks.as_strided(flat[m["off"]:], m["shape"],
                                                  [x * flat.itemsize for x in m["strides"]], writeable=False)
        built[k] = arr
        return arr

    def rebuild(obj):
        if isinstance(obj, slab_init._Ref):
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

    mj = src / "meta.json"
    return {"skel": rebuild(d["skel"]), "vgrid": d["vgrid"], "jm": int(d["jm"]), "im": int(d["im"]),
            "lm": int(d["lm"]), "meta": json.loads(mj.read_text()) if mj.is_file() else {}}


def classify(shape, jm: int, im: int) -> str:
    """grid | row | col | strip | global | other (module docstring)."""
    shape = tuple(int(x) for x in shape)
    if len(shape) >= 2 and shape[-2:] == (jm, im):
        return "grid"
    if shape == (jm,):
        return "row"
    if shape == (im,):
        return "col"
    if jm * im in shape:
        return "other"
    if jm in shape or im in shape:
        return "strip"
    return "global"


def _holds_array(v) -> bool:
    if isinstance(v, np.ndarray):
        return True
    if isinstance(v, dict):
        return any(_holds_array(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_holds_array(x) for x in v)
    if isinstance(v, types.SimpleNamespace):
        return any(_holds_array(x) for x in vars(v).values())
    return False


def _leaf_name(path: str) -> str:
    return path.rsplit("/", 1)[-1].split("[", 1)[0]


def _bits_differ(a: np.ndarray, b: np.ndarray) -> int:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return -1
    if a.dtype.itemsize in (1, 2, 4, 8):
        u = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
        return int(np.count_nonzero(a.view(u) != b.view(u)))
    return int(np.count_nonzero(a.view(np.uint8) != b.view(np.uint8)))


# ------------------------------------------------------------------------------------------------ overlay

# State arrays a scheme re-allocates in another shape on its first call: a fresh start (no step yet) still has the
# state table's shape, the restart set the run-time one. The set's shape is taken, laid over by the overlap rule.
# Breakage without it: the first hand-off of the 12 h Ampil follow run (2026-11-05 05:04Z) was refused at MPRATES,
# fresh (7, 60, 201, 172) vs set (1, 60, 201, 172).
RESHAPED_AT_RUN = {
    "MPRATES": "phys_fer_driver allocates (D_SS = 1, LM, JM, IM) on its first call; the state table's row has the "
               "tracer count in place of D_SS",
}


def overlay(ctx, rset: Path, dj: int, di: int, out_dir: Path | None = None, log=print) -> dict:
    """Lay the restart set `rset` (the old window) over the fresh start already loaded into ctx (the new window),
    shifted by (dj, di). Returns the receipt (also written to <out_dir>/move-receipt.json with move-mask.npz)."""
    import cupy as cp
    t0 = time.perf_counter()
    old = host_tree(rset)
    jm, im = int(ctx.grid.jm), int(ctx.grid.im)
    if (old["jm"], old["im"]) != (jm, im):
        raise SystemExit(f"{UNIT}: set {rset} is {old['jm']} x {old['im']}, the window is {jm} x {im}")
    if jm == im:
        raise SystemExit(f"{UNIT}: a square window ({jm} x {im}) cannot tell per-row from per-column arrays; give "
                         "the nest a different north-south size (--size-y-km)")
    o_r, o_c, n_r, n_c = overlap(jm, im, dj, di)
    ring_r, ring_c = ring_lines(jm, im, dj, di)
    mk = masks(jm, im, dj, di)
    interior = mk["interior"]
    rows_in = np.zeros(jm, bool)
    rows_in[n_r] = True
    rows_in[:EDGE_BAND] = rows_in[jm - EDGE_BAND:] = False
    rows_in[n_r.start:n_r.start + EDGE_BAND] = rows_in[max(0, n_r.stop - EDGE_BAND):n_r.stop] = False
    cols_in = np.zeros(im, bool)
    cols_in[n_c] = True
    cols_in[:EDGE_BAND] = cols_in[im - EDGE_BAND:] = False
    cols_in[n_c.start:n_c.start + EDGE_BAND] = cols_in[max(0, n_c.stop - EDGE_BAND):n_c.stop] = False
    rec = {"unit": UNIT, "set": str(rset), "shift_dj_di": [int(dj), int(di)], "grid": [jm, im],
           "overlap_new": [[n_r.start, n_r.stop], [n_c.start, n_c.stop]],
           "overlap_old": [[o_r.start, o_r.stop], [o_c.start, o_c.stop]],
           "strip_cells": int(mk["strip"].sum()), "interior_cells": int(interior.sum()),
           "old_ring_lines_fresh": {"rows": ring_r, "cols": ring_c},
           "carried": [], "window_constants": [], "geometry": [], "static": [], "rows": [], "cols": [],
           "strips": [], "globals_from_set": [], "globals_equal": 0, "fresh_only": [], "set_only_dropped": [],
           "set_only_carried": [], "relaid": [], "globals_fresh": [], "skipped_dropped": [], "refused": []}
    seen: dict = {}
    stream = ctx.stream if ctx.stream is not None else cp.cuda.get_current_stream()

    def ident(a):
        return (int(a.data.ptr), tuple(a.shape), tuple(a.strides), str(a.dtype)) if isinstance(a, cp.ndarray) \
            else (int(a.__array_interface__["data"][0]), tuple(a.shape), tuple(a.strides), str(a.dtype))

    def leaf(path, f, o):
        name = _leaf_name(path)
        if o is None:
            rec["fresh_only"].append(path)
            return f
        if f is None:                      # made lazily in the run: dropped unless CARRY_LAZY (see above)
            rec["set_only_dropped"].append(path)
            return None
        fs, os_ = tuple(f.shape), tuple(o.shape)
        if fs != os_ and name in RESHAPED_AT_RUN and str(f.dtype) == str(o.dtype):
            rec.setdefault("reshaped_at_run", []).append({"path": path, "fresh": list(fs), "set": list(os_),
                                                          "why": RESHAPED_AT_RUN[name]})
            return lazy(path, o)
        if fs != os_ or str(f.dtype) != str(o.dtype):
            if (len(fs) >= 2 and len(os_) >= 2 and fs[-2:] == os_[-2:] == (jm, im) and str(f.dtype) == str(o.dtype)
                    and isinstance(f, cp.ndarray)):
                # the run re-laid a (.., JM, IM) array after the first step (MPRATES: 7 planes at init, 1 after):
                # the set's layout, overlap from the set, strip zero (a cold start's first step zero-fills it)
                a_ = cp.zeros(os_, o.dtype)
                with stream:
                    a_[..., n_r, n_c] = cp.asarray(np.ascontiguousarray(np.asarray(o)[..., o_r, o_c]))
                if ring_r:
                    a_[..., ring_r, :] = 0
                if ring_c:
                    a_[..., :, ring_c] = 0
                rec["relaid"].append(f"{path}: fresh {fs} -> set {os_}")
                return a_
            rec["refused"].append(f"{path}: fresh {fs} {f.dtype} vs set {os_} {o.dtype}")
            return f
        kind = classify(fs, jm, im)
        key = ident(f)
        if key in seen:                     # an alias of an array already laid over (Q = TRACERS[k], ...)
            return f
        seen[key] = path
        dev = isinstance(f, cp.ndarray)
        if kind == "grid":
            if name in GEOMETRY:
                fh = cp.asnumpy(f, stream=stream) if dev else np.asarray(f)
                rec["geometry"].append({"path": path, "overlap_mismatch": _bits_differ(fh[..., n_r, n_c],
                                                                                       np.asarray(o)[..., o_r, o_c])})
                return f
            if name in STATIC_CHECK:
                fh = cp.asnumpy(f, stream=stream) if dev else np.asarray(f)
                d_ = np.asarray(fh[..., n_r, n_c] != np.asarray(o)[..., o_r, o_c])
                d_ = d_.reshape(-1, *d_.shape[-2:]).any(axis=0)
                m_int = interior[n_r, n_c]
                rec["static"].append({"path": path, "interior_mismatch": int((d_ & m_int).sum()),
                                      "band_mismatch": int((d_ & ~m_int).sum())})
            if name in WINDOW_CONSTANTS:
                fh = cp.asnumpy(f, stream=stream) if dev else np.asarray(f)
                a_new = fh[..., n_r, n_c]
                b_old = np.asarray(o)[..., o_r, o_c]
                m_int = interior[n_r, n_c]
                diff = np.asarray(a_new != b_old).reshape(-1, *m_int.shape) if a_new.ndim >= 2 else None
                n_int = int((diff & m_int).sum()) if diff is not None else -1
                rec["window_constants"].append({"path": path, "why": WINDOW_CONSTANTS[name],
                                                "interior_mismatch": n_int,
                                                "overlap_mismatch": int(diff.sum()) if diff is not None else -1})
                return f
            src = np.ascontiguousarray(np.asarray(o)[..., o_r, o_c])
            if dev:
                with stream:
                    keep_r = f[..., ring_r, :].copy() if ring_r else None
                    keep_c = f[..., :, ring_c].copy() if ring_c else None
                    f[..., n_r, n_c] = cp.asarray(src)
                    if ring_r:
                        f[..., ring_r, :] = keep_r
                    if ring_c:
                        f[..., :, ring_c] = keep_c
            else:
                keep_r = f[..., ring_r, :].copy() if ring_r else None
                keep_c = f[..., :, ring_c].copy() if ring_c else None
                f[..., n_r, n_c] = src
                if ring_r:
                    f[..., ring_r, :] = keep_r
                if ring_c:
                    f[..., :, ring_c] = keep_c
            rec["carried"].append(path)
            return f
        if kind in ("row", "col"):
            fh = cp.asnumpy(f, stream=stream) if dev else np.asarray(f)
            if kind == "row":
                nmis = _bits_differ(fh[n_r][rows_in[n_r]], np.asarray(o)[o_r][rows_in[n_r]])
            else:
                nmis = _bits_differ(fh[n_c][cols_in[n_c]], np.asarray(o)[o_c][cols_in[n_c]])
            rec["rows" if kind == "row" else "cols"].append({"path": path, "interior_mismatch": nmis})
            return f
        if kind == "strip":
            rec["strips"].append(path)
            return f
        if kind == "other":
            rec["refused"].append(f"{path}: shape {fs} has a JM x IM axis; no overlay rule (namgpu/nest/move.py)")
            return f
        # global: the set's value (state), the fresh value compared; window descriptions stay the fresh start's
        if any(f"/{k}" in path for k in FRESH_GLOBALS):
            rec["globals_fresh"].append(path)
            return f
        fh = cp.asnumpy(f, stream=stream) if dev else np.asarray(f)
        nd = _bits_differ(fh, np.asarray(o))
        if nd == 0:
            rec["globals_equal"] += 1
            return f
        rec["globals_from_set"].append({"path": path, "shape": list(fs), "differing": nd})
        if dev:
            with stream:
                f[...] = cp.asarray(np.ascontiguousarray(o))
        else:
            f[...] = o
        return f

    def name_ok(k):
        return str(k) in CARRY_LAZY

    def lazy(path, o):
        """A CARRY_LAZY entry: a new array like the set's, the overlap rule for (JM, IM), the set's otherwise."""
        kind = classify(o.shape, jm, im)
        a = cp.zeros(o.shape, o.dtype)
        with stream:
            if kind == "grid":
                a[..., n_r, n_c] = cp.asarray(np.ascontiguousarray(np.asarray(o)[..., o_r, o_c]))
            elif kind == "global":
                a[...] = cp.asarray(np.ascontiguousarray(o))
            else:
                rec["refused"].append(f"{path}: CARRY_LAZY entry of kind {kind}")
        rec["set_only_carried"].append(path)
        return a

    def walk(f, o, path):
        if isinstance(f, (np.ndarray, cp.ndarray)) or isinstance(o, np.ndarray):
            return leaf(path, f if isinstance(f, (np.ndarray, cp.ndarray)) else None,
                        o if isinstance(o, np.ndarray) else None)
        if isinstance(f, dict) and (isinstance(o, dict) or o is None):
            o = o or {}
            for k in list(f.keys()):
                f[k] = walk(f[k], o.get(k), f"{path}/{k}")
            for k in o:
                if k not in f:
                    if name_ok(k) and isinstance(o[k], np.ndarray):
                        f[k] = lazy(f"{path}/{k}", o[k])
                    else:
                        rec["set_only_dropped"].append(f"{path}/{k}")
            return f
        if isinstance(f, types.SimpleNamespace):
            ov = vars(o) if isinstance(o, types.SimpleNamespace) else {}
            for k, v in vars(f).items():
                setattr(f, k, walk(v, ov.get(k), f"{path}.{k}"))
            return f
        if isinstance(f, list) and isinstance(o, list) and len(f) == len(o):
            for i in range(len(f)):
                f[i] = walk(f[i], o[i], f"{path}[{i}]")
            return f
        if isinstance(f, tuple) and isinstance(o, tuple) and len(f) == len(o):
            return tuple(walk(u, v, f"{path}[{i}]") for i, (u, v) in enumerate(zip(f, o)))
        if f is None and o is not None and not isinstance(o, (dict, list, tuple, types.SimpleNamespace)):
            return f
        if isinstance(f, (int, float, str, bool, type(None))) and f != o and path.startswith("/tables")                 and not any(f"/{k}" in path for k in FRESH_GLOBALS):
            rec["globals_from_set"].append({"path": path, "fresh": repr(f)[:60], "set": repr(o)[:60]})
            return o
        return f

    sk = old["skel"]

    def drop_skipped(d, path):
        # objects the saver could not pickle (slab_init._Skipped, e.g. the RRTM chunk context a TDFI'd fresh start
        # made): removed, so their units rebuild them on first use (rrtm_call read the placeholder as its context)
        for k in list(d.keys()):
            v = d[k]
            if type(v).__name__ == "_Skipped":
                del d[k]
                rec["skipped_dropped"].append(f"{path}/{k}")
            elif isinstance(v, dict):
                drop_skipped(v, f"{path}/{k}")
    drop_skipped(ctx.tables, "/tables")
    drop_skipped(ctx.st.fields, "/st")
    walk(ctx.st.fields, sk["st"], "/st")
    walk(ctx.tables, sk["tables"], "/tables")
    ctx.bc = walk(ctx.bc, sk["bc"], "/bc")
    # counters, flags and the clock: the set's; the new window's files: the fresh start's
    attrs = {}
    from namgpu.init import slab_init as _si
    for k, v in sk["attrs"].items():
        if k in FRESH_ATTRS or k in _si.CTX_SKIP:
            continue
        if _holds_array(v) or type(v).__name__ == "_Skipped":
            continue                       # the filter's state (ctx.filt) and other array holders: the fresh start's
        cur = getattr(ctx, k, None)
        if repr(cur) != repr(v):
            attrs[k] = {"fresh": repr(cur)[:60], "set": repr(v)[:60]}
        setattr(ctx, k, v)
    rec["attrs_from_set"] = attrs
    vg = old["vgrid"]
    vd = {k: _bits_differ(np.asarray(getattr(ctx.grid, k)), np.asarray(vg[k])) if isinstance(vg[k], np.ndarray)
          else int(getattr(ctx.grid, k) != vg[k]) for k in vg}
    ctx.grid.set_vcoord(vg["pt"], vg["pdtop"], vg["sg1"], vg["sg2"])
    for k in ("dsg1", "dsg2", "sgml1", "sgml2"):
        setattr(ctx.grid, k, vg[k])
    rec["vgrid_fresh_differs"] = {k: v for k, v in vd.items() if v}
    stream.synchronize()
    rec["overlay_s"] = round(time.perf_counter() - t0, 2)
    rec["carried_n"] = len(rec["carried"])
    rec["ntimestep"] = int(ctx.ntimestep)
    if rec["refused"]:
        raise SystemExit(f"{UNIT}: refused (no overlay rule): {rec['refused'][:6]}")
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(out_dir) / "move-mask.npz", **{k: v for k, v in mk.items()},
                            shift=np.array([dj, di]), ntimestep=np.array(int(ctx.ntimestep)))
        (Path(out_dir) / "move-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    log(f"{UNIT}: shift ({dj}, {di}) cells, carried {rec['carried_n']} grid arrays, "
        f"{len(rec['window_constants'])} window constants, {len(rec['rows'])} row / {len(rec['cols'])} col tables, "
        f"{len(rec['strips'])} strips, {len(rec['globals_from_set'])} globals from the set, "
        f"{len(rec['set_only_dropped'])} lazy entries dropped, strip {rec['strip_cells']} cells, "
        f"geometry mismatches {sum(x['overlap_mismatch'] for x in rec['geometry'])}, static interior mismatches "
        f"{sum(x['interior_mismatch'] for x in rec['static'])}, {rec['overlay_s']} s")
    return rec


def carried_check(ctx, rset: Path, dj: int, di: int, paths: list) -> dict:
    """Read back every carried array's overlap from the card and compare it bit for bit with the set's shifted
    values (the one-cell move check: 0 mismatches)."""
    import cupy as cp
    old = host_tree(rset)
    o_r, o_c, n_r, n_c = overlap(int(ctx.grid.jm), int(ctx.grid.im), dj, di)
    rr, cc = ring_lines(int(ctx.grid.jm), int(ctx.grid.im), dj, di)
    trees = {"/st": (ctx.st.fields, old["skel"]["st"]), "/tables": (ctx.tables, old["skel"]["tables"]),
             "/bc": (ctx.bc, old["skel"]["bc"])}
    bad, n = {}, 0
    for p in paths:
        root = "/" + p.split("/")[1].split(".")[0]
        f, o = trees[root]
        for part in _split(p[len(root):]):
            f, o = _get(f, part), _get(o, part)
        fh = cp.asnumpy(f) if isinstance(f, cp.ndarray) else np.asarray(f)
        a_, b_ = np.array(fh[..., n_r, n_c]), np.array(np.asarray(o)[..., o_r, o_c])
        for r in rr:                        # the old ring lines are the fresh start's (ring_lines): not compared
            a_[..., r - n_r.start, :] = b_[..., r - n_r.start, :]
        for c in cc:
            a_[..., :, c - n_c.start] = b_[..., :, c - n_c.start]
        m = _bits_differ(a_, b_)
        n += 1
        if m:
            bad[p] = m
    return {"arrays": n, "mismatched_arrays": len(bad), "mismatches": bad}


def _split(rest: str):
    import re
    return [t for t in re.findall(r"/([^/\[\.]+)|\[(\d+)\]|\.([^/\[\.]+)", rest) for t in [next(x for x in t if x)]]


def _get(obj, part):
    if isinstance(obj, dict):
        return obj[part]
    if isinstance(obj, (list, tuple)):
        return obj[int(part)]
    return getattr(obj, part)


def static_alignment(old_npz: Path, new_npz: Path, dj: int, di: int) -> dict:
    """The two windows' static files on their shared cells: mismatches per field, in the interior (EDGE_BAND or
    more cells from both windows' edges) and in the edge band (where the static builder's terrain smoothing reads
    the window edge)."""
    a, b = np.load(old_npz, allow_pickle=False), np.load(new_npz, allow_pickle=False)
    out = {}
    for k in a.files:
        if k == "META" or a[k].dtype.kind in "OUS" or a[k].ndim < 2:
            continue
        jm, im = a[k].shape[-2:]
        o_r, o_c, n_r, n_c = overlap(jm, im, dj, di)
        m = masks(jm, im, dj, di)["interior"][n_r, n_c]
        diff = a[k][..., o_r, o_c] != b[k][..., n_r, n_c]
        diff = diff.reshape(-1, *m.shape).any(axis=0)
        out[k] = {"interior": int((diff & m).sum()), "band": int((diff & ~m).sum())}
    return out


# ------------------------------------------------------------------------------------------------ shock monitor

class Monitor:
    """Per-step shock numbers on the card: mean |dPD/dt| (Pa/s) over the window, max |W| (m/s) in the strip, the
    seam and the interior, for the steps [n0, n0 + nsteps). Device scalars only; drain() copies them at the run's
    existing hourly sync. frames: extra history writes at these step counts (the first steps after a move)."""

    def __init__(self, ctx, n0: int, nsteps: int, mk: dict | None, label: str, frames=()):
        import cupy as cp
        self.n0, self.n1, self.label = int(n0), int(n0) + int(nsteps), label
        self.dt = float(ctx.cfg.dt)
        self.buf = cp.zeros((int(nsteps), 4), cp.float32)
        self.host = np.full((int(nsteps), 4), np.nan, np.float32)
        self.prev = None
        jm, im = int(ctx.grid.jm), int(ctx.grid.im)
        mk = mk or masks(jm, im, 0, 0)
        self.m = {k: cp.asarray(mk[k].astype(np.float32)) for k in ("strip", "seam", "interior")}
        self.frames = set(int(x) for x in frames)
        self.written = []
        if int(ctx.ntimestep) == self.n0 - 1:          # created at the landing step: PD before the first step
            self.prev = ctx.st.fields["PD"].copy()

    def after_step(self, ctx):
        import cupy as cp
        n = int(ctx.ntimestep)
        if n in self.frames and not ctx.filter_active:
            ctx.driver.write_history()
            self.written.append(n)
        if not (self.n0 <= n < self.n1) and not (self.n0 - 1 == n):
            return
        pd = ctx.st.fields["PD"]
        if self.prev is not None and self.n0 <= n < self.n1:
            k = n - self.n0
            self.buf[k, 0] = cp.mean(cp.abs(pd - self.prev)) / np.float32(self.dt)
            w = cp.max(cp.abs(ctx.st.fields["W"]), axis=0)
            self.buf[k, 1] = cp.max(w * self.m["strip"])
            self.buf[k, 2] = cp.max(w * self.m["seam"])
            self.buf[k, 3] = cp.max(w * self.m["interior"])
        if self.prev is None:
            self.prev = pd.copy()
        else:
            self.prev[...] = pd

    def drain(self):
        """Called right after the run's hourly stream sync: a plain copy, no new sync."""
        self.host[...] = self.buf.get()

    def receipt(self) -> dict:
        h = self.host
        per_min = 60.0 / self.dt

        def at(minutes):
            k = int(round(minutes * per_min)) - 1
            return [round(float(x), 5) for x in h[k]] if 0 <= k < len(h) and np.isfinite(h[k, 0]) else None

        fin = np.isfinite(h[:, 0])
        return {"label": self.label, "from_step": self.n0, "steps": int(fin.sum()), "dt": self.dt,
                "columns": ["mean |dPD/dt| Pa/s", "max |W| strip m/s", "max |W| seam m/s", "max |W| interior m/s"],
                "mean_dpddt_pa_s": round(float(np.nanmean(h[:, 0])), 5) if fin.any() else None,
                "mean_dpddt_hpa_3h": round(float(np.nanmean(h[:, 0])) * 108.0, 4) if fin.any() else None,
                "max_w_strip": round(float(np.nanmax(h[:, 1])), 4) if fin.any() else None,
                "max_w_seam": round(float(np.nanmax(h[:, 2])), 4) if fin.any() else None,
                "max_w_interior": round(float(np.nanmax(h[:, 3])), 4) if fin.any() else None,
                "at_1min": at(1), "at_5min": at(5), "at_15min": at(15), "at_30min": at(30), "at_60min": at(60),
                "extra_frames_at_steps": self.written,
                "first_steps": [[round(float(x), 5) for x in r] for r in h[:6] if np.isfinite(r[0])],
                "series_every_6_steps": [[round(float(x), 5) for x in r] for r in h[::6] if np.isfinite(r[0])]}
