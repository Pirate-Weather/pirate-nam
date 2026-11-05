"""Storm-following windows: a nest window shifted by whole cells in the first window's rotated frame.

The pole stays at the first centre (generation 0). A window of generation k is the same im x jm grid with its SW
H point moved by (i0, j0) cells: wbd_k = wbd_0 + i0 dlmd, sbd_k = sbd_0 + j0 dphd. Every H and V point shared by two
windows is the same ground point, so the state in the overlap carries over by an index copy (the
exact-shift design). dlmd/dphd are written as explicit configfile port keys, because an off-centre window can no
longer derive them from wbd/sbd and im/jm (config.NamConfig.dlmd).

Files a moving run keeps under <out>/follow/ (one folder per generation, written by namgpu.nest.prepare):
  ref-grid.json             the generation-0 grid (domain.nest_grid) and its name
  g<KK>/GEN.json            generation k: offset (i0, j0), landing hour, the move list up to k; history attributes
                            (FOLLOW_*) come from it through the configfile key follow_meta
  g<KK>/nam_configfile_<w>  the window's configfile
  g<KK>/PREPARED.json       the preparer's record (walls per stage, alignment, paths)
Envelope: a window centre past rotated latitude +-ENVELOPE_TPH_DEG of the fixed pole is refused by name (NAM's own
12 km grid runs to sbd -45.036 with one dt, so dx there is the smallest NAM itself steps).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from namgpu.nest import domain

ENVELOPE_TPH_DEG = 45.0
FOLLOW_DIR = "follow"


def gen_dir(out: Path, gen: int) -> Path:
    return Path(out) / FOLLOW_DIR / f"g{int(gen):02d}"


def window_grid(ref: dict, i0: int, j0: int) -> dict:
    """The generation-0 grid `ref` (domain.nest_grid) shifted by (i0, j0) whole cells in its own rotated frame."""
    i0, j0 = int(i0), int(j0)
    g = dict(ref)
    g["wbd"] = round(ref["wbd"] + i0 * ref["dlmd"], 6)
    g["sbd"] = round(ref["sbd"] + j0 * ref["dphd"], 6)
    g["window_i0"], g["window_j0"] = i0, j0
    g["ref_wbd"], g["ref_sbd"] = ref["wbd"], ref["sbd"]
    tlm_c = g["wbd"] + (g["im"] - 1) / 2 * g["dlmd"]
    tph_c = g["sbd"] + (g["jm"] - 1) / 2 * g["dphd"]
    if abs(tph_c) > ENVELOPE_TPH_DEG:
        raise SystemExit(f"follow: window ({i0:+d}, {j0:+d}) puts its centre at rotated latitude {tph_c:.2f} deg of "
                         f"the pole {ref['tph0d']}, {ref['tlm0d']}: the window would leave NAM's rotated-latitude "
                         f"envelope (+-{ENVELOPE_TPH_DEG:g} deg, dx below NAM's own minimum for this dt)")
    g["center_rot"] = [round(tlm_c, 6), round(tph_c, 6)]
    lat, lon = rotated_to_geographic(tlm_c, tph_c, ref["tph0d"], ref["tlm0d"])
    g["center_geo"] = [round(lat, 4), round(lon, 4)]
    return g


def rotated_to_geographic(tlm: float, tph: float, tph0d: float, tlm0d: float) -> tuple[float, float]:
    """Rotated (tlm, tph) deg -> geographic (lat, lon) deg (grid.rotated_to_geographic in scalar form)."""
    d = math.radians
    st0, ct0 = math.sin(d(tph0d)), math.cos(d(tph0d))
    sphi = st0 * math.cos(d(tph)) * math.cos(d(tlm)) + ct0 * math.sin(d(tph))
    lat = math.asin(max(-1.0, min(1.0, sphi)))
    lon = d(tlm0d) + math.atan2(math.cos(d(tph)) * math.sin(d(tlm)),
                                ct0 * math.cos(d(tph)) * math.cos(d(tlm)) - st0 * math.sin(d(tph)))
    lon = (lon + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(lat), math.degrees(lon)


def offset_for(ref: dict, lat: float, lon: float) -> tuple[int, int]:
    """Whole-cell offset (i0, j0) whose window is centred within half a cell of (lat, lon)."""
    tlm, tph = domain.geographic_to_rotated(lat, lon, ref["tph0d"], ref["tlm0d"])
    return int(round(tlm / ref["dlmd"])), int(round(tph / ref["dphd"]))


def window_name(ref_name: str, i0: int, j0: int) -> str:
    return f"{ref_name}_w{int(i0):+d}{int(j0):+d}" if (i0 or j0) else ref_name


def write_configfile(path: Path, g: dict, name: str, static_npz: Path, spec_adv: bool = False,
                     follow_meta: Path | None = None, base: Path | None = None) -> dict:
    """The window's configfile. base: the reference (generation-0) configfile; the window keys are then the move step's
    (namgpu/nest/move.window_configfile: window_i0/j0, window_wbd0/sbd0 and dlmd/dphd written repr-exact from the
    reference, so every geometry kernel gives a shared ground point the same bits). Without base (tests, a grid
    given only as a dict) the reference configfile is written first from the grid. follow_meta: the window's
    GEN.json, written as the port key the history reads (FOLLOW_* attributes)."""
    from namgpu import config
    from namgpu.nest import move
    path = Path(path)
    i0, j0 = int(g.get("window_i0", 0)), int(g.get("window_j0", 0))
    if base is None:
        base = path.with_name(path.name + ".ref")
        ref = dict(g, wbd=g.get("ref_wbd", g["wbd"]), sbd=g.get("ref_sbd", g["sbd"]))
        base.write_text(domain.render((domain.PARM / "nam_configfile").read_text(),
                                      domain.config_keys(ref, name, static_npz, spec_adv), name))
    if i0 or j0:
        move.window_configfile(Path(base), path, i0, j0, name, Path(static_npz))
        text = path.read_text()
    else:
        text = domain.render(Path(base).read_text(), {"static_npz": str(static_npz), "domain_name": name}, name)
    keys = {"follow_meta": str(follow_meta)} if follow_meta is not None else {}
    path.write_text(domain.render(text, keys, name) if keys else text)
    c = config.load(path)
    if abs(c.dlmd - g["dlmd"]) > 1e-7 or abs(c.dphd - g["dphd"]) > 1e-7 or abs(c.wbd - g["wbd"]) > 1e-6:
        raise AssertionError(f"follow: window configfile reads back dlmd {c.dlmd} dphd {c.dphd} wbd {c.wbd}, "
                             f"want {g['dlmd']} {g['dphd']} {g['wbd']}")
    return {k: c.raw.get(k) for k in ("im", "jm", "tph0d", "tlm0d", "wbd", "sbd", "dlmd", "dphd", "window_i0",
                                      "window_j0", "window_wbd0", "window_sbd0", "static_npz", "domain_name",
                                      "follow_meta")}


def gen_meta(ref: dict, ref_name: str, gen: int, i0: int, j0: int, land_hour: float, moves: list[dict],
             run_start: str) -> dict:
    """The generation record: what the history attributes and the sheets read (FOLLOW_*)."""
    return {"generation": int(gen), "window_i0": int(i0), "window_j0": int(j0), "land_hour": float(land_hour),
            "run_start": run_start, "ref_name": ref_name,
            "ref_grid": {k: ref[k] for k in ("im", "jm", "lm", "tph0d", "tlm0d", "wbd", "sbd", "dlmd", "dphd",
                                             "res_km")},
            "moves": list(moves)}


def write_gen_meta(out: Path, meta: dict, d: Path | None = None) -> Path:
    """GEN.json into d (default the generation's folder gen_dir(out, generation))."""
    d = Path(d) if d is not None else gen_dir(out, meta["generation"])
    d.mkdir(parents=True, exist_ok=True)
    p = d / "GEN.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(p)
    return p


def commit_generation(prep_dir: Path, generation: int, moves: list[dict]) -> Path:
    """At the hand-off (the segment start): a prepared window becomes generation `generation` with the run's move
    list. A window is prepared before it is known which generation it will be (the controller prepares candidates
    ahead, generation -1 until committed); the history attributes are read from this file by the segment process."""
    p = Path(prep_dir) / "GEN.json"
    m = json.loads(p.read_text())
    m.update(generation=int(generation), moves=list(moves), committed=True)
    return write_gen_meta(None, m, Path(prep_dir))


def history_attrs(cfg) -> dict | None:
    """FOLLOW_* global attributes of a moving run's history (configfile key follow_meta), else None (a static
    nest's history is unchanged)."""
    p = getattr(cfg, "raw", {}).get("follow_meta")
    if not p:
        return None
    m = json.loads(Path(str(p)).read_text())
    import numpy as np
    return {"FOLLOW_GENERATION": np.int32(m["generation"]), "FOLLOW_WINDOW_I0": np.int32(m["window_i0"]),
            "FOLLOW_WINDOW_J0": np.int32(m["window_j0"]), "FOLLOW_LAND_HOUR": float(m["land_hour"]),
            "FOLLOW_REF_GRID": json.dumps(m["ref_grid"]), "FOLLOW_MOVES": json.dumps(m["moves"])}


def read_follow(nc) -> dict | None:
    """The FOLLOW_* attributes of an open history file as plain values, or None."""
    at = set(nc.ncattrs())
    if "FOLLOW_GENERATION" not in at:
        return None
    return {"generation": int(nc.FOLLOW_GENERATION), "i0": int(nc.FOLLOW_WINDOW_I0), "j0": int(nc.FOLLOW_WINDOW_J0),
            "land_hour": float(nc.FOLLOW_LAND_HOUR), "ref_grid": json.loads(str(nc.FOLLOW_REF_GRID)),
            "moves": json.loads(str(nc.FOLLOW_MOVES))}


def overlap(im: int, jm: int, a: tuple[int, int], b: tuple[int, int]):
    """Index slices of the overlap of windows at offsets a and b (each im x jm), as
    ((rows_a, cols_a), (rows_b, cols_b)), or None when they do not overlap."""
    di, dj = b[0] - a[0], b[1] - a[1]
    if abs(di) >= im or abs(dj) >= jm:
        return None
    ra = slice(max(0, dj), jm + min(0, dj))
    ca = slice(max(0, di), im + min(0, di))
    rb = slice(max(0, -dj), jm - max(0, dj))
    cb = slice(max(0, -di), im - max(0, di))
    return (ra, ca), (rb, cb)


def write_ref(out: Path, grid: dict, name: str, static=None, configfile: Path | None = None, start: str | None = None,
              hours: float | None = None, source: str | None = None, case_dir: Path | None = None,
              boco_age_h: int | None = None, parent_run: Path | None = None, boco_interval_h: int | None = None,
              spec_adv: bool = False, rrtm_rows: int | None = None) -> Path:
    """<out>/follow/ref-grid.json: the run a moving nest's preparer reads (namgpu.nest.prepare). start YYYYMMDDHH.
    The 4th argument may be the door's run dict (namgpu/nest/follow_run.py: static, cfgfile, cycle, hours, source,
    case_dir, boco_age_h, parent_run, keys, env). landable_every_h: where the source has a full state (3 h for
    gfs/ifs boundary times; every boundary interval of a parent nest, 1 h by default)."""
    if isinstance(static, dict):
        door = static
        keys, env = door.get("keys") or {}, door.get("env") or {}
        static, configfile = door["static"], door["cfgfile"]
        start, hours, source = f"{door['cycle']:%Y%m%d%H}", door["hours"], door["source"]
        case_dir, boco_age_h, parent_run = door["case_dir"], door.get("boco_age_h", 0), door.get("parent_run")
        boco_interval_h = door.get("boco_interval_h") or keys.get("boco_interval_h")
        spec_adv = str(keys.get("spec_adv", "")).lower() in ("true", ".true.")
        rrtm_rows = int(env["NAMGPU_RRTM_ROWS"]) if env.get("NAMGPU_RRTM_ROWS") else None
    if source == "parent" and not boco_interval_h:
        boco_interval_h = 1
    every = int(boco_interval_h or 3) if source == "parent" else 3
    rec = {"grid": grid, "name": name, "static": str(static), "configfile": str(configfile), "start": start,
           "hours": float(hours), "source": source, "case_dir": str(case_dir), "boco_age_h": int(boco_age_h or 0),
           "parent_run": str(parent_run) if parent_run else None, "boco_interval_h": boco_interval_h,
           "landable_every_h": every, "spec_adv": bool(spec_adv), "rrtm_rows": rrtm_rows}
    d = Path(out) / FOLLOW_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / "ref-grid.json"
    p.write_text(json.dumps(rec, indent=1))
    return p


def start_follow(out: Path, grid: dict, name: str, configfile: Path, start: str) -> Path:
    """Generation 0 of a moving run: g00/GEN.json, and the door's configfile given the follow_meta key (so its
    history has the FOLLOW_* attributes; nothing else changes). Returns GEN.json."""
    gj = write_gen_meta(out, gen_meta(grid, name, 0, 0, 0, 0.0, [], start))
    p = Path(configfile)
    p.write_text(domain.render(p.read_text(), {"follow_meta": str(gj)}, name))
    return gj


def gens_from_frames(follows) -> list:
    """The windows of a run in time order, [(landing hour, (i0, j0))], from its history files' own FOLLOW_*
    attributes (read_follow of each file, any order). Order-independent of how the controller numbered them."""
    seen = {}
    for f in follows:
        if f is not None:
            seen.setdefault((float(f["land_hour"]), int(f["i0"]), int(f["j0"])), None)
    out = [(h, (i, j)) for h, i, j in sorted(seen)]
    if not out or out[0][0] != 0.0:
        out.insert(0, (0.0, (0, 0)))
    return out


def accum_prev_on(cur: dict | None, prev: dict | None, prev_arr, tp_h: float, tc_h: float, gens=None):
    """An accumulator of the history frame at forecast hour tp_h (on that frame's window) placed on the window of the
    frame at tc_h, so cur - prev is the amount in (tp_h, tc_h]. cur/prev: read_follow() of the two files; gens: the
    run's windows (gens_from_frames; default from cur's FOLLOW_MOVES).
    A point counts only if it sat inside every window active over (tp_h, tc_h]: the strip a later move added has no
    accumulation from tp_h and is NaN (masked in the maps). A point of a strip that landed exactly at tp_h started its
    accumulators at 0 there (the move's overlay rule), so it is 0, not NaN.
    Static runs (both None) return prev_arr unchanged."""
    import numpy as np
    if cur is None and prev is None:
        return prev_arr
    # a frame without FOLLOW_* attributes in a moving run is generation 0 written by the door's own configfile
    cur = cur or {"i0": 0, "j0": 0, "land_hour": 0.0, "moves": []}
    prev = prev or {"i0": 0, "j0": 0, "land_hour": 0.0, "moves": []}
    jm, im = prev_arr.shape
    out = np.full((jm, im), np.nan)
    ov = overlap(im, jm, (prev["i0"], prev["j0"]), (cur["i0"], cur["j0"]))
    if ov is not None:
        (ra, ca), (rb, cb) = ov
        out[rb, cb] = prev_arr[ra, ca]
    if gens is None:
        gens = [(0.0, (0, 0))] + [(float(m["land_hour"]), (int(m["to"][0]), int(m["to"][1]))) for m in cur["moves"]]
    ii =cur["i0"] + np.arange(im)
    jj = cur["j0"] + np.arange(jm)
    ok = np.ones((jm, im), bool)
    for k, (land, (gi, gj)) in enumerate(gens):
        nxt = gens[k + 1][0] if k + 1 < len(gens) else float("inf")
        if land < tc_h and nxt > tp_h:
            ok &= ((jj >= gj) & (jj < gj + jm))[:, None] & ((ii >= gi) & (ii < gi + im))[None, :]
    out = np.where(ok & np.isnan(out), 0.0, out)
    out[~ok] = np.nan
    return out


def frame_union(frames) -> tuple:
    """(W, E, S, N) holding every frame (maps.frame_bounds form, E may pass 180), unwrapped around the first."""
    frames = [tuple(f) for f in frames]
    w0 = frames[0][0]
    ws, es = [], []
    for w, e, s, n in frames:
        k = round((w0 - w) / 360.0)
        ws.append(w + 360.0 * k)
        es.append(e + 360.0 * k)
    w, e = min(ws), max(es)
    shift = 360.0 * math.floor((w + 180.0) / 360.0)
    return (round(float(w - shift), 3), round(float(e - shift), 3), float(min(f[2] for f in frames)),
            float(max(f[3] for f in frames)))
