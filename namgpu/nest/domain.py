"""Nest domains from a centre, a size and a resolution: the configfile, the static cache key, the default source by
region and the VRAM estimate. Tables, not code paths: a new resolution or region is a row.

Grid (NMMB rotated lat-lon B grid, the domain's own pole at its centre):
  dlmd = 0.0105 x res, dphd = 0.009 x res (deg; NAM 12 km = 0.126 / 0.108, the 4 km CONUS domain 0.042 / 0.036);
  im - 1 = round(size / dx), jm - 1 = round(size / dy), dx = A_EARTH dlmd, dy = A_EARTH dphd (grid.A_EARTH, NMMB a);
  tph0d / tlm0d = the centre, wbd = -(im - 1)/2 dlmd, sbd = -(jm - 1)/2 dphd (config.dlmd/dphd derive them back).
Step and physics per resolution (RES_TABLE): 12 km keeps the NAM parent keys (parm/nam_configfile); 3/4/5 km take
NAM's CONUS-nest physics (nam_configfile_conusnest: convection none, gwdflg false, smag2 0.3, codamp 12, ...),
dt = 25 s x res / 12 (3 km = 6+1/4 s exactly as conusnest), filt_dt NAM's own where NAM runs that resolution
(12 km 8 s, parm/nam_configfile; 3 km 2 s, nam_configfile_conusnest filt_dt_int 2, num 0, den 4), and a divisor of
the TDFI window (nsecs_bcktdfi 600 s) on every row (write_configfile checks it), nrads/nradl = 20 min as NAM. spec_adv is
NAM's nest value true, but the port runs false until true has run (knob --spec-adv).
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path

from namgpu import paths as _paths

ROOT = _paths.work_root()          # cases/, static/, runs/ (namgpu/paths.py)
PARM = _paths.PARM                 # the NAM parm files shipped in the package
A_EARTH = 6376000.0          # grid.A_EARTH (NMMB module_CONSTANTS a)

# NAM's CONUS-nest physics keys (parm/nam_configfile_conusnest), as tools/make_domain_configs.py has used them for
# the 4/5 km CONUS domains since 2026-11-04 (spec_adv left out: see module docstring).
NEST_PHYSICS = {
    "smag2": "0.3", "codamp": "12.0", "gwdflg": "false", "cdmb": "0.5", "sigfac": "0.00",
    "convection": "none", "rhgrd": "1.0", "fres": "0.25", "fsl": "0.75", "fss": "0.75", "nodeep": ".true.",
}

# res_km -> step keys (exact fractions, NAM's own where NAM runs that resolution) and the physics set.
#   dt = dt_int + dt_num/dt_den, filt_dt likewise; nrads = nradl = radiation every 20 min (NAM 12 km 48 x 25 s,
#   conusnest 192 x 6.25 s).
RES_TABLE = {
    # handoff: the row's history carries the parent hand-off set (configfile history_handoff), so a 12 km door run is
    # a parent a --source parent nest can start from (fix round 2)
    12: {"dt": (25, 0, 3), "filt": (8, 0, 3), "nrads": 48, "physics": "parent", "handoff": True,
         "src": "parm/nam_configfile (NAM 12 km parent)"},
    5:  {"dt": (10, 0, 3), "filt": (3, 1, 3), "nrads": 120, "physics": "conusnest",
         "src": "tools/make_domain_configs.py 5km row (dt 10 s, 20 min radiation)"},
    4:  {"dt": (8, 1, 3), "filt": (2, 7, 9), "nrads": 144, "physics": "conusnest",
         "src": "tools/make_domain_configs.py 4km row (parent dt / 3)"},
    # filt_dt 2 + 0/4 s as conusnest lines 51-53 (was (2, 1, 4) = 2.25 s, which does not divide the 600 s TDFI
    # window: drv-dfi-tdfi-driver refused every 3 km nest with TDFI on, maxq/door-tdfi 2026-11-04 23:25Z)
    3:  {"dt": (6, 1, 4), "filt": (2, 0, 4), "nrads": 192, "physics": "conusnest",
         "src": "NAM parm/nam_configfile_conusnest (dt 6+1/4, filt_dt 2+0/4, nrads 192)"},
}

# Default init/boundary source by region and date: the first row whose box holds the centre, then the first of its
# "sources" whose sources.SOURCES row serves the start cycle (valid_from <= start < valid_before). GFS where NAM
# served (North America, Hawaii, Puerto Rico and the US Pacific territories); elsewhere IFS open data 0.1 deg from
# 2026-10-20 12Z, IFS 0.25 deg from 2026-02-11 12Z (the first 0.25 cycle with every record the row reads), and GFS before that (was "ifs" for any date: a 2025 Alps
# hindcast picked a row that starts on 2026-10-20 and could not be served). Not the NAM 12 km parent
# rectangle itself: its rotated corners reach Western Europe (46.5 N 8 E is at rotated 56.1, 40.8, inside
# wbd -60.039 / sbd -45.036), which is not where NAM's GFS-driven forecasts were meant to be the default.
SOURCE_BY_REGION = [
    {"region": "north-america", "kind": "box", "lat": (5.0, 84.0), "lon": (-170.0, -50.0), "sources": ["gfs"]},
    {"region": "hawaii", "kind": "box", "lat": (15.0, 26.0), "lon": (-165.0, -150.0), "sources": ["gfs"]},
    {"region": "us-pacific", "kind": "box", "lat": (10.0, 22.0), "lon": (140.0, 150.0), "sources": ["gfs"]},
    {"region": "world", "kind": "all", "sources": ["ifs", "ifs-0p25", "gfs"]},
]

# VRAM model (GiB): state + scratch per column and a fixed part, RRTM chunk scratch ROWS x IM x 0.59 MB
# (tools/run_nam_domain.sh, MULTICARD-PLAN.md section 7), and the local-memory reservation NAMGPU_LMEM_RESERVE
# (11 KiB) per resident thread after kernels.trim_local_memory. per_column_kb / fixed_gib from the first whole 48 h
# nest (41S 174E, 286 x 334, NAMGPU_RRTM_ROWS 16, 2026-11-04): CuPy pool 14.27 GiB at the end, process peak
# 17.36 GiB (nvidia-smi); 2.74 lmem + 2.51 RRTM + 11.4 per column + 1.0 = 17.6. The same nest on the merged tree on
# an RTX PRO 6000 Max-Q at RRTM rows 32 peaked 21.81 GiB (3.03 lmem + 5.03 RRTM + 1.0 leaves 12.75 GiB = 140 KB per
# column), so per_column_kb is 140, the larger of the two. A 10-minute smoke peaks far lower
# (10.56 GiB at rows 8): the pool keeps growing through the first hours, so the estimate is the whole run's.
# 2026-11-05 (fix round 1): with no pool limit the CuPy pool keeps freed blocks and grows past the working set on
# a shared card: the conus-parent proof ended at 26.77 GiB pool / 29.93 GiB process against a 21.52 GiB estimate,
# europe-ifs 21.83 / 24.93, while the same 286 x 334 nest on the 2x PRO 6000 box ended at 18.05-18.41 GiB pool
# (rows 32: 18.41 - 5.03 RRTM = 13.38 GiB = 147 KB per column). So per_column_kb is 150 and the door CAPS the pool
# (CUPY_GPU_MEMORY_LIMIT = state + RRTM + pool_headroom_gib): CuPy frees its cached blocks and retries when an
# allocation would pass the cap, so the run stays inside what the pre-flight checked instead of growing into a
# co-tenant's memory. total = cap + lmem + fixed (context).
VRAM = {"per_column_kb": 150.0, "fixed_gib": 1.0, "rrtm_mb_per_row_col": 0.59, "lmem_bytes_per_thread": 11264,
        "pool_headroom_gib": 1.0, "margin_gib": 0.75}
# Per memory path (state.resolve_memory_path: the 32 GB fit below FIT_BELOW_GIB = 64 GiB of card, the full path at and
# above it; NAMGPU_MEMORY_PATH / NAMGPU_SCRATCH_ARENA override), the pool part that does not scale with the slab.
# The door resolves the path exactly as the forecast will (memory_path below) and caps the pool with that path's row.
# Before this the door had one model, made before the full path switched on by itself at 64 GiB: on a 96 GB card the
# Milton Gulf 12 km parent (215 x 217 x 60, rows 32) was capped at 12.3 GB and died at the start of its forecast,
# 0.9 GB short on top of 12.1 GB in use (2026-11-05). Measured on that parent shape (same card, start, rows, 3 h, no
# pool cap): CuPy pool at the end 12.67 GiB on the full path, 7.84 GiB on the fit path (MEASURED_POOL_GIB).
#   fit:  the per-column part alone covers every fit-path run (parent 7.84 vs cap 11.45; 286 x 334 nest 14.27 vs 17.2).
#   full: the full path also keeps the init-only and step scratch arrays for the whole run (pool after init 2.56 vs
#         0.81 GiB on the parent); 2.5 GiB on top of the per-column part covers the parent (cap 13.95 GiB) and the
#         286 x 334 nest on a 96 GB card (18.41 GiB pool at the end of 48 h, cap 22.2).
VRAM_PATH = {"fit": {"fixed_pool_gib": 0.0}, "full": {"fixed_pool_gib": 2.5}}
MEASURED_POOL_GIB = {"milton-parent-full-r32": 12.67, "milton-parent-fit-r32": 7.84}
RRTM_ROW_CHOICES = (32, 16, 8, 4, 2)
# SMs by card name (nvidia-smi has no SM count and the pre-flight must not open a CUDA context)
CARD_SMS = (("5090", 170), ("PRO 6000", 188), ("4090", 128), ("5070 Ti", 70), ("3080", 68), ("H100", 132),
            ("B200", 148))


def geographic_to_rotated(lat, lon, tph0d, tlm0d):
    """Geographic (deg) -> rotated (tlm, tph) deg on the grid centred at (tph0d, tlm0d); the inverse of
    grid.rotated_to_geographic."""
    d = math.radians
    glat, dlam = d(lat), d(((lon - tlm0d) + 180.0) % 360.0 - 180.0)
    x, y, z = math.cos(glat) * math.cos(dlam), math.cos(glat) * math.sin(dlam), math.sin(glat)
    s0, c0 = math.sin(d(tph0d)), math.cos(d(tph0d))
    tph = math.asin(max(-1.0, min(1.0, -s0 * x + c0 * z)))
    tlm = math.atan2(y, c0 * x + s0 * z)
    return math.degrees(tlm), math.degrees(tph)


def serves(name: str, cycle) -> bool:
    """True when sources.SOURCES[name] has files for this start cycle (its valid_from / valid_before)."""
    from namgpu.init import sources
    row = sources.SOURCES.get(name)
    if row is None:
        return False
    if cycle is None:
        return True
    t = cycle.strftime("%Y-%m-%dT%H")
    return (not row.get("valid_from") or t >= row["valid_from"]) and (not row.get("valid_before")
                                                                       or t < row["valid_before"])


def _region(lat, lon) -> dict:
    for row in SOURCE_BY_REGION:
        if row["kind"] == "all":
            return row
        if row["kind"] == "box":
            lo = ((lon + 180.0) % 360.0) - 180.0
            if row["lat"][0] <= lat <= row["lat"][1] and row["lon"][0] <= lo <= row["lon"][1]:
                return row
        if row["kind"] == "rotated":
            tlm, tph = geographic_to_rotated(lat, lon, row["tph0d"], row["tlm0d"])
            if abs(tlm) <= -row["wbd"] and abs(tph) <= -row["sbd"]:
                return row
    raise AssertionError("SOURCE_BY_REGION has no 'all' row")


def default_source(lat, lon, cycle=None) -> tuple[str, str]:
    """(source, region) for a centre and start cycle: the region row's first source that serves the cycle."""
    row = _region(lat, lon)
    for name in row["sources"]:
        if serves(name, cycle):
            return name, row["region"]
    raise SystemExit(f"--source auto: no source row serves {cycle:%Y-%m-%d %H}Z in region {row['region']} "
                     f"(tried {row['sources']}; see namgpu/init/sources.py valid_from)")


def _num(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".") if x != int(x) else f"{x:.1f}"


def nest_grid(lat: float, lon: float, size_km: float, res_km: float, size_y_km: float | None = None) -> dict:
    res = int(res_km) if float(res_km) == int(res_km) else res_km
    if res not in RES_TABLE:
        raise SystemExit(f"--res-km {res_km}: no row in namgpu/nest/domain.py RES_TABLE (rows: "
                         f"{sorted(RES_TABLE)} km); a new resolution is a row with its dt/filt_dt/nrads")
    if not (-89.0 < lat < 89.0):
        raise SystemExit(f"--center latitude {lat}: the rotated grid's centre must lie off the poles")
    dlmd, dphd = 0.0105 * res, 0.009 * res
    dx = A_EARTH * math.radians(dlmd) / 1e3
    dy = A_EARTH * math.radians(dphd) / 1e3
    im = int(round(size_km / dx)) + 1
    jm = int(round((size_y_km or size_km) / dy)) + 1
    lon = ((lon + 180.0) % 360.0) - 180.0
    return {"res_km": res, "im": im, "jm": jm, "lm": 60, "tph0d": round(lat, 4), "tlm0d": round(lon, 4),
            "dlmd": dlmd, "dphd": dphd, "wbd": round(-(im - 1) / 2 * dlmd, 6), "sbd": round(-(jm - 1) / 2 * dphd, 6),
            "dx_km": round(dx, 4), "dy_km": round(dy, 4),
            "extent_km": [round((im - 1) * dx, 1), round((jm - 1) * dy, 1)]}


def _deg(x: float) -> str:
    """Centre degrees at the grid's own 4 decimals, trailing zeros cut down to 2 (38.6 -> 38.60, 38.604 -> 38.604)."""
    s = f"{abs(x):.4f}"
    while s.endswith("0") and len(s.split(".")[1]) > 2:
        s = s[:-1]
    return s


def domain_name(g: dict, size_km: float, size_y_km: float | None = None) -> str:
    """Folder and case name of a nest grid. Every grid with its own boundaries needs its own name: the case boco folder
    is boco-<name>, so the name carries the centre at the grid's 4 decimals and the north-south size when it differs
    (review round 3, #1: 38.6/38.604 and 1000/1000x600 km grids shared one name and one boco folder)."""
    lat, lon = g["tph0d"], g["tlm0d"]
    size = f"{int(round(size_km))}"
    if size_y_km and int(round(size_y_km)) != int(round(size_km)):
        size += f"x{int(round(size_y_km))}"
    return f"nest{g['res_km']}km_{_deg(lat)}{'N' if lat >= 0 else 'S'}_{_deg(lon)}{'E' if lon >= 0 else 'W'}_{size}"


def grid_key(g: dict) -> str:
    s = "|".join(str(g[k]) for k in ("im", "jm", "lm", "tph0d", "tlm0d", "wbd", "sbd"))
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def render(base: str, repl: dict, tag: str) -> str:
    """parm/nam_configfile text with the listed keys replaced in place (comments kept); keys NAM does not have are
    appended as port keys. (Moved here from tools/make_domain_configs.py, which imports it.)"""
    import re
    out, seen = [], set()
    for line in base.splitlines():
        m = re.match(r"^(\s*)([A-Za-z_0-9]+)(\s*:\s*)(\S+)(.*)$", line)
        if m and m.group(2) in repl and m.group(2) not in seen:
            k = m.group(2)
            seen.add(k)
            out.append(f"{m.group(1)}{k}{m.group(3)}{repl[k]}{m.group(5)}")
        else:
            out.append(line)
    extra = [k for k in repl if k not in seen]
    if extra:
        out.append("")
        out.append(f"# nam-gpu port keys for the {tag} domain (not NAM keys; namgpu/config.py)")
        for k in extra:
            out.append(f"{k}: {repl[k]}")
    return "\n".join(out) + "\n"


def config_keys(g: dict, name: str, static_npz: Path, spec_adv: bool = False, handoff: bool = False) -> dict:
    row = RES_TABLE[g["res_km"]]
    keys = {"im": str(g["im"]), "jm": str(g["jm"]), "tph0d": _num(g["tph0d"]), "tlm0d": _num(g["tlm0d"]),
            "wbd": f"{g['wbd']:.6f}", "sbd": f"{g['sbd']:.6f}"}
    if row["physics"] == "conusnest":
        di, dn, dd = row["dt"]
        fi, fn, fd = row["filt"]
        keys.update(dt_int=str(di), dt_num=str(dn), dt_den=str(dd), filt_dt_int=str(fi), filt_dt_num=str(fn),
                    filt_dt_den=str(fd), nrads=str(row["nrads"]), nradl=str(row["nrads"]), **NEST_PHYSICS)
        keys["spec_adv"] = ".true." if spec_adv else ".false."
    elif spec_adv:
        keys["spec_adv"] = ".true."
    keys["static_npz"] = str(static_npz)
    keys["domain_name"] = name
    if handoff:
        keys["history_handoff"] = "true"       # io_history_fields.HANDOFF_KEY
    return keys


def write_configfile(path: Path, g: dict, name: str, static_npz: Path, spec_adv: bool = False,
                     handoff: bool = False) -> dict:
    keys = config_keys(g, name, static_npz, spec_adv, handoff)
    Path(path).write_text(render((PARM / "nam_configfile").read_text(), keys, name))
    from namgpu import config
    c = config.load(Path(path))
    r = c.raw
    dt = Fraction(r["dt_int"]) + (Fraction(r["dt_num"], r["dt_den"]) if r["dt_den"] else 0)
    chk = {"im": c.im, "jm": c.jm, "dlmd": round(c.dlmd, 9), "dphd": round(c.dphd, 9), "dt": float(dt),
           "filt_dt": c.filt_dt, "nrads": c.nrads, "nradl": c.nradl, "lnsh": c.lnsh, "lnsv": c.lnsv,
           "convection": c.convection, "gwdflg": c.gwdflg, "spec_adv": r.get("spec_adv"),
           "steps_per_hour": float(3600 / dt), "res_row": RES_TABLE[g["res_km"]]["src"],
           "history_handoff": r.get("history_handoff", "false")}
    if abs(chk["dlmd"] - g["dlmd"]) > 1e-7 or abs(chk["dphd"] - g["dphd"]) > 1e-7:
        raise AssertionError(f"configfile spacing {chk['dlmd']}/{chk['dphd']} != {g['dlmd']}/{g['dphd']}")
    # TDFI (filter_method 3) needs whole filter steps in each window (drv_dfi_tdfi_driver.schedule refuses
    # otherwise, after the init has run): checked here, in exact fractions, before anything is fetched.
    if int(c.filter_method) == 3:
        fdt = Fraction(int(r["filt_dt_int"])) + (Fraction(int(r["filt_dt_num"]), int(r["filt_dt_den"]))
                                                 if int(r["filt_dt_den"]) else 0)
        for k in ("nsecs_bcktdfi", "nsecs_fwdtdfi"):
            w = Fraction(str(r[k]))
            if fdt <= 0 or (w / fdt).denominator != 1:
                raise AssertionError(f"RES_TABLE {g['res_km']} km: {k} {w} s is not a multiple of filt_dt {fdt} s "
                                     f"(TDFI needs whole filter steps)")
        chk["tdfi_steps_per_leg"] = int(Fraction(str(r["nsecs_bcktdfi"])) / fdt)
    return chk


def _visible_index() -> str:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return v if v else "0"


def card_name(index: str | None = None) -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                               index or _visible_index()], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def sm_threads(index: str | None = None) -> int:
    name = card_name(index)
    for key, sms in CARD_SMS:
        if key in name:
            return sms * 1536
    return 170 * 1536


def free_gib(index: str | None = None) -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i",
                          index or _visible_index()], capture_output=True, text=True, timeout=20).stdout
    return float(out.strip().splitlines()[0]) / 1024.0


def card_total_gib(index: str | None = None) -> float:
    """The card's total memory in GiB from nvidia-smi (no CUDA context); 0.0 when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i",
                              index or _visible_index()], capture_output=True, text=True, timeout=20).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0.0


def memory_path(index: str | None = None) -> dict:
    """The memory path the forecast will take on this card: state.resolve_memory_path on the same setting
    (NAMGPU_MEMORY_PATH, else the older NAMGPU_SCRATCH_ARENA) and the card's total memory, as state.memory_path reads
    them in the forecast process. path None when no setting is given and the card size cannot be read."""
    from namgpu import state
    want = os.environ.get("NAMGPU_MEMORY_PATH", "").strip().lower()
    old = os.environ.get("NAMGPU_SCRATCH_ARENA", "").strip()
    if not want and old in ("0", "1"):
        want = "fit" if old == "1" else "full"
    card = card_total_gib(index)
    if want not in ("fit", "full") and card <= 0.0:
        return {"path": None, "how": "unknown", "card_gib": 0.0,
                "reason": "nvidia-smi gave no card size and NAMGPU_MEMORY_PATH is not set"}
    return state.resolve_memory_path(want, card)


def vram_estimate(g: dict, rows: int, cards: int = 1, index: str | None = None, path: str | None = "auto") -> dict:
    """VRAM estimate for one card of the run. path: "fit" or "full"; "auto" = the path the forecast takes on this
    card (memory_path); None = unknown: then the larger (full) row is used for the fit check and no pool cap is set,
    because a cap from the wrong path's model either kills the run (too small) or protects no co-tenant."""
    mp = memory_path(index) if path == "auto" else {"path": path}
    row = VRAM_PATH[mp["path"] or "full"]
    cols = g["im"] * g["jm"] / max(1, cards)
    state = cols * VRAM["per_column_kb"] * 1024 / 2**30
    rrtm = rows * g["im"] * VRAM["rrtm_mb_per_row_col"] * 1e6 / 2**30
    lmem = VRAM["lmem_bytes_per_thread"] * sm_threads(index) / 2**30
    pool = state + row["fixed_pool_gib"] + rrtm + VRAM["pool_headroom_gib"]
    tot = pool + lmem + VRAM["fixed_gib"]
    capped = mp["path"] is not None
    return {"columns_per_card": int(cols), "state_gib": round(state, 2), "rrtm_gib": round(rrtm, 2),
            "lmem_gib": round(lmem, 2), "fixed_gib": VRAM["fixed_gib"], "memory_path": mp["path"],
            "path_fixed_gib": row["fixed_pool_gib"], "pool_limit_gib": round(pool, 2) if capped else None,
            "pool_limit_bytes": int(pool * 2**30) if capped else None, "total_gib": round(tot, 2),
            "rrtm_rows": rows}


def preflight(g: dict, cards: int = 1, rows: int | None = None, free: float | None = None,
              index: str | None = None) -> dict:
    """Pick NAMGPU_RRTM_ROWS (largest that fits) against the measured free memory, or refuse with the numbers."""
    free = free_gib(index) if free is None else free
    tried = []
    for r in ((rows,) if rows else RRTM_ROW_CHOICES):
        e = vram_estimate(g, r, cards, index)
        tried.append(e)
        if e["total_gib"] + VRAM["margin_gib"] <= free:
            return {"ok": True, "free_gib": round(free, 2), "pick": e, "tried": tried, "card": card_name(index)}
    return {"ok": False, "free_gib": round(free, 2), "tried": tried, "card": card_name(index)}


def preflight_cards(g: dict, cards: list[str], rows: int | None = None) -> dict:
    """preflight for a run on several cards: every card in the list must hold its slab now (nvidia-smi free memory
    per card, no CUDA context). The largest NAMGPU_RRTM_ROWS that fits on EVERY card is picked; free_gib is the
    smallest card's. Before this, --cards N > 1 skipped every free-memory check (review round 3, #2), so a rank
    could start on a card another process had filled and die with OutOfMemoryError in init-grib2-decode."""
    if len(cards) == 1:
        pf = preflight(g, 1, rows, index=cards[0])
        pf["free_by_card"] = {cards[0]: pf["free_gib"]}
        return pf
    free = {c: free_gib(c) for c in cards}
    tried = []
    for r in ((rows,) if rows else RRTM_ROW_CHOICES):
        es = {c: vram_estimate(g, r, len(cards), c) for c in cards}
        e = max(es.values(), key=lambda x: x["total_gib"])
        tried.append(e)
        if all(es[c]["total_gib"] + VRAM["margin_gib"] <= free[c] for c in cards):
            return {"ok": True, "free_gib": round(min(free.values()), 2), "free_by_card": {c: round(v, 2) for c, v in
                    free.items()}, "pick": e, "tried": tried, "card": card_name(cards[0]), "cards": cards}
    return {"ok": False, "free_gib": round(min(free.values()), 2), "free_by_card": {c: round(v, 2) for c, v in
            free.items()}, "tried": tried, "card": card_name(cards[0]), "cards": cards}
