"""init-static-build: the NAM static surface for any rotated lat-lon B grid, from WPS_GEOG, on the GPU.

Mode "new": the NPS geogrid + nemsinterp static step for an NMMB B grid (never Fortran geogrid). Writes
the layout of static/nam12_static.npz (gfs_start.STATIC_KEYS, in the units of NAM's own bgrd3d00
records) plus the fields init-static-missing otherwise fills, so a GFS start runs on any configured grid.

    python -m namgpu.init.static_build --config parm/nam_configfile_4km --date 2025-03-14 \
        --out static/nam4km_static.npz [--geog ~/WPS_GEOG]

Sources and interpolation (NAM's parm/nam_GEOGRID.TBL.NMB option names; WPS_GEOG datasets of the
ArWen/HRRR prep set):
  HGT_SFC  topo_gmted2010_30s               average_gcell(4.0)+four_pt+average_4pt, fill 0,
                                            smth-desmth_special x3 on the 3-cell processing halo
  LAND     modis_landuse_20class_30s_with_lakes: water (17) + lake (21) fraction >= 0.5 -> 0, else 1
  VGTYP    same: dominant IGBP land class (ties: lowest), water 0 (bgrd3d00 convention; NPS writes 17
           and NAM's GRIB post masks it to 0); ivegsrc 1
  SOTYP    soilgrids/texture_top (SoilGrids250m, CC BY 4.0; the same 16 STATSGO texture classes, water 14):
           dominant class; land cells take the dominant non-water class (NPS find_secondary_cat), LOAM (6)
           if no land soil pixel; land ice (IGBP 15) 16, as STATSGO/FAO carry it (SoilGrids has no texture
           under ice); water 0 (bgrd3d00 convention). NMMB's Noah reads one soil type, so the
           bottom layer (texture_bot) is not read.
           Soil set: NAMGPU_GEOG_SOIL (or --soil) "permissive" (the default, above); "wrf" is the WPS
           FAO-UNESCO/STATSGO soiltype_top_30s, an explicit opt-in that prints and records its licence hold.
  SFCR     NPS module_assorted_definitions.F90:47-52, 757-768: VGZ0TBL_IGBP(VGTYP) + Z0LAND 0.10 on land
           (no Z0LAND for snow/ice 15); water 0 as in bgrd3d00
  VEG      greenfrac_fpar_modis, monthly, mid-month linear to the valid date, percent; average_gcell(4.0)
           +four_pt+average_4pt+average_16pt+search (ARW table option for the 30 s MODIS set; NAM's table
           names four_pt for its old 0.144 deg set), masked water -> 0
  ALBDO    albedo_modis, monthly, to the valid date, percent; four_pt+average_4pt+average_16pt+search;
           water 6 (NPS SEA albedo .06, module_assorted_definitions.F90:143)
  ALBASE   ALBDO / 100 on land, 0.06 on water (fraction; NPS ALBASE)
  MXSNAL   maxsnowalb_modis / 100 (fraction), four_pt+average_4pt+average_16pt+search, water 0
  TG       soiltemp_era5_1deg (ERA5 1990-2019 annual-mean 2 m temperature reduced to sea level with
           0.0065 K/m, soiltemp_1deg's layout and convention), sixteen_pt+four_pt+average_4pt+average_16pt
           +search, masked water; NPS: TG = SOILTEMP, water with TG < 180 K -> 273.15
           (module_assorted_definitions.F90:120-124)
Licence: only permissive datasets are read (DATASETS below, source and licence lines copied from gpuwm
2.9.12); WPS's FAO-based soiltype_*_30s and ISLSCP-I soiltemp_1deg are refused by name (HELD) unless the
"wrf" soil set is selected explicitly (opt-in, never a default; its hold is recorded with the build). Every
written npz carries the table in META["datasets"] and in <static>.PROVENANCE.json beside it.
  SLOPE    slope type 1 on land, 0 on water (NPS fill_missing 0; NAM's islope data set is not in GEOG)
  EPSR     0.97 water / 1.0 land, STDH -1.E6: the init-static-missing rules, written here so the file
           is complete.
Declared differences from NAM's own static: ALBDO here is snow-free (NAM's bgrd3d00 ALBDO carries the
analysis-time snow); each new grid has its own rotated pole; no sea-ice reclassification (that is
analysis-time state). Every stage is on the GPU except tile reads, which run on all cores.
"""
from __future__ import annotations

from namgpu import cores as _cores

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "init_static_build"
ROOT = Path(__file__).resolve().parents[2]
DTR = np.pi / 180.0
HALO = 3
GCELL_RATIO = 4.0
M_PER_DEG = 6370000.0 * np.pi / 180.0
VGZ0TBL_IGBP = np.array([1.800, 2.653, 0.854, 2.467, 1.966, 0.050, 0.030, 0.856, 0.856, 0.080, 0.040,
                         0.170, 1.000, 0.500, 0.011, 0.011, 0.001, 0.076, 0.050, 0.030], np.float32)
Z0LAND = 0.10
IGBP_WATER, IGBP_LAKE, IGBP_SNOW = 17, 21, 15
SOIL_WATER, SOIL_LOAM, SOIL_LANDICE = 14, 6, 16
METHODS = {"four_pt": 1, "average_4pt": 2, "average_16pt": 3, "sixteen_pt": 4, "search": 5, "nearest": 6}
SEQ_MONTHLY = ("four_pt", "average_4pt", "average_16pt", "search")
SEQ_TG = ("sixteen_pt", "four_pt", "average_4pt", "average_16pt", "search")
# The WPS_GEOG datasets each role reads. Source and licence lines are copied, not rewritten, from gpuwm
# 2.9.12: docs/public/DATA.md's licence table (the rows gpuwm/geog_assets.py's pins speak for), NOTICE's
# SoilGrids credit, and geog_assets.GEOG_PRODUCERS["era5-annual-t2m-sea-level"] for the ERA5 soil temperature.
DATASETS = {
    "terrain": {
        "dataset": "topo_gmted2010_30s", "contents": "30-arc-sec terrain elevation",
        "source_and_licence": "USGS/NGA GMTED2010 (Danielson & Gesch 2011, USGS OFR 2011-1073); U.S. public domain"},
    "landuse": {
        "dataset": "modis_landuse_20class_30s_with_lakes",
        "contents": "Noah-modified 20-category IGBP land use + inland lakes",
        "source_and_licence": "NASA MODIS (MCD12Q1-derived); NASA data are free and open"},
    "soil_top": {
        "dataset": "soilgrids/texture_top",
        "contents": "16-category top-/bottom-layer soil texture, Noah-MP soil composition and four-layer texture",
        "source_and_licence": ("ISRIC -- World Soil Information, SoilGrids250m (Hengl et al. 2017, PLOS ONE 12(2): "
                               "e0169748), from whose 2017 texture classes NCAR generated the WPS tiles; CC BY 4.0 "
                               "(ISRIC's terms for SoilGrids products since 2019), attribution required"),
        "attribution": ("derived from SoilGrids250m by ISRIC -- World Soil Information (Hengl T. et al. 2017, "
                        "\"SoilGrids250m: Global gridded soil information based on machine learning\", PLOS ONE "
                        "12(2): e0169748), licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); NCAR "
                        "generated the WPS tiles and the mirror does not modify them.")},
    "greenfrac": {
        "dataset": "greenfrac_fpar_modis", "contents": "monthly green-vegetation-fraction climatology",
        "source_and_licence": "NASA MODIS FPAR"},
    "albedo": {
        "dataset": "albedo_modis", "contents": "monthly surface albedo climatology",
        "source_and_licence": "NASA MODIS"},
    "maxsnowalb": {
        "dataset": "maxsnowalb_modis", "contents": "maximum snow albedo",
        "source_and_licence": "MODIS-derived (Barlage et al. 2005)"},
    "soiltemp": {
        "dataset": "soiltemp_era5_1deg",
        "contents": "1-degree annual-mean deep-soil temperature (2 m temperature reduced to sea level)",
        "source_and_licence": ("ERA5 (Hersbach et al. 2020, doi:10.1002/qj.3803), 1990-2019 hourly climatology as "
                               "published by WeatherBench 2 (Rasp et al. 2024, doi:10.1029/2023MS004019), built "
                               "locally by `gpuwm fetch-geog`; Licence to Use Copernicus Products "
                               "(<https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-"
                               "products.pdf>): free, worldwide, commercial use allowed, attribution required"),
        "attribution": ("Contains modified Copernicus Climate Change Service information 2026 (ERA5, Hersbach et "
                        "al. 2020, doi:10.1002/qj.3803), 1990-2019 climatology as processed by WeatherBench 2 (Rasp "
                        "et al. 2024, doi:10.1029/2023MS004019). Neither the European Commission nor ECMWF is "
                        "responsible for any use of this information."),
        "missing_remedy": "gpuwm fetch-geog --datasets soiltemp_era5_1deg (built from ERA5, 459 MB read)"},
}
LICENCE_COPIED_FROM = ("gpuwm 2.9.12 (16c90f4cdea): docs/public/DATA.md licence table, NOTICE (SoilGrids credit), "
                       "gpuwm/geog_assets.py GEOG_PRODUCERS / FAO_DSMW_HOLD / ISLSCP_SOILTEMP_HOLD")
# Held out (licence, not physics), quoted from gpuwm/geog_assets.py FAO_DSMW_HOLD and ISLSCP_SOILTEMP_HOLD.
_FAO_DSMW_HOLD = (
    "STATSGO (USDA NRCS, public domain) inside CONUS + the FAO-UNESCO "
    "Digital Soil Map of the World elsewhere, whose notes (Version 3.6, "
    "Jan 2003, https://swat.tamu.edu/media/116411/notes.txt) say "
    "\"(C) FAO/UNESCO, 1995 All rights reserved worldwide\" and \"The user "
    "shall not reproduce for the benefit of third parties the data\" "
    "without permission from FAO's Publications Division: NOT PERMISSIVE, "
    "held out of the default geography; the default soil categories are "
    "SoilGrids250m (soilgrids/texture_top, texture_bot, CC BY 4.0)")
_ISLSCP_SOILTEMP_HOLD = (
    "1-degree annual-mean 2 m air temperature from ISLSCP-I (ECMWF "
    "1987-88 analyses, adjusted to 1000 mb) per the Noah-MP TBOT READ_ME; "
    "the ISLSCP-I Vol. 4 record at CEDA "
    "(https://catalogue.ceda.ac.uk/uuid/b9c3effaab9fe9a2206b58af500f157d) "
    "states Permitted Use \"Unstated\" and NCAR states none: licence "
    "UNCLEAR, so not cleared and held out of the default geography; the "
    "default deep-soil temperature is soiltemp_era5_1deg (ERA5, Licence "
    "to Use Copernicus Products)")
HELD = {"soiltype_top_30s": _FAO_DSMW_HOLD, "soiltype_bot_30s": _FAO_DSMW_HOLD, "soiltemp_1deg": _ISLSCP_SOILTEMP_HOLD}


def datasets_key() -> str:
    """8 hex digits naming the dataset table (role -> dataset): the nest door's static cache key carries it, so a
    file built from a different table is never reused."""
    import hashlib
    import json
    return hashlib.sha1(json.dumps(sorted((k, v["dataset"]) for k, v in DATASETS.items())).encode()).hexdigest()[:8]


class LicenceHold(ValueError):
    """A held WPS_GEOG dataset was named for a static build."""


def resolve_datasets(root, overrides=None, soil: str = "permissive") -> dict:
    """Role -> licence-table row (with the resolved path) for this build. THE BREAKAGE A HELD NAME WOULD
    CAUSE: a static (and every forecast from it) built on FAO-UNESCO DSMW soil, which may not be reproduced
    for third parties, or on ISLSCP-I soil temperature, whose licence nobody states. Refused by name."""
    import json
    rows = {k: dict(v) for k, v in DATASETS.items()}
    roots = list(root) if isinstance(root, (list, tuple)) else [Path(root)]
    if soil != "permissive":
        # an explicit opt-in soil set (NAMGPU_GEOG_SOIL / --soil, never a default): its hold is the licence line
        for role, key in (("soil_top", "soil"), ("soiltemp", "soiltemp")):
            name = SOIL_SETS[soil][key]
            rows[role] = {"dataset": name, "contents": rows[role]["contents"], "opt_in": soil,
                          "source_and_licence": f"opt-in soil set {soil}: " + HELD.get(name, "WPS_GEOG " + name)}
    for role, name in (overrides or {}).items():
        if role not in rows:
            raise ValueError(f"{UNIT}: unknown dataset role {role!r} (roles: {sorted(rows)})")
        name = name.strip().strip("/")
        held = HELD.get(name) or HELD.get(Path(name).name)
        if held:
            raise LicenceHold(f"{UNIT}: {name} is held out of the static build: {held}")
        if name != rows[role]["dataset"]:
            raise ValueError(f"{UNIT}: {role}={name} has no licence row here; the {role} dataset is "
                             f"{rows[role]['dataset']} ({rows[role]['source_and_licence']})")
    for role, row in rows.items():
        if row["dataset"] in HELD and not row.get("opt_in"):
            raise LicenceHold(f"{UNIT}: {row['dataset']} is held out of the static build: {HELD[row['dataset']]}")
        # each dataset is looked up across the roots (geog_roots: a partial NAMGPU_GEOG that links its terrain
        # into the full WPS_GEOG still finds the rest)
        path = next((r / row["dataset"] for r in roots if (r / row["dataset"] / "index").is_file()), None)
        if path is None:
            fix = row.get("missing_remedy", "gpuwm fetch-geog")
            raise FileNotFoundError(f"{UNIT}: {row['dataset']}/index missing under {[str(r) for r in roots]} "
                                    f"({role}: {row['source_and_licence']}); fetch it with {fix}")
        row["role"] = role
        row["path"] = str(path)
        prov = path / "PROVENANCE.json"
        if prov.is_file():
            pj = json.loads(prov.read_text())
            row["dataset_provenance"] = {k: pj[k] for k in ("producer", "produced_at", "licence", "attribution")
                                         if k in pj}
        rows[role] = row
    return rows


OUT_KEYS = ("HGT_SFC", "LAND", "VGTYP", "SOTYP", "SFCR", "VEG", "ALBDO",
            "ALBASE", "MXSNAL", "TG", "SLOPE", "EPSR", "STDH", "GLAT", "GLON")


#: Soil sets (NAMGPU_GEOG_SOIL selects one; default "permissive"). "permissive" is the soil set gpuwm 2.9.4 made
#: its default: SoilGrids 16-class texture (WPS 4.4+ SOILCTOP rel_path soilgrids/texture_top, the same STATSGO
#: category numbering, 14 = water) and the ERA5 1990-2019 deep-soil temperature (Copernicus; attribution in its
#: PROVENANCE.json). "wrf" is the WPS FAO-UNESCO/STATSGO soiltype_top_30s and ISLSCP-I soiltemp_1deg, whose
#: licences are held: opt-in only, never a default.
SOIL_SETS = {
    "permissive": {"soil": "soilgrids/texture_top", "soiltemp": "soiltemp_era5_1deg"},
    "wrf": {"soil": "soiltype_top_30s", "soiltemp": "soiltemp_1deg"},
}


def soil_set(name=None) -> str:
    name = (name or os.environ.get("NAMGPU_GEOG_SOIL") or "permissive").strip().lower()
    if name not in SOIL_SETS:
        raise SystemExit(f"{UNIT}: NAMGPU_GEOG_SOIL={name!r}: not one of {sorted(SOIL_SETS)}")
    return name


def geog_roots(arg=None) -> list:
    """Folders searched for each WPS_GEOG dataset, in order: --geog, NAMGPU_GEOG, the tree a link in NAMGPU_GEOG
    points into (a partial root, such as a geography folder that links topo_gmted2010_30s from the full WPS_GEOG),
    WPS_GEOG, ~/WPS_GEOG."""
    out = []
    for r in ([arg] if arg else []) + [os.environ.get("NAMGPU_GEOG")]:
        if r and Path(r).is_dir():
            out.append(Path(r))
            topo = Path(r) / "topo_gmted2010_30s"
            if topo.is_symlink():
                out.append(topo.resolve().parent)
    for r in (os.environ.get("WPS_GEOG"), str(Path.home() / "WPS_GEOG")):
        if r and Path(r).is_dir():
            out.append(Path(r))
    seen, roots = set(), []
    for r in out:
        if str(r) not in seen:
            seen.add(str(r))
            roots.append(r)
    return roots


def dataset(roots, rel: str) -> Path:
    for r in roots:
        if (r / rel / "index").is_file():
            return r / rel
    raise FileNotFoundError(f"{UNIT}: WPS_GEOG dataset {rel} not found under {[str(r) for r in roots]} "
                            "(NAMGPU_GEOG, WPS_GEOG or ~/WPS_GEOG)")


def geog_root(arg=None) -> Path:
    for r in geog_roots(arg):
        if (r / "topo_gmted2010_30s").is_dir():
            return r
    raise FileNotFoundError(f"{UNIT}: no WPS_GEOG tree (NAMGPU_GEOG or ~/WPS_GEOG)")


def _index(path: Path) -> dict:
    kv = {}
    for raw in (path / "index").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip().lower()] = v.strip().strip("'\"")
    return kv


class Source:
    """One WPS_GEOG dataset with the tiles a domain touches resident on the GPU."""

    def __init__(self, path: Path, planes=(0,)):
        import cupy as cp
        kv = _index(path)
        self.path = path
        f = lambda k, d=None: float(kv[k]) if k in kv else d   # noqa: E731
        self.dx, self.dy = f("dx"), f("dy")
        self.known_x, self.known_y = f("known_x", 1.0), f("known_y", 1.0)
        self.known_lat, self.known_lon = f("known_lat"), f("known_lon")
        self.scale = f("scale_factor", 1.0)
        self.missing = kv.get("missing_value")
        self.word = int(kv["wordsize"])
        self.signed = kv.get("signed", "no").lower() in ("yes", "true", "1")
        self.endian = kv.get("endian", "big").lower()
        self.tx, self.ty = int(kv["tile_x"]), int(kv["tile_y"])
        self.bdr = int(kv.get("tile_bdr", 0))
        if "tile_z" in kv:
            self.nz = int(kv["tile_z"])
        else:
            self.nz = int(kv.get("tile_z_end", 1)) - int(kv.get("tile_z_start", 1)) + 1
        self.ncat = int(kv["category_max"]) if kv.get("type", "") == "categorical" else 0
        self.nxg = int(round(360.0 / abs(self.dx)))
        self.nyg = int(round(180.0 / abs(self.dy)))
        self.wraps = 1
        self.planes = tuple(planes)
        base = {1: "u1", 2: "i2" if self.signed else "u2"}[self.word]
        self.dtype = np.dtype(("<" if self.endian == "little" else ">") + base) if self.word > 1 else np.dtype(base)
        self.kind = {"u1": "u8", "i2": "i16", "u2": "u16"}[base]
        self.ntx = (self.nxg + self.tx - 1) // self.tx
        self.nty = (self.nyg + self.ty - 1) // self.ty
        self.files = {}
        for p in path.iterdir():
            n = p.name.split(".")
            if len(n) == 2 and "-" in n[0] and "-" in n[1]:
                xs, ys = int(n[0].split("-")[0]), int(n[1].split("-")[0])
                self.files[((xs - 1) // self.tx, (ys - 1) // self.ty)] = p
        self.cp = cp

    def desc(self):
        cp = self.cp
        sd = cp.asarray(np.array([self.known_x, self.known_y, self.known_lat, self.known_lon,
                                  self.dx, self.dy, self.scale], np.float64))
        miss = self.missing is not None
        si = cp.asarray(np.array([self.nxg, self.nyg, self.tx, self.ty, self.ntx, self.wraps, int(miss),
                                  int(float(self.missing)) if miss else 0], np.int32))
        return sd, si

    def latlon_to_xy(self, lat_deg, lon_deg):
        xp = self.cp.get_array_module(lat_deg)
        d = xp.mod(lon_deg - self.known_lon, 360.0)
        x = self.known_x + d / self.dx
        y = self.known_y + (lat_deg - self.known_lat) / self.dy
        return x, y

    def load(self, tiles_needed):
        """Read the needed tiles (all cores), crop the border, pack them on the GPU."""
        cp = self.cp
        keys = sorted(k for k in tiles_needed if k in self.files)
        b, tx, ty = self.bdr, self.tx, self.ty
        plane_words = (ty + 2 * b) * (tx + 2 * b)

        def read(k):
            out = np.empty((len(self.planes), ty, tx), self.dtype.newbyteorder("="))
            with open(self.files[k], "rb") as fh:
                for n, z in enumerate(self.planes):
                    fh.seek(z * plane_words * self.word)
                    a = np.fromfile(fh, dtype=self.dtype, count=plane_words).reshape(ty + 2 * b, tx + 2 * b)
                    out[n] = a[b:b + ty, b:b + tx]
            return out

        with ThreadPoolExecutor(max_workers=_cores.usable()) as ex:
            got = list(ex.map(read, keys))
        packed = np.stack(got) if got else np.zeros((0, len(self.planes), ty, tx), self.dtype.newbyteorder("="))
        self.tiles = cp.asarray(packed)
        lut = np.full(self.nty * self.ntx, -1, np.int32)
        for s, (ti, tj) in enumerate(keys):
            lut[tj * self.ntx + ti] = s
        self.lut = cp.asarray(lut)
        self.slot_xs = cp.asarray(np.array([ti * tx + 1 for ti, _ in keys], np.int32))
        self.slot_ys = cp.asarray(np.array([tj * ty + 1 for _, tj in keys], np.int32))
        self.nslot = len(keys)
        return len(keys), packed.nbytes


class Domain:
    """Rotated lat-lon B grid (H points) extended by HALO cells."""

    def __init__(self, cfg, halo=HALO):
        self.im, self.jm, self.halo = int(cfg.im), int(cfg.jm), halo
        self.ie, self.je = self.im + 2 * halo, self.jm + 2 * halo
        self.tph0, self.tlm0 = cfg.tph0d * DTR, cfg.tlm0d * DTR
        self.wb, self.sb = cfg.wbd * DTR, cfg.sbd * DTR
        self.dlm, self.dph = cfg.dlmd * DTR, cfg.dphd * DTR
        self.dy_m = self.dph / DTR * M_PER_DEG

    def latlon(self, xp, ii, jj):
        """Geographic lat/lon (deg) at fractional extended-grid indices."""
        tlm = self.wb + (ii - self.halo) * self.dlm
        tph = self.sb + (jj - self.halo) * self.dph
        st0, ct0 = np.sin(self.tph0), np.cos(self.tph0)
        sphi = xp.clip(st0 * xp.cos(tph) * xp.cos(tlm) + ct0 * xp.sin(tph), -1.0, 1.0)
        lat = xp.arcsin(sphi)
        lon = self.tlm0 + xp.arctan2(xp.cos(tph) * xp.sin(tlm), ct0 * xp.cos(tph) * xp.cos(tlm) - st0 * xp.sin(tph))
        lon = (lon + np.pi) % (2 * np.pi) - np.pi
        return lat / DTR, lon / DTR

    def tiles_touched(self, src: Source, margin_cells=3):
        """Tiles holding any pixel near the extended grid: sample the grid at quarter-cell steps
        (plus margin) and every tile a sample falls in, with its neighbours where the sample sits
        within 4 source pixels of the tile edge (the 16-point stencils)."""
        cp = src.cp
        step = 0.25
        ii = cp.arange(-margin_cells, self.ie + margin_cells, step)
        jj_all = cp.arange(-margin_cells, self.je + margin_cells, step)
        # rows in blocks of ~2 M samples, so the probe stays ~100 MB on a shared card at any grid size (the
        # whole 12 km parent at once took ~1.2 GB and ran out beside other work, lane static-world 2026-11-04)
        rows = max(1, 2_000_000 // int(ii.size))
        found = set()
        for r0 in range(0, int(jj_all.size), rows):
            J, I = cp.meshgrid(jj_all[r0:r0 + rows], ii, indexing="ij")
            lat, lon = self.latlon(cp, I.ravel(), J.ravel())
            x, y = src.latlon_to_xy(lat, lon)
            del J, I, lat, lon
            keys = []
            for ox in (-4, 0, 4):
                for oy in (-4, 0, 4):
                    xi = (cp.floor(x + ox).astype(cp.int64) - 1) % src.nxg
                    yi = cp.floor(y + oy).astype(cp.int64)
                    ok = (yi >= 1) & (yi <= src.nyg)
                    keys.append(cp.unique((yi[ok] - 1) // src.ty * src.ntx + xi[ok] // src.tx))
            t = cp.unique(cp.concatenate(keys))
            found.update((int(v) % src.ntx, int(v) // src.ntx) for v in cp.asnumpy(t))
        return found


def _launch_gcell(dom, src, plane, ncat):
    cp = src.cp
    ncell = dom.ie * dom.je
    if ncat:
        sums = cp.zeros(1, cp.int64)
        counts = cp.zeros(ncat * ncell, cp.int32)
    else:
        sums = cp.zeros(ncell, cp.int64)
        counts = cp.zeros(ncell, cp.int32)
    if src.nslot:
        sd, si = src.desc()
        k = kernels.get(UNIT, f"gcell_{src.kind}")
        n = src.nslot * src.tx * src.ty
        k(((n + 255) // 256,), (256,),
          (src.tiles, src.slot_xs, src.slot_ys, np.int32(src.nslot), np.int32(plane), np.int32(len(src.planes)),
           sd, si, np.float64(dom.tph0), np.float64(dom.tlm0), np.float64(dom.wb), np.float64(dom.sb),
           np.float64(dom.dlm), np.float64(dom.dph), np.int32(dom.halo), np.int32(dom.ie), np.int32(dom.je),
           np.int32(ncat), sums, counts))
    if ncat:
        return counts.reshape(ncat, dom.je, dom.ie)
    return sums.reshape(dom.je, dom.ie), counts.reshape(dom.je, dom.ie)


def _launch_sample(dom, src, plane, seq, need, search_r=1200):
    """Interpolate at the H points where need; returns (float64 values, got mask)."""
    cp = src.cp
    out = cp.zeros((dom.je, dom.ie), cp.float64)
    got = cp.zeros((dom.je, dom.ie), cp.uint8)
    sd, si = src.desc()
    methods = cp.asarray(np.array([METHODS[m] for m in seq], np.int32))
    k = kernels.get(UNIT, f"sample_{src.kind}")
    k(kernels.grid_2d(dom.ie, dom.je), kernels.BLOCK_2D,
      (src.tiles, src.lut, np.int32(plane), np.int32(len(src.planes)), sd, si,
       np.float64(dom.tph0), np.float64(dom.tlm0), np.float64(dom.wb), np.float64(dom.sb),
       np.float64(dom.dlm), np.float64(dom.dph), np.int32(dom.halo), np.int32(dom.ie), np.int32(dom.je),
       methods, np.int32(len(seq)), np.int32(search_r), cp.ascontiguousarray(need.astype(cp.uint8)), out, got))
    return out, got.astype(bool)


def _continuous(dom, src, plane, seq, gcell, need):
    """average_gcell(4.0) (when the source is >= 4x finer) then the interpolation sequence."""
    cp = src.cp
    val = cp.full((dom.je, dom.ie), cp.nan, cp.float64)
    if gcell and dom.dy_m / (abs(src.dy) * M_PER_DEG) >= GCELL_RATIO:
        s, c = _launch_gcell(dom, src, plane, 0)
        ok = c > 0
        val = cp.where(ok, s.astype(cp.float64) / cp.maximum(c, 1) * src.scale, val)
    todo = cp.isnan(val) & need
    if bool(todo.any()):
        v, g = _launch_sample(dom, src, plane, seq, todo, search_r=max(1, int(1200 if src.dx < 0.1 else 40)))
        val = cp.where(todo & g, v, val)
    return val


def _smth_desmth_special(a, passes):
    import cupy as cp
    orig = a

    def one(x, c):
        y = x.copy()
        y[:, 1:-1] = x[:, 1:-1] + c * (0.5 * (x[:, :-2] + x[:, 2:]) - x[:, 1:-1])
        z = y.copy()
        z[1:-1, :] = y[1:-1, :] + c * (0.5 * (y[:-2, :] + y[2:, :]) - y[1:-1, :])
        return z

    out = orig
    for _ in range(passes):
        out = one(one(out, 0.5), -0.52)
    return cp.where((orig >= 0.0) & (out < 0.0), orig, out)


def _dominant(counts, exclude=()):
    """argmax over categories (lowest wins ties), excluded categories never win."""
    import cupy as cp
    c = counts.astype(cp.int64)
    for e in exclude:
        c[e - 1] = -1
    return cp.argmax(c, axis=0).astype(cp.int32) + 1, c.max(axis=0)


def month_weights(valid: datetime):
    """WPS/real mid-month linear interpolation: (month_a index, month_b index, weight of b)."""
    from datetime import datetime as dt
    y, m = valid.year, valid.month
    mid = lambda yy, mm: dt(yy + (mm - 1) // 12, (mm - 1) % 12 + 1, 15)   # noqa: E731
    if valid >= mid(y, m):
        a, b = mid(y, m), mid(y, m + 1)
    else:
        a, b = mid(y, m - 1), mid(y, m)
    w = (valid - a).total_seconds() / (b - a).total_seconds()
    return (a.month - 1, b.month - 1, float(w))


def build(cfg, valid: datetime, geog=None, log=print, datasets=None, soil=None):
    import cupy as cp
    roots = geog_roots(geog)
    if not roots:
        raise FileNotFoundError(f"{UNIT}: no WPS_GEOG tree (--geog, NAMGPU_GEOG, WPS_GEOG or ~/WPS_GEOG)")
    sset = soil_set(soil)
    ds = resolve_datasets(roots, datasets, soil=sset)
    dom = Domain(cfg)
    t0 = time.perf_counter()
    rec = {"grid": [dom.im, dom.jm], "geog": str(Path(ds["terrain"]["path"]).parent), "valid": valid.isoformat(),
           "soil_set": sset, "stages": {},
           "datasets": [{k: v for k, v in r.items() if k != "missing_remedy"} for r in ds.values()],
           "licence_copied_from": LICENCE_COPIED_FROM, "empty_cells": {}}
    log(f"{UNIT}: soil set {sset}: {ds['soil_top']['path']}, {ds['soiltemp']['path']}")
    if sset != "permissive":
        log(f"{UNIT}: opt-in soil set {sset}: {HELD.get(ds['soil_top']['dataset'], '')}")
    src = lambda role, **kw: Source(Path(ds[role]["path"]), **kw)   # noqa: E731
    H = dom.halo
    crop = (slice(H, H + dom.jm), slice(H, H + dom.im))
    allc = cp.ones((dom.je, dom.ie), bool)

    def stage(name, t):
        rec["stages"][name] = round(time.perf_counter() - t, 2)
        log(f"{UNIT}: {name} {rec['stages'][name]} s")

    # Land use: categorical accumulation, empty cells nearest neighbour.
    t = time.perf_counter()
    lu = src("landuse")
    nt, nb = lu.load(dom.tiles_touched(lu))
    cnt = _launch_gcell(dom, lu, 0, lu.ncat)
    tot = cnt.sum(axis=0)
    empty = tot == 0
    rec["empty_cells"]["landuse"] = int(empty.sum())
    if bool(empty.any()):
        v, g = _launch_sample(dom, lu, 0, ("nearest",), empty)
        cat = cp.clip(cp.rint(v).astype(cp.int64), 1, lu.ncat)
        for c in range(1, lu.ncat + 1):
            cnt[c - 1] += (empty & g & (cat == c)).astype(cp.int32)
        tot = cnt.sum(axis=0)
    water = (cnt[IGBP_WATER - 1] + cnt[IGBP_LAKE - 1]).astype(cp.float64) / cp.maximum(tot, 1)
    land = (water < 0.5) & (tot > 0)
    vg, _ = _dominant(cnt, exclude=(IGBP_WATER, IGBP_LAKE))
    vg = cp.where(land, vg, 0)
    rec["landuse_tiles"] = nt
    del lu, cnt
    stage("landuse", t)

    # Soil.
    t = time.perf_counter()
    so = src("soil_top")
    so.load(dom.tiles_touched(so))
    cnt = _launch_gcell(dom, so, 0, so.ncat)
    tot = cnt.sum(axis=0)
    empty = tot == 0
    rec["empty_cells"]["soil_top"] = int(empty.sum())
    if bool(empty.any()):
        v, g = _launch_sample(dom, so, 0, ("nearest",), empty)
        cat = cp.clip(cp.rint(v).astype(cp.int64), 1, so.ncat)
        for c in range(1, so.ncat + 1):
            cnt[c - 1] += (empty & g & (cat == c)).astype(cp.int32)
    sl, mx = _dominant(cnt, exclude=(SOIL_WATER,))
    sl = cp.where(mx > 0, sl, SOIL_LOAM)
    # Land ice (IGBP 15) takes soil class 16, "other (land-ice)": the STATSGO/FAO tiles NAM's own static came
    # from carry it over the ice sheets (12,904 of the 14,668 NAM 12 km land-ice cells), SoilGrids has no
    # texture under the ice and the cells fell to LOAM (lane static-world, 2026-11-04).
    sl = cp.where(vg == IGBP_SNOW, SOIL_LANDICE, sl)
    sl = cp.where(land, sl, 0)
    del so, cnt
    stage("soil", t)

    # Terrain.
    t = time.perf_counter()
    tp = src("terrain")
    tp.load(dom.tiles_touched(tp))
    hgt = _continuous(dom, tp, 0, ("four_pt", "average_4pt"), True, allc)
    hgt = cp.where(cp.isnan(hgt), 0.0, hgt)
    hgt = _smth_desmth_special(hgt, 3)
    del tp
    stage("terrain", t)

    # Monthly greenness and albedo to the valid date.
    ma, mb, w = month_weights(valid)
    rec["months"] = [ma + 1, mb + 1, round(w, 4)]
    out = {}
    for key, role, seq, gc in (("VEG", "greenfrac", SEQ_MONTHLY, True),
                               ("ALBDO", "albedo", SEQ_MONTHLY, False)):
        t = time.perf_counter()
        s = src(role, planes=(ma, mb))
        s.load(dom.tiles_touched(s))
        a = _continuous(dom, s, 0, seq, gc, land)
        b = _continuous(dom, s, 1, seq, gc, land)
        out[key] = (1.0 - w) * a + w * b
        del s
        stage(key.lower(), t)
    veg = cp.where(land, cp.nan_to_num(out["VEG"], nan=0.0) * 100.0, 0.0)        # fraction -> percent
    alb_land = out["ALBDO"]
    alb = cp.where(land, cp.where(cp.isnan(alb_land), 8.0, alb_land), 6.0)          # percent; fill 8 (TBL)

    t = time.perf_counter()
    ms = src("maxsnowalb")
    ms.load(dom.tiles_touched(ms))
    mxs = _continuous(dom, ms, 0, SEQ_MONTHLY, False, land)
    mxs = cp.where(land, cp.nan_to_num(mxs, nan=0.0) / 100.0, 0.0)
    del ms
    stage("maxsnowalb", t)

    t = time.perf_counter()
    st = src("soiltemp")
    st.load(dom.tiles_touched(st))
    tg = _continuous(dom, st, 0, SEQ_TG, False, land)
    tg = cp.where(land, cp.nan_to_num(tg, nan=0.0), 0.0)
    tg = cp.where(~land & (tg < 180.0), 273.15, tg)
    del st
    stage("soiltemp", t)

    sfcr = cp.where(land, cp.asarray(VGZ0TBL_IGBP)[cp.clip(vg, 1, 20) - 1]
                    + cp.where(vg == IGBP_SNOW, 0.0, Z0LAND).astype(cp.float32), 0.0)
    ii, jj = cp.meshgrid(cp.arange(dom.ie, dtype=cp.float64), cp.arange(dom.je, dtype=cp.float64))
    glat, glon = dom.latlon(cp, ii, jj)
    fields = {
        "HGT_SFC": hgt, "LAND": land.astype(cp.float32), "VGTYP": vg, "SOTYP": sl, "SFCR": sfcr,
        "VEG": veg, "ALBDO": alb, "ALBASE": cp.where(land, alb / 100.0, 0.06), "MXSNAL": mxs, "TG": tg,
        "SLOPE": land.astype(cp.float32), "EPSR": cp.where(land, 1.0, 0.97),
        "STDH": cp.full((dom.je, dom.ie), -1.0e6), "GLAT": glat, "GLON": glon,
    }
    res = {k: cp.asnumpy(cp.ascontiguousarray(v[crop])).astype(np.float32) for k, v in fields.items()}
    rec["total_s"] = round(time.perf_counter() - t0, 2)
    rec["land_frac"] = round(float(res["LAND"].mean()), 4)
    return res, rec


def run(ctx):
    """Done-rule entry: build the static of ctx.cfg's grid, valid at ctx.start (default the proof case),
    into ctx.tables['static_build'] (host arrays and the build record). Not an init-chain stage: runs write
    the file once (main) and the GFS start reads it through the configfile's static_npz."""
    valid = getattr(ctx, "start", None) or datetime(2025, 3, 14, 12)
    res, rec = build(ctx.cfg, valid)
    ctx.tables["static_build"] = {"fields": res, "record": rec}
    return rec


def write(res, rec, out: Path, cfg):
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(rec, im=int(cfg.im), jm=int(cfg.jm), tph0d=cfg.tph0d, tlm0d=cfg.tlm0d, wbd=cfg.wbd, sbd=cfg.sbd,
                dlmd=cfg.dlmd, dphd=cfg.dphd, unit=UNIT)
    import json
    np.savez_compressed(out, **res, META=np.array(json.dumps(meta)))
    prov = {"static": out.name, "unit": UNIT, "grid": meta["grid"], "valid": meta["valid"],
            "centre": [cfg.tph0d, cfg.tlm0d], "datasets": meta["datasets"],
            "licence_copied_from": meta["licence_copied_from"],
            "sea_ice": "not static: analysis-time state from the init source"}
    out.with_name(out.name + ".PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n")
    return out


def main(argv=None):
    from namgpu import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None, help="configfile (default parm/nam_configfile)")
    ap.add_argument("--date", required=True, help="valid date YYYY-MM-DD[THH]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--geog", default=None)
    ap.add_argument("--dataset", action="append", default=[], metavar="ROLE=NAME",
                    help=f"name a role's dataset (roles {sorted(DATASETS)}); a held dataset is refused")
    ap.add_argument("--soil", default=None, choices=sorted(SOIL_SETS),
                    help="soil set (default NAMGPU_GEOG_SOIL, else permissive)")
    a = ap.parse_args(argv)
    over = dict(x.split("=", 1) for x in a.dataset)
    cfg = config.load(a.config)
    valid = datetime.fromisoformat(a.date)
    res, rec = build(cfg, valid, a.geog, datasets=over, soil=a.soil)
    out = write(res, rec, a.out if a.out.is_absolute() else __import__("namgpu.paths", fromlist=["x"]).work_root() / a.out, cfg)
    print(f"{UNIT}: wrote {out} ({rec['grid'][0]}x{rec['grid'][1]}, land {rec['land_frac']}, {rec['total_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
