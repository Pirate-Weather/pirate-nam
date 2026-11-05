"""Record tables of the isobaric init sources: which GRIB2 records each source row reads, and the generic
conversions that turn them into the fields the isobaric start (gfs_start.py) and boundaries (gfs_bocos.py) use.

Adding a source is a table here plus a row in sources.SOURCES (the arbitrary acceptance test): the conversions
are named physics operations (scale, dewpoint -> q, q -> RH, conservative soil-layer remap, ratio, constant,
donor field from another source's file), never per-model code.

A record table has:
  pl            name -> (discipline, category, parameter) on isobaric levels (level type 100, Pa). Names are
                gfs_start.PL_ORDER slots; a slot a source does not publish stays 0 (the condensate species of a
                source without hydrometeors: init-gfs-vinterp's do_clouds gate then sets CW = 0).
  pl_need       slots every level used must carry (the level set is their intersection).
  pl_derive     slot -> (op, args): computed on the device from the decoded slots, before the decode sanitiser.
  sfc           name -> (d, c, p, level_type, level_value, second_level_value) single-level records.
  sfc_derive    ordered [(out, op, args)]: computed from decoded sfc fields (and earlier outputs).
  sfc_columns   the sfc records the vertical interpolation needs (boundary times decode only these, plus what
                their sfc_derive needs).
  derive_columns  the sfc_derive outputs a boundary time needs.
  donors        name -> {"file": template under the case dir, "spec": sfc selector}: a field the source does not
                publish, taken from another source's file on its own grid (e.g. ECMWF open data has no sea-ice
                concentration; the same cycle's GFS f000 ICEC is the donor, as gpuwm's fix/ifs-open-data-sea-ice).
  lat_crop      "south": keep source rows north of the domain's south edge (the GFS path as it has always been);
                "both": keep only the domain's latitude band plus the margin (0.1 deg global fields are 6.5 M points
                per record; the band keeps the device copy small).

Operations (all elementwise, float32, on the device):
  scale(src, factor)                 out = src * factor
  const(value)                       out = value everywhere
  ratio(num, den, floor)             out = num / max(den, floor)
  q_from_dewpoint(td, p)             specific humidity from dewpoint (K) and pressure (Pa):
                                     e = es(td), q = eps e / (p - (1 - eps) e)
  rh_from_q(q, t, p)                 RH (percent) that init-gfs-vinterp's own rh_to_mxrat (liquid branch,
                                     kernels/init_gfs_vinterp.cu init_gfs_vinterp_rhq) turns back into q:
                                     w = q / (1 - q), RH = 100 w / qs(t, p). So the source's q is what the model gets
                                     (inside the kernel's 0.001..100 % clamp), whatever the source's own RH definition.
  layer_remap(srcs, src_layers, tgt_layers)
                                     conservative overlap weights, target layer mean = sum_k w_k src_k
                                     (gpuwm ingest/soil_contract.py conservative_overlap_weights, same rule).
"""
from __future__ import annotations

import numpy as np

EPS = 0.622
SVP1, SVP2, SVP3, SVPT0 = 0.6112, 17.67, 29.65, 273.15     # init_gfs_vinterp_rhq's constants (hPa formula)
G0 = 9.80665                                               # WMO standard gravity: IFS z (m2 s-2) -> gpm

# GFS 0.25 deg pgrb2 (NCEP). Byte for byte the tables gfs_start has always used.
GFS = {
    "pl": {"HGT": (0, 3, 5), "TMP": (0, 0, 0), "SPFH": (0, 1, 0), "RH": (0, 1, 1), "UGRD": (0, 2, 2),
           "VGRD": (0, 2, 3), "CLMR": (0, 1, 22), "ICMR": (0, 1, 23), "RWMR": (0, 1, 24), "SNMR": (0, 1, 25)},
    "pl_need": ("HGT", "TMP", "RH", "UGRD", "VGRD"),
    "pl_derive": {},
    "sfc": {
        "PRES": (0, 3, 0, "1", "0", None), "HGT": (0, 3, 5, "1", "0", None),
        "TSK": (0, 0, 0, "1", "0", None),
        "T2": (0, 0, 0, "103", "2", None), "RH2": (0, 1, 1, "103", "2", None),
        "Q2": (0, 1, 0, "103", "2", None),
        "U10": (0, 2, 2, "103", "10", None), "V10": (0, 2, 3, "103", "10", None),
        "TSOIL1": (2, 0, 2, "106", "0", "0.1"), "TSOIL2": (2, 0, 2, "106", "0.1", "0.4"),
        "TSOIL3": (2, 0, 2, "106", "0.4", "1"), "TSOIL4": (2, 0, 2, "106", "1", "2"),
        "SOILW1": (2, 0, 192, "106", "0", "0.1"), "SOILW2": (2, 0, 192, "106", "0.1", "0.4"),
        "SOILW3": (2, 0, 192, "106", "0.4", "1"), "SOILW4": (2, 0, 192, "106", "1", "2"),
        "CNWAT": (2, 0, 196, "1", "0", None), "WEASD": (0, 1, 13, "1", "0", None),
        "SNOD": (0, 1, 11, "1", "0", None), "ICEC": (10, 2, 0, "1", "0", None),
        "LANDSRC": (2, 0, 0, "1", "0", None),
    },
    "sfc_derive": [],
    # the GFS forecast files (f006 ...) carry no CNWAT, so boundary times ask only for these (2026-11-04)
    "sfc_columns": ("PRES", "HGT", "T2", "RH2", "U10", "V10"),
    "derive_columns": (),
    "donors": {},
    "lat_crop": "south",
}

# IFS layers sot/vsw 1..4 (levtype sol, GRIB2 level type 151 levels 0-1 .. 3-4): 0-7, 7-28, 28-100, 100-289 cm
# (IFS documentation, Part IV physical processes, HTESSEL soil); Noah's four: 0-10, 10-40, 40-100, 100-200 cm.
IFS_SOIL_LAYERS = ((0.0, 0.07), (0.07, 0.28), (0.28, 1.0), (1.0, 2.89))
NOAH_SOIL_LAYERS = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))

# ECMWF open data IFS HRES (0.1 deg pl + sfc pair, or 0.25 deg single file): parameter numbers as the files
# carry them (grib2_inventory of 2026-11-03 12Z f000: sp 0-3-0, z 0-3-4, skt 0-0-17, 2t 0-0-0/103, 2d 0-0-6/103,
# 10u/10v 0-2-2/0-2-3 at 103 10 m, lsm 2-0-0, sd 0-1-254 (m of water equivalent), rsn 0-1-61 (kg m-3),
# sot 2-3-18 and vsw 2-0-25 on level type 151 layers 0-1 .. 3-4).
IFS = {
    "pl": {"HGT": (0, 3, 5), "TMP": (0, 0, 0), "SPFH": (0, 1, 0), "UGRD": (0, 2, 2), "VGRD": (0, 2, 3)},
    "pl_need": ("HGT", "TMP", "SPFH", "UGRD", "VGRD"),
    "pl_derive": {"RH": ("rh_from_q", ("SPFH", "TMP", "@p"))},
    "sfc": {
        "PRES": (0, 3, 0, "1", "0", None), "Z": (0, 3, 4, "1", "0", None),
        "TSK": (0, 0, 17, "1", "0", None),
        "T2": (0, 0, 0, "103", "2", None), "D2": (0, 0, 6, "103", "2", None),
        "U10": (0, 2, 2, "103", "10", None), "V10": (0, 2, 3, "103", "10", None),
        "SOT1": (2, 3, 18, "151", "0", "1"), "SOT2": (2, 3, 18, "151", "1", "2"),
        "SOT3": (2, 3, 18, "151", "2", "3"), "SOT4": (2, 3, 18, "151", "3", "4"),
        "VSW1": (2, 0, 25, "151", "0", "1"), "VSW2": (2, 0, 25, "151", "1", "2"),
        "VSW3": (2, 0, 25, "151", "2", "3"), "VSW4": (2, 0, 25, "151", "3", "4"),
        "SD": (0, 1, 254, "1", "0", None), "RSN": (0, 1, 61, "1", "0", None),
        "LANDSRC": (2, 0, 0, "1", "0", None),
    },
    "sfc_derive": [
        ("HGT", "scale", ("Z", 1.0 / G0)),
        ("Q2", "q_from_dewpoint", ("D2", "PRES")),
        ("RH2", "rh_from_q", ("Q2", "T2", "PRES")),
        (("TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4"), "layer_remap",
         (("SOT1", "SOT2", "SOT3", "SOT4"), IFS_SOIL_LAYERS, NOAH_SOIL_LAYERS)),
        (("SOILW1", "SOILW2", "SOILW3", "SOILW4"), "layer_remap",
         (("VSW1", "VSW2", "VSW3", "VSW4"), IFS_SOIL_LAYERS, NOAH_SOIL_LAYERS)),
        ("WEASD", "scale", ("SD", 1000.0)),                  # m of water -> kg m-2
        ("SNOD", "ratio", ("WEASD", "RSN", 50.0)),           # physical depth (m) = SWE / snow density
        ("CNWAT", "const", (0.0,)),                          # IFS open data publishes no canopy water
    ],
    "sfc_columns": ("PRES", "Z", "T2", "D2", "U10", "V10"),
    "derive_columns": ("HGT", "Q2", "RH2"),
    "donors": {"ICEC": {"file": "raw/ifs/gfs.t{cyc:02d}z.icec.f000.grib2", "spec": (10, 2, 0, "1", "0", None)}},
    # Snow: ECMWF open data carries sd/rsn only in later files (the 2025-03-14 12Z 0.25 deg file has neither). A file
    # without them takes snow water and depth from the same cycle's GFS f000 instead (receipt: donors_used).
    "sfc_optional": ("SD", "RSN"),
    "donor_fallback": {
        "WEASD": {"file": "raw/ifs/gfs.t{cyc:02d}z.weasd.f000.grib2", "spec": (0, 1, 13, "1", "0", None)},
        "SNOD": {"file": "raw/ifs/gfs.t{cyc:02d}z.snod.f000.grib2", "spec": (0, 1, 11, "1", "0", None)},
    },
    "lat_crop": "both",
}

TABLES = {"gfs_0p25": GFS, "ifs": IFS}


def table(row: dict) -> dict:
    """The record table of a SOURCES row (GFS when the row names none: the rows that predate the tables)."""
    return TABLES[row.get("records", "gfs_0p25")]


def conservative_overlap_weights(source, target) -> np.ndarray:
    """Target-by-source overlap weights for layer-mean remapping (gpuwm ingest/soil_contract.py, same rule)."""
    w = np.zeros((len(target), len(source)), dtype=np.float64)
    for ti, (t0, t1) in enumerate(target):
        for si, (s0, s1) in enumerate(source):
            w[ti, si] = max(0.0, min(t1, s1) - max(t0, s0)) / (t1 - t0)
        if abs(float(w[ti].sum()) - 1.0) > 2e-15:
            raise ValueError(f"soil remap: target layer {ti} {target[ti]} is not covered by the source layers {source}")
    return w


def _es_hpa(t):
    import cupy as cp
    return cp.float32(SVP1 * 10.0) * cp.exp(cp.float32(SVP2) * (t - cp.float32(SVPT0)) / (t - cp.float32(SVP3)))


def q_from_dewpoint(td, p):
    import cupy as cp
    e = _es_hpa(td) * cp.float32(100.0)
    return (cp.float32(EPS) * e / (p - cp.float32(1.0 - EPS) * e)).astype(cp.float32)


def rh_from_q(q, t, p):
    import cupy as cp
    es = _es_hpa(t)
    qs = cp.float32(EPS) * es / (p / cp.float32(100.0) - es)
    q = cp.maximum(q, 0.0)
    w = q / (cp.float32(1.0) - q)
    return (cp.float32(100.0) * w / qs).astype(cp.float32)


def derive_sfc(rec: dict, sfc: dict, only=None) -> None:
    """Apply rec['sfc_derive'] in order, in place on sfc (dict of cupy [ny, nx]). only: output names wanted."""
    import cupy as cp
    for out, op, args in rec["sfc_derive"]:
        outs = out if isinstance(out, tuple) else (out,)
        if only is not None and not any(o in only for o in outs):
            continue
        # an optional record the file does not carry: its outputs come from rec["donor_fallback"] instead
        if all(o in rec.get("donor_fallback", {}) for o in outs) and \
                any(isinstance(a, str) and a not in sfc for a in args):
            continue
        if op == "scale":
            sfc[out] = (sfc[args[0]] * cp.float32(args[1])).astype(cp.float32)
        elif op == "const":
            ref = next(iter(sfc.values()))
            sfc[out] = cp.full(ref.shape, args[0], dtype=cp.float32)
        elif op == "ratio":
            sfc[out] = (sfc[args[0]] / cp.maximum(sfc[args[1]], cp.float32(args[2]))).astype(cp.float32)
        elif op == "q_from_dewpoint":
            sfc[out] = q_from_dewpoint(sfc[args[0]], sfc[args[1]])
        elif op == "rh_from_q":
            sfc[out] = rh_from_q(sfc[args[0]], sfc[args[1]], sfc[args[2]])
        elif op == "layer_remap":
            srcs, sl, tl = args
            w = conservative_overlap_weights(sl, tl)
            for ti, o in enumerate(outs):
                acc = cp.zeros_like(sfc[srcs[0]])
                for si, s in enumerate(srcs):
                    if w[ti, si]:
                        acc += cp.float32(w[ti, si]) * sfc[s]
                sfc[o] = acc
        else:
            raise ValueError(f"source_records: unknown sfc_derive op {op!r}")


def derive_pl(rec: dict, pl, plev, order) -> None:
    """Apply rec['pl_derive'] in place on pl [nslot, nlev, ny, nx]; '@p' is the level pressure (Pa)."""
    import cupy as cp
    for slot, (op, args) in rec["pl_derive"].items():
        if op != "rh_from_q":
            raise ValueError(f"source_records: unknown pl_derive op {op!r}")
        q, t = pl[order.index(args[0])], pl[order.index(args[1])]
        p = cp.asarray(np.asarray(plev, np.float32))[:, None, None]
        pl[order.index(slot)] = rh_from_q(q, t, p)
