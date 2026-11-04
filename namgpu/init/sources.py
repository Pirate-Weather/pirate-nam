"""Init sources: one table row per analysis source, staged into the state the init chain reads.

This is the NPS replacement's front half. NAM was retired on 2026-11-03 12Z, so (CONTRACT "Init after
NAM retirement"):
  native  the archived NAM native B-grid analysis (bgrd3d00), valid only for cycles before the
          retirement. Same grid and levels as the model: no interpolation.
  gfs     GFS 0.25 deg analysis (pgrb2.0p25 f000) for any date, interpolated onto the rotated B grid
          and the NMMB hybrid levels by the init-gfs-* units (nemsinterp port), gfs_start.py.
  hrrr, rrfs  table rows for the same isobaric path. They are refused by name today: their files are
          on Lambert-conformal grids (GRIB2 GDT 30) and init-grib2-decode places only GDT 0 and 32769
          (a Lambert grid read as lat-lon puts every point in the wrong cell). Not done.
Adding a source is a row in SOURCES (file pattern, kind, grid), never a new code path.

Entry: stage(ctx) is what the chain stage init-grib2-decode runs when it is called with no file
(init_grib2_decode.run(ctx)). It reads ctx.init_source or ctx.cfg.raw["init_source"] (default
"native") and fills:
  ctx.st      TMP SPFH UGRD VGRD CLMR CICE RWMR SNMR PRES [LM, JM, IM] (top first), PSFC, HGT_M,
              SOILHGT, FIS [JM, IM]; for the native source also TCOND, FICE, FRAIN, RIME, TKE
              [LM, JM, IM] (the analysis' own condensate, fractions and TKE; init-native-atm reads them).
  ctx.st      PSHLTR, TSHLTR (2 m theta), QSHLTR, U10, V10 (grid-relative, H points), TH10, Q10 [JM, IM]
              from the source's 2 m / 10 m analysis (shelter_state), as NMMB reads them from its input.
  ctx.tables  raw_sfc (init-native-sfc input), soil_hydraulics (Noah STAS SOILPARM b, psi_s, theta_s),
              grid_source (receipt: source row, file, level order found).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNIT = "init-sources"
G_NMMB = np.float32(9.8060226)        # module_CONSTANTS G (FIS = HGT * G)
NCLASS = 30                           # soil_hydraulics table length (init-native-sfc)

# One row per source. "file" is relative to the case directory; {cyc} is the cycle hour.
SOURCES = {
    "native": {"kind": "nam_native", "file": "raw/nam/bgrd3d00.grib2", "gdt": 32769,
               "valid_before": "2026-11-03T12"},
    "gfs":    {"kind": "isobaric", "file": "raw/gfs/gfs.t{cyc:02d}z.pgrb2.0p25.f000", "gdt": 0,
               "grid": "gfs_0p25",
               "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{cyc:02d}/atmos/"
                      "gfs.t{cyc:02d}z.pgrb2.0p25.f000"},
    "hrrr":   {"kind": "isobaric", "file": "raw/hrrr/hrrr.t{cyc:02d}z.wrfprsf00.grib2", "gdt": 30,
               "grid": "hrrr_lambert"},
    "rrfs":   {"kind": "isobaric", "file": "raw/rrfs/rrfs.t{cyc:02d}z.prslev.f000.conus.grib2",
               "gdt": 30, "grid": "rrfs_lambert"},
}

# NAM native analysis records (UPP output of the NMMB analysis, GDT 32769, UPP resolution flags 56:
# winds grid-relative, grib2_module.f:1301), (discipline, category, parameter) on hybrid levels.
NATIVE_3D = {
    "PRES": (0, 3, 0), "HGT3D": (0, 3, 5), "TMP": (0, 0, 0), "SPFH": (0, 1, 0),
    "UGRD": (0, 2, 2), "VGRD": (0, 2, 3), "CLMR": (0, 1, 22), "CICE": (0, 6, 0),
    "RWMR": (0, 1, 24), "SNMR": (0, 1, 25),
    # NCEP local tables: 0-6-195 TCOND, 0-6-199 FICE, 0-1-202 FRAIN, 0-1-203 RIME, 0-19-11 TKE
    "TCOND": (0, 6, 195), "FICE": (0, 6, 199), "FRAIN": (0, 1, 202), "RIME": (0, 1, 203),
    "TKE": (0, 19, 11),
}
# Staged inputs dropped from the state once init-native-atm / init-vcoord-recover have read them.
STAGED_3D = ("PRES", "TMP", "SPFH", "UGRD", "VGRD", "CLMR", "CICE", "RWMR", "SNMR", "ZERO3")


def release_staged(ctx, keep=("PRES",)):
    """Drop the staged decode names (aliases of T/Q/U/V and of the zero species array, PRES) once the
    state is built; the model arrays stay in the state under their own names."""
    for n in STAGED_3D:
        if n not in keep:
            ctx.st.fields.pop(n, None)


# Surface records: name -> (d, c, p, level_type, level_value, second_level_value or None)
NATIVE_SFC = {
    "PSFC": (0, 3, 0, "1", "0", None),
    "HGT_SFC": (0, 3, 5, "1", "0", None),
    "TMP_SFC": (0, 0, 0, "1", "0", None),
    "TSOIL1": (2, 0, 2, "106", "0", "0.1"), "TSOIL2": (2, 0, 2, "106", "0.1", "0.4"),
    "TSOIL3": (2, 0, 2, "106", "0.4", "1"), "TSOIL4": (2, 0, 2, "106", "1", "2"),
    "SOILW1": (2, 0, 192, "106", "0", "0.1"), "SOILW2": (2, 0, 192, "106", "0.1", "0.4"),
    "SOILW3": (2, 0, 192, "106", "0.4", "1"), "SOILW4": (2, 0, 192, "106", "1", "2"),
    "TSOIL_3M": (2, 0, 2, "106", "3", None),
    "CNWAT": (2, 0, 196, "1", "0", None),
    "WEASD": (0, 1, 13, "1", "0", None),          # instantaneous (pdt 0); the pdt-8 twin is an accumulation
    "SNOD": (0, 1, 11, "1", "0", None),
    "SNOWC": (0, 1, 201, "1", "0", None),
    "SFCR": (2, 0, 1, "1", "0", None),
    "VEG": (2, 0, 4, "1", "0", None),
    "VGTYP": (2, 0, 198, "1", "0", None),
    "SOTYP": (2, 3, 0, "1", "0", None),
    "LAND": (2, 0, 0, "1", "0", None),
    "ICEC": (10, 2, 0, "1", "0", None),
    "WTMP": (10, 3, 0, "1", "0", None),
    "ALBDO": (0, 19, 1, "1", "0", None),
}


# 2 m / 10 m analysis records. NMMB reads tshltr, qshltr, pshltr, u10, v10, th10, q10 from its input file
# (module_INIT_READ_NEMSIO.F90:802-820, 2178-2217, 2589-2694); without them they start at the
# SOLVER_INITIALIZE defaults (TSHLTR 273, QSHLTR 0, U10 0; module_SOLVER_GRID_COMP.F90:1105-1118) and the
# 0 h history shows those defaults. Optional: a file without a record keeps the derived fallback.
NATIVE_SHELTER = {
    "T2M": (0, 0, 0, "103", "2", None), "Q2M": (0, 1, 0, "103", "2", None),
    "P2M": (0, 3, 0, "103", "2", None),
    "U10M": (0, 2, 2, "103", "10", None), "V10M": (0, 2, 3, "103", "10", None),
    "TH10M": (0, 0, 2, "103", "10", None), "Q10M": (0, 1, 0, "103", "10", None),
}
CAPA = 0.28589641          # R/CP, nmm/module_CONSTANTS.F90 (post T2 = TSHLTR * (PSHLTR*1e-5)**CAPA)
RD = 287.04
G = 9.8060226


def shelter_state(ctx, t2, q2, u10, v10, p2=None, th10=None, q10=None):
    """Fill the NMMB shelter / 10 m state (H points, [JM, IM]) from analysis 2 m and 10 m fields.
    t2 K (temperature), q2 kg/kg, u10/v10 grid-relative m/s at H points, p2 Pa (else PSFC reduced 2 m
    hydrostatically). TSHLTR and TH10 are potential temperatures (P0 = 1e5 Pa), as NMMB stores them."""
    import cupy as cp
    st = ctx.st
    jm, im = int(ctx.grid.jm), int(ctx.grid.im)
    psfc = st["PSFC"]
    if p2 is None:
        p2 = psfc * cp.exp(cp.float32(-2.0 * G / RD) / t2)
    st.add("PSHLTR", (jm, im))[...] = p2
    th2 = t2 * (cp.float32(1.0e5) / p2) ** cp.float32(CAPA)
    st.add("TSHLTR", (jm, im))[...] = th2
    st.add("QSHLTR", (jm, im))[...] = cp.maximum(q2, 0.0)
    st.add("U10", (jm, im))[...] = u10
    st.add("V10", (jm, im))[...] = v10
    st.add("TH10", (jm, im))[...] = th2 if th10 is None else th10
    st.add("Q10", (jm, im))[...] = cp.maximum(q2 if q10 is None else q10, 0.0)


def source_row(ctx) -> tuple[str, dict]:
    name = getattr(ctx, "init_source", None) or ctx.cfg.raw.get("init_source", "native")
    name = str(name).lower()
    if name not in SOURCES:
        raise ValueError(f"{UNIT}: init source {name!r} is not a SOURCES row ({sorted(SOURCES)})")
    return name, SOURCES[name]


def source_path(ctx, row) -> Path:
    cyc = int(ctx.start.hour) if ctx.start is not None else 0
    return Path(ctx.case_dir) / row["file"].format(cyc=cyc)


def soil_hydraulics():
    """Noah STAS SOILPARM b, psi_s (m), theta_s for init-native-sfc, padded to NCLASS (host)."""
    from namgpu.init import init_static_missing as ism
    noah = ism._noah()
    t = noah.load_tables(mminlu="MODIFIED_IGBP_MODIS_NOAH", mminsl="STAS", tbl_dir=ism.TBL_DIR)
    out = {}
    for key, arr in (("bexp", t.bb), ("psisat", t.satpsi), ("smcmax", t.maxsmc)):
        a = np.asarray(arr, np.float32).ravel()
        pad = np.full(NCLASS, a[-1] if a.size else 0.0, np.float32)
        pad[: min(a.size, NCLASS)] = a[:NCLASS]
        out[key] = pad
    return out


def _sel3(dcp, level):
    def sel(r):
        return (r["dcp"] == dcp and str(r["level_type"]) == "105"
                and int(float(r["level_value"])) == level
                and str(r["second_level_type"]) == "255")      # not the layer-mean records
    return sel


def _selsfc(spec):
    d, c, p, lt, lv, lv2 = spec

    def sel(r):
        if r["dcp"] != (d, c, p) or str(r["level_type"]) != lt:
            return False
        if abs(float(r["level_value"]) - float(lv)) > 1e-6:
            return False
        if lv2 is not None:
            try:
                if abs(float(r["second_level_value"]) - float(lv2)) > 1e-6:
                    return False
            except ValueError:
                return False
        return str(r["pdt"]) in ("0", "-")
    return sel


def stage(ctx):
    name, row = source_row(ctx)
    if row["kind"] == "nam_native":
        return stage_native(ctx, row)
    if row["gdt"] not in (0, 32769):
        raise NotImplementedError(
            f"{UNIT}: source {name!r} is on GRIB2 GDT {row['gdt']} (Lambert conformal); init-grib2-decode "
            "places only GDT 0 and 32769, so its fields would land in the wrong cells. Not done: a Lambert "
            "source interpolation for init-gfs-hinterp.")
    from namgpu.init import gfs_start
    return gfs_start.stage(ctx, name, row)


def decode_native(ctx, path, names3, sfc, lm, tmpdir, sink, top_first=None):
    """Decode hybrid fields one name at a time (top first, written through sink(name, [LM,JM,IM]))
    and the surface fields (returned). One name in flight keeps the device peak at one 3-D field."""
    import cupy as cp
    from namgpu.init import init_grib2_decode as dec

    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    # NAMGPU_ROW_WINDOW (config.row_window): the native file is on the parent grid; decode it there and keep the
    # band's rows (no interpolation: the band is the same grid).
    win = ctx.cfg.raw.get("row_window") if getattr(ctx, "cfg", None) is not None else None
    cut = (lambda x: x) if not win else (lambda x: x[..., win[0] - 1:win[1], :])
    kw = dict(missing_value=0.0, tmpdir=tmpdir,
              expect_shape=(im, int(ctx.cfg.raw["jm_full"]) if win else jm))
    p1 = plm = None
    if top_first is None:
        # UPP numbers hybrid levels from one end; NMMB wants l = 1 at the model top. Decide from PRES.
        pr = dec.run(ctx, path, {"P1": _sel3(NATIVE_3D["PRES"], 1),
                                 "PL": _sel3(NATIVE_3D["PRES"], lm)}, **kw)
        p1, plm = float(cp.mean(cut(pr["P1"]))), float(cp.mean(cut(pr["PL"])))
        top_first = p1 < plm
        del pr
    order = list(range(1, lm + 1)) if top_first else list(range(lm, 0, -1))
    for n in names3:
        got = dec.run(ctx, path, {f"{n}@{lev}": _sel3(NATIVE_3D[n], lev) for lev in range(1, lm + 1)}, **kw)
        a = cp.empty((lm, jm, im), dtype=cp.float32)
        for l0, lev in enumerate(order):
            a[l0] = cut(got.pop(f"{n}@{lev}"))
        sink(n, a)
        del a, got
    got = dec.run(ctx, path, {n: _selsfc(NATIVE_SFC[n]) for n in sfc}, **kw) if sfc else {}
    if win:
        got = {k: cp.ascontiguousarray(cut(v)) for k, v in got.items()}
    return got, {"top_first_in_file": top_first, "p_level1_mean": p1, "p_levelLM_mean": plm}


# Decode destinations. The native file is on the model grid and levels, so the dynamics fields are
# decoded straight into the state arrays init-native-atm writes (it reads and writes the same index,
# T = TMP etc., so the alias is safe). That keeps the init peak at the state plus about two 3-D
# fields, which is what lets init fit beside the 10.6 GiB state on a 16 GB card.
ALIAS_INTO = {"TMP": "T", "SPFH": "Q", "UGRD": "U", "VGRD": "V"}
SPECIES = ("CLMR", "CICE", "RWMR", "SNMR")
WARM_INTO = {"TCOND": "CW", "FICE": "F_ICE", "FRAIN": "F_RAIN", "RIME": "F_RIMEF", "TKE": "Q2"}


def decode_warm(ctx):
    """init-native-atm warm start: decode the analysis' TCOND, FICE, FRAIN, RIME, TKE straight into
    CW, F_ICE, F_RAIN, F_RIMEF, Q2 (one field in flight). Returns the names written."""
    import cupy as cp
    w = ctx.tables.get("native_warm")
    if not w or not w["names"]:
        return []
    st, lm = ctx.st, int(ctx.grid.lm)
    done = []

    def sink(n, a):
        dst = st[WARM_INTO[n]]
        if n == "TCOND":
            dst[...] = cp.maximum(a, 0.0)
        elif n in ("FICE", "FRAIN"):
            dst[...] = cp.clip(a, 0.0, 1.0)
        elif n == "RIME":
            dst[...] = cp.maximum(a, 1.0)
        else:                                   # TKE -> Q2 = 2 TKE, EPSQ2 floor 0.02
            dst[...] = cp.maximum(2.0 * a, cp.float32(0.02))
        done.append(WARM_INTO[n])

    decode_native(ctx, Path(w["file"]), w["names"], [], lm, w["tmpdir"], sink, top_first=w["top_first"])
    return done


def stage_native(ctx, row):
    import cupy as cp
    from namgpu.init import init_grib2_decode as dec

    st, g, cfg = ctx.st, ctx.grid, ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    path = source_path(ctx, row)
    if ctx.start is not None and ctx.start.strftime("%Y-%m-%dT%H") >= row["valid_before"]:
        raise ValueError(f"{UNIT}: no NAM native analysis exists for {ctx.start:%Y-%m-%d %HZ} (NAM retired "
                         "2026-11-03 12Z); use init_source gfs (or hrrr/rrfs once done)")
    if not path.is_file():
        raise FileNotFoundError(f"{UNIT}: {path} missing (run init-fetch-case)")
    inv = dec.inventory(path)
    have = {(r["dcp"], str(r["level_type"])) for r in inv}
    tmpdir = str(Path(ctx.case_dir).resolve().parents[1] / "scratch")
    Path(tmpdir).mkdir(exist_ok=True)

    names3 = ["PRES", "TMP", "SPFH", "UGRD", "VGRD"]
    if cfg.do_clouds:
        names3 += list(SPECIES)
    else:
        zero = st.add("ZERO3", (lm, jm, im))     # the species are not read when do_clouds is off
        for n in SPECIES:
            st.fields[n] = zero

    def sink(n, a):
        dst = ALIAS_INTO.get(n, n)
        st.add(dst, (lm, jm, im))[...] = a
        st.fields[n] = st[dst]

    got, order = decode_native(ctx, path, names3, list(NATIVE_SFC), lm, tmpdir, sink)
    shelter_keys = [k for k, spec in NATIVE_SHELTER.items() if any(_selsfc(spec)(r) for r in inv)]
    if shelter_keys:
        sh = dec.run(ctx, path, {k: _selsfc(NATIVE_SHELTER[k]) for k in shelter_keys},
                     missing_value=0.0, tmpdir=tmpdir, expect_shape=(im, jm))
    else:
        sh = {}
    warm = [n for n in WARM_INTO if (NATIVE_3D[n], "105") in have]
    ctx.tables["native_warm"] = {"file": str(path), "names": warm, "tmpdir": tmpdir,
                                 "top_first": order["top_first_in_file"]}

    st.add("PSFC", (jm, im))[...] = got["PSFC"]
    hgt = got["HGT_SFC"]
    st.add("HGT_M", (jm, im))[...] = hgt
    st.add("SOILHGT", (jm, im))[...] = hgt
    st.add("FIS", (jm, im))[...] = hgt * G_NMMB

    # Shelter / 10 m state from the analysis' own 2 m and 10 m records (UPP grid-relative winds, flags 56);
    # missing records fall back to the lowest model layer (l = LM, index LM-1).
    t2 = sh.get("T2M", st["T"][lm - 1])
    q2 = sh.get("Q2M", st["Q"][lm - 1])
    shelter_state(ctx, t2, q2, sh.get("U10M", st["U"][lm - 1]), sh.get("V10M", st["V"][lm - 1]),
                  p2=sh.get("P2M"), th10=sh.get("TH10M"), q10=sh.get("Q10M"))
    del sh

    raw = {
        "TSOIL": cp.ascontiguousarray(cp.stack([got[f"TSOIL{k}"] for k in range(1, 5)])),
        "SOILW": cp.ascontiguousarray(cp.stack([got[f"SOILW{k}"] for k in range(1, 5)])),
        "VGTYP": cp.rint(got["VGTYP"]).astype(cp.int32),
        "SOTYP": cp.rint(got["SOTYP"]).astype(cp.int32),
    }
    for k in ("TMP_SFC", "ICEC", "LAND", "TSOIL_3M", "SNOD", "WEASD", "ALBDO", "VEG", "SFCR", "CNWAT",
              "SNOWC", "WTMP"):
        raw[k] = got[k]
    # SNOWC is percent in UPP output (NCEP 0-1-201); init-native-sfc takes a fraction.
    raw["SNOWC"] = cp.clip(raw["SNOWC"] / cp.float32(100.0), 0.0, 1.0).astype(cp.float32)
    ctx.tables["raw_sfc"] = raw
    ctx.tables["soil_hydraulics"] = {k: cp.asarray(v) for k, v in soil_hydraulics().items()}
    ctx.tables["grid_source"] = {"source": "native", "file": str(path), **order}
    print(f"{UNIT}: native analysis {path.name}: {len(names3)} hybrid fields x {lm} levels, "
          f"{len(NATIVE_SFC)} surface fields, warm {warm}; file level 1 = "
          f"{'top' if order['top_first_in_file'] else 'bottom'}", flush=True)
    return ctx.tables["grid_source"]
