"""The awphys record table: one row per record of NOAA's nam.tHHz.awphysFF.tm00.grib2, in NOAA's order.

`namgpu/products/awphys.tsv` is the table. `woof-nam products` turns it into the field-set document the GRIB2
writer reads (gpuwm's Rust exporter, `rw_rrfs_upp --profile-json`), so adding a record is a table row, never a
new code path. Columns (tab-separated, `-` = empty):

  rec      NAM's record number in awphys06 (`8.1`, `8.2`: the two fields of one NCEP message). Rows are written
           in table order, which is NAM's order.
  name     wgrib2's short name (TMP, MSLET, ...): what the inventory line shows.
  dcp      discipline.category.parameter (WMO code table 4.2; >= 192 are NCEP local entries).
  level    first fixed surface, `type:value` (code table 4.5; value in the type's SI unit as the file encodes
           it: Pa for 100 and 108, m for 103 and 106, 0 for surfaces with no value).
  level2   second fixed surface for layers, `type:value`, or `-`.
  stat     `-` (product template 4.0), or acc / max / min / ave (template 4.8, code table 4.10: 1 / 2 / 3 / 0).
  window   `-` (the writer's rule: an accumulation from the start, an extreme over the last output hour) or
           a named rule the writer knows (`last-hour-label`). NAM's own buckets are listed in `note`.
  source   where the plane comes from: a writer source name (`TMP_ON_SPEC_HGT_LVL_ABOVE_GRND_2m`, the names of
           rw-grib2export's rrfs_upp.rs `Ctx::plane`), `carrier:NAME` / `carrier:NAME[k]` (a field of the
           grid-218 hand-off frame as it is; k = model level from 0), or `-` (not written yet: listed in the
           run index as left out).
  bits     bits per value for this record (complex packing), or `-` for the run's `--bits`. Used only for a
           record NOAA's precision table (below) does not cover.
  note     free text (method, NAM's bucket at f06, why a row is left out).

`namgpu/products/awphys-noaa.tsv` (core-owned, generated from NOAA's archived awphys06 and its .idx, one row per
table row, same order) carries what NOAA's own file says about each record and is not ours to choose:
  wgrib2_name, wgrib2_level   the record's text in NOAA's .idx (= `wgrib2 -s`); our .idx writes the same text
  D, E                         NOAA's decimal and binary scale factors for the record: the writer packs at that
                               precision (complex packing, bit width from the data), so each record carries the
                               same precision as NOAA's (the 2026-11-05 01:47Z coordinator decision)
  S1, S2                       the scale factors NOAA writes the first and second fixed surface with (a 0.1 m soil
                               depth is 10 x 10^-2 in NOAA's file), so section 4 carries NOAA's bytes; `-` = none

    python -m namgpu.products.spec skeleton <archived awphys06.grib2>      # a table of every record, sources `-`
    python -m namgpu.products.spec check                                    # parse the table, count rows
    python -m namgpu.products.spec noaa <archived awphys06.grib2> <its .idx>  # regenerate awphys-noaa.tsv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TABLE = Path(__file__).with_name("awphys.tsv")
NOAA = Path(__file__).with_name("awphys-noaa.tsv")
NOAA_COLUMNS = ("rec", "name", "dcp", "level", "level2", "stat", "wgrib2_name", "wgrib2_level", "D", "E", "S1", "S2")
COLUMNS = ("rec", "name", "dcp", "level", "level2", "stat", "window", "source", "bits", "note")
STAT_CODES = {"ave": 0, "acc": 1, "max": 2, "min": 3}
STAT_NAMES = {v: k for k, v in STAT_CODES.items()}
# NCEP local level types (code table 4.5, 192-254 and the entries NCEP defines locally)
LOCAL_LEVELS = {200, 204, 206, 207, 211, 212, 213, 214, 215, 220, 222, 223, 224, 232, 233, 234, 242, 243, 244, 245,
                246, 247, 248, 249, 251, 252, 253, 254}


def _surface(text: str):
    if text in ("", "-"):
        return None, None
    t, _, v = text.partition(":")
    return int(t), float(v or 0.0)


def read(path: Path = TABLE) -> list[dict]:
    """The table's rows, in order (comments and blank lines skipped). A malformed row stops the run: a misread
    level or parameter number would write a mislabelled record that a downstream ingest takes for NAM's."""
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) < len(COLUMNS) - 1:
            raise SystemExit(f"{path}:{n}: {len(cells)} columns, the table has {len(COLUMNS)} ({', '.join(COLUMNS)})")
        cells += [""] * (len(COLUMNS) - len(cells))
        r = dict(zip(COLUMNS, (c.strip() for c in cells)))
        if r["rec"] == "rec":
            continue                                       # header
        try:
            d, c, p = (int(x) for x in r["dcp"].split("."))
            t1, v1 = _surface(r["level"])
            t2, v2 = _surface(r["level2"])
        except ValueError as e:
            raise SystemExit(f"{path}:{n}: {e} in {line!r}") from None
        if t1 is None:
            raise SystemExit(f"{path}:{n}: no first fixed surface")
        if r["stat"] not in ("-", "") and r["stat"] not in STAT_CODES:
            raise SystemExit(f"{path}:{n}: stat {r['stat']!r} is not -, {', '.join(STAT_CODES)}")
        r.update(discipline=d, category=c, number=p, t1=t1, v1=v1, t2=t2, v2=v2, line=n)
        rows.append(r)
    return rows


def field(r: dict) -> dict:
    """One table row as a field of the writer's field-set document (rw-grib2export rrfs_upp.rs `Field`)."""
    stat = STAT_CODES.get(r["stat"])
    return {
        "file": "awphys", "datset": "AWPHYS", "shortname": r["source"], "pname": r["name"],
        "discipline": r["discipline"], "category": r["category"], "number": r["number"],
        "ncep_local": r["category"] >= 192 or r["number"] >= 192 or r["t1"] in LOCAL_LEVELS,
        "level_type": r["t1"], "levels": [r["v1"]],
        # no second surface: type 255 with scale factor and value 0, the bytes NCEP's post (g2lib) writes; the
        # writer's own default (255 and an all-missing scale/value) differs from NOAA's section 4 on every such record
        "level2_type": 255 if r["t2"] is None else r["t2"], "levels2": [0.0] if r["t2"] is None else [r["v2"]],
        "template": 8 if stat is not None else 0, "stat": stat, "aerosol": None, "upp_scale": 0.0,
        "window": None if r["window"] in ("", "-") else r["window"],
        "bits": None if r["bits"] in ("", "-") else int(r["bits"]),
        **({"decimal_scale": r["noaa_D"], "binary_scale": r["noaa_E"]} if r.get("noaa_D") is not None else {}),
        **({"level_scale": r["noaa_S1"]} if r.get("noaa_S1") is not None else {}),
        **({"level2_scale": r["noaa_S2"]} if r.get("noaa_S2") is not None and r["t2"] is not None else {}),
    }


def read_noaa(path: Path = NOAA) -> list[dict]:
    """NOAA's per-record facts (inventory text, precision), one row per table row, in table order."""
    if not Path(path).is_file():
        return []
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if cells[0] == "rec":
            continue
        if len(cells) != len(NOAA_COLUMNS):
            raise SystemExit(f"{path}:{n}: {len(cells)} columns, expected {len(NOAA_COLUMNS)}")
        r = dict(zip(NOAA_COLUMNS, cells))
        r["D"], r["E"] = int(r["D"]), int(r["E"])
        r["S1"], r["S2"] = (None if r[k] == "-" else int(r[k]) for k in ("S1", "S2"))
        out.append(r)
    return out


def attach_noaa(rows: list[dict], noaa: list[dict] | None = None) -> list[dict]:
    """Put NOAA's precision and inventory text on each table row. The two files are matched row by row and
    must agree on the record's identity: a shifted row would pack or label one record as another."""
    noaa = read_noaa() if noaa is None else noaa
    if not noaa:
        return rows
    if len(noaa) != len(rows):
        raise SystemExit(f"{NOAA}: {len(noaa)} rows, the record table has {len(rows)}: regenerate it "
                         f"(python -m namgpu.products.spec noaa ...)")
    for r, x in zip(rows, noaa):
        if (r["rec"], r["dcp"], r["level"], r["level2"], r["stat"] or "-") != \
                (x["rec"], x["dcp"], x["level"], x["level2"], x["stat"]):
            raise SystemExit(f"{NOAA}: row rec {x['rec']} ({x['dcp']} {x['level']}) does not match the table's "
                             f"rec {r['rec']} ({r['dcp']} {r['level']})")
        r["noaa_D"], r["noaa_E"] = x["D"], x["E"]
        r["noaa_S1"], r["noaa_S2"] = x["S1"], x["S2"]
        r["noaa_name"], r["noaa_level"] = x["wgrib2_name"], x["wgrib2_level"]
    return rows


def profile(rows: list[dict] | None = None, name: str = "nam-awphys") -> tuple[dict, list[dict]]:
    """(the writer's field-set document, the rows left out because no source is mapped yet)."""
    rows = attach_noaa(read() if rows is None else rows)
    use = [r for r in rows if r["source"] not in ("", "-")]
    left = [r for r in rows if r["source"] in ("", "-")]
    # NCEP grid 218 messages carry resolution flags 0x38: increments given, winds grid-relative
    # isobaric records made as NAM's own post makes them (NCEP UPP MDL2P, NMM): rw_rrfs_upp's upp-nmm rule
    return {"profile": name, "resolution_flags": 0x38, "isobaric_rule": "upp-nmm",
            "fields": [field(r) for r in use]}, left


def write_profile(dst: Path, rows: list[dict] | None = None) -> tuple[Path, list[dict]]:
    doc, left = profile(rows)
    dst = Path(dst)
    dst.write_text(json.dumps(doc, indent=1))
    return dst, left


def inventory_key(r: dict) -> tuple:
    """What the inventory line of a record is made of: parameter, both surfaces, template and statistic."""
    stat = STAT_CODES.get(r["stat"]) if "stat" in r and isinstance(r["stat"], str) else r.get("stat")
    return (f"{r['discipline']}.{r['category']}.{r['number']}", r["t1"], float(r["v1"] or 0.0),
            r["t2"] if r["t2"] not in (None, 255) else None,
            None if r["t2"] in (None, 255) else float(r["v2"] or 0.0), stat)


# ------------------------------------------------------------------------------------------------ skeleton

def skeleton(archive: Path) -> str:
    """A table of every record of an archived awphys file, in its order, with every source `-`."""
    from namgpu.io import products as P  # noqa: PLC0415
    recs = P.scan(Path(archive))
    out = ["# WOOF-NAM awphys record table (namgpu/products/spec.py documents the columns). Order = NAM's.",
           "\t".join(COLUMNS)]
    for r in recs:
        rec = f"{r['msg']}.{r['sub']}" if r.get("sub") else str(r["msg"])
        name = P.NAMES.get(r["key"], f"var{r['key'].replace('.', '_')}")

        def surf(t, v):
            if t in (None, 255):
                return "-"
            return f"{t}:{v:g}" if v is not None else f"{t}:0"
        stat = STAT_NAMES.get(r.get("stat"), "-") if r.get("pdt") == 8 else "-"
        note = ""
        if r.get("pdt") == 8:
            fh = P._hours(r.get("unit", 1), r.get("ftime", 0))
            end = fh + P._hours(r.get("range_unit", 1), r.get("range_len", 0))
            note = f"NAM window at this file: {fh:g}-{end:g} h"
        out.append("\t".join([rec, name, r["key"], surf(r.get("t1"), r.get("v1")), surf(r.get("t2"), r.get("v2")),
                              stat, "-", "-", "-", note]))
    return "\n".join(out) + "\n"


def noaa_table(archive: Path, idx: Path) -> str:
    """awphys-noaa.tsv from NOAA's archived awphys06 and its .idx (the .idx is wgrib2 -s of the file)."""
    from namgpu.io import products as P  # noqa: PLC0415
    recs = P.scan(Path(archive))
    lines = [ln for ln in Path(idx).read_text().splitlines() if ln.strip()]
    rows = read()
    if not (len(recs) == len(lines) == len(rows)):
        raise SystemExit(f"{archive}: {len(recs)} records, {idx}: {len(lines)} lines, table: {len(rows)} rows")
    out = ["# NOAA's own facts per awphys record (spec.py documents it). Generated: python -m namgpu.products.spec "
           f"noaa {Path(archive).name} {Path(idx).name}", "\t".join(NOAA_COLUMNS)]
    for r, rec, ln in zip(rows, recs, lines):
        k = (f"{rec['discipline']}.{rec['cat']}.{rec['param']}", rec["t1"], float(rec.get("v1") or 0.0),
             rec["t2"] if rec["t2"] != 255 else None, None if rec["t2"] in (None, 255) else float(rec.get("v2") or 0.0),
             rec.get("stat") if rec.get("pdt") == 8 else None)
        if k != inventory_key(r):
            raise SystemExit(f"table rec {r['rec']} {inventory_key(r)} is not NOAA's record {ln.split(':')[0]} {k}")
        parts = ln.split(":")
        out.append("\t".join([r["rec"], r["name"], r["dcp"], r["level"], r["level2"], r["stat"] or "-",
                              parts[3], parts[4], str(rec["D"]), str(rec["E"]),
                              "-" if rec.get("s1") in (None, 255) else str(rec["s1"]),
                              "-" if rec.get("s2") in (None, 255) else str(rec["s2"])]))
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["noaa"] and len(argv) == 3:
        NOAA.write_text(noaa_table(Path(argv[1]), Path(argv[2])))
        print(f"{NOAA}: written")
        return 0
    if argv[:1] == ["skeleton"] and len(argv) == 2:
        sys.stdout.write(skeleton(Path(argv[1])))
        return 0
    if argv[:1] == ["check"]:
        rows = read()
        doc, left = profile(rows)
        print(f"{TABLE}: {len(rows)} records, {len(doc['fields'])} with a source, {len(left)} left out")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
