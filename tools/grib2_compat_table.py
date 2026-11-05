"""The consumer-compatibility table: every record of NOAA's NAM awphys file against a WOOF-NAM file of the same hour.

For each of NOAA's records (one row of namgpu/products/awphys.tsv) it says whether ours carries it and whether the keys
a model site's code reads are identical: what wgrib2 -s prints (name, level, time), what eccodes/pygrib/cfgrib select
on (shortName, typeOfLevel, level, stepType, stepRange, units, the GRIB2 codes) and NOAA's precision (decimal and
binary scale). Values are not compared (that is tools/grib2_value_check.py).

    python -I tools/grib2_compat_table.py NOAA.grib2 OURS.grib2 [--table awphys.tsv] [--md OUT.md] [--tsv OUT.tsv]

Needs eccodes (pip install eccodes), the decoder model sites use.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import eccodes as ec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from namgpu.products import spec  # noqa: E402

KEYS = ("shortName", "typeOfLevel", "level", "stepType", "stepRange", "units", "discipline", "parameterCategory",
        "parameterNumber", "typeOfFirstFixedSurface", "scaleFactorOfFirstFixedSurface", "scaledValueOfFirstFixedSurface",
        "typeOfSecondFixedSurface", "scaleFactorOfSecondFixedSurface", "scaledValueOfSecondFixedSurface", "productDefinitionTemplateNumber", "typeOfStatisticalProcessing",
        "lengthOfTimeRange", "decimalScaleFactor", "binaryScaleFactor", "dataRepresentationTemplateNumber",
        "gridDefinitionTemplateNumber", "shapeOfTheEarth", "DxInMetres", "resolutionAndComponentFlags", "centre",
        "subCentre", "tablesVersion", "localTablesVersion")
LABEL = {"decimalScaleFactor": "D", "binaryScaleFactor": "E", "generatingProcessIdentifier": "process id"}


def fields(path: Path) -> list[dict]:
    ec.codes_grib_multi_support_on()
    out = []
    with open(path, "rb") as f:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                break
            d = {}
            for k in KEYS + ("generatingProcessIdentifier", "scaleFactorOfFirstFixedSurface",
                             "scaleFactorOfSecondFixedSurface"):
                try:
                    d[k] = ec.codes_get(h, k)
                except Exception:  # noqa: BLE001
                    d[k] = None
            ec.codes_release(h)
            out.append(d)
    return out


def _real(d: dict, which: str):
    sf, sv = d.get(f"scaleFactorOf{which}FixedSurface"), d.get(f"scaledValueOf{which}FixedSurface")
    if sv is None or sf is None or sv in (2147483647, 4294967295) or sf in (127, 255, -127):
        return 0.0
    return round(sv / 10.0 ** sf, 6)


def ident(d: dict) -> tuple:
    """The record's identity as wgrib2 prints it: parameter, both surfaces (real values), template, statistic."""
    t2 = d["typeOfSecondFixedSurface"]
    t2 = None if t2 in (255, "255", None) else t2
    return (d["discipline"], d["parameterCategory"], d["parameterNumber"], d["typeOfFirstFixedSurface"],
            _real(d, "First"), t2, None if t2 is None else _real(d, "Second"),
            d["productDefinitionTemplateNumber"], d["typeOfStatisticalProcessing"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("noaa", type=Path)
    ap.add_argument("ours", type=Path)
    ap.add_argument("--table", type=Path, default=spec.TABLE)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--tsv", type=Path)
    ap.add_argument("--plain", action="store_true", help="notes in plain words for the package docs (no internal names)")
    a = ap.parse_args(argv)
    rows = spec.attach_noaa(spec.read(a.table))
    ref, mine = fields(a.noaa), fields(a.ours)
    mine_at = collections.defaultdict(list)
    for d in mine:
        mine_at[ident(d)].append(d)
    ref_at = collections.defaultdict(list)
    for d in ref:
        ref_at[ident(d)].append(d)
    if len(ref) != len(rows):
        print(f"note: NOAA file has {len(ref)} records, the table {len(rows)} (table rows are NOAA's awphys06)")
    seen = collections.Counter()
    out = []
    counts = collections.Counter()
    for r, d in zip(rows, ref):
        k = ident(d)
        n = seen[k]
        seen[k] += 1
        name = f"{r.get('noaa_name', r['name'])} {r.get('noaa_level', r['level'])}"
        if d["productDefinitionTemplateNumber"] == 8:
            name += f" ({d['stepRange']} h {({0: 'ave', 1: 'acc', 2: 'max', 3: 'min'}).get(d['typeOfStatisticalProcessing'], 'stat')})"
        b = mine_at[k][n] if n < len(mine_at[k]) else None
        if b is None:
            if r["source"] not in ("", "-"):
                why = (f"mapped ({r['source']}); its history field is not in this run's history (next GPU run)"
                       if not a.plain else "not written yet: needs a model output field added in the next engine build")
            elif r["note"] and "left out" in r["note"].lower():
                why = r["note"] if not a.plain else "not produced (see COMPATIBILITY.md, 'Not in the files')"
            elif r["note"]:
                why = r["note"] if not a.plain else "not written yet"
            elif r["t1"] == 100:
                why = "not mapped yet: isobaric block" if not a.plain else "not written yet (isobaric records, in progress)"
            else:
                why = "not mapped yet" if not a.plain else "not written yet"
            out.append((r["rec"], name, "no", "-", why))
            counts["missing"] += 1
            continue
        diff = [f"{LABEL.get(x, x)} NOAA {d[x]} / ours {b[x]}" for x in KEYS if d[x] != b[x]]
        same = "yes" if not diff else "no"
        counts["identical" if not diff else "present, keys differ"] += 1
        notes = "; ".join(diff)
        if d["generatingProcessIdentifier"] != b["generatingProcessIdentifier"]:
            if not a.plain:
                notes = (notes + "; " if notes else "") + "process id 84 (NAM) / 254 (WOOF), by design"
        out.append((r["rec"], name, "yes", same, notes))
    head = ("rec", "NAM record (name, level)", "present", "identical keys", "notes")
    if a.tsv:
        a.tsv.write_text("\n".join("\t".join(map(str, x)) for x in [head] + out) + "\n")
    if a.md:
        md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        md += ["| " + " | ".join(str(c).replace("|", "/") for c in x) + " |" for x in out]
        a.md.write_text("\n".join(md) + "\n")
    print(f"{len(out)} NOAA records: {counts['identical']} present with identical keys, "
          f"{counts['present, keys differ']} present with differing keys, {counts['missing']} not in ours")
    return 0


if __name__ == "__main__":
    sys.exit(main())
