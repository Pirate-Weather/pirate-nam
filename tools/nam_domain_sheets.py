"""Comparison sheets for a NAM domain run (4 km / 5 km): one sheet per product per output hour.

    python tools/nam_domain_sheets.py --run runs/nam4km-24h --label "WOOF-NAM 4 km" --start YYYY-MM-DDTHH \
        --ref12 <our 12 km run maps dir> --referee cases/2025-03-14_12z/referee/png \
        --out runs/nam4km-24h/sheets [--hours 1-24]

Panels, left to right, when each exists for that hour (a missing one is left out and the title says what
is shown): MRMS (observed composite reflectivity, reflectivity only) | this run | our 12 km run | archived
NAM 12 km (grid 218). Every panel is already a rw_wrfbatch map on the referee CONUS frame (io-maps and
cases/<case>/referee), so the sheet is a composition with gpuwm.rustwx.compose_ensemble_sheet (render law:
the Rust renderer, no matplotlib for weather fields). Host work, all cores (one thread per sheet).
Output: <out>/<product>/fNN_<product>.png and <out>/INDEX.md (what each sheet holds).

Any domain (nests anywhere): --nest <source> draws referee | WOOF-NAM | source forecast sheets from the
run's own history on the domain's own frame (namgpu.io.io_referee_sheets.nest_sheets; referees by region
from namgpu.io.referee_rows: MRMS over CONUS, NASA GPM IMERG elsewhere, the source's later analysis):

    python tools/nam_domain_sheets.py --run <run dir with wrfout_d01_*> --nest gfs --out <dir> [--hours 1-24]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

PRODUCTS = {   # product -> (title, referee cf name)
    "composite_reflectivity": ("Composite reflectivity", "refc"),
    "qpf_1h": ("1 h precipitation", "qpf1h"),
    "t2m": ("2 m temperature", "t2m"),
    "d2m": ("2 m dewpoint", "d2m"),
    "wind10": ("10 m wind", "ws10m"),
    "mslp": ("MSLP", "mslp"),
}


def _hours(spec: str):
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _model_png(maps: Path, product: str, fh: int, kind: str = "") -> Path | None:
    if maps is None:
        return None
    suffix = f"_{kind}" if kind else ""
    hits = sorted(Path(maps).glob(f"{product}/*_f{fh:03d}_{product}{suffix}.png"))
    return hits[0] if hits else None


def _referee_png(referee: Path, cf: str, fh: int, valid: datetime) -> Path | None:
    if referee is None:
        return None
    p = Path(referee) / f"f{fh:02d}_cf_{cf}_{valid:%Y%m%d_%H%M}z.png"
    return p if p.is_file() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run out dir (has maps/)")
    ap.add_argument("--label", default="WOOF-NAM")
    ap.add_argument("--ref12", type=Path, default=None, help="maps dir of our 12 km run")
    ap.add_argument("--referee", type=Path, default=None, help="case referee png dir (archived NAM 12 km)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default=None, help="run start YYYY-MM-DDTHH (--nest reads it from the history)")
    ap.add_argument("--hours", default="1-24")
    ap.add_argument("--products", default=",".join(PRODUCTS))
    ap.add_argument("--nest", default=None, metavar="SOURCE",
                    help="any-domain sheets from the run's history (gfs, ifs, parent, ...)")
    ap.add_argument("--cache", type=Path, default=None, help="referee download cache (--nest)")
    ap.add_argument("--parent-run", type=Path, default=None,
                    help="--nest parent: the 12 km run, drawn as the source column on the nest frame")
    a = ap.parse_args(argv)
    if a.nest:
        import json  # noqa: PLC0415
        from namgpu.io.io_referee_sheets import NEST_PRODUCTS, nest_sheets  # noqa: PLC0415
        hist = sorted(a.run.glob("wrfout_d01_*"))
        if not hist:
            raise SystemExit(f"{a.run}: no wrfout_d01_* history")
        prods = NEST_PRODUCTS if a.products == ",".join(PRODUCTS) else tuple(a.products.split(","))
        rec = nest_sheets(hist, a.nest, a.out, _hours(a.hours), prods, a.cache, parent_run=a.parent_run)
        print(json.dumps({k: rec[k] for k in ("label", "frame", "wall_s", "stages")}, default=str))
        return 0

    from gpuwm import rustwx
    renderer = rustwx.find_renderer()
    if renderer is None:
        raise SystemExit(rustwx.renderer_remedy())
    if a.start is None:
        raise SystemExit("--start YYYY-MM-DDTHH is required without --nest (no case default in generic tools)")
    start = datetime.fromisoformat(a.start)
    maps = a.run / "maps"
    jobs = []
    for product in [p for p in a.products.split(",") if p]:
        title, cf = PRODUCTS[product]
        for fh in _hours(a.hours):
            valid = start + timedelta(hours=fh)
            cols, names = [], []
            if product == "composite_reflectivity":
                obs = _model_png(maps, product, fh, "observed")
                if obs:
                    cols.append(obs); names.append("MRMS")
            mine = _model_png(maps, product, fh)
            if mine is None:
                continue
            cols.append(mine); names.append(a.label)
            r12 = _model_png(a.ref12, product, fh)
            if r12:
                cols.append(r12); names.append("WOOF-NAM 12 km")
            ref = _referee_png(a.referee, cf, fh, valid)
            if ref:
                cols.append(ref); names.append("archived NAM 12 km")
            out = a.out / product / f"f{fh:02d}_{product}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((cols, names, out, f"{title}: " + " | ".join(names), valid))

    def one(job):
        cols, names, out, t, valid = job
        rustwx.compose_ensemble_sheet(renderer, [cols], out, title=t,
                                      valid_time=f"{valid:%Y-%m-%d %H:%M} UTC (+{int((valid - start).total_seconds() // 3600)} h)")
        return out, names

    with ThreadPoolExecutor() as ex:
        made = list(ex.map(one, jobs))
    lines = [f"# {a.label}: comparison sheets", "", f"Run: {a.run}", ""]
    for out, names in made:
        lines.append(f"- {out.relative_to(a.out)}: {' | '.join(names)}")
    (a.out / "INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"{len(made)} sheets in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
