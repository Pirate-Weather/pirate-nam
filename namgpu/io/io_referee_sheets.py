"""io-referee-sheets: referee sheets, one per product per output hour.

Panels, left to right, per product: MRMS (truth, reflectivity only) | WOOF-NAM (this port) | archived NAM
(same cycle, grid 218). Products without a truth panel in the case fetch (2 m temperature, precipitation)
are two-panel: WOOF-NAM | archived NAM. Truth for those needs a fetch (ASOS for T2, MRMS QPE for precip).

Not a driver unit: it runs after the forecast, from files, and never inside the step loop. It is not in
driver.py's lists, so it has no run(ctx) hook.

Mode "new": no Fortran counterpart and no numerics, so there is no GPU kernel for this unit. Drawing is the
Rust renderer through gpuwm.rustwx (render law: matplotlib is not used for weather fields). The panels
themselves are written by namgpu.io.io_referee_panels.

Input layout (written by io_referee_panels):
    <panels>/<product>/<source>/<YYYYmmddHH>.png
    source in {mrms, nam_gpu, nam_archive}
Output:
    <out>/<product>/<YYYYmmddHH>_referee.png

Usage:  python -m namgpu.io.io_referee_sheets --panels <dir> --out <dir> --hours 2025031412 ...
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PRODUCTS = ("composite_reflectivity", "t2", "precip_total")
# Sources drawn per product, in panel order. Truth is only MRMS reflectivity in this case fetch.
PRODUCT_SOURCES = {
    "composite_reflectivity": ("mrms", "nam_gpu", "nam_archive"),
    "t2": ("nam_gpu", "nam_archive"),
    "precip_total": ("nam_gpu", "nam_archive"),
}
CMD = {"mrms": "mrms", "nam_gpu": "woof-nam", "nam_archive": "archive"}   # panel step per source
PANEL_NAMES = {"mrms": "MRMS", "nam_gpu": "WOOF-NAM", "nam_archive": "archived NAM"}
TITLES = {
    "composite_reflectivity": "Composite reflectivity",
    "t2": "2 m temperature",
    "precip_total": "Precipitation, total since start",
}
UNIT = "io_referee_sheets"


def panel_path(panels: Path, product: str, source: str, valid: datetime) -> Path:
    return Path(panels) / product / source / f"{valid:%Y%m%d%H}.png"


def sheet_path(out: Path, product: str, valid: datetime) -> Path:
    return Path(out) / product / f"{valid:%Y%m%d%H}_referee.png"


def compose_hour(renderer: Path, panels: Path, out: Path, product: str, valid: datetime) -> Path:
    """Compose the source row for one product and hour (MRMS first where it exists)."""
    from gpuwm import rustwx  # noqa: PLC0415

    row = []
    for source in PRODUCT_SOURCES[product]:
        p = panel_path(panels, product, source, valid)
        if not p.is_file():
            raise FileNotFoundError(f"missing panel: {p} (draw it first: woof-nam side-by-side {CMD[source]} ...)")
        row.append(p)
    names = " | ".join(PANEL_NAMES[s] for s in PRODUCT_SOURCES[product])
    title = f"{TITLES[product]}: {names}"
    return rustwx.compose_ensemble_sheet(
        renderer, [row], sheet_path(out, product, valid),
        title=title, valid_time=f"{valid:%Y-%m-%d %H:%M} UTC")


def run_sheets(panels: Path, out: Path, hours: list[datetime], products=PRODUCTS) -> list[Path]:
    from gpuwm import rustwx  # noqa: PLC0415

    renderer = rustwx.find_renderer()
    if renderer is None:
        raise SystemExit(rustwx.renderer_remedy())
    made = []
    for product in products:
        for valid in hours:
            made.append(compose_hour(renderer, Path(panels), Path(out), product, valid))
    return made


def main(argv=None):
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Compose side-by-side sheets, one per product and valid time: MRMS | WOOF-NAM | "
                    "archived NAM, from the panels that `woof-nam side-by-side mrms|archive|woof-nam` rendered.",
        epilog="example: woof-nam side-by-side sheets --panels panels --out sheets --hours 2025031418",
        formatter_class=HelpFormatter)
    ap.add_argument("--panels", type=Path, required=True, help="the panel folder the panel steps wrote")
    ap.add_argument("--out", type=Path, required=True, help="sheet folder")
    ap.add_argument("--hours", nargs="+", required=True, help="YYYYmmddHH valid times")
    ap.add_argument("--products", default=",".join(PRODUCTS), help=f"comma list (default {','.join(PRODUCTS)})")
    a = ap.parse_args(argv)
    hours = [datetime.strptime(h, "%Y%m%d%H") for h in a.hours]
    products = tuple(p for p in a.products.split(",") if p)
    try:
        made = run_sheets(a.panels, a.out, hours, products)
    except FileNotFoundError as e:
        print(f"{UNIT}: {e}", file=sys.stderr)
        return 1
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
