"""The `woof-nam` command: one front door for the steps the README walks through.

Each subcommand runs the same code as its `python -m ...` form, with the same options.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import textwrap


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Wraps a one-paragraph description to the terminal; keeps multi-line text (examples) as written."""

    def _fill_text(self, text, width, indent):
        if "\n" in text:
            return super()._fill_text(text, width, indent)
        return textwrap.fill(" ".join(text.split()), width, initial_indent=indent, subsequent_indent=indent)

# name -> (module, summary). The module's main(argv) does the work and owns --help.
COMMANDS = {
    "fetch": ("namgpu.init.init_fetch_case",
              "download one case's inputs from NOAA's open data on AWS (NAM analysis, GFS, MRMS)"),
    "static": ("namgpu.init.gfs_start",
               "build the NAM grid's static surface fields once (needed by GFS starts)"),
    "init": ("namgpu.run_init_case",
             "build the initial state only and write its 0 h history and maps (no forecast)"),
    "run": ("namgpu.run_forecast",
            "run one forecast: hourly history, maps and a timing receipt"),
    "maps": ("namgpu.io.maps",
             "draw maps from history files"),
    "side-by-side": ("namgpu.io.io_referee_panels",
                     "draw MRMS | WOOF-NAM | archived NAM panels, then compose comparison sheets"),
    "licence-check": ("namgpu.licence_check",
                      "check the per-file licence notices in this install"),
}

SIDE_BY_SIDE_HELP = """\
usage: woof-nam side-by-side {mrms,archive,woof-nam,sheets} [options]

Draw the same products from each source with the same colour scales and map frame, then put them
side by side: MRMS (observed) | WOOF-NAM (your run) | archived NAM (same cycle).

steps (each has its own --help):
  mrms       panels from the fetched MRMS reflectivity:
               woof-nam side-by-side mrms --raw cases/2025-03-14_12z/raw/mrms --panels panels --hours 2025031418
  archive    panels from the archived NAM forecast of the same cycle (cycles before 2026-11-03 12Z):
               woof-nam side-by-side archive --raw cases/2025-03-14_12z/raw/nam --cycle 2025031412 --panels panels --hours 2025031418
  woof-nam   panels from your run's history files:
               woof-nam side-by-side woof-nam --history runs/2025-03-14_12z/wrfout_d01_* --panels panels --hours 2025031418
  sheets     compose the panels into one sheet per product and hour:
               woof-nam side-by-side sheets --panels panels --out sheets --hours 2025031418

--hours takes valid times as YYYYmmddHH (one or more). A sheet needs every source's panel for its hour
(reflectivity: MRMS, WOOF-NAM, archived NAM; 2 m temperature and precipitation: WOOF-NAM, archived
NAM); a missing panel stops with its path.
"""


def _usage() -> str:
    from namgpu import __version__
    w = max(len(n) for n in COMMANDS)
    lines = [f"usage: woof-nam <command> [options]      (woof-nam {__version__})", "",
             "WOOF-NAM: the NAM 12 km (NMMB) model on one NVIDIA card. Run every command from one working",
             "folder: cases/, static/ and runs/ are made there (or set NAMGPU_ROOT).", "", "commands:"]
    lines += [f"  {n:<{w}}  {s}" for n, (_, s) in COMMANDS.items()]
    lines += ["", "Each command has --help. A first run, in order:",
              "  woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z",
              "  woof-nam static cases/2025-03-14_12z/raw/nam/bgrd3d00.grib2",
              "  woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z",
              "  woof-nam side-by-side --help"]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        from namgpu import __version__
        print(f"woof-nam {__version__}")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"woof-nam: unknown command {cmd!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    mod_name = COMMANDS[cmd][0]
    if cmd == "side-by-side":
        if not rest or rest[0] in ("-h", "--help", "help"):
            print(SIDE_BY_SIDE_HELP)
            return 0
        if rest[0] == "sheets":
            mod_name, rest = "namgpu.io.io_referee_sheets", rest[1:]
            cmd = "side-by-side sheets"
    sys.argv[0] = f"woof-nam {cmd}"          # argparse names the command in its usage lines
    rc = importlib.import_module(mod_name).main(rest)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
