# Changelog

## 0.1.1.dev2 (private build, not released)

- `woof-nam run --domain conus5`: the 5 km CONUS grid (1097 x 881 x 60, GFS start), a row of `parm/domains.json`.
- Static surfaces built from WPS_GEOG (conus4, conus5) read the permissive soil set by default: SoilGrids texture
  categories and the ERA5 deep-soil temperature, as gpuwm 2.9.4 does. `NAMGPU_GEOG_SOIL=wrf` (or
  `static_build --soil wrf`) selects the WPS STATSGO/FAO set. A cached static built from another soil set is rebuilt.
- Each WPS_GEOG dataset is looked up across NAMGPU_GEOG, the tree its topo_gmted2010_30s link points into,
  WPS_GEOG and ~/WPS_GEOG, so a geography folder that holds only the terrain link still builds a 4 or 5 km static.

## 0.1.1.dev1 (private build, not released)

- `woof-nam run --domain NAME` (and `init`): grids are rows of `namgpu/parm/domains.json`; `conus4` (4 km CONUS,
  GFS start) runs from the wheel. Its static surface is built from WPS_GEOG on first use and per start month.
- A GFS start on the 12 km grid with no archived analysis in the case and no `static/nam12_static.npz` builds that
  file once from the archived NAM analysis of the same month and day.
- `woof-nam fetch --gfs-only` also fetches the start's own GFS analysis (f000); before, a fetched GFS case could not
  start. `--hours N` fetches exactly the GFS boundary times an N-hour run reads (at least 6 h, 3-hourly), and the
  init chain fetches for the run's own length (it fetched 48 h). `--inputs-only` skips the comparison files (MRMS,
  archived NAM forecast). Each fetch writes `<case>/fetch-receipt.json` (seconds, bytes, requests).
- Boundaries for a run whose length is not a multiple of 3 h cover its last hours.
- `forecast-receipt.json` has a `stages` table (wall seconds and parallelism per stage: static, init, filter,
  forecast, maps), plus domain, grid, cards and version. The memory-path log line no longer prints the card size.

## 0.1.0 (2026-11-04)

- First package of the NAM GPU port: NMMB dynamics and the NAM 12 km physics suite on one card,
  native-analysis and GFS starts, hourly history, Rust-rendered maps, timing receipt.
- `woof-nam` command: fetch, static, init, run, maps, side-by-side, licence-check (the `python -m` forms
  still work).
- Everything a run reads ships in the wheel: NAM parm files, NOAA fix files, Noah tables, UNITS.json,
  the reused ArWen files. Working files go to the folder the command runs in (or NAMGPU_ROOT).
- Side-by-side: WOOF-NAM panels are drawn on the same CONUS frame and image size as the MRMS and
  archived NAM panels, so the three compose into one sheet.
