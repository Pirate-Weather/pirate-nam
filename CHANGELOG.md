# Changelog

## Unreleased

- Composite and simulated reflectivity now match how NOAA's NAM post computed them. The history `COMPOSITE` and
  `REFL_10CM`, the maps, and the awphys GRIB2 `REFC` (record 3) and `REFD` (records 2, 295, 296) add the reflectivity
  of the convection scheme's rain (Z = 300 R^1.4, full strength below the freezing level, 20 dBZ weaker at the
  convective cloud top) to the microphysics reflectivity, as the NAM post did. Before, they carried the microphysics
  part only and ran 3-4 dB colder than the NAM's own REFC at 30 dBZ for the same state. Columns without convective
  rain are unchanged. The hourly maximum `REFDMAX` already had the term and is unchanged.

- `--source native` starts from the NAM analysis as the operational NAM forecast did. The analysis' rain and ice
  fractions and rime factor, its turbulent kinetic energy and its vertical velocity are kept (before, the rain and ice
  started as cloud water, the TKE at its floor and the vertical velocity at 0), and no second digital filter runs
  over it, because the NAM analysis file is already the NAM's filtered 0 h state. `--filter on` restores the filter;
  `--filter off|on` works for every source. GFS, ECMWF, ERA5 and parent starts are unchanged.

## 0.1.2 (2026-11-05)

- Nests run one-way by default, as NOAA's NAM did: a live parent (`--parent-center`, `--parent-size-km`,
  `--parent-size-y-km`) without `--two-way` now runs as `--concurrent`. `--two-way` enables feedback; two-way made
  Ian 27 hPa deeper than observed in our test (908.8 hPa against 937). In 0.1.1 a live parent ran two-way unless
  `--concurrent` was given.

- Maps of a domain across 180 (a nest over the Aleutians or Fiji) are no longer drawn wrong on gpuwm 2.9.12. That
  release's renderer accepts such a frame but smears every row east-west, so with it the map is not drawn and the
  log says to set `NAMGPU_RENDERER` to the patched renderer (`tools/renderer/install_renderer.sh`). The 12 km
  parent still draws on the CONUS frame, as gpuwm 2.9.12 also smears its own whole-circle frame. woof-nam keeps
  working with gpuwm 2.9.11 and 2.9.12.

- `woof-nam products` works on a plain `pip install`. In 0.1.1 it stopped every hour with `FileNotFoundError: ...
  rw_rrfs_upp ... not in a gpuwm release yet`, because the GRIB2 writer it calls ships in no gpuwm release. woof-nam
  now carries its own pinned writer: the first `woof-nam products` downloads `rw_rrfs_upp` (Linux x86-64, 13 MB,
  glibc 2.28 or later) from the woof-nam release and checks its size and SHA-256 before installing it. New command
  `woof-nam fetch-bridges` (`--from FILE`, `--where`). If the writer cannot be had, `products` and `run --products`
  stop at once with one line naming it and the command that fetches it, before any hour is read or any forecast
  step runs.

- `woof-nam nest --follow tc | low | object:refl | object:uh | path`: a nest that follows a storm. Start it with
  `--storm LAT,LON` or `--storm-id` (an IBTrACS id). The window moves by whole grid cells at the source's boundary
  times. The nest keeps its own state where the old and new windows overlap, and the new strip starts from the
  source. Maps and sheets follow the window. On Typhoon Ampil (48 h from GFS) the mean track error against IBTrACS
  was 62 km, against 95 km for the same nest held still, which lost the storm at +30 h. README "Moving nests".

## 0.1.1 (2026-11-05)

- `woof-nam nest --two-way` and `--concurrent`: a 12 km parent and a 3 km (or 4 km) nest on the parent's own grid, run
  together in one process on one card. The nest takes its lateral boundaries from the live parent every parent step
  (NMMB's PARENT_UPDATE_CHILD_PSFC / _BNDRY and UPDATE_BC_TENDS), as the operational NAM nests ran (`--concurrent`).
  `--two-way` adds NMMB's two-way feedback: box means of the nest's T, Q, CW, U and V replace the parent's in the
  overlap (GENERATE_2WAY_DATA, PARENT_2WAY_UPDATE; `--2way-wgt`, default 1.0). History for both domains in
  `<out>/parent/` and `<out>/nest/`. A live parent runs two-way by default (any `--parent-*` option); the
  `--source parent --parent-run` hand-off and the one-domain door stay one-way.

- On PyPI as `recast-woof-nam` (`pip install recast-woof-nam`); the command stays `woof-nam` and the import name
  stays `namgpu`.
- 3 km nests anywhere (`woof-nam nest`), from GFS or from a WOOF-NAM 12 km parent run.
- GRIB2 products in the NAM awphys shape (`woof-nam products`).
- ERA5 (`--source era5`) and ECMWF IFS (`--source ifs`, `ifs-0p25`) starts, beside GFS and the archived NAM analysis.
- Restarts: `woof-nam run --restart-every N` and `--resume`, byte-identical to a straight run.
- SPD10MAX is the hourly maximum 10 m speed in m/s; wind maps over 60 kt use a 0-140 kt scale.
- Maps draw coastlines, borders and states from a bare pip install (the basemap ships in the wheel).
- `woof-nam licence-check` finds the installed distribution that ships namgpu, whatever its name.

- Less GPU memory, same output: one device memory pool per run that grows only when a request needs it (never past
  `CUPY_GPU_MEMORY_LIMIT` when that is set) and hands unused blocks back after each init stage, radiation call and
  history write; the digital filter keeps its save arrays in host memory and frees everything it owns when it ends; the
  3-D history fields are computed one at a time into one buffer as they are written; init-only scratch is dropped
  after the init; banded column physics and radiation chunks free their arrays as soon as a call ends; the radiation
  and MYJ/BMJ table builds keep their work arrays in global memory. The per-thread local-memory reservation the step
  loop needs is 10,016 bytes (2.44 GiB on an RTX 5090, was 11,264), and `woof-nam run --init-from` and `--resume`
  set it before the load, as a fresh start does. Every history value is unchanged.
- Restart sets on the full memory path (cards of 64 GiB and more): a set leaves out the step, history, unit and
  init-only scratch, the RRTM chunk context and driver-prep tables and the init-only 30-arcsec terrain, exactly what the
  32 GB fit path has already released at the hour (`restart.transient`). Every one of them is written before it is read
  in a step, and the resumed run allocates them again on first use as a fresh start does. A 12 km set on a 96 GB card
  was 36.7 GiB with them; it is now the size of a fit-path set.
- `woof-nam run --resume` from a full-path set no longer stops with `AttributeError: '_Skipped' object has no attribute
  'w'`: an entry the set cannot hold (the RRTM chunk context) is left unset, so the radiation call builds it as on a
  fresh start; sets written before this fix load the same way.
- `woof-nam nest`: the CuPy pool cap follows the memory path the forecast takes on the card (the 32 GB fit below
  64 GiB of card, the full path at and above, `NAMGPU_MEMORY_PATH` as set). A 12 km parent on a 96 GB card ran out
  of memory under a cap sized for the fit path. When the card size cannot be read and no path is set, no cap is set.
- `woof-nam nest` sizes its memory check from this release's memory use: a 333 x 333 nest is admitted on an 8 GB
  card (the check asked for about 18 GiB before), and common cards are counted by their own SM count.
- `woof-nam nest --source parent --parent-run DIR` takes the parent door's `--out` folder or its `run/` folder (the
  one holding `wrfout_d01_*`); the refusal names the folders it looked in.

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
