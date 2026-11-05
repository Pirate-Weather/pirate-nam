# WOOF-NAM

WOOF-NAM is a GPU port of NOAA/NCEP's North American Mesoscale model (NAM, nam.v4.2.11): the NMMB
(Nonhydrostatic Multiscale Model on the B grid) dynamics and the NAM's operational physics, running
on one NVIDIA card or on several cards of one machine. NCEP retired the NAM on 2026-11-03 12Z; this
package keeps the model running.

Distribution name `woof-nam`, import name `namgpu`, version 0.1.0. Not on PyPI yet: install from GitHub (below) or from the wheel attached to the GitHub release.

## What it looks like

The 2025-03-14 12Z case, run from a fresh clone with `woof-nam run --source native --hours 24` (the NAM
analysis start, the digital filter, every physics scheme on). The sheets come from `woof-nam side-by-side`:
MRMS observed reflectivity, WOOF-NAM and the archived operational NAM 12 km forecast of the same cycle, on
the same map frame and colour scales.

![Composite reflectivity at +24 h: MRMS, WOOF-NAM, archived NAM](docs/evidence/f024_refc_mrms_woofnam_nam.png)

*Composite reflectivity, +24 h, valid 12Z 15 Mar 2025. MRMS | WOOF-NAM | NAM. Like the NAM, WOOF-NAM wraps
the comma head over the upper Midwest and puts the trailing line west of the observed one; neither model
has MRMS's second band across Mississippi and Alabama.*

![Composite reflectivity at +12 h: MRMS, WOOF-NAM, archived NAM](docs/evidence/f012_refc_mrms_woofnam_nam.png)

*Composite reflectivity, +12 h, valid 00Z 15 Mar 2025. WOOF-NAM and the NAM put the same curved line with a
40+ dBZ core over Iowa and Missouri; MRMS has it farther west, over Nebraska and Kansas.*

![Total precipitation at +24 h: WOOF-NAM, archived NAM](docs/evidence/f024_precip_total_woofnam_nam.png)

*Precipitation since the start, +24 h. WOOF-NAM | NAM: the same Pacific coast, upper Midwest and Ohio
Valley maxima.*

![2 m temperature at +24 h: WOOF-NAM, archived NAM](docs/evidence/f024_t2_woofnam_nam.png)

*2 m temperature, +24 h. WOOF-NAM | NAM.*

![2 m dewpoint at +24 h, WOOF-NAM](docs/evidence/f024_d2m_woofnam.png)

*2 m dewpoint, +24 h, one of the maps every `woof-nam run` draws.*

The +12 h sheets and the other maps are in [docs/evidence/](docs/evidence/).

## What you need

- **Linux** (x86_64) with Python 3.12 or newer.
- **An NVIDIA card with 32 GB (one RTX 5090) or larger; up to 85 GB on a 96 GB card without the fit.** On a
  card with less than 64 GB the run takes the 32 GB fit by itself: arrays that live inside one step are handed
  back after their last user, the radiation working set is dropped between calls and the radiation runs in
  8-row chunks. Every physics scheme stays on. The run log and `forecast-receipt.json` name the memory path
  that ran (`memory path: fit` or `full`); `NAMGPU_MEMORY_PATH=fit` or `full` picks one on any card.
  Fetching data, the static fields, maps and the side-by-side panels need no big card (the static-field
  build uses an NVIDIA card for a few seconds).
- **Or two or more 32 GB cards of one machine** (`woof-nam run --cards N`, see "Running on more than one
  card"): the 12 km grid ran on 2, 4 and 8 RTX 5090 cards at 19-31 GiB per card, with the same bits on any
  number of cards.
- **An NVIDIA driver for CUDA 13.** No system CUDA toolkit is needed: pip installs the CUDA runtime
  and the compiler headers (`cupy-cuda13x[ctk]`). The kernels are compiled on the card at first use
  (NVRTC) and cached.
- **Disk:** 3.7 GB of inputs for a 24 h case (about 7 GB for 48 h), 2 GB for the terrain set,
  and room for the hourly history. The 24 h example case is fetched in about 2.5 minutes on a fast link.
- **Internet access** to NOAA's open-data buckets on AWS (NAM, GFS and MRMS). No account is needed.

## Install

```
python3 -m venv woof-nam-venv && source woof-nam-venv/bin/activate
pip install "woof-nam @ git+https://github.com/recastsystems/woof-nam.git"   # or, in a clone: pip install .
gpuwm fetch-geog --datasets topo_gmted2010_30s --root ~/WPS_GEOG              # GMTED2010 terrain, 174 MB download
```

pip installs `woof-nam` with everything a run reads inside the package: the NAM parm files, the
NOAA radiation, CO2 and ozone fix files, the Noah tables and the CUDA kernels. It also installs
`cupy-cuda13x[ctk]`, numpy, scipy, netCDF4, Pillow and `gpuwm` (2.9.11 or newer), which supplies the
Rust GRIB2 decoder, the NetCDF writer and the map renderer `rw_wrfbatch`. The terrain for gravity-wave
drag is read from `NAMGPU_GEOG` (default `~/WPS_GEOG`).

Everything runs through one command, `woof-nam`; `woof-nam --help` lists the steps and every step has
its own `--help`. Run them from one working folder: `cases/`, `static/` and `runs/` are made there
(set `NAMGPU_ROOT` to use another folder).

## Quick start

```
mkdir nam-work && cd nam-work
woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z     # 3.7 GB, ~2.5 min
woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z
woof-nam side-by-side mrms     --raw cases/2025-03-14_12z/raw/mrms --panels panels --hours 2025031512
woof-nam side-by-side archive  --raw cases/2025-03-14_12z/raw/nam --cycle 2025031412 --panels panels --hours 2025031512
woof-nam side-by-side woof-nam --history runs/2025-03-14_12z/wrfout_d01_* --panels panels --hours 2025031512
woof-nam side-by-side sheets   --panels panels --out sheets --hours 2025031512
```

The forecast (`woof-nam run`) needs the card above; every other line runs without it. With two or more cards
in the machine, add `--cards N` to the run (see "Running on more than one card") and check it against a
one-card run with `woof-nam compare runs/<one card> runs/<N cards>`.

## Start sources

`--source` on `woof-nam fetch` and `woof-nam run` picks the analysis the forecast starts from. Every source
goes through the same init (interpolation onto the NAM grid and its 60 levels on the GPU); only the table row
that says where the files are and which fields to read differs.

| `--source` | analysis | start times | ready after the start time | boundaries (every 3 h) | licence | for |
|---|---|---|---|---|---|---|
| `native` | archived NAM native analysis | cycles before 2026-11-03 12Z | archive only | GFS cycle 6 h older | US Government | the NAM's own start |
| `gfs` | GFS 0.25 degree | from 2021-01-01 00Z | about 3.5 h | GFS cycle 6 h older | US Government | real time and recent dates |
| `ifs-0p25` | ECMWF open data IFS HRES 0.25 degree | from 2025-03-14 12Z | about 7.5 h (00/12Z), 6.4 h (06/18Z) | same IFS cycle | CC BY 4.0 | real time and recent dates |
| `ifs` | ECMWF open data IFS HRES 0.1 degree | from 2026-10-20 12Z | about 9.6 h (00/12Z), 8.5 h (06/18Z) | same IFS cycle | CC BY 4.0 | real time and recent dates |
| `era5` | ECMWF ERA5 reanalysis 0.25 degree, hourly | 00, 06, 12 or 18Z from 1940-01-01 00Z | about 6 days | ERA5 analyses | CC BY 4.0 (Copernicus) | hindcasts and case studies |

ERA5 is read without an account from Google's public ARCO-ERA5 copy, through gpuwm's own ERA5 reader.
The store publishes its last day itself, and `woof-nam fetch` refuses a run that ends past it and names
that day. On 2026-11-04 the last day was 2026-10-29: recent days are the preliminary ERA5T, and the final
ERA5 runs about four months behind. The few days ERA5 does not cover yet are what `gfs` and `ifs` are for. ERA5
has no forecast leads: the start and every boundary time are analyses, so the boundaries are as good as
the reanalysis at every hour of the run. A 24 h case fetches 9 hours, about 11.5 GB (1.3 GB per hour for
the NAM 12 km grid's latitude band, every longitude), in about 1.5 min on a fast link. The fields read and how
they are mapped to the NAM's: `namgpu/init/source_records.py` (`ERA5`).

## Get the input data

Every input comes from NOAA's open data on AWS. `woof-nam fetch` writes it to `<case-dir>/raw/` and
skips files that are already there. Name the case folder `YYYY-MM-DD_HHz`: `woof-nam run` reads the
start time from it.

**Archived NAM start (cycles before 2026-11-03 12Z).** This fetches the NAM native analysis
(`bgrd3d00`: the initial state and the static surface fields), the archived NAM forecast of the same
cycle for comparison, the GFS boundaries and MRMS reflectivity:

```
woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z
```

**GFS start (any cycle, including after the NAM's retirement).** This fetches GFS and MRMS only.
For a cycle with no NAM files on NOAA's bucket, the fetch does this by itself and says so.

```
woof-nam fetch --cycle 2026110412 --hours 24 --case-dir cases/2026-11-04_12z --gfs-only
```

A GFS start still needs the NAM grid's static surface fields (land mask, soil and vegetation type,
terrain) once. Build them from any archived NAM analysis, for example the example case's file:

```
woof-nam static cases/2025-03-14_12z/raw/nam/bgrd3d00.grib2
```

This writes `static/nam12_static.npz` (1.4 MB, a few seconds). Every later GFS start reuses it.

**IFS start (ECMWF open data, CC BY 4.0).** `--source ifs-0p25` (0.25 degree, cycles from 2025-03-14 12Z)
or `--source ifs` (0.1 degree, cycles from 2026-10-20 12Z) fetches the IFS start, its 3-hourly
boundaries from the same IFS cycle and MRMS. IFS open data has no sea-ice cover, so that one field
comes from the same cycle's GFS. An IFS start uses the same static fields as a GFS start.

```
woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z --source ifs-0p25
woof-nam run --case 2025-03-14_12z --source ifs-0p25 --hours 24 --out runs/2025-03-14_12z-ifs
```

**ERA5 start (hindcasts, CC BY 4.0).** `--source era5` fetches the ERA5 analysis at the start and every 3 h
after it, and MRMS where it exists (MRMS on NOAA's bucket starts in October 2020; older dates have no
reflectivity to compare with). ERA5 carries its own sea ice, snow, soil and sea-surface temperature, so
nothing comes from another model. An ERA5 start uses the same static fields as a GFS start.

```
woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z --source era5
woof-nam run --case 2025-03-14_12z --source era5 --hours 24 --out runs/2025-03-14_12z-era5
```

## Run one forecast

```
woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z
```

`--case` is the case folder from `woof-nam fetch` (a bare name is looked up under `cases/`).
`--source gfs` starts from GFS instead, `--source ifs-0p25` or `--source ifs` from the IFS, `--source era5`
from ERA5. The run writes:
- hourly history `wrfout_d01_<valid time>` (wrfout-shaped NetCDF);
- maps in `<out>/maps/<product>/` (composite reflectivity, 2 m temperature and dewpoint, 10 m and
  850 hPa wind, MSLP, 1 h precipitation);
- `forecast-receipt.json` (init time, ms per step, per-stage timing, end-of-run finite check).

`woof-nam init` builds and writes the initial state only (0 h history and maps), without a forecast.

## Restarts

`--restart-every 1h` writes the whole forecast state every hour. `--resume` continues from the newest set to
the run's original end, and a resumed run writes the same history files, byte for byte, as a run that was
never stopped.

```
woof-nam run --case 2025-03-14_12z --source native --hours 84 --out runs/x --restart-every 1h
# the machine goes away at 30 h; on this or another machine with woof-nam and the same case:
woof-nam run --resume runs/x/restart        # case, source, end hour and output folder come from the set
```

- **What a set holds.** A set is everything the step reads, taken right after that hour's history file:
  - every prognostic and physics field: soil, snow, the precipitation and max-field accumulators, the radiation state;
  - the digital-filter state;
  - the boundary arrays and the position in the boundary files;
  - the tables built at init;
  - the step and radiation counters and the clock.

  It is `<out>/restart/wrfrst_d01_<valid time>/`, 15-16 GiB for the 12 km grid on either memory path: on a card of
  64 GiB or more (the full path) a set leaves out the step scratch that path keeps between steps, as the 32 GB fit path
  has already released it. `<out>/restart/LATEST` names
  the newest complete set, and `forecast-receipt.json` gives its path as `restart_latest`. The run's boundary
  files are copied once into `<out>/restart/boco`, so a set also resumes on a machine whose case folder is
  somewhere else.
- **Safe to kill at any moment.** A set is written under a temporary name and renamed when it is complete, so a run
  killed mid-write keeps the previous set. `--restart-keep N` keeps the newest N sets (default 1).
- **Cost.**
  - Each set is one copy of the state from the card into pinned host memory, queued in step order. The step loop
    does not wait for it: 0.01-0.02 s of host time per set.
  - The files are written by a background thread while the forecast steps on: 34-130 s per set on a busy disk.
  - The first set of a run also allocates the 15 GiB pinned buffer: 6-10 s, once.
  - Host memory: one pinned buffer the size of the state.
- **Proof.** 2025-03-14 12Z native start, 6 h, one RTX 5090, both runs from one saved initial state:
  - a straight run, and a run killed with SIGKILL right after its 3 h set, then resumed from that set;
  - all 7 hourly history files (0-6 h) of the two runs are byte-identical (`cmp`, and `woof-nam compare`: 0 values differ);
  - the resumed process loaded the set in 5.8 s.
  - The same on the full memory path (an RTX PRO 6000 96 GB, 2023-03-31 12Z GFS start, both runs from one saved
    initial state): a 2 h run with `--restart-every 1h`, resumed to 3 h, wrote f00-f03 byte-identical to a straight
    3 h run (85 variables, 0 values differ). Each set was 16.2 GiB, written in 2-4 s; the resume loaded it in 8-9 s.
- **Scope.** One card. With `--cards N` the restart options stop with a message: each card would have to write its
  own rows and resume on the same N cards, and that path is not written yet.

A run that has to start from the same bits as another one saves the initial state with `--keep-init` (on one
card, after the init, into `<out>/init`). The second run then starts with `--init-from <out>/init`. A fresh init
is not bit-repeatable, because two of its sums use float atomics.

## Make maps

Every map is drawn by gpuwm's Rust renderer `rw_wrfbatch`. Each map is titled "WOOF-NAM 12 km" with
the init time, lead and field, plus the valid time. A forecast draws its own maps when it ends. To
redraw them from history files:

```
woof-nam maps runs/2025-03-14_12z/wrfout_d01_* --out runs/2025-03-14_12z/maps
```

To compare sources side by side (MRMS | WOOF-NAM | archived NAM, with the same colour scales, map frame
and image size), draw the panels for each source, then compose the sheets, as in the Quick start.
`--hours` takes valid times (`YYYYmmddHH`, one or more). A sheet needs every source's panel for its
hour; the MRMS and archived NAM panels can be drawn before any forecast has run.

Every `woof-nam` step is also a module (`python -m namgpu.init.init_fetch_case`, `namgpu.init.gfs_start`,
`namgpu.run_forecast`, `namgpu.run_init_case`, `namgpu.io.maps`, `namgpu.io.io_referee_panels`,
`namgpu.io.io_referee_sheets`, `namgpu.io.hist_compare`, `namgpu.licence_check`), with the same options.

## GRIB2 products (awphys shape)

`woof-nam products` writes one GRIB2 file per forecast hour in the shape of NOAA's NAM 12 km CONUS product
`nam.tHHz.awphysFF.tm00.grib2`, so code that reads NOAA's NAM files reads ours with two changes: the file name
starts with `woof-nam.`, and the generating process number is 254 where NOAA's is 84.

```
woof-nam products --run runs/2025-03-14_12z                   # every history hour of the run
woof-nam products --run runs/2025-03-14_12z --hours 0-6
woof-nam run --case 2025-03-14_12z --hours 84 --out runs/x --products   # write each hour while the forecast runs
woof-nam products --compare cases/2025-03-14_12z/raw/nam/awphys06.grib2 \
    runs/2025-03-14_12z/products/woof-nam.t12z.awphys06.tm00.grib2     # grid and inventory against NOAA's file
```

- **Files.** `<run>/products/woof-nam.tHHz.awphysFF.tm00.grib2` (HH = start hour, FF = forecast hour), a `.idx`
  beside each (exactly what `wgrib2 -s` prints, like NOAA's index files), and `index.json` for the run (file, bytes,
  records, valid time, and every record left out with its reason).
- **Grid.** NCEP grid 218: Lambert conformal, 614 x 428 points, 12.19 km, true latitude 25 N, orientation 265 E,
  NCEP's sphere. The grid section is byte-identical to NOAA's.
- **Records.** NOAA's discipline, category and parameter numbers, fixed surfaces, time ranges and record order, from
  the record table `namgpu/products/awphys.tsv` (one row per record of NOAA's file). The full 50-1000 hPa isobaric
  set every 25 hPa is in, with VVEL, DZDT, ABSV and TKE. Accumulations use NOAA's 12 h and 3 h buckets.
- **Packing.** GRIB2 complex packing, as in NOAA's files, with NOAA's own decimal and binary scale for each record.
  Values carry NOAA's precision.
- **How.** The history fields are regridded from the NMMB rotated lat-lon grid to grid 218 (bilinear; soil and
  vegetation classes and masks nearest), winds turned grid-relative as NCEP writes them, and handed to gpuwm's Rust
  GRIB2 exporter `rw_rrfs_upp`, which computes the post fields and encodes on every core. The post's column work
  uses a GPU with room, else the CPU (`--post-device`). If a shared card runs out of memory, the hour is rerun
  instead of lost, and `index.json` lists the rerun.
- **The exporter.** `woof-nam products` needs the build of `rw_rrfs_upp` with the NAM awphys profile. It is not in
  a gpuwm release yet. Until it is, set `GPUWM_BRIDGES` to a folder that holds that build.
- **Speed.** About 25 s of products per forecast hour (2 hours in flight) against about 56 s of model per forecast
  hour on one RTX 5090, so `--products` keeps up with the forecast. f00-f06 took 2 min 35 s on 24 cores.

**Checked against NOAA.** 2025-03-14 12Z, a 6 h native start, f00-f06, each file against NOAA's archived
`nam.t12z.awphysFF.tm00.grib2` of the same hour:

| hour | records | match NOAA's inventory line and order | MB (NOAA's file) |
|---|---|---|---|
| f00 | 438 | 438 of 438 | 60.4 (54.9) |
| f01 | 442 | 442 of 444 | 63.0 (57.7) |
| f02 | 442 | 442 of 444 | 63.7 (57.9) |
| f03 | 442 | 442 of 444 | 64.5 (58.7) |
| f04 | 442 | 442 of 444 | 64.9 (58.8) |
| f05 | 442 | 442 of 444 | 65.6 (59.8) |
| f06 | 444 | 444 of 446 | 66.1 (60.4) |

- The grid section is byte-identical at every hour.
- The two records missing at f01-f06 are MAXUW and MAXVW (hourly 10 m maximum wind). The history of this check was
  written before the model wrote its hourly 10 m maximum wind. A run of this version writes it.
- f00 has fewer records because NOAA's analysis file has no hourly maxima, minima or accumulations either.
- Values, our forecast against NAM's on grid 218 at f06 (correlation): surface temperature 0.997, 2 m maximum and
  minimum temperature 0.999, 2 m maximum and minimum RH 0.986 and 0.987, gust 0.925, PBL height 0.952, Haines index
  0.951. `tools/grib2_value_check.py` prints correlation and bias for every record.
- In that history the vertical velocity is the bare W. NAM's post reads W_TOT, which also carries the terrain-forced
  term, so VVEL and DZDT have NAM's structures at about 0.7 of its amplitude. A run of this version writes W_TOT, and
  the products use it.

**Not in the files.** 8 of NOAA's 454 records: visibility at cloud top, the Blackadar mixing length (NOAA's file holds
zero there), the accumulated snowfall (WEASD), the four dominant precipitation types (CSNOW, CICEP, CFRZR, CRAIN) and
lightning (LTNG). A record that is missing is left out of the file, never written as zeros or under another record's
name. `docs/COMPATIBILITY.md` lists what a site changes, and `docs/COMPATIBILITY-TABLE.md` gives every record.

## Output size

For the 12 km grid, per forecast hour:

| output | per hour | f00-f84 (85 hours) |
|---|---|---|
| awphys GRIB2 (`woof-nam products`) | 60-66 MB (f00-f06) | about 5.6 GB |
| history `wrfout_d01_*`, lossless (zlib 1 + shuffle) | about 1.0 GB | about 86 GB |
| history, uncompressed (for scale) | 2.93 GB | - |
| restart set (`--restart-every`) | 15-16 GiB per set | `--restart-keep` sets (default 1) |

- The 84 h GRIB2 figure is 85 files at the f06 size. NOAA's own files run about 10 % smaller (60.4 MB at f06).
- The history is compressed losslessly by default: 2.93 GB -> 1.01 GB an hour (2.9x), and every value reads back
  exact. Any netCDF-4 or HDF5 reader opens it with no plugin.
- `--history-fields` picks the fields of each hourly history file: `products` (default: what the maps, side-by-side
  sheets and `woof-nam products` read), `full`, `state` (the 3-D model state), or a comma list. `--full-every 3h`
  (default) makes every 3rd hour carry every field. The presets change the size little (products 1,003 MB, full
  1,012 MB, state 862 MB an hour), because the awphys records need every 3-D field and those are 98 % of the bytes.
  The way to write less is fewer history hours, or deleting the history once the GRIB2 files are written.

## Scope

- **Domain:** the NAM 12 km CONUS parent (rotated lat-lon B grid, 954 x 835 x 60, 25 s step).
- **Dynamics:** NMMB, ported from the NAM source to CUDA kernels.
- **Physics:** the NAM's suite, all on the GPU:
  - Ferrier-Aligo microphysics;
  - Betts-Miller-Janjic convection;
  - MYJ boundary layer and the Janjic surface layer;
  - Noah land surface;
  - RRTM longwave and shortwave (NCEP's RRTMG, hourly);
  - Janjic gravity-wave drag (GMTED2010 terrain).
- **Start sources:** the archived NAM native analysis (`bgrd3d00`, for cycles before the
  retirement), GFS 0.25 degree for any date from 2021, ECMWF open data IFS (0.25 degree from 2025-03-14 12Z,
  0.1 degree from 2026-10-20 12Z) or the ERA5 reanalysis (from 1940). On the NAM and GFS starts the lateral
  boundaries come from the GFS cycle 6 hours older, as in the operational NAM; on an IFS start they come
  from the same IFS cycle; on an ERA5 start they are ERA5 analyses ("Start sources" above).
- **Not bitwise with the NAM.** The port runs in single precision with fast numerics. It is judged
  by whole wet forecasts beside the archived NAM forecast of the same cycle and MRMS reflectivity.

## Speed and memory

One RTX PRO 6000, 96 GB (full memory path), fresh clone and wheel install, 2025-03-14 12Z native start:
- **Digital filter (TDFI):** 70.7 s (150 backward and 150 forward steps of 8 s), before the forecast, as the
  NAM runs it.
- **24 h forecast:** 3,456 steps in 615 s, 178 ms per step (all physics, hourly history); 731 s for the whole
  `woof-nam run`, start state and maps included. Every field finite at the end.
- **Memory:** the forecast's pool ended at 74 GiB; the card peaked at 84.5 GB.

The 32 GB fit with the digital filter: 6 h on the fit path with the memory pool capped at 31 GiB (what a 32 GB card has free), on the 96 GB card: filter 56.7 s, 299 ms per step, every field finite, card peak 30,336 MiB.

One RTX 5090, 32 GB (the 32 GB fit, chosen by itself; measured before `woof-nam run` ran the digital filter):
- **12 h forecast:** native start, all physics, 1,728 steps in 586 s, 339 ms per step (init 50 s).
  Every field finite at the end.
- **Memory:** this run's peak on the card was 31,248 of 32,607 MiB, while the start was built (the forecast
  process plus its init helpers); in the step loop the forecast process peaked at 28,482 MiB. Pool 21.6 GiB
  at the end.

## Running on more than one card

The same forecast runs on several cards of one machine. Add `--cards N`:

```bash
woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z-4c --cards 4
woof-nam run ... --cards 4 --card-list 4,5,6,7      # pick the cards (default 0..N-1; or set NAM_CARDS)
```

Each card picks its memory path as on one card: under 64 GB every rank takes the 32 GB fit by itself (radiation
8 rows at a time). The 12 km runs in the table below used the full path with 8-row radiation
(`NAMGPU_MEMORY_PATH=full NAMGPU_RRTM_ROWS=8`), which fits 4 or 8 cards of 32 GB with room to spare and 2 only on
otherwise empty cards; on 2 cards leave the fit on.

What happens:

1. **The initial state is built once**, on the first card, and saved to `<out>/init/` (about 14 GB for the
   12 km grid). The run deletes it at the end; `--keep-init` keeps it, and `--init-from <folder>` reuses a saved
   one (from an earlier `--keep-init` run of the same case, source and hours). Add
   `--managed-init` when the whole-domain init does not fit one card.
2. **One process per card.** The grid is cut into bands of whole rows (row slabs): with `JM` rows and `N` cards
   each card owns `JM // N` rows, and the first `JM % N` cards one more. Every card keeps the full west-east width
   plus 3 halo rows on each side it shares with a neighbour. On the 12 km grid (954 x 835) that is 418 + 417
   rows on 2 cards, 209/208 on 4, 105/104 on 8.
3. **Halo rows are exchanged over NCCL** at the 21 places in the time step where NMMB itself exchanges halos,
   on the GPU stream, with no host sync. The cards need no peer-to-peer link; NCCL goes through host shared
   memory when there is none.
4. **Sums over the whole domain are made the same on any number of cards**: each card sums its own rows in a
   fixed order, the row sums are gathered, and every card adds them in row order 1..JM. One card uses the same
   code, so a forecast gives the same bits run after run.
5. **History is one file per hour, as on one card.** Each card hands its owned rows to a pool of writer processes
   (`--hist-workers`, default 4) through `/dev/shm`, which writes `wrfout_d01_<valid time>`; maps follow as usual.
   `<out>/forecast-receipt.json` lists ms/step, memory and the non-finite census per card; `<out>/rank<r>.log` is
   each card's log.

**Same bits on any number of cards.** Hourly history, 7 hours (0-6 h) x 26 variables, 0 values differ in every
pair below:
- the 12 km CONUS grid, native start 2025-03-14 12Z, every physics scheme on: 2 cards vs 4 cards, and two 4-card
  runs on different cards;
- a 954 x 279 band of the same grid (its centre rows; GFS start), which fits one 32 GB card: 1 card vs 2 cards,
  1 card vs 4 cards, and one card run twice;
- the same band through the installed `woof-nam run --cards 2` with the 32 GB fit on both cards (GFS start,
  digital filter + 1 h): 2 cards vs 1 card from the same saved init, 2 hours x 26 variables.

Check your own pair with `woof-nam compare <run a> <run b>` (exit 0 when every value is identical).

**Speed and memory, 12 km CONUS (954 x 835 x 60), 6 h, every physics scheme on.** Several cards: RTX 5090
32 GB, no peer-to-peer link, NCCL 2.28 over host shared memory. ms/step is the whole-run average per card from the
receipt; the wall includes loading the saved init.

| Cards | Rows per card (owned + halo) | ms/step | Wall for 6 h (864 steps) | Memory per card |
|---|---|---|---|---|
| 1 (RTX PRO 6000 96 GB) | 835 | 186-193 | about 165 s (24 h: 643 s) | 58-76 GB (pool about 58 GB) |
| 2 | 418 + 3, 417 + 3 | 174, 175 | 469 s (258 s of it loading the init on a busy host) | 26.9, 31.1 GiB |
| 4 | 209 + 3 to 6 | 89-102 | 123-151 s | 24.1-29.9 GiB |
| 8 | 105 + 3 to 6 | 53-55 | 92 s | 19.2-20.2 GiB |

So the full 12 km grid runs on one 32 GB card (the fit, 339 ms per step), on one 96 GB card (178 ms per step),
or faster on several cards.

**Other grids.** A grid is a row of `namgpu/parm/domains.json` and a configfile in `namgpu/parm/`; `woof-nam run
--domain NAME` (also `init`) selects it. `conus12` is the NAM 12 km parent (the default). `conus4` is 1371 x 1101 x 60
over NAM's CONUS-nest footprint, GFS start only: its static surface is built on the GPU from the WPS_GEOG land sets
(`NAMGPU_GEOG`, else `~/WPS_GEOG`) on the first run and again when the start month changes, and with `--cards N` its
initial state is built in CUDA managed memory. The 4 km grid ran 24 h on 8 RTX 5090 cards (42.9 ms per step, 522 s
wall, 25-26 GB per card; from a checkout through `tools/run_nam_domain.sh`, which takes the same configfile).

Needs: one machine with N NVIDIA cards and the NCCL 2 library (`libnccl.so.2`) where CuPy can load it; about
14 GB of disk for the saved 12 km init; room in `/dev/shm` for one or two hours of history in flight.

## 3 km nests anywhere

`woof-nam nest` runs a NAM-physics domain anywhere on Earth. One command builds the static fields for the
new grid, fetches the start and the boundaries, runs the forecast and draws the maps and comparison sheets:

```
woof-nam nest --center 35.7N,139.7E --size-km 1000 --res-km 3 --source gfs \
    --start 2026110218 --hours 48 --out runs/japan-3km
```

- `--center LAT,LON` with N/S/E/W or signed degrees; `--size-km` (and `--size-y-km` for a rectangle);
  `--res-km` 3, 4, 5 or 12. The 3 km row takes the NAM CONUS-nest step and physics keys (dt 6.25 s, radiation
  every 192 steps, no parameterised convection).
- `--source auto | gfs | ifs | ifs-0p25 | parent`. `auto` takes IFS 0.1 degree where it is published and GFS
  before. `parent --parent-run DIR` is a one-way nest inside a WOOF-NAM 12 km run that wrote its hand-off set.
- Before it starts, the door checks the free memory on the card and refuses if the domain does not fit. It
  also refuses if no renderer is found. It writes `nest-receipt.json` with every stage's wall time, cores and
  VRAM. From a checkout, `tools/woof-nam nest ...` runs the same command.

Three whole 48 h runs, each 286 x 334 x 60 points at 3 km (1,000 km square), on one RTX 5090 (32 GB):

| case | start and boundaries | ms/step | forecast 48 h | VRAM (pool) | door, start to sheets |
|---|---|---|---|---|---|
| Missouri/Illinois squall line, 2025-03-14 12Z | WOOF-NAM 12 km parent (one-way) | 14.2 | 6.5 min | 26.8 GiB | 8.2 min |
| Japan, 2026-11-02 18Z | GFS 0.25 | 14.1 | 6.5 min | 22.7 GiB | 22.9 min |
| Southern Germany and the Alps, 2026-11-02 12Z | ECMWF IFS 0.1 | 13.8 | 6.4 min | 21.8 GiB | 21.1 min |

Most of the door time beyond the forecast goes on the download (GFS 7 min) and on drawing the referee
sheets. All three runs stayed finite to +48 h.

![Composite reflectivity, MRMS | WOOF-NAM 3 km nest | WOOF-NAM 12 km parent, +6 to +24 h](docs/nests/conus-parent-refc-f06-f24.png)

*Squall line, 2025-03-14 12Z (columns: MRMS, the 3 km nest, the 12 km parent; rows: +6, +12, +18, +24 h). The
nest draws the overnight line as one continuous band on about the axis MRMS has. The parent has a chain of
separate cells there. FSS at >= 40 dBZ (40 km window) is 0.24 for the nest and 0.15 for the parent at +18 h.
Two nest defects are visible from +12 h: echo along the boundary rows, and banded echo ahead of the line.*

![2 m temperature, WOOF-NAM 3 km Japan, +6, +12, +24 and +48 h](docs/nests/japan-gfs-t2m-f06-f48.png)

*Japan from GFS, 2026-11-02 18Z. The nest resolves valley, terrain and coastal structure that GFS 0.25 does
not have. Against 17-28 Japanese METARs, its 2 m temperature RMSE is 1.57, 1.93 and 1.82 C at +6, +12 and
+24 h (GFS: 1.42, 2.28, 2.50) and 1.40 C at +48 h (GFS 0.97). Its 2 m dewpoint over interior Honshu is too dry
(-1.35 C at +48 h).*

![ECMWF IFS 0.25 analysis | WOOF-NAM 3 km | ECMWF IFS 0.25 forecast, 2 m T, 2 m dewpoint, 10 m wind, MSLP and 6 h precipitation](docs/nests/europe-ifs-sheets-f12-f48.png)

*Southern Germany and the Alps from IFS 0.1, 2026-11-02 12Z (columns: the IFS analysis, the 3 km nest, the
IFS forecast). Against 139-147 METARs, the nest's 10 m wind matches IFS (lower RMSE at 9 of 12 leads). Its
2 m temperature is worse than IFS at every lead from +6 h: the gap is 0.1-0.3 C by day and largest at night.
Its 2 m dewpoint is drier than IFS, and it produces light rain and drizzle at about one fifth of the observed
frequency.*

### Data sources for nests

Static fields come from the WPS_GEOG datasets `gpuwm fetch-geog` installs, read by
`namgpu/init/static_build.py`. Start, boundary and referee data are fetched from the open archives below.
Each row's licence is also written next to its table in the code (`static_build.DATASETS`,
`fetch_case.FETCH_ROWS`, `sources.SOURCES`, `referee_rows`). No NC-licensed data is read.

| use | dataset | source and licence |
|---|---|---|
| terrain | GMTED2010 30 arc-second | USGS/NGA (Danielson & Gesch 2011); U.S. public domain |
| land use | MODIS 20-category IGBP + lakes | NASA MODIS (MCD12Q1-derived); NASA open data |
| soil texture | SoilGrids250m texture classes | ISRIC (Hengl et al. 2017); CC BY 4.0, attribution in the static receipt |
| green fraction, albedo, max snow albedo | MODIS climatologies | NASA MODIS; Barlage et al. 2005 (max snow albedo) |
| deep-soil temperature | ERA5 1990-2019 climatology (WeatherBench 2) | Copernicus licence: free, commercial use allowed, attribution required |
| start and boundaries | GFS 0.25 degree | NOAA/NCEP, U.S. public domain (AWS NODD) |
| start and boundaries | ECMWF open data IFS 0.1 / 0.25 degree | ECMWF, CC BY 4.0 (credited on every map) |
| snow, sea ice where IFS lacks them | GFS 0.25 degree f000, same cycle | NOAA/NCEP, U.S. public domain |
| parent start and boundaries | your own WOOF-NAM 12 km history | the parent's own start-source credit is carried onto the maps |
| truth (referee panels) | MRMS reflectivity and QPE; GPM IMERG | NOAA, U.S. public domain; NASA/JAXA open data |

## Layout

```
namgpu/            the package: config, grid, state, kernels/ (one .cu per unit), dynamics/,
                   boundary/, physics/, init/, io/, steps/, driver.py, cli.py (the woof-nam command)
namgpu/parm/       NAM parm files (NOAA, unmodified)
namgpu/data/       tables; fix/ NOAA radiation, CO2 and ozone fix files; fix/noah/ Noah stand-in tables
namgpu/third_party/arwen/  ArWen (gpuwm) files reused, Apache-2.0
namgpu/UNITS.json  every port unit and its status
tools/             extract and check tools (development); fetch_fix.sh refreshes namgpu/data/fix
tests/             smoke_full.py (whole forecast), unit tests
cases/<case>/      CASE.md and the referee recipe; inputs are fetched, never committed
static/            nam12_static.npz for GFS starts, built once by woof-nam static (not committed)
CONTRACT.md        array layout, step order, kernel convention, done rule
```

## Known gaps

- **Nests:** the NAM's own fixed nests (CONUS 3 km, Alaska, Hawaii, Puerto Rico, fire weather) are not
  ported as such; `woof-nam nest` ("3 km nests anywhere" above) runs a 3 km domain of your choice instead.
  Its `--cards N` path and its sheets on several cards have not run on two physical cards yet.
- **Multi-card:** one machine only (no multi-node runs). The 5 km and band configfiles are not rows of
  `domains.json` (not measured as a whole run); they run from a checkout through `tools/run_nam_domain.sh`.
- **Memory fit on several cards:** measured on the 954 x 279 band (2 cards, same bits as 1 card). The full
  12 km grid on several cards is measured on the full memory path only.
- **Restarts:** one card ("Restarts" above); the NAM's own restart files are not read.
- **GRIB2 products:** 8 of NOAA's 454 awphys records are not produced ("GRIB2 products" above), and the
  exporter build `woof-nam products` needs is not in a gpuwm release yet.
- **Noah vegetation table:** the NAM's own VEGPARM.TBL was never published. A stand-in built from
  WRF's public-domain table is used (namgpu/data/fix/noah/PROVENANCE.txt).

## Licence

Apache-2.0 (LICENSE). The RRTMG radiation files derived from AER's code carry AER's BSD-3-Clause
notice, and the reused ArWen files are Apache-2.0. NOTICE lists every third-party origin and its
terms. `woof-nam licence-check` checks the per-file notices (in an install or a checkout).
