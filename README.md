# WOOF-NAM

WOOF-NAM is a GPU port of NOAA/NCEP's North American Mesoscale model (NAM, nam.v4.2.11): the NMMB
(Nonhydrostatic Multiscale Model on the B grid) dynamics and the NAM's operational physics, running
on one NVIDIA card. NCEP retired the NAM on 2026-11-03 12Z; this package keeps the model running.

Distribution name `woof-nam`, import name `namgpu`, version 0.1.0. Private repository, not on PyPI.

## What it looks like

The 2025-03-14 12Z case: woof-nam (left), the archived operational NAM 12 km forecast of the same cycle
(middle) and MRMS observed composite reflectivity (right), at the same valid time.

![Composite reflectivity at +24 h: ours, NAM, MRMS](docs/evidence/f024_refc_ours_nam_mrms.png)

*Composite reflectivity, +24 h, valid 12Z 15 Mar 2025. Ours | NAM | MRMS, same colour scale. Both
models wrap the comma head over the upper Midwest; both our run and the real NAM put the trailing line
west of the observed one, and neither has MRMS's second band across Mississippi and Alabama.*

Surface fields, ours (left) | NAM (right), same colour scale (MRMS has no surface fields):

![2 m dewpoint at +24 h: ours, NAM](docs/evidence/f024_d2m_ours_nam_mrms.png)

*2 m dewpoint, +24 h, valid 12Z 15 Mar 2025. Ours | NAM, same colour scale.*

![2 m temperature at +24 h: ours, NAM](docs/evidence/f024_t2m_ours_nam_mrms.png)

*2 m temperature, +24 h, valid 12Z 15 Mar 2025. Ours | NAM, same colour scale.*

![2 m dewpoint at +12 h: ours, NAM](docs/evidence/f012_d2m_ours_nam_mrms.png)

*2 m dewpoint, +12 h, valid 00Z 15 Mar 2025. Ours | NAM, same colour scale.*

![2 m dewpoint at +6 h: ours, NAM](docs/evidence/f006_d2m_ours_nam_mrms.png)

*2 m dewpoint, +6 h, valid 18Z 14 Mar 2025. Ours | NAM, same colour scale.*

Every map, including our MSLP (the archived NAM set has no forecast MSLP map, so that panel reads
"not available"), is in [docs/evidence/](docs/evidence/).

## What you need

- **Linux** (x86_64) with Python 3.12 or newer.
- **An NVIDIA card with more than 76 GB of memory** for a forecast with the code published today. The
  12 km CONUS run's own memory pool is about 58 GB, and the card peaked at 75 GB in the 24 h run, which
  ran on a 96 GB RTX PRO 6000. A fit for 32 GB cards is coming; until it lands, a 24, 32 or 48 GB
  card cannot run a forecast. Fetching data, the static fields, maps and the side-by-side panels
  need no big card (the static-field build uses an NVIDIA card for a few seconds).
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

The forecast (`woof-nam run`) needs the big card above; every other line runs without it.

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

## Run one forecast

```
woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z
```

`--case` is the case folder from `woof-nam fetch` (a bare name is looked up under `cases/`).
`--source gfs` starts from GFS instead. The run writes:
- hourly history `wrfout_d01_<valid time>` (wrfout-shaped NetCDF);
- maps in `<out>/maps/<product>/` (composite reflectivity, 2 m temperature and dewpoint, 10 m and
  850 hPa wind, MSLP, 1 h precipitation);
- `forecast-receipt.json` (init time, ms per step, per-stage timing, end-of-run finite check).

`woof-nam init` builds and writes the initial state only (0 h history and maps), without a forecast.

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
`namgpu.io.io_referee_sheets`, `namgpu.licence_check`), with the same options.

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
  retirement) or GFS 0.25 degree for any date. On both starts the lateral boundaries come from the
  GFS cycle 6 hours older, as in the operational NAM.
- **Not bitwise with the NAM.** The port runs in single precision with fast numerics. It is judged
  by whole wet forecasts beside the archived NAM forecast of the same cycle and MRMS reflectivity.

## Speed and memory (one RTX PRO 6000, 96 GB)

- **24 h forecast:** 643 s at 186 ms per step (native start, all physics), which is about 21 min
  per 48 h at that rate. The one measured 48 h run, on an earlier tree with the card shared,
  took 2,734 s (395.5 ms per step) and finished with every field finite.
- **Memory:** the forecast's own pool is about 58 GB; the card peaked at 75 GB in the 24 h run.

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

- **Nests:** the NAM's nests (CONUS 3 km, Alaska, Hawaii, Puerto Rico, fire weather) are not
  ported; this is the 12 km parent only.
- **Multi-card:** in progress.
- **Memory fit:** the published code needs a card with more than 76 GB for a forecast, so it does not
  fit a 32 or 48 GB card yet, and there is no smaller example to run on one. The 32 GB fit is coming.
- **Noah vegetation table:** the NAM's own VEGPARM.TBL was never published. A stand-in built from
  WRF's public-domain table is used (namgpu/data/fix/noah/PROVENANCE.txt).

## Licence

Apache-2.0 (LICENSE). The RRTMG radiation files derived from AER's code carry AER's BSD-3-Clause
notice, and the reused ArWen files are Apache-2.0. NOTICE lists every third-party origin and its
terms. `woof-nam licence-check` checks the per-file notices (in an install or a checkout).
