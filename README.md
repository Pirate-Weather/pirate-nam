<p align="center"><img src="pirate/assets/logo.svg" width="130" alt="Pirate NAM logo: a storm cloud in a pirate hat and eye patch, above crossed lightning bolts"></p>

# Pirate NAM

Keeping the retired North American Mesoscale model sailing, on one GPU machine.

> **Experimental.** Not an official forecast, and not a NOAA or NCEP product. Runs may stop or change without notice. Don't use it for decisions that affect safety.

**[Map viewer](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/index.html) · [Data access (GRIB2 and Zarr)](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/data.html)**

## Why it's built this way

**The goal is to produce the forecast the NAM would have produced.** So Pirate NAM keeps as much of the legacy NAM pipeline as it can, and changes something only when it has to:

- **Same model design:** the NMMB model and the NAM's physics, through WOOF-NAM's GPU port.
- **Same analysis program:** NOAA's own GSI, unmodified, with the NAM's configuration.
- **Same structure:** a 6-hour catch-up before each forecast, using a GFS cycle 6 hours older than the forecast for boundaries, on the NAM's 12 km grid, out to 84 hours.

A change is made when an input no longer exists, such as the NAM's own observation files, or when testing shows it makes no measurable difference to how closely the result matches the NAM, such as running fewer analyses. Closeness to the archived NAM is the yardstick for every decision here, not accuracy against observations.

## What it is

NOAA retired the NAM on 3 November 2026. Pirate NAM runs it on a single two-card GPU machine by combining two existing pieces of work: [WOOF-NAM](https://github.com/recastsystems/woof-nam), an independent GPU port of the NAM's NMMB model, and NOAA's GSI, the Fortran program that blends observations into a model state.

The GPU port makes this practical: an 84-hour forecast takes about an hour on two cards, on a machine that costs about CAD 2.32 an hour.

## How a cycle works

Like the NAM, it runs a 6-hour catch-up before the forecast, so the model has absorbed recent observations by the time the forecast starts. For a 00Z forecast:

| Time | Step | Runs on |
|---|---|---|
| 18Z | Start from GFS | GPU |
| 20Z | GSI adds observations | CPU |
| 22Z | GSI adds observations | CPU |
| 00Z | GSI adds observations | CPU |
| 00Z to +84 h | Forecast | GPU |

WOOF-NAM forecasts on the GPU between analyses, with the model state kept on the card. At each analysis, its fields are written into a real NMMB restart file, GSI runs on that file unchanged, and whatever GSI changed goes back onto the card.

## Outputs

Every forecast is published as hourly fields from 0 to 84 hours, on the NAM's 12 km rotated latitude-longitude grid (954 × 835). The newest 8 cycles are kept, and [runs.json](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/runs.json) lists what's available.

| Product | What it's for | Where |
|---|---|---|
| **Map viewer** | Browsing a forecast in a browser | [index.html](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/index.html) |
| **GRIB2** | Full field set, for forecast tools | `<YYYYMMDDHH>/woof/grib/woofnam.t<HH>z.f<FFF>.grib2`, 85 files of about 41 MB |
| **Zarr** | A smaller field set for Python, with no server or download step | `<YYYYMMDDHH>/woof/zarr` |

All paths are under `https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/`. See the [data access page](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/data.html) for details.

**GRIB2 fields**

| Level | Fields |
|---|---|
| 1000, 925, 850, 700, 500, 300 and 250 mb | Temperature, wind, height, relative humidity |
| 2 m | Temperature, dew point |
| 10 m | Wind |
| Surface | Pressure |
| Mean sea level | Pressure |
| Whole atmosphere | Composite reflectivity, precipitable water |
| Accumulated | Precipitation, since the start and hourly |

**Zarr variables**

| Variable | Field | Units |
|---|---|---|
| `t2m`, `d2m` | 2 m temperature and dew point | °C |
| `mslp` | Sea level pressure | mb |
| `wind10`, `wind250` | 10 m and 250 mb wind speed | m/s |
| `apcp` | Precipitation since the start | mm |
| `refc` | Composite reflectivity | dBZ |
| `pwat` | Precipitable water | mm |
| `z500` | 500 mb height | dam |
| `t850` | 850 mb temperature, missing below ground | °C |

```python
import xarray as xr
ds = xr.open_zarr("https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/2026110512/woof/zarr", consolidated=True)
ds.t2m.sel(fhr=24)
```

Forecast hour is the time coordinate, and `latitude` and `longitude` are 2D. GRIB2 files have no originating centre set, because this isn't a product of any weather centre.

## Compared with NOAA's NAM and WOOF-NAM

| | NOAA NAM v4.2 | WOOF-NAM as released | Pirate NAM |
|---|---|---|---|
| **Status** | Retired 3 Nov 2026 | Open source | Experimental, running |
| **Model** | NMMB, Fortran | Same design, ported to GPU | WOOF-NAM, unchanged |
| **Numerics** | The reference | Single precision, not bit-identical to the NAM | Same as WOOF-NAM |
| **Hardware** | Supercomputer | One or more NVIDIA cards | Two V100S cards, 30 CPUs, 90 GB. One card, 15 CPUs and 45 GB also works, more slowly |
| **Grids** | 12 km parent plus higher-resolution nests | 12 km; 4 km, nests and others available | 12 km only |
| **Forecast length** | 84 h at 12 km | Your choice | 84 h |
| **Start** | Global model atmosphere plus NAM's own land, 6 h before the forecast | GFS, or a NAM analysis when one exists | GFS, 6 h before the forecast |
| **Observations** | GSI every hour through the catch-up (7 analyses) | None | GSI every 2 hours (3 analyses): more made no measured difference |
| **Observation sources** | NAM's dedicated data feed, satellite radiances, radar | None | RAP's hourly observation files and MRMS radar, since the NAM's own files ended with it. No GPS radio occultation in RAP's files. Radiances and lightning left out: no measured benefit |
| **Ensemble in GSI** | Global ensemble | None | GEFS, the newest available |
| **Digital filter** | At each catch-up start | At the start | At the start only: filtering each analysis made no measured improvement |
| **Soil and snow** | Carried from cycle to cycle, with observed rainfall | From GFS | From GFS. Seeding from the NAM archive improved scores by only 0.4% in testing |
| **SST and sea ice** | Refreshed from outside analyses | From GFS | From GFS |
| **Lateral boundaries** | GFS cycle 6 h older than the forecast | GFS cycle 6 h older than the start | GFS cycle 6 h older than the forecast |

## Results for one run

All scores are RMSE against the archived NAM forecast for the 00Z 3 November 2026 cycle, on the NAM's own output grid. Lower means closer to the NAM.

**These tests used the NAM's final observation files, which live runs can't.** Live runs use RAP's files instead, and scores with those are still to come. These tests were also scored only to 36 hours, so hours 36 to 84 are unverified.

- The GSI cycle's median error is **0.80×** plain WOOF-NAM from GFS, and it's closer to the NAM on **68 of 69** scores.
- Its median error is **1.16×** our Fortran rerun of the NAM.

![RMSE against the NAM by forecast hour, for six fields](pirate/assets/scores.png)

### Scores at 12 hours

| Field | GSI cycle | Plain WOOF | Fortran rerun | WOOF from NAM analysis |
|---|--:|--:|--:|--:|
| 500 mb height (m) | **4.22** | 5.42 | 3.48 | 0.96 |
| 250 mb height (m) | **5.62** | 7.30 | 5.27 | 1.45 |
| 700 mb humidity (%) | **11.19** | 12.96 | 9.92 | n/a |
| 2 m temperature (K) | **0.84** | 1.08 | 0.64 | 0.36 |
| 2 m dew point (K) | **1.06** | 1.44 | 0.87 | 0.33 |
| 850 mb wind (m/s) | **1.59** | 1.90 | 1.41 | 0.49 |
| 10 m wind (m/s) | **0.83** | 0.96 | 0.69 | 0.32 |
| 250 mb wind (m/s) | **2.66** | 3.14 | 2.60 | 0.52 |
| Composite reflectivity (dBZ) | **11.31** | 12.06 | 10.09 | 8.45 |
| Precipitation (mm) | **3.12** | 4.15 | 3.18 | 0.80 |
| Sea level pressure (Pa) | **46.5** | 57.5 | n/a | 25.2 |

GSI cycle: WOOF-NAM from GFS with three GSI analyses. Plain WOOF: WOOF-NAM from GFS. Fortran rerun: the original Fortran NAM and GSI rebuilt with gfortran on the same substitute inputs, with land seeded from the NAM's archive. WOOF from NAM analysis: WOOF-NAM started from the NAM's own archived analysis, which tests the port itself and isn't something a live system could do.

### Maps at 12 hours

![Maps of reflectivity, sea level pressure and 500 mb height at 12 hours, with differences from the NAM](pirate/assets/maps_f12.png)

Valid 12Z 3 November 2026. Red is higher than the NAM and blue is lower. The noisy pressure differences over the western mountains appear in both WOOF-NAM runs, and sea level pressure is an extrapolated quantity there.

### How long the observations help

The median ratio is the GSI cycle's error relative to a plain GFS start: below 1 means closer to the NAM. Below that, each cell is RMSE for the GSI cycle / plain start.

| Field | 0 h | 6 h | 12 h | 18 h | 24 h | 30 h | 36 h |
|---|--:|--:|--:|--:|--:|--:|--:|
| **Median ratio, all fields** | **0.78** | **0.81** | **0.82** | **0.83** | **0.87** | **0.87** | **0.87** |
| 500 mb height (m) | 4.64 / 5.93 | 4.38 / 6.07 | 4.22 / 5.42 | 4.68 / 5.94 | 4.96 / 6.33 | 5.47 / 6.99 | 6.27 / 8.53 |
| 250 mb height (m) | 7.07 / 6.94 | 5.82 / 6.91 | 5.60 / 7.30 | 6.49 / 8.29 | 7.39 / 9.66 | 8.71 / 11.62 | 10.36 / 13.86 |
| 700 mb humidity (%) | 9.48 / 11.53 | 9.90 / 12.05 | 11.19 / 12.96 | 12.40 / 13.97 | 13.86 / 15.80 | 15.41 / 16.39 | 15.98 / 17.21 |
| 2 m temperature (K) | 0.81 / 1.04 | 0.72 / 1.01 | 0.85 / 1.08 | 0.99 / 1.19 | 0.89 / 1.00 | 0.89 / 1.04 | 1.06 / 1.20 |
| 2 m dew point (K) | 1.20 / 1.92 | 0.98 / 1.50 | 1.06 / 1.44 | 1.15 / 1.40 | 1.21 / 1.42 | 1.15 / 1.35 | 1.20 / 1.37 |
| 850 mb wind (m/s) | 1.09 / 1.60 | 1.39 / 1.63 | 1.59 / 1.90 | 1.73 / 1.99 | 1.70 / 1.89 | 1.85 / 2.10 | 1.97 / 2.25 |

### How to read this

- **This is one case.** Other weather could score differently.
- **Close to the NAM isn't the same as accurate.** These scores don't compare any of the runs with observations. The NAM's own errors count as the truth here, and the NAM also used GSI, which favours the GSI cycle on this measure.
- **The Fortran rerun has an advantage near the surface:** its land was seeded from the same archive it's scored against. A live run couldn't do that.

## What it takes to run

| 84-hour forecast | Time | Cost (OVHcloud) | Status |
|---|---|---|---|
| GSI cycle, two cards (t2-90), the production setup | About 2 h | About CAD 4.60 | Estimated from measured parts; times to be replaced with measured ones |
| GSI cycle, one card (t2-45) | About 3.2 h | About CAD 3.70 | Estimated from measured parts |
| WOOF-NAM from GFS, one card | About 2.2 h | About CAD 2.55 | Scaled from a measured 48-hour run |

Times are for the model and GSI. Fetching inputs, post-processing and upload add roughly an hour. The code is in [`pirate/`](pirate/README.md).

## Credits

The NAM, NMMB and GSI were developed by NOAA/NCEP and its collaborators. WOOF-NAM is by Recast Systems, under the Apache 2.0 licence; its `LICENSE` and `NOTICE` are unchanged here. For how the model itself works, see [WOOF-NAM's README](https://github.com/recastsystems/woof-nam#readme). Input data comes from NOAA's GFS, GEFS, RAP and MRMS. Pirate NAM isn't affiliated with or endorsed by NOAA, NCEP or Recast Systems.
