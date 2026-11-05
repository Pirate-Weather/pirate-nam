<p align="center"><img src="pirate/assets/logo.svg" width="130" alt="Pirate NAM logo: a storm cloud in a pirate hat and eye patch, above crossed lightning bolts"></p>

# Pirate NAM

Keeping the retired North American Mesoscale model sailing, on one GPU machine.

> **Experimental.** Not an official forecast, and not a NOAA or NCEP product. Runs may stop or change without notice. Don't use it for decisions that affect safety.

**[Map viewer](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/index.html) · [Data access (GRIB2 and Zarr)](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/data.html)**

## What it is

NOAA retired the NAM on 3 November 2026. Pirate NAM runs the same model design on a much smaller machine by combining two existing pieces of work: [WOOF-NAM](https://github.com/recastsystems/woof-nam), an independent GPU port of the NAM's NMMB model, and NOAA's GSI, the Fortran program that blends observations into a model state. Its goal is to stay as close as practical to what the NAM would have produced.

**Where this stands:** Pirate NAM is being set up to run 84-hour forecasts, the same length as the NAM's 12 km forecast, with GSI adding observations. That setup hasn't completed a live run yet. The forecast on the map viewer today is a 48-hour run of WOOF-NAM started directly from GFS, without observations.

## How the tested cycle works

Like the NAM, it runs a 6-hour catch-up before the forecast, so the model has absorbed recent observations by the time the forecast starts. For a 00Z forecast:

| Time | Step | Runs on |
|---|---|---|
| 18Z | Start from GFS | GPU |
| 20Z | GSI adds observations | CPU |
| 22Z | GSI adds observations | CPU |
| 00Z | GSI adds observations | CPU |
| 00Z to +84 h | Forecast | GPU |

WOOF-NAM forecasts on the GPU between analyses, with the model state kept on the card. At each analysis, its fields are written into a real NMMB restart file, GSI runs on that file unchanged, and whatever GSI changed goes back onto the card.

## Compared with NOAA's NAM and WOOF-NAM

| | NOAA NAM v4.2 | WOOF-NAM as released | Pirate NAM (tested cycle) |
|---|---|---|---|
| **Status** | Retired 3 Nov 2026 | Open source | Experimental; GFS-start version live |
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

**These tests used the NAM's final observation files, which live runs can't.** Live runs use RAP's files instead, and scores with those are still to come. These tests were also scored only to 36 hours, so the 84-hour forecast is unverified beyond that.

