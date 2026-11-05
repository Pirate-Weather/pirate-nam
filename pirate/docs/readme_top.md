<p align="center"><img src="pirate/assets/logo.svg" width="130" alt="Pirate NAM logo: a storm cloud in a pirate hat and eye patch, above crossed lightning bolts"></p>

# Pirate NAM

Keeping the retired North American Mesoscale model sailing, on one GPU machine.

> **Experimental.** Not an official forecast, and not a NOAA or NCEP product. Runs may stop or change without notice. Don't use it for decisions that affect safety.

**[Map viewer](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/index.html) · [Data access (GRIB2 and Zarr)](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/data.html)**

## What it is

NOAA retired the NAM on 3 November 2026. Pirate NAM runs the same model design on a much smaller machine by combining two existing pieces of work: [WOOF-NAM](https://github.com/recastsystems/woof-nam), an independent GPU port of the NAM's NMMB model, and NOAA's GSI, the Fortran program that blends observations into a model state. Its goal is to stay as close as practical to what the NAM would have produced.

**Where this stands:** the forecasts on the map viewer today use WOOF-NAM started directly from GFS, without observations. The cycle with GSI described below has only been tested on one past case, and isn't running live yet.

## How the tested cycle works

Like the NAM, it runs a 6-hour catch-up before the forecast, so the model has absorbed recent observations by the time the forecast starts. For a 00Z forecast:

| Time | Step | Runs on |
|---|---|---|
| 18Z | Start from GFS | GPU |
| 20Z | GSI adds observations | CPU |
| 22Z | GSI adds observations | CPU |
| 00Z | GSI adds observations | CPU |
| 00Z to +48 h | Forecast | GPU |

WOOF-NAM forecasts on the GPU between analyses, with the model state kept on the card. At each analysis, its fields are written into a real NMMB restart file, GSI runs on that file unchanged, and whatever GSI changed goes back onto the card.

## Compared with NOAA's NAM and WOOF-NAM

| | NOAA NAM v4.2 | WOOF-NAM as released | Pirate NAM (tested cycle) |
|---|---|---|---|
| **Status** | Retired 3 Nov 2026 | Open source | Experimental; GFS-start version live |
| **Model** | NMMB, Fortran | Same design, ported to GPU | WOOF-NAM, unchanged |
| **Numerics** | The reference | Single precision, not bit-identical to the NAM | Same as WOOF-NAM |
| **Hardware** | Supercomputer | One or more NVIDIA cards | One V100S card, 15 CPUs, 45 GB |
| **Grids** | 12 km parent plus higher-resolution nests | 12 km; 4 km, nests and others available | 12 km only |
| **Forecast length** | 84 h at 12 km | Your choice | 48 h |
| **Start** | Global model atmosphere plus NAM's own land, 6 h before the forecast | GFS, or a NAM analysis when one exists | GFS, 6 h before the forecast |
| **Observations** | GSI every hour through the catch-up (7 analyses) | None | GSI every 2 hours (3 analyses): more made no measured difference |
| **Observation sources** | NAM's dedicated data feed, satellite radiances, radar | None | RAP observation files and MRMS radar. Radiances and lightning left out: no measured benefit |
| **Ensemble in GSI** | Global ensemble | None | GEFS |
| **Digital filter** | At each catch-up start | At the start | At the start only: filtering each analysis made no measured improvement |
| **Soil and snow** | Carried from cycle to cycle, with observed rainfall | From GFS | From GFS. Seeding from the NAM archive improved scores by only 0.4% in testing |
| **SST and sea ice** | Refreshed from outside analyses | From GFS | From GFS |
| **Lateral boundaries** | GFS cycle 6 h older than the forecast | GFS cycle 6 h older than the start | GFS cycle 6 h older than the forecast |

## Results for one run

All scores are RMSE against the archived NAM forecast for the 00Z 3 November 2026 cycle, on the NAM's own output grid. Lower means closer to the NAM.

