<p align="center"><img src="pirate/assets/logo.svg" width="130" alt="Pirate NAM logo: a storm cloud in a pirate hat and eye patch, above crossed lightning bolts"></p>

# Pirate NAM

Keeping the retired North American Mesoscale model sailing, on one GPU machine. Pirate Weather was founded on the idea of keeping legacy weather alive, so this is exactly what we like to do!

> **Experimental.** Not an official forecast, and not a NOAA or NCEP product. Runs may stop or change without notice, and there's no regular schedule yet. Don't use it for decisions that affect safety.

> **Support this project:** running a model like this costs real money. Help keep it going by [sponsoring me on GitHub](https://github.com/sponsors/alexander0042) or [signing up for Pirate Weather](https://pirate-weather.apiable.io/).

**[Map viewer](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/index.html) · [Data access (GRIB2 and Zarr)](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/data.html) · [Benchmarks](BENCHMARKS.md)**

## Why it's built this way

NOAA retired the NAM on 3 November 2026. **The goal here is to produce the forecast the NAM would have produced,** so Pirate NAM keeps as much of the legacy pipeline as it can:

- **Same model design:** the NMMB model and the NAM's physics, through the GPU port [WOOF-NAM](https://github.com/recastsystems/woof-nam).
- **Same analysis program:** NOAA's own GSI, unmodified, with the NAM's configuration.
- **Same structure:** a 6-hour catch-up before each forecast, on the NAM's 12 km grid, out to 84 hours.

Something changes only when an input no longer exists, or when testing shows it makes little difference to how closely the result matches the NAM. That closeness is the yardstick, not accuracy against observations.

## How a cycle works

For a 00Z forecast:

| Time | Step | Runs on |
|---|---|---|
| 18Z | Start from GFS | GPU |
| 20Z, 22Z, 00Z | GSI adds observations | CPU |
| 00Z to +84 h | Forecast | GPU |

Lateral boundaries come from the GFS cycle 6 hours older than the forecast.

## How close it gets

Not quite the same as NOAA's, due to some missing underlying files; however, pretty close! See [BENCHMARKS.md](BENCHMARKS.md) for scores, maps, the test behind each decision and the full comparison with the NAM. Worth noting, this implementation is around 20% closer than plain WOOF-NAM, due to the data assimilatuion step.

## Outputs

Hourly fields from 0 to 84 hours, on the NAM's 12 km rotated latitude-longitude grid (954 × 835). The newest 8 cycles are kept, and [runs.json](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/runs.json) lists what's available. The [data access page](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/data.html) lists every field.

| Product | What it's for | Path |
|---|---|---|
| **Map viewer** | Browsing a forecast | [index.html](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/index.html) |
| **GRIB2** | Full field set, for forecast tools | `<YYYYMMDDHH>/woof/grib/woofnam.t<HH>z.f<FFF>.grib2`, 85 files of about 43 MB |
| **WMS** | Map layers for GIS tools and web maps, newest run only | `https://nam-wms.pirateweather.net/datasets/pirate-nam/wms` |
| **Zarr v3** | 10 key fields for Python, with no server or download step | `<YYYYMMDDHH>/woof/zarr`, about 680 MB |

Zarr and GRIB2 paths are under `https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/`.

```python
import xarray as xr
ds = xr.open_zarr("https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/2026110518/woof/zarr")
ds.t2m.sel(fhr=24)
```

### WMS

The [WMS 1.3.0 service](https://nam-wms.pirateweather.net/datasets/pirate-nam/wms?service=WMS&version=1.3.0&request=GetCapabilities) serves the same 10 fields as the Zarr store: `t2m`, `d2m`, `mslp`, `wind10`, `wind250`, `apcp`, `refc`, `pwat`, `z500` and `t850`. Here's 2 m temperature for one tile, valid 00Z on 6 November:

```
https://nam-wms.pirateweather.net/datasets/pirate-nam/wms?service=WMS&version=1.3.0&request=GetMap&format=image/png&transparent=true&styles=raster/default&crs=EPSG:3857&width=256&height=256&layers=t2m&colorscalerange=-40,40&autoscale=false&time=2026-11-06T00:00:00Z&bbox=-10018754,3757033,-7514065,6261721
```

- **Fields are regridded:** the NAM's grid contains the North Pole, which the WMS software draws wrongly, so layers come from a 0.1° lat-lon copy using the nearest model point. Use the Zarr or GRIB2 data for native-grid values.
- **Newest run only:** a new run appears within about 20 minutes of being published, and older times stop being available.
- **Use EPSG:3857:** if you use EPSG:4326, give the bounding box as `lon,lat`, not the `lat,lon` order WMS 1.3.0 specifies. The other order returns an empty response.
- **Limits:** images can be up to 2048 pixels on each side, and requests are limited to 40 per second per address, with short bursts allowed.

## Main differences from the NAM

| | NOAA NAM | Pirate NAM |
|---|---|---|
| **Hardware** | Supercomputer | Two GPU cards |
| **Grids** | 12 km plus nests | 12 km only |
| **Start** | Global model plus NAM's own land | GFS, including land |
| **Analyses** | 7 in the catch-up | 3 in the catch-up |
| **Observations** | NAM's own data feed, radiances, radar | RAP's hourly files and MRMS radar |

## What it takes to run

The first live run took **under 3 hours and about CAD 6.50** on a two-card OVHcloud t2-90 machine already set up, from fetching inputs to publishing. The 84-hour forecast itself took 74 minutes. The code is in [`pirate/`](pirate/README.md).

## Credits

The NAM, NMMB and GSI were developed by NOAA/NCEP and its collaborators. WOOF-NAM is by Recast Systems, under the Apache 2.0 licence; its `LICENSE` and `NOTICE` are unchanged here. For how the model itself works, see [WOOF-NAM's README](https://github.com/recastsystems/woof-nam#readme). Input data comes from NOAA's GFS, GEFS, RAP and MRMS. Pirate NAM isn't affiliated with or endorsed by NOAA, NCEP or Recast Systems.
