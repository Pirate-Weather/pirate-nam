# Pirate NAM code

This folder holds everything added to WOOF-NAM. Upstream's `namgpu/` is left unchanged, so `git merge upstream/main` stays simple.

| Folder | What it does |
|---|---|
| `prod/` | Runs one live cycle end to end: inputs, catch-up with GSI on one card, forecast on both, then publishing |
| `cycle/` | Cycles WOOF-NAM with NOAA's GSI (`gsi_cycle.py`), and moves fields between the model and NEMSIO restart files (`nemsio_fields.f90`) |
| `publish/` | Turns history files into GRIB2 and Zarr, uploads them, and updates the run list |
| `verify/` | Scores a run against the archived NAM and builds the README's results |
| `inputs/` | Substitute inputs: GEFS ensemble for GSI, MRMS radar |
| `build/` | Builds GSI and its libraries in a container, and the job harness |
| `infra/` | Creates and sets up an OVHcloud GPU machine |
| `viewer/` | The static map viewer and data page |

**This isn't turnkey yet.** `prod/run_cycle.sh` runs a live cycle, but it's new and assumes a machine already set up by `infra/`. Its first live run is still in progress.

**Tested against upstream commit `93d85e3`.** This fork's `main` is newer, and upstream has changed `driver.py` and `bnd_read_bc.py` since, both of which the cycle depends on. Retest before relying on a newer version.

**Not yet run as committed:** the stricter error handling in `nemsio_fields.f90`, and these scripts' switch from hard-coded paths and IDs to environment variables.

Scripts read their settings from the environment: `OVH_CONFIG`, `OVH_PROJECT`, `OVH_USER`, `OVH_SSH_KEY_NAME`, `NAM_SSH_KEY`, `GPU_IP`, `GPU_SOURCE_IP`, `PIRATE_DATA` and `PIRATE_VIEWER`. No credentials belong on the GPU machines: uploads run from a separate host.
