# Status

## Paused on 9 November 2026

Daily runs and the WMS server are stopped. Everything already published stays in the bucket. The last run is
2026110900, and [runs.json](https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/runs.json) lists what's available.

| Piece | State | To resume |
|---|---|---|
| Daily runs (`pirate-nam-control` container on the control host) | Stopped with `docker stop` | `docker start pirate-nam-control` |
| WMS server (`nam-wms`, d2-8 in BHS5) | Shelved; the snapshot `nam-wms-shelved` (5.6 GB) is all that's billed | Unshelve it, then check its public IP still matches the `nam-wms.pirateweather.net` DNS record |
| GPU machines (`nam-auto-<cycle>`) | None running | Created and deleted by the daily runs |

The daily-run code (`pirate/control/daily.py`, the upload scripts and both Dockerfiles) is on the `auto-daily` branch, not
`main` yet.

## Daily runs, 6 to 9 November

Each day's 00Z cycle ran on a throwaway t2-90 machine (2 GPU cards) in DE1, created from the `nam-auto-image`
snapshot.

| Cycle | Result | What happened |
|---|---|---|
| 2026110600 | Published (manual rerun) | Failed twice on OVH API errors, then a machine took about 70 min to delete; the rerun built in 7 min |
| 2026110700 | Published | Attempt 1: a GPU card reported `CUDA_ERROR_ECC_UNCORRECTABLE` while making boundaries. Attempt 2, on another machine, ran clean |
| 2026110800 | **Not published** | One machine was still building after 90 min; the next stopped in the radar step and hit the 6-hour limit; the third failed the SSH and GPU check |
| 2026110900 | Published | Clean on the first attempt |

Once a machine is up, a cycle is steady:

| Step | Time |
|---|---|
| Inputs (GFS, observations, radar, 30-member ensemble) | about 30 min |
| Boundaries and catch-up, three GSI analyses at about 12 min each, peak about 39.5 GB of 70 GB | about 50 min |
| 84 h forecast on 2 cards | about 72 min |
| Publish (952 objects, 4.3 GB) | about 11 min |

## Known issues

- **Machine builds are slow and vary a lot.** From the 67 GB snapshot (800 GB minimum disk) they took 7, 49, 70, more
  than 90, 87, 34 and 41 min. The daily runs wait up to 90 min (`BUILD_WAIT_MINUTES`).
- **The radar step has no timeout.** `prod/fetch_inputs.sh` runs `make_refd3d.py` with nothing to stop it, and on
  8 November it hung until the 6-hour limit. The script already carries on without radar when the step fails (GSI
  skips its cloud analysis), so a timeout would turn a hang into a run without radar instead of a lost cycle. Why it
  hung isn't known: the machine was deleted, and only the start of the log was saved.
- **GPU hosts can be faulty.** One host's card had an uncorrectable memory (ECC) error. A second attempt on a new
  machine dealt with it.
- **The OVH API often fails passing calls.** About 1 call in 12 failed at a 15 s spacing; `daily.py` retries them.

## Next steps when resuming

1. Add a timeout to the radar step.
2. Make builds faster: start from OVH's stock `Ubuntu 24.04 - NVIDIA` image and fetch the static files from a private
   bucket, or keep the snapshot and trim it. Of the 72 GB on disk, about 25 GB is `nam.hold`, 11 GB `/nam/gpuda`,
   5.5 GB NAM fix files, 4.1 GB the current venv and 3.5 GB `WPS_GEOG`. An old venv and repo (4.2 GB) and
   `~/.cache` (2.4 GB) look removable.
3. Merge `auto-daily` into `main`.
