# Case 2025-03-14 12Z: the 14-16 March 2025 severe outbreak

## Why this day

A large, well-observed severe outbreak across the Mississippi and Tennessee valleys and the
Deep South: the evening and night of 14 March (Missouri, Arkansas, Illinois) and the day of
15 March (Mississippi, Alabama), with long-track tornadoes, plus a high-wind and blowing-dust
event over the southern and central Plains on the 14th. MRMS, the NEXRAD network, ASOS and the
NAM itself all ran normally, so every referee exists. The NAM 12Z cycle of 14 March, run to
48 h, covers both days.

## Run

- Initial time 2025-03-14 12:00 UTC, forecast 48 h (to 2025-03-16 12:00 UTC), 12 km NAM parent
  (954 x 835 x 60, dt 25 s, 6,912 steps), all physics on (Ferrier-Aligo, BMJ, MYJ, JSFC, Noah,
  NMMB RRTM, GWD), TDFI on, hourly history.
- Command: `python tests/smoke_full.py --case 2025-03-14_12z --hours 48 --out runs/2025-03-14_12z`

## Inputs (raw/, fetched by `python -m namgpu.init.fetch_case --cycle 2025031412 --hours 48`)

| what | files | source | used for |
|---|---|---|---|
| NAM native B-grid analysis | `raw/nam/bgrd3d00.grib2` (450 MB): 954 x 835, GDT 3.32769, 60 hybrid levels of TMP, SPFH, UGRD, VGRD, PRES, HGT, DZDT, VVEL, TKE, CLMR, CICE, RWMR, SNMR, FICE, FRAIN, RIME, TCOND, plus soil TSOIL/SOILW (4 layers), LAND, VGTYP, SOTYP, VEG, ALBDO, SFCR, SNOD, WEASD, ICEC, CNWAT, NLAT/ELON, LMH/LMV | `s3://noaa-nam-pds/nam.20250314/nam.t12z.bgrd3d00.tm00.grib2` | initial state (no vertical interpolation needed: same grid, same levels) |
| NAM native surface, 0-48 h every 3 h | `raw/nam/bgrdsfFF.grib2` (17 x ~48 MB) | `nam.t12z.bgrdsfFF.tm00.grib2` | native-grid surface referee |
| NAM 218 forecast, same cycle | `raw/nam/awphysFF.grib2` every 3 h (17 x ~60 MB) and `awphysFF.sub.grib2` hourly to 36 h (REFC, APCP, 2 m T/Td, 10 m wind, MSLP) | `nam.t12z.awphysFF.tm00.grib2` | picture referee: what the operational NAM forecast |
| GFS 0.25 deg boundaries | `raw/gfs/gfs.t06z.fFFF.grib2`, f006-f054 every 3 h (17 x ~270 MB, 418 records each: HGT/TMP/SPFH/RH/UGRD/VGRD/CLMR/ICMR/RWMR/SNMR/GRLE/O3MR on pressure levels + surface) | `s3://noaa-gfs-bdp-pds/gfs.20250314/06/atmos/` | lateral boundaries every 3 h (the 6-h-old GFS, as operational NAM) |
| MRMS composite reflectivity | `raw/mrms/mrms_refc_YYYYMMDDHH.grib2`, the file nearest each hour 12Z 14th - 12Z 16th (49) | `s3://noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/` | truth |

Total about 7 GB. Raw input is regenerable by the fetch command and follows the no-hoarding
rule: delete `raw/` once the run's sheets and evidence are kept.

## Referees and how the run is judged

- Maps, hour by hour, side by side: MRMS | nam-gpu | archived NAM (composite reflectivity),
  plus 2 m temperature and dewpoint, 10 m wind, MSLP and accumulated precipitation beside the
  NAM 218 forecast. Look at every sheet and the time sequence: storm placement, timing, mode
  (discrete cells vs lines), the dryline and cold front, the low-level jet, the Plains wind and
  dust-day gradients.
- Scores beside the pictures, not instead of them.
- Not a bitwise comparison (project rule, 2026-11-04).

## Notes

- The archived NAM is a forecast from the NAM's own GSI/NDAS analysis; our initial state is
  that same analysis on its native grid, so the two runs start from the same state. Our
  boundaries come from GFS 0.25 deg pressure-level output, while NAM's came from GFS spectral
  coefficients through NPS: a declared difference in the boundaries.
- GWD orography statistics and snow-free/max-snow albedo are not in the analysis file; unit
  init-gwd-orog and init-static-missing supply them.
