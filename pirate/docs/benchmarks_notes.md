## Full comparison

| | NOAA NAM v4.2 | WOOF-NAM as released | Pirate NAM |
|---|---|---|---|
| **Status** | Retired 3 Nov 2026 | Open source | Experimental |
| **Model** | NMMB, Fortran | Same design, ported to GPU | WOOF-NAM, unchanged |
| **Numerics** | The reference | Single precision, not bit-identical to the NAM | Same as WOOF-NAM |
| **Hardware** | Supercomputer | One or more NVIDIA cards | Two V100S cards, 30 CPUs, 90 GB. One card, 15 CPUs and 45 GB also works, more slowly |
| **Grids** | 12 km parent plus higher-resolution nests | 12 km; 4 km, nests and others available | 12 km only |
| **Forecast length** | 84 h at 12 km | Your choice | 84 h |
| **Start** | Global model atmosphere plus NAM's own land, 6 h before the forecast | GFS, or a NAM analysis when one exists | GFS, 6 h before the forecast |
| **Analyses** | GSI every hour through the catch-up (7 analyses) | None | GSI every 2 hours (3 analyses) |
| **Observation sources** | NAM's dedicated data feed, satellite radiances, radar | None | RAP's hourly observation files and MRMS radar, since the NAM's own files ended with it. RAP's files have no GPS radio occultation. Radiances and lightning left out |
| **Ensemble in GSI** | Global ensemble | None | GEFS, the newest available |
| **Digital filter** | At each catch-up start | At the start | At the start only |
| **Soil and snow** | Carried from cycle to cycle, with observed rainfall | From GFS | From GFS |
| **SST and sea ice** | Refreshed from outside analyses | From GFS | From GFS |
| **Lateral boundaries** | GFS cycle 6 h older than the forecast | GFS cycle 6 h older than the start | GFS cycle 6 h older than the forecast |
