
### How to read this

- **This is one case.** Other weather could score differently.
- **Close to the NAM isn't the same as accurate.** These scores don't compare any of the runs with observations. The NAM's own errors count as the truth here, and the NAM also used GSI, which favours the GSI cycle on this measure.
- **The Fortran rerun has an advantage near the surface:** its land was seeded from the same archive it's scored against. A live run couldn't do that.

## What it takes to run

| 48-hour forecast | Time | Cost (OVHcloud) | Status |
|---|---|---|---|
| WOOF-NAM from GFS, one card | 75 min | About CAD 1.45 | Measured, live |
| GSI cycle, one card (t2-45) | About 2.4 h | About CAD 2.80 | Estimated from measured parts |
| GSI cycle, two cards (t2-90) | About 1.5 h | About CAD 3.40 | Estimated from measured parts; not yet run end to end |

Times are for the model and GSI, before post-processing and upload. The code is in [`pirate/`](pirate/README.md).

## Credits

The NAM, NMMB and GSI were developed by NOAA/NCEP and its collaborators. WOOF-NAM is by Recast Systems, under the Apache 2.0 licence; its `LICENSE` and `NOTICE` are unchanged here. Input data comes from NOAA's GFS, GEFS, RAP and MRMS. Pirate NAM isn't affiliated with or endorsed by NOAA, NCEP or Recast Systems.

---

*Everything below is WOOF-NAM's own README, unchanged.*

