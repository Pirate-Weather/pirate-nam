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
