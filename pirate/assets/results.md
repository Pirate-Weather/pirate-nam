- The GSI cycle's median error is **0.80×** plain WOOF-NAM from GFS, and it's closer to the NAM on **68 of 69** scores.
- Its median error is **1.16×** our Fortran rerun of the NAM.

![RMSE against the NAM by forecast hour, for six fields](pirate/assets/scores.png)

### Scores at 12 hours

| Field | GSI cycle | Plain WOOF | Fortran rerun | WOOF from NAM analysis |
|---|--:|--:|--:|--:|
| 500 mb height (m) | **4.22** | 5.43 | 3.48 | 0.96 |
| 250 mb height (m) | **5.62** | 7.30 | 5.27 | 1.45 |
| 700 mb humidity (%) | **11.19** | 13.01 | 9.92 | n/a |
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
