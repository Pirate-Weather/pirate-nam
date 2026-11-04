# Evidence: 2025-03-14 12Z, MRMS | WOOF-NAM | archived NAM

One forecast from a fresh clone and wheel install: `woof-nam run --case 2025-03-14_12z --source native
--hours 24` (the NAM 12Z analysis start, the digital filter, every physics scheme on) on one RTX PRO 6000.
The sheets are `woof-nam side-by-side` output (MRMS observed, WOOF-NAM, the archived operational NAM 12 km
forecast of the same cycle) on one map frame and colour scale per product; the single maps are `woof-nam maps`.

| Lead | Sheets | Maps |
|---|---|---|
| +12 h | f012_refc_mrms_woofnam_nam.png, f012_t2_woofnam_nam.png, f012_precip_total_woofnam_nam.png | f012_d2m_woofnam.png |
| +24 h | f024_refc_mrms_woofnam_nam.png, f024_t2_woofnam_nam.png, f024_precip_total_woofnam_nam.png | f024_d2m_woofnam.png, f024_qpf1h_woofnam.png |

- refc: composite reflectivity (MRMS | WOOF-NAM | NAM).
- t2: 2 m temperature; precip_total: precipitation since the start (WOOF-NAM | NAM; MRMS has neither).
- d2m: 2 m dewpoint; qpf1h: 1 h precipitation.

At +24 h, like the NAM, WOOF-NAM wraps the comma head over the upper Midwest and puts the trailing line west
of the observed one, and neither model has MRMS's second band across Mississippi and Alabama.
