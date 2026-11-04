#!/bin/bash
# NCEP fix files the NMMB run directory carries (exnam_fcst.sh links them into the run dir; the
# radiation init opens them by name from the working directory). Public NOAA data, fetched from
# the NOAA global fix bucket (fix/am/20220805). Usage: tools/fetch_fix.sh [dest]   (default namgpu/data/fix/, where the package ships them)
set -euo pipefail
dest=${1:-$(dirname "$0")/../namgpu/data/fix}
mkdir -p "$dest"
base=https://noaa-nws-global-pds.s3.amazonaws.com/fix/am/20220805
get() { [ -s "$dest/$2" ] || curl -sf --retry 3 --max-time 120 -o "$dest/$2" "$base/$1"; echo "fix: $2"; }
get global_o3clim.txt                          global_o3clim.txt
get global_solarconstant_noaa_an.txt           solarconstant_noaa_an.txt
get global_solarconstant_noaa_a0.txt           solarconstant_noaa_a0.txt
get global_sfc_emissivity_idx.txt              sfc_emissivity_idx.txt
get global_co2historicaldata_glob.txt          co2historicaldata_glob.txt
get co2monthlycyc.txt                          co2monthlycyc.txt
for y in $(seq 2015 2024); do get co2dat_4a/global_co2historicaldata_$y.txt co2historicaldata_$y.txt; done
for y in 2025 2026; do get co2dat_4a/global_co2historicaldata_$y.txt_proj co2historicaldata_$y.txt; done
