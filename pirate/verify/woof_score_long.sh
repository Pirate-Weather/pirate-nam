#!/bin/bash
# Score a run at the 6-hourly leads out to 36 h, from a 2026-11-03 00Z forecast. Usage: woof_score_long.sh RUN (GPU_IP optional)
K=${NAM_SSH_KEY:?}; I=${GPU_IP:?}; R=$1; D=${PIRATE_DATA:?}/woof/scored/${R}_long; mkdir -p $D
ssh -i $K ubuntu@$I "cd /woof/work/runs/$R && mkdir -p npzl && for p in 00:2026-11-03_00 06:2026-11-03_06 12:2026-11-03_12 18:2026-11-03_18 24:2026-11-04_00 30:2026-11-04_06 36:2026-11-04_12; do h=\${p%%:*}; f=\${p#*:}; /woof/venv/bin/python /woof/woof_fields.py wrfout_d01_\$f:00:00 npzl/f\$h.npz & done; wait; cp forecast-receipt.json npzl/" 2>&1 | grep -vE "Ubuntu 24|documentation"
rsync -a -e "ssh -i $K" ubuntu@$I:/woof/work/runs/$R/npzl/ $D/
# Python with xarray, cfgrib, eccodes and wgrib2 on PATH expected here.; export OMP_NUM_THREADS=2
python -W ignore $(dirname "$0")/woof_compare.py $D ${PIRATE_DATA:?}/compare/f36/arc $D/series.csv 0 6 12 18 24 30 36 2>&1 | tail -1 | cut -c1-120
