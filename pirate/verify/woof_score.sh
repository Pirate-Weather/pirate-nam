#!/bin/bash
# Convert one WOOF-NAM run on the GPU instance and score it here against the archive.
# Usage: woof_score.sh RUN [YYYY-MM-DD] [ARC_DIR]; defaults are the 2026-11-03 00Z test cycle.
K=${NAM_SSH_KEY:?}; I=${GPU_IP:?}; R=$1; DAY=${2:-2026-11-03}; ARC=${3:-${PIRATE_DATA:?}/compare/f36/arc}; D=${PIRATE_DATA:?}/woof/scored/$R; mkdir -p $D
ssh -i $K ubuntu@$I "cd /woof/work/runs/$R && mkdir -p npz && for h in 00 03 06 09 12; do /woof/venv/bin/python /woof/woof_fields.py wrfout_d01_${DAY}_\$h:00:00 npz/f\$h.npz & done; wait; cp forecast-receipt.json npz/ 2>/dev/null; true" || exit 1
rsync -a -e "ssh -i $K" ubuntu@$I:/woof/work/runs/$R/npz/ $D/
# Python with xarray, cfgrib, eccodes and wgrib2 on PATH expected here.; export OMP_NUM_THREADS=2
python -W ignore $(dirname "$0")/woof_compare.py $D $ARC $D/series.csv
