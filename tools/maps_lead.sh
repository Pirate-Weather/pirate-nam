#!/bin/bash
# lane N: draw one lead's maps (all 7 products) from a run's history, then the ours|NAM|MRMS montages for that lead.
# usage: CASE=<case> maps_n.sh <run> <lead_h> [montage-dir]
set -u
run=$1; f=$2; mdir=${3:-/work/nam-gpu/runs/N/$1/side-by-side}
case=${CASE:?set CASE=<YYYY-MM-DD_HHz> (no case default in generic tools)}
start="${case:0:10} ${case:11:2}:00 UTC"
cd /work/nam-gpu; source /work/N-nam/env-n.sh; export PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=""
t0=$(date -d "$start + $f hours" -u +%Y-%m-%d_%H:%M:%S); t1=$(date -d "$start + $((f-1)) hours" -u +%Y-%m-%d_%H:%M:%S)
R=runs/N/$run
python -m namgpu.io.maps "$R/wrfout_d01_$t1" "$R/wrfout_d01_$t0" --out $R/maps --frames 1 --mrms cases/$case/raw/mrms > $R/maps_f$f.log 2>&1
tail -1 $R/maps_f$f.log
python /work/N-nam/montage.py $R/maps cases/$case/referee/png $mdir $f --start "${case:0:10}T${case:11:2}" > /dev/null 2>&1; ls $mdir | grep -c "f$(printf %03d $f)_"
