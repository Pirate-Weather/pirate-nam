#!/bin/bash
# Run one Pirate NAM cycle end to end from the control host, with credentials staying here: inputs and model on the GPU machine,
# then GRIB2, Zarr and upload here. A failed cycle publishes nothing, so the last good forecast stays listed.
# Usage: run_cycle.sh GPU_IP [YYYYMMDDHH]   With no cycle, picks the newest whose inputs are all out.
set -uo pipefail
export GPU_IP=$1; C=${2:-}; export FCST_HOURS=${FCST_HOURS:-84}
K=${NAM_SSH_KEY:?}; SSH="ssh -i $K -o ServerAliveInterval=60 -o StrictHostKeyChecking=accept-new ubuntu@$GPU_IP"; H=$(cd "$(dirname "$0")/.." && pwd)
stop() { echo "CYCLE_FAILED $*"; exit 1; }
at() { date -u -d "${1:0:4}-${1:4:2}-${1:6:2} ${1:8:2}:00 UTC -$2 hours" +"$3"; }   # UTC is required: "12:00 -6" would read as a timezone
ready() { # a cycle is ready when RAP's analysis-hour observations and the older GFS run's last boundary file are out
  local G=$(at $1 6 %Y%m%d%H)
  curl -sfI -m 20 "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.${G:0:8}/${G:8:2}/atmos/gfs.t${G:8:2}z.pgrb2.0p25.f$(printf %03d $((FCST_HOURS + 6))).idx" >/dev/null &&
  curl -sfI -m 20 "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/rap.${1:0:8}/rap.t${1:8:2}z.prepbufr.tm00.nr" >/dev/null
}
if [ -z "$C" ]; then
  now=$(date -u +%Y%m%d)$(printf %02d $(( $(date -u +%-H) / 6 * 6 )))
  for back in 0 6 12 18; do t=$(at $now $back %Y%m%d%H); if ready $t; then C=$t; break; fi; done
  [ -n "$C" ] || stop "no cycle has its inputs yet"
fi
echo "CYCLE $C on $GPU_IP, $FCST_HOURS h, started $(date -u +%H:%M:%S)"
scp -q -i $K $H/cycle/gsi_cycle.py $H/publish/product_fields.py ubuntu@$GPU_IP:/woof/work/gpuda/ || stop "sync"
$SSH "mkdir -p /nam/prod /nam/gpuda" 2>/dev/null; scp -q -i $K $H/prod/fetch_inputs.sh $H/prod/run_on_gpu.sh $H/inputs/gefs_init_to_atmf.py $H/inputs/make_refd3d.py ubuntu@$GPU_IP:/nam/prod/ || stop "sync"
scp -q -i $K $H/cycle/gsi_mem.sh ubuntu@$GPU_IP:/nam/gpuda/ || stop "sync"
STEP=$(mktemp)
$SSH "FCST_HOURS=$FCST_HOURS bash /nam/prod/fetch_inputs.sh $C" 2>&1 | grep --line-buffered -vE "Ubuntu 24|documentation" | tee $STEP; grep -q INPUTS_READY $STEP || stop "inputs"
$SSH "FCST_HOURS=$FCST_HOURS bash /nam/prod/run_on_gpu.sh $C" 2>&1 | grep --line-buffered -vE "Ubuntu 24|documentation" | tee $STEP; grep -q GPU_DONE $STEP || stop "model"
rm -f $STEP
echo "publish $(date -u +%H:%M:%S)"
$H/publish/ship_run.sh $C/fc $C 0 $FCST_HOURS || stop "publish"
echo "CYCLE_DONE $C $(date -u +%H:%M:%S)"
