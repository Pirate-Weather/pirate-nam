#!/bin/bash
# The GPU side of one cycle: boundaries, catch-up with GSI on one card, then the forecast on both cards. Usage: run_on_gpu.sh YYYYMMDDHH
set -uo pipefail
C=$1; HH=${C:8:2}; FCST_HOURS=${FCST_HOURS:-84}; W=/woof/work; R=$W/runs/$C; mkdir -p $R
at() { date -u -d "${C:0:4}-${C:4:2}-${C:6:2} ${HH}:00 UTC -$1 hours" +"$2"; }   # UTC is required: "12:00 -6" would read as a timezone
CA=$(at 6 %Y-%m-%d_%Hz); CB=$(at 0 %Y-%m-%d_%Hz)
NCCL=$(dirname $(find /woof/venv -name libnccl.so.2 | head -1)); export LD_LIBRARY_PATH=$NCCL:${LD_LIBRARY_PATH:-} NAMGPU_GEOG=/woof/WPS_GEOG NAM_CYCLE=$C NAM_HOST=/nam
cd $W; PY=/woof/venv/bin/python
stop() { echo "CYCLE_FAILED $*"; exit 1; }

echo "boundaries $(date -u +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=1 $PY -m namgpu.init.gfs_bocos --case $CB --hours $FCST_HOURS > $R/boco.log 2>&1 || stop "boundaries, see $R/boco.log"

echo "catch-up $(date -u +%H:%M:%S)"
rm -rf $R/catchup $R/init $R/fc
CUDA_VISIBLE_DEVICES=0 GSI_TIMES=04,02,00 GSI_FILTER=0 GSI_CPUS=${GSI_CPUS:-28} GSI_MEM_GB=${GSI_MEM_GB:-70} GSI_HANDOFF=$R/init \
  $PY gpuda/gsi_cycle.py --case $CA --source gfs --hours 6 --no-maps --out $R/catchup > $R/catchup.log 2>&1
grep -E "GSI_MEM|GSI cost|tm0.: patch|handoff|boundaries reloaded|RUN ERROR" $R/catchup.log | cut -c1-120
grep -q "RUN ERROR" $R/catchup.log && stop "catch-up, see $R/catchup.log"
[ -s $R/init/save-receipt.json ] || ls $R/init >/dev/null 2>&1 || stop "no handoff state"

echo "forecast $(date -u +%H:%M:%S)"
NAMGPU_SKIP_TDFI=1 $PY -m namgpu.run_forecast --case $CB --source gfs --hours $FCST_HOURS --no-maps --cards 2 --init-from $R/init --out $R/fc > $R/fc.log 2>&1 || stop "forecast, see $R/fc.log and $R/fc/rank0.log"
grep -E "ranks exit" $R/fc.log | cut -c1-120; ls $R/fc | grep -c wrfout; rm -rf $R/init
echo "GPU_DONE $C $(date -u +%H:%M:%S)"
