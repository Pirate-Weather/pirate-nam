#!/bin/bash
# The GPU side of one cycle: boundaries, catch-up with GSI on one card, then the forecast on both cards. Usage: run_on_gpu.sh YYYYMMDDHH
set -uo pipefail
C=$1; HH=${C:8:2}; FCST_HOURS=${FCST_HOURS:-84}; W=${WOOF_WORK:-/woof/work}; R=$W/runs/$C; mkdir -p $R
at() { date -u -d "${C:0:4}-${C:4:2}-${C:6:2} ${HH}:00 UTC -$1 hours" +"$2"; }   # UTC is required: "12:00 -6" would read as a timezone
CA=$(at 6 %Y-%m-%d_%Hz); CB=$(at 0 %Y-%m-%d_%Hz)
VENV=${WOOF_VENV:-/woof/venv}; NCCL=$(dirname $(find $VENV -name libnccl.so.2 | head -1)); export LD_LIBRARY_PATH=$NCCL:${LD_LIBRARY_PATH:-} NAMGPU_GEOG=/woof/WPS_GEOG NAM_CYCLE=$C NAM_HOST=/nam
cd $W; PY=$VENV/bin/python
stop() { echo "CYCLE_FAILED $*"; exit 1; }

echo "boundaries $(date -u +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=1 $PY -m namgpu.init.gfs_bocos --case $CB --hours $FCST_HOURS > $R/boco.log 2>&1 || stop "boundaries, see $R/boco.log"

echo "catch-up $(date -u +%H:%M:%S)"
# Fresh GSI state every time, so a rerun doesn't start from a half-used restart file or an earlier attempt's bias file.
PDY=${C:0:8}; COM=/nam/com/nam/v4.2/nam.$PDY; GES=/nam/com/nam/v4.2/nwges/nam.$PDY
docker ps -aq --filter name=gsi_nam_ | xargs -r docker rm -f >/dev/null 2>&1; rm -rf /dev/shm/gsi_cycle_nam
rm -f $GES/nam.t${HH}z.nmm_b_restart_nemsio*; cp /nam/gpuda/restart_template.nemsio $GES/nam.t${HH}z.nmm_b_restart_nemsio.tm06
for t in 05 04 03 02 01 00; do rm -f $COM/nam.t${HH}z.satbias.tm$t $COM/nam.t${HH}z.satbias_pc.tm$t; done
cp /nam/gpuda/satbias.tm06 $COM/nam.t${HH}z.satbias.tm06; cp /nam/gpuda/satbias_pc.tm06 $COM/nam.t${HH}z.satbias_pc.tm06
rm -rf $R/catchup $R/init $R/fc
CUDA_VISIBLE_DEVICES=0 GSI_TIMES=04,02,00 GSI_FILTER=0 GSI_CPUS=${GSI_CPUS:-28} GSI_MEM_GB=${GSI_MEM_GB:-70} GSI_HANDOFF=$R/init \
  $PY gpuda/gsi_cycle.py --case $CA --source gfs --hours 6 --no-maps --out $R/catchup > $R/catchup.log 2>&1
grep -E "GSI_MEM|GSI cost|tm0.: patch|handoff|boundaries reloaded|RUN ERROR" $R/catchup.log | cut -c1-120
grep -q "RUN ERROR" $R/catchup.log && stop "catch-up, see $R/catchup.log"
[ -s $R/init/save-receipt.json ] || ls $R/init >/dev/null 2>&1 || stop "no handoff state"

echo "forecast $(date -u +%H:%M:%S)"
# --filter off: the saved state is an analysis, and a filter here would throw away its rain and ice fractions.
$PY -m namgpu.run_forecast --case $CB --source gfs --filter off --hours $FCST_HOURS --no-maps --cards 2 --init-from $R/init --out $R/fc > $R/fc.log 2>&1 || stop "forecast, see $R/fc.log and $R/fc/rank0.log"
grep -E "ranks exit" $R/fc.log | cut -c1-120; ls $R/fc | grep -c wrfout; rm -rf $R/init
echo "GPU_DONE $C $(date -u +%H:%M:%S)"
