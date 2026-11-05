#!/bin/bash
# Run one production GSI analysis under the t2-45 limits and record memory. Usage: gsi_mem.sh TMMARK MEM_GB CPUS
# Anonymous memory is what can't be evicted; page cache of the 9 GB restart file is reclaimable, so it doesn't count against the limit.
TM=$1; MEM=$2; CPUS=$3; HOST=${NAM_HOST:-/nam}; NAME=gsi_$(basename $HOST)_$TM
docker rm -f $NAME >/dev/null 2>&1
rm -rf $HOST/tmpnwprd/nam_${NAM_CYCLE:8:2}_${TM}_prod
T0=$(date +%s)
docker run -d --name $NAME --user 1000:1000 --ipc=host --ulimit stack=-1:-1 --memory=${MEM}g --memory-swap=${MEM}g --cpus=$CPUS -e HOME=/nam/home -v $HOST:/nam namhar bash /nam/harness/run_job.sh JNAM_ANALYSIS ${NAM_CYCLE:?} $TM NAM_NP=$CPUS NAM_NENS=${NAM_NENS:-31} >/dev/null
ID=$(docker inspect -f '{{.Id}}' $NAME); C=/sys/fs/cgroup/system.slice/docker-$ID.scope; PEAK=0
while [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ]; do
  A=$(awk '$1=="anon"{print int($2/1048576)}' $C/memory.stat 2>/dev/null); [ -n "$A" ] && [ "$A" -gt "$PEAK" ] && PEAK=$A; sleep 2
done
RC=$(docker inspect -f '{{.State.ExitCode}}' $NAME); OOM=$(docker inspect -f '{{.State.OOMKilled}}' $NAME)
echo "GSI_MEM tm=$TM limit=${MEM}g cpus=$CPUS rc=$RC oom=$OOM wall=$(( $(date +%s)-T0 ))s peak_anon_mb=$PEAK"
docker rm $NAME >/dev/null
