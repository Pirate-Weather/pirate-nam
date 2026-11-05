#!/bin/bash
# Wait for a run on the GPU machine to end, however it ends: a dead process counts, because grep on the log alone waited forever after an OOM kill.
# Usage: wait_run.sh RUN LOGNAME [score]
R=$1; L=$2; K=${NAM_SSH_KEY:?}; I=${GPU_IP:?}
ssh -i $K -o ServerAliveInterval=60 ubuntu@$I "cd /woof/work; while pgrep -f 'out runs/[$(echo $R|cut -c1)]$(echo $R|cut -c2-)( |\$)' >/dev/null; do sleep 30; done; date -u +%H:%M; grep -E 'tm0.: patch|tm0.: filter|GSI_MEM|RUN ERROR|Traceback|end stats' $L | cut -c1-130; tail -2 $L | cut -c1-150; ls runs/$R | grep -c wrfout" 2>&1 | grep -vE "Ubuntu 24|documentation"
[ "${3:-}" = score ] && { rm -rf ${PIRATE_DATA:?}/woof/scored/$R; $(dirname "$0")/woof_score.sh $R 2>&1 | tail -1; }
