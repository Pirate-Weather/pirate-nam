#!/bin/bash
# Turn a finished WOOF-NAM run on the GPU machine into published products. Fields, GRIB and Zarr are all made there, so the control
# host needs no compute. Credentials stay on the control host: upload_run.py hands the GPU machine signed URLs instead.
# Usage: ship_run.sh RUN CYCLE FIRST_HOUR HOURS [--no-upload]
# FIRST_HOUR is the run hour of the forecast's 0 h: 0 for a direct start, 6 for a run with a catch-up window.
set -euo pipefail
R=$1; CYCLE=$2; FIRST=$3; HOURS=$4; UP=${5:-}
K=${NAM_SSH_KEY:?}; I=${GPU_IP:?}; D=/woof/work/runs/$R
SSH="ssh -i $K -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -o LogLevel=ERROR ubuntu@$I"
$SSH "cd /woof/work/runs/$R && rm -rf pf && mkdir pf && /woof/venv/bin/python - <<PY
import datetime as dt, subprocess
from concurrent.futures import ThreadPoolExecutor
c = dt.datetime.strptime('$CYCLE', '%Y%m%d%H')
def one(h):
    f = (c + dt.timedelta(hours=h)).strftime('wrfout_d01_%Y-%m-%d_%H:00:00')
    subprocess.run(['/woof/venv/bin/python', '/woof/work/gpuda/product_fields.py', f, f'pf/f{h:02d}.npz'], check=True)
with ThreadPoolExecutor(6) as ex: list(ex.map(one, range($HOURS + 1)))
print('fields done')
PY"
$SSH "set -o pipefail; rm -rf $D/out; docker run --rm --user 1000:1000 -v /woof/work:/woof/work -w $D pirate-nam-publish python /woof/work/gpuda/publish_product.py pf out $CYCLE --no-upload | tail -1"
[ "$UP" = "--no-upload" ] && exit 0
python "$(dirname "$0")/upload_run.py" $D/out $CYCLE
