#!/bin/bash
# Turn a finished WOOF-NAM run on the GPU machine into published products. Fields are made there; GRIB, Zarr and the upload
# happen here, so credentials never reach that machine. Usage: ship_run.sh RUN CYCLE FIRST_HOUR HOURS [--no-upload]
# FIRST_HOUR is the run hour of the forecast's 0 h: 0 for a direct start, 6 for a run with a catch-up window.
set -euo pipefail
R=$1; CYCLE=$2; FIRST=$3; HOURS=$4; UP=${5:-}
K=${NAM_SSH_KEY:?}; I=${GPU_IP:?}; W=${PIRATE_DATA:?}/woof/product/$CYCLE; rm -rf $W; mkdir -p $W/npz
ssh -i $K ubuntu@$I "cd /woof/work/runs/$R && rm -rf pf && mkdir pf && /woof/venv/bin/python - <<PY
import datetime as dt, subprocess
from concurrent.futures import ThreadPoolExecutor
c = dt.datetime.strptime('$CYCLE', '%Y%m%d%H')
def one(h):
    f = (c + dt.timedelta(hours=h)).strftime('wrfout_d01_%Y-%m-%d_%H:00:00')
    subprocess.run(['/woof/venv/bin/python', '/woof/work/gpuda/product_fields.py', f, f'pf/f{h:02d}.npz'], check=True)
with ThreadPoolExecutor(6) as ex: list(ex.map(one, range($HOURS + 1)))
print('fields done')
PY"
rsync -a -e "ssh -i $K" ubuntu@$I:/woof/work/runs/$R/pf/ $W/npz/
# Python with xarray, cfgrib, eccodes and wgrib2 on PATH expected here.
python $(dirname "$0")/publish_product.py $W/npz $W/out $CYCLE --no-upload | tail -1
[ "$UP" = "--no-upload" ] && exit 0
# Python with boto3 and ovh expected here.
 python $(dirname "$0")/upload_run.py $W/out $CYCLE
