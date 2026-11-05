#!/bin/bash
# The big NAM domain run on N cards: forecast, receipt, maps, comparison sheets.
#
#   tools/run_nam_domain.sh <4km|5km|band12> <hours> <cards> [out]      (source env.sh first)
#   tools/run_nam_domain.sh nest --center LAT,LON --size-km 1000 --res-km 3 ...   (any domain: tools/woof-nam nest)
#
# - Domain: parm/nam_configfile_<domain> through NAMGPU_CONFIGFILE (namgpu/config.py); its static is the
#   configfile's static_npz (init-static-build), its boundaries <case>/boco-<domain_name>/ (init-gfs-bocos
#   builds any that are missing, GFS 0.25 every 3 h).
# - Case: CASE=<YYYY-MM-DD_HHz> (required: no case default in generic tools), from the GFS 0.25 start
#   (--source gfs).
# - Cards: N > 1 = namgpu.init.slab_init save (once, --managed) then tools/run_forecast.py --cards N --init-from
#   (lane E1's rank launcher, one process per card). With 1 card both are left off: the single-card path.
# - Receipt: <out>/forecast-receipt.json (driver timing, ms/step, peak VRAM) plus <out>/domain-receipt.json
#   (domain, configfile, static file and its build record, boco folder, wall per stage, card count).
# - Maps: run_forecast's post-run maps tail (rw_wrfbatch on the referee CONUS frame, MRMS overlay).
# - Sheets: tools/nam_domain_sheets.py, f01..f<hours>: MRMS | this run | our 12 km | archived NAM 12 km,
#   into <out>/sheets/ and, when DELIVER is set, copied to that folder too.
set -euo pipefail
cd "$(dirname "$0")/.."
# A nest anywhere (centre, size, resolution, source) goes through the door (tools/woof-nam, namgpu/nest/cli.py)
if [ "${1:-}" = nest ]; then exec tools/woof-nam "$@"; fi
DOM=${1:?domain 4km|5km|band12}; HOURS=${2:?hours}; CARDS=${3:?cards}
CASE=${CASE:?set CASE=YYYY-MM-DD_HHz (the case folder under cases/)}
OUT=${4:-runs/nam${DOM}-${HOURS}h-${CARDS}c}
REF12=${REF12:-}                     # maps dir of our 12 km run (optional column)
export NAMGPU_CONFIGFILE=parm/nam_configfile_${DOM}
# RRTM chunk scratch is ROWS x IM x 0.59 MB: 26 GB at ROWS 32 x 1371 (4 km), 6.5 GB at 8 (MULTICARD-PLAN.md section 7;
# per column, no answer change). 8 on 32 GB cards unless the caller sets it.
export NAMGPU_RRTM_ROWS=${NAMGPU_RRTM_ROWS:-8}
mkdir -p "$OUT"
t0=$(date +%s)
CARDARG=()
# rc is captured with "|| rc=${PIPESTATUS[0]}": under set -euo pipefail a bare failing "cmd | tee" ends the script at
# once, so a failed init or forecast never reached the sheets or domain-receipt.json (review round 3, #6).
rc=0
stage=forecast
INIT_TMP=
if [ "$CARDS" -gt 1 ]; then
  # N cards: the init runs once (lane E1/E2 slab_init: whole domain, CUDA managed memory when it is bigger than
  # one card), every rank loads its own rows (--init-from).
  [ -z "${INIT:-}" ] && [ -z "${REUSE_INIT:-}" ] && INIT_TMP=/dev/shm/nam${DOM}-init   # our own /dev/shm copy: removed below (kept with REUSE_INIT)
  INIT=${INIT:-/dev/shm/nam${DOM}-init}
  if [ -n "${REUSE_INIT:-}" ] && [ -f "$INIT/skeleton.pkl" ]; then
    echo "reusing saved init $INIT" | tee "$OUT/init.log"      # saved earlier on one card (same domain, case, hours)
  else
    python -m namgpu.init.slab_init save --case "$CASE" --source gfs --hours "$HOURS" --out "$INIT" --managed         2>&1 | tee "$OUT/init.log" || { rc=${PIPESTATUS[0]}; stage=init; }
  fi
  CARDARG=(--cards "$CARDS" --init-from "$INIT")
fi
if [ "$rc" -eq 0 ]; then
  python tools/run_forecast.py --case "$CASE" --source gfs --hours "$HOURS" --out "$OUT" "${CARDARG[@]}" \
      2>&1 | tee "$OUT/run.log" || rc=${PIPESTATUS[0]}
fi
[ -n "$INIT_TMP" ] && rm -rf "$INIT_TMP"
[ "$rc" -eq 0 ] && stage=
t1=$(date +%s)
python tools/nam_domain_sheets.py --run "$OUT" --label "WOOF-NAM ${DOM} (${CARDS} cards)" \
    --start "$(python -c "import sys; from datetime import datetime; print(datetime.strptime(sys.argv[1], '%Y-%m-%d_%Hz').strftime('%Y-%m-%dT%H'))" "$CASE")" \
    ${REF12:+--ref12 "$REF12"} --referee "cases/$CASE/referee/png" --out "$OUT/sheets" --hours "1-${HOURS%.*}" \
    2>&1 | tee -a "$OUT/run.log" || true
t2=$(date +%s)
python - "$OUT" "$DOM" "$CARDS" "$rc" "$((t1 - t0))" "$((t2 - t1))" "$stage" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from namgpu import config
out, dom, cards, rc, run_s, sheets_s, failed = sys.argv[1:]
cfg = config.load()
st = config.static_npz(cfg)
meta = json.loads(str(np.load(st)["META"])) if st and st.is_file() and "META" in np.load(st).files else {}
rec = {"domain": dom, "configfile": str(config.configfile_path(None)), "grid": [cfg.im, cfg.jm, cfg.lm],
       "dt": cfg.dt, "nrads": cfg.nrads, "convection": cfg.convection, "gwdflg": cfg.gwdflg,
       "static_npz": str(st), "static_build": meta, "boco_dir": config.boco_dirname(cfg), "cards": int(cards),
       "run_rc": int(rc), "run_wall_s": int(run_s), "sheets_wall_s": int(sheets_s)}
if failed:
    rec["failed_stage"] = f"{failed} rc {rc} (see {out}/{'init' if failed == 'init' else 'run'}.log)"
Path(out, "domain-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
print(json.dumps(rec, default=str)[:400])
PY
if [ -n "${DELIVER:-}" ]; then mkdir -p "$DELIVER"; cp -r "$OUT/sheets/." "$DELIVER/"; cp "$OUT"/*receipt.json "$DELIVER/"; fi
exit "$rc"
