#!/bin/bash
# Fetch and lay out one cycle's inputs on the GPU machine: public data only, no credentials here. Usage: fetch_inputs.sh YYYYMMDDHH
# Boundaries and the 6-hour-early start come from the GFS cycle 6 h before the forecast, as the NAM did: a 12Z forecast uses 06Z GFS.
set -uo pipefail
C=$1; HH=${C:8:2}; PDY=${C:0:8}; FCST_HOURS=${FCST_HOURS:-84}; ANALYSIS_LEADS="04 02 00"
at() { date -u -d "${C:0:4}-${C:4:2}-${C:6:2} ${HH}:00 UTC -$1 hours" +"$2"; }   # UTC is required: "12:00 -6" would read as a timezone
G=$(at 6 %Y%m%d%H); GD=${G:0:8}; GH=${G:8:2}; AGED=$(at 12 %H)
RAW=/woof/work/gfs/$G; A=/woof/work/cases/$(at 6 %Y-%m-%d_%Hz); B=/woof/work/cases/$(at 0 %Y-%m-%d_%Hz); mkdir -p $RAW $A/raw/gfs $B/raw/gfs
fail() { echo "FETCH_FAILED $*"; exit 1; }

fetch_one() { # URL OUT: bounded retries, written to .part so a dead transfer can't look like a file
  [ -s "$2" ] && return 0
  for a in 1 2 3 4; do curl -sf --http1.1 -m 1500 -o "$2.part" "$1" && { mv "$2.part" "$2"; return 0; }; sleep $((a * 20)); done
  echo "MISSING $1"; return 1
}
export -f fetch_one

echo "gfs $G $(date -u +%H:%M:%S)"
HOURS=$(seq -f %03g 0 3 $((FCST_HOURS + 6)))
for h in $HOURS; do echo "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.$GD/$GH/atmos/gfs.t${GH}z.pgrb2.0p25.f$h $RAW/gfs.t${GH}z.pgrb2.0p25.f$h"; done \
  | xargs -P 8 -L 1 bash -c 'fetch_one "$0" "$1"' || fail gfs
for h in $HOURS; do [ -s $RAW/gfs.t${GH}z.pgrb2.0p25.f$h ] || fail "gfs f$h"; done
# The catch-up case starts from f000 and takes its boundaries from f000-f006. WOOF-NAM looks for the cycle 6 h older than its start,
# under that cycle's name, so this cycle's files are linked in under that name: the catch-up then uses the same GFS run as the forecast.
ln -sf $RAW/gfs.t${GH}z.pgrb2.0p25.f000 $A/raw/gfs/gfs.t${GH}z.pgrb2.0p25.f000
for k in 000 003 006; do ln -sf $RAW/gfs.t${GH}z.pgrb2.0p25.f$k $A/raw/gfs/gfs.t${AGED}z.f$(printf %03d $((10#$k + 6))).grib2; done
# The forecast case's boundaries are simply this GFS cycle's f006 onwards, WOOF-NAM's normal rule.
for h in $(seq -f %03g 6 3 $((FCST_HOURS + 6))); do ln -sf $RAW/gfs.t${GH}z.pgrb2.0p25.f$h $B/raw/gfs/gfs.t${GH}z.f$h.grib2; done

echo "observations $(date -u +%H:%M:%S)"
N=https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod
mkdir -p /nam/ens_tmp   # make_refd3d.py's scratch space
for lead in $ANALYSIS_LEADS; do
  AD=$(at $((10#$lead)) %Y%m%d); AH=$(at $((10#$lead)) %H); O=/nam/com/obsproc/v1.3/rap.$AD; mkdir -p $O
  # NAM's own observation files ended with the NAM, so GSI's second choice is what's used: RAP's hourly files for each analysis hour.
  fetch_one $N/rap.$AD/rap.t${AH}z.prepbufr.tm00.nr $O/rap.t${AH}z.prepbufr.tm00.nr || fail "prepbufr $AD $AH"
  ln -sf rap.t${AH}z.prepbufr.tm00.nr $O/rap.t${AH}z.prepbufr.tm00
  # Optional streams: GSI runs without them, so a missing one is logged but doesn't stop the cycle.
  for t in satwnd nexrad lgycld; do fetch_one $N/rap.$AD/rap.t${AH}z.$t.tm00.bufr_d $O/rap.t${AH}z.$t.tm00.bufr_d; sleep 2; done
  docker run --rm --user 1000:1000 -v /nam:/nam namhar python /nam/prod/make_refd3d.py $AD$AH | tail -1 || echo "radar missing for $AD$AH: GSI skips its cloud analysis"
done

echo "ensemble $(date -u +%H:%M:%S)"
# GEFS init at T stands in for the EnKF's 6 h forecast from T-6. Use the newest init that exists, no later than the forecast's start.
for back in 0 6 12; do T=$(at $back %Y%m%d%H); curl -sf -I -m 20 https://noaa-gefs-pds.s3.amazonaws.com/gefs.${T:0:8}/${T:8:2}/atmos/init/p30/gfs_data.tile1.nc >/dev/null && break; T=""; done
[ -n "$T" ] || fail "no GEFS init"
/woof/venv/bin/python /nam/prod/gefs_init_to_atmf.py $T ${GEFS_PROCS:-8} | tail -2 || fail "ensemble $T"

echo "gsi tree $(date -u +%H:%M:%S)"
COM=/nam/com/nam/v4.2/nam.$PDY; GES=/nam/com/nam/v4.2/nwges/nam.$PDY; mkdir -p $COM $GES /nam/com/obsproc/v1.3/nam.$PDY /nam/com/radarl2/v1.2/radar.$PDY
cat > $GES/nam.t${HH}z.envir.sh <<V
export CYCLE=$C
export GUESS=GDAS
export RUNTYPE=CATCHUP
export GUESStm00=NAM
V
rm -f $GES/nam.t${HH}z.nmm_b_restart_nemsio*; cp /nam/gpuda/restart_template.nemsio $GES/nam.t${HH}z.nmm_b_restart_nemsio.tm06
for t in 05 04 03 02 01 00; do rm -f $COM/nam.t${HH}z.satbias.tm$t $COM/nam.t${HH}z.satbias_pc.tm$t; done
cp /nam/gpuda/satbias.tm06 $COM/nam.t${HH}z.satbias.tm06; cp /nam/gpuda/satbias_pc.tm06 $COM/nam.t${HH}z.satbias_pc.tm06
df -h /woof | tail -1; echo "INPUTS_READY $C $(date -u +%H:%M:%S)"
