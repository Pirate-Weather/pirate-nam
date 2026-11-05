"""Cycle WOOF-NAM with NOAA's GSI, on one machine: the catch-up hourly forecasts run on the GPU and GSI on the CPUs.

WOOF-NAM has no restart, so its state stays on the card. At each analysis time its fields overwrite the matching records
of a real NMMB restart file (a template from one Fortran run), GSI runs unchanged on that file, and what GSI changed goes
back onto the live state. Records WOOF-NAM can't supply keep the template's values; GSI doesn't read them.

Sequence for a 00Z cycle: GFS start at tm06, replaced by the coldstart-and-GSI tm06 analysis file (which also carries the
seeded land), then GSI at tm05..tm00, then the free forecast.
Usage: gsi_cycle.py woof-nam-run-args...       Environment: GSI_CPUS, GSI_MEM_GB
"""
import os, shutil, struct, subprocess, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, cupy as cp
from namgpu import state as namgpu_state
from namgpu.driver import Driver
from namgpu import run_forecast
from namgpu.init import init_derive_state
from namgpu.steps import drv_dfi_tdfi_driver
from namgpu.boundary import bnd_read_bc
from namgpu.init import slab_init
# The step after an analysis runs as a first step, which skips the units writing these, while later units still read them mid-run.
# WOOF-NAM 0.1.1 started dropping them after each step, so they have to stay in memory here, as they did in 0.1.0 (0.8 GB).
namgpu_state.STEP_SCRATCH = namgpu_state.STEP_SCRATCH - {"DEF3D", "FMLX_3D", "FMLY_3D", "RTOP"}

C = os.environ["NAM_CYCLE"]; CYC_HH = C[8:]    # the forecast's start, YYYYMMDDHH
CYCLE_TIME = datetime.strptime(C, "%Y%m%d%H")
# Containers always see the tree as /nam; NAM_HOST lets two experiments run side by side from separate trees on the host.
HOST = os.environ.get("NAM_HOST", "/nam")
GES_IN = f"/nam/com/nam/v4.2/nwges/nam.{C[:8]}"
GES = HOST + GES_IN[4:]
COLD_ANL = f"{GES_IN}/nam.t{CYC_HH}z.input_domain_01_nemsio.anl.tm06"
SHM = Path("/dev/shm/gsi_cycle_" + os.path.basename(HOST))
STEPS_PER_HOUR = 144
QMIN = 1e-8
FILTER_EACH_CYCLE = os.environ.get("GSI_FILTER", "0") == "1"
RESET_NH = os.environ.get("GSI_RESET_NH", "0") == "1"
HANDOFF = os.environ.get("GSI_HANDOFF")
LAND_SEED = os.environ.get("GSI_LAND_SEED", "0") == "1"
RRFS_START = bool(os.environ.get("RRFS_NPZ"))
if RRFS_START: import rrfs_start
CPUS = os.environ.get("GSI_CPUS", "15"); MEM = os.environ.get("GSI_MEM_GB", "39")
NX, NY, LM = 954, 835, 60
# Which hours get a GSI analysis. tm06 always applies the cold-start analysis; fewer hours trade fit for speed.
ANALYSES = os.environ.get("GSI_TIMES", "06,05,04,03,02,01,00").split(",")
schedule = {(6 - int(k)) * STEPS_PER_HOUR: f"tm{k}" for k in ANALYSES}
done = set()
orig_step = Driver.step

def container(*args):
    subprocess.run(["docker", "run", "--rm", "--user", "1000:1000", "-v", f"{HOST}:/nam", "-v", "/dev/shm:/dev/shm", "namhar", *args], check=True,
                   stdout=subprocess.DEVNULL)

# Fields handed to GSI. (record file, state name). Levels are top first in both, so they map directly.
GUESS_3D = {"tmp__mid_layer": "T", "spfh__mid_layer": "Q", "clwmr__mid_layer": "CW", "f_ice__mid_layer": "F_ICE", "f_rain__mid_layer": "F_RAIN",
            "f_rimef__mid_layer": "F_RIMEF", "ugrd__mid_layer": "U", "vgrd__mid_layer": "V", "dwdt__mid_layer": "DWDT", "o3mr__mid_layer": "O3",
            "refl_10cm__mid_layer": "REFL_10CM", "pres__layer": "PINT", "stc__soil_layer": "STC", "smc__soil_layer": "SMC", "sh2o__soil_layer": "SH2O"}
GUESS_2D = {"tskin__sfc": "TSKIN", "tg__sfc": "TG", "sno__sfc": "SNO", "si__sfc": "SI", "cmc__sfc": "CMC", "ths__sfc": "THS",
            "u10__10_m_above_gnd": "U10", "v10__10_m_above_gnd": "V10", "tshltr__sfc": "TSHLTR", "qshltr__sfc": "QSHLTR", "pshltr__sfc": "PSHLTR"}
# What GSI may have changed, copied back whole. PD is handled apart, as a difference, because the file stores ps - ptop.
ANALYSED = {"tmp__mid_layer": "T", "spfh__mid_layer": "Q", "clwmr__mid_layer": "CW", "f_ice__mid_layer": "F_ICE", "f_rain__mid_layer": "F_RAIN",
            "f_rimef__mid_layer": "F_RIMEF", "ugrd__mid_layer": "U", "vgrd__mid_layer": "V", "o3mr__mid_layer": "O3", "refl_10cm__mid_layer": "REFL_10CM",
            "tg__sfc": "TG"}
LAND_START = {"stc__soil_layer": "STC", "smc__soil_layer": "SMC", "sh2o__soil_layer": "SH2O", "tskin__sfc": "TSKIN", "sice__sfc": "SICE", "tg__sfc": "TG", "sno__sfc": "SNO", "si__sfc": "SI", "cmc__sfc": "CMC", "ths__sfc": "THS"}
READ_NAMES = "tmp,spfh,clwmr,f_ice,f_rain,f_rimef,ugrd,vgrd,o3mr,refl_10cm,dfi_tten,tsea,tg,dpres,stc,smc,sh2o,tskin,sno,si,cmc,ths,sice"
HEADER_IDATE_OFFSET = 64        # byte offset of idate(1) in a big-endian NEMSIO header
T_MIN, T_MAX = 150.0, 340.0     # kelvin; outside this the model state is broken and GSI must not see it

def read_rec(d, name, shape):
    path = d / f"{name}.bin"
    return np.fromfile(path, "f4").reshape(shape) if path.exists() else None

def export_guess(ctx, d):
    st = ctx.st
    ctx.stream.synchronize()
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    for name, field in {**GUESS_3D, **GUESS_2D}.items():
        if field in st.fields: cp.asnumpy(st[field]).astype("f4").tofile(d / f"{name}.bin")
        else: print("not in state, template kept:", field, flush=True)
    pint = st["PINT"]
    cp.asnumpy(pint[-1] - pint[0]).astype("f4").tofile(d / "dpres__hybrid_sig_lev.bin")
    # NMMB files hold tsea on open water only and zero elsewhere, while WOOF-NAM keeps SST everywhere.
    open_water = (st["SM"] > 0.5) & (st["SICE"] < 0.5)
    cp.asnumpy(cp.where(open_water, st["SST"], 0.0)).astype("f4").tofile(d / "tsea__sfc.bin")

def set_guess_time(path, k):
    """The template is one Fortran restart, so its header holds that run's date whichever hour it now holds. GSI takes the guess
    time from there, and with the wrong one it silently matches no observations (cost 0). A tmNN guess is a 1 h forecast
    (the header's forecast hour stays 1), so its start is 1 h before the tmNN analysis time."""
    start = CYCLE_TIME - timedelta(hours=k + 1)
    with open(path, "r+b") as f:
        f.seek(HEADER_IDATE_OFFSET)
        f.write(struct.pack(">4i", start.year, start.month, start.day, start.hour))

def check_state(ctx, label):
    """A broken model state or a NaN analysis must stop the cycle, because GSI still exits 0 on NaN input."""
    st = ctx.st
    ctx.stream.synchronize()
    t = st["T"]; low, high = float(t.min()), float(t.max())
    finite = {n: bool(cp.isfinite(st[n]).all()) for n in ("T", "Q", "U", "V", "PD")}
    print(f"{label}: T {low:.1f}..{high:.1f}", flush=True)
    if low < T_MIN or high > T_MAX or not all(finite.values()):
        raise RuntimeError(f"{label}: invalid state, T {low}..{high}, finite {finite}")

def apply_analysis(ctx, d, tm, mapping, guess_dpres=None):
    st = ctx.st
    # init_derive_state on a non-restart run does the cold fill, so whatever the fill would wipe has to be held around it.
    # W and DWDT: the fill also rebuilds PINT hydrostatically from PD, so keeping them can pair a nonhydrostatic W with a
    # hydrostatic pressure. GSI_RESET_NH=1 lets the fill zero them, which is at least self-consistent.
    keep_names = ("Q2", "OMGALF", "TOLD") if RESET_NH else ("Q2", "OMGALF", "TOLD", "W", "DWDT")
    keep = {n: st[n].copy() for n in keep_names if n in st.fields}
    for name, field in mapping.items():
        shape = st[field].shape
        a = read_rec(d, name, shape)
        if a is None: print("analysis missing", name, flush=True); continue
        st[field][...] = cp.asarray(a)
    st["Q"][...] = cp.maximum(st["Q"], QMIN)
    # Surface pressure: the file holds ps - ptop, WOOF-NAM's PD excludes a hybrid part, so move PD by the analysed change.
    dpres = read_rec(d, "dpres__hybrid_sig_lev", (NY, NX))
    pint = st["PINT"]
    st["PD"][...] += cp.asarray(dpres) - (pint[-1] - pint[0])
    # Open water only (see export_guess); ice and land keep the skin temperature WOOF-NAM carries in SST.
    tsea = read_rec(d, "tsea__sfc", (NY, NX))
    st["SST"][...] = cp.where(cp.asarray(tsea) > 0.0, cp.asarray(tsea), st["TSKIN"])
    tten = read_rec(d, "dfi_tten__mid_layer", (LM, NY, NX))
    if tten is not None and "DFI_TTEN" in st.fields: st["DFI_TTEN"][...] = cp.asarray(tten)
    # O3 and REFL_10CM are read back above and would be wiped too; REFL_10CM is what the 0 h composite is written from.
    analysed = {n: st[n].copy() for n in ("O3", "REFL_10CM") if n in st.fields}
    init_derive_state.run(ctx)
    ctx.stream.synchronize()
    for n, a in {**keep, **analysed}.items(): st[n][...] = a
    ctx.first_step = True

def analyse(ctx, tm):
    t0 = time.time(); st = ctx.st
    anl_dir = SHM / f"anl_{tm}"
    shutil.rmtree(anl_dir, ignore_errors=True); anl_dir.mkdir(parents=True); os.chmod(anl_dir, 0o777)
    if tm == "tm06":
        # Production's first analysis is on the cold-start file, not on a forecast. Reuse that analysis, seeded land included.
        container("/nam/exec/nemsio_fields", "dump", COLD_ANL, str(anl_dir), READ_NAMES)
        apply_analysis(ctx, anl_dir, tm, {**{k: v for k, v in ANALYSED.items() if k.startswith(("tmp", "spfh", "clwmr", "ugrd", "vgrd", "o3mr"))}, **LAND_START})
        # WOOF-NAM's init rule for snow cover, since SNO has just changed underneath it.
        st["SNOWC"][...] = cp.where(st["SNO"] > 0.0, 0.98, 0.0)
        print(f"tm06 start applied in {time.time() - t0:.0f} s", flush=True)
        shutil.rmtree(anl_dir); return
    # The template restart moves to the name this analysis expects, so only one 9 GB file ever exists.
    k = int(tm[2:]); guess_file = f"{GES}/nam.t{CYC_HH}z.nmm_b_restart_nemsio.tm{k + 1:02d}"
    previous = [f for f in Path(GES).glob(f"nam.t{CYC_HH}z.nmm_b_restart_nemsio.tm*")][0]
    if str(previous) != guess_file: previous.rename(guess_file)
    gd = SHM / f"guess_{tm}"
    export_guess(ctx, gd); os.chmod(gd, 0o777)
    t1 = time.time()
    container("/nam/exec/nemsio_fields", "apply", guess_file.replace(HOST, "/nam", 1), str(gd))
    set_guess_time(guess_file, k)
    t2 = time.time()
    check_state(ctx, f"guess {tm}")
    # The fields are in the restart file now. Holding them in /dev/shm through GSI cost 2.3 GB that a t2-45 hasn't got spare.
    shutil.rmtree(gd)
    # One GSI at a time on this machine: two at once (37 GB each, plus two forecasts' memory) got a forecast OOM-killed.
    rec = subprocess.run(["flock", "/woof/work/gsi.lock", "bash", "/nam/gpuda/gsi_mem.sh", tm, MEM, CPUS], capture_output=True, text=True, env=dict(os.environ, NAM_HOST=HOST))
    print(rec.stdout.strip(), flush=True)
    if "rc=0" not in rec.stdout: raise RuntimeError(f"GSI failed at {tm}: {rec.stdout} {rec.stderr[-400:]}")
    t3 = time.time()
    log_text = Path(f"{HOST}/logs/{C}_{tm}_JNAM_ANALYSIS.log").read_text(errors="replace")
    costs = [ln.split("=")[1].strip() for ln in log_text.splitlines() if ln.startswith(("Initial cost function", "Final   cost function"))]
    print(f"{tm}: GSI cost (each outer loop) {costs}", flush=True)
    if not costs or any("NaN" in c or float(c.replace("D", "E")) <= 0 for c in costs): raise RuntimeError(f"GSI cost invalid at {tm}: {costs}")
    anl_file = f"{GES}/nam.t{CYC_HH}z.nmm_b_restart_nemsio_anl.{tm}"
    container("/nam/exec/nemsio_fields", "dump", anl_file.replace(HOST, "/nam", 1), str(anl_dir), READ_NAMES)
    apply_analysis(ctx, anl_dir, tm, ANALYSED)
    check_state(ctx, f"analysis {tm}")
    os.remove(anl_file); shutil.rmtree(anl_dir)
    print(f"{tm}: patch {t2 - t1:.0f} s, GSI {t3 - t2:.0f} s, read back and apply {time.time() - t3:.0f} s, total {time.time() - t0:.0f} s", flush=True)

def run_filter(ctx, tm):
    """The NAM filters every catch-up forecast's start (filter_method 3). Here the filter runs on its own clock inside the
    run, so the forecast clock has to be put back afterwards."""
    t0 = time.time(); n = ctx.ntimestep
    ctx.filter_active = True
    try:
        drv_dfi_tdfi_driver.run(ctx)
    finally:
        ctx.filter_active = False
    ctx.ntimestep = n; ctx.first_step = True; ctx.dt = ctx.cfg.dt
    ctx.stream.synchronize()
    check_state(ctx, f"filter {tm}")
    print(f"{tm}: filter {time.time() - t0:.0f} s", flush=True)

def reload_boundaries(ctx, n):
    """Put the lateral boundary strips where an uninterrupted run would have them at step n.

    first_step=True skips the not_first block for one step, and bnd-read-bc lives there: an analysis landing on a 3 h boundary
    time left the previous interval's tendency running for 3 more hours, and the next file then snapped the edges back. That
    launched a surface pressure shock from all four edges (measured: 138 Pa/h on the edge rows at tm00+3 h, about 50 otherwise).
    The filter also rewrites the strips and tendencies for its own use (bnd_write_bc at dfi_start), so it needs this as well."""
    tboco = float(ctx.tables.get("consts", {}).get("tboco", 10800.0))
    dt = float(ctx.cfg.dt); nboco = int(round(tboco / abs(dt))); g = ctx.grid
    n_start = n - n % nboco
    dims = dict(bnd_read_bc.PARENT_DIMS, im=g.im, jm=g.jm, lm=g.lm, lnsh=ctx.cfg.lnsh, lnsv=ctx.cfg.lnsv)
    pend = bnd_read_bc.stage_interval(ctx, ctx.boco_dir, n_start, dt, dims, bnd_read_bc.dict_destinations(ctx))
    for name in bnd_read_bc.CLASSES:
        cls = getattr(pend, name)
        if cls is not None: bnd_read_bc.place(ctx, cls, int(g.im), int(g.jm), expect_slots=pend.n_bc_3d_h if name == "h3d" else None)
    ctx.stream.synchronize()
    # Slot 0 is the working value, slot 1 the tendency per second. Same arithmetic as bnd-bocoh-update, for the steps already run.
    for key, a in ctx.bc.items():
        if key[:2] in ("S_", "N_", "W_", "E_") and getattr(a, "ndim", 0) >= 1 and a.shape[0] == 2: a[0] += a[1] * float((n - n_start) * dt)
    print(f"boundaries reloaded at step {n} from interval start {n_start}", flush=True)

def seed_land(ctx):
    """Land-only tm06 start: soil and snow from the seeded tm06 file, atmosphere left as the GFS start. Separates the land
    effect from the Fortran start's extra analysis. Test only: the seed comes from archived NAM output, so it has no live source."""
    anl_dir = SHM / "land_seed"
    shutil.rmtree(anl_dir, ignore_errors=True); anl_dir.mkdir(parents=True); os.chmod(anl_dir, 0o777)
    container("/nam/exec/nemsio_fields", "dump", COLD_ANL, str(anl_dir), READ_NAMES)
    st = ctx.st
    for name, field in LAND_START.items():
        a = read_rec(anl_dir, name, st[field].shape)
        if a is None: raise RuntimeError(f"land seed record missing: {name}")
        st[field][...] = cp.asarray(a)
    st["SNOWC"][...] = cp.where(st["SNO"] > 0.0, 0.98, 0.0)
    ctx.stream.synchronize(); shutil.rmtree(anl_dir)
    print("land seeded at tm06", flush=True)

def save_handoff(ctx):
    """Save the analysed state for the two-card forecast. It starts its own clock, so the saved context has to describe the
    forecast's case: its start time, and boundary files cut for its window. The catch-up's last hour isn't needed afterwards."""
    case_dir = ctx.case_dir.parent / CYCLE_TIME.strftime("%Y-%m-%d_%Hz")
    keep = (ctx.start, ctx.start_time, ctx.case_dir, ctx.boco_dir, ctx.ntimestep, ctx.wrote_first_hist, ctx.first_step)
    ctx.start = ctx.start_time = CYCLE_TIME; ctx.case_dir = case_dir; ctx.boco_dir = str(case_dir / "boco")
    # The forecast's 0 h file is the analysis, and it's only written when the saved context says hour 0 hasn't been yet.
    ctx.ntimestep = 0; ctx.wrote_first_hist = False; ctx.first_step = True
    # RDTEMP caches its start date and hour on first use. Saved as is, the forecast would take the sun angle it scales
    # shortwave heating by from the catch-up's start, 6 h early, and lose most of the morning's sun. Dropped, each run rebuilds it.
    ctx.tables.pop("RDTEMP_CAL", None)
    t0 = time.time(); rec = slab_init.save(ctx, HANDOFF)
    ctx.start, ctx.start_time, ctx.case_dir, ctx.boco_dir, ctx.ntimestep, ctx.wrote_first_hist, ctx.first_step = keep
    if rec.get("skipped"): print("handoff skipped:", rec["skipped"], flush=True)
    print(f"handoff saved to {HANDOFF}: {rec['gib']} GiB in {time.time() - t0:.0f} s", flush=True)

def analyse_at(ctx, n):
    if n in schedule and n not in done and not ctx.filter_active:
        done.add(n); analyse(ctx, schedule[n])
        if FILTER_EACH_CYCLE: run_filter(ctx, schedule[n])
        # With a handoff, tm00 is this run's last step and the forecast reads its own boundaries, so there's nothing to reload:
        # the catch-up case has no boundary file past its end. tm06 is the first step, whose boundaries arrive normally at step 1.
        if schedule[n] == "tm00" and HANDOFF: save_handoff(ctx)
        elif n > 0: reload_boundaries(ctx, n)

def step(self):
    # tm06 goes in before the first step. Later analyses go in straight after the step that reaches their hour, because
    # the loop writes that hour's history next: otherwise the 0 h file would hold the guess, not the analysis.
    if self.ctx.ntimestep == 0 and not self.ctx.filter_active:
        if RRFS_START: rrfs_start.apply_rrfs(self.ctx)
        if LAND_SEED and "land" not in done: done.add("land"); seed_land(self.ctx)
    if self.ctx.ntimestep == 0: analyse_at(self.ctx, 0)
    orig_step(self)
    if self.ctx.ntimestep > 0: analyse_at(self.ctx, self.ctx.ntimestep)

Driver.step = step
if RRFS_START:
    # The start's filter must see the RRFS atmosphere, as in rrfs_start.py run alone.
    run_tdfi = drv_dfi_tdfi_driver.run
    def filter_after_rrfs(ctx, *a, **k):
        rrfs_start.apply_rrfs(ctx)
        return run_tdfi(ctx, *a, **k)
    drv_dfi_tdfi_driver.run = filter_after_rrfs
sys.exit(run_forecast.main(sys.argv[1:]))
