"""The NMMB time loop, in SOLVER_RUN order (module_SOLVER_GRID_COMP.F90:2453-7168), for the
regional (GLOBAL=false) nonhydrostatic 12 km parent.

Every entry below is a UNIT id from UNITS.json.  A unit's wrapper is the Python module named in
UNITS.json "wrapper" and exports ``run(ctx)``: it launches its kernel(s) on ctx.stream and
returns.  No wrapper synchronises the device or copies to the host.

Global-only pieces of SOLVER_RUN (FFTFHN/FFTFWN/FFTFUVN polar filters, POAVHN, POLEHN/POLEWN,
SWAPHN/SWAPWN) are mode "skip" in UNITS.json: the NAM parent is regional.  MPI halo exchanges
are dropped: one card holds the whole domain (halo-free layout, CONTRACT.md section 1).
"""
from __future__ import annotations

import importlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .steps import drv_timing_receipt as timing
from . import devmem
from .state import UNIT_SCRATCH as _UNIT_SCRATCH, STEP_SCRATCH_TABLE_PREFIXES as _TPFX, fit_on as _fit_on
from .state import STEP_SCRATCH_TABLE_KEEP as _TKEEP

from . import paths

ROOT = Path(__file__).resolve().parents[1]


def load_units() -> dict:
    return {u["id"]: u for u in json.loads((paths.UNITS_JSON).read_text())["units"]}


# ---- step plan --------------------------------------------------------------------------------
# (unit id, cadence key[, kwargs]).  Cadence keys are evaluated by Driver._due(); kwargs go to the
# wrapper's run().  CADENCE_KEYS is the list check_plan validates against.
CADENCE_KEYS = ("every", "first", "not_first", "not_first_hdiff", "q_tracer", "tracers", "nphs",
                "nprecip", "cltend_pre", "cltend_post", "cltend_inc", "radiation", "history",
                "update_water", "max_fields")
# Halo-exchange rows ("drv-halo-exch", cadence, {"x": "Xn"}) sit where module_SOLVER_GRID_COMP.F90 calls HALO_EXCH
# (MULTICARD-PLAN.md section 3; Fortran lines in namgpu/steps/drv_halo_exch.py EXCH). On one card they are skipped
# (Driver.step), so the one-card plan and output are unchanged.
FIRST_STEP = [                      # SOLVER_RUN firststep block, 3390-3542
    ("drv-halo-exch", "first", {"x": "X1"}),
    ("dyn-pgforce-geopot", "first"),
    ("dyn-pgforce-pgf", "first"),        # pgf scratch (module_DYNAMICS_ROUTINES.F90:249-289)
    ("dyn-pgforce-divcor", "first"),
    ("dyn-pgforce-uvupdate", "first"),   # pgforce:452 switch: restart -> TCU/TCV update, else zero
    ("drv-halo-exch", "first", {"x": "X2"}),
    ("dyn-dht", "first"),
    ("drv-halo-exch", "first", {"x": "X3"}),
]
DYNAMICS = [                        # not_firststep block 3546-4228, then 4245-4857
    ("dyn-hdiff-coef", "not_first_hdiff"),     # HDIFF gated on HDIFF_ON>0 (SOLVER_RUN 3577-3590)
    ("dyn-hdiff-apply", "not_first_hdiff"),
    ("drv-halo-exch", "not_first", {"x": "X4"}),
    ("bnd-read-bc", "not_first"),          # wrapper acts only when a 3-h interval starts
    ("bnd-bocoh-update", "not_first"),
    ("bnd-bocoh-blend", "not_first"),
    ("bnd-bocoh-smooth", "not_first"),
    ("dyn-pgforce-geopot", "not_first"),
    ("dyn-pgforce-pgf", "not_first"),
    ("dyn-pgforce-divcor", "not_first"),
    ("dyn-pgforce-uvupdate", "not_first"),
    ("dyn-updateuv", "not_first"),
    ("drv-halo-exch", "not_first", {"x": "X6"}),
    ("bnd-bocov-update", "not_first"),
    ("bnd-bocov-blend", "not_first"),
    ("dyn-dht", "not_first"),
    ("drv-halo-exch", "not_first", {"x": "X7"}),
    ("dyn-ddamp-extmode", "not_first_hdiff"),  # DDAMP gated on HDIFF_ON>0 (SOLVER_RUN 4176-4187)
    ("drv-halo-exch", "not_first", {"x": "X8"}),
    ("dyn-pdtsdt", "every"),
    ("drv-halo-exch", "every", {"x": "X9"}),
    ("dyn-adv1-vert-t", "every"),
    ("dyn-adv1-horiz-t", "every"),
    ("dyn-adv1-bndry-t", "every"),
    ("dyn-adv1-vert-uv", "every"),
    ("dyn-adv1-coriolis", "every"),
    ("dyn-adv1-horiz-uv", "every"),
    ("dyn-adv1-bndry-uv", "every"),
    # SOLVER_RUN 4448 (RIME_FACTOR_INPUT=.TRUE.), before the ADV2 block (4468). Gated inside the
    # wrapper by RIME_FACTOR_ADVECT (fer_hires and spec_adv); dormant when spec_adv=.false.
    ("drv-rime-factor-update", "tracers", {"branch": "before"}),
    ("dyn-adv2-q2-mid", "q_tracer", {"kss": "INDX_Q", "kse": "INDX_Q"}),
    ("dyn-adv2-vert", "q_tracer", {"kss": "INDX_Q", "kse": "INDX_Q", "idtadt": "IDTADTQ"}),
    ("dyn-adv2-horiz", "q_tracer", {"kss": "INDX_Q", "kse": "INDX_Q", "idtadt": "IDTADTQ"}),
    # ADV2 regional strips, same ADV2 call as dyn-adv2-vert: Q call 4321-4325, tracer call 4459-4461.
    ("dyn-adv2-bndry", "q_tracer", {"kss": "INDX_Q", "kse": "INDX_Q", "idtadt": "IDTADTQ"}),
    ("dyn-mono-fix", "q_tracer", {"kss": "INDX_Q", "kse": "INDX_Q", "idtadt": "IDTADTQ"}),
    ("dyn-mono-global-sum", "q_tracer", {"kss": "INDX_Q", "kse": "INDX_Q"}),
    # SOLVER_RUN 4321-4392: KSS=INDX_Q, KSE1=INDX_Q (1..1) in the q_tracer block.
    ("dyn-updates", "q_tracer", {"kss": "INDX_Q", "kse": "INDX_Q"}),
    ("drv-halo-exch", "q_tracer", {"x": "X10"}),
    ("dyn-adv2-q2-mid", "tracers", {"kss": "INDX_CW", "kse": "INDX_Q2"}),
    ("dyn-adv2-vert", "tracers", {"kss": "INDX_CW", "kse": "KSE_TRACERS", "idtadt": "IDTADT"}),
    ("dyn-adv2-horiz", "tracers", {"kss": "INDX_CW", "kse": "KSE_TRACERS", "idtadt": "IDTADT"}),
    ("dyn-adv2-bndry", "tracers", {"kss": "INDX_CW", "kse": "KSE_TRACERS", "idtadt": "IDTADT"}),
    ("dyn-mono-fix", "tracers", {"kss": "INDX_CW", "kse": "KSE_TRACERS", "idtadt": "IDTADT"}),
    ("dyn-mono-global-sum", "tracers", {"kss": "INDX_CW", "kse": "KSE_TRACERS"}),
    ("dyn-mono-q2-back", "tracers", {"kss": "INDX_CW", "kse": "INDX_Q2"}),
    # SOLVER_RUN 4459-4461 and 4540 (UPDATES): SPEC_ADV=.false. KSE1=INDX_Q2 (4456-4462);
    # SPEC_ADV=.true. KSE1=NUM_TRACERS_TOTAL (4442-4443). KSE_TRACERS picks the branch from cfg spec_adv.
    ("dyn-updates", "tracers", {"kss": "INDX_CW", "kse": "KSE_TRACERS"}),
    ("drv-rime-factor-update", "tracers"),
    ("drv-halo-exch", "tracers", {"x": "X11"}),
    ("dyn-vtoa", "every"),
    ("dyn-updatet", "every"),
    ("drv-halo-exch", "every", {"x": "X12"}),
    ("dyn-cdzdt-nonhydro", "every"),
    ("dyn-cdzdt-advect-z", "every"),
    ("dyn-cdzdt-extmode", "every"),
    ("drv-halo-exch", "every", {"x": "X13"}),
    ("dyn-cdwdt-boundary-smooth", "every"),
    ("dyn-cdwdt-local-deriv", "every"),
    ("dyn-cdwdt-diffusion", "every"),
    ("dyn-cdwdt-vert-adv", "every"),
    ("dyn-cdwdt-horiz-adv", "every"),
    ("dyn-cdwdt-extmode", "every"),
    ("dyn-cdwdt-filter", "every"),
    ("drv-halo-exch", "every", {"x": "X14"}),
    ("dyn-vsound", "every"),
    ("drv-halo-exch", "every", {"x": "X15"}),
]
PHYSICS = [                         # SOLVER_RUN 4929-6077 (INTEGER_DT>0, non-GFS branch), Fortran order
    ("io-max-fields", "max_fields"),         # 5000-5167: NSTEPS_PER_CHECK, FILTER_METHOD==0, MICROPHYSICS==fer_hires
    ("drv-update-water", "update_water"),    # 5188-5219: micro in fer/fer_hires/gfs/wsm6/thompson, any CALL_*
    ("phys-rad-driver", "radiation"),        # 5274-5360: on NTIMESTEP_RAD (5312), SW and LW flags separately
    ("drv-bucket-reset", "every", {"blocks": "nclod"}),        # 5367-5376, before RDTEMP
    ("drv-rdtemp", "every"),                 # 5384
    ("drv-bucket-reset", "every", {"blocks": "post_rdtemp"}),  # 5418-5467, after RDTEMP (own cadences)
    ("phys-turbl-driver", "nphs"),          # JSFC, NOAHLSM, SFCDIAGS, MYJPBL, GWD inside
    ("drv-halo-exch", "nphs", {"x": "X16"}),
    ("drv-h-to-v-tend", "nphs"),
    ("drv-halo-exch", "nphs", {"x": "X17"}),
    ("drv-halo-exch", "nphs", {"x": "X18"}),
    ("drv-cltend", "cltend_pre", {"icltend": -1}),   # CLTEND 1 of 3, SOLVER_RUN 5705-5714
    ("phys-bmj-driver", "nprecip"),                  # CUCNVC, 5732
    ("drv-halo-exch", "nprecip", {"x": "X19"}),
    ("drv-h-to-v-tend", "nprecip", {"branch": "bmj"}),  # wind block 5813-5830, NSTEPS=NPRECIP
    ("drv-halo-exch", "nprecip", {"x": "X20"}),
    ("phys-fer-driver", "nprecip"),                  # GSMDRIVE, 5880
    ("drv-cltend", "cltend_post", {"icltend": 0}),   # CLTEND 2 of 3, 5925-5934
    ("drv-cltend", "cltend_inc", {"icltend": 1}),    # CLTEND 3 of 3, 6005-6014 (before TQADJUST 6022)
    ("drv-tqadjust", "every"),
    ("drv-halo-exch", "every", {"x": "X21"}),
    # No reset after TQADJUST: SOLVER_RUN 6348-6395 (NRDLW/NRDSW/NSRFC/NPREC resets) sit in the
    # ELSE gfs_phys_test branch (6077), GFS physics only. The NAM resets are the post_rdtemp block
    # (5418-5467). Running them here emptied CUPREC/ACPREC and the flux buckets right after the
    # physics that filled them (CUPREC stayed 0 at step 0 while CPRATE > 0).
]
# Plan entries the configfile switches off, as the Fortran IF around the call does.
CONFIG_GATES = {
    "phys-bmj-driver": lambda cfg: str(cfg.convection).strip() != "none",   # SOLVER_RUN 5722
}
DIAGNOSTICS = []                    # io-max-fields moved to the head of PHYSICS (Fortran 5000 < 5173)
# History is not a plan entry. Fortran NMM_INTEGRATE writes it after the clock advance on the
# post-advance count (module_NMM_INTEGRATE.F90:1153-1180), and writes the 0 h file before the first
# advance (794-813). Driver.write_history() owns both; see Driver.run.
# io-quasipost-driver gives the history its MSLP (module_QUASIPOST, the NMMB in-model post).
HISTORY_UNITS = ("io-wind-rotate", "io-quasipost-driver", "io-history-fields", "io-history")
# Units the TDFI sums are moved off the card for, inside the filter, on the fit path (Driver.step).
FILTER_SPILL_UNITS = frozenset({"phys-rad-driver"})


def restart_flag(cfg) -> bool:
    """cfg.raw['restart'] (parm key `restart:`). True/.true./t/yes/1 mean a restarted run; the
    REST placeholder (substituted by exnam_fcst.sh) and anything else reads as a cold start."""
    v = cfg.raw.get("restart", False)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", ".true.", "t", "yes", "1")


@dataclass
class Ctx:
    cfg: object
    grid: object
    st: object
    stream: object = None
    ntimestep: int = 0
    first_step: bool = True
    cdwdt_skip: bool = False               # set each step by dyn-cdwdt-boundary-smooth: Fortran CDWDT
                                           # returned at module_DYNAMICS_ROUTINES.F90:3427 (setup branch)
    dt: float = 0.0
    filt: object = None                    # DFI state when inside the filter
    bc: object = None                      # boundary arrays (boundary group)
    tables: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    case_dir: object = None                # cases/<case>
    start: object = None                   # datetime of the initial time
    out_dir: object = None                 # run output (history, maps, receipt)
    comm: object = None                    # namgpu.parallel.Comm on several cards, else None (one card)
    slab: object = None                    # namgpu.slab.Slab on several cards, else None (one card)
    decomp: object = None                  # (mype, inpes, jnpes) for drv-loop-limits; None = one task
    driver: object = None                  # the Driver, for DFI integrations
    restart: bool = None                   # Fortran int_state%RESTART; resolved from cfg in __post_init__
    filter_active: bool = False            # True only while the TDFI filter runs (Fortran FILTER_METHOD /= 0)
    wrote_first_hist: bool = False         # Fortran domain_int_state%WROTE_1ST_HIST
    history_fields: object = None          # set by io-history-fields before each history write
    start_time: object = None              # datetime of the initial time (io-history labels files with it)
    ntimestep_rad: int = 0                 # Fortran NTIMESTEP_RAD (SOLVER_RUN 4938-4942): 0 at step 0, else n+1
    hdiff_on: bool = True                  # Fortran HDIFF_ON: 1 for forward integration (3807, 4516, 5163)
    physics_on: bool = True                # Fortran IF(INTEGER_DT>0): forward legs only

    def __post_init__(self):
        if self.restart is None:
            self.restart = restart_flag(self.cfg)

    @property
    def clock(self):
        """The model clock radiation reads (ctx.clock.jdate / .idate / .solhr).

        Same time phys-rad-driver's clock_jdat uses: start + NTIMESTEP*DT, as NMMB JDAT
        (yr, mon, day, tz, hr, min, sec, ms); IDAT = JDAT in a forecast (radupdate docstring).
        solhr = mod(hr + min/60, 24) as coszmn computes it. Run lead 0930Z: nothing set ctx.clock,
        so radiation stopped at its first call with AttributeError.
        """
        import datetime as _dt
        import types as _types
        start = self.start if self.start is not None else self.start_time
        if start is None:
            jd = list(self.tables.get("JDAT") or ())
            if len(jd) != 8:
                raise RuntimeError("ctx.clock: no ctx.start and no ctx.tables['JDAT']")
        else:
            t = start + _dt.timedelta(seconds=float(self.ntimestep) * float(self.dt or self.cfg.dt))
            jd = [t.year, t.month, t.day, 0, t.hour, t.minute, t.second, 0]
        solhr = math.fmod(float(jd[4]) + float(jd[5]) / 60.0, 24.0)
        return _types.SimpleNamespace(jdate=jd, idate=list(jd), solhr=solhr)


class Driver:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.units = load_units()
        self._mods = {}
        self.groups = (("first", FIRST_STEP), ("dyn", DYNAMICS), ("phys", PHYSICS), ("diag", DIAGNOSTICS))
        self._last_use = None
        self._phys_lu = None
        self.plan = FIRST_STEP + DYNAMICS + PHYSICS + DIAGNOSTICS
        missing = [e[0] for e in self.plan + [(u, "") for u in HISTORY_UNITS]
                   if e[0] not in self.units]
        if missing:
            raise KeyError(f"step plan names units not in UNITS.json: {sorted(set(missing))}")

    def _scratch_last_use(self):
        """{position in DYNAMICS: [scratch names whose last user in the dynamics plan is that entry]}.

        A name's users are the wrapper modules whose source names it ('NAME' / "NAME" / .NAME); its release
        point is the last of them in DYNAMICS. A name with a user outside the dynamics plan stays on the
        end-of-group release. Computed once; NAMGPU_SCRATCH_POISON=1 proves no later reader exists."""
        import re
        from .state import STEP_SCRATCH
        root = Path(__file__).resolve().parent
        dyn = [e[0] for e in DYNAMICS]
        src = {}
        for uid in dict.fromkeys(dyn):
            mod = self.units[uid].get("wrapper", "")
            f = root.parent / (mod.replace(".", "/") + ".py")
            if f.is_file():
                src[uid] = f.read_text()
        allsrc = {f: f.read_text() for f in root.rglob("*.py") if f.name not in ("state.py", "driver.py")}
        dyn_files = {str(root.parent / (self.units[u]["wrapper"].replace(".", "/") + ".py")) for u in src}
        out = {}
        for n in STEP_SCRATCH:
            pat = re.compile(r"[\x27\x22.]" + re.escape(n) + r"\b")
            users = [str(f) for f, t in allsrc.items() if pat.search(t)]
            if any(u not in dyn_files for u in users):
                continue                      # used outside the dynamics wrappers: end-of-group release
            last = max((i for i, u in enumerate(dyn) if u in src and pat.search(src[u])), default=None)
            if last is not None:
                out.setdefault(last, []).append(n)
        return out

    def _phys_last_use(self):
        """{position in PHYSICS: [PHYS_SCRATCH names whose last user is that entry]}. A helper module counts
        as its driver unit (myj/jsfc/noah/gwd -> phys-turbl-driver, bmj -> phys-bmj-driver, fer_hires ->
        phys-fer-driver); a name with any other user (radiation, dynamics, init, io, other steps) is kept."""
        import re
        from .state import PHYS_SCRATCH
        root = Path(__file__).resolve().parent
        owner = {"physics/driver/phys_turbl_driver.py": "phys-turbl-driver", "physics/myj/": "phys-turbl-driver",
                 "physics/jsfc/": "phys-turbl-driver", "physics/noah/": "phys-turbl-driver",
                 "physics/gwd/": "phys-turbl-driver", "physics/driver/phys_bmj_driver.py": "phys-bmj-driver",
                 "physics/bmj/": "phys-bmj-driver", "physics/driver/phys_fer_driver.py": "phys-fer-driver",
                 "physics/fer_hires/": "phys-fer-driver", "steps/drv_h_to_v_tend.py": "drv-h-to-v-tend"}
        files = {str(f.relative_to(root)): f.read_text() for f in root.rglob("*.py")
                 if f.name not in ("state.py", "driver.py")}
        phys = [e[0] for e in PHYSICS]
        out = {}
        for n in PHYS_SCRATCH:
            pat = re.compile(r"[\x27\x22.]" + re.escape(n) + r"\b")
            users = set()
            ok = True
            for rel, t in files.items():
                if not pat.search(t):
                    continue
                u = next((v for k, v in owner.items() if rel == k or (k.endswith("/") and rel.startswith(k))), None)
                if u is None:
                    ok = False
                    break
                users.add(u)
            if not ok or not users:
                continue
            last = max(i for i, u in enumerate(phys) if u in users)
            out.setdefault(last, []).append(n)
        return out

    def _release_dyn(self, pos):
        """Drop the dynamics locals whose last user in DYNAMICS is entry `pos` (32 GB fit path)."""
        if self._last_use is None:
            self._last_use = self._scratch_last_use()
        for _n in self._last_use.get(pos, ()):
            self.ctx.st.fields.pop(_n, None)

    def _wrapper(self, uid):
        if uid not in self._mods:
            self._mods[uid] = importlib.import_module(self.units[uid]["wrapper"])
        return self._mods[uid]

    def check_ready(self):
        """Import every wrapper on the plan once; report ALL missing ones in one message."""
        bad = []
        for uid in dict.fromkeys([e[0] for e in self.plan] + list(HISTORY_UNITS)):
            if self.units[uid]["mode"] == "skip":
                continue
            try:
                self._wrapper(uid)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{uid}: {type(e).__name__}: {e}")
        if bad:
            raise RuntimeError("units not ready:\n  " + "\n  ".join(bad))

    def _due(self, key: str) -> bool:
        c, n = self.ctx.cfg, self.ctx.ntimestep
        if key == "every":
            return True
        if key == "first":
            return self.ctx.first_step
        if key == "not_first":
            return not self.ctx.first_step
        if key == "not_first_hdiff":          # HDIFF_ON>0 (Fortran 3577, 4176) inside the not_firststep block
            return (not self.ctx.first_step) and bool(self.ctx.hdiff_on)
        if key == "q_tracer":
            # IDTADTQ (module_SOLVER_GRID_COMP.F90:2858-2862): IDTADT when global, 1 when regional.
            idtadq = c.idtad if c.global_ else 1
            return n % max(1, idtadq) == 0
        if key == "tracers":
            return n % max(1, c.idtadt) == 0
        if key == "nphs":
            return n % c.nphs == 0
        if key == "nprecip":
            return n % c.nprecip == 0
        if key == "radiation":
            # SW/LW gate on NTIMESTEP_RAD (SOLVER_RUN 5274-5312): 0 at step 0, n+1 after.
            nr = self.ctx.ntimestep_rad
            return nr % c.nrads == 0 or nr % c.nradl == 0
        if key == "update_water":             # SOLVER_RUN 5188-5199
            if str(c.microphysics).strip() not in ("fer", "fer_hires", "gfs", "wsm6", "thompson"):
                return False
            nr = self.ctx.ntimestep_rad
            return (nr % c.nrads == 0 or nr % c.nradl == 0 or n % c.nphs == 0
                    or n % c.nprecip == 0)
        if key == "max_fields":               # SOLVER_RUN 5000-5001: NSTEPS_PER_CHECK, FILTER_METHOD==0, 'fer_hires'
            nsc = max(2, int(math.floor(40.0 / float(c.dt) + 0.5)))   # NINT(40/DT), 818
            # FILTER_METHOD is set to 0 once the filter is done (module_NMM_GRID_COMP.F90:3257), so the
            # call is skipped only inside the filter legs (ctx.filter_active). MAX_FIELDS_HR is the
            # 'fer_hires' branch (module_SOLVER_GRID_COMP.F90:5004-5049).
            # The hourly reset (SOLVER_RUN 4963, NTIMESTEP == 0 or MOD(NTIMESTEP, NSTEPS_PER_RESET) == 0) sits
            # before that gate, on its own cadence, so the unit is also due on reset steps; io-max-fields
            # then resets and returns. At 12 km (dt 25 s) every reset step is a check step, so this adds no
            # call there; at 4 km (dt 25/3 s: check 5, reset 432) it keeps the reset NAM does.
            nrst = max(1, int(math.floor(float(c.raw.get("avg_max_length", 3600)) / float(c.dt) + 0.5)))
            due = n % nsc == 0 or n == 0 or n % nrst == 0
            return due and not self.ctx.filter_active and str(c.microphysics).strip() == "fer_hires"
        if key in ("cltend_pre", "cltend_post"):
            # CLTEND modes -1 and 0 (SOLVER_RUN 5705, 5925): inside IF(CALL_PRECIP), CALL_PRECIP =
            # MOD(NTIMESTEP,NPRECIP)==0 (5176), and only when NPRECIP>1 (cld_tend1, cld_tend2).
            return c.nprecip > 1 and n % c.nprecip == 0
        if key == "cltend_inc":
            # CLTEND mode 1 (SOLVER_RUN 6005): every step, NPRECIP>1 only.
            return c.nprecip > 1
        if key == "history":
            mins = c.raw.get("minutes_history", 60)
            steps = max(1, round(mins * 60.0 / c.dt))
            return n % steps == 0
        raise KeyError(key)

    def _indx(self, name: str) -> int:
        """A tracer slot from ctx.tables["indx"] (drv-tracer-index), or its host mirror."""
        tab = self.ctx.tables.get("indx")
        if tab is None:
            from .steps.drv_tracer_index import host_index  # noqa: PLC0415
            tab = host_index(self.ctx.cfg.microphysics, self.ctx.cfg.raw.get("num_tracers_chem", 0))
        return int(tab[name])

    def _resolve(self, v):
        """A plan kwarg: a literal int (icltend), an INDX_* tracer slot, or a config cadence
        (IDTADT, IDTADTQ)."""
        if isinstance(v, int):
            return v
        c = self.ctx.cfg
        if v == "KSE_TRACERS":                # SOLVER_RUN 4442-4443 (SPEC_ADV=T) vs 4456-4461 (SPEC_ADV=F)
            spec = str(c.raw.get("spec_adv", False)).strip().lower() in ("true", ".true.", "t", "yes", "1")
            return self._indx("NUM_TRACERS_TOTAL" if spec else "INDX_Q2")
        if v in ("nclod", "post_rdtemp", "post_turb"):     # drv-bucket-reset block selector
            return v
        if v == "IDTADT":
            return int(c.idtadt)
        if v == "IDTADTQ":   # module_SOLVER_GRID_COMP.F90:2858-2862
            return int(c.idtadt) if c.global_ else 1
        if v in ("before", "after"):   # drv-rime-factor-update branch selector
            return v
        return self._indx(v)

    def step(self):
        ctx = self.ctx
        devmem.forecast_start(ctx.stream)          # first step (TDFI or forecast) ends init: lift the init pool cap
        if not ctx.tables.get("filter_consts_bound"):
            # one array per int_state constant (FAH, FAD, HDACX/Y, HDACVX/Y): drv-filter-consts
            from .steps import drv_filter_consts  # noqa: PLC0415
            drv_filter_consts.bind(ctx)
        # Fortran NTIMESTEP_RAD (SOLVER_RUN 4938-4942), read by the radiation cadence and wrapper.
        ctx.ntimestep_rad = 0 if ctx.ntimestep == 0 else ctx.ntimestep + 1
        from . import parallel  # noqa: PLC0415
        from .steps import drv_halo_exch as halo  # noqa: PLC0415
        multi = parallel.get(ctx)                  # None on one card
        _fit = _fit_on()
        # Fit path, inside the TDFI legs: the filter's Dolph sums (1.3 GiB at 12 km) wait in pinned host memory
        # while the units in FILTER_SPILL_UNITS run (radiation: the filter-phase high-water mark), and are
        # copied back before the next dyn-sum. Byte copies; drv_dfi_tdfi_driver.spill_sums.
        _spill = _fit and bool(getattr(ctx, "filter_active", False))
        for name, entries in self.groups:
            if name == "phys" and not ctx.physics_on:    # IF(INTEGER_DT>0) (SOLVER_RUN 4929)
                continue
            for _pos, entry in enumerate(entries):
                uid, key = entry[0], entry[1]
                if ((uid == "drv-halo-exch" and multi is None)          # one card: no exchange, plan as before
                        or self.units[uid]["mode"] == "skip" or not self._due(key)
                        or (uid in CONFIG_GATES and not CONFIG_GATES[uid](ctx.cfg))):
                    if _fit and name == "dyn":
                        # A dynamics local whose last user is an entry not due this step (the not_first entries on
                        # the first step) was last used earlier in this step: release it here too. Before, the
                        # first-step PGFORCE/DHT locals (~2 GiB at 12 km) stayed to the end of the group and set
                        # the card peak. Every STEP_SCRATCH name is dropped at the group end anyway, so nothing
                        # crosses a step either way.
                        self._release_dyn(_pos)
                    continue
                # optional third element: kwargs (INDX_* tracer ranges, icltend, branch, bucket block, halo row)
                kw = ({k: (v if k in ("branch", "x") else self._resolve(v)) for k, v in entry[2].items()}
                  if len(entry) > 2 else {})
                timing.stamp(ctx, f"{uid}/{key}")      # stage start, for the run receipt
                if _spill and uid in FILTER_SPILL_UNITS:
                    from .steps import drv_dfi_tdfi_driver as _tdfi  # noqa: PLC0415
                    _tdfi.spill_sums(ctx)
                    self._wrapper(uid).run(ctx, **kw)
                    _tdfi.unspill_sums(ctx)
                else:
                    self._wrapper(uid).run(ctx, **kw)
                if halo.HALO_ALL and multi is not None:
                    halo.exchange_all(ctx)
                if halo.UNIT_HASH:
                    halo.hash_unit(ctx, f"{uid}/{key}" + (f"/{kw['x']}" if "x" in kw else ""))
                if _fit and uid in _UNIT_SCRATCH:
                    ctx.st.release_scratch(uid)      # that unit's locals back to the pool (state.py)
                if _fit and name == "phys":
                    if self._phys_lu is None:
                        self._phys_lu = self._phys_last_use()
                    for _n in self._phys_lu.get(_pos, ()):
                        ctx.st.fields.pop(_n, None)  # last physics user of this driver-level array is done
                if _fit and name == "dyn":
                    self._release_dyn(_pos)          # last dynamics user of these locals is done
            if name == "dyn" and _fit:
                ctx.st.release_scratch("dyn")    # dynamics locals back to the pool before physics (state.py)
                for _k in [k for k in ctx.tables if k.startswith(_TPFX) and not k.startswith(_TKEEP)]:
                    del ctx.tables[_k]
        ctx.first_step = False
        ctx.ntimestep += 1

    def write_history(self):
        """One history output: io-wind-rotate, io-history-fields (sets ctx.history_fields), io-history.
        The file is labelled with ctx.ntimestep at the call, so the caller sets the count first."""
        ctx = self.ctx
        # io-history-fields hands the 3-D derived fields over as io_history_fields.StreamField: each is computed
        # into one shared buffer (H_STREAM) right before its copy to the host, instead of eight full 3-D arrays
        # at once. Only this path sets it; tools that call io-history-fields directly get plain arrays.
        ctx.history_stream = True
        for uid in HISTORY_UNITS:
            # stamped like plan units, so the receipt splits the hourly history cost by unit instead of
            # folding it into the last step stage (it showed as a 'gap after drv-tqadjust')
            timing.stamp(ctx, f"{uid}/history")
            if uid == "io-history-fields":
                ctx.history_fields = self._wrapper(uid).run(ctx)
            else:
                self._wrapper(uid).run(ctx)
        if _fit_on():
            # history-only device buffers: io-history queued their D2H copies on this stream already
            ctx.history_fields = None
            ctx.st.release_scratch("history")

    def _initial_history(self):
        """Fortran history_output_0_a (NMM_INTEGRATE.F90:794-813): NTIMESTEP==0, not a restarted
        run, FILTER_METHOD==0, not yet written.  Restarted runs write it only with RST_OUT_00
        (history_output_0_b, 815-837), which needs the restart reader (not ported; see run())."""
        ctx = self.ctx
        if ctx.filter_active or ctx.wrote_first_hist or ctx.ntimestep != 0:
            return
        ctx.wrote_first_hist = True
        self.write_history()

    def run(self, hours: float, log=print):
        """Every unit of the step loop runs with ctx.stream as the CURRENT CuPy stream. Wrappers launch
        their kernels on ctx.stream (non_blocking) but do their elementwise CuPy work and host checks on
        the current stream; outside this scope that is the legacy null stream, which a non_blocking
        stream does not wait for, so reads raced the kernels (run lead 0958Z: phys-rrtm-getozn read
        RAD_PRSLK1 all zero, 61x32x954 of 61x32x954, right after grrad-setup wrote it; the
        run_rrtm_real harness passed because it wraps the call in `with ctx.stream`)."""
        import contextlib
        scope = self.ctx.stream if self.ctx.stream is not None else contextlib.nullcontext()
        with scope:
            return self._run(hours, log=log)

    def _run(self, hours: float, log=print):
        ctx = self.ctx
        ctx.dt = ctx.cfg.dt
        ctx.start_time = ctx.start
        from .state import release_init_leftovers  # noqa: PLC0415
        release_init_leftovers(ctx)                  # init-only scratch (state.py); idempotent
        if ctx.out_dir is None and ctx.start is not None:
            # Run output (history, receipt) goes to <work root>/runs/<start> unless the caller names one.
            ctx.out_dir = paths.work_root() / "runs" / f"{ctx.start:%Y%m%d_%H}"
        if ctx.restart:
            # Refused by name: RESET_ALARMS RESTARTED_RUN (NMM_INTEGRATE.F90:1682-1694), ALARM_RESTART
            # (1865-1875) and RST_OUT_00 (816-827) have no port yet, and no restart file is read or written.
            raise NotImplementedError("drv-time-loop: the NAM's own restart files are not read (woof-nam run "
                                      "--restart-every / --resume write and read woof-nam restart sets); not ported "
                                      "(NMM_INTEGRATE.F90 1682-1694, 1865-1875, 816-827)")
        nsteps = int(round(hours * 3600.0 / ctx.cfg.dt))
        steps_per_hour = max(1, int(round(3600.0 / ctx.cfg.dt)))
        t0 = time.perf_counter()
        self._initial_history()
        # A resumed run (woof-nam run --resume) starts at the restart set's step count and runs to the same end.
        n0 = int(ctx.ntimestep)
        rw = getattr(ctx, "restart_writer", None)
        for _ in range(max(0, nsteps - n0)):
            self.step()
            # Fortran: the clock is advanced inside the step, then history is written on the
            # post-advance count (NMM_INTEGRATE.F90:1153+).  step() has already incremented ntimestep.
            if not ctx.filter_active and self._due("history"):
                self.write_history()
            if rw is not None and not ctx.filter_active and rw.due(ctx.ntimestep):
                timing.stamp(ctx, "restart/stage")
                rw.write(ctx)                        # stream-ordered copy to pinned memory; files on a thread
            if ctx.ntimestep % steps_per_hour == 0:
                # once per forecast hour: the only host sync, for the progress line
                if ctx.stream is not None:
                    ctx.stream.synchronize()
                timing.drain(ctx)                    # timing stamps, read at the sync we already pay
                devmem.release(ctx.stream)           # unused pool blocks back to the driver once an hour
                wall = time.perf_counter() - t0
                log(f"fh {ctx.ntimestep * ctx.cfg.dt / 3600.0:6.2f}  step {ctx.ntimestep}"
                    f"  wall {wall:8.1f} s  {wall / max(1, ctx.ntimestep - n0) * 1e3:7.1f} ms/step")
        if ctx.stream is not None:
            ctx.stream.synchronize()
        from .io import io_history
        io_history.drain(ctx)                        # last history file is on disk before the receipt
        if rw is not None:
            rw.drain()                               # last restart set is on disk before the receipt
        from .steps import drv_halo_exch as _halo    # noqa: PLC0415
        if _halo.UNIT_HASH and ctx.out_dir is not None:
            Path(ctx.out_dir).mkdir(parents=True, exist_ok=True)
            log(f"unit hashes -> {_halo.write_hashes(ctx, ctx.out_dir)}")
        if ctx.out_dir is not None:
            rec = timing.write_receipt(ctx, Path(ctx.out_dir))
            log(f"timing receipt: {rec['stamped']} stamps, dropped {rec['dropped']}, "
                f"{rec['drains']} drains -> {Path(ctx.out_dir) / 'timing_receipt.json'}")
        return nsteps
