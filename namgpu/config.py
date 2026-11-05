"""NAM configfile parser.

The NAM 12 km parent is configured by ``parm/nam_configfile`` (copied unmodified from the
NAM clone).  Lines are ``key: value   # comment``.  Placeholders that the NAM job script
substitutes with sed (``YSTART``, ``LENGTH``, ``FILTEROPTION`` ...) are filled from
``SCRIPT_SUBSTITUTIONS`` below, which records what ``scripts/exnam_fcst.sh`` sets for a
12 km parent free forecast (tm00), so a bare ``load()`` gives the operational parent run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import paths  # noqa: E402
from .paths import PARM  # noqa: E402  (shipped inside the package)

DEFAULT_CONFIGFILE = PARM / "nam_configfile"
# Domain selection: NAMGPU_CONFIGFILE names the configfile that every default config.load() reads (a relative
# path is looked up from the current folder, then the work root, then the package: parm/nam_configfile_4km), so one setting moves the whole run (init, bocos, driver, done-rule stubs,
# every rank) to another grid. Unset = parm/nam_configfile, the NAM 12 km parent, unchanged; an explicit path
# given to load() always wins. A domain configfile may carry two port keys NAM does not have: static_npz (the
# init-static-build file for that grid) and domain_name (the case boco folder is boco-<domain_name>, so the
# 12 km boco files are never overwritten).
CONFIG_ENV = "NAMGPU_CONFIGFILE"


def configfile_path(path=None) -> Path:
    """The configfile to read: an explicit non-default path; else NAMGPU_CONFIGFILE; else parm/nam_configfile."""
    p = Path(path) if path is not None else DEFAULT_CONFIGFILE
    if p == DEFAULT_CONFIGFILE and os.environ.get(CONFIG_ENV):
        p = Path(os.environ[CONFIG_ENV])
    if p.is_absolute() or p.exists():
        return p
    for base in (paths.work_root(), paths.PKG):
        if (base / p).exists():
            return base / p
    return paths.PKG / p

# What exnam_fcst.sh substitutes for the tm00 parent forecast (scripts/exnam_fcst.sh:270-300).
SCRIPT_SUBSTITUTIONS = {
    "FILTEROPTION": "3",      # exnam_fcst.sh:280-285, TDFI on every cold-start forecast
    "USE_RADAR": "0",         # no GSI radar heating (we start from the archived analysis)
    "PFLAG": "false",         # no precip assimilation in the free forecast
    "FREE": "true",
    "REST": "false",
    "OUTRST": "false",
    "WRTLAST": "false",
    "RESINT": "720",
    "NUMDOM": "1",
    "NCHILD": "0",
    "PRECIPBUCKET": "3",
    "HEATBUCKET": "3",
    "CLOUDBUCKET": "3",
    "LWBUCKET": "3",
    "SWBUCKET": "3",
    "EVAPBUCKET": "3",
}


def _coerce(text: str):
    t = text.strip().strip("'")
    low = t.lower()
    if low in ("true", ".true."):
        return True
    if low in ("false", ".false."):
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def parse_configfile(path: Path, subs: dict[str, str] | None = None) -> dict:
    subs = dict(SCRIPT_SUBSTITUTIONS, **(subs or {}))
    out: dict = {}
    for raw in configfile_path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip()
        if val in subs:
            val = subs[val]
        out[key.strip()] = _coerce(val)
    return out


@dataclass
class NamConfig:
    raw: dict
    # grid
    im: int
    jm: int
    lm: int
    tph0d: float
    tlm0d: float
    wbd: float
    sbd: float
    # time
    dt: float
    filt_dt: float
    nphs: int
    nprecip: int
    nrads: int
    nradl: int
    idtad: int
    idtadt: int
    # dynamics
    lnsh: int
    lnsv: int
    secadv: bool
    smag2: float
    codamp: float
    wcor: float
    hydro: bool
    global_: bool
    # physics
    microphysics: str
    convection: str
    turbulence: str
    sfc_layer: str
    land_surface: str
    shortwave: str
    longwave: str
    gwdflg: bool
    rhgrd: float
    np3d: int
    filter_method: int
    # hybrid levels (nam_namelist.nps &nemsinterp)
    pt: float = 200.0
    ptsgm: float = 30000.0
    coord_levs: list = field(default_factory=list)
    # NPS nemsinterp flags that change the cold-start mapping (nam_namelist.nps &nemsinterp;
    # defaults are gridinfo_module.F90 defaults).
    vcoord: int = 1              # 1 = PD = PSFC - PT (NAM); 0 = PD = PSFC - PDTOP - PT
    direct_temp: bool = True     # .true. = TMP already direct temperature (no devirtualisation)
    do_clouds: bool = False      # .false. = NPS writes CWM = 0 everywhere (CONUS coldstart)
    # RRTM control values NAM sets in module_SOLVER_GRID_COMP.F90 (NAM v4.2.11 83df4e13c1), not in a file.
    iaer: int = 11               # :10658 3-digit aerosol flag
    iems: int = 0                # :10681 surface emissivity scheme
    ntcw: int = 3                # :10684 cloud condensate index (>0 prognostic cloud, icldflg 1)
    ntoz: int = 0                # :10691 0 = climatological ozone (loz1st path)
    # physparam.f defaults, not in parm/nam_configfile (the NAM 12 km configfile does not set them).
    # Read by phys-rrtm-radupdate (radupdate_nmmb) and its sibling units.
    isolar: int = 0             # solar constant scheme control flag (physparam.f:102)
    ictmflg: int = 0            # external data time/date control flag (physparam.f:126)
    ioznflg: int = 0            # runtime value: rad_initialize_nmmb.f:139 ioznflg = NTOZ = 0 (physparam.f:127 default 1 is overwritten)
    ico2flg: int = 1            # runtime value: rad_initialize_nmmb.f:138 ico2flg = ICO2 = 1 (module_SOLVER_GRID_COMP.F90:10656)
    ivflip: int = 0             # runtime value: rad_initialize_nmmb.f:183 ivflip = IFLIP = 0 (module_SOLVER_GRID_COMP.F90:10714)
    isubcsw: int = 0            # module_SOLVER_GRID_COMP.F90:10697 ISUBCSW=0 (McICA sub-columns off on the NAM path)
    iovrsw: int = 1             # module_SOLVER_GRID_COMP.F90:10693 IOVR_SW=1 (max-random overlap, sw)
    iovrlw: int = 1             # module_SOLVER_GRID_COMP.F90:10695 IOVR_LW=1 (max-random overlap, lw)
    isubclw: int = 0            # module_SOLVER_GRID_COMP.F90:10702 ISUBCLW=0 (McICA sub-columns off, lw)
    ialbflg: int = 2            # module_SOLVER_GRID_COMP.F90:10676 IALB=2 -> rad_initialize ialbflg (externally provided albedo)
    co2dat_file: str = "co2historicaldata_2004.txt"  # physparam.f:133 (year written at 19:22)
    co2gbl_file: str = "co2historicaldata_glob.txt"  # physparam.f:134
    co2usr_file: str = "co2userdata.txt"             # physparam.f:135
    co2cyc_file: str = "co2monthlycyc.txt"           # physparam.f:136
    lfdncmp: bool = True         # swrad always fills FDNCMP: grrad_nmmb.f:1910 scmpsw is unconditional (phys-rrtm-grrad-sw)

    # A grid-associated nest (module_NESTING.F90:4263-4592 SET_NEST_GRIDS) sits off centre on its parent's pole, so its
    # spacing cannot be derived from wbd/sbd: its configfile carries the port keys dlmd/dphd (namgpu/nest/domain.py
    # nest_grid_associated). Without them a grid is symmetric about its pole and the derived value is used, as before.
    @property
    def dlmd(self) -> float:
        if "dlmd" in self.raw:
            return float(self.raw["dlmd"])
        return -2.0 * self.wbd / (self.im - 1)

    @property
    def dphd(self) -> float:
        if "dphd" in self.raw:
            return float(self.raw["dphd"])
        return -2.0 * self.sbd / (self.jm - 1)


def static_npz(cfg) -> Path | None:
    """The domain's own static file (configfile key static_npz), or None for the NAM grid."""
    v = cfg.raw.get("static_npz")
    if not v:
        return None
    p = Path(str(v))
    return p if p.is_absolute() else paths.work_root() / p


def boco_dirname(cfg) -> str:
    """Case boco folder name: boco for the NAM grid, boco-<domain_name> for another domain."""
    d = cfg.raw.get("domain_name")
    return f"boco-{d}" if d else "boco"


def grid_stamp(cfg) -> dict:
    """The grid a boco folder was built on (written into its source marker and compared on reuse): the case folder
    boco-<domain_name> is shared by every domain of that cycle with the same name, so a folder built on another
    grid (other jm, or a centre that rounds to the same name) must be rebuilt, never reused (review round 3, #1)."""
    return {"im": int(cfg.im), "jm": int(cfg.jm), "lm": int(cfg.lm), "tph0d": round(float(cfg.tph0d), 4),
            "tlm0d": round(float(cfg.tlm0d), 4), "wbd": round(float(cfg.wbd), 6), "sbd": round(float(cfg.sbd), 6)}


def parse_nps_coord_levs(path: Path) -> tuple[float, float, list[float]]:
    pt = ptsgm = None
    levs: list[float] = []
    for raw in Path(path).read_text().splitlines():
        s = raw.strip().rstrip(",")
        if s.startswith("pt ") or s.startswith("pt="):
            pt = float(s.split("=", 1)[1])
        elif s.startswith("ptsgm"):
            ptsgm = float(s.split("=", 1)[1])
        elif s.startswith("coord_levs"):
            levs = [float(x) for x in s.split("=", 1)[1].split(",") if x.strip()]
    return pt, ptsgm, levs


def parse_nps_flags(path: Path) -> dict:
    """vcoord, direct_temp and do_clouds from the &nemsinterp namelist (absent keys -> NPS defaults)."""
    out: dict = {}
    for raw in Path(path).read_text().splitlines():
        key, _, val = raw.strip().rstrip(",").partition("=")
        key, val = key.strip(), val.strip().lower()
        if key == "vcoord":
            out["vcoord"] = int(val)
        elif key in ("direct_temp", "do_clouds"):
            out[key] = val in (".true.", ".t.", "t", "true")
    return out


def row_window(r: dict) -> dict:
    """NAMGPU_ROW_WINDOW=a:b (1-based rows of the configured grid, symmetric about its rotated equator): the
    centre-band sub-domain of the same grid (MULTICARD-PLAN.md section 10.2, the 954 x 279 proof band = 279:557 of
    the 12 km parent, sbd -15.012). jm and sbd change, every other key stays; jm_full and row_window record the
    parent so the native static decode can cut the same rows. Symmetric only: config derives dphd from sbd and jm,
    so an off-centre window would change the row spacing."""
    import os
    win = os.environ.get("NAMGPU_ROW_WINDOW", "").strip()
    if not win:
        return r
    a, b = (int(x) for x in win.split(":"))
    jf = int(r["jm"])
    if not (1 <= a <= b <= jf) or (a - 1) != (jf - b):
        raise ValueError(f"NAMGPU_ROW_WINDOW={win}: rows must lie in 1..{jf} and be symmetric (a-1 == jm-b) so the "
                         "row spacing dphd = -2 sbd/(jm-1) is unchanged")
    dph = -2.0 * float(r["sbd"]) / (jf - 1)
    r = dict(r, jm_full=jf, row_window=(a, b), jm=b - a + 1, sbd=round(float(r["sbd"]) + (a - 1) * dph, 6))
    return r


def load(configfile: Path | None = None, npsfile: Path | None = None) -> NamConfig:
    r = row_window(parse_configfile(configfile or PARM / "nam_configfile"))
    pt, ptsgm, levs = parse_nps_coord_levs(npsfile or PARM / "nam_namelist.nps")
    flags = parse_nps_flags(npsfile or PARM / "nam_namelist.nps")
    dt = r["dt_int"] + (r["dt_num"] / r["dt_den"] if r["dt_den"] else 0.0)
    fdt = r["filt_dt_int"] + (r["filt_dt_num"] / r["filt_dt_den"] if r["filt_dt_den"] else 0.0)
    return NamConfig(
        raw=r, im=r["im"], jm=r["jm"], lm=r["lm"], tph0d=r["tph0d"], tlm0d=r["tlm0d"],
        wbd=r["wbd"], sbd=r["sbd"], dt=dt, filt_dt=fdt, nphs=r["nphs"], nprecip=r["nprecip"],
        nrads=r["nrads"], nradl=r["nradl"], idtad=r["idtad"], idtadt=r["idtadt"],
        lnsh=r["lnsh"], lnsv=r["lnsv"], secadv=r["secadv"], smag2=r["smag2"],
        codamp=r["codamp"], wcor=r["wcor"], hydro=r["hydro"], global_=r["global"],
        microphysics=r["microphysics"], convection=r["convection"],
        turbulence=r["turbulence"], sfc_layer=r["sfc_layer"],
        land_surface=r["land_surface"], shortwave=r["shortwave"], longwave=r["longwave"],
        gwdflg=r["gwdflg"], rhgrd=r["rhgrd"], np3d=r["np3d"],
        filter_method=int(r["filter_method"]), pt=pt, ptsgm=ptsgm, coord_levs=levs,
        **flags,
    )


if __name__ == "__main__":
    c = load()
    print({k: v for k, v in c.__dict__.items() if k not in ("raw", "coord_levs")})
    print("dlmd", c.dlmd, "dphd", c.dphd, "nlev_int", len(c.coord_levs))
