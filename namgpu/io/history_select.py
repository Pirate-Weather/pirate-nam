"""Which fields each history file carries: `woof-nam run --history-fields <preset|list> --full-every <hours>`.

Presets (names as in the history file, io_history_fields.py):
  full      every field (the whole history, as before this option existed).
  products  what woof-nam's own readers and WOOF's column store read: the maps (composite reflectivity, 2 m
            temperature and dewpoint, 10 m and 850 hPa wind, MSLP, 1 h precipitation), the side-by-side panels,
            the any-point sounding column store (T, QVAPOR, the four condensates, PRES, PH, U, V on the model
            levels) and `woof-nam products` (the awphys GRIB2: also W, REFL_10CM, and the physics carriers of
            PRODUCT_CARRIERS: fluxes, surface layer, radiation, land/snow state, cloud cover, TKE and the hourly
            extremes). Left out: any field no reader uses.
  state     the 3-D model state on the model levels (T, QVAPOR, the condensates, PRES, PH, U, V, W) plus PSFC
            and HGT, without the 2-D diagnostics.
  a list    comma-separated field names, e.g. T2,Q2,U10,V10,RAINNC,RAINC,COMPOSITE.
XLAT, XLONG, XTIME and Times are in every file, and so is any field another option adds to the history (a nest
hand-off set): the presets choose among the base fields only. A field the model has not produced at that hour (MSLP before the
first post, COMPOSITE before the first microphysics call) is absent, as before.

--full-every N (hours, default 3): every Nth output hour (0, N, 2N, ...) is written with every field whatever the
preset, so a run always has full states to restart analysis from and to check against. 0 = never.
Default: --history-fields products --full-every 3h.
"""
from __future__ import annotations

# The NCEP product carriers io-history-fields writes when the state holds them (io_history_fields.py:
# PRODUCT_CARRIERS_2D/3D and TKE_PBL): what woof-nam products (the awphys GRIB2) reads beyond the maps' set.
PRODUCT_CARRIERS = (
    "TWBS", "QWBS", "USTAR", "Z0", "PBLH", "AKMS", "ALBEDO", "RSWIN", "RLWIN", "RSWOUT", "RADOT", "THS", "SST",
    "SICE", "SM", "SNO", "SI", "SNOWC", "CMC", "TG", "VEGFRC", "ISLTYP", "IVGTYP", "CFRACL", "CFRACM", "CFRACH",
    "PSHLTR", "POTEVP", "SFCEVP",
    "T02MAX", "T02MIN", "RH02MAX", "RH02MIN", "SPD10MAX", "REFDMAX", "UPVVELMAX", "DNVVELMAX", "PRATEMAX",
    "UPHLMAX25", "UPHLMIN25", "UPHLMAX03", "UPHLMIN03", "CLDFRA", "TKE_PBL", "U10MAX", "V10MAX", "W_TOT", "PINT",
    "SMSTAV", "SMSTOT", "CZEN", "CZMEAN", "SIGT4", "CNVBOT", "CNVTOP", "DRAGCO",
)
PRESETS = {
    "full": None,
    # W is in: the awphys VVEL/DZDT records need it every hour
    "products": ("PSFC", "HGT", "T2", "Q2", "U10", "V10", "MSLP", "RAINNC", "RAINC", "COMPOSITE",
                 "T", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "PRES", "PH", "U", "V", "W", "REFL_10CM")
    + PRODUCT_CARRIERS,
    "state": ("PSFC", "HGT", "T", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "PRES", "PH", "U", "V", "W"),
}
DEFAULT_PRESET = "products"
DEFAULT_FULL_EVERY = "3h"
# Every name io-history-fields can write (io_history_fields.run), for checking a list before the run starts.
KNOWN = ("PSFC", "HGT", "T", "PRES", "PH", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "U10", "V10", "RAINNC",
         "RAINC", "U", "V", "W", "T2", "Q2", "MSLP", "REFL_10CM", "COMPOSITE") + PRODUCT_CARRIERS


def parse(spec: str):
    """A preset name or a comma list -> (label, names or None for every field). Unknown names stop the run."""
    s = (spec or DEFAULT_PRESET).strip()
    if s in PRESETS:
        return s, PRESETS[s]
    names = tuple(n.strip() for n in s.split(",") if n.strip())
    bad = [n for n in names if n not in KNOWN]
    if bad or not names:
        raise SystemExit(f"--history-fields {spec!r}: not a preset ({', '.join(PRESETS)}) and not history field "
                         f"names: {bad or 'empty'}; the fields are {', '.join(KNOWN)}")
    return "list", names


class Selection:
    def __init__(self, spec: str, full_every_steps: int):
        self.label, self.names = parse(spec)
        self.spec = spec
        self.full_every = int(full_every_steps)

    def pick(self, ntimestep: int, fields: dict) -> dict:
        if self.names is None or (self.full_every > 0 and int(ntimestep) % self.full_every == 0):
            return fields
        # only the base fields are chosen; a field another option adds (e.g. a nest hand-off set) always stays
        return {k: v for k, v in fields.items() if k in self.names or k not in KNOWN}

    def receipt(self) -> dict:
        return {"fields": self.spec, "preset": self.label, "names": list(self.names) if self.names else "all",
                "full_every_steps": self.full_every}


def select(ctx, fields: dict) -> dict:
    sel = getattr(ctx, "history_select", None)
    return fields if sel is None else sel.pick(int(ctx.ntimestep), fields)
