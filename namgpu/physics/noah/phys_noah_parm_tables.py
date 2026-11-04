"""phys-noah-parm-tables: LSM_PARM_INIT + SOIL_VEG_GEN_PARM (NAM v4.2.11, phys/module_LS_NOAHLSM.F90:1298-1547).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1). NOAA US government work; no licence file. The module's
NOAA header is kept as the notice.

The unit is a one-time host parse of three list-directed tables (VEGPARM.TBL, SOILPARM.TBL,
GENPARM.TBL) followed by one upload. No per-cell maths, so no kernel: the output is device vectors
bound at ctx.tables["noah_parm"] (flat keys "veg.<NAME>", "soil.<NAME>", "gen.<NAME>") and the host
copy at ctx.tables["noah_parm_host"] (scalars, bare, lucats, slcats). phys-noah-redprm reads those
bindings; nothing else re-parses the files.

The reads follow gfortran list-directed input, not a line-per-READ approximation:
  * a READ with n items takes n items from successive records; the rest of the last record is
    discarded; a record with too few items continues onto the next record;
  * a bare READ (19,*) skips one whole record;
  * READ (19,2000) with FORMAT (A4) takes the first four characters of one record, blank padded;
  * INTEGER items must be integers (gfortran rejects "4.0"); REAL items are converted once by
    strtof (libgfortran convert_real), so each value is the correctly rounded single;
  * a repeat count (r*c) is refused: no NAM table uses one and it would be misread.

VEG  (IVEGSRC 1 -> MMINLU 'IGBP'): skip 1; A4 LUTYPE; LUCATS, IINDEX; if LUTYPE /= MMINLU abort;
     LUCATS rows of 11 items (IINDEX, ALBTBL, Z0TBL, SHDTBL, NROTBL (INTEGER), RSTBL, RGLTBL,
     HSTBL, SNUPTBL, LAITBL, MAXALB); then five label+value pairs (TOPT, CMCMAX, CFACTR, RSMAX, BARE).
SOIL (MMINSL 'STAS'): skip 1; A4 SLTYPE; SLCATS, IINDEX; if SLTYPE /= 'STAS' abort; SLCATS rows of
     11 items (IINDEX, BB, DRYSMC, F11, MAXSMC, REFSMC, SATPSI, SATDK, SATDW, WLTSMC, QTZ); the rest
     of each record is ignored.
GEN:  skip 2; NUM_SLOPE; NUM_SLOPE slope values; then 11 label+value pairs (SBETA, FXEXP, CSOIL,
     SALP, REFDK, REFKDT, FRZK, ZBOT, CZIL, SMLOW, SMHIGH). The Fortran's LVCOEF line is not read.
The "abort" branches call ESMF_FINALIZE in Fortran; here they raise NoahParmTableError with the
Fortran message. Arrays are 0-based (Fortran index LC is stored at LC-1), padded to NLUS=50,
NSLTYPE=30, NSLOPE=30 as declared in module_LS_NOAHLSM.F90.

Tables come only from NAM_NOAH_FIX_DIR (the NCEP fix directory). There is no default and no synthetic
fallback: a missing directory is a refusal. The in-tree copies in namgpu/data/noah_tables are WRF
v4.6.1 files with a different VEG layout; the VEG parse refuses them.

Declared residue (stated, not hidden):
  * MPI_BCAST sends NROTBL as MPI_REAL while it is INTEGER (module_LS_NOAHLSM.F90:153, :1410). Both
    are 4 bytes, so the bits arrive unchanged; the port keeps the INTEGER type.
  * The NAM fix VEGPARM.TBL is not available. The VEG section follows the Fortran read rules, but its
    values have not been checked against the NAM file. That check is open (see UNITS.json note).
"""
from __future__ import annotations

import ctypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NLUS = 50          # module_LS_NOAHLSM.F90:151
NSLTYPE = 30       # :162
NSLOPE = 30        # :171
NSOIL = 4          # NAM soil layers used by NROTBL (module_LS_NOAHLSM.F90:632 NSOIL=num_soil_layers)
UNIT = "phys_noah_parm_tables"
ENV_DIR = "NAM_NOAH_FIX_DIR"   # NCEP fix directory holding VEGPARM.TBL, SOILPARM.TBL, GENPARM.TBL
MMINLU = {0: "USGS", 1: "IGBP"}   # LSM_PARM_INIT :1304-1306
MMINSL = "STAS"                    # LSM_PARM_INIT :1307
KEY_DEV = "noah_parm"
KEY_HOST = "noah_parm_host"
KEY_SOIL = "noah_soil"

_VEG_ARRAYS = ("ALBTBL", "Z0TBL", "SHDTBL", "RSTBL", "RGLTBL", "HSTBL",
               "SNUPTBL", "LAITBL", "MAXALB")
_SOIL_ARRAYS = ("BB", "DRYSMC", "F11", "MAXSMC", "REFSMC", "SATPSI", "SATDK", "SATDW",
                "WLTSMC", "QTZ")
_GEN_SCALARS = ("SBETA", "FXEXP", "CSOIL", "SALP", "REFDK", "REFKDT", "FRZK", "ZBOT",
                "CZIL", "SMLOW", "SMHIGH")

_INT_RE = re.compile(r"[+-]?\d+\Z")
_REAL_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?\Z")
_TOKEN_RE = re.compile(r"'[^']*'|\"[^\"]*\"|[^\s,/]+|/")


class NoahParmTableError(RuntimeError):
    """A table the NAM Fortran would abort on (ESMF_FINALIZE branch), or a read gfortran would fail."""


def _int(tok: str) -> int:
    """List-directed INTEGER item: gfortran rejects '4.0' and quoted text."""
    if not _INT_RE.match(tok):
        raise NoahParmTableError(f"bad integer {tok!r} in list-directed input (gfortran rejects it)")
    return int(tok)


_STRTOF = None


def _strtof(tok: str) -> np.float32:
    """List-directed REAL(4) item: libgfortran converts with strtof, so the value is rounded once."""
    global _STRTOF
    if not _REAL_RE.match(tok):
        raise NoahParmTableError(f"bad real {tok!r} in list-directed input")
    if _STRTOF is None:
        f = ctypes.CDLL(None).strtof
        f.restype = ctypes.c_float
        f.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
        _STRTOF = f
    raw = tok.replace("d", "e").replace("D", "E").encode()
    end = ctypes.c_char_p()
    val = _STRTOF(raw, ctypes.byref(end))
    if end.value not in (b"", None):
        raise NoahParmTableError(f"bad real {tok!r}: strtof stopped at {end.value!r}")
    return np.float32(val)


def _tokens(line: str) -> list[str]:
    """List-directed items of one record: blanks and commas separate, a slash ends the input."""
    out = []
    for m in _TOKEN_RE.finditer(line):
        t = m.group(0)
        if t == "/":
            break
        if "*" in t and not t.startswith(("'", '"')):
            raise NoahParmTableError(f"repeat count {t!r} in list-directed input is not supported")
        out.append(t)
    return out


class _Reader:
    def __init__(self, path: Path):
        self.path = path
        self.lines = path.read_text().splitlines()
        self.pos = 0

    def _record(self) -> str:
        if self.pos >= len(self.lines):
            raise NoahParmTableError(f"{self.path}: unexpected end of file at record {self.pos + 1}")
        line = self.lines[self.pos]
        self.pos += 1
        return line

    def read(self, n: int) -> list[str]:
        """One list-directed READ of n items. n == 0 is a bare READ (19,*): one record skipped."""
        if n == 0:
            self._record()
            return []
        toks: list[str] = []
        while len(toks) < n:
            toks.extend(_tokens(self._record()))
        return toks[:n]

    def skip(self, n: int = 1) -> None:
        for _ in range(n):
            self.read(0)

    def a4(self) -> str:
        """READ (19,2000) LUTYPE with FORMAT (A4): the first four characters, blank padded."""
        return self._record()[:4].ljust(4)

    def integers(self, n: int) -> list[int]:
        return [_int(t) for t in self.read(n)]

    def reals(self, n: int) -> list[np.float32]:
        return [_strtof(t) for t in self.read(n)]


@dataclass
class NoahParmTables:
    mminlu: str
    veg: dict = field(default_factory=dict)     # name -> array [NLUS] plus scalars
    soil: dict = field(default_factory=dict)    # name -> float32 [NSLTYPE]
    gen: dict = field(default_factory=dict)     # SLOPE_DATA [NSLOPE] and the scalars
    lucats: int = 0
    bare: int = 0
    slcats: int = 0
    slpcats: int = 0

    def arrays(self) -> dict:
        out = {}
        out.update({f"veg.{k}": v for k, v in self.veg.items()})
        out.update({f"soil.{k}": v for k, v in self.soil.items()})
        out.update({f"gen.{k}": v for k, v in self.gen.items()})
        return out

    def assert_finite(self) -> None:
        for name, arr in self.arrays().items():
            a = np.asarray(arr)
            if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a)):
                raise NoahParmTableError(f"non-finite value in {name}")


def _veg(r: _Reader, mminlu: str, out: NoahParmTables) -> None:
    r.skip(1)                                   # READ (19,*)
    lutype = r.a4()                             # READ (19,2000) LUTYPE
    lucats, _iindex = r.integers(2)             # READ (19,*) LUCATS, IINDEX
    if lucats <= 0:
        raise NoahParmTableError("Problem reading vegetation tables (VEGPARM.TBL)")
    if lutype != mminlu.ljust(4):
        raise NoahParmTableError(
            f"Mismatched vegetation tables (VEGPARM.TBL): MMINLU={mminlu} and LUTYPE={lutype.strip()} "
            "(LU_test, module_LS_NOAHLSM.F90:1373-1395; only the first section is read)")
    if lucats > NLUS:
        raise NoahParmTableError(f"LUCATS={lucats} exceeds NLUS={NLUS}")
    arr = {k: np.zeros(NLUS, dtype=np.float32) for k in _VEG_ARRAYS}
    arr["NROTBL"] = np.zeros(NLUS, dtype=np.int32)
    for lc in range(lucats):
        # IINDEX ALBTBL Z0TBL SHDTBL NROTBL RSTBL RGLTBL HSTBL SNUPTBL LAITBL MAXALB
        v = r.read(11)
        _int(v[0])
        nrot = _int(v[4])
        if not 1 <= nrot <= NSOIL:
            raise NoahParmTableError(f"VEGPARM row {lc + 1}: NROTBL={nrot} outside 1..NSOIL={NSOIL}")
        arr["NROTBL"][lc] = nrot
        arr["ALBTBL"][lc] = _strtof(v[1])
        arr["Z0TBL"][lc] = _strtof(v[2])
        arr["SHDTBL"][lc] = _strtof(v[3])
        arr["RSTBL"][lc] = _strtof(v[5])
        arr["RGLTBL"][lc] = _strtof(v[6])
        arr["HSTBL"][lc] = _strtof(v[7])
        arr["SNUPTBL"][lc] = _strtof(v[8])
        arr["LAITBL"][lc] = _strtof(v[9])
        arr["MAXALB"][lc] = _strtof(v[10])
    r.skip(1); topt = r.reals(1)[0]             # label record skipped, then the value
    r.skip(1); cmcmax = r.reals(1)[0]
    r.skip(1); cfactr = r.reals(1)[0]
    r.skip(1); rsmax = r.reals(1)[0]
    r.skip(1); bare = r.integers(1)[0]
    out.veg = {**arr,
               "TOPT_DATA": topt, "CMCMAX_DATA": cmcmax,
               "CFACTR_DATA": cfactr, "RSMAX_DATA": rsmax}
    out.lucats, out.bare = lucats, bare


def _soil(r: _Reader, mminsl: str, out: NoahParmTables) -> None:
    r.skip(1)                                   # READ (19,*)
    sltype = r.a4()                             # READ (19,2000) SLTYPE
    slcats, _iindex = r.integers(2)             # READ (19,*) SLCATS, IINDEX
    if slcats <= 0:
        raise NoahParmTableError("Problem reading soil tables (SOILPARM.TBL)")
    if sltype != mminsl.ljust(4):
        raise NoahParmTableError(
            f"Mismatched soil tables (SOILPARM.TBL): MMINSL={mminsl} and LUTYPE={sltype.strip()}")
    if slcats > NSLTYPE:
        raise NoahParmTableError(f"SLCATS={slcats} exceeds NSLTYPE={NSLTYPE}")
    arr = {k: np.zeros(NSLTYPE, dtype=np.float32) for k in _SOIL_ARRAYS}
    for lc in range(slcats):
        # IINDEX BB DRYSMC F11 MAXSMC REFSMC SATPSI SATDK SATDW WLTSMC QTZ; the rest is ignored
        v = r.read(11)
        _int(v[0])
        for k, s in zip(_SOIL_ARRAYS, v[1:11]):
            arr[k][lc] = _strtof(s)
    out.soil = arr
    out.slcats = slcats


def _gen(r: _Reader, out: NoahParmTables) -> None:
    r.skip(2)                                   # READ, READ ("General Parameters", "SLOPE_DATA")
    num_slope = r.integers(1)[0]                # READ (19,*) NUM_SLOPE
    if not 0 < num_slope <= NSLOPE:
        raise NoahParmTableError(f"NUM_SLOPE={num_slope} outside 1..{NSLOPE}")
    slope = np.zeros(NSLOPE, dtype=np.float32)
    for lc in range(num_slope):
        slope[lc] = r.reals(1)[0]               # READ (19,*) SLOPE_DATA(LC)
    gen = {"SLOPE_DATA": slope}
    for name in _GEN_SCALARS:
        r.skip(1)                               # the "<NAME>_DATA" label record
        gen[f"{name}_DATA"] = r.reals(1)[0]
    out.gen = gen
    out.slpcats = num_slope


def parse_tables(tbl_dir: str | Path, ivegsrc: int = 1) -> NoahParmTables:
    """Read VEGPARM.TBL, SOILPARM.TBL, GENPARM.TBL the way SOIL_VEG_GEN_PARM does."""
    if ivegsrc not in MMINLU:
        raise NoahParmTableError(f"IVEGSRC={ivegsrc}: LSM_PARM_INIT leaves MMINLU blank (no table)")
    tbl_dir = Path(tbl_dir)
    out = NoahParmTables(mminlu=MMINLU[ivegsrc])
    _veg(_Reader(tbl_dir / "VEGPARM.TBL"), MMINLU[ivegsrc], out)
    _soil(_Reader(tbl_dir / "SOILPARM.TBL"), MMINSL, out)
    _gen(_Reader(tbl_dir / "GENPARM.TBL"), out)
    out.assert_finite()
    return out


def parse_soil_table(tbl_dir: str | Path, ivegsrc: int = 1) -> NoahParmTables:
    """SOILPARM.TBL only (the STAS section that SOIL_VEG_GEN_PARM reads). For units that need only
    the soil-type columns (QTZ, MAXSMC, ...). The VEG and GEN sections are not read, so this does
    not depend on the vegetation table matching the requested IVEGSRC."""
    if ivegsrc not in MMINLU:
        raise NoahParmTableError(f"IVEGSRC={ivegsrc}: LSM_PARM_INIT leaves MMINLU blank (no table)")
    out = NoahParmTables(mminlu=MMINLU[ivegsrc])
    _soil(_Reader(Path(tbl_dir) / "SOILPARM.TBL"), MMINSL, out)
    out.assert_finite()
    return out


def run_soil(ctx, tbl_dir: str | Path | None = None):
    """Soil-only bind: device arrays at ctx.tables["noah_soil"] with flat keys "soil.<NAME>"."""
    if KEY_SOIL in ctx.tables:
        return ctx.tables[KEY_SOIL]
    tbl_dir = tbl_dir or os.environ.get(ENV_DIR)
    if not tbl_dir:
        raise NoahParmTableError(f"no NAM SOILPARM directory: set {ENV_DIR}")
    host = parse_soil_table(tbl_dir, int(ctx.cfg.raw["ivegsrc"]))
    import cupy as cp
    dev = {f"soil.{name}": cp.asarray(arr) for name, arr in host.soil.items()}
    ctx.tables[KEY_SOIL] = dev
    return dev


def run(ctx, tbl_dir: str | Path | None = None, ivegsrc: int | None = None):
    """Parse once. Binds the device copy at ctx.tables["noah_parm"] and the host copy at
    ctx.tables["noah_parm_host"]; a later call returns the existing binding."""
    if KEY_DEV in ctx.tables:
        return ctx.tables[KEY_DEV]
    tbl_dir = tbl_dir or os.environ.get(ENV_DIR)
    if not tbl_dir:
        raise NoahParmTableError(f"no NAM VEGPARM/SOILPARM/GENPARM directory: set {ENV_DIR}")
    if ivegsrc is None:
        ivegsrc = int(ctx.cfg.raw["ivegsrc"])
    host = parse_tables(tbl_dir, ivegsrc)
    import cupy as cp
    dev = {name: cp.asarray(arr) for name, arr in host.arrays().items()}
    ctx.tables[KEY_HOST] = host
    ctx.tables[KEY_DEV] = dev
    return dev
