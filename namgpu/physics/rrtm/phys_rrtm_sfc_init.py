"""phys-rrtm-sfc-init: surface albedo and emissivity scheme init (sfc_init, sfcinit).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_surface_nmmb.f:117-401 (subroutines
sfc_init and sfcinit of module_radiation_surface_nmmb). NOAA US government work; the NOAA module
history block is kept as the notice in the kernel file.
Kernels: namgpu/kernels/phys_rrtm_sfc_init.cu (sfc_init_flags, sfc_init_idxems).

NMMB calls sfc_init(me) from grrad_nmmb.f:413 (mode 0). sfcinit(LMAX, iflip, IALB, IEMS, me) is the
GFS/FIM variant (mode 1); it is ported because the unit covers both routines.

In (run arguments):
  mode      0 = sfc_init (NMMB), 1 = sfcinit
  ialb      mode 1 only: IALB, surface albedo scheme (0 climatology, 1 MODIS, 2 external).
  iems      mode 1 only: IEMS (0 or 1). Other values: Fortran stop.
  ems_dir   directory the Fortran inquire() runs in (default the working directory, as in NMMB).
  iemslw_prev  mode 1 only: prior module value of iemslw (default 0), kept by sfcinit.
Mode 0 reads IALB and IEMS from ctx.tables['rad_physparam'] (slots RI_IALBFLG, RI_IEMSFLG, written by
phys-rrtm-rad-initialize). Those are the module values (ialbflg, iemsflg from physparam) that sfc_init
reads and never assigns. The table must exist first (RuntimeError otherwise). Mode 0 uses
iemslw = mod(iemsflg, 10) with Fortran (truncating) semantics; any value other than 0 or 1 stops.
Out (ctx.tables, allocated once):
  sfc_flags    int32[3]          [ialbflg, iemsflg, iemslw]. phys-rrtm-setemis reads sfc_flags[0] and
                                 sfc_flags[2]; this is how sfc_init's results reach setemis.
  sfc_idxems   int32[180, 360]   emissivity-type index, [j, i]; only present when a table is read.
                                 phys-rrtm-setemis reads it only when sfc_flags[2] == 1.
run() returns (flags_dev, idxems_dev_or_None, info dict).

This unit computes no surface-albedo components. ALVSF, ALNSF, ALVWF, ALNWF, FACSF and FACWF are
phys-rrtm-setalb inputs supplied by the caller; no unit in the tree produces them yet.

Host work (init only, not on the step path): the physparam read (26 ints) and the file read happen
here, because inquire/open/read are host I/O. The file is sfc_emissivity_idx.txt (physparam
semis_file). Line 1 is a text tag. The next records are 80-character lines read with format 80i1
into idxems(360,180) in column-major order. The NAM run links the operational table into this name
(exngac_fcst.sh: EMMISSIVITY_FILE); that table is not on node-2 and is not needed for the default
path (IEMS=0, fixed emissivity 1.0).

Launch: two kernel launches on ctx.stream, no host sync inside the kernels. The host branches
(validation, file presence) are control flow only.
"""
from __future__ import annotations

import contextlib
import math
import os

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm.phys_rrtm_rad_initialize import KEY as RAD_KEY, RI_IALBFLG, RI_IEMSFLG

UNIT = "phys_rrtm_sfc_init"
KERNEL_FLAGS = "sfc_init_flags"
KERNEL_IDX = "sfc_init_idxems"
KEY_FLAGS = "sfc_flags"
KEY_IDXEMS = "sfc_idxems"

IMXEMS = 360        # radiation_surface_nmmb.f  IMXEMS
JMXEMS = 180        # radiation_surface_nmmb.f  JMXEMS
NREC = (IMXEMS * JMXEMS + 79) // 80   # 810 records of 80i1 digits
SEMIS_FILE = "sfc_emissivity_idx.txt"  # physparam.f:158 semis_file (cfems in sfcinit, same name)


def _decode_table(path: str) -> np.ndarray:
    """Decode the emissivity index file in Fortran read order (uint8, IMXEMS*JMXEMS digits).

    Line 1 is read as a80 and skipped. The following records are read as format 80i1: each record
    gives 80 one-digit integers, and idxems is filled column-major (first index fastest). Blank
    characters read as 0 (gfortran default BLANK='NULL'); any other non-digit is a read error.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    lines = raw.split(b"\n")
    body = lines[1:]                                    # line 1 is the a80 tag, skipped
    if len(body) < NREC:
        raise ValueError(f"sfc_init: {path} has {len(body)} data records, needs {NREC} (80i1 read hits EOF)")
    recs = b"".join(ln.rstrip(b"\r")[:80].ljust(80, b" ") for ln in body[:NREC])
    arr = np.frombuffer(recs, dtype=np.uint8)
    blank = arr == 32
    digit = (arr >= 48) & (arr <= 57)
    if not np.all(blank | digit):
        bad = int(arr[~(blank | digit)][0])
        raise ValueError(f"sfc_init: non-digit character {bad!r} in {path} (format 80i1 read error)")
    vals = np.where(blank, np.uint8(0), (arr - np.uint8(48)).astype(np.uint8))
    return np.ascontiguousarray(vals[: IMXEMS * JMXEMS])


def _fortran_mod10(n: int) -> int:
    """Fortran mod(n, 10): the remainder takes the sign of the dividend (truncating), as in C."""
    return int(math.fmod(n, 10))


def run(ctx, *, mode: int = 0, ialb: int = 0, iems: int = 0, ems_dir: str | None = None,
        iemslw_prev: int = 0) -> tuple:
    import cupy as cp

    if mode == 0:
        # sfc_init reads the physparam module values (ialbflg, iemsflg); it never assigns them.
        pp = ctx.tables.get(RAD_KEY)
        if pp is None:
            raise RuntimeError(
                "phys-rrtm-sfc-init: ctx.tables['rad_physparam'] is missing; run phys-rrtm-rad-initialize first")
        pph = cp.asnumpy(pp)                            # init-time read of 26 ints, not the step path
        ialb = int(pph[RI_IALBFLG])
        iems = int(pph[RI_IEMSFLG])

    # Fortran "stop" conditions are refusals: they name the bad flag.
    if ialb not in (0, 1, 2):
        raise ValueError(f"sfc_init: ERROR in Albedo Scheme Setting, IALB={ialb} (Fortran stop)")
    if mode == 0:
        iemslw_req = _fortran_mod10(iems)               # iemslw = mod(iemsflg, 10), Fortran semantics
        if iemslw_req not in (0, 1):
            raise ValueError(f"sfc_init: ERROR in Emissivity Scheme Setting, IEMS={iems} (Fortran stop)")
    elif mode == 1:
        if iems not in (0, 1):
            raise ValueError(f"sfcinit: ERROR in Emissivity Scheme Setting, IEMS={iems} (Fortran stop)")
        iemslw_req = iems
    else:
        raise ValueError(f"sfc_init: mode must be 0 (sfc_init) or 1 (sfcinit), got {mode}")

    base = ems_dir if ems_dir is not None else os.getcwd()
    path = os.path.join(base, SEMIS_FILE)
    file_ok = int(os.path.isfile(path))

    if KEY_FLAGS not in ctx.tables:
        ctx.tables[KEY_FLAGS] = cp.zeros(3, dtype=cp.int32)
    flags = ctx.tables[KEY_FLAGS]

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kernels.get(UNIT, KERNEL_FLAGS)(
            (1, 1, 1), (1, 1, 1),
            (np.int32(mode), np.int32(ialb), np.int32(iems), np.int32(iemslw_prev),
             np.int32(file_ok), flags))

    # The table is needed only when the emissivity-index scheme is selected and the file is present.
    want_table = (iemslw_req == 1) if mode == 0 else (iems == 1)
    idxems = None
    if want_table and file_ok:
        if KEY_IDXEMS not in ctx.tables:
            ctx.tables[KEY_IDXEMS] = cp.zeros((JMXEMS, IMXEMS), dtype=cp.int32)
        idxems = ctx.tables[KEY_IDXEMS]
        n = IMXEMS * JMXEMS
        stream = cp.asarray(_decode_table(path))
        with scope:
            kernels.get(UNIT, KERNEL_IDX)(
                ((n + 127) // 128, 1, 1), (128, 1, 1), (stream, idxems, np.int32(n)))

    info = {"file": path, "file_present": bool(file_ok), "table_read": idxems is not None}
    return flags, idxems, info
