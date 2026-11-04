#!/usr/bin/env python3
"""Compare phys-rrtm-sw-taumol22 (GPU, chunk path) with the real NAM taumol22 subroutine.

The reference is SUBROUTINE taumol22 from nam.v4.2.11 (phys/radsw_main_nmmb.f:4085-4201), extracted unchanged,
compiled with gfortran in the SINGLE_PREC build (kind_taum = 4, the precision the port assumes) against the NAM
module sources, and driven by tools/taumol22-fortran/harness.f90 in groups of 8 columns (the Fortran CHK).

Declared deviations, both kept by the port and counted here separately (kernel header):
  - k == laytrop where laytrop exceeds the minimum of its 8-column group: the Fortran overwrites absa with absb,
    the port keeps absa. Reported as "declared"; everything else must agree.
  - speccomb == 0: not exercised by random inputs; witnessed separately (gfortran MIN(oneminus, NaN) = oneminus).

usage: check-phys-rrtm-sw-taumol22-fortran.py [--nam-phys DIR] [--work DIR] [--groups G] [--seed S]
Run on the GPU host under a pool lease (it compiles and launches the kernel once on a 4096-column chunk).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
sys.path.insert(0, str(TREE))

CHK = 8
NLAY = 60
NGPT = 112
NS22 = 54


def build_reference(phys: Path, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    lines = (phys / "radsw_main_nmmb.f").read_text(encoding="latin-1").splitlines()
    s = next(i for i, l in enumerate(lines) if re.match(r"^\s*subroutine taumol22\s*$", l, re.I))
    e = next(i for i in range(s, len(lines)) if re.search(r"end\s+subroutine\s+taumol22\b", lines[i], re.I))
    body = "\n".join(lines[s:e + 1]).replace("use module_radsw_kgb22\n", "use module_radsw_kgb22_nmmb\n")
    body = re.sub(r"(?m)^(\s*use\s+)module_radsw_kgb22\s*$", r"\1module_radsw_kgb22_nmmb", body)
    wrap = work / "taumol22_wrap.f"
    wrap.write_text(
        "      module taumol22_mod\n"
        "      use hostvars\n"
        "      use machine, only : kind_taum\n"
        "      use module_radsw_parameters_nmmb, only : NG22, NS22\n"
        "      use module_radsw_sflux_nmmb, only : strrat\n"
        "      implicit none\n"
        "      contains\n"
        + body + "\n"
        "      end module taumol22_mod\n",
        encoding="ascii")
    flags = ["-cpp", "-DSINGLE_PREC", "-O2", "-fno-range-check", "-J", str(work)]
    fixed = ["-ffixed-line-length-none"]
    free = ["-ffree-form", "-ffree-line-length-none"]
    here = HERE / "taumol22-fortran"
    steps = [
        ([*flags, *free, "-c", str(phys / "machine.f"), "-o", str(work / "machine.o")]),
        ([*flags, *free, "-c", str(phys / "radsw_param_nmmb.f"), "-o", str(work / "param.o")]),
        ([*flags, *free, "-c", str(phys / "radsw_datatb_nmmb.f"), "-o", str(work / "datatb.o")]),
        ([*flags, *free, "-c", str(here / "hostvars.f90"), "-o", str(work / "hostvars.o")]),
        ([*flags, *free, "-c", str(wrap), "-o", str(work / "wrap.o")]),
        ([*flags, *free, "-c", str(here / "harness.f90"), "-o", str(work / "harness.o")]),
        ([*flags, str(work / "machine.o"), str(work / "param.o"), str(work / "datatb.o"),
          str(work / "hostvars.o"), str(work / "wrap.o"), str(work / "harness.o"),
          "-o", str(work / "harness")]),
    ]
    for args in steps:
        subprocess.run(["gfortran", *args], check=True, cwd=work)
    return work / "harness"


def make_inputs(n_groups: int, seed: int, work: Path) -> dict:
    rng = np.random.default_rng(seed)
    n = n_groups * CHK
    shp = (n, NLAY)
    f32 = np.float32

    def u(lo, hi):
        return rng.uniform(lo, hi, shp).astype(f32)

    inp = {
        "colmol": np.exp(rng.uniform(np.log(1e20), np.log(1e25), shp)).astype(f32),
        "colamt": rng.uniform(0.0, 1.0e24, (n, NLAY, 6)).astype(f32),
        "laytrop": rng.integers(1, NLAY + 1, n).astype(np.int32),
        "fac00": u(0, 1), "fac01": u(0, 1), "fac10": u(0, 1), "fac11": u(0, 1),
        "selffac": u(0, 1), "selffrac": u(0, 1), "forfac": u(0, 1), "forfrac": u(0, 1),
        "indself": rng.integers(1, 10, shp).astype(np.int32),
        "indfor": rng.integers(1, 3, shp).astype(np.int32),
        "jp": rng.integers(1, 14, shp).astype(np.int32),
        "jt": rng.integers(1, 6, shp).astype(np.int32),
        "jt1": rng.integers(1, 6, shp).astype(np.int32),
    }
    # Fortran stream layout, first index fastest: (CHK, NLAY, G) is stored as (G, NLAY, CHK) in C order.
    def fl(a):
        return np.ascontiguousarray(a.reshape(n_groups, CHK, NLAY).transpose(0, 2, 1))
    for name in ("colmol", "fac00", "fac01", "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac"):
        fl(inp[name]).tofile(work / f"{name}.f32")
    np.ascontiguousarray(inp["colamt"].reshape(n_groups, CHK, NLAY, 6).transpose(0, 3, 2, 1)).tofile(
        work / "colamt.f32")
    inp["laytrop"].reshape(n_groups, CHK).astype(np.int32).tofile(work / "laytrop.i32")
    for name in ("indself", "indfor", "jp", "jt", "jt1"):
        fl(inp[name]).astype(np.int32).tofile(work / f"{name}.i32")
    return inp


def run_gpu(inp: dict, n: int):
    import cupy as cp

    from namgpu.physics.rrtm import phys_rrtm_sw_taumol22 as unit
    from namgpu.physics.rrtm.phys_rrtm_sw_swrad import chunk_scratch

    ch = chunk_scratch(cp, n, NLAY, NLAY + 1)
    for name in ("colmol", "colamt", "laytrop", "fac00", "fac01", "fac10", "fac11", "selffac", "selffrac",
                 "forfac", "forfrac", "indself", "indfor", "jp", "jt", "jt1"):
        ch[name] = cp.asarray(inp[name])
    ctx = SimpleNamespace(tables={}, st={"SW_NDAY": cp.asarray([n], dtype=cp.int32)}, stream=None)
    unit.run_chunk(ctx, ch, 0, n)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(ch["taug"][:, :, NS22:NS22 + 2]), cp.asnumpy(ch["taur"][:, :, NS22:NS22 + 2])


def read_fortran(out: Path, n_groups: int):
    raw = np.fromfile(out, dtype="<f4")
    half = raw.size // 2
    def unpack(part):
        a = part.reshape(n_groups, 2, NLAY, CHK)          # (G, p, k, i) C order
        return np.ascontiguousarray(a.transpose(0, 3, 2, 1)).reshape(n_groups * CHK, NLAY, 2)
    return unpack(raw[:half]), unpack(raw[half:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nam-phys", default=os.path.expandvars("$NMMB_SRC/phys"))
    ap.add_argument("--work", default="scratch/taumol22-fortran")
    ap.add_argument("--groups", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    work = Path(a.work)
    exe = build_reference(Path(a.nam_phys), work)
    inp = make_inputs(a.groups, a.seed, work)
    out = work / "taug_taur.f32"
    subprocess.run([str(exe), str(work), str(out), str(a.groups)], check=True, cwd=work)
    f_taug, f_taur = read_fortran(out, a.groups)

    n = a.groups * CHK
    g_taug, g_taur = run_gpu(inp, n)

    L = inp["laytrop"].astype(np.int64)
    gmin = np.repeat(L.reshape(a.groups, CHK).min(axis=1), CHK)
    k = np.arange(1, NLAY + 1)[None, :, None]
    declared = np.broadcast_to((k == L[:, None, None]) & (L[:, None, None] > gmin[:, None, None]), f_taug.shape)

    def cmp(g, f):
        diff = np.abs(g.astype(np.float64) - f.astype(np.float64))
        bad = diff > 1e-4 * np.abs(f.astype(np.float64)) + 1e-30
        return diff, bad

    dg, bg = cmp(g_taug, f_taug)
    dr, br = cmp(g_taur, f_taur)
    und = ~declared
    summary = {
        "columns": int(n), "groups_of_CHK": int(a.groups), "layers": NLAY, "seed": a.seed,
        "taug_cells": int(f_taug.size),
        "taug_undeclared_cells": int(und.sum()),
        "taug_undeclared_bad": int((bg & und).sum()),
        "taug_undeclared_max_rel_diff": float((dg[und] / np.maximum(np.abs(f_taug[und].astype(np.float64)), 1e-30)).max()),
        "taug_declared_cells": int(declared.sum()),
        "taug_declared_bad": int((bg & declared).sum()),
        "taug_declared_max_rel_diff": float((dg[declared] / np.maximum(np.abs(f_taug[declared].astype(np.float64)), 1e-30)).max()) if declared.any() else 0.0,
        "taur_cells": int(f_taur.size),
        "taur_bad": int(br.sum()),
        "taur_max_abs_diff": float(dr.max()),
    }
    print(json.dumps(summary, indent=2))
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    ok = summary["taug_undeclared_bad"] == 0 and summary["taur_bad"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
