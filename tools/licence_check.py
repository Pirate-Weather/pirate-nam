#!/usr/bin/env python3
"""Licence check for the AER RRTMG-derived radiation code (owner ruling, 2026-11-04).

The NAM port's RRTMG radiation is AER code (NCEP's NMMB copy) and ships under AER's BSD-3-Clause
terms. BSD-3 requires AER's notice on every copy, so this check fails when:
  - an AER-derived file (list below, from LICENSE-AUDIT.md sections 1b and 5) lacks the AER header;
  - a new phys_rrtm_lw_* / phys_rrtm_sw_* kernel or wrapper appears that is not in the list;
  - an AER-derived file labels itself plain "US government work" (only NCEP's modifications are);
  - the old AER "not sold" wording survives in an AER-derived file, NOTICE or README;
  - NOTICE lacks the BSD-3 text, the RRTMG table entry, or the other origin notices.
Run from anywhere: python3 tools/licence_check.py. Exit 0 = pass.
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LW = ["cldprop", "lwrad", "mcica_subcol", "params", "rlwinit", "rtrn", "rtrnmc", "rtrnmr",
      "setcoef", "table_cldprlw"] + ["taugb%02d" % b for b in range(1, 17)]
SW = ["cldprop", "mcica_subcol", "params", "rswinit", "setcoef", "spcvrtc", "swflux", "swrad",
      "table_cldprtb", "table_sflux", "taumol"] + ["taumol%d" % b for b in range(16, 30)]
KERNELS = (["namgpu/kernels/phys_rrtm_lw_%s.cu" % n for n in LW]
           + ["namgpu/kernels/phys_rrtm_sw_%s.cu" % n for n in SW]
           + ["namgpu/kernels/rrtm_sw_bands.cuh"])
WRAPPERS = (["namgpu/physics/rrtm/phys_rrtm_lw_%s.py" % n for n in LW + ["table_avplank"]]
            + ["namgpu/physics/rrtm/phys_rrtm_sw_%s.py" % n for n in SW + ["table_ref"]]
            + ["namgpu/physics/rrtm/lw_core.py", "namgpu/physics/rrtm/sw_seam.py",
               "namgpu/physics/rrtm/rrtm_lw_ref.py"])
AER_FILES = KERNELS + WRAPPERS

HEADER = (
    "SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause",
    "RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).",
    "BSD-3-Clause; see NOTICE.",
)
NOTICE_NEEDS = (
    "Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER)",
    "Redistribution and use in source and binary forms, with or without modification, are permitted",
    "Neither the name of the copyright holder nor the names of its contributors may be used to",
    'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
    "namgpu/data/rrtm_lw/",
    "namgpu/data/rrtm_sw/",
    "namgpu/data/rrtm_*.npz",
    "US Government work",
    "Apache License, Version 2.0",
    "public domain for use by any person or entity",
    "Hess, Koepke and Schult (1998)",
    "GMTED2010",
)
GOV = re.compile(r"US\s+(?:(?://|#)\s*)?government\s+work", re.I)


def main() -> int:
    bad: list[str] = []
    for rel in AER_FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            bad.append(f"{rel}: listed AER-derived file is missing")
            continue
        text = open(p, encoding="utf-8").read()
        head = text[:800]
        for h in HEADER:
            if h not in head:
                bad.append(f"{rel}: AER header line missing: {h!r}")
        if "not sold" in text.lower():
            bad.append(f"{rel}: old AER 'not sold' wording present")
        for m in GOV.finditer(text):
            if "modification" not in text[max(0, m.start() - 90):m.start()]:
                ln = text.count("\n", 0, m.start()) + 1
                bad.append(f"{rel}:{ln}: labelled US government work without naming AER (NCEP's modifications only)")
    listed = set(AER_FILES)
    for pat in ("namgpu/kernels/phys_rrtm_lw_*", "namgpu/kernels/phys_rrtm_sw_*",
                "namgpu/physics/rrtm/phys_rrtm_lw_*", "namgpu/physics/rrtm/phys_rrtm_sw_*"):
        for p in glob.glob(os.path.join(ROOT, pat)):
            rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
            if rel.endswith((".cu", ".cuh", ".py")) and rel not in listed:
                bad.append(f"{rel}: RRTM LW/SW file not in the AER list; add it and give it the AER header")
    npz = sorted(glob.glob(os.path.join(ROOT, "namgpu/data/rrtm_*.npz"))
                 + glob.glob(os.path.join(ROOT, "namgpu/data/rrtm_lw/*.npz"))
                 + glob.glob(os.path.join(ROOT, "namgpu/data/rrtm_sw/*.npz")))
    notice_p = os.path.join(ROOT, "NOTICE")
    notice = open(notice_p, encoding="utf-8").read() if os.path.exists(notice_p) else ""
    flat = " ".join(notice.split())
    if not notice:
        bad.append("NOTICE: missing")
    for need in NOTICE_NEEDS:
        if " ".join(need.split()) not in flat:
            bad.append(f"NOTICE: missing {need!r}")
    for name in ("NOTICE", "README.md"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p) and "not sold" in open(p, encoding="utf-8").read().lower():
            bad.append(f"{name}: old AER 'not sold' wording present")
    if bad:
        print("licence_check: FAIL (%d)" % len(bad))
        for b in bad:
            print("  " + b)
        return 1
    print("licence_check: PASS (%d AER-derived files carry the AER BSD-3-Clause header; %d RRTMG .npz "
          "tables covered by NOTICE)" % (len(AER_FILES), len(npz)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
