"""Build fix/noah/{VEGPARM,SOILPARM,GENPARM}.TBL in the layout NMMB's LSM_PARM_INIT reads
(phys/module_LS_NOAHLSM.F90:1298-1547): VEGPARM first section 'IGBP', 11 items per row
(IINDEX, ALBEDO, Z0, SHDFAC, NROOT, RS, RGL, HS, SNUP, LAI, MAXALB).

Why: the NCEP NAM fix VEGPARM.TBL (IGBP first, old 10-column layout) is not published anywhere we
can fetch. This builds the IGBP rows from WRF V3.0's MODIFIED_IGBP_MODIS_NOAH section (WRF public
domain, www2.mmm.ucar.edu/wrf/src/WRFV3.0.TAR.gz): SHDFAC, NROOT, RS, RGL, HS, SNUP, MAXALB as given (NROOT 0 on water raised to 1, as in the WRF V2 USGS water row);
ALBEDO, Z0 and LAI are the midpoints of WRF's min/max pairs. SOILPARM (STAS first) and GENPARM are
WRF V3.0's files unchanged. NAMED GAP: replace with the NCEP NAM fix tables when found; the NAM state
carries its own ALBASE / Z0BASE / VEGFRC, so the table ALBEDO/Z0 are fallbacks in NOAH_LSM_INIT.

    python tools/build_noah_fix.py <WRFV3/run dir> [dest=fix/noah]
"""
import re
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "namgpu" / "data" / "fix" / "noah"
dest.mkdir(parents=True, exist_ok=True)
lines = (src / "VEGPARM.TBL").read_text().splitlines()
i = next(k for k, l in enumerate(lines) if l.strip() == "MODIFIED_IGBP_MODIS_NOAH")
n = int(lines[i + 1].split(",")[0])
rows = []
for l in lines[i + 2:i + 2 + n]:
    name = re.search(r"'([^']*)'", l).group(1)
    v = [t for t in re.split(r"[,\s]+", l.split("'")[0].strip()) if t]
    idx, shd, nrt, rs, rgl, hs, snup, maxalb, laimin, laimax, emin, emax, amin, amax, zmin, zmax = v[:16]
    f = float
    alb = (f(amin) + f(amax)) / 2.0
    z0 = (f(zmin) + f(zmax)) / 2.0
    lai = (f(laimin) + f(laimax)) / 2.0
    rows.append(f"{int(idx)},  {alb:.4f}, {z0:.5f}, {f(shd):.2f}, {max(1, int(nrt))}, {f(rs):.1f}, {f(rgl):.1f}, "
                f"{f(hs):.2f}, {f(snup):.3f}, {lai:.3f}, {f(maxalb):.1f}, '{name}'")
j = i + 2 + n
tail = []
while j < len(lines) and not lines[j].startswith("Vegetation Parameters"):
    tail.append(lines[j]); j += 1
out = ["Vegetation Parameters", "IGBP",
       f"{n},1, 'ALBEDO    Z0   SHDFAC NROOT   RS      RGL      HS      SNUP    LAI   MAXALB'"] + rows + tail
(dest / "VEGPARM.TBL").write_text("\n".join(out) + "\n")
shutil.copy(src / "SOILPARM.TBL", dest / "SOILPARM.TBL")
shutil.copy(src / "GENPARM.TBL", dest / "GENPARM.TBL")
(dest / "PROVENANCE.txt").write_text(__doc__)
print(f"build_noah_fix: {n} IGBP rows -> {dest}")
