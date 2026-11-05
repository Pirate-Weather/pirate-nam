"""io-referee-sheets: referee sheets, one per product per output hour.

Panels, left to right, per product: MRMS (truth, reflectivity only) | WOOF-NAM (this port) | archived NAM
(same cycle, grid 218). Products without a truth panel in the case fetch (2 m temperature, precipitation)
are two-panel: WOOF-NAM | archived NAM. Truth for those needs a fetch (ASOS for T2, MRMS QPE for precip).

Not a driver unit: it runs after the forecast, from files, and never inside the step loop. It is not in
driver.py's lists, so it has no run(ctx) hook.

Mode "new": no Fortran counterpart and no numerics, so there is no GPU kernel for this unit. Drawing is the
Rust renderer through gpuwm.rustwx (render law: matplotlib is not used for weather fields). The panels
themselves are written by namgpu.io.io_referee_panels.

Input layout (written by io_referee_panels):
    <panels>/<product>/<source>/<YYYYmmddHH>.png
    source in {mrms, nam_gpu, nam_archive}
Output:
    <out>/<product>/<YYYYmmddHH>_referee.png

Usage:  python -m namgpu.io.io_referee_sheets --panels <dir> --out <dir> --hours 2025031412 ...

Nests anywhere (any domain, any region):
        python -m namgpu.io.io_referee_sheets nest --history <run>/wrfout_d01_* --source gfs --out <dir>
            [--leads 1-24] [--cache <dir>]
    Per product and lead: referee | WOOF-NAM | source forecast. The referee and source rows come from
    namgpu.io.referee_rows by region and date: MRMS over CONUS, NASA GPM IMERG for 1 h / 6 h
    precipitation elsewhere, the source's own later analysis (GFS f000 / IFS step 0 of the valid cycle)
    for MSLP, 2 m T, 2 m Td and 10 m wind, and the source's forecast from the same init. Every referee
    field is sampled onto the nest's own grid and drawn on the nest's frame (dateline-aware), so the
    three panels share mesh, frame and colour table. Output: <out>/sheets/<product>/fNNN_<product>.png,
    <out>/panels/..., <out>/INDEX.md (what each sheet holds, why a panel is missing, data licences) and
    <out>/sheets-receipt.json (wall and parallelism per stage, download size).
"""
from __future__ import annotations

from namgpu import cores as _cores

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

_NAME = re.compile(r"_(\d{8})_(\d{2})z_f(\d{3})_")      # maps.py panel names: namgpu_YYYYmmdd_HHz_fNNN_<product>

PRODUCTS = ("composite_reflectivity", "t2", "precip_total")
# Sources drawn per product, in panel order. Truth is only MRMS reflectivity in this case fetch.
PRODUCT_SOURCES = {
    "composite_reflectivity": ("mrms", "nam_gpu", "nam_archive"),
    "t2": ("nam_gpu", "nam_archive"),
    "precip_total": ("nam_gpu", "nam_archive"),
}
CMD = {"mrms": "mrms", "nam_gpu": "woof-nam", "nam_archive": "archive"}   # panel step per source
PANEL_NAMES = {"mrms": "MRMS", "nam_gpu": "WOOF-NAM", "nam_archive": "archived NAM"}
TITLES = {
    "composite_reflectivity": "Composite reflectivity",
    "t2": "2 m temperature",
    "precip_total": "Precipitation, total since start",
}
UNIT = "io_referee_sheets"


def panel_path(panels: Path, product: str, source: str, valid: datetime) -> Path:
    return Path(panels) / product / source / f"{valid:%Y%m%d%H}.png"


def sheet_path(out: Path, product: str, valid: datetime) -> Path:
    return Path(out) / product / f"{valid:%Y%m%d%H}_referee.png"


def _parent_licence(parent_run) -> str:
    """INDEX licence line of the parent column: our own WOOF-NAM run, credited with what IT started from and its static
    data (the door's parent credit: the parent history's start-source record, or its door receipt). It said "(no outside
    data)", but a parent starts from GFS or IFS (CC BY 4.0) and its static uses SoilGrids (CC BY 4.0) and ERA5, and the
    same INDEX carried the correct "parent run, ..." lines (review round 3, #4)."""
    name = Path(parent_run).resolve()
    name = name.parent.name if name.name == "run" else name.name
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    try:      # the door passes its licence lines (with a --parent-credit given for an older parent history)
        door = [ln[len("parent run, "):] for ln in json.loads(os.environ.get("NAMGPU_DATA_LICENCES") or "[]")
                if str(ln).startswith("parent run, ")]
    except ValueError:
        door = []
    if door:
        lines, credit = door, ""
    else:
        try:
            from namgpu.nest.cli import parent_credit  # noqa: PLC0415
            pc = parent_credit(parent_run)
        except (Exception, SystemExit) as e:  # noqa: BLE001
            return f"our own WOOF-NAM run {name}; its start-source credit could not be read ({type(e).__name__}: {e})"
        lines, credit = [x for x in (pc.get("lines") or []) if x], pc.get("credit") or ""
    body = "; ".join(lines) if lines else credit
    return f"our own WOOF-NAM run {name}, which used: {body}" if body else (
        f"our own WOOF-NAM run {name}; it records no start-source credit")


def compose_hour(renderer: Path, panels: Path, out: Path, product: str, valid: datetime) -> Path:
    """Compose the source row for one product and hour (MRMS first where it exists)."""
    from gpuwm import rustwx  # noqa: PLC0415

    row = []
    for source in PRODUCT_SOURCES[product]:
        p = panel_path(panels, product, source, valid)
        if not p.is_file():
            raise FileNotFoundError(f"missing panel: {p} (draw it first: woof-nam side-by-side {CMD[source]} ...)")
        row.append(p)
    names = " | ".join(PANEL_NAMES[s] for s in PRODUCT_SOURCES[product])
    title = f"{TITLES[product]}: {names}"
    return rustwx.compose_ensemble_sheet(
        renderer, [row], sheet_path(out, product, valid),
        title=title, valid_time=f"{valid:%Y-%m-%d %H:%M}")  # the renderer appends UTC


def run_sheets(panels: Path, out: Path, hours: list[datetime], products=PRODUCTS) -> list[Path]:
    from gpuwm import rustwx  # noqa: PLC0415

    renderer = rustwx.find_renderer()
    if renderer is None:
        raise SystemExit(rustwx.renderer_remedy())
    made = []
    for product in products:
        for valid in hours:
            made.append(compose_hour(renderer, Path(panels), Path(out), product, valid))
    return made


# ================================================================================ nests anywhere
# Sheets for a WOOF-NAM run on any domain: referee | WOOF-NAM | source forecast, per product and lead.
# The referee and source are table rows (namgpu.io.referee_rows), chosen by the domain frame, the run's
# source and its dates. Every panel is the Rust renderer's CF route on the nest's own mesh and frame.

NEST_PRODUCTS = ("composite_reflectivity", "qpf_1h", "qpf_6h", "mslp", "t2m", "d2m", "wind10")


def _sheet_subtitle_ok(renderer) -> bool:
    """True when the renderer takes the request's "subtitle" and pads panels of two sizes
    (tools/renderer/04-ensemble-sheet-pad-subtitle.patch, listed in the REVISION.txt install_renderer writes)."""
    rev = Path(renderer).with_name("REVISION.txt")
    return rev.is_file() and "04-ensemble-sheet" in rev.read_text()


def _compose(renderer, rows, output: Path, title: str, valid_time: str, subtitle: str | None = None) -> Path:
    """rw_wrfbatch --ensemble-sheet (the layout gpuwm.rustwx.compose_ensemble_sheet drives), called
    directly so the nam venv needs no gpuwm import. subtitle: the whole header line (patch 04); a renderer
    without it draws "valid {valid_time} UTC"."""
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    desc = {"rows": [[str(Path(p).resolve()) for p in row] for row in rows], "output": str(output.resolve()),
            "title": str(title), "valid_time": str(valid_time)}
    if subtitle:
        desc["subtitle"] = str(subtitle)
    with tempfile.TemporaryDirectory(prefix="sheet-", dir=output.parent) as td:
        req = Path(td) / "request.json"
        req.write_text(json.dumps(desc))
        r = subprocess.run([str(renderer), "--ensemble-sheet", str(req)], capture_output=True, text=True)
    if r.returncode or not output.is_file():
        raise RuntimeError(f"sheet {output.name}: {(r.stderr or r.stdout).strip()[-300:]}")
    return output


def _referee_job(job):
    """One referee or source panel's data: fetch, decode and sample onto the nest grid, write the CF file
    and its one-row table. Runs in a worker process. Returns (key, cf, var, table, error, bytes, seconds)."""
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from namgpu.io import maps, referee_rows  # noqa: PLC0415
    (key, row, product, start, valid, grid_npz, cache, work, res_label) = job
    t0 = time.time()
    g = np.load(grid_npz)
    lat, lon = g["lat"], g["lon"]
    fx = referee_rows.Fetcher(Path(cache), lat, lon)
    try:
        vals, units = fx.product(row, product, start, valid)
    except Exception as e:  # noqa: BLE001  -- the reason is the panel's caption, not a crash of the sheet set
        return key, None, None, None, f"{type(e).__name__}: {e}", fx.bytes, time.time() - t0
    if not np.isfinite(vals).any():
        return key, None, None, None, "no finite values on the nest grid", fx.bytes, time.time() - t0
    var = maps.TABLE[product][0]
    if product in maps.WINDOW_H:
        first = maps.TABLE[product][1]["scale"]["levels"][0] * 25.4
        vals = np.where(vals >= first, vals, np.nan)
    if not np.isfinite(vals).any():          # a dry window: draw an empty map, not a missing panel
        vals = np.zeros_like(vals)
    cf = Path(work) / f"{row['name']}_{product}_{valid:%Y%m%d%H}.nc"
    maps._write_cf(cf, lat, (lon + 180.0) % 360.0 - 180.0, valid, {var: (vals, units)}, row["label"])
    field = maps.TABLE[product][1]["title"]
    if row["role"] == "truth":
        title = f"{field}  valid {valid:%Y-%m-%d %H}Z  (sampled to the {res_label} grid)"
    else:
        title = maps.panel_title(field, start, valid)
    table = cf.with_suffix(".json")
    # the row's own credit line names the licensor (IMERG: the NCAR GDEX mirror, not NASA); was the label plus
    # " open data, CC BY 4.0" for any CC BY row, which credited NASA for a GDEX licence (fix round 2)
    note = row.get("credit")
    table.write_text(json.dumps([maps.cf_row(product, title, note)]))
    return key, str(cf), var, str(table), None, fx.bytes, time.time() - t0


def nest_sheets(history, source: str, out: Path, leads=None, products=NEST_PRODUCTS,
                cache: Path | None = None, width=1200, height=900, parent_run: Path | None = None) -> dict:
    """Draw the WOOF-NAM panels, the referee and source panels and the sheets for one run.
    history: the run's wrfout_d01_* files (f00..fNN). Returns the receipt dict (also written as
    <out>/sheets-receipt.json); INDEX.md lists every sheet, what it holds and why a panel is missing.
    parent_run: for a --source parent nest, the 12 km run it was started from; its history is drawn on the
    nest's frame as the source column (same renderer, same colour table), at every lead the parent wrote."""
    import json  # noqa: PLC0415
    import multiprocessing as mp  # noqa: PLC0415
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import time  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: PLC0415
    from datetime import timedelta  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from namgpu.io import maps, referee_rows  # noqa: PLC0415

    t_all = time.time()
    stages = {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache or out / "referee-cache")
    files = sorted(Path(h) for h in history)
    times = {f: maps._times(f) for f in files}
    files.sort(key=lambda f: times[f][0])
    start = times[files[0]][1]
    by_lead = {int(round((times[f][0] - start).total_seconds() / 3600)): f for f in files}
    leads = sorted(by_lead) if leads is None else [h for h in leads if h in by_lead]
    res = maps.resolution_km(files[-1])
    label = maps.source_label(res)
    rw = maps.renderer()
    cores = _cores.usable()

    # 1. WOOF-NAM panels (maps.render: frame from XLAT/XLONG, titles, all cores)
    t0 = time.time()
    model_dir = out / "panels" / "woof_nam"
    res_maps = maps.render([*map(str, files), "--out", str(model_dir), "--products", ",".join(products),
                            "--width", str(width), "--height", str(height)])
    frame = tuple(float(v) for v in res_maps["frame"])
    stages["woof_nam_panels"] = {"wall_s": round(time.time() - t0, 1), "parallel": f"{cores // 2} procs + threads",
                                 "written": len(res_maps["written"]), "failures": res_maps["failures"]}
    _, _, f0 = maps._read(files[0], {"XLAT", "XLONG"})
    work = Path(tempfile.mkdtemp(prefix=".ref-", dir=out))
    grid_npz = work / "grid.npz"
    np.savez(grid_npz, lat=f0["XLAT"], lon=f0["XLONG"])

    # 2. rows by region, source and dates
    hours = max(leads) if leads else 0
    picks = referee_rows.rows_for(frame, source, start, hours)
    # data credit on every sheet (CC BY sources: ECMWF open data, SoilGrids; ERA5): the run's own (the door sets
    # NAMGPU_DATA_CREDIT / NAMGPU_DATA_LICENCES; else from the source row and the static table) plus CC BY rows
    import os  # noqa: PLC0415
    if os.environ.get("NAMGPU_DATA_LICENCES"):
        run_lic = json.loads(os.environ["NAMGPU_DATA_LICENCES"])
        credit = os.environ.get("NAMGPU_DATA_CREDIT", "")
    else:
        from namgpu.nest.cli import data_credits  # noqa: PLC0415
        credit, run_lic = data_credits(source, parent_run=parent_run)
    # each sheet adds the credit of the rows drawn on it (below), not of every row the sheet set uses: the
    # IMERG credit sat on mslp/wind/reflectivity sheets that have no IMERG panel (fix round 2)
    jobs = []
    for prod in products:
        for role in ("truth", "source"):
            row = picks[prod][role]
            if row is None:
                continue
            for h in leads:
                valid = start + timedelta(hours=h)
                if row["role"] == "truth" and row["kind"] != "imerg" and valid.hour % row["cadence_h"]:
                    continue
                if row.get("lead") == "analysis" and valid.hour % row["cadence_h"]:
                    continue
                jobs.append(((prod, role, h), row, prod, start, valid, str(grid_npz), str(cache), str(work),
                             label))

    # 3. fetch + decode + sample on all cores (processes), then draw every panel (threads)
    t0 = time.time()
    results = {}
    nbytes = 0
    errors = {}
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, min(len(jobs), cores // 2, 24)),
                                 mp_context=mp.get_context("spawn")) as ex:
            for key, cf, var, table, err, b, _s in ex.map(_referee_job, jobs):
                nbytes += b
                if err:
                    errors[key] = err
                else:
                    results[key] = (cf, var, table)
    stages["referee_fetch_sample"] = {"wall_s": round(time.time() - t0, 1), "jobs": len(jobs),
                                      "parallel": f"{max(1, min(len(jobs), cores // 2, 24))} procs",
                                      "download_mb": round(nbytes / 1e6, 1), "errors": len(errors)}

    t0 = time.time()
    panel_dir = out / "panels"

    def draw(item):
        (prod, role, h), (cf, var, table) = item
        row = picks[prod][role]
        valid = start + timedelta(hours=h)
        od = Path(tempfile.mkdtemp(prefix="r-", dir=work))
        cmd = [rw, "--cf", cf, "--variable", var, "--out-dir", od, "--valid", f"{valid:%Y-%m-%dT%H:%M:%SZ}",
               "--bounds", maps.bounds_arg(frame), "--cf-table", table, "--width", str(width),
               "--height", str(height)]
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
        pngs = sorted(od.rglob("*.png"))
        if p.returncode or not pngs:
            return (prod, role, h), None, (p.stderr or p.stdout).strip()[-300:]
        dst = panel_dir / prod / row["name"] / f"f{h:03d}_{prod}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pngs[0]), dst)
        return (prod, role, h), dst, None

    drawn = {}
    with ThreadPoolExecutor(max_workers=max(1, min(len(results) or 1, cores, 48))) as ex:
        for key, dst, err in ex.map(draw, results.items()):
            if err:
                errors[key] = f"renderer: {err}"
            else:
                drawn[key] = dst
    stages["referee_panels"] = {"wall_s": round(time.time() - t0, 1), "panels": len(drawn),
                                "parallel": f"{min(len(results) or 1, cores, 48)} threads"}

    # 3b. the parent column (nests started from our own 12 km run): the parent's own history on the nest frame
    if parent_run is not None:
        t0 = time.time()
        pfiles = sorted(Path(parent_run).glob("wrfout_d01_*"))
        if not pfiles:
            raise SystemExit(f"--parent-run {parent_run}: no wrfout_d01_* history")
        ptime = {f: maps._times(f) for f in pfiles}
        pvalid = {ptime[f][0]: f for f in pfiles}
        want = set()
        for h in leads:
            v = start + timedelta(hours=h)
            want.add(v)
            want.update(v - timedelta(hours=maps.WINDOW_H[pr]) for pr in products if pr in maps.WINDOW_H)
        sel = [str(pvalid[v]) for v in sorted(want) if v in pvalid]
        pres = maps.render([*sel, "--out", str(panel_dir / "parent"), "--products", ",".join(products),
                            "--width", str(width), "--height", str(height), "--bounds", maps.bounds_arg(frame)])
        plabel = maps.source_label(maps.resolution_km(pfiles[-1])) + " parent"
        steps = sorted(pvalid)
        gaps = [b - a for a, b in zip(steps, steps[1:])]
        cad = max(set(gaps), key=gaps.count) if gaps else timedelta(hours=1)    # the usual spacing
        prow = dict(name="parent", role="source", label=plabel, cadence_h=max(1, int(cad.total_seconds() // 3600)),
                    lead="forecast", licence=_parent_licence(parent_run))
        pstart = ptime[pfiles[0]][1]
        for pth in pres["written"]:
            pth = Path(pth)
            m = _NAME.search(pth.name)
            if not m or any(t in pth.name for t in ("_comparison", "_observed", "_extra")):
                continue
            v = pstart + timedelta(hours=int(m.group(3)))
            h = int(round((v - start).total_seconds() / 3600))
            if h in leads:
                drawn[(pth.parent.name, "source", h)] = pth
        for prod in products:
            picks[prod]["source"] = prow
            for h in leads:
                if (prod, "source", h) in drawn:
                    continue
                v = start + timedelta(hours=h)
                back = maps.WINDOW_H.get(prod)
                if v not in pvalid:
                    errors[(prod, "source", h)] = f"the parent history has no frame at {v:%Y-%m-%d %H}Z"
                elif back and (v - timedelta(hours=back)) not in pvalid:
                    errors[(prod, "source", h)] = (f"the parent history has no frame {back} h before "
                                                   f"{v:%Y-%m-%d %H}Z (it was kept every {prow['cadence_h']} h)")
                else:
                    errors[(prod, "source", h)] = "parent panel not drawn (see sheets-receipt.json)"
        stages["parent_panels"] = {"wall_s": round(time.time() - t0, 1), "frames": len(sel),
                                   "written": len(pres["written"]), "failures": pres["failures"],
                                   "parallel": f"{cores // 2} procs + threads"}

    # 4. sheets: referee | WOOF-NAM | source, per product and lead
    t0 = time.time()
    model = {}
    for pth in res_maps["written"]:
        pth = Path(pth)
        if "_comparison" in pth.name or "_observed" in pth.name or "_extra" in pth.name:
            continue
        m = _NAME.search(pth.name)
        if m:
            model[(pth.parent.name, int(m.group(3)))] = pth
    sheets_dir = out / "sheets"
    sheet_jobs, index = [], []
    has_sub = _sheet_subtitle_ok(rw)
    for prod in products:
        field = maps.TABLE[prod][1]["title"]
        for h in leads:
            mine = model.get((prod, h))
            if mine is None:
                continue
            valid = start + timedelta(hours=h)
            cols, names, missing = [], [], []
            for role in ("truth", "model", "source"):
                if role == "model":
                    cols.append(mine)
                    names.append(label)
                    continue
                row = picks[prod][role]
                key = (prod, role, h)
                if key in drawn:
                    cols.append(drawn[key])
                    names.append(row["label"] + (" (later analysis)" if row.get("lead") == "analysis" else ""))
                elif row is None:
                    missing.append(f"{role}: {picks[prod]['why'].get(role, 'no row')}")
                elif key in errors:
                    missing.append(f"{role} {row['label']}: {errors[key]}")
                else:
                    missing.append(f"{role} {row['label']}: not at this hour (every {row['cadence_h']} h)")
            dst = sheets_dir / prod / f"f{h:03d}_{prod}.png"
            title = f"{label}  {field}: " + " | ".join(names)
            here = [picks[prod][r_]["credit"] for r_ in ("truth", "source")
                    if (prod, r_, h) in drawn and picks[prod][r_] and picks[prod][r_].get("credit")]
            cred = ", ".join([credit] * bool(credit) + [c_ for c_ in dict.fromkeys(here) if c_ not in credit])
            sub = (f"valid {valid:%Y-%m-%d %H:%M} UTC (+{h} h)  init {start:%Y-%m-%d %H:%M} UTC"
                   + (f"  | data: {cred}" if cred else ""))
            if has_sub:
                sheet_jobs.append((cols, dst, title, f"{valid:%Y-%m-%d %H:%M}", sub))
            else:   # an unpatched renderer appends " UTC" to valid_time: the credit is kept, the order is not
                sheet_jobs.append((cols, dst, title, sub[len("valid "):], None))
            index.append((prod, h, dst, names, missing))

    def comp(j):
        cols, dst, title, vt, sub = j
        return _compose(rw, [cols], dst, title, vt, sub)

    with ThreadPoolExecutor(max_workers=max(1, min(len(sheet_jobs) or 1, cores, 48))) as ex:
        made = list(ex.map(comp, sheet_jobs))
    stages["sheets"] = {"wall_s": round(time.time() - t0, 1), "sheets": len(made),
                        "parallel": f"{min(len(sheet_jobs) or 1, cores, 48)} threads"}
    shutil.rmtree(work, ignore_errors=True)

    used = [picks[p][r] for p in products for r in ("truth", "source") if picks[p][r] is not None]
    lic = referee_rows.licences(used)
    failed_model = res_maps["failures"]
    lines = [f"# {label} sheets: init {start:%Y-%m-%d %H}Z, source {source}", "",
             f"- Frame (W, E, S, N): {frame} (from the history's XLAT/XLONG; E past 180 = across the dateline)",
             f"- Renderer: {rw}", f"- Leads: {leads[0] if leads else '-'}..{leads[-1] if leads else '-'} h", "",
             "## Panels per product", ""]
    for prod in products:
        t, s_ = picks[prod]["truth"], picks[prod]["source"]
        lines.append(f"- {prod}: truth {t['label'] if t else 'none (' + picks[prod]['why'].get('truth', '') + ')'}; "
                     f"source {s_['label'] if s_ else 'none (' + picks[prod]['why'].get('source', '') + ')'}")
    lines += ["", "## Data sources and licences", "", "WOOF-NAM run:"] + [f"- {x}" for x in run_lic]
    lines += ["", "Referee and source panels:"] + [f"- {x}" for x in lic]
    if failed_model:
        lines += ["", "## WOOF-NAM panels that failed to draw (their sheets are missing)", ""]
        lines += [f"- {x}" for x in failed_model]
    lines += ["", "## Sheets", ""]
    for prod, h, dst, names, missing in index:
        lines.append(f"- {dst.relative_to(out)}: {' | '.join(names)}" + (f"  (missing: {'; '.join(missing)})" if missing else ""))
    (out / "INDEX.md").write_text("\n".join(lines) + "\n")
    receipt = {"run_start": f"{start:%Y-%m-%d %H}Z", "source": source, "label": label, "frame": frame,
               "renderer": str(rw), "leads": leads, "stages": stages, "licences": lic,
               "wall_s": round(time.time() - t_all, 1), "cores": cores,
               "missing": {f"{k[0]} {k[1]} f{k[2]:03d}": v for k, v in errors.items()}}
    (out / "sheets-receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
    return receipt


def _leads(spec):
    if spec is None:
        return None
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main_nest(argv):
    ap = argparse.ArgumentParser(prog="io_referee_sheets nest")
    ap.add_argument("--history", type=Path, nargs="+", required=True)
    ap.add_argument("--source", default="gfs", help="the run's init/boundary source (gfs, ifs, parent, ...)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--leads", default=None, help="e.g. 1-24 (default: every frame)")
    ap.add_argument("--products", default=",".join(NEST_PRODUCTS))
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--parent-run", type=Path, default=None, help="--source parent: draw the parent's history "
                    "as the source column")
    a = ap.parse_args(argv)
    import json  # noqa: PLC0415
    rec = nest_sheets(a.history, a.source, a.out, _leads(a.leads), tuple(p for p in a.products.split(",") if p),
                      a.cache, parent_run=a.parent_run)
    print(json.dumps({k: rec[k] for k in ("label", "frame", "wall_s", "stages")}, default=str))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "nest":
        return main_nest(argv[1:])
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Compose side-by-side sheets, one per product and valid time: MRMS | WOOF-NAM | "
                    "archived NAM, from the panels that `woof-nam side-by-side mrms|archive|woof-nam` rendered.",
        epilog="example: woof-nam side-by-side sheets --panels panels --out sheets --hours 2025031418",
        formatter_class=HelpFormatter)
    ap.add_argument("--panels", type=Path, required=True, help="the panel folder the panel steps wrote")
    ap.add_argument("--out", type=Path, required=True, help="sheet folder")
    ap.add_argument("--hours", nargs="+", required=True, help="YYYYmmddHH valid times")
    ap.add_argument("--products", default=",".join(PRODUCTS), help=f"comma list (default {','.join(PRODUCTS)})")
    a = ap.parse_args(argv)
    hours = [datetime.strptime(h, "%Y%m%d%H") for h in a.hours]
    products = tuple(p for p in a.products.split(",") if p)
    try:
        made = run_sheets(a.panels, a.out, hours, products)
    except FileNotFoundError as e:
        print(f"{UNIT}: {e}", file=sys.stderr)
        return 1
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
