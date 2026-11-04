"""History writer pool for several cards (woof-nam run --cards N starts it).

Every rank writes its owned rows of each history field into $SHM/<valid>/<field>.f32 (full-domain float32, global rows)
and a done.r<rank> flag; rank 0 adds meta.json and xlat/xlon (namgpu/io/io_history.py _run_slab). This pool of
--workers processes claims each complete hour (mkdir claim), writes the NetCDF with the same writer as one card
(namgpu.io.history.write_payload), and removes the hour from /dev/shm. One hour per writer, so the zlib encode of a
4 km hour never stalls the next. Exits when $SHM/END exists and no hour is left.

    python -m namgpu.io.history_merge --shm /dev/shm/namgpu-hist-X --out <run out> --nranks N --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np



def _complete(hour: Path, nranks: int) -> bool:
    return (hour / "meta.json").exists() and all((hour / f"done.r{r}").exists() for r in range(nranks))


def _write_hour(hour: Path, out: Path) -> Path:
    from namgpu.io.history import write_payload
    meta = json.loads((hour / "meta.json").read_text())
    d = meta["dims"]
    jm, im = int(d["south_north"]), int(d["west_east"])
    fields = []
    for name, dims, lead in meta["fields"]:
        shape = tuple(lead) + (jm, im)
        fields.append((name, np.fromfile(hour / f"{name}.f32", dtype=np.float32).reshape(shape), tuple(dims)))
    payload = {
        "valid": datetime.strptime(meta["valid"], "%Y-%m-%d_%H:%M:%S"),
        "start": datetime.strptime(meta["start"], "%Y-%m-%d_%H:%M:%S"),
        "xlat": np.fromfile(hour / "xlat.f32", dtype=np.float32).reshape(jm, im),
        "xlon": np.fromfile(hour / "xlon.f32", dtype=np.float32).reshape(jm, im),
        "xtime_min": meta["xtime_min"], "dims": d,
        "cen_lat": meta["cen_lat"], "cen_lon": meta["cen_lon"],
        "pole_lat": meta["pole_lat"], "pole_lon": meta["pole_lon"],
        "dx": meta["dx"], "dy": meta["dy"], "fields": fields,
    }
    path = out / f"wrfout_d01_{meta['valid']}"
    tmp = out / f".wrfout_d01_{meta['valid']}.part"
    write_payload(tmp, payload)
    tmp.replace(path)
    return path


def worker(shm: str, out: str, nranks: int, wid: int):
    shm_p, out_p = Path(shm), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    log = out_p / f"history_merge.w{wid}.log"
    while True:
        did = False
        for hour in sorted(p for p in shm_p.iterdir() if p.is_dir()):
            if not _complete(hour, nranks):
                continue
            try:
                (hour / "claim").mkdir()
            except FileExistsError:
                continue
            t0 = time.perf_counter()
            path = _write_hour(hour, out_p)
            shutil.rmtree(hour, ignore_errors=True)
            with open(log, "a") as f:
                f.write(f"{datetime.utcnow():%H:%M:%S} {path.name} {time.perf_counter() - t0:.1f} s\n")
            did = True
        if not did:
            if (shm_p / "END").exists() and not any(p.is_dir() and _complete(p, nranks) and not (p / "claim").exists()
                                                    for p in shm_p.iterdir()):
                return
            time.sleep(0.5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nranks", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    Path(a.shm).mkdir(parents=True, exist_ok=True)
    ps = [mp.Process(target=worker, args=(a.shm, a.out, a.nranks, w)) for w in range(a.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    left = [p.name for p in Path(a.shm).iterdir() if p.is_dir()]
    if left:
        print(f"history_merge: incomplete hours left in {a.shm}: {left}", flush=True)
    return 1 if any(p.exitcode for p in ps) else 0


if __name__ == "__main__":
    sys.exit(main())
