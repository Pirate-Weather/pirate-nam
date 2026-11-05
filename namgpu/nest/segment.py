"""A moving-nest segment's side of the controller's control.json (woof-nam nest --follow).

A segment (namgpu.run_forecast with --follow-control) runs to the end of the forecast unless the controller
(namgpu/nest/follow.py) says otherwise. At every landable time L (control.json landing_every_s; GFS/IFS every 3 h,
our 12 km parent every hour) it reads control.json's committed decision for L at the hourly sync it already pays:
  continue -> it steps on;
  stop     -> it writes its restart set at L (right after L's history) and ends, so the next segment can lay that
              set over the new window's fresh start (namgpu/nest/move.py overlay).
The controller commits L at the frame before it (L - 1 h), so the decision is normally there when the segment
reaches L. When it is not yet there the segment waits for it (the wait is in the receipt), at most wait_s, then
continues (a late or dead controller never stops a forecast; the window just does not move there).
No device work: one small JSON read per landable hour.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class FollowStop:
    def __init__(self, path, dt: float, wait_s: float = 900.0, poll_s: float = 0.25):
        self.path, self.dt, self.wait_s, self.poll_s = Path(path), float(dt), float(wait_s), float(poll_s)
        c = self._read()
        if c is None:
            raise SystemExit(f"--follow-control {self.path}: no control.json there (the controller writes it before "
                             f"the first segment starts)")
        self.land_s = float(c["landing_every_s"])
        self.end_s = float(c["hours"]) * 3600.0
        k = self.land_s / self.dt
        if abs(k - round(k)) > 1e-9:
            raise SystemExit(f"--follow-control: landing interval {self.land_s:.0f} s is not a whole number of "
                             f"{self.dt} s steps")
        self.k = int(round(k))
        self.rows: list = []

    def _read(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    def landing_steps(self, n0: int, n_end: int) -> list:
        return [n for n in range(self.k, n_end, self.k) if n > n0 and n * self.dt < self.end_s - 1e-6]

    def is_landing(self, n: int) -> bool:
        return n > 0 and n % self.k == 0 and n * self.dt < self.end_s - 1e-6

    def decide(self, n: int, log=print) -> bool:
        """True: stop at step n (write the restart set and end the segment)."""
        key = f"{n * self.dt:.0f}"
        t0 = time.time()
        row = None
        while True:
            c = self._read()
            row = ((c or {}).get("landings") or {}).get(key)
            if row is not None or time.time() - t0 > self.wait_s:
                break
            time.sleep(self.poll_s)
        waited = round(time.time() - t0, 2)
        act = (row or {}).get("action", "continue (no decision)")
        self.rows.append({"landing_s": float(key), "step": int(n), "action": act, "waited_s": waited})
        if row is None:
            log(f"follow: no decision for the landing at {key} s after {waited:.0f} s: continuing in this window")
        elif waited > 1.0:
            log(f"follow: waited {waited:.1f} s at {key} s for the controller's decision ({act})")
        return act == "stop"

    def receipt(self) -> dict:
        return {"control": str(self.path), "landing_every_s": self.land_s, "landings": self.rows,
                "waited_s_total": round(sum(r["waited_s"] for r in self.rows), 2)}
