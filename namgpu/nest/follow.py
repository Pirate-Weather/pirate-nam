"""woof-nam nest --follow: the storm-following controller (a moving 3 km nest is a series of windows on one frame).

The window. A moving nest keeps the rotated lat-lon frame of its first grid: the pole stays at the first centre
(tph0d, tlm0d) and the spacing (dlmd, dphd) and size (im, jm) never change. A move shifts the window by whole cells
(i0, j0) in that frame, so the overlap with the previous window is a pure index copy (no interpolation, no wind
rotation) and the static there is the same bitwise (the move step). A window whose centre would sit past rotated
latitude +-45 deg (NAM's own 12 km grid spans sbd -45.036) is refused by name; re-poling at every move was rejected
because it needs interpolation and terrain re-fits, which is the restart-shock source this design avoids.

The safe box. The tracked core must stay inside the window's interior (the lnsh boundary rows excluded, configfile
lnsh) by the target row's core margin (track.TARGETS: 150 km for a TC) plus the placement policy's fix and motion
uncertainty (hex PlacementPolicy: 25 km + 6 km/h of lead). edge_km is measured in km in the rotated frame: x room
times cos(rotated latitude), y room as is.

When moves land. Only at a landable time of the source row (SOURCE_LANDING: every 3 h of a GFS/IFS cycle, where the
source has a whole start record set; every hour from our own 12 km WOOF-NAM parent written with boco_interval_h 1).
The controller reads every closed hourly history frame (check_minutes 60) and commits the decision for a landing L
at the last frame before it (t = L - 60 min): "continue" (the segment runs on through L) or "stop" at L with the
next window. The hex placement policy decides with its granularity set to the landing interval: the current window
must hold the predicted core from L until the next landing, else the window with the longest predicted dwell from
L (whole cells, the turning prediction for preparation only, as hex) is chosen.

Plan ahead. At every frame the policy is replayed on fixes dead-reckoned along the predicted track (turning at the
rate of the last three fixes, hex b2156436 plan_predicted_path), and the windows of the next prestage_moves moves
are prepared in the background (static, a fresh start at the landing time and its boundaries: the preparer's
namgpu.nest.prepare), so a hand-off rarely waits.

Forced moves (hex b2156436). A move whose current window cannot hold the predicted core until L is forced. With no
window ready that admits the core, it moves into the ready window nearest the storm when that is nearer than the
current window (a smaller margin beats a hold), else it holds for the policy's window; with every candidate refused
(rotated latitude) it is a forced stay. The controller never raises on the storm's motion.

Hand-off at L (the move step owns the segment side): the running segment, told by control.json, writes its restart
set at L and exits; the next segment lays that set over the prepared fresh start of the new window (the overlap
shifted by (di, dj), the strip from the fresh start). control.json lists every committed landing.

Subcommands:
  python -m namgpu.nest.follow replay --track tests/data/ibtracs_ampil_2024.json [--hours 48] [--land-every-h 3]
  python -m namgpu.nest.follow observe --run RUN_DIR --storm LAT,LON [--target tc] [--source gfs]
  python -m namgpu.nest.follow run ...   (the live controller: needs the move step's segment and the preparer's preparer)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from namgpu.nest import track as st

A_EARTH = 6376000.0            # domain.A_EARTH (NMMB module_CONSTANTS a)
KM_PER_DEG = A_EARTH * math.pi / 180.0 / 1000.0
MAX_ROTATED_LAT_DEG = 45.0     # NAM's 12 km parent spans rotated latitude +-45.036 (sbd -45.036)
SCHEMA_CONTROL = "namgpu.follow-control/v1"
SCHEMA_STATE = "namgpu.follow-state/v1"

# Landable times by source row: a move lands only where the source has a whole start state (soil included) for
# the fresh start of the new window. A new source is a row.
SOURCE_LANDING = {
    "gfs": {"every_h": 3.0, "why": "GFS 0.25: a whole start record set every 3 h of the cycle (f003, f006, ...)"},
    "ifs": {"every_h": 3.0, "why": "IFS open data 0.1: a whole start record set every 3 h of the cycle"},
    "ifs-0p25": {"every_h": 3.0, "why": "IFS open data 0.25: a whole start record set every 3 h of the cycle"},
    "parent": {"every_h": 1.0, "why": "our own 12 km WOOF-NAM history, hourly, with the parent's boco_interval_h 1"},
}


def landing_every_h(source: str, given: float | None = None) -> float:
    if given:
        return float(given)
    if source not in SOURCE_LANDING:
        raise SystemExit(f"--follow: source {source!r} has no row in namgpu/nest/follow.py SOURCE_LANDING (rows: "
                         f"{', '.join(SOURCE_LANDING)}); a source without a whole start state at its leads cannot "
                         f"land a move")
    return SOURCE_LANDING[source]["every_h"]


# ------------------------------------------------------------------------------------------ rotated frame

def to_rotated(lat, lon, tph0d, tlm0d):
    """Geographic (deg) -> rotated (tlm, tph) deg on the frame centred at (tph0d, tlm0d); arrays allowed
    (domain.geographic_to_rotated, vectorised)."""
    glat = np.radians(np.asarray(lat, float))
    dlam = np.radians(((np.asarray(lon, float) - tlm0d) + 180.0) % 360.0 - 180.0)
    x, y, z = np.cos(glat) * np.cos(dlam), np.cos(glat) * np.sin(dlam), np.sin(glat)
    s0, c0 = math.sin(math.radians(tph0d)), math.cos(math.radians(tph0d))
    tph = np.arcsin(np.clip(-s0 * x + c0 * z, -1.0, 1.0))
    tlm = np.arctan2(y, c0 * x + s0 * z)
    return np.degrees(tlm), np.degrees(tph)


def to_geographic(tlm, tph, tph0d, tlm0d):
    """Rotated (tlm, tph) deg -> geographic (lat, lon) deg (grid.rotated_to_geographic, degrees)."""
    tlm, tph = np.radians(np.asarray(tlm, float)), np.radians(np.asarray(tph, float))
    s0, c0 = math.sin(math.radians(tph0d)), math.cos(math.radians(tph0d))
    sphi = np.clip(s0 * np.cos(tph) * np.cos(tlm) + c0 * np.sin(tph), -1.0, 1.0)
    dlam = np.arctan2(np.cos(tph) * np.sin(tlm), c0 * np.cos(tph) * np.cos(tlm) - s0 * np.sin(tph))
    lon = (tlm0d + np.degrees(dlam) + 180.0) % 360.0 - 180.0
    return np.degrees(np.arcsin(sphi)), lon


@dataclass(frozen=True)
class Window:
    """One window of a moving nest: the reference grid (pole, spacing, size) and its whole-cell offset (i0, j0)."""
    tph0d: float
    tlm0d: float
    dlmd: float
    dphd: float
    im: int
    jm: int
    lnsh: int = 5
    i0: int = 0
    j0: int = 0

    @classmethod
    def from_grid(cls, g: dict, lnsh: int = 5) -> "Window":
        """From a domain.nest_grid dict (a symmetric grid: its centre is the pole)."""
        return cls(float(g["tph0d"]), float(g["tlm0d"]), float(g["dlmd"]), float(g["dphd"]), int(g["im"]),
                   int(g["jm"]), int(lnsh))

    @classmethod
    def from_config(cls, cfg) -> "Window":
        """From a loaded configfile (namgpu.config.load). A shifted window (wbd != -(im-1)/2 dlmd) needs the explicit
        dlmd/dphd keys of the move step; this tree derives the spacing from wbd, so only the symmetric grid is read."""
        dl, dp = float(cfg.dlmd), float(cfg.dphd)
        i0 = int(round((float(cfg.wbd) + (cfg.im - 1) / 2.0 * dl) / dl))
        j0 = int(round((float(cfg.sbd) + (cfg.jm - 1) / 2.0 * dp) / dp))
        return cls(float(cfg.tph0d), float(cfg.tlm0d), dl, dp, int(cfg.im), int(cfg.jm), int(cfg.lnsh), i0, j0)

    def at(self, i0: int, j0: int) -> "Window":
        return replace(self, i0=int(i0), j0=int(j0))

    @property
    def key(self) -> str:
        return f"i{self.i0:+d}j{self.j0:+d}"

    @property
    def half_x(self) -> float:
        return (self.im - 1) / 2.0 * self.dlmd

    @property
    def half_y(self) -> float:
        return (self.jm - 1) / 2.0 * self.dphd

    @property
    def centre_rot(self) -> tuple[float, float]:
        return self.i0 * self.dlmd, self.j0 * self.dphd

    def centre(self) -> tuple[float, float]:
        lat, lon = to_geographic(*self.centre_rot, self.tph0d, self.tlm0d)
        return float(lat), float(lon)

    @property
    def wbd(self) -> float:
        return -self.half_x + self.i0 * self.dlmd

    @property
    def sbd(self) -> float:
        return -self.half_y + self.j0 * self.dphd

    def rotated(self, lat, lon):
        return to_rotated(lat, lon, self.tph0d, self.tlm0d)

    def edge_km(self, lat, lon):
        """Signed km from a point (or arrays of points) to the window's interior edge (inside the lnsh boundary
        rows): the smaller of the x room (times cos of the rotated latitude) and the y room. Negative = outside."""
        tlm, tph = self.rotated(lat, lon)
        cx, cy = self.centre_rot
        xr = (self.half_x - self.lnsh * self.dlmd - np.abs(tlm - cx)) * KM_PER_DEG * np.cos(np.radians(tph))
        yr = (self.half_y - self.lnsh * self.dphd - np.abs(tph - cy)) * KM_PER_DEG
        out = np.minimum(xr, yr)
        return float(out) if np.ndim(out) == 0 else out

    def interior_half_km(self) -> float:
        cy = math.radians(self.centre_rot[1])
        return min((self.half_x - self.lnsh * self.dlmd) * KM_PER_DEG * math.cos(cy),
                   (self.half_y - self.lnsh * self.dphd) * KM_PER_DEG)

    def offset_for(self, lat, lon) -> tuple[int, int]:
        """The whole-cell offset of the window centred nearest (lat, lon)."""
        tlm, tph = self.rotated(lat, lon)
        return int(round(float(tlm) / self.dlmd)), int(round(float(tph) / self.dphd))

    def refusal(self) -> str | None:
        """Why this window cannot be used, or None."""
        tph = self.centre_rot[1]
        if abs(tph) > MAX_ROTATED_LAT_DEG:
            return (f"window {self.key} centre at rotated latitude {tph:+.2f} deg is past +-{MAX_ROTATED_LAT_DEG:g} "
                    f"deg of the nest's frame (pole {self.tph0d:g}, {self.tlm0d:g}): the rotated grid would need a "
                    f"new pole, which this design never does (no interpolated overlap)")
        return None

    def grid(self, stride: int = 1):
        """Geographic lat/lon of the window's H points (every stride-th), [jm', im']."""
        i = np.arange(0, self.im, stride)
        j = np.arange(0, self.jm, stride)
        tlm = self.wbd + i * self.dlmd
        tph = self.sbd + j * self.dphd
        T, P = np.meshgrid(tlm, tph)
        return to_geographic(T, P, self.tph0d, self.tlm0d)

    def as_dict(self) -> dict:
        lat, lon = self.centre()
        return dict(asdict(self), key=self.key, wbd=round(self.wbd, 6), sbd=round(self.sbd, 6),
                    centre_deg=[round(lat, 4), round(lon, 4)])

    @classmethod
    def from_dict(cls, d: dict) -> "Window":
        return cls(*(d[k] for k in ("tph0d", "tlm0d", "dlmd", "dphd", "im", "jm", "lnsh", "i0", "j0")))


# ------------------------------------------------------------------------------------------ placement on windows

class WindowPolicy(st.PlacementPolicy):
    """The hex placement policy with the safe region = the window rectangle and candidates = whole-cell windows.

    decide()/anticipate() are hex's own; dwell() and choose() are re-based on Window.edge_km and Window.offset_for,
    so every centre the policy returns is the exact centre of a whole-cell window of the nest's frame. ``omega``
    (rad/h) turns the predicted path (turning_motion); None = hex's straight path."""

    def __init__(self, config: st.PlacementConfig, motion: st.MotionModel, window: Window, omega=None):
        super().__init__(config, motion)
        self.window = window
        self.omega = omega
        self.refused: dict[str, str] = {}

    def _path(self, origin, velocity, minutes):
        if not self.omega:
            return super()._path(origin, velocity, minutes)
        h = np.atleast_1d(np.asarray(minutes, float)) / 60.0
        dx, dy = zip(*(st.turning_motion(velocity, self.omega, float(x))[0] for x in h))
        lat, lon = st.unproject(origin[0], origin[1], np.array(dx), np.array(dy))
        return np.atleast_1d(lat), np.atleast_1d(lon)

    def win(self, centre) -> Window:
        return self.window.at(*self.window.offset_for(float(centre[0]), float(centre[1])))

    def room_km(self, centre, lat, lon):
        return self.win(centre).edge_km(lat, lon) - self.c.core_margin_km

    def dwell(self, centre, origin, velocity, start_minutes=0.0, horizon_minutes=None):
        c = self.c
        horizon = c.max_dwell_minutes if horizon_minutes is None else horizon_minutes
        s = np.arange(0.0, horizon + 1e-9, self.SUBSTEP_MIN)
        lat, lon = self._path(origin, velocity, start_minutes + s)
        sigma = c.fix_error_km + c.motion_error_kmh * (start_minutes + s) / 60.0
        ok = np.atleast_1d(self.room_km(centre, lat, lon)) - sigma >= 0.0
        if not ok[0]:
            return -1.0
        bad = np.flatnonzero(~ok)
        limit = horizon if bad.size == 0 else s[bad[0]] - self.SUBSTEP_MIN
        return float(math.floor(limit / c.check_minutes + 1e-9) * c.check_minutes)

    def choose(self, origin, velocity, current, available=(), start_minutes=0.0, excluded=()):
        c = self.c
        self.refused = {}
        leads = np.arange(0.0, c.max_lead_minutes + 1e-9, self.SUBSTEP_MIN)
        lat, lon = self._path(origin, velocity, start_minutes + leads)
        offs = {self.window.offset_for(a, b) for a, b in zip(lat, lon)}
        avail = {self.window.offset_for(float(a[0]), float(a[1])) for a in available}
        offs |= avail
        bad = {self.window.offset_for(float(e[0]), float(e[1])) for e in excluded}
        here = self.win(current)
        scored = []
        for o in sorted(offs - bad):
            w = self.window.at(*o)
            why = w.refusal()
            if why:
                self.refused[w.key] = why
                continue
            cen = w.centre()
            d = self.dwell(cen, origin, velocity, start_minutes, c.max_lead_minutes)
            moved = float(st.distance_km(*here.centre(), *cen))
            scored.append((d, -moved, o in avail, cen))
        if not scored:
            cen = here.centre()
            return [cen[0], cen[1]], -1.0, 0, False
        best = max(scored, key=lambda r: (r[0], r[1]))
        # hex keeps a prepared member when it holds two checks; here a check is a whole landing interval, so a
        # prepared window is kept when it holds the core for one interval and 75 % of the best dwell
        prepared = [r for r in scored if r[2] and r[0] >= max(c.check_minutes, c.prefer_prepared_fraction * best[0])]
        pick, used = (max(prepared, key=lambda r: (r[0], r[1])), True) if prepared else (best, False)
        return [pick[3][0], pick[3][1]], min(pick[0], c.max_dwell_minutes), len(scored), used


# ------------------------------------------------------------------------------------------ the controller

@dataclass
class FollowConfig:
    target: str = "tc"
    source: str = "gfs"
    hours: float = 48.0
    land_every_h: float | None = None       # None = SOURCE_LANDING[source]
    check_minutes: float = 60.0             # the history frame interval: one closed frame per hour
    core_margin_km: float | None = None     # None = track.TARGETS[target]
    fix_error_km: float = 25.0
    motion_error_kmh: float = 6.0
    max_lead_minutes: float = 900.0
    min_move_km: float = 20.0
    plan_hours: float = 12.0                # plan-ahead horizon for background preparation
    prestage_moves: int = 2                 # windows of the next N planned moves prepared ahead
    tracker: dict = field(default_factory=dict)   # overrides of the target's tracker block
    path_rows: list = field(default_factory=list)
    dt_s: float | None = None               # model step (s): control.json's stop_step = landing / dt_s

    @property
    def land_s(self) -> float:
        return landing_every_h(self.source, self.land_every_h) * 3600.0

    @property
    def check_s(self) -> float:
        return self.check_minutes * 60.0

    @property
    def margin_km(self) -> float:
        return float(self.core_margin_km if self.core_margin_km is not None
                     else st.target_row(self.target)["core_margin_km"])

    def margin_for(self, interior_half_km: float) -> float:
        """The core margin for a window: an explicit core_margin_km, else the target row's core_margin_km, raised
        to core_margin_frac of the interior half-width where the row has one (a margin that keeps the storm's
        rain shield, not just its eye, off the boundary rows on a large window)."""
        if self.core_margin_km is not None:
            return float(self.core_margin_km)
        row = st.target_row(self.target)
        return float(max(row["core_margin_km"], row.get("core_margin_frac", 0.0) * interior_half_km))


def _atomic_json(path: Path, data) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float) + "\n")
    os.replace(tmp, path)


class Controller:
    """Drives one moving nest: observe() each closed hourly frame, commit landings, plan ahead, hand off."""

    def __init__(self, cfg: FollowConfig, window0: Window, start: datetime, initial_center, launcher, out: Path,
                 log=print):
        self.cfg, self.window0, self.start, self.launcher, self.out, self.log = cfg, window0, start, launcher, \
            Path(out), log
        self.out.mkdir(parents=True, exist_ok=True)
        if cfg.land_s % cfg.check_s:
            raise SystemExit(f"--follow: landing interval {cfg.land_s:.0f} s is not a whole number of "
                             f"{cfg.check_s:.0f} s history frames")
        placement = dict(interior_radius_km=window0.interior_half_km(), core_margin_km=cfg.margin_for(window0.interior_half_km()),
                         check_minutes=cfg.land_s / 60.0, max_dwell_minutes=max(360.0, cfg.land_s / 60.0),
                         max_lead_minutes=cfg.max_lead_minutes, snap_deg=0.0, fix_error_km=cfg.fix_error_km,
                         motion_error_kmh=cfg.motion_error_kmh, min_move_km=cfg.min_move_km, follow="lazy")
        tcfg = st.tracker_config(cfg.target, initial_center, placement, cfg.tracker, cfg.path_rows)
        self.session = st.TrackSession(tcfg)
        self._policy()
        why = window0.refusal()
        if why:
            raise SystemExit(f"--follow: the start {why}")
        self.state = dict(schema=SCHEMA_STATE, t=-1.0, current=[window0.i0, window0.j0], generation=0,
                          committed={}, moves=[], prepared={}, frames=[], events=[], holds=[])

    # ---------------------------------------------------------------- helpers
    def _policy(self):
        p = self.session.policy
        self.session.policy = WindowPolicy(p.c, self.session.motion, self.window0)

    @property
    def policy(self) -> WindowPolicy:
        return self.session.policy

    @property
    def duration(self) -> float:
        return self.cfg.hours * 3600.0

    def current(self) -> Window:
        return self.window0.at(*self.state["current"])

    def utc(self, seconds: float) -> str:
        return (self.start + timedelta(seconds=float(seconds))).strftime("%Y-%m-%dT%H:%MZ")

    def landings(self):
        n = int(math.floor((self.duration - 1e-6) / self.cfg.land_s))
        return [k * self.cfg.land_s for k in range(1, n + 1)]

    def next_landing(self, t):
        for L in self.landings():
            if L > t + 1e-6:
                return L
        return None

    def event(self, kind, **kw):
        row = dict(kind=kind, **kw)
        self.state["events"].append(row)
        return row

    @staticmethod
    def pkey(w: Window, landing: float) -> str:
        """A preparation is one window for one landing: its fresh start is the source state at that time."""
        return f"{w.key}@{landing:.0f}"

    def windows_for(self, landing: float, t: float, states=("ready",)) -> dict:
        """Windows prepared (or preparing) for a hand-off at ``landing``: window key -> Window."""
        out = {}
        for k, p in self.state["prepared"].items():
            if abs(p["landing_s"] - landing) < 1e-6 and self.launcher.status(k, t) in states:
                w = Window.from_dict(p["window"])
                out[w.key] = w
        return out

    def request(self, w: Window, landing: float, t: float, why: str):
        """Prepare window w for a hand-off at ``landing`` (background), once per window and landing."""
        k = self.pkey(w, landing)
        if k in self.state["prepared"] and self.launcher.status(k, t) in ("ready", "preparing"):
            return
        self.state["prepared"][k] = dict(window=w.as_dict(), landing_s=float(landing), requested_s=float(t),
                                         why=why)
        self.launcher.prepare(k, w, landing, t)
        self.event("prepare", window=w.key, landing_s=landing, at_s=t, why=why)

    # ---------------------------------------------------------------- one frame
    def on_frame(self, t: float, fields: st.StateFields) -> dict:
        t = float(t)
        fix = self.session.observe(t, fields)
        cur = self.current()
        frame = dict(t=t, utc=self.utc(t), window=cur.key, generation=self.state["generation"],
                     fix=fix.as_dict())
        if fix.ok:
            frame["room_km"] = round(float(self.policy.room_km(cur.centre(), fix.lat, fix.lon)), 1)
        last, velocity, _ = self._motion()
        if last is not None:
            # whole landing intervals the current window holds the predicted core from now (fix + motion error)
            frame["safe_for_min"] = self.policy.dwell(cur.centre(), (last.lat, last.lon), velocity,
                                                      (t - last.seconds) / 60.0, self.cfg.max_lead_minutes)
        # 1. a landing at this frame: hand off
        com = self.state["committed"].get(f"{t:.0f}")
        if com and com["action"] == "stop":
            frame["handoff"] = self.handoff(t, com, fix)
        # 2. plan ahead and prepare in the background
        frame["plan"] = self.plan_ahead(t)
        for row in frame["plan"][: self.cfg.prestage_moves]:
            self.request(self.window0.at(*row["offset"]), row["landing_s"], t, f"planned ({row['why']})")
        # 3. commit the next landing at the last frame before it
        L = self.next_landing(t)
        if L is not None and L - t <= self.cfg.check_s + 1e-6 and f"{L:.0f}" not in self.state["committed"]:
            frame["commit"] = self.commit(t, L, fix)
        self.state["t"] = t
        self.state["frames"].append(frame)
        self.save()
        return frame

    def _motion(self):
        good = [f for f in self.session.fixes if f.ok]
        if not good:
            return None, None, None
        velocity, _ = self.session.motion.velocity(self.session.fixes)
        return good[-1], velocity, st.turning_rate(self.session.fixes, self.session.motion.c.max_speed_kmh)

    def commit(self, t: float, L: float, fix) -> dict:
        """Decide the landing L at frame t (the last frame before it): continue, or stop at L with a window."""
        last, velocity, _ = self._motion()
        cur = self.current()
        pol, land_min = self.policy, self.cfg.land_s / 60.0
        if last is None or self.session.lost or self.session.misses:
            why = ("storm lost" if self.session.lost else "no fix yet" if last is None
                   else f"fix missed ({self.session.misses}/{pol.c.max_misses})")
            return self._commit(L, "continue", None, f"{why}: the window holds (never moves on dead reckoning)")
        lag = (L - last.seconds) / 60.0
        origin = (last.lat, last.lon)
        stay = pol.dwell(cur.centre(), origin, velocity, lag)
        if stay >= land_min:
            return self._commit(L, "continue", None, f"core safe in {cur.key} for {stay:.0f} min after the landing")
        known = self.windows_for(L, t, ("ready", "preparing"))
        ready = self.windows_for(L, t)
        available = [w.centre() for k, w in known.items() if k != cur.key]
        centre, dwell, n, used = pol.choose(origin, velocity, cur.centre(), available, lag)
        target = pol.win(centre)
        moved = float(st.distance_km(*cur.centre(), *target.centre()))
        forced = stay < 0          # the current window cannot hold the predicted core until the landing
        if n == 0 or target.key == cur.key:
            if pol.refused:
                self.event("forced-stay", at_s=t, landing_s=L, refused=dict(pol.refused))
                return self._commit(L, "continue", None, "forced stay: every other candidate window is refused ("
                                    + "; ".join(pol.refused.values()) + ")")
            return self._commit(L, "continue", None, f"stay: {cur.key} is the best candidate window")
        if moved < pol.c.min_move_km and stay >= 0:
            return self._commit(L, "continue", None, f"best window only {moved:.0f} km away; holding")
        kind = "forced" if forced else "planned"
        reason = (f"{cur.key} holds the core {max(stay, 0):.0f} min after the landing < one landing interval; "
                  f"move {moved:.0f} km to {target.key} (dwell {dwell:.0f} min{', prepared' if used else ''})")
        if dwell < land_min:
            reason += "; WARNING the new window cannot hold the predicted core for one landing interval"
        if forced and target.key not in ready:
            # hex b2156436: a forced move with no admitted ready window takes the ready window nearest the storm
            # when it is nearer than the current one (a smaller margin beats a hold); else it holds for the target.
            here = float(st.distance_km(*origin, *cur.centre()))
            rows = sorted((float(st.distance_km(*origin, *w.centre())), k) for k, w in ready.items() if k != cur.key)
            if rows and rows[0][0] < here - 1e-6:
                k = rows[0][1]
                self.event("forced-move", at_s=t, landing_s=L, window=k, wanted=target.key, storm_km=rows[0][0])
                reason += f"; forced into ready window {k} ({rows[0][0]:.0f} km from the storm), {target.key} " \
                          f"keeps preparing"
                self.request(target, L + self.cfg.land_s, t, "forced: wanted window, for the next landing")
                target, kind = ready[k], "forced-ready"
        self.request(target, L, t, f"committed {kind}")
        return self._commit(L, "stop", target, reason, kind=kind, dwell_min=dwell,
                            room_at_commit_km=float(pol.room_km(cur.centre(), fix.lat, fix.lon)) if fix.ok else None)

    def _commit(self, L, action, target: Window | None, reason, **kw) -> dict:
        cur = self.current()
        row = dict(landing_s=float(L), utc=self.utc(L), action=action, reason=reason, **kw)
        if target is not None:
            row.update(next_window=target.as_dict(), di=target.i0 - cur.i0, dj=target.j0 - cur.j0,
                       stop_seconds=float(L))
            if self.cfg.dt_s:
                n = L / self.cfg.dt_s
                if abs(n - round(n)) > 1e-9:
                    raise SystemExit(f"--follow: landing {L:.0f} s is not a whole number of {self.cfg.dt_s} s steps")
                row["stop_step"] = int(round(n))      # steps from the run's start (the restart carries ntimestep)
        self.state["committed"][f"{L:.0f}"] = row
        self.write_control()
        self.log(f"follow: {self.utc(L)} {action}" + (f" -> {target.key} (di {row['di']:+d}, dj {row['dj']:+d})"
                                                     if target is not None else "") + f": {reason}")
        return row

    def handoff(self, L: float, com: dict, fix) -> dict:
        """The segment stopped at L: move into the committed window (wait for its preparation if needed)."""
        cur = self.current()
        target = Window.from_dict(com["next_window"])
        pol = self.policy
        ready = self.windows_for(L, L)
        note = ""
        if fix.ok and target.key in ready and pol.room_km(target.centre(), fix.lat, fix.lon) < 0:
            # the storm at L is outside the committed window's safe box: the ready window that holds it best
            rows = sorted(((float(pol.room_km(w.centre(), fix.lat, fix.lon)), k) for k, w in ready.items()),
                          reverse=True)
            if rows and rows[0][0] > pol.room_km(target.centre(), fix.lat, fix.lon) and rows[0][1] != cur.key:
                note = f"; at the landing the storm is outside {target.key}'s safe box: into ready {rows[0][1]}"
                target = ready[rows[0][1]]
        k = self.pkey(target, L)
        held = self.launcher.wait_ready(k, target, L)
        if held > 0:
            self.state["holds"].append(dict(at_s=L, window=target.key, held_s=held))
        self.state["prepared"].setdefault(k, dict(window=target.as_dict(), landing_s=float(L), requested_s=float(L),
                                                  why="hand-off"))["ready_s"] = L
        self.launcher.handoff(cur, target, L, self.state["generation"] + 1)
        self.state["current"] = [target.i0, target.j0]
        self.state["generation"] += 1
        room = float(pol.room_km(target.centre(), fix.lat, fix.lon)) if fix.ok else None
        move = dict(at_s=L, utc=self.utc(L), kind=com.get("kind", "planned"), frm=cur.key, to=target.key,
                    di=target.i0 - cur.i0, dj=target.j0 - cur.j0, i0=target.i0, j0=target.j0,
                    centre=[round(v, 3) for v in target.centre()], generation=self.state["generation"],
                    held_s=held, room_after_km=None if room is None else round(room, 1),
                    reason=com["reason"] + note)
        self.state["moves"].append(move)
        self.log(f"follow: moved at {move['utc']} {cur.key} -> {target.key} ({move['kind']}, di {move['di']:+d} dj "
                 f"{move['dj']:+d}, centre {move['centre']}){' held %.0f s' % held if held else ''}")
        return move

    # ---------------------------------------------------------------- plan ahead (hex plan_predicted_path)
    def plan_ahead(self, t: float) -> list:
        """The moves the commit rule makes if the storm keeps its predicted track (turning at the rate of the last
        three fixes), from the state now: [{landing_s, offset, centre, why}]."""
        last, velocity, omega = self._motion()
        good = [f for f in self.session.fixes if f.ok]
        if last is None or len(good) < 2 or self.session.lost:
            return []

        def track(rate):
            def at(dt):
                (dx, dy), moving = st.turning_motion(velocity, rate, dt / 3600.0)
                lat, lon = st.unproject(last.lat, last.lon, dx, dy)
                return (float(lat), float(lon)), moving
            return at

        rows = self._plan_along(t, track(omega or 0.0), "predicted" if not omega else
                                f"predicted turning {math.degrees(omega):+.0f} deg/h")
        if omega and abs(omega) > 1e-9:
            straight = track(0.0)
            times = [r["landing_s"] for r in rows] or [t + self.cfg.land_s]
            parted = max(float(st.distance_km(*track(omega)(x - last.seconds)[0], *straight(x - last.seconds)[0]))
                         for x in times)
            if parted > self.policy.c.safe_km():
                alt = [r for r in self._plan_along(t, straight, "predicted straight")
                       if all(r["offset"] != p["offset"] for p in rows)]
                for r in alt:
                    r["branch"], r["parted_km"] = "straight", round(parted, 1)
                rows = rows + alt
        return rows

    def _plan_along(self, t, predict, label) -> list:
        sim = st.TrackSession.restore(self.session.to_json())
        sim.policy = WindowPolicy(self.policy.c, sim.motion, self.window0)
        last = [f for f in sim.fixes if f.ok][-1]
        region = self.current()
        com = [c for k, c in sorted(self.state["committed"].items(), key=lambda kv: float(kv[0]))
               if float(k) > t and c["action"] == "stop"]
        if com:
            region = Window.from_dict(com[-1]["next_window"])
        out = []
        horizon = min(self.duration, t + self.cfg.plan_hours * 3600.0)
        for L in self.landings():
            if L <= t + self.cfg.check_s + 1e-6 or L > horizon or f"{L:.0f}" in self.state["committed"]:
                continue
            tc = L - self.cfg.check_s
            (lat, lon), moving = predict(tc - last.seconds)
            sim.fixes.append(st.Fix(seconds=float(tc), ok=True, lat=lat, lon=lon, reason="dead-reckoned",
                                    steering_ms=[float(moving[0]) / 3.6, float(moving[1]) / 3.6]))
            v, _ = sim.motion.velocity(sim.fixes)
            lag = (L - tc) / 60.0
            pol = sim.policy
            if pol.dwell(region.centre(), (lat, lon), v, lag) >= self.cfg.land_s / 60.0:
                continue
            avail = [w.centre() for w in self.windows_for(L, t, ("ready", "preparing")).values()]
            centre, dwell, n, _ = pol.choose((lat, lon), v, region.centre(), avail, lag)
            w = pol.win(centre)
            if n == 0 or w.key == region.key:
                continue
            out.append(dict(landing_s=float(L), utc=self.utc(L), offset=[w.i0, w.j0],
                            centre=[round(c, 3) for c in w.centre()], why=label, dwell_min=dwell))
            region = w
        return out

    # ---------------------------------------------------------------- files
    def write_control(self):
        cur = self.current()
        _atomic_json(self.out / "control.json", dict(
            schema=SCHEMA_CONTROL, start=self.start.isoformat(), hours=self.cfg.hours, frame_s=self.cfg.check_s,
            landing_every_s=self.cfg.land_s, reference=self.window0.as_dict(), current=cur.as_dict(),
            generation=self.state["generation"],
            landings={k: v for k, v in sorted(self.state["committed"].items(), key=lambda kv: float(kv[0]))}))

    def save(self):
        _atomic_json(self.out / "follow-state.json", dict(
            self.state, config=asdict(self.cfg), window0=self.window0.as_dict(), start=self.start.isoformat(),
            session=self.session.to_json()))

    def move_list(self) -> list:
        return list(self.state["moves"])

    @classmethod
    def resume(cls, path: Path, launcher, log=print) -> "Controller":
        """Rebuild a controller from its follow-state.json (--resume)."""
        d = json.loads(Path(path).read_text())
        cfg = FollowConfig(**d["config"])
        w0 = Window.from_dict(d["window0"])
        self = cls.__new__(cls)
        self.cfg, self.window0, self.start, self.launcher, self.out, self.log = cfg, w0, \
            datetime.fromisoformat(d["start"]), launcher, Path(path).parent, log
        self.session = st.TrackSession.restore(d["session"])
        self._policy()
        self.state = {k: d[k] for k in ("schema", "t", "current", "generation", "committed", "moves", "prepared",
                                        "frames", "events", "holds")}
        launcher.restore(self.state["prepared"])
        return self

    def frames_left(self):
        t0 = self.state["t"] + self.cfg.check_s if self.state["t"] >= 0 else 0.0
        n = int(round((self.duration - t0) / self.cfg.check_s))
        return [t0 + k * self.cfg.check_s for k in range(n + 1)]


# ------------------------------------------------------------------------------------------ launchers

class SimLauncher:
    """Simulated clock: a preparation is ready ``prep_s`` model seconds after it is requested (the forecast runs at
    a fixed speed, so model seconds stand for wall); a hand-off waiting for one is a hold of that many model
    seconds of wall."""

    def __init__(self, prep_s: float = 4 * 3600.0):
        self.prep_s = float(prep_s)
        self.req: dict[str, float] = {}
        self.handoffs: list = []

    def prepare(self, key: str, w: Window, landing: float, t: float):
        self.req[key] = float(t)

    def status(self, key: str, t: float) -> str:
        if key not in self.req:
            return "none"
        return "ready" if t + 1e-6 >= self.req[key] + self.prep_s else "preparing"

    def wait_ready(self, key: str, w: Window, t: float) -> float:
        if key not in self.req:
            self.req[key] = float(t)
        held = max(0.0, self.req[key] + self.prep_s - t)
        self.req[key] = min(self.req[key], t - self.prep_s)
        return held

    def handoff(self, frm: Window, to: Window, L: float, generation: int):
        self.handoffs.append((L, frm.key, to.key, generation))

    def restore(self, prepared: dict):
        for key, p in prepared.items():
            self.req[key] = float(p["requested_s"])
            if "ready_s" in p:
                self.req[key] = min(self.req[key], float(p["ready_s"]) - self.prep_s)


class ProcessLauncher:
    """The live launcher: preparers and segments in their own process groups, killed by PID (killpg).

    The preparer is the preparer's namgpu.nest.prepare (static, a fresh start at the landing time, boundaries from
    that hour into <out>/gen-<key>/, READY.json when done); the segment is the move step's run_forecast segment start
    (the restart set at L laid over the fresh start, stop at control.json's next landing). Either missing from this
    tree is refused by name before anything starts."""

    PREPARE = ("namgpu.nest.prepare", "the preparer")
    SEGMENT = ("namgpu.nest.move", "the move step")

    def __init__(self, out: Path, prepare_argv, segment_argv, env=None, check=True):
        import importlib.util
        if check:
            for mod, piece in (self.PREPARE, self.SEGMENT):
                if importlib.util.find_spec(mod) is None:
                    raise SystemExit(f"--follow run: {mod} ({piece}) is not in this tree; the live controller needs "
                                     f"it to prepare windows and start segments")
        self.out, self.prepare_argv, self.segment_argv, self.env = Path(out), prepare_argv, segment_argv, env
        self.procs: dict[str, subprocess.Popen] = {}
        self.segment: subprocess.Popen | None = None
        # At most max_prep background preparations run at once (NAMGPU_FOLLOW_PREP_MAX, default 2); the rest
        # queue, earliest landing first, and a hand-off that waits on a queued window starts it at once.
        # Breakage it prevents: 2026-11-05, a 48 h --follow door ran 6 preparers together (each fanning
        # out grib2_inventory children), a load of 74 on 30.7 cores, slowing every other process on the machine.
        self.max_prep = max(1, int(os.environ.get("NAMGPU_FOLLOW_PREP_MAX", "2")))
        self.pending: dict[str, tuple] = {}

    def gen_dir(self, key: str) -> Path:
        return self.out / f"gen-{key.replace('@', '-at')}"

    def _start(self, argv, log: Path) -> subprocess.Popen:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            return subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                                    env=self.env)

    def _live(self) -> list:
        return [k for k, p in self.procs.items() if p.poll() is None]

    def _launch(self, key: str, w: Window, landing: float):
        self.pending.pop(key, None)
        d = self.gen_dir(key)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_json(d / "window.json", dict(w.as_dict(), landing_s=landing))
        self.procs[key] = self._start(self.prepare_argv(w, landing, d), d / "prepare.log")

    def _pump(self):
        while self.pending and len(self._live()) < self.max_prep:
            key = min(self.pending, key=lambda k: self.pending[k][1])
            self._launch(key, *self.pending[key])

    def prepare(self, key: str, w: Window, landing: float, t: float, urgent: bool = False):
        old = self.procs.get(key)
        if old is not None and old.poll() is None:
            return
        if urgent or len(self._live()) < self.max_prep:
            self._launch(key, w, landing)
        else:
            self.pending[key] = (w, landing)

    def status(self, key: str, t: float) -> str:
        self._pump()
        if (self.gen_dir(key) / "READY.json").is_file():
            return "ready"
        if key in self.pending:
            return "preparing"
        p = self.procs.get(key)
        if p is None:
            return "none"
        return "preparing" if p.poll() is None else "failed"

    def wait_ready(self, key: str, w: Window, t: float, poll_s: float = 2.0) -> float:
        t0, tries = time.time(), 0
        if key in self.pending:
            self.prepare(key, *self.pending[key], t, urgent=True)
        while self.status(key, t) != "ready":
            if self.status(key, t) in ("failed", "none"):
                tries += 1
                if tries > 2:
                    raise SystemExit(f"--follow: the preparation of {key} failed twice (see "
                                     f"{self.gen_dir(key) / 'prepare.log'})")
                self.prepare(key, w, t, t)
            time.sleep(poll_s)
        return time.time() - t0

    def handoff(self, frm: Window, to: Window, L: float, generation: int):
        if self.segment is not None and self.segment.poll() is None:
            self.segment.wait()
        d = self.gen_dir(f"{to.key}@{L:.0f}")
        self.segment = self._start(self.segment_argv(frm, to, L, generation, d), d / f"segment-{generation}.log")

    def stop(self, grace_s: float = 30.0):
        """Kill every child's process group by PID (never by name)."""
        self.pending.clear()
        for p in list(self.procs.values()) + ([self.segment] if self.segment else []):
            if p is None or p.poll() is not None:
                continue
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def restore(self, prepared: dict):
        pass


class DryLauncher(SimLauncher):
    """observe mode on a finished run: nothing is prepared or started; every window counts as ready at once."""

    def __init__(self):
        super().__init__(prep_s=0.0)


# ------------------------------------------------------------------------------------------ frames

def wait_closed(path: Path, timeout_s: float = 3600.0, poll_s: float = 2.0) -> Path:
    """Wait until a history frame is closed: it exists, its size holds for two polls and netCDF opens it."""
    import netCDF4
    t0, last = time.time(), -1
    while True:
        if path.is_file():
            size = path.stat().st_size
            if size == last and size > 0:
                try:
                    with netCDF4.Dataset(str(path)):
                        return path
                except OSError:
                    pass
            last = size
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"--follow: history frame {path} not closed after {timeout_s:.0f} s")
        time.sleep(poll_s)


def frame_path(run_dir: Path, valid: datetime) -> Path:
    return Path(run_dir) / f"wrfout_d01_{valid:%Y-%m-%d_%H:%M:%S}"


def observe_frame(ctl: Controller, path: Path, t: float, cloud_cache: dict) -> dict:
    """Read one frame with the NAM adapter (3-D winds only near the predicted storm) and run on_frame."""
    guess, since = ctl.session.predicted(t)
    radius = ctl.session.search_radius(since) + 600.0
    key = ctl.current().key
    fields = st.frame_fields(path, ctl.cfg.target, guess=guess, radius_km=radius, cloud=cloud_cache.get(key))
    cloud_cache[key] = fields.points()
    return dict(fields=fields, frame=ctl.on_frame(t, fields))


# ------------------------------------------------------------------------------------------ simulated storms

def load_track(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    d["t0"] = datetime.strptime(d["rows"][0]["time"], "%Y-%m-%d %H:%M:%S")
    return d


class TrackClock:
    """Best-track rows -> position, pressure and motion at any second after a start (great-circle interpolation in
    lon unwrapped across 180)."""

    def __init__(self, track: dict, start: datetime):
        self.start = start
        t = [(datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S") - start).total_seconds() for r in track["rows"]]
        self.rows = [[s, float(r["lat"]), float(r["lon"])] for s, r in zip(t, track["rows"])]
        self.t = np.array(t)
        self.p = np.array([float(r["pres_hpa"]) for r in track["rows"]])

    def at(self, s: float):
        return st.path_position(self.rows, s)

    def pres(self, s: float) -> float:
        return float(np.interp(s, self.t, self.p))

    def motion_ms(self, s: float):
        a, b = self.at(s - 5400.0), self.at(s + 5400.0)
        x, y = st.project(a[0], a[1], b[0], b[1])
        return float(x) * 1000.0 / 10800.0, float(y) * 1000.0 / 10800.0


def synthetic_fields(clock: TrackClock, s: float, w: Window, stride: int = 3, rm_km: float = 40.0,
                     vmax_ms: float = 35.0, holland_b: float = 1.4) -> st.StateFields:
    """Model-like fields of the window at second s: a Holland vortex at the best-track centre and pressure (MSLP),
    its cyclonic tangential wind at 850/700 hPa plus the best track's own motion as the environmental flow at every
    steering level. Only the window's own grid points exist, so a storm outside the window cannot be fixed."""
    lat, lon = (a.ravel() for a in w.grid(stride))
    clat, clon = clock.at(s)
    x, y = st.project(clat, clon, lat, lon)
    r = np.maximum(np.hypot(x, y), 1e-3)
    pc, penv = clock.pres(s) * 100.0, 101000.0
    mslp = pc + (penv - pc) * np.exp(-(rm_km / r) ** holland_b)
    vt = np.where(r < rm_km, vmax_ms * r / rm_km, vmax_ms * (rm_km / r) ** 0.6)
    sign = 1.0 if clat >= 0 else -1.0
    uv, vv = -sign * vt * y / r, sign * vt * x / r
    ue, ve = clock.motion_ms(s)
    u_p = {85000.0: uv + ue, 70000.0: 0.85 * uv + ue, 50000.0: 0.3 * uv + ue, 30000.0: np.full_like(r, ue),
           20000.0: np.full_like(r, ue)}
    v_p = {85000.0: vv + ve, 70000.0: 0.85 * vv + ve, 50000.0: 0.3 * vv + ve, 30000.0: np.full_like(r, ve),
           20000.0: np.full_like(r, ve)}
    return st.StateFields(lat=lat, lon=st.wrap_lon(lon), mslp_pa=mslp, u_p=u_p, v_p=v_p, wind10=0.8 * vt,
                          land=np.zeros(lat.size, bool))


def replay(track: dict, hours: float = 48.0, size_km: float = 1000.0, res_km: float = 3.0, source: str = "gfs",
           land_every_h: float | None = None, prep_s: float = 4 * 3600.0, out: Path | None = None,
           start: datetime | None = None, stop_after_s: float | None = None, resume_from: Path | None = None,
           log=print) -> dict:
    """Drive the real Controller on a simulated clock with a best track: every closed hourly frame is a synthetic
    vortex at the best-track position on the current window's grid. Returns moves, per-hour truth room (km to the
    safe edge: the best-track centre's room in the window of that frame, and in the new window at a landing), holds
    and events."""
    from namgpu.nest import domain
    start = start or track["t0"]
    clock = TrackClock(track, start)
    out = Path(out or Path.cwd() / f"follow-replay-{track['name'].lower()}")
    launcher = SimLauncher(prep_s)
    if resume_from:
        ctl = Controller.resume(resume_from, launcher, log=log)
    else:
        lat0, lon0 = clock.at(0.0)
        g = domain.nest_grid(lat0, lon0, size_km, res_km)
        di, dn, dd = domain.RES_TABLE[g["res_km"]]["dt"]
        cfg = FollowConfig(target="tc", source=source, hours=hours, land_every_h=land_every_h,
                           dt_s=di + (dn / dd if dd else 0.0))
        ctl = Controller(cfg, Window.from_grid(g, lnsh=5), start, (lat0, lon0), launcher, out, log=log)
    truth = []
    for t in ctl.frames_left():
        if stop_after_s is not None and t > stop_after_s + 1e-6:
            break
        before = ctl.current()
        frame = ctl.on_frame(t, synthetic_fields(clock, t, before))
        tl = clock.at(t)
        row = dict(t=t, utc=ctl.utc(t), best_track=[round(tl[0], 3), round(tl[1], 3)], window=before.key,
                   room_km=round(float(ctl.policy.room_km(before.centre(), *tl)), 1))
        if frame["fix"].get("ok"):
            row["fix_error_km"] = round(float(st.distance_km(frame["fix"]["lat"], frame["fix"]["lon"], *tl)), 1)
        after = ctl.current()
        if after.key != before.key:
            row["new_window"], row["room_new_km"] = after.key, round(float(ctl.policy.room_km(after.centre(), *tl)), 1)
        truth.append(row)
    res = dict(name=track["name"], sid=track.get("sid"), agency=track.get("agency"), start=start.isoformat(),
               hours=hours, land_every_s=ctl.cfg.land_s, prep_s=prep_s, window0=ctl.window0.as_dict(),
               moves=ctl.move_list(), truth=truth, holds=ctl.state["holds"],
               events=[e for e in ctl.state["events"] if e["kind"] != "prepare"],
               prepared=len(ctl.state["prepared"]))
    res["min_room_km"] = min(min(r["room_km"], r.get("room_new_km", 1e9)) for r in truth) if truth else None
    _atomic_json(ctl.out / "replay.json", res)
    return res


def print_moves(res: dict, log=print):
    log(f"{res['name']} ({res.get('agency')}) from {res['start']}, {res['hours']:.0f} h, landings every "
        f"{res['land_every_s'] / 3600:.0f} h, preparation {res['prep_s'] / 3600:.1f} h of model time")
    log(f"  {'time (UTC)':17s} {'kind':12s} {'from':12s} {'to':12s} {'di':>5s} {'dj':>5s} {'centre':>18s} "
        f"{'room after':>10s} {'held':>6s}")
    for m in res["moves"]:
        log(f"  {m['utc']:17s} {m['kind']:12s} {m['frm']:12s} {m['to']:12s} {m['di']:+5d} {m['dj']:+5d} "
            f"{m['centre'][0]:8.3f},{m['centre'][1]:9.3f} {m['room_after_km'] or 0:10.1f} {m['held_s']:6.0f}")
    log(f"  {len(res['moves'])} moves; min best-track room to the safe edge over every hour {res['min_room_km']} km; "
        f"holds {len(res['holds'])}; events {[e['kind'] for e in res['events']]}")


# ------------------------------------------------------------------------------------------ CLI

def _center(text):
    from namgpu.nest.cli import parse_center
    return parse_center(text)


def cmd_replay(a) -> int:
    track = load_track(a.track)
    res = replay(track, hours=a.hours, size_km=a.size_km, source=a.source, land_every_h=a.land_every_h,
                 prep_s=a.prep_hours * 3600.0, out=a.out)
    print_moves(res)
    bad = [r for r in res["truth"] if r["room_km"] < 0 or r.get("room_new_km", 0.0) < 0]
    print(f"  hours with the best-track centre outside the safe box: {len(bad)}")
    return 1 if bad else 0


def cmd_observe(a) -> int:
    """The tracker and the commit rule on the closed hourly frames of a finished run (nothing is started)."""
    from namgpu import config
    run = Path(a.run)
    hist = run / "run" if (run / "run").is_dir() else run
    cfgs = sorted(run.glob("nam_configfile_*")) or sorted(run.parent.glob("nam_configfile_*"))
    if not cfgs:
        raise SystemExit(f"observe: no nam_configfile_* in {run}")
    c = config.load(cfgs[0])
    w0 = Window.from_config(c)
    r = c.raw
    dt = float(r["dt_int"]) + (float(r["dt_num"]) / float(r["dt_den"]) if int(r["dt_den"]) else 0.0)
    frames = sorted(hist.glob("wrfout_d01_*"))
    if not frames:
        raise SystemExit(f"observe: no wrfout_d01_* in {hist}")
    start = st.frame_time(frames[0])
    hours = (st.frame_time(frames[-1]) - start).total_seconds() / 3600.0
    cfg = FollowConfig(target=a.target, source=a.source, hours=hours, dt_s=dt)
    out = Path(a.out or run / "follow-observe")
    ctl = Controller(cfg, w0, start, _center(a.storm), DryLauncher(), out)
    rows, cache = [], {}
    print(f"observe: {len(frames)} frames of {hist}, window {w0.key} centre {w0.centre()}, target {a.target}, "
          f"landings every {cfg.land_s / 3600:.0f} h")
    for p in frames:
        t = (st.frame_time(p) - start).total_seconds()
        r = observe_frame(ctl, p, t, cache)
        fr, fields = r["frame"], r["fields"]
        fx = fr["fix"]
        row = dict(t=t, utc=fr["utc"], ok=fx.get("ok"), fix=[fx.get("lat"), fx.get("lon")], reason=fx.get("reason"),
                   mslp_fix_hpa=fx.get("mslp_min_hpa"), room_km=fr.get("room_km"))
        if fields.mslp_pa is not None:
            mlat, mlon, mp = st.mslp_minimum(fields)
            row["mslp_min"] = [round(mlat, 3), round(mlon, 3), round(mp, 1)]
            if fx.get("ok"):
                row["fix_to_mslp_min_km"] = round(float(st.distance_km(fx["lat"], fx["lon"], mlat, mlon)), 1)
        com = fr.get("commit")
        L = ctl.next_landing(t)
        if com:
            row["decision"] = (f"commit {com['utc']}: {com['action']}" + (f" -> {com['next_window']['key']} (di "
                               f"{com['di']:+d} dj {com['dj']:+d})" if com.get("next_window") else "") +
                               f" -- {com['reason']}")
        elif L is None:
            row["decision"] = "no landing left in the run"
        else:
            row["decision"] = (f"next landing {ctl.utc(L)} is decided at {ctl.utc(L - ctl.cfg.check_s)}; "
                               f"{fr['window']} holds the predicted core {fr.get('safe_for_min', -1):.0f} min")
        row["plan"] = [(p_["utc"], p_["offset"]) for p_ in fr["plan"]]
        rows.append(row)
        print(f"  {row['utc']}  fix {'%.3f,%.3f' % tuple(row['fix']) if row['ok'] else 'MISS'} "
              f"({row['reason']}); MSLP min {row.get('mslp_min')}; fix-to-min {row.get('fix_to_mslp_min_km')} km; "
              f"room {row['room_km']} km; {row['decision']}; plan {row['plan']}")
    _atomic_json(out / "observe.json", dict(run=str(run), window0=w0.as_dict(), target=a.target, rows=rows))
    far = [r for r in rows if r.get("fix_to_mslp_min_km") is None or r["fix_to_mslp_min_km"] > a.max_km]
    print(f"observe: {len(rows)} frames, {len(far)} with the fix more than {a.max_km:g} km from the frame's MSLP "
          f"minimum (or no fix)")
    return 1 if far else 0


def _segment_argv(out: Path):
    """The segment command of the move step's namgpu.nest.move (the contract this controller writes control.json
    for): segment <generation> runs from the prepared fresh start of its window with the previous segment's restart
    set laid over it (generation 0: the plain start), writes hourly history into out/seg-<generation>/, and stops
    at the first landing whose control.json action is "stop" after writing its restart set there."""
    def argv(frm, to, L, generation, prepared):
        return [sys.executable, "-m", "namgpu.nest.move", "segment", "--control", str(out / "control.json"),
                "--generation", str(generation), "--window", json.dumps(to.as_dict()), "--landing-s", f"{L:.0f}",
                "--prepared", str(prepared), "--restart-from", str(out / f"seg-{generation - 1}"),
                "--out", str(out / f"seg-{generation}")]
    return argv


def _prepare_argv(a, out: Path):
    """The preparer command of the preparer's namgpu.nest.prepare: static, the fresh start at the landing time and
    the boundaries from that hour for one window, into the preparation folder, then READY.json."""
    def argv(w, landing, d):
        return [sys.executable, "-m", "namgpu.nest.prepare", "--window", str(d / "window.json"), "--landing-s",
                f"{landing:.0f}", "--start", a.start, "--hours", f"{a.hours:g}", "--source", a.source, "--out", str(d)]
    return argv


def cmd_run(a) -> int:
    """The live controller: watch the closed hourly frames of the running segment, commit landings into
    control.json, prepare windows in the background and hand off at each landing. --resume continues from
    follow-state.json (the running segment and preparations are found on disk)."""
    from namgpu.nest import domain
    out = Path(a.out).resolve()
    launcher = ProcessLauncher(out, _prepare_argv(a, out), _segment_argv(out))
    if a.resume:
        ctl = Controller.resume(out / "follow-state.json", launcher)
    else:
        lat, lon = _center(a.storm)
        g = domain.nest_grid(lat, lon, a.size_km, a.res_km)
        di, dn, dd = domain.RES_TABLE[g["res_km"]]["dt"]
        cfg = FollowConfig(target=a.target, source=a.source, hours=a.hours, land_every_h=a.land_every_h,
                           dt_s=di + (dn / dd if dd else 0.0),
                           tracker=json.loads(a.follow_config) if a.follow_config else {})
        st.refuse_history(a.history_fields, a.target)
        ctl = Controller(cfg, Window.from_grid(g, lnsh=a.lnsh), datetime.strptime(a.start, "%Y%m%d%H"), (lat, lon),
                         launcher, out)
        ctl.write_control()
        launcher.handoff(ctl.window0, ctl.window0, 0.0, 0)
    cache = {}
    try:
        for t in ctl.frames_left():
            seg = out / f"seg-{ctl.state['generation']}"
            path = wait_closed(frame_path(seg, ctl.start + timedelta(seconds=t)), timeout_s=a.frame_timeout_s)
            observe_frame(ctl, path, t, cache)
    finally:
        launcher.stop()
    _atomic_json(out / "moves.json", ctl.move_list())
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m namgpu.nest.follow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="the controller on a simulated clock with a best track")
    r.add_argument("--track", type=Path, required=True)
    r.add_argument("--hours", type=float, default=48.0)
    r.add_argument("--size-km", type=float, default=1000.0)
    r.add_argument("--source", default="gfs")
    r.add_argument("--land-every-h", type=float, default=None)
    r.add_argument("--prep-hours", type=float, default=4.0)
    r.add_argument("--out", type=Path, default=None)
    o = sub.add_parser("observe", help="the tracker and commit rule on a finished run's hourly frames")
    o.add_argument("--run", required=True)
    o.add_argument("--storm", required=True, help="LAT,LON at the start")
    o.add_argument("--target", default="tc")
    o.add_argument("--source", default="gfs")
    o.add_argument("--max-km", type=float, default=20.0)
    o.add_argument("--out", default=None)
    n = sub.add_parser("run", help="the live controller (needs the move step's segment and the preparer's preparer)")
    n.add_argument("--out", required=True)
    n.add_argument("--resume", action="store_true")
    n.add_argument("--storm", default=None, help="LAT,LON at the start")
    n.add_argument("--target", default="tc", help=" | ".join(st.TARGETS))
    n.add_argument("--source", default="gfs")
    n.add_argument("--start", default=None, help="YYYYMMDDHH")
    n.add_argument("--hours", type=float, default=48.0)
    n.add_argument("--size-km", type=float, default=1000.0)
    n.add_argument("--res-km", type=float, default=3.0)
    n.add_argument("--lnsh", type=int, default=5, help="boundary rows (the configfile's lnsh)")
    n.add_argument("--land-every-h", type=float, default=None)
    n.add_argument("--follow-config", default=None, help="JSON overrides of the target's tracker block")
    n.add_argument("--history-fields", default=None)
    n.add_argument("--frame-timeout-s", type=float, default=3600.0)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return {"replay": cmd_replay, "observe": cmd_observe, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
