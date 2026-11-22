"""Loader for the real TurtleBot4 step-response recordings (read-only).

Source: ~/Sim2Real/Adaptive-Frame-Rate-Control-Sim2Real-TurtleBot/RealRobot/motion_dynamic_charac/runs/<run>/
  commands.csv  : t_s, cmd_linear, cmd_angular, phase (5 Hz commands, t_s = monotonic time since start of recording)
  odometry.csv  : t_s (receive time, same clock), v_linear, v_angular (/odom twist)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

DEFAULT_ROOT = os.path.expanduser(
    "~/Sim2Real/Adaptive-Frame-Rate-Control-Sim2Real-TurtleBot/RealRobot/motion_dynamic_charac/runs")


@dataclass
class RealRun:
    name: str
    kind: str                 # linear | angular | combined
    cmd_t: np.ndarray         # (K,) command publish times [s]
    cmd_v: np.ndarray
    cmd_w: np.ndarray
    cmd_phase: list
    odo_t: np.ndarray         # (M,) odometry receive times [s]
    odo_v: np.ndarray
    odo_w: np.ndarray

    @property
    def duration(self) -> float:
        return float(max(self.cmd_t[-1], self.odo_t[-1]))

    def command_at(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Zero-order-hold command at times t (0 before the first command)."""
        idx = np.searchsorted(self.cmd_t, t, side="right") - 1
        v = np.where(idx >= 0, self.cmd_v[np.clip(idx, 0, None)], 0.0)
        w = np.where(idx >= 0, self.cmd_w[np.clip(idx, 0, None)], 0.0)
        return v, w


def _read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def load_run(run_dir: str) -> RealRun:
    name = os.path.basename(run_dir.rstrip("/"))
    kind = "angular" if name.startswith("angular") else "combined" if name.startswith("combined") else "linear"
    c = _read(os.path.join(run_dir, "commands.csv"))
    o = _read(os.path.join(run_dir, "odometry.csv"))
    f = lambda rows, k: np.array([float(r[k]) for r in rows], dtype=np.float64)
    return RealRun(name=name, kind=kind, cmd_t=f(c, "t_s"), cmd_v=f(c, "cmd_linear"), cmd_w=f(c, "cmd_angular"),
                   cmd_phase=[r["phase"] for r in c], odo_t=f(o, "t_s"), odo_v=f(o, "v_linear"),
                   odo_w=f(o, "v_angular"))


def load_all(root: str = DEFAULT_ROOT) -> list[RealRun]:
    runs = []
    for name in sorted(os.listdir(root), key=lambda s: (s.split("_run_")[0] if "_run_" in s else "linear",
                                                         int(s.rsplit("_", 1)[-1]))):
        d = os.path.join(root, name)
        if os.path.isfile(os.path.join(d, "commands.csv")) and os.path.isfile(os.path.join(d, "odometry.csv")):
            runs.append(load_run(d))
    return runs


def transitions(run: RealRun, rest_s: float = 0.8):
    """Command changes: list of dicts (t, v0, w0, v1, w1, from_rest), from_rest = command was (0, 0) for >= rest_s."""
    out = []
    last_change_t, prev = 0.0, (0.0, 0.0)
    for t, v, w in zip(run.cmd_t, run.cmd_v, run.cmd_w):
        cur = (float(v), float(w))
        if cur != prev:
            from_rest = prev == (0.0, 0.0) and (t - last_change_t) >= rest_s
            out.append(dict(t=float(t), v0=prev[0], w0=prev[1], v1=cur[0], w1=cur[1], from_rest=from_rest))
            prev, last_change_t = cur, float(t)
    return out


def startup_mask(run: RealRun, t: np.ndarray, window_s: float = 2.5) -> np.ndarray:
    """True where t lies within window_s after a start-from-rest transition (firmware start-up delay regime)."""
    m = np.zeros_like(t, dtype=bool)
    for tr in transitions(run):
        if tr["from_rest"]:
            m |= (t >= tr["t"]) & (t < tr["t"] + window_s)
    return m


def step_metrics(t: np.ndarray, y: np.ndarray, t0: float, y0: float, y1: float, t_end: float, relative: bool = True):
    """Delay (time to 10 % of the step) and rise time (10 -> 90 %) of a response y(t) to a command step y0 -> y1 at t0,
    evaluated on [t0, t_end). Returns (delay, rise) or (nan, nan) when the response never crosses the levels.
    relative=True (default) normalizes by the response's own pre-step and settled levels instead of the commanded
    values, so that steady-state gain errors do not distort the timing (same definition for real and simulated data)."""
    y0_cmd, y1_cmd = y0, y1
    sel = (t >= t0) & (t < t_end)
    if sel.sum() < 3 or abs(y1 - y0) < 1e-6:
        return np.nan, np.nan
    if relative:
        # levels of the response itself: mean over 0.25 s before the step, mean over the last 40 % of the window
        pre = (t >= t0 - 0.25) & (t < t0)
        fin = (t >= t0 + 0.6 * (t_end - t0)) & (t < t_end)
        if pre.sum() < 1 or fin.sum() < 1:
            return np.nan, np.nan
        y0, y1 = float(y[pre].mean()), float(y[fin].mean())
        if abs(y1 - y0) < 0.3 * abs(y1_cmd - y0_cmd):    # response did not follow the step: no timing defined
            return np.nan, np.nan
    tt, yy = t[sel], (y[sel] - y0) / (y1 - y0)
    def cross(level):
        i = np.argmax(yy >= level)
        if yy[i] < level:
            return np.nan
        if i == 0:
            return tt[0]
        return tt[i - 1] + (level - yy[i - 1]) / (yy[i] - yy[i - 1]) * (tt[i] - tt[i - 1])
    t10, t90 = cross(0.1), cross(0.9)
    return t10 - t0, t90 - t10
