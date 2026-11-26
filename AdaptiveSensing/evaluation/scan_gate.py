"""LiDAR scan gate: the real-robot gate of the TurtleBot sim-to-real evaluation, vectorized over envs.

Port of `_on_raw_scan` / `reset_scan_gate_for_episode` in
~/Sim2Real/Adaptive-Frame-Rate-Control-Sim2Real-TurtleBot/AdaptiveFPS/scripts/real_eval_fixed.py:

    on every NEW native scan (time `now`):
        first scan of the episode  -> forward it; next_sample_time = now + period
        now <  next_sample_time    -> ignore it (the held scan stays)
        now >= next_sample_time    -> forward it; while next_sample_time <= now: next_sample_time += period

and of `set_target_fps` in real_eval_adaptive.py (adaptive sensing, TMP02; also the reference simulator,
adaptive_sensor_policy@dc5b53f envs/turtlebot_var_scan_rate.py `_set_target_fps`):

    rate unchanged             -> nothing (the schedule continues)
    rate changed               -> period = 1 / fps; next_sample_time = last_forward_time + period

A forwarded scan replaces `last_sampled_obs`, which the navigation policy reads at every control step. The schedule
advances on the ideal lattice (first sample + n * period), never from `now`, so late native scans do not shift it.

Time is kept in integer physics ticks (the native scan time `scan_t` is a multiple of the physics step), so the
comparisons are exact; every period must be a whole number of physics ticks (4, 2, 1, 0.5, 0.2 Hz at 100 Hz physics:
25, 50, 100, 200, 500 ticks). The period is per env; `reset` restores the gate's reset rate (`hz`). Masked updates use
torch.where (no host synchronization). Torch only (no Isaac Sim), so it is unit-tested without the simulator.
"""
from __future__ import annotations

import torch


def period_ticks(hz: float, physics_dt: float) -> int:
    if hz <= 0:
        raise ValueError("the requested sensing rate must be > 0")
    p = 1.0 / (hz * physics_dt)
    if abs(p - round(p)) > 1e-6:
        raise ValueError(f"period 1/{hz:g} s is not a whole number of physics steps ({physics_dt} s)")
    return int(round(p))


class ScanGate:
    def __init__(self, num_envs: int, hz: float, physics_dt: float, device="cpu"):
        """hz: rate in force after every reset (fixed-rate evaluation: the fixed rate; adaptive sensing: the maximum)."""
        self.hz, self.physics_dt, self.period = float(hz), float(physics_dt), period_ticks(hz, physics_dt)
        n, dev = num_envs, device
        i64 = dict(dtype=torch.int64, device=dev)
        self.last_obs = None                                                  # last_sampled_obs (N, ...)
        self.periods = torch.full((n,), self.period, **i64)                   # period in force (ticks), per env
        self.last_native = torch.full((n,), -1, **i64)                        # tick of the latest native scan seen
        self.next_sample = torch.zeros(n, **i64)                              # next_sample_time (ticks)
        self.last_forward = torch.zeros(n, **i64)                             # tick of the scan in last_sampled_obs
        # per-episode counters (reset scan included in both)
        self.n_native = torch.zeros(n, **i64)                                 # genuinely new native scans
        self.n_forward = torch.zeros(n, **i64)                                # forwarded (accepted) scans
        self.int_min = torch.zeros(n, **i64)                                  # min / max interval between forwards (ticks)
        self.int_max = torch.zeros(n, **i64)

    def ticks(self, scan_t: torch.Tensor) -> torch.Tensor:
        return torch.round(scan_t.double() / self.physics_dt).to(torch.int64)

    def hz_in_force(self) -> torch.Tensor:
        return 1.0 / (self.periods.double() * self.physics_dt)

    def reset(self, env_ids: torch.Tensor, lidar: torch.Tensor, scan_t: torch.Tensor):
        """Episode start: the reset scan is always forwarded (force_next_fresh); the rate returns to `hz`.
        env_ids: index tensor or (N,) bool mask."""
        m = self._mask(env_ids)
        now = self.ticks(scan_t)
        if self.last_obs is None:
            self.last_obs = lidar.clone()
        self.last_obs = torch.where(m.view(-1, *([1] * (lidar.dim() - 1))), lidar, self.last_obs)
        w = lambda new, old: torch.where(m, new, old)  # noqa: E731
        self.periods = w(torch.full_like(self.periods, self.period), self.periods)
        self.last_native = w(now, self.last_native)
        self.last_forward = w(now, self.last_forward)
        self.next_sample = w(now + self.periods, self.next_sample)
        self.n_native = w(torch.ones_like(self.n_native), self.n_native)
        self.n_forward = w(torch.ones_like(self.n_forward), self.n_forward)
        self.int_min = w(torch.full_like(self.int_min, torch.iinfo(torch.int64).max), self.int_min)
        self.int_max = w(torch.zeros_like(self.int_max), self.int_max)

    def step(self, lidar: torch.Tensor, scan_t: torch.Tensor):
        """Native observation (held scan + its time) -> (new, forwarded) masks; updates last_sampled_obs."""
        now = self.ticks(scan_t)
        new = now != self.last_native                                         # repeated native scans are ignored
        self.last_native = now
        fwd = new & (now >= self.next_sample)
        self.n_native += new
        self.n_forward += fwd
        gap = now - self.last_forward
        self.int_min = torch.where(fwd, torch.minimum(self.int_min, gap), self.int_min)
        self.int_max = torch.where(fwd, torch.maximum(self.int_max, gap), self.int_max)
        self.last_forward = torch.where(fwd, now, self.last_forward)
        # while next_sample_time <= now: next_sample_time += period
        adv = torch.div(now - self.next_sample, self.periods, rounding_mode="floor") + 1
        self.next_sample = torch.where(fwd, self.next_sample + adv * self.periods, self.next_sample)
        self.last_obs = torch.where(fwd.view(-1, *([1] * (lidar.dim() - 1))), lidar, self.last_obs)
        return new, fwd

    def set_rate(self, mask: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:
        """set_target_fps on the envs in `mask` (those that just forwarded a scan): a changed period re-anchors the
        deadline at the forwarded scan. periods: (N,) ticks. Returns the mask of envs whose rate changed."""
        changed = mask & (periods != self.periods)
        self.periods = torch.where(changed, periods, self.periods)
        self.next_sample = torch.where(changed, self.last_forward + periods, self.next_sample)
        return changed

    def _mask(self, env_ids):
        if env_ids.dtype == torch.bool:
            return env_ids
        m = torch.zeros_like(self.last_native, dtype=torch.bool)
        m[env_ids] = True
        return m


FixedRateScanGate = ScanGate     # TMP01 name (fixed rate: set_rate is never called)
