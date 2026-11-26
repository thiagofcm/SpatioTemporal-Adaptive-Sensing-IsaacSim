"""Per-section LiDAR scan gate (spatiotemporal sensing, STS): the real-robot gate of scan_gate.py, run independently
for every horizontal section of the range image, vectorized over (envs, sections).

All sections see the same native scans (the env's 4 Hz LiDAR, held between the 5 Hz control instants). For every
genuinely NEW native scan at physics tick `now`, section s applies the rule of scan_gate.ScanGate:

    now <  next_sample[s]  -> section s ignores the scan (its columns of the held image stay as they are)
    now >= next_sample[s]  -> section s accepts it: its columns of the held image <- the new scan's columns;
                              last_forward[s] = now; next_sample[s] advances by whole periods (no drift)

and `set_rate` is set_target_fps per section (rate unchanged: nothing; changed: next = last_forward + new period). The
held image (`last_obs`) is the composite the navigation policy receives: every column holds the section's latest
accepted scan. Episode start (`reset`): the reset scan is accepted by every section, each section's rate returns to
its reset rate, counters restart (reset scan included).

With S = 1 (or every section at the same rate and the same history) this is exactly ScanGate: same formulas, same
integer ticks (tests/unit/test_section_scan_gate.py checks it bitwise).

Consumption: `rays` counts the rays acquired in the episode (reset scan included); full-scan equivalents = rays /
layout.total_rays. Torch only (no Isaac Sim); masked updates with torch.where (no host synchronization).
"""
from __future__ import annotations

import torch

from AdaptiveSensing.evaluation.scan_gate import period_ticks
from AdaptiveSensing.evaluation.sections import SectionLayout


class SectionScanGate:
    def __init__(self, num_envs: int, layout: SectionLayout, hz, physics_dt: float, device="cpu"):
        """hz: reset rate of every section (sequence of S rates, or one rate for all)."""
        S = layout.n_sections
        hz = [float(hz)] * S if isinstance(hz, (int, float)) else [float(h) for h in hz]
        if len(hz) != S:
            raise ValueError(f"{len(hz)} reset rates for {S} sections")
        self.layout, self.physics_dt, self.hz = layout, float(physics_dt), tuple(hz)
        n, dev = num_envs, device
        self.n, self.S, self.device = n, S, dev
        i64 = dict(dtype=torch.int64, device=dev)
        self.reset_periods = torch.tensor([period_ticks(h, physics_dt) for h in hz], **i64)      # (S,)
        self.col_section = layout.column_section(dev)                                            # (W,)
        self.section_rays = layout.rays_tensor(dev)                                              # (S,)
        self.last_obs = None                                                  # composite held image (N, ..., W)
        self.periods = self.reset_periods.expand(n, S).clone()                # period in force (ticks), (N, S)
        self.last_native = torch.full((n,), -1, **i64)                        # tick of the latest native scan seen
        self.next_sample = torch.zeros(n, S, **i64)                           # next_sample_time per section (ticks)
        self.last_forward = torch.zeros(n, S, **i64)                          # tick of each section's held scan
        # per-episode counters (reset scan included)
        self.n_native = torch.zeros(n, **i64)                                 # genuinely new native scans
        self.n_forward = torch.zeros(n, S, **i64)                             # acquisitions per section
        self.n_any = torch.zeros(n, **i64)                                    # native scans accepted by >= 1 section
        self.rays = torch.zeros(n, **i64)                                     # rays acquired
        self.int_min = torch.zeros(n, S, **i64)                               # min / max interval between acquisitions
        self.int_max = torch.zeros(n, S, **i64)

    # ------------------------------------------------------------------------------------------------ helpers
    def ticks(self, scan_t: torch.Tensor) -> torch.Tensor:
        return torch.round(scan_t.double() / self.physics_dt).to(torch.int64)

    def hz_in_force(self) -> torch.Tensor:
        """(N, S) rates in force (Hz, float64)."""
        return 1.0 / (self.periods.double() * self.physics_dt)

    def fse(self) -> torch.Tensor:
        """(N,) full-scan equivalents of the episode so far (reset scan = 1), float64."""
        return self.rays.double() / self.layout.total_rays

    def column_mask(self, sec_mask: torch.Tensor, ndim: int) -> torch.Tensor:
        """(N, S) section mask -> (N, 1, ..., 1, W) column mask broadcastable to an image of `ndim` dims."""
        cols = sec_mask[:, self.col_section]                                  # (N, W)
        return cols.view(cols.shape[0], *([1] * (ndim - 2)), cols.shape[1])

    def _mask(self, env_ids):
        if env_ids.dtype == torch.bool:
            return env_ids
        m = torch.zeros(self.n, dtype=torch.bool, device=self.last_native.device)
        m[env_ids] = True
        return m

    # ------------------------------------------------------------------------------------------------ gate
    def reset(self, env_ids: torch.Tensor, lidar: torch.Tensor, scan_t: torch.Tensor):
        """Episode start: every section accepts the reset scan; rates return to the reset rates. env_ids: index tensor
        or (N,) bool mask."""
        m = self._mask(env_ids)
        mS = m[:, None]
        now = self.ticks(scan_t)
        if self.last_obs is None:
            self.last_obs = lidar.clone()
        self.last_obs = torch.where(m.view(-1, *([1] * (lidar.dim() - 1))), lidar, self.last_obs)
        rp = self.reset_periods.expand(self.n, self.S)
        self.periods = torch.where(mS, rp, self.periods)
        self.last_native = torch.where(m, now, self.last_native)
        self.last_forward = torch.where(mS, now[:, None].expand(-1, self.S), self.last_forward)
        self.next_sample = torch.where(mS, now[:, None] + rp, self.next_sample)
        one = torch.ones_like(self.n_native)
        self.n_native = torch.where(m, one, self.n_native)
        self.n_any = torch.where(m, one, self.n_any)
        self.n_forward = torch.where(mS, torch.ones_like(self.n_forward), self.n_forward)
        self.rays = torch.where(m, torch.full_like(self.rays, self.layout.total_rays), self.rays)
        self.int_min = torch.where(mS, torch.full_like(self.int_min, torch.iinfo(torch.int64).max), self.int_min)
        self.int_max = torch.where(mS, torch.zeros_like(self.int_max), self.int_max)

    def step(self, lidar: torch.Tensor, scan_t: torch.Tensor):
        """Native observation (image + its time) -> (new (N,), accepted (N, S)); updates the composite held image."""
        now = self.ticks(scan_t)
        new = now != self.last_native                                         # repeated native scans are ignored
        self.last_native = now
        nowS = now[:, None]
        fwd = new[:, None] & (nowS >= self.next_sample)                       # (N, S)
        self.n_native += new
        self.n_forward += fwd
        self.n_any += fwd.any(dim=1)
        self.rays += (fwd.long() * self.section_rays).sum(dim=1)
        gap = nowS - self.last_forward
        self.int_min = torch.where(fwd, torch.minimum(self.int_min, gap), self.int_min)
        self.int_max = torch.where(fwd, torch.maximum(self.int_max, gap), self.int_max)
        self.last_forward = torch.where(fwd, nowS, self.last_forward)
        # while next_sample_time <= now: next_sample_time += period   (per section)
        adv = torch.div(nowS - self.next_sample, self.periods, rounding_mode="floor") + 1
        self.next_sample = torch.where(fwd, self.next_sample + adv * self.periods, self.next_sample)
        self.last_obs = torch.where(self.column_mask(fwd, lidar.dim()), lidar, self.last_obs)
        return new, fwd

    def set_rate(self, mask: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:
        """set_target_fps on the (env, section) pairs in `mask` (those that just accepted a scan): a changed period
        re-anchors the section's deadline at its accepted scan. mask, periods: (N, S). Returns the changed mask."""
        changed = mask & (periods != self.periods)
        self.periods = torch.where(changed, periods, self.periods)
        self.next_sample = torch.where(changed, self.last_forward + periods, self.next_sample)
        return changed
