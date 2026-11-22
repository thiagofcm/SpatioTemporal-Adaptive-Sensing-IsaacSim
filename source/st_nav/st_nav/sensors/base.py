"""Sensor-agnostic interface between range sensors and the navigation MDP.

Every range sensor (2D LiDAR now, Mirrorcle 3D LiDAR and programmable spatiotemporal patterns later) produces a
`SensorOutput`. Observation encoders and logging consume only this record, never sensor internals, so the sensor can be
replaced without touching rewards, terminations or the policy observation layout logic.

Design notes for the later phases (already supported by the fields, unused by the 2D LiDAR):
  * `valid` lets a sensor return fewer than R_max rays (sparse / adaptive patterns): unused slots are masked.
  * `t_stamp` carries the acquisition time of every ray (temporal sensing; rays acquired at different times).
  * `dirs_b` are the ray directions in the robot base frame for this measurement (spatial sensing: per-env, per-step).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SensorOutput:
    ranges: torch.Tensor          # (N, R) metres; invalid / no-return rays hold range_max
    ranges_clean: torch.Tensor    # (N, R) same before noise (diagnostics / privileged critic; never for the policy)
    hit: torch.Tensor             # (N, R) bool, a surface was hit within range_max (before noise)
    valid: torch.Tensor           # (N, R) bool, the ray slot was acquired this step
    dirs_b: torch.Tensor          # (N, R, 3) unit ray directions in the base frame
    t_stamp: torch.Tensor         # (N, R) seconds (simulation time of acquisition)
    range_max: float
    range_min: float

    @property
    def normalized(self) -> torch.Tensor:
        """ranges / range_max in [0, 1] (the reference simplified environment's observation scaling)."""
        return torch.clamp(self.ranges / self.range_max, 0.0, 1.0)


class RangeSensor:
    """Minimal interface. Implementations wrap an Isaac Lab sensor (or an analytic model)."""

    num_rays: int
    range_max: float

    def reset(self, env_ids=None):
        raise NotImplementedError

    def read(self, sim_time: torch.Tensor | float) -> SensorOutput:
        """Compute / fetch the current measurement (called once per control step)."""
        raise NotImplementedError
