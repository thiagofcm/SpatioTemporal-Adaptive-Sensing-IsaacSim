"""Ray patterns for Isaac Lab ray casters, defined by explicit (azimuth, elevation) lists.

Unlike `LidarPatternCfg` (a regular grid starting at the FOV minimum), the beam order and directions are stated
explicitly, which (i) fixes the beam indexing convention of the navigation stack and (ii) is the entry point for
programmable patterns (Mirrorcle scan trajectories, sparse / adaptive ray sets).

Convention (= reference simplified environment and the real-robot beam selection, drl_environment_real.py):
  azimuth measured in the sensor frame, counter-clockwise from +x (robot forward), beam i at azimuths_deg[i].
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass


def beam_list_pattern(cfg: "BeamListPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    az = torch.deg2rad(torch.tensor(cfg.azimuths_deg, dtype=torch.float32))
    el = torch.deg2rad(torch.tensor(cfg.elevations_deg if cfg.elevations_deg is not None else [0.0] * len(cfg.azimuths_deg),
                                    dtype=torch.float32))
    if el.numel() != az.numel():
        raise ValueError("azimuths_deg and elevations_deg must have the same length")
    dirs = torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], dim=-1)
    starts = torch.zeros_like(dirs)
    return starts.to(device), dirs.to(device)


@configclass
class BeamListPatternCfg(PatternBaseCfg):
    func = beam_list_pattern
    azimuths_deg: Sequence[float] = ()
    elevations_deg: Sequence[float] | None = None


def uniform_planar_azimuths(num_beams: int = 40, start_deg: float = 0.0) -> list[float]:
    """num_beams azimuths, uniform over 360 deg, beam 0 at start_deg, CCW (40 beams -> 9 deg spacing)."""
    step = 360.0 / num_beams
    return [start_deg + i * step for i in range(num_beams)]


def reference_linspace_azimuths(num_beams: int = 40, max_rad: float = 6.28) -> list[float]:
    """The reference simplified environment's beams: linspace(0, 6.28, 40) incl. both ends (9.23 deg spacing; beam 39
    is 0.18 deg short of beam 0). Kept for comparison only; the approved design uses uniform 9 deg spacing."""
    return [math.degrees(i * max_rad / (num_beams - 1)) for i in range(num_beams)]


from .ray_tables import RAY_ORDERS, grid_pattern, rpy_to_quat_wxyz  # noqa: E402,F401  (Kit-free)
