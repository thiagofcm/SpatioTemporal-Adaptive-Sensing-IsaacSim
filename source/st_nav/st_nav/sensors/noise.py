"""Range noise models (batched torch).

DatasheetUniformRangeNoise reproduces the reference simplified environment (turtlebot_var_scan_rate.py, `_raycast`):
  error ~ U(-a(d) * d, +a(d) * d), drawn independently per beam and per scan, from the TRUE distance d, where
  a(d) is the datasheet relative accuracy of the first band whose upper bound >= d
  (LIDAR_ACCURACY = ((3.0, 0.01), (5.0, 0.02), (25.0, 0.025)): 1 % up to 3 m, 2 % up to 5 m, 2.5 % beyond).
  Rays without a return get a zero-width interval (stay at the cap); the result is clipped to [0, range_max].

Reproducible noise: with keys = (seed (N,), tick (N,)) the uniform of beam b is st_nav.crng.uniform(seed, tick, b), a
pure function of the episode's noise seed, the acquisition time (physics tick) and the beam: independent across envs
and episodes, and the same measurement (beam, time) always gets the same noise whatever else is acquired (needed for
adaptive sensing, where policies acquire different beams at different times). Without keys: the generator below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from st_nav import crng

REFERENCE_ACCURACY_BANDS = ((3.0, 0.01), (5.0, 0.02), (25.0, 0.025))


@dataclass
class DatasheetUniformRangeNoiseCfg:
    bands: tuple = REFERENCE_ACCURACY_BANDS
    enabled: bool = True
    seed: int | None = None


class DatasheetUniformRangeNoise:
    def __init__(self, cfg: DatasheetUniformRangeNoiseCfg, device):
        self.cfg = cfg
        self.device = device
        self.upper = torch.tensor([b[0] for b in cfg.bands], device=device)
        self.acc = torch.tensor([b[1] for b in cfg.bands], device=device)
        self.gen = torch.Generator(device=device)
        if cfg.seed is not None:
            self.gen.manual_seed(cfg.seed)
        else:
            self.gen.seed()

    def relative_accuracy(self, d: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.upper, d.contiguous(), right=False).clamp(max=len(self.cfg.bands) - 1)
        return self.acc[idx]

    def __call__(self, d: torch.Tensor, hit: torch.Tensor, range_max: float, keys=None) -> torch.Tensor:
        if not self.cfg.enabled:
            return d
        e_max = torch.where(hit, self.relative_accuracy(d) * d, torch.zeros_like(d))
        if keys is not None:                                   # (seed (N,), tick (N,)) -> per (env, time, beam)
            seed, tick = keys
            beam = torch.arange(d.shape[-1], device=d.device).view(1, -1)
            u = crng.uniform(seed.view(-1, 1), tick.view(-1, 1), beam) * 2.0 - 1.0
        else:
            u = torch.rand(d.shape, generator=self.gen, device=self.device) * 2.0 - 1.0
        return torch.clamp(d + u * e_max, 0.0, range_max)
