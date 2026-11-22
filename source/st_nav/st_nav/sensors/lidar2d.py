"""Conventional 2D LiDAR (40 beams, 9 deg, 3.5 m) on Isaac Lab's GPU mesh ray caster (MultiMeshRayCaster).

Mounting (TurtleBot4 Standard URDF, verified in M1): rplidar_link at (-0.040, 0, 0.1929) m in base_link. The rays are
defined directly in the base frame (beam 0 = robot forward, CCW), i.e. the ray caster frame is base_link translated to
the LiDAR origin; the physical rplidar_link is yawed +90 deg, which only matters when mapping real scans to beams
(the real-robot node selects, for every robot-frame beam direction, the nearest native sample after removing that yaw).

Isaac Lab detail (verified in M2): the ray caster applies cfg.offset to the pattern's ray START points, while
`data.pos_w` / `data.quat_w` stay the pose of the attach frame (base_link). Ranges are therefore measured from the
true ray origin `origin_w` = pos_w + R(quat_w) * mount_pos, never from data.pos_w.

Data generation (matches the reference simplified environment unless stated):
  * true range d = |hit - origin| of the first mesh hit within range_max; no return -> range_max, hit = False
  * optional minimum range (reference: none). below_min_policy="max" reproduces the real pipeline, where the
    RPLIDAR reports inf below its minimum range and the node maps inf to the cap.
  * noise: DatasheetUniformRangeNoise (reference model), applied to rays with a return, then clipped to [0, range_max]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils.math import quat_apply

from st_nav.sensors.base import RangeSensor, SensorOutput
from st_nav.sensors.noise import DatasheetUniformRangeNoise, DatasheetUniformRangeNoiseCfg
from st_nav.sensors.patterns import BeamListPatternCfg, uniform_planar_azimuths

TB4_LIDAR_MOUNT = (-0.04, 0.0, 0.192915)


@dataclass
class Lidar2DCfg:
    attach_prim: str = "{ENV_REGEX_NS}/Robot/base_link"
    mount_pos: tuple = TB4_LIDAR_MOUNT
    azimuths_deg: list = field(default_factory=lambda: uniform_planar_azimuths(40))
    range_max: float = 3.5
    range_min: float = 0.0
    below_min_policy: str = "none"          # "none" (reference) | "max" (real RPLIDAR pipeline: inf -> cap)
    ray_alignment: str = "base"             # rays follow the full base orientation (physical mounting)
    targets: list = field(default_factory=list)   # str or MultiMeshRayCasterCfg.RaycastTargetCfg
    noise: DatasheetUniformRangeNoiseCfg = field(default_factory=DatasheetUniformRangeNoiseCfg)

    def raycaster_cfg(self) -> MultiMeshRayCasterCfg:
        return MultiMeshRayCasterCfg(
            prim_path=self.attach_prim,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=self.mount_pos),
            mesh_prim_paths=list(self.targets),
            ray_alignment=self.ray_alignment,
            pattern_cfg=BeamListPatternCfg(azimuths_deg=list(self.azimuths_deg)),
            max_distance=self.range_max,
            update_period=0.0,
        )


class Lidar2D(RangeSensor):
    def __init__(self, cfg: Lidar2DCfg, raycaster, device):
        self.cfg = cfg
        self.rc = raycaster                       # the scene's MultiMeshRayCaster instance
        self.device = device
        self.range_max = cfg.range_max
        self.num_rays = len(cfg.azimuths_deg)
        self.noise = DatasheetUniformRangeNoise(cfg.noise, device)
        az = torch.deg2rad(torch.tensor(cfg.azimuths_deg, dtype=torch.float32, device=device))
        self._dirs_b = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], -1)
        self._mount = torch.tensor(cfg.mount_pos, dtype=torch.float32, device=device)
        if cfg.ray_alignment != "base":
            raise NotImplementedError("Lidar2D origin computation implemented for ray_alignment='base'")

    @property
    def origin_w(self) -> torch.Tensor:
        """(N, 3) world position of the ray origin (LiDAR optical centre)."""
        d = self.rc.data
        return d.pos_w + quat_apply(d.quat_w, self._mount.expand(d.pos_w.shape[0], -1))

    def reset(self, env_ids=None):
        self.rc.reset(env_ids)

    def read(self, sim_time=0.0) -> SensorOutput:
        data = self.rc.data                        # triggers the (lazy) ray cast
        d = torch.linalg.norm(data.ray_hits_w - self.origin_w.unsqueeze(1), dim=-1)
        hit = torch.isfinite(d) & (d <= self.range_max)
        d = torch.where(hit, d, torch.full_like(d, self.range_max))
        if self.cfg.range_min > 0 and self.cfg.below_min_policy == "max":
            short = hit & (d < self.cfg.range_min)
            d = torch.where(short, torch.full_like(d, self.range_max), d)
            hit = hit & ~short
        clean = d
        noisy = self.noise(d, hit, self.range_max)
        n = d.shape[0]
        t = torch.as_tensor(sim_time, dtype=torch.float32, device=self.device)
        return SensorOutput(ranges=noisy, ranges_clean=clean, hit=hit, valid=torch.ones_like(hit),
                            dirs_b=self._dirs_b.unsqueeze(0).expand(n, -1, -1), t_stamp=t.expand(n, self.num_rays).clone()
                            if t.dim() == 0 else t[:, None].expand(-1, self.num_rays).clone(),
                            range_max=self.range_max, range_min=self.cfg.range_min)
