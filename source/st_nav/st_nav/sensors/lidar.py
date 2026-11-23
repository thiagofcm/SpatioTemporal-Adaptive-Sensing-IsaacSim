"""Ray-casting LiDAR on Isaac Lab's GPU mesh ray caster (MultiMeshRayCaster): one implementation for every ray set.

The sensor is defined by an explicit ray table (azimuth, elevation per ray, in the SENSOR frame: x forward, y left, z up;
azimuth CCW from +x, elevation up), a mount pose in base_link (position + roll/pitch/yaw, REP-103: positive pitch tilts
the sensor's forward axis DOWN), a range model and a noise model. Presets:
  * 2D LiDAR (M2): RayLidarCfg defaults = 40 beams, 9 deg, elevation 0, 3.5 m, TB4 RPLIDAR mount, datasheet noise
    (st_nav.sensors.lidar2d keeps the historical names Lidar2DCfg / Lidar2D for it).
  * 3D LiDAR (M7, Mirrorcle raster approximation): Lidar3DCfg (YAML-facing) builds a RayLidarCfg from a uniform
    rectangular grid (st_nav.sensors.patterns.grid_pattern) or an explicit ray table.

Isaac Lab details (verified in M2 / M7): the ray caster rotates the pattern's directions by offset.rot and adds
offset.pos to the ray STARTS, while data.pos_w / data.quat_w stay the pose of the attach frame (base_link); ranges are
measured from origin_w = pos_w + R(quat_w) * mount_pos. ray_alignment = "base": rays follow the full base orientation.

Data generation: true range d = |hit - origin| of the first mesh hit within range_max; no return -> range_max, hit =
False; optional minimum range; then the noise model (reproducible per episode / acquisition tick / ray when keyed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from st_nav.sensors.base import RangeSensor, SensorOutput
from st_nav.sensors.noise import DatasheetUniformRangeNoiseCfg, GaussianRangeNoiseCfg, make_range_noise
from st_nav.sensors.patterns import BeamListPatternCfg, grid_pattern, rpy_to_quat_wxyz, uniform_planar_azimuths

TB4_LIDAR_MOUNT = (-0.04, 0.0, 0.192915)          # RPLIDAR A1 optical centre in base_link (M1)
TB4_TOP_PLATE_Z = 0.3498                          # top surface of the TB4 top platform (base_link frame, robot USD)
# Floor as a ray-cast target (3D sensors): a finite box whose top face is the ground (z = 0), covering every env plus
# the LiDAR range; no collider, invisible, so physics keeps using the ground plane. Not the ground plane itself: Isaac
# Lab meshes a USD Plane as a 2000 km square of 2 triangles, and float32 ray-triangle tests on it err by ~1-2 mm (M7).
FLOOR_PRIM = "/World/LidarFloor"
GROUND_PLANE_PRIM = "/World/ground/terrain/GroundPlane/CollisionPlane"


@dataclass
class RayLidarCfg:
    sensor_type: str = "lidar2d"
    attach_prim: str = "{ENV_REGEX_NS}/Robot/base_link"
    mount_pos: tuple = TB4_LIDAR_MOUNT
    mount_rpy_deg: tuple = (0.0, 0.0, 0.0)          # roll, pitch, yaw (REP-103; pitch > 0 = looking down)
    azimuths_deg: list = field(default_factory=lambda: uniform_planar_azimuths(40))
    elevations_deg: list | None = None              # None = all 0 (planar)
    grid_shape: tuple | None = None                 # (rows, cols) for range images (3D grids)
    grid_row: list | None = None                    # per-ray cell (row 0 = top), per-ray column (0 = left)
    grid_col: list | None = None
    ray_order: str = "planar_ccw"
    range_max: float = 3.5
    range_min: float = 0.0
    below_min_policy: str = "none"                  # "none" (reference) | "max" (real RPLIDAR pipeline: inf -> cap)
    ray_alignment: str = "base"                     # rays follow the full base orientation (physical mounting)
    include_floor: bool = False                     # ray-cast against the ground plane (3D sensors)
    targets: list = field(default_factory=list)     # set by the scene runtime
    noise: object = field(default_factory=DatasheetUniformRangeNoiseCfg)

    @property
    def num_rays(self) -> int:
        return len(self.azimuths_deg)

    def raycaster_cfg(self) -> MultiMeshRayCasterCfg:
        return MultiMeshRayCasterCfg(
            prim_path=self.attach_prim,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=self.mount_pos, rot=rpy_to_quat_wxyz(self.mount_rpy_deg)),
            mesh_prim_paths=list(self.targets),
            ray_alignment=self.ray_alignment,
            pattern_cfg=BeamListPatternCfg(azimuths_deg=list(self.azimuths_deg),
                                           elevations_deg=None if self.elevations_deg is None else list(self.elevations_deg)),
            max_distance=self.range_max,
            update_period=0.0,
        )


@configclass
class Lidar3DCfg:
    """YAML-facing configuration of the 3D LiDAR (env cfg field `lidar3d`; select it with `sensor: lidar3d`).

    Default = Mirrorcle raster approximation: uniform rectangular grid, 139 x 34 rays at cell centres over -14..+14 deg
    azimuth and -7..+7 deg elevation, row-major from the top-left (provisional ordering), 4 Hz, 12 m, Gaussian noise
    sigma 1 cm, on the TB4 top platform (5 cm above it, centred, facing forward) pitched 5 deg down.
    A measured ray table replaces the grid, either inline (azimuths_deg / elevations_deg, optionally grid_row / grid_col)
    or from a file (ray_table: .csv with a header azimuth_deg, elevation_deg[, grid_row, grid_col], or .npz with these
    arrays; relative paths from the repository root). Ray i of the table = ray i of the observation, recording and noise.
    """
    h_fov_deg: tuple = (-14.0, 14.0)
    v_fov_deg: tuple = (-7.0, 7.0)
    cols: int = 139
    rows: int = 34
    placement: str = "cell_center"                 # cell_center | endpoints
    ray_order: str = "row_major_top_left"           # see st_nav.sensors.patterns.RAY_ORDERS
    ray_table: str | None = None                    # ray table file (overrides the grid and the inline table)
    azimuths_deg: list | None = None                # explicit ray table (overrides the grid when given)
    elevations_deg: list | None = None
    grid_row: list | None = None
    grid_col: list | None = None
    mount_pos: tuple = (0.0, 0.0, round(TB4_TOP_PLATE_Z + 0.05, 4))    # centred on the top platform, 5 cm above it
    mount_rpy_deg: tuple = (0.0, 5.0, 0.0)          # 5 deg downward tilt
    range_max: float = 12.0
    range_min: float = 0.0
    noise_type: str = "gaussian"                    # gaussian | none
    noise_sigma: float = 0.01                       # m
    period_s: float = 0.25                          # acquisition period (4 Hz); scans at control instants (scheduler)
    include_floor: bool = True

    def explicit_table(self):
        """(azimuths, elevations, grid_row, grid_col) of the file / inline ray table, or None (use the grid)."""
        if self.ray_table:
            return load_ray_table(self.ray_table)
        if self.azimuths_deg is not None:
            el = list(self.elevations_deg) if self.elevations_deg is not None else [0.0] * len(self.azimuths_deg)
            if len(el) != len(self.azimuths_deg):
                raise ValueError("lidar3d.azimuths_deg and lidar3d.elevations_deg must have the same length")
            return ([float(a) for a in self.azimuths_deg], [float(e) for e in el],
                    None if self.grid_row is None else [int(r) for r in self.grid_row],
                    None if self.grid_col is None else [int(c) for c in self.grid_col])
        return None

    def to_ray_lidar_cfg(self) -> RayLidarCfg:
        table = self.explicit_table()
        if table is not None:
            az, el, row, col = table
            shape = (max(row) + 1, max(col) + 1) if row is not None and col is not None else None
            order = "explicit"
        else:
            az, el, row, col = grid_pattern(tuple(self.h_fov_deg), tuple(self.v_fov_deg), int(self.cols), int(self.rows),
                                            self.placement, self.ray_order)
            shape, order = (int(self.rows), int(self.cols)), self.ray_order
        if self.noise_type not in ("gaussian", "none"):
            raise ValueError(f"lidar3d.noise_type must be 'gaussian' or 'none', got {self.noise_type!r}")
        return RayLidarCfg(sensor_type="lidar3d", mount_pos=tuple(float(v) for v in self.mount_pos),
                           mount_rpy_deg=tuple(float(v) for v in self.mount_rpy_deg),
                           azimuths_deg=az, elevations_deg=el, grid_shape=shape, grid_row=row, grid_col=col,
                           ray_order=order, range_max=float(self.range_max), range_min=float(self.range_min),
                           include_floor=bool(self.include_floor),
                           noise=GaussianRangeNoiseCfg(sigma=float(self.noise_sigma), enabled=self.noise_type == "gaussian"))


def load_ray_table(path: str):
    """Ray table file -> (azimuths_deg, elevations_deg, grid_row | None, grid_col | None). Columns / arrays:
    azimuth_deg, elevation_deg (sensor frame: x forward, y left, z up; azimuth CCW from +x, elevation up), optional
    grid_row (0 = top), grid_col (0 = left). Row order = ray index."""
    import csv
    import os

    import numpy as np

    if not os.path.isabs(path):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        path = os.path.join(root, path)
    if path.endswith(".npz"):
        d = dict(np.load(path))
    else:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    if "azimuth_deg" not in d or "elevation_deg" not in d:
        raise ValueError(f"ray table {path}: needs azimuth_deg and elevation_deg")
    az, el = [float(v) for v in d["azimuth_deg"]], [float(v) for v in d["elevation_deg"]]
    if len(az) != len(el):
        raise ValueError(f"ray table {path}: azimuth_deg / elevation_deg lengths differ")
    has_grid = "grid_row" in d and "grid_col" in d
    return (az, el, [int(v) for v in d["grid_row"]] if has_grid else None,
            [int(v) for v in d["grid_col"]] if has_grid else None)


class RayLidar(RangeSensor):
    def __init__(self, cfg: RayLidarCfg, raycaster, device):
        self.cfg = cfg
        self.rc = raycaster                       # the scene's MultiMeshRayCaster instance
        self.device = device
        self.range_max = cfg.range_max
        self.num_rays = cfg.num_rays
        self.noise = make_range_noise(cfg.noise, device)
        az = torch.deg2rad(torch.tensor(cfg.azimuths_deg, dtype=torch.float32, device=device))
        if cfg.elevations_deg is None:            # planar: exactly the historical 2D directions
            dirs_s = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], -1)
        else:
            el = torch.deg2rad(torch.tensor(cfg.elevations_deg, dtype=torch.float32, device=device))
            dirs_s = torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], -1)
        self._dirs_s = dirs_s                     # sensor frame
        q = torch.tensor(rpy_to_quat_wxyz(cfg.mount_rpy_deg), dtype=torch.float32, device=device)
        self._dirs_b = dirs_s if tuple(cfg.mount_rpy_deg) == (0.0, 0.0, 0.0) else quat_apply(q.expand(len(dirs_s), 4), dirs_s)
        self._mount = torch.tensor(cfg.mount_pos, dtype=torch.float32, device=device)
        if cfg.ray_alignment != "base":
            raise NotImplementedError("RayLidar origin computation implemented for ray_alignment='base'")

    @property
    def dirs_s(self) -> torch.Tensor:
        """(R, 3) ray directions in the sensor frame."""
        return self._dirs_s

    @property
    def dirs_b(self) -> torch.Tensor:
        """(R, 3) ray directions in base_link (mount rotation applied)."""
        return self._dirs_b

    @property
    def origin_w(self) -> torch.Tensor:
        """(N, 3) world position of the ray origin (optical centre)."""
        d = self.rc.data
        return d.pos_w + quat_apply(d.quat_w, self._mount.expand(d.pos_w.shape[0], -1))

    def reset(self, env_ids=None):
        self.rc.reset(env_ids)

    def read(self, sim_time=0.0, noise_keys=None) -> SensorOutput:
        """noise_keys = (seed (N,), tick (N,)): reproducible per-(env, acquisition tick, ray) noise (see noise.py)."""
        data = self.rc.data                        # triggers the (lazy) ray cast
        d = torch.linalg.norm(data.ray_hits_w - self.origin_w.unsqueeze(1), dim=-1)
        hit = torch.isfinite(d) & (d <= self.range_max)
        d = torch.where(hit, d, torch.full_like(d, self.range_max))
        if self.cfg.range_min > 0 and self.cfg.below_min_policy == "max":
            short = hit & (d < self.cfg.range_min)
            d = torch.where(short, torch.full_like(d, self.range_max), d)
            hit = hit & ~short
        clean = d
        noisy = self.noise(d, hit, self.range_max, keys=noise_keys)
        n = d.shape[0]
        t = torch.as_tensor(sim_time, dtype=torch.float32, device=self.device)
        return SensorOutput(ranges=noisy, ranges_clean=clean, hit=hit, valid=torch.ones_like(hit),
                            dirs_b=self._dirs_b.unsqueeze(0).expand(n, -1, -1), t_stamp=t.expand(n, self.num_rays).clone()
                            if t.dim() == 0 else t[:, None].expand(-1, self.num_rays).clone(),
                            range_max=self.range_max, range_min=self.cfg.range_min)
