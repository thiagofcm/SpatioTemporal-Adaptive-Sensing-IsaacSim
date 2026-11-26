"""Navigation env config with the fixed 3D LiDAR (Mirrorcle raster approximation) and image observations (M8).

Same environment class, scenes, dynamics, rewards, action limits and terminations as CorridorNavEnvCfg; only the sensor
and the observation layout differ:
  * sensor lidar3d, R4 mount (mount study, infrastructure/M7_lidar3d/mount_study): (0, 0, 0.40) m in base_link,
    pitch 6.75 deg down; full FoV 28 x 14 deg, 139 x 34 rays (cell centres, row-major top-left), 12 m, Gaussian noise
    sigma 0.01 m, 4 Hz (scans at control instants; the latest complete scan is held between acquisitions);
  * obs_layout "image": {"policy": 7 navigation features, "lidar": (1, 34, 139) range image / 12 m}.
"""
from isaaclab.utils import configclass

from st_nav.sensors.lidar import Lidar3DCfg

from .corridor_env_cfg import CorridorNavEnvCfg

R4_MOUNT_POS = (0.0, 0.0, 0.40)
R4_MOUNT_RPY_DEG = (0.0, 6.75, 0.0)


@configclass
class CorridorNavLidar3DEnvCfg(CorridorNavEnvCfg):
    sensor: str = "lidar3d"
    lidar3d: Lidar3DCfg = Lidar3DCfg(mount_pos=R4_MOUNT_POS, mount_rpy_deg=R4_MOUNT_RPY_DEG)
    obs_layout: str = "image"
    observation_space: int = 7                          # "policy" group (navigation features); the image is "lidar"
