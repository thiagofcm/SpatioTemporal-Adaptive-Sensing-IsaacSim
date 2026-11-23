"""Conventional 2D LiDAR (40 beams, 9 deg, 3.5 m, TB4 RPLIDAR mount, datasheet noise): the default preset of the
generic ray-casting LiDAR (st_nav.sensors.lidar). Kept under its historical names for the M2-M6 code and scripts.

Mounting (TurtleBot4 Standard URDF, verified in M1): rplidar_link at (-0.040, 0, 0.1929) m in base_link. The rays are
defined directly in the base frame (beam 0 = robot forward, CCW); the physical rplidar_link is yawed +90 deg, which only
matters when mapping real scans to beams.
"""
from st_nav.sensors.lidar import TB4_LIDAR_MOUNT  # noqa: F401
from st_nav.sensors.lidar import RayLidar as Lidar2D  # noqa: F401
from st_nav.sensors.lidar import RayLidarCfg as Lidar2DCfg  # noqa: F401
