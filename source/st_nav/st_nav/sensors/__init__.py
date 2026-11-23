"""Range sensors behind a common SensorOutput interface: one ray-casting LiDAR (lidar.py) with a 2D preset and a 3D
(Mirrorcle raster approximation) configuration."""
from .base import RangeSensor, SensorOutput  # noqa: F401
