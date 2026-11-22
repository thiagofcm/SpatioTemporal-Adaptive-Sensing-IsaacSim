"""Default simulation configuration of the navigation environment (adopted after M2, 2026-11-22).

  physics 100 Hz, PhysX solver 16 position / 2 velocity iterations, control 5 Hz (decimation 20),
  40-beam 360 deg 2D LiDAR at 5 Hz (one scan per control step), range 3.5 m,
  calibrated actuator model (M1), validated LiDAR + noise model (M2), headless GPU simulation.
The number of parallel environments is a run-time choice (best measured throughput: 16,384 envs).
"""
from dataclasses import dataclass

PHYSICS_HZ = 100.0
CONTROL_HZ = 5.0
DECIMATION = int(round(PHYSICS_HZ / CONTROL_HZ))      # 20
SOLVER_POSITION_ITERATIONS = 16
SOLVER_VELOCITY_ITERATIONS = 2
LIDAR_BEAMS = 40
LIDAR_RANGE_MAX = 3.5
LIDAR_HZ = CONTROL_HZ                                   # one scan per control step
WALL_HEIGHT = 0.6
OBSTACLE_HEIGHT = 0.5


@dataclass
class SimDefaults:
    physics_hz: float = PHYSICS_HZ
    control_hz: float = CONTROL_HZ
    solver_position_iterations: int = SOLVER_POSITION_ITERATIONS
    solver_velocity_iterations: int = SOLVER_VELOCITY_ITERATIONS

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def decimation(self) -> int:
        return int(round(self.physics_hz / self.control_hz))
