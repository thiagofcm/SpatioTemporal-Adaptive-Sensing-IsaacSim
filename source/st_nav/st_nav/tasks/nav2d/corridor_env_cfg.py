"""DirectRLEnv configuration: TB4 navigation in the randomized static corridor (corridor_static_object_random)."""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from st_nav import config as C
from st_nav.calibration.sim_harness import GROUND_FRICTION
from st_nav.sensors.lidar import Lidar3DCfg


@configclass
class RewardCfg:
    progress_scale: float = 10.0          # r_progress = k_p (d_prev - d)
    proximity_scale: float = 1.0          # r_proximity = -k_o max(0, 1 - c / c_safe)^2
    proximity_clearance: float = 0.15     # c_safe [m]
    time_penalty_per_s: float = 0.1       # r_time = -k_t * control_dt (= -0.02 per 0.2 s step)
    goal_reward: float = 30.0
    collision_penalty: float = 30.0       # physical collision or safety violation (applied once, on termination)
    timeout_penalty: float = 0.0          # timeouts are truncations (no terminal reward by default)


@configclass
class CorridorNavEnvCfg(DirectRLEnvCfg):
    # ---- timing (adopted configuration) ----
    decimation: int = C.DECIMATION                       # 20 physics steps per 5 Hz control step
    episode_length_s: float = 70.0                       # 350 control steps (reference episode)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / C.PHYSICS_HZ, render_interval=C.DECIMATION,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=GROUND_FRICTION,
                                                        dynamic_friction=GROUND_FRICTION, restitution=0.0),
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=2 ** 19))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=7.0, replicate_physics=False)
    # ---- spaces ----
    action_space: int = 2                                # (v, w) in [-1, 1] -> +-0.22 m/s, +-2.0 rad/s
    observation_space: int = 47                         # set by the env from the sensor: num_rays + 7 (2D: 40 + 7)
    state_space: int = 0
    # ---- actions ----
    v_max: float = 0.22
    w_max: float = 2.0
    # ---- sensing (independent of the control rate) ----
    sensor: str = "lidar2d"                              # lidar2d (40 beams, 3.5 m, M2) | lidar3d (block below)
    lidar3d: Lidar3DCfg = Lidar3DCfg()                   # 3D LiDAR: grid / ray table, mount pose, range, noise, period
    lidar_period_s: float = 1.0 / C.LIDAR_HZ             # 2D LiDAR period: 0.2 s, one scan per control step
    lidar_noise: bool = True                             # master switch (2D datasheet noise; 3D: and lidar3d.noise_type)
    # ---- observation normalization ----
    goal_distance_norm: float = 7.0                      # > max start-goal distance 6.79 m (reference MAX_GOAL_DISTANCE
    #                                                      5.9397 saturated obs[40] at episode start; changed after M4)
    # ---- terminations ----
    goal_tolerance: float = 0.20
    safety_margin: float = 0.05
    # ---- rewards ----
    rewards: RewardCfg = RewardCfg()
    # ---- scenario ----
    wall_height: float = C.WALL_HEIGHT                   # m, fixed walls (geometry only; the robot is planar)
    baffle_height: float = C.WALL_HEIGHT                 # m, Z1-Z4 baffles
    obstacle_height: float = C.OBSTACLE_HEIGHT           # m, moving obstacles OB1 / OB2
    scene_name: str = "corridor_static_random"           # st_nav.scenes: corridor_static_random | corridor_dynamic_chase_random
    # moving obstacles (dynamic scenes; reference flag names): fixed active set and per-episode randomization
    active_obstacles: tuple = (1, 2)                     # e.g. (1,) = OB1 only, (1, 2) = OB1 + OB2
    obstacle_presence_dr: bool = False
    obstacle_speed_dr: bool = False
    obstacle_y_dr: bool = False
    obstacle_radius_dr: bool = False                     # not supported (mesh sizes baked at start-up)
    start_pose_dr: bool = True                           # reference ST04-07: 5 start points, yaw pi +- 10 deg, per reset (M5)
    bank_size: int = 1024
    privileged_critic: bool = False                      # optional critic observation group (clean scan + GT)
    # ---- evaluation overrides (None/defaults = training behaviour; see CorridorSceneCfg) ----
    bank_seed: int = 0
    bank_assignment: str = "random"
    offsets_from_bank: bool = False
    start_poses: tuple | None = None
    goals: tuple | None = None                           # None: reference ST01 9-goal list
