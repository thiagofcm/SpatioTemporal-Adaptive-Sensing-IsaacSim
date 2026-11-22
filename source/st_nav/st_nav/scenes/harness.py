"""Multi-env TurtleBot4 + arena + 2D LiDAR scene (no RL wrapper), for M2 validation and benchmarks and as the building
block of the DirectRLEnv in M3/M4. Must be imported after the app has been launched."""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from st_nav.assets.turtlebot4 import TB4_CFG
from st_nav.calibration.sim_harness import CASTER_FRICTION, GROUND_FRICTION, bind_caster_material
from st_nav.scenes.builder import SceneSpec, add_scene_assets
from st_nav.sensors.lidar2d import Lidar2D, Lidar2DCfg


def make_nav_scene(num_envs: int, physics_dt: float, device: str, spec: SceneSpec, lidar_cfg: Lidar2DCfg | None,
                   robot_cfg=None, env_spacing: float = 7.0, lazy_sensor_update: bool = True):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=physics_dt, device=device,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=GROUND_FRICTION,
                                                        dynamic_friction=GROUND_FRICTION, restitution=0.0),
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=2 ** 19)))
    scene_cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing, lazy_sensor_update=lazy_sensor_update)
    scene_cfg.robot = (robot_cfg or TB4_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    targets = add_scene_assets(scene_cfg, spec, ground_friction=GROUND_FRICTION)
    if lidar_cfg is not None:
        lidar_cfg.targets = targets
        scene_cfg.lidar = lidar_cfg.raycaster_cfg()
    scene = InteractiveScene(scene_cfg)
    mat_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=CASTER_FRICTION, dynamic_friction=CASTER_FRICTION,
                                             restitution=0.0, friction_combine_mode="min")
    mat_cfg.func("/World/Materials/caster", mat_cfg)
    bind_caster_material(num_envs, "/World/Materials/caster")
    sim.reset()
    robot = scene["robot"]
    wheel_ids, names = robot.find_joints(["left_wheel_joint", "right_wheel_joint"], preserve_order=True)
    lidar = Lidar2D(lidar_cfg, scene["lidar"], device) if lidar_cfg is not None else None
    movers = [scene[f"mover_{k}"] for k in range(len(spec.moving_cylinders))]
    return sim, scene, robot, wheel_ids, lidar, movers
