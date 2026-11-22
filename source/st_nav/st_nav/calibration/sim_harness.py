"""Minimal multi-env scene with TurtleBot4 robots on a flat ground, for M1 validation and calibration.

Must be imported after the Isaac Sim app has been launched (AppLauncher).
"""
from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

from st_nav.assets.turtlebot4 import TB4_CASTER_COLLIDER, TB4_CFG
from st_nav.control.diff_drive import CommandModel, DiffDriveGeometry

GROUND_FRICTION = 1.0       # Gazebo <mu1>/<mu2> = 1.0 on the Create3 wheels
CASTER_FRICTION = 0.0       # ball caster: rolling, modelled as frictionless contact ("min" combine overrides ground)


def bind_caster_material(num_envs: int, material_path: str, robot_rel: str = "Robot"):
    """Bind a physics material to the caster collider of every robot.

    The URDF importer puts each link's colliders under an *instanceable* prim, and instance proxies cannot carry a
    material binding (the binding silently does nothing). The caster's collider group is therefore de-instanced
    (local copy of a 1-sphere subtree) before binding. Returns the number of bound colliders."""
    from pxr import Usd, UsdPhysics

    stage = sim_utils.get_current_stage()
    n = 0
    for i in range(num_envs):
        group = stage.GetPrimAtPath(f"/World/envs/env_{i}/{robot_rel}/{TB4_CASTER_COLLIDER}")
        # de-instance the nearest instanceable ancestor (or the group itself)
        p = group
        while p and p.IsValid() and not p.IsInstance():
            p = p.GetParent()
        if p and p.IsValid() and p.IsInstance():
            p.SetInstanceable(False)
        group = stage.GetPrimAtPath(f"/World/envs/env_{i}/{robot_rel}/{TB4_CASTER_COLLIDER}")
        for prim in Usd.PrimRange(group):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                sim_utils.bind_physics_material(str(prim.GetPath()), material_path)
                n += 1
    if n != num_envs:
        raise RuntimeError(f"caster material bound to {n} colliders, expected {num_envs}")
    return n


def make_scene(num_envs: int, physics_dt: float, device: str, env_spacing: float = 3.0, robot_cfg=None,
               contact_sensor: bool = False):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=physics_dt, device=device,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=GROUND_FRICTION,
                                                        dynamic_friction=GROUND_FRICTION, restitution=0.0)))

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane",
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=GROUND_FRICTION,
                                                            dynamic_friction=GROUND_FRICTION, restitution=0.0))
        robot = (robot_cfg or TB4_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
        contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1) if contact_sensor else None

    scene = InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=env_spacing))
    # frictionless caster in every env (bound before the physics is initialised)
    mat_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=CASTER_FRICTION, dynamic_friction=CASTER_FRICTION,
                                             restitution=0.0, friction_combine_mode="min")
    mat_cfg.func("/World/Materials/caster", mat_cfg)
    bind_caster_material(num_envs, "/World/Materials/caster")
    sim.reset()
    robot = scene["robot"]
    wheel_ids, wheel_names = robot.find_joints(["left_wheel_joint", "right_wheel_joint"], preserve_order=True)
    assert wheel_names == ["left_wheel_joint", "right_wheel_joint"], wheel_names
    return sim, scene, robot, wheel_ids


class Driver:
    """Steps the CommandModel at the physics rate and writes wheel velocity targets."""

    def __init__(self, sim, scene, robot, wheel_ids, cmd_model: CommandModel):
        self.sim, self.scene, self.robot, self.wheel_ids, self.cm = sim, scene, robot, wheel_ids, cmd_model
        self.dt = sim.get_physics_dt()

    def physics_step(self):
        wl, wr, v_eff, w_eff = self.cm.step()
        self.robot.set_joint_velocity_target(torch.stack([wl, wr], dim=-1), joint_ids=self.wheel_ids)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        return v_eff, w_eff

    def twist(self):
        """Measured base twist in the body frame: (v_x, w_z)."""
        return self.robot.data.root_lin_vel_b[:, 0], self.robot.data.root_ang_vel_b[:, 2]

    def pose2d(self):
        p = self.robot.data.root_pos_w - self.scene.env_origins
        r, pch, y = euler_xyz_from_quat(self.robot.data.root_quat_w)
        return p, r, pch, y


def default_geometry() -> DiffDriveGeometry:
    return DiffDriveGeometry()
