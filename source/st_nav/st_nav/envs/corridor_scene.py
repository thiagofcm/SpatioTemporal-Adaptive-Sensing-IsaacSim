"""M3 scene manager for `corridor_static_object_random` (articulated TB4, 2D LiDAR, GPU-vectorized resets).

This is the simulation layer the DirectRLEnv (M4) wraps: scene construction, randomization, resets, robot stepping
with the calibrated command model, ground-truth geometry and collision flags, and the sensor. No MDP logic here.

Single source of geometry: per-env wall tables (fixed walls + Z1-Z4 baffles, [cx, cy, hx, hy, yaw]) feed
  * the USD/PhysX scene (static walls; kinematic baffle bodies posed at reset),
  * the ray-caster targets (walls static, baffles tracked),
  * the analytic ground truth (footprint clearance, collisions) — independent of the sensor.
Must be imported after the app is launched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import euler_xyz_from_quat
from pxr import Gf, UsdGeom

from st_nav import config as C
from st_nav.assets.turtlebot4 import TB4_CFG, TB4_COMMAND_MODEL, TB4_GEOMETRY
from st_nav.calibration.sim_harness import CASTER_FRICTION, GROUND_FRICTION, bind_caster_material
from st_nav.control.diff_drive import CommandModel
from st_nav.geometry.analytic2d import wall_clearance, walls_from_specs
from st_nav.scenes import corridor_static_random as S
from st_nav.scenes.spawners import StaticPrimitiveGroupCfg
from st_nav.sensors.lidar2d import Lidar2D, Lidar2DCfg

FOOTPRINT_RADIUS = 0.177       # m, TB4 bumper (front); rear base is 0.164 m -> conservative by 13 mm at the rear
SAFETY_MARGIN = 0.05           # m, reference margin (THRESHOLD_COLLISION 0.22 = 0.17 + 0.05)


@dataclass
class CorridorSceneCfg:
    num_envs: int = 1024
    physics_hz: float = C.PHYSICS_HZ
    decimation: int = C.DECIMATION
    env_spacing: float = 7.0
    bank_size: int = 1024                   # distinct Z size configurations (reference sampler, seeds bank_seed..)
    bank_seed: int = 0
    offset_candidates: int = 32
    start_pose_dr: bool = False             # reference ST01: fixed start; ST04-07: start-pose DR
    goals: tuple = S.GOAL_LIST_ST01
    footprint_radius: float = FOOTPRINT_RADIUS
    safety_margin: float = SAFETY_MARGIN
    wall_height: float = C.WALL_HEIGHT
    baffle_height: float = C.WALL_HEIGHT    # baffles are wall segments in the reference (same 0.6 m height)
    contact_force_threshold: float = 1.0    # N, horizontal contact force on base_link counted as physical contact
    lidar: Lidar2DCfg = field(default_factory=Lidar2DCfg)
    seed: int = 0
    # ---- evaluation options (defaults = training behaviour) ----
    bank_assignment: str = "random"         # "sequential": env i uses bank entry i (seed bank_seed + i)
    offsets_from_bank: bool = False         # True: Z offsets = the bank entry's own (reference seed) offsets, no resampling
    start_poses: tuple | None = None        # per-env (x, y, yaw) used at every reset instead of sampling


class CorridorScene:
    """Standalone use (M3 scripts): CorridorScene(cfg, device) builds simulation + scene and resets all envs.
    Inside a DirectRLEnv (M4): CorridorScene.deferred(cfg, device) -> build_scene_cfg() for the env's scene ->
    prepare_stage() in _setup_scene (before physics starts) -> attach(scene, sim) after the env started physics."""

    def __init__(self, cfg: CorridorSceneCfg, device: str = "cuda:0", _deferred: bool = False):
        self._init_bank(cfg, device)
        if _deferred:
            return
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, device=device,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=GROUND_FRICTION,
                                                            dynamic_friction=GROUND_FRICTION, restitution=0.0),
            physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=2 ** 19)))
        scene = InteractiveScene(self.build_scene_cfg())
        self.prepare_stage()
        self.sim.reset()
        self.attach(scene, self.sim)
        self.reset(torch.arange(self.n, device=device))

    @classmethod
    def deferred(cls, cfg: CorridorSceneCfg, device: str):
        return cls(cfg, device, _deferred=True)

    def _init_bank(self, cfg, device):
        self.cfg, self.device, self.n = cfg, device, cfg.num_envs
        self.dt = 1.0 / cfg.physics_hz
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(cfg.seed)
        n = self.n
        # size bank (reference sampler) and per-env assignment
        self.bank = S.sample_bank(min(cfg.bank_size, n), cfg.bank_seed)
        k = len(self.bank["seeds"])
        rng = np.random.default_rng(cfg.seed)
        if cfg.bank_assignment == "sequential":
            self.bank_idx = torch.arange(n, device=device) % k
        else:
            self.bank_idx = torch.tensor(rng.integers(0, k, n), device=device)
        bl = torch.tensor(self.bank["lengths"], dtype=torch.float32, device=device)
        bt = torch.tensor(self.bank["thicknesses"], dtype=torch.float32, device=device)
        bo = torch.tensor(self.bank["offsets"], dtype=torch.float32, device=device)
        self.z_len, self.z_thick, self.z_bank_off = bl[self.bank_idx], bt[self.bank_idx], bo[self.bank_idx]
        self.z_off = self.z_bank_off.clone()

    def build_scene_cfg(self, scene_cfg: InteractiveSceneCfg | None = None) -> InteractiveSceneCfg:
        """Adds terrain, robot, walls, baffles, LiDAR and contact sensor to scene_cfg (or a new one)."""
        cfg = self.cfg
        sc = scene_cfg or InteractiveSceneCfg(num_envs=self.n, env_spacing=cfg.env_spacing)
        sc.num_envs, sc.env_spacing, sc.replicate_physics = self.n, cfg.env_spacing, False
        sc.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane",
                                        physics_material=sim_utils.RigidBodyMaterialCfg(
                                            static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION))
        sc.robot = TB4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        boxes = [(w[0], w[1], w[2], S.WALL_THICKNESS, cfg.wall_height, w[3]) for w in S.FIXED_WALLS]
        sc.walls = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Walls", spawn=StaticPrimitiveGroupCfg(boxes=boxes, cylinders=[]))
        for i in range(4):
            setattr(sc, f"z{i + 1}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Z{i + 1}",
                spawn=sim_utils.CuboidCfg(
                    size=(1.0, 1.0, cfg.baffle_height),          # unit footprint, scaled per env in prepare_stage
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.15))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, cfg.baffle_height / 2))))
        targets = [MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Walls", track_mesh_transforms=False)]
        targets += [MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/Z{i + 1}",
                                                           track_mesh_transforms=True) for i in range(4)]
        self.lidar_cfg = cfg.lidar
        self.lidar_cfg.targets = targets
        sc.lidar = self.lidar_cfg.raycaster_cfg()
        sc.contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link", history_length=1)
        return sc

    def prepare_stage(self):
        """USD edits that must precede physics start: per-env baffle sizes and the frictionless caster."""
        self._scale_baffles()
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=CASTER_FRICTION, dynamic_friction=CASTER_FRICTION,
                                             restitution=0.0, friction_combine_mode="min")
        mat.func("/World/Materials/caster", mat)
        bind_caster_material(self.n, "/World/Materials/caster")

    def attach(self, scene: InteractiveScene, sim):
        """Runtime handles (after physics started)."""
        device, n = self.device, self.n
        self.scene, self.sim = scene, sim
        self.robot = self.scene["robot"]
        self.baffles = [self.scene[f"z{i + 1}"] for i in range(4)]
        self.wheel_ids, _ = self.robot.find_joints(["left_wheel_joint", "right_wheel_joint"], preserve_order=True)
        self.lidar = Lidar2D(self.lidar_cfg, self.scene["lidar"], device)
        self.contact = self.scene["contact"]
        self.cm = CommandModel(n, self.dt, TB4_GEOMETRY, TB4_COMMAND_MODEL, device=device)
        self.origins = self.scene.env_origins
        self.fixed_walls = walls_from_specs(S.FIXED_WALLS, device=device)            # (10, 5) [cx, cy, hx, hy, yaw]
        self.walls = torch.zeros(n, len(S.FIXED_WALLS) + 4, 5, device=device)
        self.goal = torch.zeros(n, 2, device=device)
        self.start = torch.zeros(n, 3, device=device)
        self.sim_time = torch.zeros(n, device=device)
        self.offset_fallbacks = 0

    # ------------------------------------------------------------------ construction helpers
    def _scale_baffles(self):
        """Per-env baffle sizes as USD scale on the (unit-footprint) rigid-body prims, before physics starts."""
        stage = sim_utils.get_current_stage()
        L, T = self.z_len.cpu().numpy(), self.z_thick.cpu().numpy()
        for e in range(self.n):
            for i in range(4):
                prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Z{i + 1}")
                xf = UsdGeom.Xformable(prim)
                op = next((o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeScale), None)
                if op is None:
                    op = xf.AddScaleOp()
                op.Set(Gf.Vec3d(float(L[e, i]), float(T[e, i]), 1.0))

    # ------------------------------------------------------------------ resets
    def reset(self, env_ids: torch.Tensor):
        k = len(env_ids)
        if k == 0:
            return
        if self.cfg.offsets_from_bank:
            off = self.z_bank_off[env_ids]
        else:
            off, fb = S.sample_offsets(self.z_len[env_ids], self.z_thick[env_ids], self.z_bank_off[env_ids],
                                       self.cfg.offset_candidates, self.gen)
            self.offset_fallbacks += int(fb.sum())
        self.z_off[env_ids] = off
        zw = S.z_walls(self.z_len[env_ids], self.z_thick[env_ids], off)                 # (k, 4, 5) [cx,cy,L,yaw,T]
        self.walls[env_ids, :len(S.FIXED_WALLS)] = self.fixed_walls
        self.walls[env_ids, len(S.FIXED_WALLS):] = torch.stack(
            [zw[..., 0], zw[..., 1], zw[..., 2] / 2, zw[..., 4] / 2, zw[..., 3]], -1)
        org = self.origins[env_ids]
        for i, b in enumerate(self.baffles):
            yaw = zw[:, i, 3]
            pose = torch.cat([org[:, :2] + zw[:, i, :2], torch.full((k, 1), self.cfg.baffle_height / 2, device=self.device),
                              torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], 1)], 1)
            b.write_root_pose_to_sim(pose, env_ids=env_ids)
        if self.cfg.start_poses is not None:
            self.start[env_ids] = torch.tensor(self.cfg.start_poses, dtype=torch.float32, device=self.device)[env_ids]
        else:
            self.start[env_ids] = S.sample_start(k, self.cfg.start_pose_dr, self.device, self.gen)
        self.goal[env_ids] = S.sample_goal(k, self.cfg.goals, self.device, self.gen)
        st = self.start[env_ids]
        rpos = torch.cat([org[:, :2] + st[:, :2], torch.zeros(k, 1, device=self.device)], 1)
        rq = torch.stack([torch.cos(st[:, 2] / 2), 0 * st[:, 2], 0 * st[:, 2], torch.sin(st[:, 2] / 2)], 1)
        self.robot.write_root_pose_to_sim(torch.cat([rpos, rq], 1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(k, 6, device=self.device), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(torch.zeros(k, 2, device=self.device), torch.zeros(k, 2, device=self.device),
                                            joint_ids=self.wheel_ids, env_ids=env_ids)
        self.robot.set_joint_velocity_target(torch.zeros(k, 2, device=self.device), joint_ids=self.wheel_ids,
                                             env_ids=env_ids)
        self.cm.reset(env_ids)
        self.sim_time[env_ids] = 0.0
        self.scene.reset(env_ids)

    # ------------------------------------------------------------------ stepping
    def step(self, v: torch.Tensor, w: torch.Tensor):
        """One control step (decimation physics steps) with the calibrated command model."""
        self.cm.set_command(v, w)
        for _ in range(self.cfg.decimation):
            wl, wr, _, _ = self.cm.step()
            self.robot.set_joint_velocity_target(torch.stack([wl, wr], -1), joint_ids=self.wheel_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.dt)
        self.sim_time += self.cfg.decimation * self.dt

    # ------------------------------------------------------------------ ground truth (sensor-independent)
    def robot_pose2d(self):
        p = self.robot.data.root_pos_w[:, :2] - self.origins[:, :2]
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        return p, torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi

    def footprint_clearance(self) -> torch.Tensor:
        """c: minimum distance from the robot footprint (circle r_fp at base_link) to any wall / baffle surface."""
        p, _ = self.robot_pose2d()
        return wall_clearance(p, self.walls) - self.cfg.footprint_radius

    def goal_distance(self) -> torch.Tensor:
        p, _ = self.robot_pose2d()
        return torch.linalg.norm(self.goal - p, dim=1)

    def contact_force_xy(self) -> torch.Tensor:
        """Horizontal contact force on base_link (bumper/shell/base); the caster is frictionless, so on flat ground
        this is non-zero only for contact with walls / baffles."""
        return torch.linalg.norm(self.contact.data.net_forces_w[:, 0, :2], dim=-1)

    def collision_flags(self):
        c = self.footprint_clearance()
        return {"clearance": c, "physical_collision": c <= 0.0, "safety_violation": c < self.cfg.safety_margin,
                "contact": self.contact_force_xy() > self.cfg.contact_force_threshold}

    def read_lidar(self):
        return self.lidar.read(self.sim_time)
