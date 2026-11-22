"""Scene construction from the reference simplified environment's 2D definitions (walls as oriented boxes, obstacles as
cylinders), extruded to 3D for Isaac Lab.

The same 2D tables also feed the analytic geometry (st_nav.geometry.analytic2d) used for ground truth; both views of
the scene come from one source, so the ray caster, the physics and the ground-truth clearance see the same geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.terrains import TerrainImporterCfg

from st_nav.scenes.spawners import StaticPrimitiveGroupCfg

WALL_HEIGHT = 0.6           # m (approved)
OBSTACLE_HEIGHT = 0.5       # m (approved, configurable)
DEFAULT_WALL_THICKNESS = 0.15


def yaw_quat(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


@dataclass
class SceneSpec:
    walls: list                                   # reference wall specs (cx, cy, length, yaw[, thickness])
    static_cylinders: list = field(default_factory=list)   # (cx, cy, r)
    moving_cylinders: list = field(default_factory=list)   # radii; positions set at runtime
    wall_height: float = WALL_HEIGHT
    obstacle_height: float = OBSTACLE_HEIGHT


def add_scene_assets(scene_cfg: InteractiveSceneCfg, spec: SceneSpec, ground_friction: float = 1.0) -> list:
    """Adds ground, walls, static and moving cylinders to scene_cfg (in place). Returns the ray-cast target list."""
    scene_cfg.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=ground_friction, dynamic_friction=ground_friction,
                                                        restitution=0.0))
    boxes = [(w[0], w[1], w[2], (w[4] if len(w) > 4 else DEFAULT_WALL_THICKNESS), spec.wall_height, w[3])
             for w in spec.walls]
    scene_cfg.walls = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Walls",
                                   spawn=StaticPrimitiveGroupCfg(boxes=boxes, cylinders=[]))
    if spec.static_cylinders:
        scene_cfg.cylinders = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Cylinders",
            spawn=StaticPrimitiveGroupCfg(boxes=[], cylinders=[(cx, cy, r, spec.obstacle_height)
                                                                for cx, cy, r in spec.static_cylinders]))
    for k, r in enumerate(spec.moving_cylinders):
        setattr(scene_cfg, f"mover_{k}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Mover_{k}",
            spawn=sim_utils.CylinderCfg(
                radius=r, height=spec.obstacle_height, axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, spec.obstacle_height / 2 + 0.001))))
    targets = [MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Walls", track_mesh_transforms=False)]
    if spec.static_cylinders:
        targets.append(MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Cylinders",
                                                              track_mesh_transforms=False))
    for k in range(len(spec.moving_cylinders)):
        targets.append(MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/Mover_{k}",
                                                              track_mesh_transforms=True))
    return targets
