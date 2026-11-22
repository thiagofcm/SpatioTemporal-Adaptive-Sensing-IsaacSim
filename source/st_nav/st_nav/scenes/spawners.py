"""Spawner for a static group of primitive colliders (all walls of a layout, or all static cylinders) under one Xform.

One prim per group keeps the scene graph small and lets the mesh ray caster merge the whole group into one mesh per
environment (identical groups are stored once and referenced by every environment)."""
from __future__ import annotations

from dataclasses import MISSING

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage, standardize_xform_ops
from isaaclab.utils import configclass


@clone
def spawn_static_primitive_group(prim_path: str, cfg: "StaticPrimitiveGroupCfg", translation=None, orientation=None,
                                 **kwargs) -> Usd.Prim:
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    standardize_xform_ops(root.GetPrim(), translation=translation, orientation=orientation)
    for i, (cx, cy, length, thickness, height, yaw) in enumerate(cfg.boxes):
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/box_{i:02d}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*cfg.box_color)])
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, height / 2.0))
        xf.AddRotateZOp().Set(float(yaw) * 180.0 / 3.141592653589793)
        xf.AddScaleOp().Set(Gf.Vec3f(length, thickness, height))
        if cfg.collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    for j, (cx, cy, r, height) in enumerate(cfg.cylinders):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/cyl_{j:02d}")
        cyl.CreateRadiusAttr(float(r))
        cyl.CreateHeightAttr(float(height))
        cyl.CreateAxisAttr("Z")
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*cfg.cylinder_color)])
        UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(cx, cy, height / 2.0))
        if cfg.collision:
            UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return root.GetPrim()


@configclass
class StaticPrimitiveGroupCfg(SpawnerCfg):
    func = spawn_static_primitive_group
    boxes: list = MISSING            # (cx, cy, length, thickness, height, yaw)
    cylinders: list = []             # (cx, cy, r, height)
    collision: bool = True
    box_color: tuple = (0.55, 0.55, 0.6)
    cylinder_color: tuple = (0.75, 0.45, 0.2)
