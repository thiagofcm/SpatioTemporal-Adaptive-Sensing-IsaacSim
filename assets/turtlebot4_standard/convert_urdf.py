"""Convert the cleaned TurtleBot4 Standard URDF to USD with Isaac Lab's UrdfConverter and report the result.

    isaacrun -g auto -- python assets/turtlebot4_standard/convert_urdf.py --headless

Output: assets/turtlebot4_standard/usd/turtlebot4_standard.usd (+ configuration/ sublayers written by the importer).
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--merge_fixed_joints", type=int, default=1)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = UrdfConverterCfg(
    asset_path=os.path.join(HERE, "turtlebot4_standard.urdf"),
    usd_dir=os.path.join(HERE, "usd"),
    usd_file_name="turtlebot4_standard.usd",
    force_usd_conversion=True,
    make_instanceable=False,
    fix_base=False,
    root_link_name="base_link",
    merge_fixed_joints=bool(args.merge_fixed_joints),
    collider_type="convex_hull",
    self_collision=False,
    # Wheel joints: velocity drives. Gains are overridden at runtime by the ArticulationCfg actuator.
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="velocity",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=10.0),
    ),
)
conv = UrdfConverter(cfg)
print(f"[CONVERT] usd: {conv.usd_path}", flush=True)

stage = Usd.Stage.Open(conv.usd_path)
print("[CONVERT] defaultPrim:", stage.GetDefaultPrim().GetPath(), flush=True)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
total_mass = 0.0
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        m = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
        total_mass += m or 0.0
        print(f"[BODY ] {path} mass={m}", flush=True)
    elif prim.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(prim)
        print(f"[JOINT] {path} type={prim.GetTypeName()} b0={j.GetBody0Rel().GetTargets()} "
              f"b1={j.GetBody1Rel().GetTargets()}", flush=True)
    elif prim.HasAPI(UsdPhysics.CollisionAPI):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = r.GetMin(), r.GetMax()
        approx = prim.GetAttribute("physics:approximation").Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else "-"
        print(f"[COLL ] {path} [{prim.GetTypeName()}] approx={approx} x[{mn[0]:+.3f},{mx[0]:+.3f}] "
              f"y[{mn[1]:+.3f},{mx[1]:+.3f}] z[{mn[2]:+.3f},{mx[2]:+.3f}]", flush=True)
    elif "rplidar" in path.lower() and prim.IsA(UsdGeom.Xformable) and path.count("/") <= 3:
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        print(f"[FRAME] {path} pos={tuple(round(v, 4) for v in t.ExtractTranslation())}", flush=True)
print(f"[CONVERT] total rigid-body mass = {total_mass:.3f}", flush=True)
app.close()
