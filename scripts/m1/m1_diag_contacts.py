"""M1 diagnostic: where does the TurtleBot4 rest? (normal force per body, CoM, caster material binding)."""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--physics_hz", type=float, default=200.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from pxr import Usd, UsdPhysics, UsdShade  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402

from st_nav.calibration.sim_harness import make_scene  # noqa: E402

sim, scene, robot, wheel_ids = make_scene(2, 1.0 / args.physics_hz, args.device)
stage = sim_utils.get_current_stage()
root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
        print(f"[COLL] {prim.GetPath()} instance_proxy={prim.IsInstanceProxy()} material={mat.GetPath() if mat else None}",
              flush=True)
    if "mesh_2" in str(prim.GetPath()) and prim.GetName() == "mesh_2":
        print(f"[MESH2] {prim.GetPath()} instanceable={prim.IsInstance()} proxy={prim.IsInstanceProxy()} "
              f"rels={[r.GetName() for r in prim.GetRelationships()]}", flush=True)
print("[MASS] body names:", robot.body_names, flush=True)
print("[MASS] masses:", robot.root_physx_view.get_masses()[0].tolist(), flush=True)
print("[MASS] coms (pos,quat):", robot.root_physx_view.get_coms()[0].tolist(), flush=True)
cs = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=1))
sim.reset()
dt = sim.get_physics_dt()
for _ in range(int(2.0 / dt)):
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)
    cs.update(dt)
f = cs.data.net_forces_w[0]
print("[CONTACT] bodies:", cs.body_names, flush=True)
print("[CONTACT] net normal z [N]:", [round(float(v), 3) for v in f[:, 2]], " total weight:",
      float(robot.root_physx_view.get_masses()[0].sum()) * 9.81, flush=True)
print("[POSE] base z, quat:", robot.data.root_pos_w[0].tolist(), robot.data.root_quat_w[0].tolist(), flush=True)
os._exit(0)
