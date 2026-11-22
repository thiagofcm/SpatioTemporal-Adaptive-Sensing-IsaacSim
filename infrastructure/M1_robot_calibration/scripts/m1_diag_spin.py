"""M1 diagnostic: in-place turn (w=0.5 and 2.0) and reverse under one drive/contact variant; PhysX values confirmed."""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--effort", type=float, default=1.0); p.add_argument("--damping", type=float, default=5.0)
p.add_argument("--no_caster", action="store_true"); p.add_argument("--physics_hz", type=float, default=200.0)
p.add_argument("--pos_iters", type=int, default=4); p.add_argument("--vel_iters", type=int, default=1); p.add_argument("--in_air", action="store_true")
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import torch
from pxr import Usd, UsdPhysics
import isaaclab.sim as sim_utils
from st_nav.assets.turtlebot4 import TB4_CFG
from st_nav.calibration import sim_harness as H
from st_nav.control.diff_drive import CommandModel
cfg = TB4_CFG.replace(actuators={"wheels": TB4_CFG.actuators["wheels"].replace(effort_limit_sim=args.effort, damping=args.damping)})
cfg.spawn.articulation_props = cfg.spawn.articulation_props.replace(solver_position_iteration_count=args.pos_iters, solver_velocity_iteration_count=args.vel_iters, fix_root_link=args.in_air)
if args.in_air: cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, 0.5))
if args.no_caster:
    _orig = H.bind_caster_material
    def _nocaster(num_envs, mat, robot_rel="Robot"):
        n = _orig(num_envs, mat, robot_rel)
        st = sim_utils.get_current_stage()
        for i in range(num_envs):
            for prim in Usd.PrimRange(st.GetPrimAtPath(f"/World/envs/env_{i}/{robot_rel}/{H.TB4_CASTER_COLLIDER}")):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        return n
    H.bind_caster_material = _nocaster
CMDS = [(0.0, 0.5), (0.0, 2.0), (-0.22, 0.0), (0.22, 0.0)]
N = len(CMDS); dt = 1.0 / args.physics_hz
sim, scene, robot, wid = H.make_scene(N, dt, args.device, robot_cfg=cfg)
v = robot.root_physx_view
print(f"[SPIN] variant effort={args.effort} damping={args.damping} no_caster={args.no_caster} iters={args.pos_iters}/{args.vel_iters} air={args.in_air} hz={args.physics_hz} | physx damp={v.get_dof_dampings()[0].tolist()} maxF={v.get_dof_max_forces()[0].tolist()}", flush=True)
cm = CommandModel(N, dt, H.default_geometry(), device=args.device); drv = H.Driver(sim, scene, robot, wid, cm)
cv = torch.tensor([c[0] for c in CMDS], device=args.device); cw = torch.tensor([c[1] for c in CMDS], device=args.device)
acc = torch.zeros(N, 4, device=args.device); n = 0
for k in range(int(4.0 / dt)):
    cm.set_command(cv if k * dt > 1 else 0 * cv, cw if k * dt > 1 else 0 * cw); drv.physics_step()
    if k * dt > 2.5:
        b_v, b_w = drv.twist(); acc += torch.stack([b_v, b_w, robot.data.joint_vel[:, wid[0]], robot.data.joint_vel[:, wid[1]]], 1); n += 1
acc /= n
for i, c in enumerate(CMDS):
    print(f"[SPIN] cmd={c} body v,w=({acc[i,0]:+.3f},{acc[i,1]:+.3f}) wheels=({acc[i,2]:+.2f},{acc[i,3]:+.2f})", flush=True)
os._exit(0)
