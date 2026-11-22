"""M1 diagnostic: what resists in-place rotation? Effort limit from CLI; full contact forces per body while spinning."""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--effort", type=float, default=10.0); p.add_argument("--physics_hz", type=float, default=200.0)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import torch
from st_nav.assets.turtlebot4 import TB4_CFG
from st_nav.calibration.sim_harness import Driver, default_geometry, make_scene
from st_nav.control.diff_drive import CommandModel
cfg = TB4_CFG.copy(); cfg.actuators["wheels"].effort_limit_sim = args.effort
CMDS = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.0, 2.0), (0.22, 0.0), (-0.22, 0.0)]
N = len(CMDS); dt = 1.0 / args.physics_hz; geo = default_geometry()
sim, scene, robot, wid = make_scene(N, dt, args.device, robot_cfg=cfg, contact_sensor=True)
cs = scene["contacts"]
cm = CommandModel(N, dt, geo, device=args.device); drv = Driver(sim, scene, robot, wid, cm)
cv = torch.tensor([c[0] for c in CMDS], device=args.device); cw = torch.tensor([c[1] for c in CMDS], device=args.device)
acc = None; n = 0
for k in range(int(4.0 / dt)):
    cm.set_command(cv if k * dt > 1 else 0 * cv, cw if k * dt > 1 else 0 * cw); drv.physics_step()
    if k * dt > 2.5:
        v, w = drv.twist(); jv = robot.data.joint_vel[:, wid]; tq = robot.data.applied_torque[:, wid]
        f = cs.data.net_forces_w  # (N, B, 3)
        row = torch.cat([v[:, None], w[:, None], jv, tq, f.reshape(N, -1)], 1)
        acc = row if acc is None else acc + row; n += 1
acc /= n
print("[RES] bodies", cs.body_names, "effort", args.effort, flush=True)
for i, c in enumerate(CMDS):
    a = acc[i].tolist()
    fs = [tuple(round(x, 2) for x in a[6 + 3 * b: 9 + 3 * b]) for b in range(len(cs.body_names))]
    print(f"[RES] cmd={c} body v,w=({a[0]:+.3f},{a[1]:+.3f}) wheel=({a[2]:+.2f},{a[3]:+.2f}) tau=({a[4]:+.3f},{a[5]:+.3f}) F_xyz={fs}", flush=True)
os._exit(0)
