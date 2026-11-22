"""M1 diagnostic: PhysX-side drive parameters and wheel/base time series during an in-place turn."""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--physics_hz", type=float, default=200.0)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import numpy as np, torch
from st_nav.calibration.sim_harness import Driver, default_geometry, make_scene
from st_nav.control.diff_drive import CommandModel
dt = 1.0 / args.physics_hz; geo = default_geometry()
sim, scene, robot, wid = make_scene(1, dt, args.device)
v = robot.root_physx_view
print("[PX] dof names", robot.joint_names, flush=True)
for name in ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_max_forces", "get_dof_max_velocities", "get_dof_armatures",
             "get_dof_friction_coefficients"):
    try: print(f"[PX] {name}", getattr(v, name)().tolist(), flush=True)
    except Exception as e: print(f"[PX] {name} n/a {e}", flush=True)
cm = CommandModel(1, dt, geo, device=args.device); drv = Driver(sim, scene, robot, wid, cm)
rows = []
for k in range(int(3.0 / dt)):
    cm.set_command(torch.zeros(1, device=args.device), torch.full((1,), 0.5 if k * dt > 0.5 else 0.0, device=args.device))
    drv.physics_step()
    jv = robot.data.joint_vel[0, wid].tolist(); tg = robot.data.joint_vel_target[0, wid].tolist()
    bz = robot.data.body_pos_w[0, :, 2].tolist()
    rows.append([k * dt] + jv + tg + [float(robot.data.root_ang_vel_b[0, 2])] + bz)
a = np.array(rows)
sel = a[:, 0] > 1.5
print("[TS] mean jl,jr", a[sel, 1].mean(), a[sel, 2].mean(), "std", a[sel, 1].std(), a[sel, 2].std(), flush=True)
print("[TS] target", a[sel, 3].mean(), a[sel, 4].mean(), " w body", a[sel, 5].mean(), a[sel, 5].std(), flush=True)
print("[TS] body z (base, lw, rw) min/max", a[sel, 6:].min(0).tolist(), a[sel, 6:].max(0).tolist(), flush=True)
for r in a[(a[:, 0] > 1.5) & (a[:, 0] < 1.6)]: print("[TS]", np.round(r[:6], 3).tolist(), flush=True)
os._exit(0)
