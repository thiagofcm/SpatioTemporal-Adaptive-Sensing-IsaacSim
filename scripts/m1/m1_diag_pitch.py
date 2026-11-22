"""M1 diagnostic: pitch / wheel-odometry vs base twist when starting from rest (calibrated command model)."""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--amax", default="0,1.0,0.6"); p.add_argument("--physics_hz", type=float, default=200.0)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import numpy as np, torch
from st_nav.calibration.sim_harness import Driver, default_geometry, make_scene
from st_nav.control.diff_drive import CommandModel
amax = [float(x) for x in args.amax.split(",")]
N = len(amax); dt = 1.0 / args.physics_hz; geo = default_geometry()
sim, scene, robot, wid = make_scene(N, dt, args.device)
cm = CommandModel(N, dt, geo, device=args.device)
cm.set_params(delay_s=0.16, tau_v_s=0.05, delay_w_s=0.12, tau_w_s=0.05, k_w=0.9)
drv = Driver(sim, scene, robot, wid, cm)
a_lim = torch.tensor(amax, device=args.device)
prev = torch.zeros(N, device=args.device)
log = []
for k in range(int(3.0 / dt)):
    tk = k * dt
    cm.set_command(torch.full((N,), 0.15 if tk > 0.5 else 0.0, device=args.device), torch.full((N,), 0.3 if tk > 0.5 else 0.0, device=args.device))
    # optional acceleration limit on the lagged linear state (diagnostic only)
    wl, wr, v_eff, w_eff = cm.step()
    lim = torch.where(a_lim > 0, a_lim * dt, torch.full_like(a_lim, 1e9))
    v_new = prev + torch.clamp(cm.state[:, 0] - prev, -lim, lim)
    cm.state[:, 0] = v_new; prev = v_new.clone()
    half = 0.5 * geo.track_width
    tgt = torch.stack([(v_new - cm.state[:, 1] * half) / geo.wheel_radius, (v_new + cm.state[:, 1] * half) / geo.wheel_radius], -1)
    robot.set_joint_velocity_target(tgt, joint_ids=wid); scene.write_data_to_sim(); sim.step(render=False); scene.update(dt)
    b_v, _ = drv.twist(); jv = robot.data.joint_vel[:, wid]
    odo_v = geo.wheel_radius * jv.mean(1)
    _, pitch, _ = drv.pose2d()[1:]
    log.append(torch.stack([b_v, odo_v, torch.rad2deg(pitch)], 1).cpu().numpy())
L = np.array(log)
for i, a in enumerate(amax):
    print(f"[PITCH] amax={a}: max base v {L[:, i, 0].max():.3f}  max wheel-odom v {L[:, i, 1].max():.3f}  pitch min/max {L[:, i, 2].min():+.2f}/{L[:, i, 2].max():+.2f} deg", flush=True)
os._exit(0)
