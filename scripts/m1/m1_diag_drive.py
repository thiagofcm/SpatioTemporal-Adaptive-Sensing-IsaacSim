"""M1 diagnostic: drive tracking vs wheel slip. Per env: (damping, effort limit) variant x command.
Reports wheel joint velocity vs target (drive error) and body twist vs wheel kinematics (slip)."""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--physics_hz", type=float, default=200.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from st_nav.calibration.sim_harness import Driver, default_geometry, make_scene  # noqa: E402
from st_nav.control.diff_drive import CommandModel  # noqa: E402

VARIANTS = [(5.0, 1.0), (5.0, 10.0), (50.0, 10.0), (500.0, 10.0), (5000.0, 100.0)]
CMDS = [(0.0, 0.5), (0.0, 2.0), (-0.22, 0.0), (0.22, 0.0), (0.1, 1.0)]
N = len(VARIANTS) * len(CMDS)
dt = 1.0 / args.physics_hz
geo = default_geometry()
print('[DBG] make_scene', flush=True)
sim, scene, robot, wheel_ids = make_scene(N, dt, args.device)
print('[DBG] scene ok', flush=True)
dev = args.device
damp = torch.tensor([v[0] for v in VARIANTS for _ in CMDS], device=dev)
eff = torch.tensor([v[1] for v in VARIANTS for _ in CMDS], device=dev)
robot.write_joint_damping_to_sim(damp[:, None].repeat(1, 2), joint_ids=wheel_ids)
robot.write_joint_effort_limit_to_sim(eff[:, None].repeat(1, 2), joint_ids=wheel_ids)
print('[DBG] gains written', flush=True)
cm = CommandModel(N, dt, geo, device=dev)
drv = Driver(sim, scene, robot, wheel_ids, cm)
cv = torch.tensor([c[0] for _ in VARIANTS for c in CMDS], device=dev)
cw = torch.tensor([c[1] for _ in VARIANTS for c in CMDS], device=dev)
acc = {k: torch.zeros(N, device=dev) for k in ("v", "w", "jl", "jr", "tl", "tr", "tau_l", "tau_r")}
n = 0
for k in range(int(5.0 / dt)):
    cm.set_command(cv if k * dt > 1.0 else 0 * cv, cw if k * dt > 1.0 else 0 * cw)
    drv.physics_step()
    if k % 200 == 0: print('[DBG] step', k, flush=True)
    if k * dt > 3.0:
        v, w = drv.twist()
        jv = robot.data.joint_vel[:, wheel_ids]
        tg = robot.data.joint_vel_target[:, wheel_ids]
        tq = robot.data.applied_torque[:, wheel_ids]
        for key, val in (("v", v), ("w", w), ("jl", jv[:, 0]), ("jr", jv[:, 1]), ("tl", tg[:, 0]), ("tr", tg[:, 1]),
                         ("tau_l", tq[:, 0]), ("tau_r", tq[:, 1])):
            acc[key] += val
        n += 1
r, L = geo.wheel_radius, geo.track_width
for i in range(N):
    a = {k: float(v[i] / n) for k, v in acc.items()}
    v_kin = r * (a["jl"] + a["jr"]) / 2
    w_kin = r * (a["jr"] - a["jl"]) / L
    print(f"[DRV] D={VARIANTS[i // len(CMDS)][0]:7.1f} E={VARIANTS[i // len(CMDS)][1]:5.1f} cmd=({cv[i]:+.2f},{cw[i]:+.1f}) "
          f"wheel tgt=({a['tl']:+.2f},{a['tr']:+.2f}) meas=({a['jl']:+.2f},{a['jr']:+.2f}) tau=({a['tau_l']:+.3f},{a['tau_r']:+.3f}) "
          f"| kin v,w=({v_kin:+.3f},{w_kin:+.3f}) body v,w=({a['v']:+.3f},{a['w']:+.3f})", flush=True)
os._exit(0)
