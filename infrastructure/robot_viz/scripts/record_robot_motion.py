"""Record body poses of the articulated TB4 executing a command program (calibrated command model), for offline
rendering. Headless physics only (no RTX).

    isaacrun -g auto -- python infrastructure/robot_viz/scripts/record_robot_motion.py --headless
"""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--fps", type=float, default=30.0)
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "infrastructure", "robot_viz", "results"))
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import numpy as np, torch
from st_nav.assets.turtlebot4 import TB4_COMMAND_MODEL, TB4_GEOMETRY, TB4_PHYSICS_DT
from st_nav.calibration.sim_harness import Driver, make_scene
from st_nav.control.diff_drive import CommandModel
# (t_start, v, w, label)
PROGRAM = [(0.0, 0.0, 0.0, "rest"), (1.0, 0.20, 0.0, "forward 0.20 m/s"), (5.0, 0.0, 0.0, "stop"),
           (6.0, -0.15, 0.0, "reverse -0.15 m/s"), (9.0, 0.0, 0.0, "stop"), (10.0, 0.0, 1.0, "turn in place +1.0 rad/s"),
           (14.0, 0.0, 0.0, "stop"), (15.0, 0.15, 0.5, "arc v=0.15, w=+0.5"), (19.0, 0.22, -2.0, "v=0.22, w=-2.0 (wheel-saturated)"),
           (21.0, 0.0, 0.0, "stop"), (23.0, None, None, "end")]
dt = TB4_PHYSICS_DT
sim, scene, robot, wid = make_scene(1, dt, args.device)
cm = CommandModel(1, dt, TB4_GEOMETRY, TB4_COMMAND_MODEL, device=args.device)
drv = Driver(sim, scene, robot, wid, cm)
names = robot.body_names
every = int(round(1.0 / (args.fps * dt)))
rec = {"t": [], "pos": [], "quat": [], "cmd": [], "twist": [], "label": []}
k = 0
seg = 0
t_end = PROGRAM[-1][0]
origin = scene.env_origins[0].clone()
while k * dt < t_end:
    tk = k * dt
    while seg + 1 < len(PROGRAM) and tk >= PROGRAM[seg + 1][0]:
        seg += 1
    _, v, w, lab = PROGRAM[seg]
    cm.set_command(torch.tensor([v], device=args.device), torch.tensor([w], device=args.device))
    drv.physics_step()
    if k % every == 0:
        bv, bw = drv.twist()
        rec["t"].append(tk); rec["label"].append(lab)
        rec["pos"].append((robot.data.body_pos_w[0] - origin).cpu().numpy())
        rec["quat"].append(robot.data.body_quat_w[0].cpu().numpy())
        rec["cmd"].append([v, w]); rec["twist"].append([bv.item(), bw.item()])
    k += 1
os.makedirs(args.out, exist_ok=True)
np.savez_compressed(os.path.join(args.out, "tb4_motion.npz"), t=np.array(rec["t"]), pos=np.array(rec["pos"]),
                    quat=np.array(rec["quat"]), cmd=np.array(rec["cmd"]), twist=np.array(rec["twist"]),
                    label=np.array(rec["label"]), body_names=np.array(names))
print("[DONE] frames", len(rec["t"]), "bodies", names, flush=True)
os._exit(0)
