"""M2 benchmark: throughput, GPU memory and stability of the articulated TB4 + arena + 40-beam LiDAR.

One process = one (num_envs, physics_hz, solver iterations) configuration, three phases on the same scene:
  A  physics only          : wheel velocity targets (random, held per control step), physics + robot/mover state update
  B  physics + LiDAR       : A + one LiDAR read (GPU ray cast of 40 beams + post-processing) per 5 Hz control step
  C  full control loop     : B + calibrated command model every physics step, scripted moving obstacle (pose write per
                             control step), ground-truth clearance (analytic), 47-dim observation assembly
Robots receive random (v, w) commands, so they also collide with walls (contact stress).

    isaacrun -g auto -- python scripts/m2/m2_benchmark.py --headless --num_envs 4096 --physics_hz 200
"""
import argparse
import json
import math
import os
import subprocess
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--physics_hz", type=float, default=200.0)
p.add_argument("--pos_iters", type=int, default=16)
p.add_argument("--vel_iters", type=int, default=2)
p.add_argument("--control_steps", type=int, default=0, help="per phase (0 = auto)")
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "results", "m2", "benchmark_clean.jsonl"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
t_launch = time.time()
app = AppLauncher(args).app

import torch  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from st_nav.assets.turtlebot4 import TB4_CFG, TB4_COMMAND_MODEL, TB4_GEOMETRY  # noqa: E402
from st_nav.control.diff_drive import CommandModel  # noqa: E402
from st_nav.geometry.analytic2d import circle_clearance, wall_clearance, walls_from_specs  # noqa: E402
from st_nav.scenes.builder import SceneSpec  # noqa: E402
from st_nav.scenes.harness import make_nav_scene  # noqa: E402
from st_nav.scenes.reference import reference_scene_module  # noqa: E402
from st_nav.sensors.lidar2d import Lidar2DCfg  # noqa: E402

dev = args.device
N = args.num_envs
DT = 1.0 / args.physics_hz
DEC = int(round(0.2 / DT))
K = args.control_steps or (100 if N <= 2048 else 50 if N <= 8192 else 30)
STATIC = [(1.9, 1.6, 0.17), (-0.6, -0.9, 0.10), (2.2, -1.2, 0.05)]
ref_walls = list(reference_scene_module("corridor_static_object").WALLS)
spec = SceneSpec(walls=ref_walls, static_cylinders=STATIC, moving_cylinders=[0.17])
lcfg = Lidar2DCfg()
robot_cfg = TB4_CFG.copy()
robot_cfg.spawn.articulation_props = robot_cfg.spawn.articulation_props.replace(
    solver_position_iteration_count=args.pos_iters, solver_velocity_iteration_count=args.vel_iters)
t0 = time.time()
sim, scene, robot, wid, lidar, (mover,) = make_nav_scene(N, DT, dev, spec, lcfg, robot_cfg=robot_cfg)
t_setup = time.time() - t0
origins = scene.env_origins
walls = walls_from_specs(ref_walls, device=dev)
static_c = torch.tensor(STATIC, device=dev)
cm = CommandModel(N, DT, TB4_GEOMETRY, TB4_COMMAND_MODEL, device=dev)
ar = torch.arange(N, device=dev)

# start poses in free space
lim = 2.78
cand = (torch.rand(N, 64, 2, device=dev) * 2 - 1) * lim
pts = cand.reshape(-1, 2)
ok = ((wall_clearance(pts, walls) > 0.3) & (circle_clearance(pts, static_c[None].expand(pts.shape[0], -1, -1)) > 0.3)).reshape(N, 64)
xy = cand[ar, torch.argmax(ok.int(), 1)]
yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
pos = torch.cat([origins[:, :2] + xy, torch.zeros(N, 1, device=dev)], 1)
quat = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], 1)
robot.write_root_pose_to_sim(torch.cat([pos, quat], 1))
mc = torch.tensor([[0.9, 0.0]], device=dev).expand(N, -1)


def mover_pose(t):
    th = 0.3 * t
    xy_m = mc + 0.5 * torch.stack([torch.cos(th + ar * 0.1), torch.sin(th + ar * 0.1)], 1)
    p_ = torch.cat([origins[:, :2] + xy_m, torch.full((N, 1), 0.251, device=dev)], 1)
    return torch.cat([p_, torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(N, -1)], 1)


unstable = torch.zeros(N, dtype=torch.bool, device=dev)
nan_seen = False


def check_stability():
    global nan_seen
    rs = robot.data.root_state_w
    if not torch.isfinite(rs).all():
        nan_seen = True
    r_, p_, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    tilt = torch.maximum(torch.sin(r_).abs(), torch.sin(p_).abs())
    z = robot.data.root_pos_w[:, 2]
    unstable.__ior__((tilt > math.sin(math.radians(30))) | (z < -0.05) | (z > 0.15) | ~torch.isfinite(rs).all(1))


def gpu_mem_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            pid, mem = (x.strip() for x in line.split(","))
            if int(pid) == os.getpid():
                return int(mem)
    except Exception:
        pass
    return None


def run_phase(mode, k_steps):
    wl = torch.zeros(N, 2, device=dev)
    sim_t = 0.0
    torch.cuda.synchronize()
    t_start = time.time()
    obs = None
    for k in range(k_steps):
        v = (torch.rand(N, device=dev) * 2 - 1) * 0.22
        w = (torch.rand(N, device=dev) * 2 - 1) * 2.0
        if mode == "C":
            cm.set_command(v, w)
            mover.write_root_pose_to_sim(mover_pose(sim_t))
        else:
            half = 0.5 * TB4_GEOMETRY.track_width
            wl = torch.stack([(v - w * half), (v + w * half)], -1) / TB4_GEOMETRY.wheel_radius
        for _ in range(DEC):
            if mode == "C":
                a, b, _, _ = cm.step()
                robot.set_joint_velocity_target(torch.stack([a, b], -1), joint_ids=wid)
            else:
                robot.set_joint_velocity_target(wl, joint_ids=wid)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(DT)
            sim_t += DT
        if mode in ("B", "C"):
            out = lidar.read(sim_t)
        if mode == "C":
            p2 = robot.data.root_pos_w[:, :2] - origins[:, :2]
            m2 = mover.data.root_pos_w[:, :2] - origins[:, :2]
            circ = torch.cat([static_c[None].expand(N, -1, -1), torch.cat([m2, torch.full((N, 1), 0.17, device=dev)], 1)[:, None]], 1)
            clear = torch.minimum(wall_clearance(p2, walls), circle_clearance(p2, circ)) - 0.177
            goal = torch.tensor([[-1.8, -2.5]], device=dev) - p2
            _, _, yw = euler_xyz_from_quat(robot.data.root_quat_w)
            th = torch.atan2(goal[:, 1], goal[:, 0]) - yw
            obs = torch.cat([out.normalized, (goal.norm(dim=1, keepdim=True) / 5.94).clamp(0, 1), torch.sin(th)[:, None],
                             torch.cos(th)[:, None], torch.stack([v / 0.22, w / 2.0], 1),
                             robot.data.root_lin_vel_b[:, :1] / 0.22, robot.data.root_ang_vel_b[:, 2:3] / 2.0], 1)
            _ = clear.min()
        check_stability()
    torch.cuda.synchronize()
    el = time.time() - t_start
    return {"control_steps_per_s": k_steps * N / el, "physics_env_steps_per_s": k_steps * DEC * N / el,
            "wall_s": el, "control_steps": k_steps, "obs_dim": None if obs is None else int(obs.shape[1])}


res = {"num_envs": N, "physics_hz": args.physics_hz, "decimation": DEC, "solver_iters": [args.pos_iters, args.vel_iters],
       "setup_s": t_setup, "launch_to_ready_s": time.time() - t_launch}
torch.cuda.reset_peak_memory_stats()
run_phase("A", 5)                                       # warm-up
for mode in ("A", "B", "C"):
    res[mode] = run_phase(mode, K)
    res[mode]["gpu_mem_process_mib"] = gpu_mem_mib()
res["torch_peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
res["unstable_envs"] = int(unstable.sum())
res["nan_seen"] = nan_seen
res["gpu_name"] = torch.cuda.get_device_name()
print("[BENCH]", json.dumps(res), flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "a") as f:
    f.write(json.dumps(res) + "\n")
os._exit(0)
