"""M3 limited benchmark: complete corridor_static_object_random environment at the adopted configuration
(100 Hz physics, 16/2 solver iterations, decimation 20, 5 Hz control and LiDAR), including automatic resets.

Per control step: random (v, w) -> calibrated command model -> 20 physics steps -> LiDAR (40 beams, noise on) ->
ground-truth clearance and collision flags -> goal distance -> partial reset of envs with a safety violation,
goal reached (< 0.20 m) or timeout (350 steps = 70 s, reference episode length).

    isaacrun -g auto -- python scripts/m3/m3_benchmark.py --headless --num_envs 16384
"""
import argparse
import json
import math
import os
import subprocess
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--control_steps", type=int, default=150)
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "results", "m3", "benchmark.jsonl"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
t_launch = time.time()
app = AppLauncher(args).app

import torch  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from st_nav.envs.corridor_scene import CorridorScene, CorridorSceneCfg  # noqa: E402

dev = args.device
t0 = time.time()
sc = CorridorScene(CorridorSceneCfg(num_envs=args.num_envs), dev)
t_setup = time.time() - t0
N = sc.n
steps = torch.zeros(N, dtype=torch.long, device=dev)
counts = {"safety": 0, "physical": 0, "goal": 0, "timeout": 0, "contact_steps": 0}
unstable = torch.zeros(N, dtype=torch.bool, device=dev)
nan_seen = False


def gpu_mem():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        pid, mem = (x.strip() for x in line.split(","))
        if int(pid) == os.getpid():
            return int(mem)
    return None


def control_step():
    global nan_seen
    v = (torch.rand(N, device=dev) * 2 - 1) * 0.22
    w = (torch.rand(N, device=dev) * 2 - 1) * 2.0
    sc.step(v, w)
    out = sc.read_lidar()
    f = sc.collision_flags()
    d = sc.goal_distance()
    steps.add_(1)
    goal = d < 0.20
    timeout = steps >= 350
    done = f["safety_violation"] | goal | timeout
    counts["safety"] += int(f["safety_violation"].sum())
    counts["physical"] += int(f["physical_collision"].sum())
    counts["goal"] += int(goal.sum())
    counts["timeout"] += int((timeout & ~f["safety_violation"] & ~goal).sum())
    counts["contact_steps"] += int(f["contact"].sum())
    rs = sc.robot.data.root_state_w
    if not torch.isfinite(rs).all() or not torch.isfinite(out.ranges).all():
        nan_seen = True
    r_, p_, _ = euler_xyz_from_quat(sc.robot.data.root_quat_w)
    unstable.__ior__((torch.sin(r_).abs() > 0.5) | (torch.sin(p_).abs() > 0.5) | (rs[:, 2] > 0.15) | (rs[:, 2] < -0.05))
    ids = done.nonzero(as_tuple=False).squeeze(-1)
    if len(ids):
        sc.reset(ids)
        steps[ids] = 0
    return len(ids)


for _ in range(10):
    control_step()
for k in counts:
    counts[k] = 0
torch.cuda.synchronize()
t1 = time.time()
resets = 0
for _ in range(args.control_steps):
    resets += control_step()
torch.cuda.synchronize()
el = time.time() - t1
res = {"num_envs": N, "physics_hz": sc.cfg.physics_hz, "decimation": sc.cfg.decimation, "solver_iters": [16, 2],
       "bank_size": len(sc.bank["seeds"]), "setup_s": t_setup, "launch_to_ready_s": time.time() - t_launch - el,
       "control_steps": args.control_steps, "control_steps_per_s": args.control_steps * N / el,
       "physics_env_steps_per_s": args.control_steps * N * sc.cfg.decimation / el, "wall_s": el,
       "resets": resets, "resets_per_control_step": resets / args.control_steps, "terminations": counts,
       "offset_sampler_fallbacks": sc.offset_fallbacks, "unstable_envs": int(unstable.sum()), "nan_seen": nan_seen,
       "gpu_mem_process_mib": gpu_mem(), "torch_peak_alloc_mib": torch.cuda.max_memory_allocated() / 2 ** 20}
print("[BENCH]", json.dumps(res), flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "a") as f:
    f.write(json.dumps(res) + "\n")
os._exit(0)
