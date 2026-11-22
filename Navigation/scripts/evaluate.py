"""Navigation evaluation. Protocol ref_static_c10 = the reference protocol (adaptive_sensor_policy/AdaptiveFPS/evaluation):

* episode i (i = 0..n-1) uses the Z1-Z4 configuration of reference seed STATIC_OBSTACLES_SEED + i (sizes AND offsets from
  the reference sampler; seeds 3000.. are outside the training bank 0..1023, i.e. held-out obstacle sizes),
* start pose from the reference sample_start_pose(START_POSE_SEED + i) (5 points, yaw pi +- 10 deg),
* fixed goal EVAL_GOAL = (-2.0, -2.0) (reference "C10"; not in the 9-goal training list, inside the goal region),
* timeout 80 s = 400 steps (reference EVAL_TIMEOUT_S); deterministic policy (actor mean); LiDAR noise on.
All episodes run in parallel (env i = episode i); only each env's first episode is counted.

    isaacrun -g 5 -- python Navigation/scripts/evaluate.py --headless \
        --checkpoint Navigation/runs/<ID>/<run>/model_200.pt [--protocol ref_static_c10]

Output (default): Navigation/eval/<ID>/<protocol>/<run>_<checkpoint>/ (ID taken from the checkpoint path):
episodes.csv, summary.json, run_config.json, trajectories.npy, recording.npz. Existing folders need --overwrite.
The training run's env overrides (params/config.yaml, if present) are applied before the protocol's settings.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools import experiments as X  # noqa: E402

# named evaluation protocols: every policy evaluated under one protocol sees identical episodes
PROTOCOLS = {"ref_static_c10": dict(episodes=100, obstacle_seed=3000, start_seed=4000, goal=(-2.0, -2.0), timeout_s=80.0)}

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--protocol", default="ref_static_c10", choices=sorted(PROTOCOLS))
p.add_argument("--episodes", type=int, default=None)            # the remaining options override the protocol
p.add_argument("--obstacle_seed", type=int, default=None)       # reference STATIC_OBSTACLES_SEED
p.add_argument("--start_seed", type=int, default=None)          # reference START_POSE_SEED
p.add_argument("--goal", type=float, nargs=2, default=None)
p.add_argument("--timeout_s", type=float, default=None)
p.add_argument("--seed", type=int, default=42)                  # reference RUN_SEED (LiDAR noise generator)
p.add_argument("--out_dir", default=None)
p.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
custom = [k for k in PROTOCOLS[args.protocol] if getattr(args, k) is not None]
for k, v in PROTOCOLS[args.protocol].items():
    setattr(args, k, v if getattr(args, k) is None else getattr(args, k))
run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
exp_id = X.exp_id_from_run_path(args.checkpoint)
if args.out_dir is None:
    if exp_id is None:
        raise SystemExit("checkpoint is not inside <line>/runs/<ID>/<run>/: pass --out_dir")
    scenario = args.protocol + ("_custom-" + "-".join(custom) if custom else "")
    policy_name = os.path.basename(run_dir) + "_" + os.path.basename(args.checkpoint).replace(".pt", "")
    args.out_dir = str(X.paths(exp_id)["eval"] / scenario / policy_name)
if os.path.exists(os.path.join(args.out_dir, "episodes.csv")) and not args.overwrite:
    raise SystemExit(f"{args.out_dir} already has results; pass --overwrite (and note why in the experiment card)")
run_cfg_path = os.path.join(run_dir, "params", "config.yaml")
run_exp = X.yaml.safe_load(open(run_cfg_path)) if os.path.exists(run_cfg_path) else None
t_start = time.time()
app = AppLauncher(args).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from st_nav.scenes import corridor_static_random as S  # noqa: E402
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cfg import CorridorNavPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import OUTCOME_NAMES  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402


def reference_start_pose(seed):
    """= adaptive_sensor_policy envs/turtlebot_var_scan_rate.sample_start_pose (verified identical for seeds 4000-4099)."""
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(S.START_POSE_LIST)))
    dyaw = float(rng.uniform(-S.START_YAW_NOISE, S.START_YAW_NOISE))
    return i, (*S.START_POSE_LIST[i], (S.START_YAW_CENTER + dyaw + math.pi) % (2 * math.pi) - math.pi)


n = args.episodes
starts = [reference_start_pose(args.start_seed + i) for i in range(n)]
cfg = CorridorNavEnvCfg()
if run_exp:                                                      # the training run's env settings
    X.apply_overrides(cfg, X.flatten(run_exp.get("env") or {}), "env")
    X.apply_overrides(cfg, (run_exp.get("set") or {}).get("env") or {}, "env")
cfg.scene.num_envs = n
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.episode_length_s = args.timeout_s
cfg.bank_seed, cfg.bank_size = args.obstacle_seed, n
cfg.bank_assignment, cfg.offsets_from_bank = "sequential", True
cfg.start_poses = tuple(s for _, s in starts)
cfg.goals = (tuple(args.goal),)
genv = gym.make("STNav-TB4-Corridor-Static-v0", cfg=cfg)
agent = handle_deprecated_rsl_rl_cfg(CorridorNavPPORunnerCfg(), version("rsl-rl-lib"))
agent.device = args.device
env = RslRlVecEnvWrapper(genv, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=args.device)
t_setup = time.time() - t_start
u = genv.unwrapped
corr, dev = u.corr, u.device
_IM = torch.inference_mode()
_IM.__enter__()

# capture terminal state before the auto-reset
term = {}
_orig = u._get_dones


def _hook():
    out = _orig()
    term["pose"] = torch.cat([corr.robot_pose2d()[0], corr.robot_pose2d()[1][:, None]], 1).clone()
    term["dist"], term["clear"] = u._goal_dist.clone(), u._clear.clone()
    return out


u._get_dones = _hook
obs = env.get_observations()
p0, y0 = corr.robot_pose2d()
chk_start = float(torch.max((p0 - corr.start[:, :2]).norm(dim=1)))
chk_layout = float((corr.z_off - torch.tensor(corr.bank["offsets"], device=dev, dtype=torch.float32)).abs().max())
chk_goal = float((corr.goal - torch.tensor(args.goal, device=dev)).abs().max())
done_once = torch.zeros(n, dtype=torch.bool, device=dev)
ret = torch.zeros(n, device=dev)
path = torch.zeros(n, device=dev)
minc = torch.full((n,), float("inf"), device=dev)
res = {k: torch.zeros(n, device=dev) for k in ("outcome", "steps", "final_dist", "min_clear", "ret", "path")}
traj = [p0.cpu().numpy().copy()]
# per-step record for animations: pose (x, y, yaw) and the observed LiDAR ranges [m] (NaN after the first episode)
rec_pose = [torch.cat([p0, y0[:, None]], 1).cpu().numpy()]
rec_scan = [(obs["policy"][:, :40] * corr.lidar.range_max).cpu().numpy()]
prev = p0.clone()
t_loop = time.time()
for k in range(int(u.max_episode_length) + 2):
    obs, rew, dones, ext = env.step(policy(obs))
    alive = ~done_once
    d = dones.bool()
    pos_end = torch.where(d[:, None], term["pose"][:, :2], corr.robot_pose2d()[0])
    path += torch.where(alive, (pos_end - prev).norm(dim=1), torch.zeros_like(path))
    ret += torch.where(alive, rew, torch.zeros_like(rew))
    minc = torch.where(alive, torch.minimum(minc, term["clear"]), minc)
    new = d & alive
    if new.any():
        res["outcome"][new] = ext["outcome"][new].float()
        res["steps"][new] = float(k + 1)
        res["final_dist"][new] = term["dist"][new]
        res["min_clear"][new], res["ret"][new], res["path"][new] = minc[new], ret[new], path[new]
        done_once |= new
    traj.append(torch.where(alive[:, None], pos_end, torch.full_like(pos_end, float("nan"))).cpu().numpy())
    pose_end = torch.where(d[:, None], term["pose"], torch.cat([corr.robot_pose2d()[0], corr.robot_pose2d()[1][:, None]], 1))
    rec_pose.append(torch.where(alive[:, None], pose_end, torch.full_like(pose_end, float("nan"))).cpu().numpy())
    sc = (obs["policy"][:, :40] * corr.lidar.range_max)          # after an auto-reset this is the new episode's scan
    rec_scan.append(torch.where((alive & ~d)[:, None], sc, torch.full_like(sc, float("nan"))).cpu().numpy())
    prev = corr.robot_pose2d()[0].clone()
    if bool(done_once.all()):
        break
t_eval = time.time() - t_loop

R = {k: v.cpu().numpy() for k, v in res.items()}
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "run_config.json"), "w") as f:
    json.dump({"exp_id": exp_id, "protocol": args.protocol, "custom": custom,
               "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), X.REPO_ROOT),
               "settings": {k: getattr(args, k) for k in ("episodes", "obstacle_seed", "start_seed", "goal", "timeout_s", "seed")},
               "training_run_config": run_cfg_path if run_exp else None, "command": sys.argv,
               "repo": X.git_state(X.REPO_ROOT), "reference_repo": X.git_state(X.REFERENCE_ROOT)}, f, indent=1)
rows = []
for i in range(n):
    oc = OUTCOME_NAMES.get(int(R["outcome"][i]), "running")
    rows.append({"episode": i, "static_obstacles_seed": args.obstacle_seed + i, "start_pose_seed": args.start_seed + i,
                 "start_pose_index": starts[i][0], "start_x": starts[i][1][0], "start_y": starts[i][1][1],
                 "start_theta": round(starts[i][1][2], 5), "goal_x": args.goal[0], "goal_y": args.goal[1],
                 "z12_mode": corr.bank["pair_modes"][i][0], "z34_mode": corr.bank["pair_modes"][i][1],
                 **{f"Z{j + 1}_length": round(float(corr.bank["lengths"][i][j]), 4) for j in range(4)},
                 **{f"Z{j + 1}_thickness": round(float(corr.bank["thicknesses"][i][j]), 4) for j in range(4)},
                 **{n_: round(float(corr.bank["offsets"][i][j]), 4) for j, n_ in enumerate(("Z1_x_offset", "Z2_x_offset", "Z3_y_offset", "Z4_y_offset"))},
                 "success": oc == "goal", "outcome": oc, "n_steps": int(R["steps"][i]), "duration_s": round(R["steps"][i] * u.step_dt, 2),
                 "episode_return": round(float(R["ret"][i]), 3), "final_goal_distance_m": round(float(R["final_dist"][i]), 4),
                 "distance_traveled_m": round(float(R["path"][i]), 3), "min_clearance_m": round(float(R["min_clear"][i]), 4)})
with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
np.save(os.path.join(out_dir, "trajectories.npy"), np.stack(traj))
np.savez_compressed(os.path.join(out_dir, "recording.npz"), pose=np.stack(rec_pose), scan=np.stack(rec_scan),
                    beam_angles=np.arange(40) * 2 * np.pi / 40,                         # beam 0 = forward, CCW, 9 deg
                    step_dt=u.step_dt)
succ = np.array([r["success"] for r in rows])
oc_counts = {name: int(sum(r["outcome"] == name for r in rows)) for name in list(OUTCOME_NAMES.values()) + ["running"]}


def ms(x):
    return [float(np.mean(x)), float(np.std(x))] if len(x) else [float("nan"), float("nan")]


S_ = {"checkpoint": args.checkpoint, "episodes": n, "goal": list(args.goal), "timeout_s": args.timeout_s,
      "obstacle_seeds": [args.obstacle_seed, args.obstacle_seed + n - 1], "start_seeds": [args.start_seed, args.start_seed + n - 1],
      "success_rate": float(succ.mean()), "outcomes": oc_counts,
      "return_mean_std": ms([r["episode_return"] for r in rows]),
      "success_duration_s_mean_std": ms([r["duration_s"] for r in rows if r["success"]]),
      "success_path_m_mean_std": ms([r["distance_traveled_m"] for r in rows if r["success"]]),
      "success_mean_speed_mps": float(np.mean([r["distance_traveled_m"] / r["duration_s"] for r in rows if r["success"]])) if succ.any() else None,
      "min_clearance_m_mean_min": [float(np.mean([r["min_clearance_m"] for r in rows])), float(np.min([r["min_clearance_m"] for r in rows]))],
      "success_by_start_index": {str(i): [int(sum(r["success"] for r in rows if r["start_pose_index"] == i)),
                                          int(sum(r["start_pose_index"] == i for r in rows))] for i in range(5)},
      "success_by_pair_modes": {m: [int(sum(r["success"] for r in rows if f'{r["z12_mode"]}/{r["z34_mode"]}' == m)),
                                    int(sum(f'{r["z12_mode"]}/{r["z34_mode"]}' == m for r in rows))]
                                for m in sorted({f'{r["z12_mode"]}/{r["z34_mode"]}' for r in rows})},
      "checks": {"max_start_placement_err_m": chk_start, "max_offset_vs_reference_err": chk_layout, "max_goal_err": chk_goal},
      "timing_s": {"launch_and_setup": round(t_setup, 1), "episodes": round(t_eval, 1), "total": round(time.time() - t_start, 1)}}
json.dump(S_, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
print("[EVAL]", json.dumps(S_), flush=True)
os._exit(0)
