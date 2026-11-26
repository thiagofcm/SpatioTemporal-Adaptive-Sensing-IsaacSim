"""Navigation with the 3D LiDAR: evaluation of a CNN policy (train_3d.py) under the same protocols as the 2D policies.
Derived from evaluate.py (2D, unchanged): identical protocols, episode seeding, recording format and summaries; it differs
only in the task (STNav-TB4-Corridor-Lidar3D-v0: range-image observations), the agent config (CNN, with the training
run's agent overrides) and the output root. Keep the two in sync when protocols change.

    isaacrun -g 5 -- python Navigation/scripts/evaluate_3d.py --headless \
        --checkpoint Navigation/runs_3d/<ID>/<run>/model_<N>.pt [--protocol ref_static_c10] [--episodes 100]

Output (default): Navigation/eval_3d/<ID>/<protocol>/<run>_<checkpoint>/ (same files as evaluate.py; lidar.npz in the
per-scan layout, tools/episode_data.py). The visualization tools (tools/make_video.py, plot_episode.py, plot_eval.py)
read it unchanged.
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
# ref_static_c10: static scene, Z1-Z4 of reference seed obstacle_seed + i. ref_chase_c10: dynamic scenes (moving OB1 / OB2;
# the whole obstacle realization and the LiDAR noise of episode i are fixed by (--seed, i), see the docstring).
PROTOCOLS = {"ref_static_c10": dict(episodes=100, obstacle_seed=3000, start_seed=4000, goal=(-2.0, -2.0), timeout_s=70.0),
             "ref_chase_c10": dict(episodes=100, obstacle_seed=None, start_seed=4000, goal=(-2.0, -2.0), timeout_s=70.0)}

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--protocol", default=None, choices=sorted(PROTOCOLS),
               help="default: ref_static_c10 for the static scene, ref_chase_c10 for dynamic scenes")
p.add_argument("--episodes", type=int, default=None)            # the remaining options override the protocol
p.add_argument("--obstacle_seed", type=int, default=None)       # reference STATIC_OBSTACLES_SEED
p.add_argument("--start_seed", type=int, default=None)          # reference START_POSE_SEED
p.add_argument("--goal", type=float, nargs=2, default=None)
p.add_argument("--timeout_s", type=float, default=None)
p.add_argument("--seed", type=int, default=42)                  # reference RUN_SEED (LiDAR noise generator)
p.add_argument("--out_dir", default=None)
p.add_argument("--overwrite", action="store_true")
p.add_argument("--fast_path", default="none", help="env performance fast path: none | all | comma-separated items (st_nav.envs.corridor_scene.FAST_PATH_ITEMS); same results, faster")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
run_cfg_path = os.path.join(run_dir, "params", "config.yaml")
run_exp = X.yaml.safe_load(open(run_cfg_path)) if os.path.exists(run_cfg_path) else None
run_env = {**X.flatten((run_exp or {}).get("env") or {}), **(((run_exp or {}).get("set") or {}).get("env") or {})}
dynamic = run_env.get("scene_name", "corridor_static_random") != "corridor_static_random"
if args.protocol is None:
    args.protocol = "ref_chase_c10" if dynamic else "ref_static_c10"
if dynamic != (args.protocol == "ref_chase_c10"):
    raise SystemExit(f"protocol {args.protocol} does not match the run's scene {run_env.get('scene_name', 'corridor_static_random')}")
custom = [k for k in PROTOCOLS[args.protocol] if getattr(args, k) is not None]
for k, v in PROTOCOLS[args.protocol].items():
    setattr(args, k, v if getattr(args, k) is None else getattr(args, k))
exp_id = X.exp_id_from_run_path_3d(args.checkpoint)
if args.out_dir is None:
    if exp_id is None:
        raise SystemExit("checkpoint is not inside Navigation/runs_3d/<ID>/<run>/: pass --out_dir")
    scenario = args.protocol + ("_custom-" + "-".join(custom) if custom else "")
    policy_name = os.path.basename(run_dir) + "_" + os.path.basename(args.checkpoint).replace(".pt", "")
    args.out_dir = str(X.paths_3d(exp_id)["eval"] / scenario / policy_name)
if os.path.exists(os.path.join(args.out_dir, "episodes.csv")) and not args.overwrite:
    raise SystemExit(f"{args.out_dir} already has results; pass --overwrite (and note why in the experiment card)")
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
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cnn_cfg import CorridorNavCNNPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import OUTCOME_NAMES  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_lidar3d_cfg import CorridorNavLidar3DEnvCfg  # noqa: E402
from Navigation.evaluation.recorder import EpisodeRecorder  # noqa: E402
from tools import episode_data as ED  # noqa: E402


def reference_start_pose(seed):
    """= adaptive_sensor_policy envs/turtlebot_var_scan_rate.sample_start_pose (verified identical for seeds 4000-4099)."""
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(S.START_POSE_LIST)))
    dyaw = float(rng.uniform(-S.START_YAW_NOISE, S.START_YAW_NOISE))
    return i, (*S.START_POSE_LIST[i], (S.START_YAW_CENTER + dyaw + math.pi) % (2 * math.pi) - math.pi)


n = args.episodes
starts = [reference_start_pose(args.start_seed + i) for i in range(n)]
cfg = CorridorNavLidar3DEnvCfg()
if run_exp:                                                      # the training run's env settings
    X.apply_overrides(cfg, run_env, "env")
cfg.scene.num_envs = n
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.episode_length_s = args.timeout_s
if not dynamic:                                                  # Z1-Z4 of reference seed obstacle_seed + i
    cfg.bank_seed, cfg.bank_size = args.obstacle_seed, n
    cfg.bank_assignment, cfg.offsets_from_bank = "sequential", True
cfg.start_poses = tuple(s for _, s in starts)
cfg.goals = (tuple(args.goal),)
cfg.fast_path = () if args.fast_path == "none" else tuple(x.strip() for x in args.fast_path.split(","))
if cfg.sensor != "lidar3d" or cfg.obs_layout != "image":
    raise SystemExit("evaluate_3d.py expects a 3D-LiDAR run (env sensor lidar3d, obs_layout image); use evaluate.py for 2D")
genv = gym.make("STNav-TB4-Corridor-Lidar3D-v0", cfg=cfg)
agent = CorridorNavCNNPPORunnerCfg()
run_agent = {**X.flatten((run_exp or {}).get("agent") or {}), **(((run_exp or {}).get("set") or {}).get("agent") or {})}
X.apply_overrides(agent, {k: v for k, v in run_agent.items() if k not in ("seed", "max_iterations")}, "agent")
agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
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

rec = EpisodeRecorder(u)                                         # per-episode recording (episode_NNNN/)
obs = env.get_observations()
rec.record_reset(obs)
p0, y0 = corr.robot_pose2d()
chk_start = float(torch.max((p0 - corr.start[:, :2]).norm(dim=1)))
chk_layout = None if dynamic else float((corr.z_off - torch.tensor(corr.bank["offsets"], device=dev, dtype=torch.float32)).abs().max())
om = corr.obs_model                                              # episode obstacle settings (first episode of every env)
ob_ep = None if om is None else {"active": om.active.cpu().numpy().copy(), "speed": om.speed.cpu().numpy().copy(),
                                 "path_y": om.path_y.cpu().numpy().copy()}
chk_goal = float((corr.goal - torch.tensor(args.goal, device=dev)).abs().max())
done_once = torch.zeros(n, dtype=torch.bool, device=dev)
ret = torch.zeros(n, device=dev)
path = torch.zeros(n, device=dev)
minc = torch.full((n,), float("inf"), device=dev)
res = {k: torch.zeros(n, device=dev) for k in ("outcome", "steps", "final_dist", "min_clear", "ret", "path")}
prev = p0.clone()
t_loop = time.time()
for k in range(int(u.max_episode_length) + 2):
    act = policy(obs)
    obs, rew, dones, ext = env.step(act)
    rec.record_step(act, rew, dones, ext, obs)
    term = rec.end_state                                         # state at the end of the step, before any auto-reset
    alive = ~done_once
    d = dones.bool()
    pos_end = torch.stack([term["x"], term["y"]], 1)
    path += torch.where(alive, (pos_end - prev).norm(dim=1), torch.zeros_like(path))
    ret += torch.where(alive, rew, torch.zeros_like(rew))
    minc = torch.where(alive, torch.minimum(minc, term["clear"]), minc)
    new = d & alive
    if new.any():
        res["outcome"][new] = ext["outcome"][new].float()
        res["steps"][new] = float(k + 1)
        res["final_dist"][new] = term["goal_d"][new]
        res["min_clear"][new], res["ret"][new], res["path"][new] = minc[new], ret[new], path[new]
        done_once |= new
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
               "repo": X.git_state(X.REPO_ROOT), "recording_schema": ED.SCHEMA_VERSION,
               "env_cfg_resolved": cfg.to_dict()}, f, indent=1, default=str)
rows = []
for i in range(n):
    oc = OUTCOME_NAMES.get(int(R["outcome"][i]), "running")
    rows.append({"episode": i, "static_obstacles_seed": None if dynamic else args.obstacle_seed + i, "start_pose_seed": args.start_seed + i,
                 "start_pose_index": starts[i][0], "start_x": starts[i][1][0], "start_y": starts[i][1][1],
                 "start_theta": round(starts[i][1][2], 5), "goal_x": args.goal[0], "goal_y": args.goal[1],
                 **({} if dynamic else {
                     "z12_mode": corr.bank["pair_modes"][i][0], "z34_mode": corr.bank["pair_modes"][i][1],
                     **{f"Z{j + 1}_length": round(float(corr.bank["lengths"][i][j]), 4) for j in range(4)},
                     **{f"Z{j + 1}_thickness": round(float(corr.bank["thicknesses"][i][j]), 4) for j in range(4)},
                     **{n_: round(float(corr.bank["offsets"][i][j]), 4) for j, n_ in enumerate(("Z1_x_offset", "Z2_x_offset", "Z3_y_offset", "Z4_y_offset"))}}),
                 **({} if ob_ep is None else {
                     **{f"ob{j + 1}_active": int(ob_ep["active"][i][j]) for j in range(ob_ep["active"].shape[1])},
                     **{f"ob{j + 1}_speed_mps": round(float(ob_ep["speed"][i][j]), 4) for j in range(ob_ep["speed"].shape[1])},
                     **{f"ob{j + 1}_path_y_m": round(float(ob_ep["path_y"][i][j]), 4) for j in range(ob_ep["path_y"].shape[1])}}),
                 "success": oc == "goal", "outcome": oc, "n_steps": int(R["steps"][i]), "duration_s": round(R["steps"][i] * u.step_dt, 2),
                 "episode_return": round(float(R["ret"][i]), 3), "final_goal_distance_m": round(float(R["final_dist"][i]), 4),
                 "distance_traveled_m": round(float(R["path"][i]), 3), "min_clearance_m": round(float(R["min_clear"][i]), 4)})
with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
policy_name = os.path.basename(os.path.normpath(out_dir))


def episode_meta(e, row):
    m = {"episode_index": e, "exp_id": exp_id, "protocol": args.protocol, "policy": policy_name,
         "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), X.REPO_ROOT),
         "seeds": {"start_pose_seed": row["start_pose_seed"], "static_obstacles_seed": row["static_obstacles_seed"],
                   "env_seed": args.seed},
         "start": {"x": row["start_x"], "y": row["start_y"], "yaw": row["start_theta"], "index": row["start_pose_index"]},
         "goal": {"x": row["goal_x"], "y": row["goal_y"]}, "deterministic_policy": True}
    m.update({k: row[k] for k in ("outcome", "success", "n_steps", "duration_s", "episode_return", "final_goal_distance_m",
                                  "distance_traveled_m", "min_clearance_m")})
    if not dynamic:
        m["static_obstacles"] = {"pair_modes": list(corr.bank["pair_modes"][e]), "lengths": corr.bank["lengths"][e].tolist(),
                                 "thicknesses": corr.bank["thicknesses"][e].tolist(), "offsets": corr.bank["offsets"][e].tolist()}
    if ob_ep is not None:
        m["moving_obstacles"] = {"ids": [f"OB{j + 1}" for j in range(ob_ep["active"].shape[1])],
                                 "active": ob_ep["active"][e].tolist(), "speed_mps": ob_ep["speed"][e].tolist(),
                                 "path_y_m": ob_ep["path_y"][e].tolist()}
    return m


static_meta = rec.static_metadata(cfg)
rec.export(out_dir, R["steps"], rows, static_meta, episode_meta)
ED.write_json(os.path.join(out_dir, "world_geometry.json"), rec.fixed_geometry())
succ = np.array([r["success"] for r in rows])
oc_counts = {name: int(sum(r["outcome"] == name for r in rows)) for name in list(OUTCOME_NAMES.values()) + ["running"]}


def ms(x):
    return [float(np.mean(x)), float(np.std(x))] if len(x) else [float("nan"), float("nan")]


S_ = {"checkpoint": args.checkpoint, "episodes": n, "goal": list(args.goal), "timeout_s": args.timeout_s,
      "obstacle_seeds": None if dynamic else [args.obstacle_seed, args.obstacle_seed + n - 1], "start_seeds": [args.start_seed, args.start_seed + n - 1],
      "success_rate": float(succ.mean()), "outcomes": oc_counts,
      "return_mean_std": ms([r["episode_return"] for r in rows]),
      "success_duration_s_mean_std": ms([r["duration_s"] for r in rows if r["success"]]),
      "success_path_m_mean_std": ms([r["distance_traveled_m"] for r in rows if r["success"]]),
      "success_mean_speed_mps": float(np.mean([r["distance_traveled_m"] / r["duration_s"] for r in rows if r["success"]])) if succ.any() else None,
      "min_clearance_m_mean_min": [float(np.mean([r["min_clearance_m"] for r in rows])), float(np.min([r["min_clearance_m"] for r in rows]))],
      "success_by_start_index": {str(i): [int(sum(r["success"] for r in rows if r["start_pose_index"] == i)),
                                          int(sum(r["start_pose_index"] == i for r in rows))] for i in range(5)},
      **({} if dynamic else {"success_by_pair_modes": {
          m: [int(sum(r["success"] for r in rows if f'{r["z12_mode"]}/{r["z34_mode"]}' == m)),
              int(sum(f'{r["z12_mode"]}/{r["z34_mode"]}' == m for r in rows))]
          for m in sorted({f'{r["z12_mode"]}/{r["z34_mode"]}' for r in rows})}}),
      **({} if ob_ep is None else {"success_by_active_obstacles": {
          a: [int(sum(r["success"] for r in rows if "".join(str(r[k]) for k in sorted(r) if k.endswith("_active")) == a)),
              int(sum("".join(str(r[k]) for k in sorted(r) if k.endswith("_active")) == a for r in rows))]
          for a in sorted({"".join(str(r[k]) for k in sorted(r) if k.endswith("_active")) for r in rows})}}),
      "checks": {"max_start_placement_err_m": chk_start, "max_offset_vs_reference_err": chk_layout, "max_goal_err": chk_goal},
      "timing_s": {"launch_and_setup": round(t_setup, 1), "episodes": round(t_eval, 1), "total": round(time.time() - t_start, 1)}}
json.dump(S_, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
ED.update_protocol_summary(os.path.dirname(os.path.normpath(out_dir)), {
    "policy": policy_name, "checkpoint": S_["checkpoint"], "episodes": n, "success_rate": S_["success_rate"],
    **{f"n_{k}": v for k, v in oc_counts.items() if k != "running"}, "return_mean": S_["return_mean_std"][0],
    "success_duration_s_mean": S_["success_duration_s_mean_std"][0], "min_clearance_m_mean": S_["min_clearance_m_mean_min"][0],
    "min_clearance_m_min": S_["min_clearance_m_mean_min"][1]})
print("[EVAL]", json.dumps(S_), flush=True)
os._exit(0)
