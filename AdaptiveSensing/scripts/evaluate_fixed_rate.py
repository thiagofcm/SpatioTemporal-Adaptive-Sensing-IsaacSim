"""Fixed-rate sensing evaluation of a frozen 3D-LiDAR navigation policy (temporal sensing sensitivity, TMP01).

Same evaluation as Navigation/scripts/evaluate_3d.py (protocols, episode seeding, scenario, recording format, navigation
metrics), with the real-robot scan gate (real_eval_fixed.py, AdaptiveSensing/evaluation/scan_gate.py) between the
environment and the policy:

  * the env and its native LiDAR are unchanged: 4 Hz scans (lidar3d.period_s = 0.25, taken at the 5 Hz control
    instants and held in between), navigation policy at 5 Hz;
  * every genuinely NEW native scan (env scan_t changed) goes through the gate: it is forwarded into last_sampled_obs
    when its time has reached next_sample_time (then next_sample_time advances by whole periods), otherwise ignored;
    repeated native scans are never counted;
  * the policy receives {"policy": the env's current navigation features, "lidar": last_sampled_obs}.

    bin/isaacrun -g 0 -- python AdaptiveSensing/scripts/evaluate_fixed_rate.py --headless --fixed 2

Output: AdaptiveSensing/eval/temporal/<exp_id>/<protocol>/<run>_<checkpoint>_fixed<hz>Hz/ (the evaluate_3d.py files;
episodes.csv and the protocol summary.csv in the reference schema (AdaptiveSensing/evaluation/metrics.py; adaptive_return
with --frame_cost, default 0, and --budget); the sensing diagnostics (native / accepted scans, measured rate, intervals,
observation age) are in every episode's metadata.json (scan_gate) and in summary.json; steps.csv scan_fresh and
lidar.npz = FORWARDED scans).
"""
import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools import experiments as X  # noqa: E402

NAV05 = ("Navigation/runs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04/2026-11-23_20-21-14_ppo_cnn_n4096_s42/"
         "model_1499.pt")
NATIVE_PERIOD_S = 0.25                                           # native LiDAR: 4 Hz in every condition
# identical to evaluate_3d.py: every policy / sensing rate evaluated under one protocol sees identical episodes
PROTOCOLS = {"ref_static_c10": dict(episodes=100, obstacle_seed=3000, start_seed=4000, goal=(-2.0, -2.0), timeout_s=70.0),
             "ref_chase_c10": dict(episodes=100, obstacle_seed=None, start_seed=4000, goal=(-2.0, -2.0), timeout_s=70.0)}

p = argparse.ArgumentParser()
p.add_argument("--fixed", type=float, required=True, help="requested sensing rate (Hz) of the scan gate, e.g. 4, 2, 1")
p.add_argument("--checkpoint", default=NAV05)
p.add_argument("--exp_id", default="TMP01_chase-ob1ob2_fixed-rate-nav05")
p.add_argument("--protocol", default=None, choices=sorted(PROTOCOLS),
               help="default: ref_static_c10 for the static scene, ref_chase_c10 for dynamic scenes")
p.add_argument("--episodes", type=int, default=None)            # the remaining options override the protocol
p.add_argument("--obstacle_seed", type=int, default=None)
p.add_argument("--start_seed", type=int, default=None)
p.add_argument("--goal", type=float, nargs=2, default=None)
p.add_argument("--timeout_s", type=float, default=None)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--frame_cost", type=float, default=0.0, help="sensing cost per charged frame in adaptive_return")
p.add_argument("--budget", type=int, default=500, help="frame budget of adaptive_return (TMP02 value)")
p.add_argument("--budget_penalty", type=float, default=15.0)
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
hz_tag = f"fixed{args.fixed:g}Hz"
if args.out_dir is None:
    scenario = args.protocol + ("_custom-" + "-".join(custom) if custom else "")
    policy_name = os.path.basename(run_dir) + "_" + os.path.basename(args.checkpoint).replace(".pt", "") + "_" + hz_tag
    args.out_dir = str(X.paths(args.exp_id)["eval"] / scenario / policy_name)
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
from AdaptiveSensing.evaluation import metrics as M  # noqa: E402
from AdaptiveSensing.evaluation.scan_gate import FixedRateScanGate  # noqa: E402
from Navigation.evaluation.recorder import EpisodeRecorder  # noqa: E402
from tools import episode_data as ED  # noqa: E402


def reference_start_pose(seed):
    """= adaptive_sensor_policy envs/turtlebot_var_scan_rate.sample_start_pose (as evaluate_3d.py)."""
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
if not dynamic:
    cfg.bank_seed, cfg.bank_size = args.obstacle_seed, n
    cfg.bank_assignment, cfg.offsets_from_bank = "sequential", True
cfg.start_poses = tuple(s for _, s in starts)
cfg.goals = (tuple(args.goal),)
cfg.fast_path = () if args.fast_path == "none" else tuple(x.strip() for x in args.fast_path.split(","))
if cfg.sensor != "lidar3d" or cfg.obs_layout != "image":
    raise SystemExit("evaluate_fixed_rate.py expects a 3D-LiDAR run (env sensor lidar3d, obs_layout image)")
if abs(cfg.lidar3d.period_s - NATIVE_PERIOD_S) > 1e-9:
    raise SystemExit(f"native LiDAR period is {cfg.lidar3d.period_s} s; this study keeps it at {NATIVE_PERIOD_S} s (4 Hz)")
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
if abs(u.scan_period - NATIVE_PERIOD_S) > 1e-9 or abs(u.step_dt - 0.2) > 1e-9:
    raise SystemExit(f"unexpected timing: native period {u.scan_period} s, control step {u.step_dt} s")
_IM = torch.inference_mode()
_IM.__enter__()

gate = FixedRateScanGate(n, args.fixed, u.physics_dt, device=dev)
all_ids = torch.arange(n, device=dev)


def gated(o):
    """Policy input: the env's current navigation features + last_sampled_obs."""
    g = o.clone()
    g["lidar"] = gate.last_obs.clone()
    return g


rec = EpisodeRecorder(u)
obs = env.get_observations()
gate.reset(all_ids, obs["lidar"], u.scan_t)
rec.record_reset(gated(obs))
p0, y0 = corr.robot_pose2d()
chk_start = float(torch.max((p0 - corr.start[:, :2]).norm(dim=1)))
chk_layout = None if dynamic else float((corr.z_off - torch.tensor(corr.bank["offsets"], device=dev, dtype=torch.float32)).abs().max())
om = corr.obs_model
ob_ep = None if om is None else {"active": om.active.cpu().numpy().copy(), "speed": om.speed.cpu().numpy().copy(),
                                 "path_y": om.path_y.cpu().numpy().copy()}
chk_goal = float((corr.goal - torch.tensor(args.goal, device=dev)).abs().max())
done_once = torch.zeros(n, dtype=torch.bool, device=dev)
ret = torch.zeros(n, device=dev)
path = torch.zeros(n, device=dev)
minc = torch.full((n,), float("inf"), device=dev)
res = {k: torch.zeros(n, device=dev) for k in ("outcome", "steps", "final_dist", "min_clear", "ret", "path")}
# sensing: staleness of last_sampled_obs at every policy call (ticks), and the gate counters at the episode's end
i64 = dict(dtype=torch.int64, device=dev)
age_sum, age_max = torch.zeros(n, **i64), torch.zeros(n, **i64)
sens = {k: torch.zeros(n, **i64) for k in ("n_native", "n_forward", "last_forward", "int_min", "int_max", "age_sum", "age_max")}
chk = {"native_interval_violations": 0, "native_intervals_ticks": set(), "forward_ne_new_4hz": 0,
       "gated_ne_native_4hz": 0, "held_changed_without_forward": 0, "nav_features_not_current": 0,
       "forward_without_new_scan": 0}
prev = p0.clone()
t_loop = time.time()
for k in range(int(u.max_episode_length) + 2):
    g_obs = gated(obs)
    alive = ~done_once
    age = gate.ticks(u.ep_time) - gate.last_forward                     # age of last_sampled_obs at this policy call
    age_sum += torch.where(alive, age, torch.zeros_like(age))
    age_max = torch.where(alive, torch.maximum(age_max, age), age_max)
    if not torch.equal(g_obs["policy"], obs["policy"]):
        chk["nav_features_not_current"] += 1
    act = policy(g_obs)
    obs, rew, dones, ext = env.step(act)
    d = dones.bool()
    term = rec.end_state
    new_end = d & alive
    if new_end.any():                                                   # first episode ends: freeze its sensing record
        for key, v in (("n_native", gate.n_native), ("n_forward", gate.n_forward), ("last_forward", gate.last_forward),
                       ("int_min", gate.int_min), ("int_max", gate.int_max), ("age_sum", age_sum), ("age_max", age_max)):
            sens[key][new_end] = v[new_end]
    held_before, native_before = gate.last_obs.clone(), gate.last_native.clone()
    new, fwd = gate.step(obs["lidar"], u.scan_t)                       # done envs: overwritten by the reset below
    live = alive & ~d                                                   # first episode still running after this step
    gap = gate.last_native - native_before
    bad = live & new & (gap != 20) & (gap != 40)                        # native 4 Hz on the 5 Hz grid: 0.2 / 0.4 s
    chk["native_interval_violations"] += int(bad.sum())
    chk["native_intervals_ticks"] |= set(gap[live & new].unique().tolist())
    chk["forward_without_new_scan"] += int((live & fwd & ~new).sum())
    chk["held_changed_without_forward"] += int((live & ~fwd).sum()) - int(
        (gate.last_obs[live & ~fwd] == held_before[live & ~fwd]).flatten(1).all(1).sum())
    if args.fixed == 4.0:
        chk["forward_ne_new_4hz"] += int((live & (fwd != new)).sum())
        chk["gated_ne_native_4hz"] += int(live.sum()) - int(
            (gate.last_obs[live] == obs["lidar"][live]).flatten(1).all(1).sum())
    if d.any():                                                         # auto-reset envs: new episode, reset scan forwarded
        gate.reset(torch.nonzero(d).squeeze(1), obs["lidar"], u.scan_t)
        age_sum[d], age_max[d] = 0, 0
    rec.record_step(act, rew, dones, ext, gated(obs), fresh=fwd, t_scan=gate.last_forward.float() * u.physics_dt)
    pos_end = torch.stack([term["x"], term["y"]], 1)
    path += torch.where(alive, (pos_end - prev).norm(dim=1), torch.zeros_like(path))
    ret += torch.where(alive, rew, torch.zeros_like(rew))
    minc = torch.where(alive, torch.minimum(minc, term["clear"]), minc)
    if new_end.any():
        res["outcome"][new_end] = ext["outcome"][new_end].float()
        res["steps"][new_end] = float(k + 1)
        res["final_dist"][new_end] = term["goal_d"][new_end]
        res["min_clear"][new_end], res["ret"][new_end], res["path"][new_end] = minc[new_end], ret[new_end], path[new_end]
        done_once |= new_end
    prev = corr.robot_pose2d()[0].clone()
    if bool(done_once.all()):
        break
t_eval = time.time() - t_loop
chk["native_intervals_ticks"] = sorted(chk["native_intervals_ticks"])

R = {k: v.cpu().numpy() for k, v in res.items()}
SE = {k: v.cpu().numpy() for k, v in sens.items()}
dtp = u.physics_dt
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
sensing_cfg = {"gate": "real-robot fixed-rate scan gate (Sim2Real real_eval_fixed.py _on_raw_scan), "
                       "AdaptiveSensing/evaluation/scan_gate.py",
               "requested_hz": args.fixed, "period_s": 1.0 / args.fixed, "period_physics_ticks": gate.period,
               "native_lidar_period_s": u.scan_period, "control_dt_s": u.step_dt, "physics_dt_s": dtp,
               "rule": "every new native scan: forward if t >= next_sample_time, then next_sample_time += period until "
                       "> t; reset scan always forwarded, next_sample_time = t0 + period",
               "policy_input": "{'policy': current env navigation features, 'lidar': last_sampled_obs}"}
with open(os.path.join(out_dir, "run_config.json"), "w") as f:
    json.dump({"exp_id": args.exp_id, "protocol": args.protocol, "custom": custom, "sensing": sensing_cfg,
               "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), X.REPO_ROOT),
               "settings": {k: getattr(args, k) for k in ("fixed", "episodes", "obstacle_seed", "start_seed", "goal",
                                                          "timeout_s", "seed")},
               "training_run_config": run_cfg_path if run_exp else None, "command": sys.argv,
               "repo": X.git_state(X.REPO_ROOT), "recording_schema": ED.SCHEMA_VERSION,
               "env_cfg_resolved": cfg.to_dict()}, f, indent=1, default=str)
rows = []
for i in range(n):
    oc = OUTCOME_NAMES.get(int(R["outcome"][i]), "running")
    dur = R["steps"][i] * u.step_dt
    nf, nn = int(SE["n_forward"][i]), int(SE["n_native"][i])
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
                 "success": oc == "goal", "outcome": oc, "n_steps": int(R["steps"][i]), "duration_s": round(dur, 2),
                 "episode_return": round(float(R["ret"][i]), 3), "final_goal_distance_m": round(float(R["final_dist"][i]), 4),
                 "distance_traveled_m": round(float(R["path"][i]), 3), "min_clearance_m": round(float(R["min_clear"][i]), 4),
                 # sensing (first episode; counts include the reset scan, rates exclude it; the terminal state is not observed)
                 "requested_hz": args.fixed, "native_scans": nn, "accepted_scans": nf,
                 "native_hz": round((nn - 1) / dur, 4) if dur > 0 else None,
                 "measured_hz": round((nf - 1) / dur, 4) if dur > 0 else None,
                 "mean_interval_s": round(SE["last_forward"][i] * dtp / (nf - 1), 4) if nf > 1 else None,
                 "min_interval_s": round(SE["int_min"][i] * dtp, 4) if nf > 1 else None,
                 "max_interval_s": round(SE["int_max"][i] * dtp, 4) if nf > 1 else None,
                 "mean_obs_age_s": round(SE["age_sum"][i] * dtp / R["steps"][i], 4) if R["steps"][i] > 0 else None,
                 "max_obs_age_s": round(SE["age_max"][i] * dtp, 4)})
policy_name = os.path.basename(os.path.normpath(out_dir))


def episode_meta(e, row):
    m = {"episode_index": e, "exp_id": args.exp_id, "protocol": args.protocol, "policy": policy_name,
         "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), X.REPO_ROOT),
         "seeds": {"start_pose_seed": row["start_pose_seed"], "static_obstacles_seed": row["static_obstacles_seed"],
                   "env_seed": args.seed},
         "start": {"x": row["start_x"], "y": row["start_y"], "yaw": row["start_theta"], "index": row["start_pose_index"]},
         "goal": {"x": row["goal_x"], "y": row["goal_y"]}, "deterministic_policy": True,
         "scan_gate": {**sensing_cfg, **{k: row[k] for k in ("native_scans", "accepted_scans", "native_hz", "measured_hz",
                                                             "mean_interval_s", "min_interval_s", "max_interval_s",
                                                             "mean_obs_age_s", "max_obs_age_s")},
                       "note": "lidar.npz and steps.csv scan_fresh hold the FORWARDED scans (the policy's observation)"}}
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
# episodes.csv (reference schema) from the evaluation values and the written recordings (terminal pose, collision type)
table = []
for r in rows:
    epd = os.path.join(out_dir, f"episode_{r['episode']:04d}")
    seed_ep = json.load(open(os.path.join(epd, "metadata.json")))["random_seeds"]["episode_seed"]
    table.append(M.episode_row(epd, episode=r["episode"], seed=seed_ep, policy_type="fixed", fixed_fps=args.fixed,
                               mean_fps=args.fixed, env_outcome=r["outcome"], n_steps=r["n_steps"],
                               nav_return=r["episode_return"], frame_count=r["accepted_scans"],
                               final_goal_distance_m=r["final_goal_distance_m"], distance_traveled_m=r["distance_traveled_m"],
                               start=(r["start_x"], r["start_y"], r["start_theta"]), goal=(r["goal_x"], r["goal_y"]),
                               frame_cost=args.frame_cost, budget=args.budget, budget_penalty=args.budget_penalty,
                               footprint_radius=float(corr.cfg.footprint_radius)))
M.write_episodes(os.path.join(out_dir, "episodes.csv"), table)
ED.write_json(os.path.join(out_dir, "world_geometry.json"), rec.fixed_geometry())
succ = np.array([r["success"] for r in rows])
oc_counts = {name: int(sum(r["outcome"] == name for r in rows)) for name in list(OUTCOME_NAMES.values()) + ["running"]}


def ms(x):
    return [float(np.mean(x)), float(np.std(x))] if len(x) else [float("nan"), float("nan")]


dur_all = np.array([r["duration_s"] for r in rows])
nf_all, nn_all = np.array([r["accepted_scans"] for r in rows]), np.array([r["native_scans"] for r in rows])
mi = [r["min_interval_s"] for r in rows if r["min_interval_s"] is not None]
mx = [r["max_interval_s"] for r in rows if r["max_interval_s"] is not None]
sensing_summary = {
    "requested_hz": args.fixed, "native_lidar_hz": 1.0 / u.scan_period, "control_hz": 1.0 / u.step_dt,
    "measured_hz_pooled": float((nf_all - 1).sum() / dur_all.sum()),
    "measured_hz_mean_std": ms([r["measured_hz"] for r in rows if r["measured_hz"] is not None]),
    "native_hz_pooled": float((nn_all - 1).sum() / dur_all.sum()),
    "accepted_scans_total": int(nf_all.sum()), "native_scans_total": int(nn_all.sum()),
    "interval_s_min_max": [min(mi) if mi else None, max(mx) if mx else None],
    "obs_age_s_mean": float(sum(r["mean_obs_age_s"] * r["n_steps"] for r in rows) / sum(r["n_steps"] for r in rows)),
    "obs_age_s_max": float(max(r["max_obs_age_s"] for r in rows))}
S_ = {"checkpoint": args.checkpoint, "episodes": n, "goal": list(args.goal), "timeout_s": args.timeout_s,
      "obstacle_seeds": None if dynamic else [args.obstacle_seed, args.obstacle_seed + n - 1], "start_seeds": [args.start_seed, args.start_seed + n - 1],
      "sensing": sensing_summary,
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
      "checks": {"max_start_placement_err_m": chk_start, "max_offset_vs_reference_err": chk_layout, "max_goal_err": chk_goal,
                 "sensing": chk},
      "timing_s": {"launch_and_setup": round(t_setup, 1), "episodes": round(t_eval, 1), "total": round(time.time() - t_start, 1)}}
S_["summary"] = M.summarize(table, policy_name)             # reference schema (= this run's summary.csv row)
S_["tables_schema"] = "AdaptiveSensing/evaluation/metrics.py (reference schema)"
json.dump(S_, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
M.update_summary_csv(os.path.join(os.path.dirname(os.path.normpath(out_dir)), "summary.csv"), S_["summary"])
print("[EVAL]", json.dumps(S_), flush=True)
os._exit(0)
