"""Evaluation of a trained temporal sensing policy (TMP02 checkpoint) over the frozen NAV05 policy, on the TMP01
protocol (ref_chase_c10: the same 100 episodes, seeds, scene and recordings as the fixed-rate evaluations).

Sensing exactly as in training (AdaptiveSensing/training/temporal_sensing.py: real-robot gate, decision at each control
instant gating that instant's native scan, set_target_fps on acceptance, reset rate 4 Hz); the sensing policy acts
deterministically (argmax, as the reference evaluator) with its LSTM state reset at every episode start. Tables in the
reference schema (AdaptiveSensing/evaluation/metrics.py); adaptive_return with the checkpoint's training frame cost
unless --frame_cost is given. Recordings as evaluate_fixed_rate.py (steps.csv scan_fresh / lidar.npz = accepted scans).

    bin/isaacrun -g 7 -- python AdaptiveSensing/scripts/evaluate_adaptive.py --headless --checkpoint <run>/checkpoints/model_00400.pt
    --force_fps <hz>: constant action <hz> and gate reset rate <hz> instead of the policy (equivalence check: must
                      reproduce the TMP01 fixed-rate evaluation)
Output: AdaptiveSensing/eval/temporal/<exp_id>/<protocol>/<run>_model_<iteration>/ (+ the protocol summary.csv row)
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

NATIVE_PERIOD_S = 0.25
PROTOCOLS = {"ref_chase_c10": dict(episodes=100, start_seed=4000, goal=(-2.0, -2.0), timeout_s=70.0)}

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True, help="TMP02 checkpoint (train_temporal_ppo.py model_<iteration>.pt)")
p.add_argument("--protocol", default="ref_chase_c10", choices=sorted(PROTOCOLS))
p.add_argument("--seed", type=int, default=42)                  # env seed of the protocol (as TMP01)
p.add_argument("--frame_cost", type=float, default=None, help="default: the checkpoint's training frame cost")
p.add_argument("--budget", type=int, default=None, help="default: the checkpoint's budget")
p.add_argument("--budget_penalty", type=float, default=None, help="default: the checkpoint's budget penalty")
p.add_argument("--force_fps", type=float, default=None, help="validation: constant sensing rate instead of the policy")
p.add_argument("--fast_path", default="all", help="env performance fast path (bitwise-equivalent): all | none | items")
p.add_argument("--out_dir", default=None)
p.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
for k, v in PROTOCOLS[args.protocol].items():
    setattr(args, k, v)
ck_path = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(os.path.dirname(ck_path))
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from st_nav.scenes import corridor_static_random as S  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import OUTCOME_NAMES  # noqa: E402
from AdaptiveSensing.evaluation import metrics as M  # noqa: E402
from AdaptiveSensing.training.sensing_agent import SensingAgent  # noqa: E402
from AdaptiveSensing.training.temporal_sensing import TemporalSensing, load_nav_policy, nav_run_cfgs  # noqa: E402
from Navigation.evaluation.recorder import EpisodeRecorder  # noqa: E402
from tools import episode_data as ED  # noqa: E402

t_start = time.time()
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
A_ = ck["args"]
exp_id = ck.get("exp_id", "TMP02_chase-ob1ob2_adaptive-nav05")
frame_cost = float(A_["frame_cost"] if args.frame_cost is None else args.frame_cost)
budget = int(A_["budget"] if args.budget is None else args.budget)
budget_penalty = float(A_["budget_penalty"] if args.budget_penalty is None else args.budget_penalty)
fps_choices = [float(f) for f in ck["fps_choices"]]
if args.force_fps is not None and args.force_fps not in fps_choices:
    raise SystemExit(f"--force_fps must be one of {fps_choices}")
policy_name = os.path.basename(run_dir) + f"_model_{int(ck['iteration']):05d}" + (
    f"_forced{args.force_fps:g}Hz" if args.force_fps is not None else "")
if args.out_dir is None:
    args.out_dir = str(X.paths(exp_id)["eval"] / args.protocol / policy_name)
if os.path.exists(os.path.join(args.out_dir, "episodes.csv")) and not args.overwrite:
    raise SystemExit(f"{args.out_dir} already has results; pass --overwrite")
nav_ck = A_["nav_checkpoint"] if os.path.isabs(A_["nav_checkpoint"]) else os.path.join(X.REPO_ROOT, A_["nav_checkpoint"])


def reference_start_pose(seed):                    # = evaluate_3d.py / evaluate_fixed_rate.py
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(S.START_POSE_LIST)))
    dyaw = float(rng.uniform(-S.START_YAW_NOISE, S.START_YAW_NOISE))
    return i, (*S.START_POSE_LIST[i], (S.START_YAW_CENTER + dyaw + math.pi) % (2 * math.pi) - math.pi)


dev = args.device
n = args.episodes
starts = [reference_start_pose(args.start_seed + i) for i in range(n)]
nav_agent_cfg, nav_run_exp, cfg = nav_run_cfgs(nav_ck, dev)      # NAV05's env settings (= TMP01) + protocol
cfg.scene.num_envs, cfg.sim.device, cfg.seed, cfg.episode_length_s = n, dev, args.seed, args.timeout_s
cfg.start_poses = tuple(s for _, s in starts)
cfg.goals = (tuple(args.goal),)
cfg.fast_path = () if args.fast_path == "none" else tuple(x.strip() for x in args.fast_path.split(","))
genv = gym.make("STNav-TB4-Corridor-Lidar3D-v0", cfg=cfg)
env = RslRlVecEnvWrapper(genv, clip_actions=nav_agent_cfg.clip_actions)
nav = load_nav_policy(env, nav_agent_cfg, nav_ck, dev)
u = genv.unwrapped
corr = u.corr
if abs(u.scan_period - NATIVE_PERIOD_S) > 1e-9 or abs(u.step_dt - 0.2) > 1e-9:
    raise SystemExit(f"unexpected timing: native period {u.scan_period} s, control step {u.step_dt} s")
agent = SensingAgent(tuple(ck["img_shape"]), 3, len(fps_choices), A_["lstm_hidden_size"], tuple(A_["cnn_channels"]),
                     A_["cnn_kernel"], A_["cnn_stride"], A_["cnn_projection_dim"]).to(dev)
agent.load_state_dict(ck["model_state_dict"])
agent.eval()
sense = TemporalSensing(env, nav, fps_choices, frame_cost, budget, budget_penalty, reset_hz=args.force_fps)
gate = sense.gate
t_setup = time.time() - t_start
torch.set_grad_enabled(False)

lstm = agent.initial_state(n, dev)
force_idx = None if args.force_fps is None else torch.full((n,), fps_choices.index(args.force_fps), device=dev)


def decide(img, aux, done):
    """Sensing action of this instant (argmax; LSTM reset where done)."""
    global lstm
    if force_idx is not None:
        return force_idx
    a, _, _, _, lstm = agent.get_action_and_value(img, aux, lstm, done, deterministic=True)
    return a


def nav_obs():
    o = sense.obs.clone()
    o["lidar"] = gate.last_obs.clone()
    return o


rec = EpisodeRecorder(u)
img, aux = sense.reset()
rec.record_reset(nav_obs())
p0, _ = corr.robot_pose2d()
chk_start = float(torch.max((p0 - corr.start[:, :2]).norm(dim=1)))
chk_goal = float((corr.goal - torch.tensor(args.goal, device=dev)).abs().max())
om = corr.obs_model
ob_ep = {"active": om.active.cpu().numpy().copy(), "speed": om.speed.cpu().numpy().copy(),
         "path_y": om.path_y.cpu().numpy().copy()}
done_once = torch.zeros(n, dtype=torch.bool, device=dev)
ret = torch.zeros(n, device=dev)
path = torch.zeros(n, device=dev)
minc = torch.full((n,), float("inf"), device=dev)
res = {k: torch.zeros(n, device=dev) for k in ("outcome", "steps", "final_dist", "min_clear", "ret", "path")}
SN = ("frames", "native_scans", "causal", "fps_inforce_sum", "fps_applied_sum", "return", "age_ticks_sum", "age_ticks_max",
      "length", "budget_exceeded")
sens = {k: torch.zeros(n, device=dev) for k in SN + ("last_forward", "int_min", "int_max")}
applied_hist = torch.zeros(n, len(fps_choices), device=dev)
a = decide(img, aux, torch.zeros(n, device=dev))
sense.gate_step(a)                                                   # tick 0 (the reset scan is not new)
prev = p0.clone()
t_loop = time.time()
for k in range(int(u.max_episode_length) + 2):
    alive = ~done_once
    applied_hist += torch.nn.functional.one_hot(sense._tick[0], len(fps_choices)).float() * (
        sense._tick[2] & alive).float()[:, None]
    (img, aux), r, d, info = sense.advance()
    term = rec.end_state
    new_end = d & alive
    if bool(new_end.any()):                                          # first episode ends: freeze its sensing record
        for key in SN:
            sens[key] = torch.where(new_end, info["episode"][key].float(), sens[key])
        for key in ("last_forward", "int_min", "int_max"):
            sens[key] = torch.where(new_end, info["gate_end"][key].float(), sens[key])
    a = decide(img, aux, d.float())                                  # next instant: decide and gate its scan, so the
    fwd = sense.gate_step(a)                                         # recorded next observation is what NAV05 sees
    rew, dones, ext = info["r_env"], info["dones"], info["extras"]
    rec.record_step(info["nav_action"], rew, dones, ext, nav_obs(), fresh=fwd,
                    t_scan=gate.last_forward.float() * u.physics_dt)
    pos_end = torch.stack([term["x"], term["y"]], 1)
    path += torch.where(alive, (pos_end - prev).norm(dim=1), torch.zeros_like(path))
    ret += torch.where(alive, rew, torch.zeros_like(rew))
    minc = torch.where(alive, torch.minimum(minc, term["clear"]), minc)
    if bool(new_end.any()):
        res["outcome"][new_end] = ext["outcome"][new_end].float()
        res["steps"][new_end] = float(k + 1)
        res["final_dist"][new_end] = term["goal_d"][new_end]
        res["min_clear"][new_end], res["ret"][new_end], res["path"][new_end] = minc[new_end], ret[new_end], path[new_end]
        done_once |= new_end
    prev = corr.robot_pose2d()[0].clone()
    if bool(done_once.all()):
        break
t_eval = time.time() - t_loop

R = {k: v.cpu().numpy() for k, v in res.items()}
SE = {k: v.cpu().numpy() for k, v in sens.items()}
H = applied_hist.cpu().numpy()
dtp = u.physics_dt
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
sensing_cfg = {"policy": "forced constant rate (validation)" if force_idx is not None else
               "trained sensing policy, deterministic (argmax), LSTM reset per episode",
               "checkpoint": os.path.relpath(ck_path, X.REPO_ROOT), "iteration": int(ck["iteration"]),
               "global_step": int(ck["global_step"]), "fps_choices": fps_choices,
               "reset_hz": sense.reset_hz, "frame_cost": frame_cost, "budget": budget, "budget_penalty": budget_penalty,
               "gate": "real-robot gate + set_target_fps (AdaptiveSensing/evaluation/scan_gate.py), decision at each "
                       "control instant gates that instant's native scan (temporal_sensing.py)",
               "native_lidar_period_s": u.scan_period, "control_dt_s": u.step_dt, "physics_dt_s": dtp}
with open(os.path.join(out_dir, "run_config.json"), "w") as f:
    json.dump({"exp_id": exp_id, "protocol": args.protocol, "sensing": sensing_cfg,
               "nav_checkpoint": os.path.relpath(nav_ck, X.REPO_ROOT), "training_run": os.path.relpath(run_dir, X.REPO_ROOT),
               "settings": {k: getattr(args, k) for k in ("episodes", "start_seed", "goal", "timeout_s", "seed",
                                                          "force_fps", "fast_path")},
               "command": sys.argv, "repo": X.git_state(X.REPO_ROOT), "recording_schema": ED.SCHEMA_VERSION,
               "env_cfg_resolved": cfg.to_dict()}, f, indent=1, default=str)
rows = []
for i in range(n):
    oc = OUTCOME_NAMES.get(int(R["outcome"][i]), "running")
    dur = R["steps"][i] * u.step_dt
    nf, nn, L = int(SE["frames"][i]), int(SE["native_scans"][i]), max(int(SE["length"][i]), 1)
    rows.append({"episode": i, "start_pose_seed": args.start_seed + i, "start_pose_index": starts[i][0],
                 "start_x": starts[i][1][0], "start_y": starts[i][1][1], "start_theta": round(starts[i][1][2], 5),
                 "goal_x": args.goal[0], "goal_y": args.goal[1], "success": oc == "goal", "outcome": oc,
                 "n_steps": int(R["steps"][i]), "duration_s": round(dur, 2), "episode_return": round(float(R["ret"][i]), 3),
                 "final_goal_distance_m": round(float(R["final_dist"][i]), 4),
                 "distance_traveled_m": round(float(R["path"][i]), 3), "min_clearance_m": round(float(R["min_clear"][i]), 4),
                 "mean_fps": float(SE["fps_inforce_sum"][i]) / L, "sensing_return": float(SE["return"][i]),
                 "native_scans": nn, "accepted_scans": nf, "charged_frames": int(SE["causal"][i]),
                 "native_hz": round((nn - 1) / dur, 4) if dur > 0 else None,
                 "measured_hz": round((nf - 1) / dur, 4) if dur > 0 else None,
                 "mean_applied_fps": round(float(SE["fps_applied_sum"][i]) / SE["causal"][i], 4) if SE["causal"][i] else None,
                 "applied_decisions_by_fps": {f"{f:g}": int(H[i, j]) for j, f in enumerate(fps_choices)},
                 "mean_interval_s": round(SE["last_forward"][i] * dtp / (nf - 1), 4) if nf > 1 else None,
                 "min_interval_s": round(SE["int_min"][i] * dtp, 4) if nf > 1 else None,
                 "max_interval_s": round(SE["int_max"][i] * dtp, 4) if nf > 1 else None,
                 "mean_obs_age_s": round(SE["age_ticks_sum"][i] * dtp / R["steps"][i], 4) if R["steps"][i] > 0 else None,
                 "max_obs_age_s": round(SE["age_ticks_max"][i] * dtp, 4)})
DIAG = ("mean_fps", "sensing_return", "native_scans", "accepted_scans", "charged_frames", "native_hz", "measured_hz",
        "mean_applied_fps", "applied_decisions_by_fps", "mean_interval_s", "min_interval_s", "max_interval_s",
        "mean_obs_age_s", "max_obs_age_s")


def episode_meta(e, row):
    m = {"episode_index": e, "exp_id": exp_id, "protocol": args.protocol, "policy": policy_name,
         "checkpoint": os.path.relpath(ck_path, X.REPO_ROOT),
         "seeds": {"start_pose_seed": row["start_pose_seed"], "static_obstacles_seed": None, "env_seed": args.seed},
         "start": {"x": row["start_x"], "y": row["start_y"], "yaw": row["start_theta"], "index": row["start_pose_index"]},
         "goal": {"x": row["goal_x"], "y": row["goal_y"]}, "deterministic_policy": True,
         "scan_gate": {**sensing_cfg, **{k: row[k] for k in DIAG},
                       "note": "lidar.npz and steps.csv scan_fresh hold the ACCEPTED scans (NAV05's observation)"}}
    m.update({k: row[k] for k in ("outcome", "success", "n_steps", "duration_s", "episode_return", "final_goal_distance_m",
                                  "distance_traveled_m", "min_clearance_m")})
    m["moving_obstacles"] = {"ids": [f"OB{j + 1}" for j in range(ob_ep["active"].shape[1])],
                             "active": ob_ep["active"][e].tolist(), "speed_mps": ob_ep["speed"][e].tolist(),
                             "path_y_m": ob_ep["path_y"][e].tolist()}
    return m


rec.export(out_dir, R["steps"], rows, rec.static_metadata(cfg), episode_meta)
ED.write_json(os.path.join(out_dir, "world_geometry.json"), rec.fixed_geometry())
table = []
chk = {"adaptive_return_vs_reward_sum_max_diff": 0.0}
for r in rows:
    epd = os.path.join(out_dir, f"episode_{r['episode']:04d}")
    seed_ep = json.load(open(os.path.join(epd, "metadata.json")))["random_seeds"]["episode_seed"]
    row = M.episode_row(epd, episode=r["episode"], seed=seed_ep,
                        policy_type="fixed" if force_idx is not None else "adaptive",
                        fixed_fps=args.force_fps, mean_fps=r["mean_fps"], env_outcome=r["outcome"], n_steps=r["n_steps"],
                        nav_return=r["episode_return"], frame_count=r["accepted_scans"],
                        final_goal_distance_m=r["final_goal_distance_m"], distance_traveled_m=r["distance_traveled_m"],
                        start=(r["start_x"], r["start_y"], r["start_theta"]), goal=(r["goal_x"], r["goal_y"]),
                        frame_cost=frame_cost, budget=budget, budget_penalty=budget_penalty,
                        footprint_radius=float(corr.cfg.footprint_radius))
    chk["adaptive_return_vs_reward_sum_max_diff"] = max(chk["adaptive_return_vs_reward_sum_max_diff"],
                                                        abs(row["adaptive_return"] - r["sensing_return"]))
    table.append(row)
M.write_episodes(os.path.join(out_dir, "episodes.csv"), table)
summary = M.summarize(table, policy_name)
dur_all = np.array([r["duration_s"] for r in rows])
nf_all = np.array([r["accepted_scans"] for r in rows])
S_ = {"summary": summary, "tables_schema": "AdaptiveSensing/evaluation/metrics.py (reference schema)",
      "sensing_policy": sensing_cfg,
      "outcomes": {name: int(sum(r["outcome"] == name for r in rows)) for name in list(OUTCOME_NAMES.values()) + ["running"]},
      "sensing": {"measured_hz_pooled": float((nf_all - 1).sum() / dur_all.sum()),
                  "native_hz_pooled": float(sum(r["native_scans"] - 1 for r in rows) / dur_all.sum()),
                  "accepted_scans_total": int(nf_all.sum()), "charged_frames_total": int(sum(r["charged_frames"] for r in rows)),
                  "applied_decisions_by_fps": {f"{f:g}": int(H[:, j].sum()) for j, f in enumerate(fps_choices)},
                  "obs_age_s_mean": float(sum(r["mean_obs_age_s"] * r["n_steps"] for r in rows) / sum(r["n_steps"] for r in rows)),
                  "obs_age_s_max": float(max(r["max_obs_age_s"] for r in rows))},
      "min_clearance_m_mean_min": [float(np.mean([r["min_clearance_m"] for r in rows])),
                                   float(np.min([r["min_clearance_m"] for r in rows]))],
      "checks": {"max_start_placement_err_m": chk_start, "max_goal_err": chk_goal, **chk},
      "timing_s": {"launch_and_setup": round(t_setup, 1), "episodes": round(t_eval, 1), "total": round(time.time() - t_start, 1)}}
json.dump(S_, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
M.update_summary_csv(os.path.join(os.path.dirname(os.path.normpath(out_dir)), "summary.csv"), summary)
print("[EVAL-ADAPTIVE]", json.dumps({"summary": summary, "sensing": S_["sensing"], "checks": S_["checks"]}), flush=True)
os._exit(0)
