"""Fixed spatiotemporal sensing evaluation of the frozen NAV05 3D-LiDAR navigation policy (STS01).

The TMP01 fixed-rate evaluation (AdaptiveSensing/scripts/evaluate_fixed_rate.py: protocol ref_chase_c10, episode seeds,
start poses, goal, env, frozen NAV05 loading, first episode of each of 100 envs) with the range image split into
horizontal sections (default 46 / 47 / 46 columns = left / center / right; AdaptiveSensing/evaluation/sections.py), each
gated independently at its own constant rate by the real-robot gate (AdaptiveSensing/evaluation/section_scan_gate.py):

  * the env and its native LiDAR are unchanged: 4 Hz scans at the 5 Hz control instants, held in between;
  * every new native scan is accepted by the sections whose deadline it has reached; only their columns of the held
    image change (the other sections keep their last accepted scan); every section starts at its own rate with the
    reset scan;
  * NAV05 receives {"policy": current navigation features, "lidar": the composite 34 x 139 image}.
Consumption: full-scan equivalents FSE = rays acquired (reset scan included) / 4,726.

Checks during the run (summary.json "checks"; all must be 0): an independent literal per-section gate (accept /
deadline / counters), the composite image against a history of the native scans (every section's columns = the
native scan of its last acquisition), untouched columns of non-accepting sections, native intervals (0.2 / 0.4 s),
nav features current; with every section at 4 Hz, composite = native scan.

    bin/isaacrun -g 6 -- python AdaptiveSensing/scripts/evaluate_fixed_sections.py --headless --sections 4 1 0.2

Output: AdaptiveSensing/eval/spatiotemporal/<exp_id>/<protocol>/<run>_<checkpoint>_sections<fL>-<fC>-<fR>Hz/
episodes.csv, summary.json, run_config.json (+ episode_NNNN/ recordings with --record), and one row of
<protocol>/summary.csv. Tables: AdaptiveSensing/evaluation/section_metrics.py.
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
NATIVE_PERIOD_S = 0.25
# identical to evaluate_fixed_rate.py / evaluate_3d.py
PROTOCOLS = {"ref_static_c10": dict(episodes=100, obstacle_seed=3000, start_seed=4000, goal=(-2.0, -2.0), timeout_s=70.0),
             "ref_chase_c10": dict(episodes=100, obstacle_seed=None, start_seed=4000, goal=(-2.0, -2.0), timeout_s=70.0)}
HIST = 32                                                        # native scans kept for the image check (> 0.2 Hz age)

p = argparse.ArgumentParser()
p.add_argument("--sections", type=float, nargs="+", required=True, help="constant rate (Hz) of every section, left to right")
p.add_argument("--columns", type=int, nargs="+", default=[46, 47, 46], help="columns per section, left to right")
p.add_argument("--checkpoint", default=NAV05)
p.add_argument("--exp_id", default="STS01_chase-ob1ob2_fixed-sections-nav05")
p.add_argument("--protocol", default=None, choices=sorted(PROTOCOLS))
p.add_argument("--episodes", type=int, default=None)
p.add_argument("--obstacle_seed", type=int, default=None)
p.add_argument("--start_seed", type=int, default=None)
p.add_argument("--goal", type=float, nargs=2, default=None)
p.add_argument("--timeout_s", type=float, default=None)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--frame_cost", type=float, default=0.0, help="sensing cost per full-scan equivalent in adaptive_return")
p.add_argument("--budget", type=int, default=500)
p.add_argument("--budget_penalty", type=float, default=15.0)
p.add_argument("--record", action="store_true", help="also write per-episode recordings (episode_NNNN/, section-aware lidar.npz)")
p.add_argument("--out_dir", default=None)
p.add_argument("--overwrite", action="store_true")
p.add_argument("--fast_path", default="none", help="env performance fast path: none | all | comma-separated items")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if len(args.sections) != len(args.columns):
    raise SystemExit(f"{len(args.sections)} rates for {len(args.columns)} sections")
run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
run_cfg_path = os.path.join(run_dir, "params", "config.yaml")
run_exp = X.yaml.safe_load(open(run_cfg_path)) if os.path.exists(run_cfg_path) else None
run_env = {**X.flatten((run_exp or {}).get("env") or {}), **(((run_exp or {}).get("set") or {}).get("env") or {})}
dynamic = run_env.get("scene_name", "corridor_static_random") != "corridor_static_random"
if args.protocol is None:
    args.protocol = "ref_chase_c10" if dynamic else "ref_static_c10"
if dynamic != (args.protocol == "ref_chase_c10"):
    raise SystemExit(f"protocol {args.protocol} does not match the run's scene")
custom = [k for k in PROTOCOLS[args.protocol] if getattr(args, k) is not None]
for k, v in PROTOCOLS[args.protocol].items():
    setattr(args, k, v if getattr(args, k) is None else getattr(args, k))
config = "-".join(f"{h:g}" for h in args.sections)
hz_tag = f"sections{config}Hz" + ("" if args.columns == [46, 47, 46] else "_cols" + "-".join(map(str, args.columns)))
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
from st_nav.scenes._obstacle_motion import OBSTACLE_RADIUS  # noqa: E402
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cnn_cfg import CorridorNavCNNPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import OUTCOME_NAMES  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_lidar3d_cfg import CorridorNavLidar3DEnvCfg  # noqa: E402
from AdaptiveSensing.evaluation import section_metrics as SM  # noqa: E402
from AdaptiveSensing.evaluation.scan_gate import period_ticks  # noqa: E402
from AdaptiveSensing.evaluation.section_recorder import SectionEpisodeRecorder  # noqa: E402
from AdaptiveSensing.evaluation.section_scan_gate import SectionScanGate  # noqa: E402
from AdaptiveSensing.evaluation.sections import SectionLayout  # noqa: E402
from tools import episode_data as ED  # noqa: E402


def reference_start_pose(seed):
    """= evaluate_fixed_rate.py / evaluate_3d.py."""
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(S.START_POSE_LIST)))
    dyaw = float(rng.uniform(-S.START_YAW_NOISE, S.START_YAW_NOISE))
    return i, (*S.START_POSE_LIST[i], (S.START_YAW_CENTER + dyaw + math.pi) % (2 * math.pi) - math.pi)


n = args.episodes
starts = [reference_start_pose(args.start_seed + i) for i in range(n)]
cfg = CorridorNavLidar3DEnvCfg()
if run_exp:
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
    raise SystemExit("evaluate_fixed_sections.py expects a 3D-LiDAR run (env sensor lidar3d, obs_layout image)")
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

lc = corr.lidar.cfg
layout = SectionLayout.from_grid(lc.grid_col, u.img_shape[2], args.columns)
NS, names = layout.n_sections, layout.names
gate = SectionScanGate(n, layout, args.sections, u.physics_dt, device=dev)
P = gate.reset_periods.clone()                                          # (S,) constant periods (fixed rates)
rays_t = layout.rays_tensor(dev)
all_ids = torch.arange(n, device=dev)
col_bounds = layout.col_bounds


def gated(o):
    """Policy input: the env's current navigation features + the composite held image."""
    g = o.clone()
    g["lidar"] = gate.last_obs.clone()
    return g


rec = SectionEpisodeRecorder(u, layout, capture_lidar=args.record)
obs = env.get_observations()
gate.reset(all_ids, obs["lidar"], u.scan_t)
rec.record_reset(obs)                                                   # reset: composite = native scan
p0, _ = corr.robot_pose2d()
chk_start = float(torch.max((p0 - corr.start[:, :2]).norm(dim=1)))
om = corr.obs_model
ob_ep = None if om is None else {"active": om.active.cpu().numpy().copy(), "speed": om.speed.cpu().numpy().copy(),
                                 "path_y": om.path_y.cpu().numpy().copy()}
chk_goal = float((corr.goal - torch.tensor(args.goal, device=dev)).abs().max())

# ---- independent literal gate (while-loop form, real_eval_fixed.py) and native-scan history for the image check
i64 = dict(dtype=torch.int64, device=dev)
now0 = gate.ticks(u.scan_t)
lit = {"native": now0.clone(), "next": now0[:, None] + P, "last": now0[:, None].expand(-1, NS).clone(),
       "n": torch.ones(n, NS, **i64), "n_any": torch.ones(n, **i64), "rays": torch.full((n,), layout.total_rays, **i64)}
hist_img = torch.zeros((HIST,) + tuple(obs["lidar"].shape), device=dev)
hist_t = torch.full((HIST, n), -1, **i64)
hist_ptr = torch.zeros(n, **i64)


def hist_push(mask, lidar, ticks_):
    idx = torch.nonzero(mask).squeeze(1)
    if idx.numel():
        hist_img[hist_ptr[idx], idx] = lidar[idx]
        hist_t[hist_ptr[idx], idx] = ticks_[idx]
        hist_ptr[idx] = (hist_ptr[idx] + 1) % HIST


def lit_reset(mask, ticks_):
    m, mS = mask, mask[:, None]
    lit["native"] = torch.where(m, ticks_, lit["native"])
    lit["next"] = torch.where(mS, ticks_[:, None] + P, lit["next"])
    lit["last"] = torch.where(mS, ticks_[:, None].expand(-1, NS), lit["last"])
    lit["n"] = torch.where(mS, torch.ones_like(lit["n"]), lit["n"])
    lit["n_any"] = torch.where(m, torch.ones_like(lit["n_any"]), lit["n_any"])
    lit["rays"] = torch.where(m, torch.full_like(lit["rays"], layout.total_rays), lit["rays"])
    hist_t[:, m] = -1
    hist_push(m, obs["lidar"], ticks_)


hist_push(torch.ones(n, dtype=torch.bool, device=dev), obs["lidar"], now0)

done_once = torch.zeros(n, dtype=torch.bool, device=dev)
ret = torch.zeros(n, device=dev)
path = torch.zeros(n, device=dev)
res = {k: torch.zeros(n, device=dev) for k in ("outcome", "steps", "final_dist", "ret", "path")}
age_sum, age_max = torch.zeros(n, NS, **i64), torch.zeros(n, NS, **i64)
sens = {"n_native": torch.zeros(n, **i64), "n_any": torch.zeros(n, **i64), "rays": torch.zeros(n, **i64),
        **{k: torch.zeros(n, NS, **i64) for k in ("n_forward", "last_forward", "int_min", "int_max", "age_sum", "age_max")}}
chk = {k: 0 for k in ("literal_accept_mismatch", "literal_deadline_mismatch", "literal_last_mismatch",
                      "literal_counter_mismatch", "image_section_mismatch", "image_scan_not_in_history",
                      "untouched_columns_changed", "native_interval_violations", "forward_without_new_scan",
                      "composite_ne_native_all4hz", "nav_features_not_current", "reset_composite_ne_native")}
chk["native_intervals_ticks"] = set()
all4 = all(h == 4.0 for h in args.sections)
prev = p0.clone()
t_loop = time.time()
for k in range(int(u.max_episode_length) + 2):
    g_obs = gated(obs)
    alive = ~done_once
    age = gate.ticks(u.ep_time)[:, None] - gate.last_forward                # (N, S) section ages at this policy call
    age_sum += torch.where(alive[:, None], age, torch.zeros_like(age))
    age_max = torch.where(alive[:, None], torch.maximum(age_max, age), age_max)
    if not torch.equal(g_obs["policy"], obs["policy"]):
        chk["nav_features_not_current"] += 1
    act = policy(g_obs)
    obs, rew, dones, ext = env.step(act)
    d = dones.bool()
    term = rec.end_state
    new_end = d & alive
    if new_end.any():                                                   # first episode ends: freeze its sensing record
        for key, v in (("n_native", gate.n_native), ("n_any", gate.n_any), ("rays", gate.rays),
                       ("n_forward", gate.n_forward), ("last_forward", gate.last_forward), ("int_min", gate.int_min),
                       ("int_max", gate.int_max), ("age_sum", age_sum), ("age_max", age_max)):
            sens[key][new_end] = v[new_end]
        bad_cnt = new_end & ((gate.n_forward != lit["n"]).any(1) | (gate.n_any != lit["n_any"]) | (gate.rays != lit["rays"]))
        chk["literal_counter_mismatch"] += int(bad_cnt.sum())
    held_before = gate.last_obs.clone()
    nat_before = gate.last_native.clone()
    new, fwd = gate.step(obs["lidar"], u.scan_t)                       # done envs: overwritten by the reset below
    live = alive & ~d
    # ---- literal per-section gate on the same native scan
    now = gate.ticks(u.scan_t)
    l_new = now != lit["native"]
    lit["native"] = now
    l_acc = l_new[:, None] & (now[:, None] >= lit["next"])
    lit["last"] = torch.where(l_acc, now[:, None], lit["last"])
    while True:                                                         # while next_sample_time <= now: += period
        mm = l_acc & (lit["next"] <= now[:, None])
        if not bool(mm.any()):
            break
        lit["next"] = torch.where(mm, lit["next"] + P, lit["next"])
    lit["n"] += l_acc
    lit["n_any"] += l_acc.any(1)
    lit["rays"] += (l_acc.long() * rays_t).sum(1)
    lv = live[:, None]
    chk["literal_accept_mismatch"] += int((lv & (l_acc != fwd)).sum())
    chk["literal_deadline_mismatch"] += int((lv & (lit["next"] != gate.next_sample)).sum())
    chk["literal_last_mismatch"] += int((lv & (lit["last"] != gate.last_forward)).sum())
    # ---- composite image: each section's columns = the native scan of its last acquisition
    hist_push(live & new, obs["lidar"], now)
    for s, (c0, c1) in enumerate(col_bounds):
        match = hist_t == gate.last_forward[:, s][None]                  # (HIST, N)
        found = match.any(0)
        ref = hist_img[match.long().argmax(0), all_ids]
        same = (gate.last_obs[..., c0:c1] == ref[..., c0:c1]).flatten(1).all(1)
        chk["image_scan_not_in_history"] += int((live & ~found).sum())
        chk["image_section_mismatch"] += int((live & found & ~same).sum())
        untouched = (gate.last_obs[..., c0:c1] == held_before[..., c0:c1]).flatten(1).all(1)
        chk["untouched_columns_changed"] += int((live & ~fwd[:, s] & ~untouched).sum())
    gap = gate.last_native - nat_before
    chk["native_interval_violations"] += int((live & new & (gap != 20) & (gap != 40)).sum())
    chk["native_intervals_ticks"] |= set(gap[live & new].unique().tolist())
    chk["forward_without_new_scan"] += int((lv & fwd & ~new[:, None]).sum())
    if all4:
        chk["composite_ne_native_all4hz"] += int(live.sum()) - int(
            (gate.last_obs[live] == obs["lidar"][live]).flatten(1).all(1).sum())
    if d.any():                                                         # auto-reset envs: new episode, reset scan in every section
        gate.reset(d, obs["lidar"], u.scan_t)
        lit_reset(d, now)
        age_sum[d], age_max[d] = 0, 0
        chk["reset_composite_ne_native"] += int(d.sum()) - int((gate.last_obs[d] == obs["lidar"][d]).flatten(1).all(1).sum())
    rec.record_step_sections(act, rew, dones, ext, obs, fwd & ~d[:, None], t_scan=gate.last_forward.max(1).values.float() * u.physics_dt)
    pos_end = torch.stack([term["x"], term["y"]], 1)
    path += torch.where(alive, (pos_end - prev).norm(dim=1), torch.zeros_like(path))
    ret += torch.where(alive, rew, torch.zeros_like(rew))
    if new_end.any():
        res["outcome"][new_end] = ext["outcome"][new_end].float()
        res["steps"][new_end] = float(k + 1)
        res["final_dist"][new_end] = term["goal_d"][new_end]
        res["ret"][new_end], res["path"][new_end] = ret[new_end], path[new_end]
        done_once |= new_end
    prev = corr.robot_pose2d()[0].clone()
    if bool(done_once.all()):
        break
t_eval = time.time() - t_loop
chk["native_intervals_ticks"] = sorted(chk["native_intervals_ticks"])

# ------------------------------------------------------------------ tables
R = {k: v.cpu().numpy() for k, v in res.items()}
SE = {k: v.cpu().numpy() for k, v in sens.items()}
dtp = u.physics_dt
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
sensing_cfg = {"gate": "real-robot fixed-rate scan gate per section (Sim2Real real_eval_fixed.py _on_raw_scan), "
                       "AdaptiveSensing/evaluation/section_scan_gate.py",
               "sections": layout.to_dict(), "section_hz": list(args.sections),
               "section_period_physics_ticks": P.tolist(), "native_lidar_period_s": u.scan_period,
               "control_dt_s": u.step_dt, "physics_dt_s": dtp,
               "rule": "every new native scan: section s accepts it if t >= next_sample_time[s] (its columns of the held "
                       "image <- the scan), then next_sample_time[s] += period[s] until > t; reset scan accepted by every "
                       "section, next_sample_time[s] = t0 + period[s]",
               "policy_input": "{'policy': current env navigation features, 'lidar': composite held image}"}
with open(os.path.join(out_dir, "run_config.json"), "w") as f:
    json.dump({"exp_id": args.exp_id, "protocol": args.protocol, "custom": custom, "sensing": sensing_cfg,
               "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), X.REPO_ROOT),
               "settings": {k: getattr(args, k) for k in ("sections", "columns", "episodes", "obstacle_seed", "start_seed",
                                                          "goal", "timeout_s", "seed", "frame_cost", "budget",
                                                          "budget_penalty", "record", "fast_path")},
               "training_run_config": run_cfg_path if run_exp else None, "command": sys.argv,
               "repo": X.git_state(X.REPO_ROOT), "env_cfg_resolved": cfg.to_dict()}, f, indent=1, default=str)

n_obs = 0 if ob_ep is None else ob_ep["active"].shape[1]
fields = SM.episode_fields(names, n_obs)
weights = layout.weights
rows = []
for i in range(n):
    oc = OUTCOME_NAMES.get(int(R["outcome"][i]), "running")
    ns = int(R["steps"][i])
    last = rec.rows[ns]                                                 # terminal state (= trajectory.csv last row)
    fx, fy, fth = (round(float(last[c][i]), 6) for c in ("x", "y", "yaw"))
    ob_xy = None if last["obst"] is None else last["obst"][i][:, :2]
    kind, _ = SM.collision_type_mem(fx, fy, rec.geometry(i)["walls"], ob_xy,
                                    None if ob_ep is None else ob_ep["active"][i], OBSTACLE_RADIUS)
    nf = SE["n_forward"][i]
    extra = {}
    for s, nm in enumerate(names):
        extra[f"hz_{nm}"] = args.sections[s]
        extra[f"acq_{nm}"] = int(nf[s])
        extra[f"mean_interval_s_{nm}"] = round(SE["last_forward"][i, s] * dtp / (nf[s] - 1), 4) if nf[s] > 1 else ""
        extra[f"mean_obs_age_s_{nm}"] = round(SE["age_sum"][i, s] * dtp / ns, 4) if ns > 0 else ""
        extra[f"max_obs_age_s_{nm}"] = round(SE["age_max"][i, s] * dtp, 4)
    # ray-weighted mean age with integer rays (exactly the common age when the sections are synchronized)
    extra["mean_obs_age_s"] = round(float((SE["age_sum"][i] * np.array(layout.rays)).sum()) / layout.total_rays * dtp / ns, 4)
    extra["max_obs_age_s"] = round(SE["age_max"][i].max() * dtp, 4)
    extra["accepted_scans"], extra["native_scans"] = int(SE["n_any"][i]), int(SE["n_native"][i])
    extra["start_pose_index"] = starts[i][0]
    for j in range(n_obs):
        extra[f"ob{j + 1}_speed_mps"] = round(float(ob_ep["speed"][i][j]), 4)
        extra[f"ob{j + 1}_path_y_m"] = round(float(ob_ep["path_y"][i][j]), 4)
    rows.append(SM.episode_row(
        episode=i, seed=rec.seeds0["episode_seed"][i], hz=args.sections, fse=SE["rays"][i] / layout.total_rays,
        env_outcome=oc, n_steps=ns, nav_return=round(float(R["ret"][i]), 3),
        final_goal_distance_m=round(float(R["final_dist"][i]), 4), distance_traveled_m=round(float(R["path"][i]), 3),
        final_pose=(fx, fy, fth), collision_kind=kind, start=(starts[i][1][0], starts[i][1][1], round(starts[i][1][2], 5)),
        goal=args.goal, frame_cost=args.frame_cost, budget=args.budget, budget_penalty=args.budget_penalty,
        weights=weights, extra=extra))
policy_name = os.path.basename(os.path.normpath(out_dir))

if args.record:                                                         # optional per-episode recordings
    def episode_meta(e, row):
        return {"episode_index": e, "exp_id": args.exp_id, "protocol": args.protocol, "policy": policy_name,
                "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), X.REPO_ROOT),
                "seeds": {"start_pose_seed": args.start_seed + e, "static_obstacles_seed": None, "env_seed": args.seed},
                "goal": {"x": args.goal[0], "y": args.goal[1]}, "deterministic_policy": True,
                "section_gate": {**sensing_cfg, **{k: row[k] for k in fields if k.startswith(("acq_", "hz_", "mean_obs",
                                                                                              "max_obs"))},
                                 "accepted_scans": row["accepted_scans"], "native_scans": row["native_scans"],
                                 "fse": row["episode_frame_count"]},
                **{k: row[k] for k in ("outcome", "success", "n_steps", "nav_return", "final_goal_distance_m",
                                       "distance_traveled_m")}}
    rec.export(out_dir, R["steps"], rows, rec.static_metadata(cfg), episode_meta)
    ED.write_json(os.path.join(out_dir, "world_geometry.json"), rec.fixed_geometry())
SM.write_episodes(os.path.join(out_dir, "episodes.csv"), rows, fields)

succ = np.array([r["success"] for r in rows])
oc_counts = {nm: int(sum(r["outcome"] == nm for r in rows)) for nm in ("success", "collision", "timeout")}
summary_row = SM.summarize(rows, policy_name, names, args.sections, config)
checks_ok = all(v == 0 for k_, v in chk.items() if k_ != "native_intervals_ticks") and \
    set(chk["native_intervals_ticks"]) <= {20, 40}
S_ = {"checkpoint": args.checkpoint, "episodes": n, "goal": list(args.goal), "timeout_s": args.timeout_s,
      "start_seeds": [args.start_seed, args.start_seed + n - 1], "section_hz": list(args.sections),
      "sections": layout.to_dict(), "success_rate": float(succ.mean()), "outcomes": oc_counts,
      "fse_mean_std": [float(np.mean([float(r["episode_frame_count"]) for r in rows])),
                       float(np.std([float(r["episode_frame_count"]) for r in rows]))],
      "checks": {"max_start_placement_err_m": chk_start, "max_goal_err": chk_goal, "sensing": chk, "pass": checks_ok},
      "timing_s": {"launch_and_setup": round(t_setup, 1), "episodes": round(t_eval, 1),
                   "total": round(time.time() - t_start, 1)},
      "summary": summary_row, "tables_schema": "AdaptiveSensing/evaluation/section_metrics.py"}
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(S_, f, indent=1, default=str)
SM.update_summary_csv(os.path.join(os.path.dirname(os.path.normpath(out_dir)), "summary.csv"), summary_row,
                      SM.summary_fields(names))
print("[STS-EVAL]", json.dumps({k: S_[k] for k in ("section_hz", "success_rate", "outcomes", "fse_mean_std", "checks",
                                                   "timing_s")}, default=str), flush=True)
os._exit(0)
