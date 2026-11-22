"""M4 follow-up diagnostics (requested after the M4 review). Nothing is trained or saved except diagnostics output.

D1  default configuration actually used: randomization, start poses, goal distribution
D2  benchmark outcome: deterministic (inference) vs stochastic policy; actions, trajectories, termination counters
D3  measured linear velocity > v_max: per-physics-substep decomposition (CoM vs link velocity, wheel odometry,
    finite-difference motion, command-model output, contact force, steps since reset)
D4  timeout bootstrapping with the installed RSL-RL (RslRlVecEnvWrapper + PPO.process_env_step)

    isaacrun -g 5 -- python scripts/m4/m4_diagnostics.py --headless --num_envs 1024
"""
import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--out_dir", default=os.path.join(os.environ.get("ST_ROOT", "."), "results", "m4", "diag"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from st_nav.geometry.analytic2d import wall_clearance  # noqa: E402
from st_nav.scenes import corridor_static_random as S  # noqa: E402
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cfg import CorridorNavPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import OUTCOME_NAMES  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

os.makedirs(args.out_dir, exist_ok=True)
cfg = CorridorNavEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = 11                                            # same seed as the M4 benchmark
t0 = time.time()
genv = gym.make("STNav-TB4-Corridor-Static-v0", cfg=cfg)
agent = handle_deprecated_rsl_rl_cfg(CorridorNavPPORunnerCfg(), version("rsl-rl-lib"))
agent.device = args.device
env = RslRlVecEnvWrapper(genv, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
policy = runner.get_inference_policy(device=args.device)
_IM = torch.inference_mode(); _IM.__enter__()                     # as in OnPolicyRunner rollouts (all env I/O below)
u = genv.unwrapped
corr, rd, N, dev = u.corr, u.corr.robot.data, u.num_envs, u.device
print(f"[DIAG] setup {time.time() - t0:.0f} s, N={N}", flush=True)
R = {"num_envs": N, "rsl_rl_version": version("rsl-rl-lib")}
NW = corr.walls.shape[1]
NF = len(S.FIXED_WALLS)


def per_wall_clearance():
    pp, _ = corr.robot_pose2d()
    return torch.stack([wall_clearance(pp, corr.walls[:, i:i + 1]) for i in range(NW)], 1) - corr.cfg.footprint_radius


def wall_name(i):
    return f"fixed{i}" if i < NF else f"Z{i - NF + 1}"


# ------------------------------------------------------------------ hooks: terminal information before reset
_orig_dones = u._get_dones
term_info = {}


def _dones_hook():
    out = _orig_dones()
    term_info["wall"] = per_wall_clearance().argmin(1)
    term_info["pose"] = torch.cat(corr.robot_pose2d()[0:1] + (corr.robot_pose2d()[1][:, None],), 1).clone()
    term_info["len"] = u.episode_length_buf.clone() + 0          # value at the end of this step (pre-reset)
    return out


u._get_dones = _dones_hook

# ================================================================== D1 configuration and randomization
gh = {}
starts = set()
z_off_samples, fb0 = [], corr.offset_fallbacks


def record_reset_stats(ids):
    for g in corr.goal[ids].tolist():
        k = tuple(round(x, 2) for x in g)
        gh[k] = gh.get(k, 0) + 1
    for s in corr.start[ids].tolist():
        starts.add(tuple(round(x, 4) for x in s))
    z_off_samples.append(corr.z_off[ids].clone())


record_reset_stats(torch.arange(N, device=dev))
R["D1_config"] = {
    "start_pose_dr": cfg.start_pose_dr, "bank_size": cfg.bank_size, "distinct_bank_entries_used": int(corr.bank_idx.unique().numel()),
    "goals": [list(g) for g in corr.cfg.goals], "goal_tolerance": cfg.goal_tolerance, "safety_margin": cfg.safety_margin,
    "goal_distance_norm": cfg.goal_distance_norm, "lidar_noise": cfg.lidar_noise, "lidar_period_s": cfg.lidar_period_s,
    "episode_steps": int(u.max_episode_length), "v_max": cfg.v_max, "w_max": cfg.w_max,
    "default_start": [*S.STARTING_POS, S.STARTING_YAW],
}
d0 = corr.goal_distance()
R["D1_config"]["start_goal_distance_m"] = [float(d0.min()), float(d0.max())]
R["D1_config"]["obs40_at_reset_range"] = [float(d0.min() / cfg.goal_distance_norm), float(d0.max() / cfg.goal_distance_norm)]

# ================================================================== D2a deterministic inference policy (as benchmark)
def rollout(act_fn, steps, tag):
    obs = env.get_observations()
    first_actions, act_std, pos_std, act_mean = None, [], [], []
    outcomes = {n: 0 for n in OUTCOME_NAMES.values()}
    per_step_ends, lens, walls = [], [], {}
    traj = []
    for k in range(steps):
        with torch.inference_mode():
            a = act_fn(obs)
        ac = torch.clamp(a, -1, 1)
        if first_actions is None:
            first_actions = ac[:8].tolist()
        act_mean.append(ac.mean(0).tolist())
        act_std.append(ac.std(0).tolist())
        pp, _ = corr.robot_pose2d()
        pos_std.append(pp.std(0).tolist())
        traj.append(pp[:64].cpu().numpy().copy())
        with torch.inference_mode():
            obs, rew, dones, ext = env.step(a)
        dmask = dones.bool()
        per_step_ends.append(int(dmask.sum()))
        if dmask.any():
            oc = ext["outcome"][dmask]
            for code, name in OUTCOME_NAMES.items():
                outcomes[name] += int((oc == code).sum())
            lens += term_info["len"][dmask].tolist()
            for w in term_info["wall"][dmask].tolist():
                walls[wall_name(w)] = walls.get(wall_name(w), 0) + 1
            record_reset_stats(dmask.nonzero().squeeze(1))
    ends = [(k + 1, c) for k, c in enumerate(per_step_ends) if c]
    lens_t = torch.tensor(lens, dtype=torch.float32) if lens else torch.zeros(1)
    res = {"steps": steps, "outcomes": outcomes, "first_actions_env0_7": first_actions,
           "action_mean_step0": act_mean[0], "action_std_across_envs_step0": act_std[0],
           "action_std_across_envs_max_over_rollout": np.max(np.array(act_std), 0).tolist(),
           "position_std_across_envs_m_at_steps_[0,10,30,49]": [pos_std[i] for i in (0, 10, 30, 49) if i < steps],
           "termination_steps_(step, count)_first12": ends[:12], "episode_length_unique": sorted(set(int(x) for x in lens))[:20],
           "episode_length_mean_std": [float(lens_t.mean()), float(lens_t.std()) if len(lens) > 1 else 0.0],
           "terminal_nearest_wall": walls}
    np.save(os.path.join(args.out_dir, f"traj_{tag}.npy"), np.stack(traj))
    return res


genv.reset()
R["D2a_deterministic_inference_policy"] = rollout(policy, 120, "deterministic")
genv.reset()
R["D2b_stochastic_policy_alg_act"] = rollout(lambda o: runner.alg.act(o), 120, "stochastic")
runner.alg.transition.clear()
offs = torch.cat(z_off_samples)
R["D1_config"]["resets_sampled"] = int(offs.shape[0])
R["D1_config"]["goal_histogram"] = {str(k): v for k, v in sorted(gh.items())}
R["D1_config"]["distinct_start_poses"] = [list(s) for s in sorted(starts)][:10]
R["D1_config"]["z_offset_std_per_Z_over_resets"] = offs.reshape(offs.shape[0], -1).std(0).tolist()
R["D1_config"]["offset_fallbacks"] = int(corr.offset_fallbacks - fb0)
print("[DIAG] D1/D2 done", flush=True)

# ================================================================== D3 velocity decomposition per physics substep
genv.reset()
G = N // 4
grp = torch.arange(N, device=dev) // G
fixed_act = torch.tensor([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=torch.float32, device=dev)[grp.clamp(max=3)]
names = ["v_com_b", "v_link_b", "w_z", "v_wheel", "v_fd", "v_cmd_eff", "contact_xy", "since_reset", "lat_link_b"]
rec = {n: [] for n in names}
prev = {"pos": rd.root_link_pos_w[:, :2].clone(), "len": u.episode_length_buf.clone()}
r_w = corr.cm.geo.wheel_radius
_orig_apply = u._apply_action


def snap():
    pos = rd.root_link_pos_w[:, :2]
    _, yaw = corr.robot_pose2d()
    hd = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
    v_fd = ((pos - prev["pos"]) * hd).sum(1) / corr.dt
    v_fd[u.episode_length_buf < prev["len"]] = float("nan")            # reset between samples: no finite difference
    jv = rd.joint_vel[:, corr.wheel_ids]
    vals = [rd.root_com_lin_vel_b[:, 0], rd.root_link_lin_vel_b[:, 0], rd.root_com_ang_vel_b[:, 2],
            r_w * jv.mean(1), v_fd, corr.cm.state[:, 0], corr.contact_force_xy(), u.episode_length_buf.float(),
            rd.root_link_lin_vel_b[:, 1]]
    for n, v in zip(names, vals):
        rec[n].append(v.clone())
    prev["pos"], prev["len"] = pos.clone(), u.episode_length_buf.clone()


def _apply_hook():
    snap()                                     # state after the previous physics substep
    _orig_apply()


u._apply_action = _apply_hook
obs_v = []
obs = env.get_observations()
for k in range(150):
    a = fixed_act.clone()
    a[grp == 0] = torch.clamp(torch.randn(int((grp == 0).sum()), 2, device=dev) * 2, -1, 1)   # T6-style actions
    with torch.inference_mode():
        obs, rew, dones, ext = env.step(a)
    obs_v.append(torch.stack([obs["policy"][:, 45] * cfg.v_max, u.episode_length_buf.float(),
                              rd.root_link_lin_vel_b[:, 0], rd.root_com_ang_vel_b[:, 2]], 1))
u._apply_action = _orig_apply
T = {n: torch.stack(v) for n, v in rec.items()}             # (substeps, N)
OV = torch.stack(obs_v)                                     # (control steps, N, 4)

com_b = rd.body_com_pos_b[0, 0].tolist() if hasattr(rd, "body_com_pos_b") else None
thr = cfg.v_max * 1.02
d3 = {"root_body": corr.robot.body_names[0], "root_body_com_offset_in_link_frame_m": com_b,
      "threshold_v": thr, "group_actions": ["random clip(2*randn)", "(1,1)", "(1,0)", "(0,1)"]}
for n in ("v_com_b", "v_link_b", "v_wheel", "v_fd", "v_cmd_eff"):
    x = T[n][torch.isfinite(T[n])]
    d3[f"max_{n}"] = float(x.max())
d3["obs45_times_vmax_max"] = float(OV[..., 0].max())
d3["obs45_max_per_group"] = [float(OV[:, grp == g, 0].max()) for g in range(4)]
d3["obs_link_v_max_per_group"] = [float(OV[:, grp == g, 2].max()) for g in range(4)]
# lever-arm model: v_com_x = v_link_x - w * r_y  (+ w * r_x in y)
if com_b is not None:
    pred = T["v_link_b"] - T["w_z"] * com_b[1]
    d3["com_lever_arm_model_max_abs_err"] = float((pred - T["v_com_b"]).abs().max())
    d3["com_minus_link_max"] = float((T["v_com_b"] - T["v_link_b"]).max())
ev = T["v_com_b"] > thr
d3["substep_samples_over_threshold"] = int(ev.sum())
if ev.any():
    def frac(m):
        return float((m & ev).sum() / ev.sum())
    d3["of_those_fraction"] = {
        "link_velocity_also_over": frac(T["v_link_b"] > thr),
        "wheel_odometry_also_over": frac(T["v_wheel"] > thr),
        "finite_difference_also_over": frac(torch.nan_to_num(T["v_fd"]) > thr),
        "command_model_output_over": frac(T["v_cmd_eff"] > thr),
        "contact_force_gt_1N": frac(T["contact_xy"] > 1.0),
        "within_2_steps_of_reset": frac(T["since_reset"] <= 2),
        "abs_w_gt_1": frac(T["w_z"].abs() > 1.0),
    }
    d3["over_threshold_by_group"] = [int(ev[:, grp == g].sum()) for g in range(4)]
evl = T["v_link_b"] > thr
d3["link_over_threshold_samples"] = int(evl.sum())
if evl.any():
    d3["link_over_with_contact_gt_1N"] = float(((T["contact_xy"] > 1.0) & evl).sum() / evl.sum())
    d3["link_over_wheel_over"] = float(((T["v_wheel"] > thr) & evl).sum() / evl.sum())
# steady state for (1, 1): v and w after 3 s, free of contact
ss = (grp == 1)[None] & (T["since_reset"] >= 15) & (T["contact_xy"] < 1e-3)
if ss.any():
    d3["group(1,1)_steady_means"] = {n: float(T[n][ss].mean()) for n in ("v_com_b", "v_link_b", "w_z", "v_wheel", "v_cmd_eff")}
    d3["group(1,1)_steady_link_lateral_v"] = float(T["lat_link_b"][ss].mean())
R["D3_velocity"] = d3
e = int(T["v_com_b"].max(0).values.argmax())
np.savez(os.path.join(args.out_dir, "velocity_traces.npz"), env=e, group=int(grp[e]), dt=corr.dt,
         **{n: T[n][:, e].cpu().numpy() for n in names},
         **{f"g1_{n}": T[n][:, int((grp == 1).nonzero()[0])].cpu().numpy() for n in names},
         scatter_w=T["w_z"].flatten()[::7].cpu().numpy(), scatter_diff=(T["v_com_b"] - T["v_link_b"]).flatten()[::7].cpu().numpy())
print("[DIAG] D3 done", flush=True)

# ================================================================== D4 timeout bootstrapping (installed RSL-RL)
genv.reset()
u.cfg.episode_length_s = 8.0                               # 40-step episodes for this test only
L = int(u.max_episode_length)
genv.reset()
alg, st = runner.alg, runner.alg.storage
st.clear()
half = torch.arange(N, device=dev) < N // 2               # first half: stand still (time out); second: drive (1, 0)
act = torch.zeros(N, 2, device=dev)
act[~half, 0] = 1.0
obs = env.get_observations()
chk = {"is_finite_horizon": bool(u.cfg.is_finite_horizon), "episode_steps": L, "gamma": alg.gamma,
       "timeouts": 0, "terminations": 0, "timeout_and_terminated_overlap": 0, "time_outs_equals_truncated": True,
       "dones_equals_term_or_trunc": True, "max_err_timeout_bootstrap": 0.0, "max_err_non_timeout_unchanged": 0.0,
       "timeout_step_values": [], "first_termination_step": None}
_orig_step = u.step
last = {}


def _step_hook(a):
    out = _orig_step(a)
    last["term"], last["trunc"] = out[2].clone(), out[3].clone()
    return out


u.step = _step_hook
for k in range(L + 5):
    if st.step >= st.num_transitions_per_env:
        st.clear()
    alg.act(obs)
    v_before = alg.transition.values.squeeze(1).clone()
    obs, rew, dones, ext = env.step(act)
    to = ext["time_outs"].bool() if "time_outs" in ext else torch.zeros(N, dtype=torch.bool, device=dev)
    alg.process_env_step(obs, rew, dones, ext)
    stored = st.rewards[st.step - 1, :, 0]
    term, trunc = last["term"].bool(), last["trunc"].bool()
    chk["time_outs_equals_truncated"] &= bool((to == trunc).all())
    chk["dones_equals_term_or_trunc"] &= bool((dones.bool() == (term | trunc)).all())
    chk["timeout_and_terminated_overlap"] += int((term & trunc).sum())
    chk["timeouts"] += int(to.sum())
    chk["timeouts_in_driving_half"] = chk.get("timeouts_in_driving_half", 0) + int((to & ~half).sum())
    chk["terminations_in_standing_half"] = chk.get("terminations_in_standing_half", 0) + int((term & half).sum())
    chk["terminations"] += int(term.sum())
    if term.any() and chk["first_termination_step"] is None:
        chk["first_termination_step"] = k + 1
    if to.any():
        chk["timeout_step_values"].append([k + 1, int(to.sum())])
        chk["max_err_timeout_bootstrap"] = max(chk["max_err_timeout_bootstrap"],
                                               float((stored[to] - (rew[to] + alg.gamma * v_before[to])).abs().max()))
    if (~to).any():
        chk["max_err_non_timeout_unchanged"] = max(chk["max_err_non_timeout_unchanged"], float((stored[~to] - rew[~to]).abs().max()))
u.step = _orig_step
st.clear()
R["D4_timeout_bootstrap"] = chk
print("[DIAG] D4 done", flush=True)

with open(os.path.join(args.out_dir, "m4_diagnostics.json"), "w") as f:
    json.dump(R, f, indent=1, default=str)
print("[DIAG-JSON]", json.dumps(R, default=str), flush=True)
os._exit(0)
