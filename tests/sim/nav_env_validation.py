"""M4 validation of the DirectRLEnv STNav-TB4-Corridor-Static-v0 (LiDAR noise off so observations can be compared
exactly with the analytic oracle; the noise model itself was validated in M2).

T1 spaces + observation layout/normalization      T2 action scaling + actuator integration
T3 reward components and total (independent recomputation from hooked inputs)
T4 termination semantics: goal, safety violation, physical collision (margin 0), timeout = truncation at step 350
T5 partial resets: prev distance / actions / actuator state / fresh LiDAR scan; no artificial progress
T6 numerical stability (random actions)            T7 RSL-RL 5.0.1 compatibility (wrapper + 1-iteration API smoke)

    isaacrun -g auto -- python tests/sim/nav_env_validation.py --headless --num_envs 1024
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "tests", "sim", "results"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from st_nav.geometry.analytic2d import raycast_2d, wall_clearance  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import GOAL, PHYSICAL_COLLISION, SAFETY_VIOLATION, TIMEOUT  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

os.makedirs(args.out, exist_ok=True)
cfg = CorridorNavEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.lidar_noise = False
cfg.seed = 7
genv = gym.make("STNav-TB4-Corridor-Static-v0", cfg=cfg)
env = genv.unwrapped
N, dev = env.num_envs, env.device
ar = torch.arange(N, device=dev)
corr = env.corr
AZ = torch.deg2rad(torch.tensor(corr.lidar_cfg.azimuths_deg, device=dev))
res = {"num_envs": N, "physics_dt": env.physics_dt, "step_dt": env.step_dt, "decimation": cfg.decimation,
       "max_episode_length": env.max_episode_length}
checks = {}


def ok(name, cond, detail=None):
    checks[name] = {"pass": bool(cond), **({"detail": detail} if detail is not None else {})}
    print(f"[CHECK] {'PASS' if cond else 'FAIL'} {name} {detail if detail is not None else ''}", flush=True)


# ---- hook: capture reward inputs and terms inside the step ----
cap = {}
_orig_rewards = env._get_rewards


def hooked_rewards():
    cap["prev"] = env.prev_goal_dist.clone()
    cap["d"] = env._goal_dist.clone()
    cap["c"] = env._clear.clone()
    cap["flags"] = {k: v.clone() for k, v in env._flags.items()}
    p2, _ = corr.robot_pose2d()
    cap["pose"] = p2.clone()
    cap["goal"] = corr.goal.clone()
    cap["walls"] = corr.walls.clone()
    cap["ep_len"] = env.episode_length_buf.clone()
    r = _orig_rewards()
    cap["terms"] = {k: v.clone() for k, v in env.last_reward_terms.items()}
    cap["total"] = r.clone()
    return r


env._get_rewards = hooked_rewards


def oracle_scan_norm():
    org = corr.lidar.origin_w - corr.origins
    _, _, ly = __import__("isaaclab.utils.math", fromlist=["euler_xyz_from_quat"]).euler_xyz_from_quat(corr.lidar.rc.data.quat_w)
    return raycast_2d(org[:, :2], ly[:, None] + AZ[None], corr.walls, None, max_range=3.5) / 3.5


def expected_goal_feats():
    p2, yaw = corr.robot_pose2d()
    g = corr.goal - p2
    d = torch.linalg.norm(g, dim=1)
    a = torch.atan2(g[:, 1], g[:, 0]) - yaw
    return torch.clamp(d / cfg.goal_distance_norm, 0, 1), torch.sin(a), torch.cos(a), d


# ===================================================================== T1
obs, _ = genv.reset()
o = obs["policy"]
ok("T1_obs_shape_47", tuple(o.shape) == (N, 47), list(o.shape))
ok("T1_action_space_2", env.single_action_space.shape == (2,), str(env.single_action_space))
dn, s, c, d0 = expected_goal_feats()
lid_err = (o[:, :40] - oracle_scan_norm()).abs().max().item()
ok("T1_lidar_block_[0:40]_equals_oracle/3.5_at_reset", lid_err < 0.003, f"max abs {lid_err:.2e} (normalized)")
ok("T1_goal_distance_[40]", (o[:, 40] - dn).abs().max().item() < 1e-5)
ok("T1_sin_cos_goal_angle_[41:43]", max((o[:, 41] - s).abs().max().item(), (o[:, 42] - c).abs().max().item()) < 1e-5)
ok("T1_prev_action_[43:45]_zero_at_reset", o[:, 43:45].abs().max().item() == 0.0)
ok("T1_measured_velocity_[45:47]_near_zero_at_reset", o[:, 45:47].abs().max().item() < 0.02, o[:, 45:47].abs().max().item())
ok("T1_lidar_in_[0,1]", bool((o[:, :40] >= 0).all() and (o[:, :40] <= 1).all()))

# ===================================================================== T2
grp = ar % 4
act = torch.zeros(N, 2, device=dev)
act[grp == 0] = torch.tensor([0.5, 0.0], device=dev)        # v = 0.11 m/s
act[grp == 1] = torch.tensor([1.7, 0.0], device=dev)        # clipped to 1.0 -> 0.22 m/s
act[grp == 2] = torch.tensor([0.0, 0.5], device=dev)        # w = 1.0 rad/s (k_w = 0.9 -> ~0.9)
act[grp == 3] = torch.tensor([0.0, -2.0], device=dev)       # clipped -> -2.0 rad/s (k_w -> ~-1.8)
alive = torch.ones(N, dtype=torch.bool, device=dev)
for k in range(12):                                           # 2.4 s
    obs, rew, term, trunc, ext = genv.step(act)
    alive &= ~(term | trunc)
o = obs["policy"]
v_meas, w_meas = o[:, 45] * cfg.v_max, o[:, 46] * cfg.w_max
det = {}
for gi, (ev, ew) in enumerate(((0.11, 0.0), (0.22, 0.0), (0.0, 0.9), (0.0, -1.8))):
    m = (grp == gi) & alive
    det[f"group{gi}"] = {"n_alive": int(m.sum()), "v_mean": float(v_meas[m].mean()), "w_mean": float(w_meas[m].mean()),
                         "expected_v": ev, "expected_w": ew}
ok("T2_clipped_action_stored_as_prev_action", torch.allclose(o[alive, 43:45], torch.clamp(act[alive], -1, 1)), None)
tol = all(abs(det[f"group{g}"]["v_mean"] - det[f"group{g}"]["expected_v"]) < 0.01 and
          abs(det[f"group{g}"]["w_mean"] - det[f"group{g}"]["expected_w"]) < 0.06 for g in range(4))
ok("T2_actuated_velocity_matches_scaled_command", tol, det)

# ===================================================================== T3 + T5 (random rollout with resets)
genv.reset()
rec = []
for k in range(80):
    a = (torch.rand(N, 2, device=dev) * 2 - 1) * 1.2
    obs, rew, term, trunc, ext = genv.step(a)
    rc = cfg.rewards
    t = cap["terms"]
    prog = rc.progress_scale * (cap["prev"] - cap["d"])
    d_ind = torch.linalg.norm(cap["goal"] - cap["pose"], dim=1)
    c_ind = wall_clearance(cap["pose"], cap["walls"]) - corr.cfg.footprint_radius
    prox = -rc.proximity_scale * torch.clamp(1 - c_ind / rc.proximity_clearance, min=0) ** 2
    goal_f = d_ind < cfg.goal_tolerance
    saf_f = (c_ind < cfg.safety_margin) & ~goal_f
    tot = prog + prox - rc.time_penalty_per_s * env.step_dt + rc.goal_reward * goal_f - rc.collision_penalty * saf_f
    done = term | trunc
    o = obs["policy"]
    rec.append(dict(
        d_err=(d_ind - cap["d"]).abs().max().item(), c_err=(c_ind - cap["c"]).abs().max().item(),
        prog_err=(t["progress"] - prog).abs().max().item(), prox_err=(t["proximity"] - prox).abs().max().item(),
        time_ok=bool(torch.allclose(t["time"], torch.full_like(t["time"], -0.02))),
        total_err=(rew - tot).abs().max().item(), sum_err=(sum(t.values()) - rew).abs().max().item(),
        flags_ok=bool(torch.equal(goal_f, cap["flags"]["goal"]) and torch.equal(saf_f, cap["flags"]["safety"])),
        term_ok=bool(torch.equal(term, goal_f | saf_f)),
        terminal_once=bool(((t["goal"] != 0) | (t["collision"] != 0))[~done].sum() == 0),
        n_done=int(done.sum()),
        # T5 checks on envs reset in this step (observation is the new initial state)
        reset_prev_action_zero=bool(o[done, 43:45].abs().max().item() == 0.0) if done.any() else True,
        reset_prev_dist_ok=float((env.prev_goal_dist[done] - corr.goal_distance()[done]).abs().max().item()) if done.any() else 0.0,
        reset_cm_zero=bool(corr.cm.state[done].abs().max().item() == 0.0 and corr.cm._fifo[done].abs().max().item() == 0.0)
        if done.any() else True,
        reset_scan_fresh_err=float((o[done, :40] - oracle_scan_norm()[done]).abs().max().item()) if done.any() else 0.0,
        reset_ep_len_zero=bool((env.episode_length_buf[done] == 0).all()) if done.any() else True,
        progress_bound=float(t["progress"][(cap["ep_len"] == 1)].abs().max().item()) if bool((cap["ep_len"] == 1).any()) else 0.0,
    ))
R = {k: (max(r[k] for r in rec) if isinstance(rec[0][k], float) else all(r[k] for r in rec)) for k in rec[0] if k != "n_done"}
R["episodes_ended"] = sum(r["n_done"] for r in rec)
res["T3_T5_rollout"] = R
ok("T3_goal_distance_matches_independent", R["d_err"] < 1e-5, R["d_err"])
ok("T3_clearance_matches_independent", R["c_err"] < 1e-5, R["c_err"])
ok("T3_progress_term", R["prog_err"] < 1e-4, R["prog_err"])
ok("T3_proximity_term", R["prox_err"] < 1e-4, R["prox_err"])
ok("T3_time_term_-0.02", R["time_ok"])
ok("T3_total_reward_equals_sum_of_terms", R["sum_err"] < 1e-5 and R["total_err"] < 1e-3, [R["sum_err"], R["total_err"]])
ok("T3_flags_and_terminated", R["flags_ok"] and R["term_ok"])
ok("T3_terminal_rewards_only_on_terminal_steps", R["terminal_once"])
ok("T5_reset_prev_action_zero", R["reset_prev_action_zero"])
ok("T5_reset_prev_distance_equals_new_distance", R["reset_prev_dist_ok"] < 1e-5, R["reset_prev_dist_ok"])
ok("T5_reset_actuator_state_zero", R["reset_cm_zero"])
ok("T5_reset_lidar_fresh_scan", R["reset_scan_fresh_err"] < 0.003, R["reset_scan_fresh_err"])
ok("T5_reset_episode_length_zero", R["reset_ep_len_zero"])
max_disp = cfg.v_max * env.step_dt
ok("T5_no_artificial_progress_first_step", R["progress_bound"] <= cfg.rewards.progress_scale * (max_disp + 0.01),
   f"max |progress| on first steps {R['progress_bound']:.4f} <= {cfg.rewards.progress_scale * (max_disp + 0.01):.3f}")

# ===================================================================== T4 terminations
def place(xy, yaw):
    pos = torch.cat([corr.origins[:, :2] + xy, torch.zeros(N, 1, device=dev)], 1)
    q = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], 1)
    corr.robot.write_root_pose_to_sim(torch.cat([pos, q], 1))
    corr.robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))
    corr.robot.write_joint_state_to_sim(torch.zeros(N, 2, device=dev), torch.zeros(N, 2, device=dev), joint_ids=corr.wheel_ids)
    corr.cm.reset()
    env.prev_goal_dist[:] = corr.goal_distance()


zero2 = torch.zeros(N, 2, device=dev)
# goal: 0.10 m from the goal, standing still
genv.reset()
yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
place(corr.goal + 0.10 * torch.stack([torch.cos(yaw), torch.sin(yaw)], 1), yaw)
obs, rew, term, trunc, ext = genv.step(zero2)
ok("T4_goal_terminates_with_+30_once", bool(term.all() and (~trunc).all() and (cap["terms"]["goal"] == 30).all()
                                            and (cap["terms"]["collision"] == 0).all() and (ext["outcome"] == GOAL).all()),
   {"terminated": int(term.sum()), "goal_outcomes": int((ext["outcome"] == GOAL).sum())})
# safety violation: footprint clearance ~0.03 m from the arena's east wall (x = 2.835), facing it, still
genv.reset()
# y in [0.3, 2.3]: between corridor wall C5 (top face y = -0.25) and the north wall, away from the baffles
xy = torch.stack([torch.full((N,), 2.835 - 0.177 - 0.03, device=dev), torch.rand(N, device=dev) * 2.0 + 0.3], 1)
place(xy, torch.zeros(N, device=dev))
obs, rew, term, trunc, ext = genv.step(zero2)
ok("T4_safety_violation_terminates_with_-30", bool(term.all() and (ext["outcome"] == SAFETY_VIOLATION).all()
                                                    and (cap["terms"]["collision"] == -30).all() and (cap["c"] > 0).all()),
   {"terminated": int(term.sum()), "c_range": [float(cap["c"].min()), float(cap["c"].max())]})
# physical collision: margin 0, drive forward into the east wall from 0.15 m
env.cfg.safety_margin = 0.0
genv.reset()
xy[:, 0] = 2.835 - 0.177 - 0.15
place(xy, torch.zeros(N, device=dev))
first_term, outcome_at, c_at = torch.full((N,), -1, device=dev), torch.zeros(N, dtype=torch.long, device=dev), torch.zeros(N, device=dev)
for k in range(15):
    obs, rew, term, trunc, ext = genv.step(torch.tensor([[1.0, 0.0]], device=dev).expand(N, -1))
    new = term & (first_term < 0)
    first_term[new] = k
    outcome_at[new] = ext["outcome"][new]
    c_at[new] = cap["c"][new]
ok("T4_physical_collision_with_margin_0", bool((first_term >= 0).all() and (outcome_at == PHYSICAL_COLLISION).all()
                                               and (c_at <= 0).all()),
   {"terminated": int((first_term >= 0).sum()), "c_at_termination_min_max_m": [float(c_at.min()), float(c_at.max())]})
env.cfg.safety_margin = 0.05
# timeout: stand still at the start for the whole episode
genv.reset()
term_any = torch.zeros(N, dtype=torch.bool, device=dev)
trunc_step = torch.full((N,), -1, device=dev)
rew_sum = torch.zeros(N, device=dev)
for k in range(1, 352):
    obs, rew, term, trunc, ext = genv.step(zero2)
    rew_sum += rew
    term_any |= term
    trunc_step[trunc & (trunc_step < 0)] = k
    if k == 350:
        out350 = ext["outcome"].clone()
        tl = {kk: v.clone() for kk, v in cap["terms"].items()}
ok("T4_timeout_is_truncation_at_step_350", bool((~term_any).all() and (trunc_step == 350).all() and (out350 == TIMEOUT).all()
                                                and (tl["goal"] == 0).all() and (tl["collision"] == 0).all()),
   {"truncated_at": torch.unique(trunc_step).tolist(), "terminated_any": int(term_any.sum())})

# ===================================================================== T6 stability
genv.reset()
bad = 0
lo, hi = torch.full((47,), 1e9, device=dev), torch.full((47,), -1e9, device=dev)
rmin, rmax = 1e9, -1e9
for k in range(200):
    obs, rew, term, trunc, ext = genv.step(torch.randn(N, 2, device=dev) * 2)
    o = obs["policy"]
    bad += int((~torch.isfinite(o)).sum() + (~torch.isfinite(rew)).sum())
    lo, hi = torch.minimum(lo, o.min(0).values), torch.maximum(hi, o.max(0).values)
    rmin, rmax = min(rmin, rew.min().item()), max(rmax, rew.max().item())
res["T6_ranges"] = {"obs_min": lo.tolist(), "obs_max": hi.tolist(), "reward_min": rmin, "reward_max": rmax}
ok("T6_finite_obs_and_rewards", bad == 0, bad)
ok("T6_obs_ranges", bool(lo[:43].min() >= -1 - 1e-6 and hi[:43].max() <= 1 + 1e-6 and lo[43:45].min() >= -1 and hi[43:45].max() <= 1
                         and lo[45:].min() > -2 and hi[45:].max() < 2),
   {"lidar/goal": [float(lo[:43].min()), float(hi[:43].max())], "vel": [float(lo[45:].min()), float(hi[45:].max())]})

# ===================================================================== T7 RSL-RL
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import rsl_rl  # noqa: E402

from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cfg import CorridorNavPPORunnerCfg  # noqa: E402

agent = CorridorNavPPORunnerCfg()
agent.num_steps_per_env = 8
agent.device = dev
wrapped = RslRlVecEnvWrapper(genv, clip_actions=agent.clip_actions)
td = wrapped.get_observations()
try:
    from importlib.metadata import version
    rv = version("rsl-rl-lib")
except Exception:
    rv = "unknown"
agent = handle_deprecated_rsl_rl_cfg(agent, rv)          # same conversion as Isaac Lab's rsl_rl/train.py
runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=dev)
runner.learn(num_learning_iterations=1)
ok("T7_rsl_rl_wrapper_and_one_iteration_api_smoke", True, {"rsl_rl_version": rv, "obs_keys": list(td.keys()),
                                                            "policy_obs_shape": list(td["policy"].shape)})

res["checks"] = checks
res["all_pass"] = all(v["pass"] for v in checks.values())
with open(os.path.join(args.out, "m4_validation.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("[DONE] all_pass =", res["all_pass"], flush=True)
os._exit(0)
