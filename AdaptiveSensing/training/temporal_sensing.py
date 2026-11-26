"""Temporal adaptive sensing around the unchanged Isaac Lab navigation env (TMP02): frozen NAV05 + real-robot scan
gate + sensing reward, vectorized over the envs; the sensing policy acts through `step(action)`.

Reference: adaptive_sensor_policy@dc5b53f envs/turtlebot_var_scan_rate.py `step` (1356), `_advance_with_native_scans`
(1114), `_set_target_fps` (1034), `_get_augmented_obs` (749), `reset` (1214); real robot: Sim2Real
real_eval_adaptive.py control loop (944-1027) and `set_target_fps` (613).

Order of one control tick k at time t_k (the native scan rendered at t_k is in the env observation returned by the
previous env.step; it is gated at the START of tick k, by the action selected at t_k):

    s_k = (held scan BEFORE gating the t_k scan, fps / max(fps), clip(steps_since / 25), clip(frames / budget))
    a_k = sensing policy(s_k)                                    (outside; LSTM steps every tick)
    steps_since += 1
    gate the t_k scan (new native scan and t_k >= next_sample_time, rate in force) -> fwd_k
        on fwd_k: held <- scan(t_k); deadline advanced by whole periods; set_target_fps(fps[a_k]) (a changed rate
                  re-anchors: next = t_k + 1 / fps); steps_since = 0; frames += 1
        otherwise a_k is discarded
    NAV05({"policy": current navigation features, "lidar": held})   (frozen; most recent accepted scan, as TMP01)
    env.step (20 physics steps; native 4 Hz scheduler unchanged)
    r_k = r_env - frame_cost * fwd_k - budget_penalty * [first tick with frames > budget];   mask_k = fwd_k
    done envs: episode closed with the counters above; gate.reset (reset scan forwarded, rate = reset rate,
               steps_since = 0, frames = 1, not charged): the reset observation belongs to the next episode only

Terminal step: the last tick gates its own t_k scan like any other tick (frame counted, action applied, cost
charged). The scan the env would render at the terminal instant itself is never gated: there is no decision after the
episode end (Isaac Lab returns the next episode's reset observation there). The reference native-LiDAR simulator
gated scans in (t_start, t_end] of every step, so a scan arriving inside the last step was counted there; here every
scan sits on a control instant and is gated by the decision taken at that instant.
"""
from __future__ import annotations

import torch

from AdaptiveSensing.evaluation.scan_gate import ScanGate, period_ticks

EP_KEYS = ("return", "nav_return", "frame_cost_sum", "budget_penalty_sum", "length", "frames", "native_scans",
           "causal", "fps_inforce_sum", "fps_selected_sum", "fps_applied_sum", "age_ticks_sum", "age_ticks_max",
           "budget_exceeded", "outcome", "time_out")
NAV_CONTRACT = ("sensor", "obs_layout", "lidar3d", "goal_distance_norm", "v_max", "w_max", "decimation", "sim.dt")


def _get(cfg, path):
    for p in path.split("."):
        cfg = getattr(cfg, p)
    return cfg.to_dict() if hasattr(cfg, "to_dict") else cfg


def nav_contract_mismatches(train_cfg, nav_cfg) -> list[str]:
    """Env settings that define NAV05's input (sensor, image layout, normalizations, timing) that differ."""
    return [k for k in NAV_CONTRACT if _get(train_cfg, k) != _get(nav_cfg, k)]


def nav_run_cfgs(checkpoint: str, device):
    """Settings of the navigation checkpoint's training run, resolved as in TMP01 / evaluate_3d.py: (RSL-RL agent cfg,
    run config dict, env cfg of the run). The agent cfg also gives the env wrapper's clip_actions."""
    import os
    from importlib.metadata import version

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cnn_cfg import CorridorNavCNNPPORunnerCfg
    from st_nav.tasks.nav2d.corridor_env_lidar3d_cfg import CorridorNavLidar3DEnvCfg
    from tools import experiments as X

    run_dir = os.path.dirname(os.path.abspath(checkpoint))
    run_exp = X.yaml.safe_load(open(os.path.join(run_dir, "params", "config.yaml")))
    run_env = {**X.flatten(run_exp.get("env") or {}), **((run_exp.get("set") or {}).get("env") or {})}
    run_agent = {**X.flatten(run_exp.get("agent") or {}), **((run_exp.get("set") or {}).get("agent") or {})}
    nav_env_cfg = CorridorNavLidar3DEnvCfg()
    X.apply_overrides(nav_env_cfg, run_env, "env")
    agent = CorridorNavCNNPPORunnerCfg()
    X.apply_overrides(agent, {k: v for k, v in run_agent.items() if k not in ("seed", "max_iterations")}, "agent")
    agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
    agent.device = str(device)
    return agent, run_exp, nav_env_cfg


def load_nav_policy(env, agent_cfg, checkpoint: str, device):
    """Frozen navigation policy (OnPolicyRunner.load + get_inference_policy, as TMP01); no parameter needs grad."""
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
    runner.load(checkpoint, map_location=str(device))
    policy = runner.get_inference_policy(device=str(device))
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    runner.alg.storage = None                     # NAV05's rollout storage is never used
    return policy


class TemporalSensing:
    def __init__(self, env, nav_policy, fps_choices, frame_cost: float, budget: int, budget_penalty: float,
                 reset_hz: float | None = None):
        """env: RslRlVecEnvWrapper of the navigation env. reset_hz: rate after every reset (default max(fps_choices);
        fixed-rate validation: the fixed rate)."""
        self.env, self.u, self.nav = env, env.unwrapped, nav_policy
        u = self.u
        self.n, self.dev = u.num_envs, u.device
        self.fps_choices = [float(f) for f in fps_choices]
        self.max_fps, self.min_fps = max(self.fps_choices), min(self.fps_choices)
        self.frame_cost, self.budget, self.budget_penalty = float(frame_cost), int(budget), float(budget_penalty)
        self.control_hz = 1.0 / u.step_dt
        self.max_obs_interval = self.control_hz / self.min_fps           # reference: control_frequency / min(fps)
        self.reset_hz = self.max_fps if reset_hz is None else float(reset_hz)
        self.periods = torch.tensor([period_ticks(f, u.physics_dt) for f in self.fps_choices], device=self.dev)
        self.fps_t = torch.tensor(self.fps_choices, device=self.dev)
        self.gate = ScanGate(self.n, self.reset_hz, u.physics_dt, device=self.dev)
        z = lambda dt=torch.float32: torch.zeros(self.n, dtype=dt, device=self.dev)  # noqa: E731
        self.steps_since = z(torch.int64)
        self.budget_applied = z(torch.bool)
        self.acc = {k: z(torch.int64 if k == "age_ticks_max" else torch.float32) for k in EP_KEYS}
        self.obs = None

    # ------------------------------------------------------------------------------------------------ observations
    def sensing_obs(self):
        """(image (N, 1, H, W), ratios (N, 3)) from the held scan BEFORE the scan of the current instant is gated."""
        g = self.gate
        fps_ratio = (g.hz_in_force() / self.max_fps).float()
        age_ratio = torch.clamp(self.steps_since.float() / self.max_obs_interval, 0.0, 1.0)
        frame_ratio = torch.clamp(g.n_forward.float() / self.budget, 0.0, 1.0)
        return g.last_obs, torch.stack([fps_ratio, age_ratio, frame_ratio], dim=1)

    def reset(self):
        self.obs = self.env.get_observations()
        allm = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        self.gate.reset(allm, self.obs["lidar"], self.u.scan_t)
        self.steps_since.zero_()
        self.budget_applied.zero_()
        for v in self.acc.values():
            v.zero_()
        return self.sensing_obs()

    # ------------------------------------------------------------------------------------------------ one tick
    def step(self, action: torch.Tensor):
        """One control tick: gate the scan of this instant with `action`, then navigate and step the env."""
        self.gate_step(action)
        return self.advance()

    def gate_step(self, action: torch.Tensor) -> torch.Tensor:
        """First half of a tick: the sensing action selected at this instant gates the native scan of this instant
        (accept / discard, set_target_fps on acceptance). Returns the accepted mask."""
        g, u = self.gate, self.u
        a = action.long()
        self.steps_since += 1
        new, fwd = g.step(self.obs["lidar"], u.scan_t)                    # the scan of this instant
        g.set_rate(fwd, self.periods[a])                                  # effective only on an accepted scan
        self.steps_since = torch.where(fwd, torch.zeros_like(self.steps_since), self.steps_since)
        first = (g.n_forward > self.budget) & ~self.budget_applied
        self.budget_applied |= first
        self._tick = (a, new, fwd, first)
        return fwd

    def advance(self):
        """Second half of a tick: NAV05 on the held scan, env step, sensing reward, episode bookkeeping and resets."""
        g, u = self.gate, self.u
        a, new, fwd, first = self._tick
        age = g.ticks(u.ep_time) - g.last_forward                         # age of the held scan NAV05 receives
        nav_obs = self.obs.clone(recurse=False)
        nav_obs["lidar"] = g.last_obs
        with torch.no_grad():
            nav_act = self.nav(nav_obs)
        obs, r_env, dones, extras = self.env.step(nav_act)
        fwd_f, first_f = fwd.float(), first.float()
        reward = r_env - self.frame_cost * fwd_f - self.budget_penalty * first_f
        d = dones.bool()
        A = self.acc
        A["return"] += reward
        A["nav_return"] += r_env
        A["frame_cost_sum"] += self.frame_cost * fwd_f
        A["budget_penalty_sum"] += self.budget_penalty * first_f
        A["length"] += 1
        A["causal"] += fwd_f
        A["fps_inforce_sum"] += g.hz_in_force().float()                  # reference mean_fps: rate in force per tick
        A["fps_selected_sum"] += self.fps_t[a]
        A["fps_applied_sum"] += self.fps_t[a] * fwd_f
        A["age_ticks_sum"] += age.float()
        A["age_ticks_max"] = torch.maximum(A["age_ticks_max"], age)
        A["frames"] = g.n_forward.float()
        A["native_scans"] = g.n_native.float()
        A["budget_exceeded"] = self.budget_applied.float()
        A["outcome"] = extras["outcome"].float() if "outcome" in extras else torch.zeros_like(r_env)
        A["time_out"] = extras["time_outs"].float() if "time_outs" in extras else torch.zeros_like(r_env)
        episode = {k: v.clone() for k, v in A.items()}                    # valid where done
        gate_end = {"last_forward": g.last_forward.clone(), "int_min": g.int_min.clone(), "int_max": g.int_max.clone()}
        info = {"mask": fwd_f, "new": new, "budget_first": first, "r_env": r_env, "episode": episode,
                "fps_in_force": g.hz_in_force().float(), "age_ticks": age, "gate_end": gate_end,
                "nav_action": nav_act, "dones": dones, "extras": extras}
        # episode ends: the reset observation starts the next episode (frame 1, reset rate, not charged)
        g.reset(d, obs["lidar"], u.scan_t)
        self.steps_since = torch.where(d, torch.zeros_like(self.steps_since), self.steps_since)
        self.budget_applied &= ~d
        for v in A.values():
            v.masked_fill_(d, 0)
        self.obs = obs
        return self.sensing_obs(), reward, d, info
