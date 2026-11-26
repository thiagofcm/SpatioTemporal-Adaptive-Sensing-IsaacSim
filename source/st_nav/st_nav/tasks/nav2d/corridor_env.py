"""DirectRLEnv: articulated TurtleBot4 navigation in the corridor scenes with a ray-casting LiDAR (2D or 3D).

Simulation layer: st_nav.envs.corridor_scene.CorridorScene (validated in M3), used in its deferred mode:
scene config -> prepare_stage() in _setup_scene() -> attach() after the base class started physics.

Step (DirectRLEnv order): _pre_physics_step -> decimation x (_apply_action, physics) -> _get_dones -> _get_rewards ->
_reset_idx (finished envs) -> _get_observations. Dones and rewards therefore use the post-step state of THIS control
step; observations of reset envs are those of their new initial state.

Observation (R + 7, R = rays of the sensor: 2D LiDAR 40 -> 47; 3D LiDAR 139 x 34 = 4,726 -> 4,733):
[0:R] LiDAR ranges / range_max (latest scan, noisy, sensor ray order) | [R] goal distance / goal_distance_norm (clipped
[0, 1]) | [R+1] sin(goal angle) | [R+2] cos(goal angle) | [R+3:R+5] previous action (v, w) in [-1, 1] |
[45] base_link v / v_max | [46] base_link w / w_max         (goal angle = bearing to goal in the robot frame)
Rewards and terminations use ground-truth geometry only (footprint clearance, goal distance), never the LiDAR.
"""
from __future__ import annotations

import math

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

from st_nav.envs.corridor_scene import CorridorScene, CorridorSceneCfg
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg

# outcome codes (per env, set on the step an episode ends)
RUNNING, GOAL, PHYSICAL_COLLISION, SAFETY_VIOLATION, TIMEOUT = 0, 1, 2, 3, 4
OUTCOME_NAMES = {GOAL: "goal", PHYSICAL_COLLISION: "physical_collision", SAFETY_VIOLATION: "safety_violation",
                 TIMEOUT: "timeout"}
REWARD_TERMS = ("progress", "proximity", "time", "goal", "collision", "timeout")


class CorridorNavEnv(DirectRLEnv):
    cfg: CorridorNavEnvCfg

    def __init__(self, cfg: CorridorNavEnvCfg, render_mode: str | None = None, **kwargs):
        scfg = CorridorSceneCfg(num_envs=cfg.scene.num_envs, physics_hz=1.0 / cfg.sim.dt, decimation=cfg.decimation,
                                env_spacing=cfg.scene.env_spacing, bank_size=cfg.bank_size,
                                start_pose_dr=cfg.start_pose_dr, safety_margin=cfg.safety_margin,
                                seed=cfg.seed if cfg.seed is not None else 0, bank_seed=cfg.bank_seed,
                                bank_assignment=cfg.bank_assignment, offsets_from_bank=cfg.offsets_from_bank,
                                start_poses=cfg.start_poses, scene_name=cfg.scene_name,
                                active_obstacles=tuple(cfg.active_obstacles), obstacle_presence_dr=cfg.obstacle_presence_dr,
                                obstacle_speed_dr=cfg.obstacle_speed_dr, obstacle_y_dr=cfg.obstacle_y_dr,
                                obstacle_radius_dr=cfg.obstacle_radius_dr, max_episode_s=cfg.episode_length_s,
                                fast_path=tuple(cfg.fast_path))
        if cfg.goals is not None:
            scfg.goals = tuple(tuple(g) for g in cfg.goals)
        scfg.wall_height, scfg.baffle_height, scfg.obstacle_height = cfg.wall_height, cfg.baffle_height, cfg.obstacle_height
        if cfg.sensor == "lidar3d":
            scfg.lidar = cfg.lidar3d.to_ray_lidar_cfg()
            scfg.lidar.noise.enabled = scfg.lidar.noise.enabled and cfg.lidar_noise
            self.scan_period = float(cfg.lidar3d.period_s)
        elif cfg.sensor == "lidar2d":
            scfg.lidar.noise.enabled = cfg.lidar_noise
            self.scan_period = float(cfg.lidar_period_s)
        else:
            raise ValueError(f"unknown sensor {cfg.sensor!r} (lidar2d | lidar3d)")
        cfg.observation_space = scfg.lidar.num_rays + 7            # [scan | goal (3) | previous action (2) | v, w]
        if cfg.obs_layout == "image":                               # {"policy": 7 nav features, "lidar": range image}
            lc = scfg.lidar
            if lc.grid_shape is None or lc.grid_row is None or lc.grid_col is None:
                raise ValueError("obs_layout='image' needs a grid ray set (lidar3d grid or a ray table with grid_row/col)")
            if cfg.privileged_critic:
                raise NotImplementedError("privileged_critic is implemented for obs_layout='flat' only")
            self.img_shape = (1, int(lc.grid_shape[0]), int(lc.grid_shape[1]))
            self._img_idx = [int(r) * self.img_shape[2] + int(c) for r, c in zip(lc.grid_row, lc.grid_col)]
            if len(set(self._img_idx)) != lc.num_rays:
                raise ValueError("obs_layout='image': two rays map to the same range-image cell")
            cfg.observation_space = 7                               # the "policy" group; the image is the "lidar" group
        elif cfg.obs_layout != "flat":
            raise ValueError(f"unknown obs_layout {cfg.obs_layout!r} (flat | image)")
        self.corr = CorridorScene.deferred(scfg, cfg.sim.device)
        self.corr.build_scene_cfg(cfg.scene)
        super().__init__(cfg, render_mode, **kwargs)
        self.corr.attach(self.scene, self.sim)
        n, dev = self.num_envs, self.device
        if cfg.obs_layout == "image":
            self._img_idx = torch.tensor(self._img_idx, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 2, device=dev)          # last applied (clipped) action, = previous action obs
        self.prev_goal_dist = torch.zeros(n, device=dev)
        self.ep_time = torch.zeros(n, device=dev)
        self.next_scan_t = torch.zeros(n, device=dev)
        self.scan = torch.ones(n, self.corr.lidar.num_rays, device=dev)
        self.scan_clean = torch.ones_like(self.scan)
        self.scan_t = torch.zeros(n, device=dev)
        self.outcome = torch.zeros(n, dtype=torch.long, device=dev)
        self.ep_sums = {k: torch.zeros(n, device=dev) for k in REWARD_TERMS}
        self.ep_min_clearance = torch.full((n,), float("inf"), device=dev)
        self._goal_dist = torch.zeros(n, device=dev)
        self._clear = torch.zeros(n, device=dev)
        self._flags = {}
        self.last_reward_terms = {k: torch.zeros(n, device=dev) for k in REWARD_TERMS}

    # ------------------------------------------------------------------------------------------------ setup
    def _setup_scene(self):
        self.corr.prepare_stage()

    # ------------------------------------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions, -1.0, 1.0)
        self.corr.cm.set_command(self.actions[:, 0] * self.cfg.v_max, self.actions[:, 1] * self.cfg.w_max)

    def _apply_action(self):
        self.corr.advance_obstacles(self.physics_dt)              # moving obstacles (no-op in static scenes)
        wl, wr, _, _ = self.corr.cm.step()
        self.corr.robot.set_joint_velocity_target(torch.stack([wl, wr], -1), joint_ids=self.corr.wheel_joint_ids)

    # ------------------------------------------------------------------------------------------------ dones
    def _get_dones(self):
        self.extras.pop("log", None)                             # episode log is per step (see _reset_idx)
        self.ep_time += self.step_dt
        self.corr.sim_time += self.step_dt
        self._goal_dist = self.corr.goal_distance()
        self._clear = self.corr.footprint_clearance()
        goal = self._goal_dist < self.cfg.goal_tolerance
        physical = (self._clear <= 0.0) & ~goal                  # goal has priority (reference order)
        safety = (self._clear < self.cfg.safety_margin) & ~goal  # includes physical collisions
        terminated = goal | safety
        time_out = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        self.outcome[:] = RUNNING
        self.outcome[time_out] = TIMEOUT
        self.outcome[safety] = SAFETY_VIOLATION
        self.outcome[physical] = PHYSICAL_COLLISION
        self.outcome[goal] = GOAL
        self._flags = {"goal": goal, "physical": physical, "safety": safety, "time_out": time_out}
        self.ep_min_clearance = torch.minimum(self.ep_min_clearance, self._clear)
        return terminated, time_out

    # ------------------------------------------------------------------------------------------------ rewards
    def _get_rewards(self) -> torch.Tensor:
        rc = self.cfg.rewards
        d, c, f = self._goal_dist, self._clear, self._flags
        terms = {
            "progress": rc.progress_scale * (self.prev_goal_dist - d),
            "proximity": -rc.proximity_scale * torch.clamp(1.0 - c / rc.proximity_clearance, min=0.0) ** 2,
            "time": torch.full_like(d, -rc.time_penalty_per_s * self.step_dt),
            "goal": rc.goal_reward * f["goal"].float(),
            "collision": -rc.collision_penalty * f["safety"].float(),
            "timeout": -rc.timeout_penalty * f["time_out"].float(),
        }
        self.prev_goal_dist = d.clone()
        total = torch.zeros_like(d)
        for k, v in terms.items():
            total += v
            self.ep_sums[k] += v
        self.last_reward_terms = terms
        return total

    # ------------------------------------------------------------------------------------------------ resets
    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ended = env_ids[self.episode_length_buf[env_ids] > 0]
        # Per-episode tensors (no .item() sync): the RSL-RL logger concatenates them over the iteration, so its means
        # are per episode. "log" exists only on steps where episodes ended (cleared every step in _get_dones).
        if len(ended):
            log = {}
            for k in REWARD_TERMS:
                log[f"Episode_Reward/{k}"] = self.ep_sums[k][ended].clone()
            log["Episode/length_steps"] = self.episode_length_buf[ended].float()
            log["Episode/min_clearance_m"] = self.ep_min_clearance[ended].clone()
            for code, name in OUTCOME_NAMES.items():
                log[f"Episode_Termination/{name}"] = (self.outcome[ended] == code).float()
            self.extras["log"] = log
        self.extras["outcome"] = self.outcome.clone()                     # per env; valid for envs reset this step
        super()._reset_idx(env_ids)                                       # scene.reset, episode_length_buf = 0
        self.corr.reset(env_ids)                                          # layout, start/goal, robot, actuator model
        self.actions[env_ids] = 0.0
        self.ep_time[env_ids] = 0.0
        self.corr.sim_time[env_ids] = 0.0
        self.next_scan_t[env_ids] = 0.0                                   # force a fresh scan in this step's obs
        for k in REWARD_TERMS:
            self.ep_sums[k][env_ids] = 0.0
        self.ep_min_clearance[env_ids] = float("inf")
        self.prev_goal_dist[env_ids] = self.corr.goal_distance()[env_ids]  # no artificial progress after reset

    # ------------------------------------------------------------------------------------------------ observations
    def _robot_state(self):
        p, yaw = self.corr.robot_pose2d()
        return p, yaw

    def lidar_block(self, obs: dict) -> torch.Tensor:
        """(N, num_rays) normalized LiDAR ranges of an observation dict, in ray order, for either obs_layout."""
        if self.cfg.obs_layout == "image":
            return obs["lidar"].reshape(obs["lidar"].shape[0], -1)[:, self._img_idx]
        return obs["policy"][:, :self.corr.lidar.num_rays]

    def _get_observations(self) -> dict:
        # tolerance half a physics step: ep_time accumulates step_dt in float32 (error ~1e-5 s after tens of seconds), so
        # a 1e-6 tolerance misses due times of periods that are not multiples of step_dt (3D LiDAR 0.25 s; M7)
        tol = 0.5 * self.physics_dt
        due = self.ep_time + tol >= self.next_scan_t
        if bool(due.any()):
            tick = torch.round(self.ep_time / self.physics_dt).to(torch.int64)   # acquisition time (physics steps)
            out = self.corr.lidar.read(self.ep_time, noise_keys=(self.corr.lidar_seed, tick))
            self.scan[due] = out.normalized[due]
            self.scan_clean[due] = (out.ranges_clean / out.range_max)[due]
            self.scan_t[due] = self.ep_time[due]
            self.next_scan_t[due] = self.next_scan_t[due] + self.scan_period * torch.ceil(
                (self.ep_time[due] + tol - self.next_scan_t[due]) / self.scan_period).clamp(min=1)
        p, yaw = self._robot_state()
        g = self.corr.goal - p
        dist = torch.linalg.norm(g, dim=1)
        ang = torch.atan2(g[:, 1], g[:, 0]) - yaw
        rd = self.corr.robot.data
        if self.cfg.obs_layout == "image":
            nav = torch.cat([
                torch.clamp(dist / self.cfg.goal_distance_norm, 0.0, 1.0)[:, None],
                torch.sin(ang)[:, None], torch.cos(ang)[:, None],
                self.actions,
                (rd.root_link_lin_vel_b[:, 0] / self.cfg.v_max)[:, None],
                (rd.root_link_ang_vel_b[:, 2] / self.cfg.w_max)[:, None],
            ], dim=1)
            n = self.scan.shape[0]
            img = torch.ones(n, self.img_shape[1] * self.img_shape[2], device=self.scan.device)   # cells without a ray: 1
            img[:, self._img_idx] = self.scan                       # latest complete scan (held between acquisitions)
            return {"policy": nav, "lidar": img.view(n, *self.img_shape)}
        obs = torch.cat([
            self.scan,
            torch.clamp(dist / self.cfg.goal_distance_norm, 0.0, 1.0)[:, None],
            torch.sin(ang)[:, None], torch.cos(ang)[:, None],
            self.actions,
            # base_link frame velocity (= wheel-odometry twist of the real robot). NOT root_lin_vel_b, which in
            # Isaac Lab is the CoM velocity: the CoM sits 0.137 m above base_link, so body pitch rocking adds
            # w_y * 0.137 (up to ~0.13 m/s) to it (M4 follow-up diagnostics, infrastructure/M4_rl_environment/results/diag).
            (rd.root_link_lin_vel_b[:, 0] / self.cfg.v_max)[:, None],
            (rd.root_link_ang_vel_b[:, 2] / self.cfg.w_max)[:, None],
        ], dim=1)
        out = {"policy": obs}
        if self.cfg.privileged_critic:
            out["critic"] = torch.cat([obs, self.scan_clean, self.corr.footprint_clearance()[:, None],
                                       (g / self.cfg.goal_distance_norm)], dim=1)
        return out
