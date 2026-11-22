"""Per-episode evaluation recording for the navigation env (format: tools/episode_data.py).

The evaluation runs all environments in parallel; EpisodeRecorder captures, every control step and for every env,
the state at the END of the step (hooked into _get_dones, i.e. before any auto-reset) plus what the policy observed,
and finally exports each env's first episode to its own folder. Recording does not change the simulation: the clean
LiDAR ranges are computed from the ray caster's hit points (the noise generator is not touched) and no state is
written. Import after the Isaac Sim app has been launched.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from isaaclab.utils.math import euler_xyz_from_quat
from st_nav.assets.turtlebot4 import TB4_COMMAND_MODEL, TB4_GEOMETRY
from st_nav.scenes import _common as SC
from st_nav.scenes._obstacle_motion import OBSTACLE_RADIUS, OBSTACLE_TURN_PAUSE_RANGE_S
from st_nav.tasks.nav2d.corridor_env import OUTCOME_NAMES, REWARD_TERMS

from tools import episode_data as ED

WALL_NAMES_FIXED = ["outer_W", "outer_N", "outer_E", "outer_S", "C1", "C2", "C3", "C4", "C5", "C6"]


def _np(t):
    return t.detach().float().cpu().numpy().copy()


class EpisodeRecorder:
    def __init__(self, env_unwrapped):
        self.u = u = env_unwrapped
        self.corr = c = u.corr
        self.R = float(c.lidar.range_max)
        self.rows, self.steps = [], []
        self._end = None
        self._orig_dones = u._get_dones
        u._get_dones = self._dones_hook
        self.walls0 = _np(c.walls)                                 # walls of every env's first episode
        om = c.obs_model
        self.ob0 = None if om is None else {"active": om.active.cpu().numpy().copy(), "speed": _np(om.speed),
                                            "path_y": _np(om.path_y)}
        self._origin_prev = None
        self.seeds0 = {"episode_seed": c.episode_seed.cpu().tolist(), "obstacle_seed": c.obstacle_seed.cpu().tolist(),
                       "lidar_noise_seed": c.lidar_seed.cpu().tolist(), "episode_number": c.episode_count.cpu().tolist()}

    # ------------------------------------------------------------------ state capture
    def _lidar_clean(self):
        """Noise-free ranges (N, B) from the ray caster hits, and the sensor origin (env-local x, y, yaw)."""
        lid = self.corr.lidar
        data = lid.rc.data                                         # updates the ray cast if outdated (no noise drawn)
        org = lid.origin_w
        d = torch.linalg.norm(data.ray_hits_w - org.unsqueeze(1), dim=-1)
        d = torch.where(torch.isfinite(d) & (d <= self.R), d, torch.full_like(d, self.R))
        _, _, yaw = euler_xyz_from_quat(data.quat_w)
        o = org - self.corr.origins
        return d, torch.stack([o[:, 0], o[:, 1], torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi], 1)

    def _state(self):
        c, u = self.corr, self.u
        p, yaw = c.robot_pose2d()
        rd = c.robot.data
        clean, origin = self._lidar_clean()
        circ = c.obstacle_circles()
        return {"x": p[:, 0], "y": p[:, 1], "yaw": yaw, "v": rd.root_link_lin_vel_b[:, 0], "w": rd.root_link_ang_vel_b[:, 2],
                "goal_d": c.goal_distance(), "clear": c.footprint_clearance(), "clean": clean, "origin": origin,
                "obst": circ, "act": c.cm.state.clone()}

    def _dones_hook(self):
        out = self._orig_dones()
        self._end = {k: (v.clone() if v is not None else None) for k, v in self._state().items()}
        return out

    @property
    def end_state(self):
        return self._end

    # ------------------------------------------------------------------ per-step records
    def record_reset(self, obs):
        st = self._state()
        observed = obs["policy"][:, :40] * self.R
        self._origin_prev = st["origin"].clone()
        self.rows.append(self._row(st, observed, torch.ones_like(st["x"], dtype=torch.bool), torch.zeros_like(st["x"]),
                                   st["origin"]))

    def record_step(self, actions, rew, dones, ext, obs):
        u, st = self.u, self._end
        d = dones.bool()
        observed = torch.where(d[:, None], torch.full_like(st["clean"], float("nan")), obs["policy"][:, :40] * self.R)
        fresh = (~d) & ((u.scan_t - u.ep_time).abs() < 1e-6)       # a scan was acquired at this step
        t_scan = torch.where(d, torch.full_like(st["x"], float("nan")), u.scan_t)
        origin = torch.where((fresh | d)[:, None], st["origin"], self._origin_prev)   # pose at acquisition of `ranges`
        self._origin_prev = torch.where(d[:, None], self._origin_prev, origin)   # reset envs restart from row 0 only
        self.rows.append(self._row(st, observed, fresh, t_scan, origin))
        a = torch.clamp(actions, -1.0, 1.0)
        terms = u.last_reward_terms
        self.steps.append({"action": _np(a), "cmd": _np(a * torch.tensor([u.cfg.v_max, u.cfg.w_max], device=a.device)),
                           "act": _np(st["act"]), "reward": _np(rew), "terms": {k: _np(terms[k]) for k in REWARD_TERMS},
                           "fresh": fresh.cpu().numpy(), "done": d.cpu().numpy(),
                           "outcome": torch.where(d, ext["outcome"], torch.zeros_like(ext["outcome"])).cpu().numpy()})

    def _row(self, st, observed, fresh, t_scan, origin):
        circ = st["obst"]
        return {"x": _np(st["x"]), "y": _np(st["y"]), "yaw": _np(st["yaw"]), "v": _np(st["v"]), "w": _np(st["w"]),
                "goal_d": _np(st["goal_d"]), "clear": _np(st["clear"]), "clean": _np(st["clean"]),
                "observed": _np(observed), "fresh": fresh.cpu().numpy(), "t_scan": _np(t_scan), "origin": _np(origin),
                "obst": None if circ is None else _np(circ)}

    # ------------------------------------------------------------------ metadata
    def static_metadata(self, cfg) -> dict:
        u, c = self.u, self.corr
        return build_static_metadata(cfg, c.cfg, c.lidar.cfg, c.scene_mod, u.step_dt, u.physics_dt, int(u.max_episode_length),
                                     bool(c.has_baffles), c.obs_model is not None)

    def geometry(self, e: int) -> dict:
        ob = None if self.ob0 is None else {"active": self.ob0["active"][e], "path_y": self.ob0["path_y"][e]}
        return episode_geometry(self.corr.cfg.scene_name, self.walls0[e], self.corr.scene_mod, ob)

    def fixed_geometry(self) -> dict:
        return fixed_geometry(self.corr.cfg.scene_name)

    # ------------------------------------------------------------------ export
    def export(self, out_dir, n_steps, ep_rows: list, static_meta: dict, episode_meta_fn):
        """Write episode_NNNN/ for every env (its first episode, n_steps[e] control steps)."""
        u = self.u
        dt = u.step_dt
        B = self.rows[0]["clean"].shape[1]
        beam_ang = np.array(static_meta["lidar"]["beam_angles_rad"], dtype=np.float32)
        for e, n in enumerate(n_steps):
            n = int(n)
            R = self.rows[:n + 1]
            S = self.steps[:n]
            col = lambda k: np.array([r[k][e] for r in R])  # noqa: E731
            x, y = col("x"), col("y")
            dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            traj = {"step": np.arange(n + 1), "t_sim": np.arange(n + 1) * dt, "x": x, "y": y, "yaw": col("yaw"),
                    "v_mps": col("v"), "w_radps": col("w"), "goal_distance_m": col("goal_d"), "clearance_m": col("clear"),
                    "distance_traveled_m": dist}
            observed = np.stack([r["observed"][e] for r in R]).astype(np.float32)
            clean = np.stack([r["clean"][e] for r in R]).astype(np.float32)
            fresh = col("fresh").astype(bool)
            t_scan = col("t_scan")
            origin = np.stack([r["origin"][e] for r in R])
            rew = np.array([s["reward"][e] for s in S])
            min_l = np.where(np.isfinite(observed), observed, clean)[1:].min(1)    # observed scan; clean on the terminal row
            oc = np.array([s["outcome"][e] for s in S])
            steps = {"step": np.arange(1, n + 1), "t_sim": np.arange(1, n + 1) * dt,
                     "action_v": np.array([s["action"][e, 0] for s in S]), "action_w": np.array([s["action"][e, 1] for s in S]),
                     "cmd_v_mps": np.array([s["cmd"][e, 0] for s in S]), "cmd_w_radps": np.array([s["cmd"][e, 1] for s in S]),
                     "act_v_mps": np.array([s["act"][e, 0] for s in S]), "act_w_radps": np.array([s["act"][e, 1] for s in S]),
                     "reward": rew, **{f"r_{k}": np.array([s["terms"][k][e] for s in S]) for k in REWARD_TERMS},
                     "cumulative_reward": np.cumsum(rew), "scan_fresh": fresh[1:], "beams_acquired": np.where(fresh[1:], B, 0),
                     "min_lidar_m": min_l, "goal_distance_m": traj["goal_distance_m"][1:], "clearance_m": traj["clearance_m"][1:],
                     "done": np.array([s["done"][e] for s in S]),
                     "outcome": np.array([OUTCOME_NAMES.get(int(o), "") if dn else "" for o, dn in
                                          zip(oc, [s["done"][e] for s in S])])}
            lidar = {"t_sim": np.arange(n + 1) * dt, "ranges": observed, "ranges_clean": clean, "fresh": fresh,
                     "beam_acquired": np.repeat(fresh[:, None], B, 1), "beam_t_acq": np.repeat(t_scan[:, None], B, 1),
                     "origin_x": origin[:, 0], "origin_y": origin[:, 1], "origin_yaw": origin[:, 2],
                     "beam_angle_rad": beam_ang, "range_max": np.float64(self.R),
                     "mount_offset_xyz": np.array(static_meta["lidar"]["mount_offset_xyz_m"], dtype=np.float64)}
            if R[0]["obst"] is not None:
                ob = np.stack([r["obst"][e] for r in R])                       # (rows, K, 3)
                K = ob.shape[1]
                act = self.ob0["active"][e]
                obstacles = {"t_sim": np.arange(n + 1) * dt, "ids": np.array([f"OB{j + 1}" for j in range(K)]),
                             "x": ob[..., 0], "y": ob[..., 1], "radius": np.full(K, OBSTACLE_RADIUS), "active": act,
                             "speed_mps": self.ob0["speed"][e], "path_y_m": self.ob0["path_y"][e]}
            else:
                obstacles = {"t_sim": np.arange(n + 1) * dt, "ids": np.zeros(0, dtype="<U4"), "x": np.zeros((n + 1, 0)),
                             "y": np.zeros((n + 1, 0)), "radius": np.zeros(0), "active": np.zeros(0, dtype=bool)}
            meta = {**episode_meta_fn(e, ep_rows[e]), **static_meta,
                    "random_seeds": {k: v[e] for k, v in self.seeds0.items()} | {
                        "note": "counter-based (st_nav/crng.py): episode_seed = key(env seed, env index, episode number); "
                                "obstacle draws = uniform(obstacle_seed, stream, obstacle, counter); LiDAR noise of beam b "
                                "acquired at physics tick t = uniform(lidar_noise_seed, t, b)"},
                    "timebase": "t_sim = simulated seconds since reset; trajectory / lidar / obstacles row k = after step k "
                                "(row 0 = reset); steps row k = step k",
                    "lidar_rows_note": "ranges = the observation the policy received after step k (row 0: reset observation; "
                                       "last row NaN: the terminal state is not observed); ranges_clean = noise-free ranges "
                                       "at the end of step k for every row; origin_* = sensor pose when `ranges` was acquired "
                                       "(terminal row: the pose of ranges_clean)"}
            ED.write_episode(f"{out_dir}/{ED.episode_dir_name(e)}", meta, self.geometry(e), traj, steps, lidar, obstacles)


# ---------------------------------------------------------------------------------------------- shared builders
def build_static_metadata(cfg, scene_cfg, lidar_cfg, scene_mod, step_dt, physics_dt, max_steps, has_baffles, has_obstacles):
    """Setup shared by all episodes of an evaluation (robot, actuator, LiDAR, sensing, scene, rewards, DR)."""
    lc, sm = lidar_cfg, scene_mod
    az = [math.radians(a) for a in lc.azimuths_deg]
    meta = {
        "task": "navigation", "scene": scene_cfg.scene_name,
        "control": {"control_dt_s": step_dt, "physics_dt_s": physics_dt, "decimation": int(cfg.decimation),
                    "episode_length_s": float(cfg.episode_length_s), "max_steps": int(max_steps)},
        "robot": {"model": "TurtleBot4 Standard (articulated, PhysX)", "footprint_radius_m": scene_cfg.footprint_radius,
                  "wheel_radius_m": TB4_GEOMETRY.wheel_radius, "track_width_m": TB4_GEOMETRY.track_width,
                  "wheel_speed_max_mps": TB4_GEOMETRY.wheel_speed_max, "v_max_mps": cfg.v_max, "w_max_radps": cfg.w_max,
                  "action": "a in [-1, 1]^2 -> command (a0 * v_max, a1 * w_max)"},
        "actuator": {"model": "calibrated command model (delay, first-order lag, gains, wheel saturation)",
                     **dataclasses.asdict(TB4_COMMAND_MODEL)},
        "lidar": {"type": "lidar2d", "num_beams": len(az), "beam_angles_rad": az, "range_max_m": lc.range_max,
                  "range_min_m": lc.range_min, "mount_offset_xyz_m": list(lc.mount_pos), "ray_alignment": lc.ray_alignment,
                  "noise": {"enabled": bool(cfg.lidar_noise), "model": "datasheet uniform: error ~ U(+-a(d) * d), then cap",
                            "seeding": "per episode, acquisition physics tick and beam (see random_seeds)",
                            "bands": [list(b) for b in lc.noise.bands]},
                  "geometry_note": "beam b of lidar row k points at origin_yaw[k] + beam_angle_rad[b] from (origin_x, origin_y)[k]"},
        "sensing": {"mode": "fixed_rate", "period_s": float(cfg.lidar_period_s), "beams_per_scan": len(az),
                    "note": "lidar.npz: fresh = a scan was acquired at this step; beam_acquired / beam_t_acq give the "
                            "per-beam acquisition flag and time (all beams at the scan time here). Adaptive temporal / "
                            "spatial / spatiotemporal sensing varies these per beam (and may add an event log)."},
        "terminations": {"goal_tolerance_m": cfg.goal_tolerance, "safety_margin_m": cfg.safety_margin,
                         "clearance": "footprint clearance = distance to the nearest wall / active obstacle - footprint radius"},
        "rewards": cfg.rewards.to_dict(),
        "dr": {k: getattr(cfg, k) for k in ("start_pose_dr", "obstacle_presence_dr", "obstacle_speed_dr", "obstacle_y_dr",
                                            "obstacle_radius_dr", "lidar_noise")} | {"static_obstacle_dr": bool(has_baffles)},
    }
    if has_obstacles:
        meta["moving_obstacles_spec"] = {
            "ids": [f"OB{j + 1}" for j in range(len(sm.MOVING_OBSTACLES))], "radius_m": OBSTACLE_RADIUS,
            "height_m": scene_cfg.obstacle_height, "motion": [dict(o) for o in sm.MOVING_OBSTACLES],
            "turn_pause_range_s": list(OBSTACLE_TURN_PAUSE_RANGE_S), "active_obstacles": list(scene_cfg.active_obstacles),
            "presence_prob": list(sm.OBSTACLE_PRESENCE_PROB) if cfg.obstacle_presence_dr else None,
            "speed_ranges_mps": [list(r) for r in sm.OBSTACLE_SPEED_RANGES] if cfg.obstacle_speed_dr else None,
            "path_y_ranges_m": [list(r) for r in sm.OBSTACLE_Y_RANGES] if cfg.obstacle_y_dr else None,
            "path_y_gap_range_m": list(sm.OBSTACLE_Y_GAP_RANGE) if cfg.obstacle_y_dr else None}
    return meta


def episode_geometry(scene_name, walls, scene_mod, ob=None) -> dict:
    """All walls of one episode (walls: (W, 5) [cx, cy, hx, hy, yaw]) + the moving-obstacle paths
    (ob: {"active": (K,), "path_y": (K,)} or None)."""
    names = WALL_NAMES_FIXED + [f"Z{i + 1}" for i in range(len(walls) - len(WALL_NAMES_FIXED))]
    g = {"scene": scene_name, "arena_interior_half_m": SC.ARENA_INTERIOR_HALF,
         "walls": [{"name": nm, "cx": float(w[0]), "cy": float(w[1]), "length": float(2 * w[2]), "thickness": float(2 * w[3]),
                    "yaw": float(w[4])} for nm, w in zip(names, walls)], "moving_obstacles": []}
    if ob is not None:
        for j, o in enumerate(scene_mod.MOVING_OBSTACLES):
            y = float(ob["path_y"][j])
            dy = o["end"][1] - o["start"][1]
            g["moving_obstacles"].append({"id": f"OB{j + 1}", "active": bool(ob["active"][j]), "radius": OBSTACLE_RADIUS,
                                          "path": [[o["start"][0], y], [o["end"][0], y + dy]],
                                          "reversal_start": list(o["reversal_start"]), "reversal_end": list(o["reversal_end"])})
    return g


def fixed_geometry(scene_name) -> dict:
    walls = [{"name": nm, "cx": w[0], "cy": w[1], "length": w[2], "thickness": SC.WALL_THICKNESS, "yaw": w[3]}
             for nm, w in zip(WALL_NAMES_FIXED, SC.FIXED_WALLS)]
    return {"scene": scene_name, "arena_interior_half_m": SC.ARENA_INTERIOR_HALF, "walls": walls,
            "note": "fixed walls only; every episode_NNNN/world_geometry.json is the complete geometry of that episode"}
