"""Evaluation recording format: one folder per episode (layout of the reference simplified environment,
adaptive_sensor_policy AdaptiveFPS/evaluation/results.py), written by Navigation/evaluation/recorder.py and read by the
visualization tools. Pure Python (numpy, csv, json): no st_nav, Isaac Lab or PyTorch.

<eval dir> = <line>/eval/<ID>/<protocol>/<policy>/
    run_config.json        resolved evaluation and environment configuration
    episodes.csv           one row per episode (aggregate metrics)
    summary.json           evaluation summary
    world_geometry.json    static walls of the scene (fixed part; each episode has its own complete copy)
    episode_NNNN/          NNNN = episode index of episodes.csv (0-based), one folder per episode:
        metadata.json        episode, outcome, seeds, robot, actuator, LiDAR, sensing, obstacles, DR, file index
        world_geometry.json  every wall of THIS episode (fixed walls + randomized baffles), moving-obstacle paths
        trajectory.csv       row k = robot state after step k (row 0 = reset state)
        steps.csv            row k = step k (1..n): action, command, rewards, sensing, termination
        lidar.npz            row k = scan of the observation after step k (row 0 = reset observation)
        obstacles.npz        moving obstacles, row k = after step k (empty arrays in static scenes)
<line>/eval/<ID>/<protocol>/summary.csv   one row per evaluated policy (same episodes, directly comparable)

Conventions: env-local frame (m), yaw in rad CCW from +x, t_sim = simulated seconds since reset.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
TRAJECTORY_FIELDS = ["step", "t_sim", "x", "y", "yaw", "v_mps", "w_radps", "goal_distance_m", "clearance_m",
                     "distance_traveled_m"]
STEP_FIELDS = ["step", "t_sim", "action_v", "action_w", "cmd_v_mps", "cmd_w_radps", "act_v_mps", "act_w_radps",
               "reward", "r_progress", "r_proximity", "r_time", "r_goal", "r_collision", "r_timeout", "cumulative_reward",
               "scan_fresh", "beams_acquired", "min_lidar_m", "goal_distance_m", "clearance_m", "done", "outcome"]
FILE_INDEX = {
    "metadata.json": "episode, outcome, seeds, robot, actuator, LiDAR, sensing, moving obstacles, DR flags",
    "world_geometry.json": "all walls of this episode (cx, cy, length, thickness, yaw) and moving-obstacle paths",
    "trajectory.csv": "row k = robot state after step k (row 0 = reset): pose, base_link velocity, goal distance, clearance",
    "steps.csv": "row k = step k: policy action in [-1, 1], command (m/s, rad/s), actuator output, reward terms, sensing, done",
    "lidar.npz": "row k = scan of the observation after step k: ranges (observed, noisy; NaN where not observed), "
                 "ranges_clean, per-beam acquisition time/flag, sensor origin pose at acquisition",
    "obstacles.npz": "moving obstacles: ids, radius, active, x/y per row (row k = after step k)",
}


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def _clean(o):
    """JSON-safe copy: NaN / inf -> null (json.dump would write invalid NaN tokens)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=1, default=_jsonable)


def _write_table(path, fields, cols: dict, n: int):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(n):
            row = []
            for k in fields:
                v = cols[k][i] if k in cols else ""
                if isinstance(v, (float, np.floating)):
                    v = "" if not math.isfinite(float(v)) else repr(round(float(v), 6))
                elif isinstance(v, (bool, np.bool_)):
                    v = int(v)
                elif isinstance(v, np.integer):
                    v = int(v)
                row.append(v)
            w.writerow(row)


def episode_dir_name(index: int) -> str:
    return f"episode_{index:04d}"


def write_episode(ep_dir, meta: dict, geometry: dict, traj: dict, steps: dict, lidar: dict, obstacles: dict):
    """Write one episode folder. traj / steps: column name -> 1-D array (TRAJECTORY_FIELDS / STEP_FIELDS);
    lidar / obstacles: arrays stored as-is in the npz files."""
    ep_dir = Path(ep_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(traj["step"])
    meta = {"schema_version": SCHEMA_VERSION, **meta, "files": FILE_INDEX, "rows": {
        "trajectory": n_rows, "steps": len(steps["step"]), "lidar": int(np.asarray(lidar["ranges"]).shape[0]),
        "obstacles": int(np.asarray(obstacles["x"]).shape[0])}}
    write_json(ep_dir / "metadata.json", meta)
    write_json(ep_dir / "world_geometry.json", geometry)
    _write_table(ep_dir / "trajectory.csv", TRAJECTORY_FIELDS, traj, n_rows)
    _write_table(ep_dir / "steps.csv", STEP_FIELDS, steps, len(steps["step"]))
    np.savez_compressed(ep_dir / "lidar.npz", **{k: np.asarray(v) for k, v in lidar.items()})
    np.savez_compressed(ep_dir / "obstacles.npz", **{k: np.asarray(v) for k, v in obstacles.items()})


def update_protocol_summary(protocol_dir, row: dict):
    """<protocol>/summary.csv: one row per policy (replaces the policy's previous row)."""
    path = Path(protocol_dir) / "summary.csv"
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r.get("policy") != row["policy"]]
    rows.append({k: v for k, v in row.items()})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------------------------- reading
def _read_table(path) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in (rows[0].keys() if rows else []):
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            out[k] = np.array(vals)
    return out


def load_episode(ep_dir) -> dict:
    """Everything needed to reconstruct one episode: meta, geometry, traj, steps, lidar, obstacles."""
    d = Path(ep_dir)
    lidar = dict(np.load(d / "lidar.npz"))
    obst = dict(np.load(d / "obstacles.npz"))
    return {"dir": d, "meta": json.load(open(d / "metadata.json")), "geometry": json.load(open(d / "world_geometry.json")),
            "traj": _read_table(d / "trajectory.csv"), "steps": _read_table(d / "steps.csv"), "lidar": lidar,
            "obstacles": obst}


def episode_dirs(eval_dir, episodes=None) -> list:
    d = Path(eval_dir)
    if episodes is None:
        return sorted(p for p in d.glob("episode_*") if p.is_dir())
    return [d / episode_dir_name(int(i)) for i in episodes]


def load_eval(eval_dir) -> dict:
    """Evaluation-level files: run_config, summary, episodes.csv rows (as dicts of strings), episode folders."""
    d = Path(eval_dir)
    with open(d / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    rc = d / "run_config.json"
    return {"dir": d, "run_config": json.load(open(rc)) if rc.exists() else {},
            "summary": json.load(open(d / "summary.json")), "episodes": rows, "episode_dirs": episode_dirs(d)}


# --------------------------------------------------------------------------------------------- geometry helpers
def wall_polygon(w: dict) -> np.ndarray:
    """(4, 2) corners of a wall {cx, cy, length, thickness, yaw}."""
    c, s = math.cos(w["yaw"]), math.sin(w["yaw"])
    hl, ht = w["length"] / 2, w["thickness"] / 2
    pts = [(-hl, -ht), (hl, -ht), (hl, ht), (-hl, ht)]
    return np.array([(w["cx"] + c * x - s * y, w["cy"] + s * x + c * y) for x, y in pts])


def lidar_points(ep: dict, row: int):
    """World (env-local) end points of the beams of lidar row `row`, from the sensor pose at acquisition.
    Returns (origin (2,), ends (B, 2), hit (B,) bool)."""
    L = ep["lidar"]
    ranges = L["ranges"][row]
    if not np.isfinite(ranges).any():
        ranges = L["ranges_clean"][row]
    ox, oy, oyaw = L["origin_x"][row], L["origin_y"][row], L["origin_yaw"][row]
    ang = oyaw + L["beam_angle_rad"]
    ends = np.stack([ox + ranges * np.cos(ang), oy + ranges * np.sin(ang)], -1)
    return np.array([ox, oy]), ends, ranges < float(L["range_max"]) - 1e-6
