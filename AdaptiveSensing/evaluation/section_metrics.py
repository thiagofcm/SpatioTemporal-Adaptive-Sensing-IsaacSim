"""Evaluation tables of the fixed-section (spatiotemporal) evaluation, STS01.

episodes.csv = the reference schema of AdaptiveSensing/evaluation/metrics.py (same columns, order and decimals), with
    policy_type          fixed_sections
    fixed_fps            the common rate when every section has the same rate (synchronized), else empty
    mean_fps             ray-weighted rate sum_s w_s f_s
    episode_frame_count  full-scan equivalents FSE = rays acquired (reset scan incl.) / total rays (6 decimals; an
                         integer when synchronized, then identical to the temporal frame count)
    adaptive_return      nav_return - frame_cost * (FSE - 1) - budget_penalty * [FSE > budget]
followed by section columns (per section s, left to right):
    hz_<s>, acq_<s> (acquisitions incl. the reset scan), mean_interval_s_<s>, mean_obs_age_s_<s>, max_obs_age_s_<s>,
    accepted_scans (native scans accepted by >= 1 section), native_scans, mean_obs_age_s (ray-weighted),
    max_obs_age_s (max over sections), start_pose_index, ob<j>_speed_mps, ob<j>_path_y_m.
The terminal pose and collision type are computed from the recorder's in-memory terminal state with exactly the inputs
metrics.collision_type reads from a recording (trajectory.csv values rounded to 6 decimals, the episode's walls,
obstacle circles), so no per-episode files are needed.

summary.csv (protocol folder, one row per configuration) = metrics.summarize + section columns; updated under a file
lock (several evaluations may finish at the same time). Torch / NumPy only (no Isaac Sim).
"""
from __future__ import annotations

import csv
import fcntl
import os

import numpy as np
import torch

from AdaptiveSensing.evaluation import metrics as M


def episode_fields(names, n_obstacles):
    sec = [f"{k}_{s}" for k in ("hz", "acq", "mean_interval_s", "mean_obs_age_s", "max_obs_age_s") for s in names]
    obs = [f"ob{j + 1}_{k}" for j in range(n_obstacles) for k in ("speed_mps", "path_y_m")]
    return M.EPISODE_FIELDS + sec + ["accepted_scans", "native_scans", "mean_obs_age_s", "max_obs_age_s",
                                     "start_pose_index"] + obs


def summary_fields(names):
    return M.SUMMARY_FIELDS + [f"hz_{s}" for s in names] + [f"mean_acq_{s}" for s in names] + [
        "mean_fse_success", "mean_obs_age_s", "max_obs_age_s", "config"]


def collision_type_mem(x, y, walls: list, ob_xy=None, ob_active=None, ob_radius=None):
    """= metrics.collision_type on the values a recording would hold. x, y: terminal position (already rounded to 6
    decimals as in trajectory.csv); walls: episode_geometry()["walls"]; ob_xy (K, 2) float32 terminal obstacle centres,
    ob_active (K,), ob_radius (scalar). Returns (kind, clearance + footprint radius)."""
    from st_nav.geometry.analytic2d import circle_clearance, wall_clearance

    p = torch.tensor([[float(x), float(y)]], dtype=torch.float32)
    wt = torch.tensor([[w["cx"], w["cy"], w["length"] / 2.0, w["thickness"] / 2.0, w["yaw"]] for w in walls],
                      dtype=torch.float32)
    d_wall = float(wall_clearance(p, wt)[0])
    d_obs = float("inf")
    if ob_xy is not None and len(ob_xy):
        r = np.where(np.asarray(ob_active).astype(bool), ob_radius, 0.0)
        circles = torch.tensor(np.stack([ob_xy[:, 0], ob_xy[:, 1], r], -1)[None], dtype=torch.float32)
        d_obs = float(circle_clearance(p, circles)[0])
    return ("obstacle" if d_obs < d_wall else "wall"), min(d_wall, d_obs)


def episode_row(*, episode, seed, hz, fse, env_outcome, n_steps, nav_return, final_goal_distance_m,
                distance_traveled_m, final_pose, collision_kind, start, goal, frame_cost, budget, budget_penalty,
                weights, extra: dict):
    outcome = M.OUTCOME_MAP[env_outcome]
    same = len(set(hz)) == 1
    fse_v = int(round(fse)) if abs(fse - round(fse)) < 1e-12 else round(float(fse), 6)
    row = {"episode": int(episode), "seed": int(seed), "policy_type": "fixed_sections",
           "fixed_fps": float(hz[0]) if same else "", "mean_fps": float(sum(w * h for w, h in zip(weights, hz))),
           "success": outcome == "success", "outcome": outcome,
           "collision_type": collision_kind if outcome == "collision" else "", "n_steps": int(n_steps),
           "adaptive_return": float(nav_return) - float(frame_cost) * (fse - 1.0)
           - (float(budget_penalty) if fse > budget else 0.0),
           "nav_return": float(nav_return), "episode_frame_count": fse_v,
           "final_goal_distance_m": float(final_goal_distance_m), "distance_traveled_m": float(distance_traveled_m),
           "final_x": float(final_pose[0]), "final_y": float(final_pose[1]), "final_theta": float(final_pose[2]),
           "start_x": float(start[0]), "start_y": float(start[1]), "start_theta": float(start[2]),
           "goal_x": float(goal[0]), "goal_y": float(goal[1]), "frame_cost": float(frame_cost), "budget": budget}
    if same:
        row["mean_fps"] = float(hz[0])
    for k, d in M.DECIMALS.items():
        if row[k] != "":
            row[k] = round(row[k], d)
    row.update(extra)
    return row


def write_episodes(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, run_name, names, hz, config):
    s = M.summarize(rows, run_name)
    col = lambda k, rs=rows: np.array([float(r[k]) for r in rs])  # noqa: E731
    succ = [r for r in rows if str(r["success"]) == "True"]
    s.update({**{f"hz_{n}": h for n, h in zip(names, hz)},
              **{f"mean_acq_{n}": float(col(f"acq_{n}").mean()) for n in names},
              "mean_fse_success": float(col("episode_frame_count", succ).mean()) if succ else float("nan"),
              "mean_obs_age_s": float(col("mean_obs_age_s").mean()), "max_obs_age_s": float(col("max_obs_age_s").max()),
              "config": config})
    return s


def update_summary_csv(path, row, fields):
    """Replace the run's row (by run_name), keep the others, sort by (mean_fps, config); under an exclusive lock."""
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = []
        if os.path.exists(path):
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if r["run_name"] != row["run_name"]]
        rows.append(row)
        rows.sort(key=lambda r: (float(r["mean_fps"]), str(r["config"])))
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
