"""Evaluation tables of the sensing experiments (temporal and spatiotemporal; fixed-rate and adaptive policies).

Schema of the reference evaluator (adaptive_sensor_policy@dc5b53f AdaptiveFPS/scripts/eval_adaptive_fps.py:
EPISODE_FIELDS / SUMMARY_FIELDS / summarize), with `episode_return` renamed `adaptive_return`:

episodes.csv (one row per episode, in this order)
    episode               episode index; 0-based here (= episode_NNNN/ folder) where the reference counts from 1
    seed                  the episode's environment seed (here the counter-based episode seed, st_nav/crng.py)
    policy_type           fixed | adaptive
    fixed_fps             requested rate of a fixed-rate policy (Hz); empty for adaptive policies
    mean_fps              mean over the episode's control steps of the sensing rate in force after each step
                          (reference: mean of info["current_fps"] over every control step)
    success               outcome == success
    outcome               success | collision | timeout (collision = the env's safety-violation termination, which
                          includes physical contact; the physical / safety distinction stays in metadata.json)
    collision_type        nearest ground-truth surface at a collision: wall | obstacle; empty otherwise
    n_steps               control steps of the episode
    adaptive_return       total sensing reward: nav_return - frame_cost * charged frames - budget penalty (charged
                          frames = episode_frame_count - 1; budget penalty once if episode_frame_count > budget)
    nav_return            total navigation (env) reward
    episode_frame_count   acquired observations incl. the initial reset observation (full-scan equivalents)
    final_goal_distance_m, distance_traveled_m, final_x, final_y, final_theta, start_x, start_y, start_theta,
    goal_x, goal_y        env-local frame, metres / radians
    frame_cost, budget    the evaluation's values

summary.csv (one row per evaluated run, in the protocol folder, sorted by mean_fps; reference aggregation: means
and population standard deviations over all episodes; n_collision counts outcome == collision)

Sensing diagnostics (native scans, measured rate, sampling intervals, observation age) are not in these tables: they
are in every episode's metadata.json and the run's summary.json. Torch / NumPy only (no Isaac Sim).
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

EPISODE_FIELDS = ["episode", "seed", "policy_type", "fixed_fps", "mean_fps", "success", "outcome", "collision_type",
                  "n_steps", "adaptive_return", "nav_return", "episode_frame_count", "final_goal_distance_m",
                  "distance_traveled_m", "final_x", "final_y", "final_theta", "start_x", "start_y", "start_theta",
                  "goal_x", "goal_y", "frame_cost", "budget"]
SUMMARY_FIELDS = ["run_name", "mean_fps", "n_episodes", "success_rate", "mean_adaptive_return", "std_adaptive_return",
                  "mean_nav_return", "std_nav_return", "mean_n_steps", "std_n_steps", "mean_final_goal_distance_m",
                  "mean_distance_traveled_m", "std_distance_traveled_m", "mean_frame_count", "std_frame_count",
                  "n_success", "n_collision", "n_timeout", "policy_type", "frame_cost", "budget"]
# env outcome names (st_nav.tasks.nav2d.corridor_env.OUTCOME_NAMES) -> reference outcomes
OUTCOME_MAP = {"goal": "success", "physical_collision": "collision", "safety_violation": "collision",
               "timeout": "timeout"}
# fixed decimals of the written values (identical whether a row comes from a live evaluation or from recordings)
DECIMALS = {"mean_fps": 6, "adaptive_return": 3, "nav_return": 3, "final_goal_distance_m": 4,
            "distance_traveled_m": 3, "final_x": 6, "final_y": 6, "final_theta": 6, "start_x": 6, "start_y": 6,
            "start_theta": 5, "goal_x": 6, "goal_y": 6}


def adaptive_return(nav_return, frame_count, frame_cost, budget, budget_penalty):
    charged = frame_count - 1                                   # the reset observation is free
    return nav_return - frame_cost * charged - (budget_penalty if frame_count > budget else 0.0)


def collision_type(ep_dir, footprint_radius):
    """(type, clearance) at the terminal state of an episode recording: nearest ground-truth surface among the walls
    of its world_geometry.json and the active moving obstacles of obstacles.npz (the env's analytic geometry,
    st_nav.geometry.analytic2d). clearance = min distance - footprint radius (= trajectory.csv clearance_m)."""
    import torch

    from st_nav.geometry.analytic2d import circle_clearance, wall_clearance

    with open(os.path.join(ep_dir, "trajectory.csv")) as f:
        last = list(csv.DictReader(f))[-1]
    p = torch.tensor([[float(last["x"]), float(last["y"])]], dtype=torch.float32)
    geo = json.load(open(os.path.join(ep_dir, "world_geometry.json")))
    walls = torch.tensor([[w["cx"], w["cy"], w["length"] / 2.0, w["thickness"] / 2.0, w["yaw"]] for w in geo["walls"]],
                         dtype=torch.float32)
    d_wall = float(wall_clearance(p, walls)[0])
    d_obs = float("inf")
    ob = np.load(os.path.join(ep_dir, "obstacles.npz"))
    if ob["x"].size:
        r = np.where(ob["active"].astype(bool), ob["radius"], 0.0)
        circles = torch.tensor(np.stack([ob["x"][-1], ob["y"][-1], r], -1)[None], dtype=torch.float32)
        d_obs = float(circle_clearance(p, circles)[0])
    kind = "obstacle" if d_obs < d_wall else "wall"           # reference tie rule: wall unless the obstacle is nearer
    return kind, min(d_wall, d_obs) - footprint_radius, last


def episode_row(ep_dir, *, episode, seed, policy_type, fixed_fps, mean_fps, env_outcome, n_steps, nav_return,
                frame_count, final_goal_distance_m, distance_traveled_m, start, goal, frame_cost, budget,
                budget_penalty, footprint_radius):
    """One episodes.csv row from the evaluation values and the episode's recording (terminal pose, collision type)."""
    outcome = OUTCOME_MAP[env_outcome]
    kind, _, last = collision_type(ep_dir, footprint_radius)
    row = {"episode": int(episode), "seed": int(seed), "policy_type": policy_type,
           "fixed_fps": "" if fixed_fps is None else float(fixed_fps), "mean_fps": float(mean_fps),
           "success": outcome == "success", "outcome": outcome,
           "collision_type": kind if outcome == "collision" else "", "n_steps": int(n_steps),
           "adaptive_return": adaptive_return(float(nav_return), int(frame_count), float(frame_cost), budget,
                                              float(budget_penalty)),
           "nav_return": float(nav_return), "episode_frame_count": int(frame_count),
           "final_goal_distance_m": float(final_goal_distance_m), "distance_traveled_m": float(distance_traveled_m),
           "final_x": float(last["x"]), "final_y": float(last["y"]), "final_theta": float(last["yaw"]),
           "start_x": float(start[0]), "start_y": float(start[1]), "start_theta": float(start[2]),
           "goal_x": float(goal[0]), "goal_y": float(goal[1]), "frame_cost": float(frame_cost), "budget": budget}
    for k, d in DECIMALS.items():
        if row[k] != "":
            row[k] = round(row[k], d)
    return row


def write_episodes(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, run_name):
    """Reference summarize(): means and population standard deviations (np.std) over all episodes."""
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    ret, nav, steps, frames, dist = (col("adaptive_return"), col("nav_return"), col("n_steps"),
                                     col("episode_frame_count"), col("distance_traveled_m"))
    outcomes = [r["outcome"] for r in rows]
    return {"run_name": run_name, "mean_fps": float(col("mean_fps").mean()), "n_episodes": len(rows),
            "success_rate": float(np.mean([str(r["success"]) == "True" for r in rows])),
            "mean_adaptive_return": float(ret.mean()), "std_adaptive_return": float(ret.std()),
            "mean_nav_return": float(nav.mean()), "std_nav_return": float(nav.std()),
            "mean_n_steps": float(steps.mean()), "std_n_steps": float(steps.std()),
            "mean_final_goal_distance_m": float(col("final_goal_distance_m").mean()),
            "mean_distance_traveled_m": float(dist.mean()), "std_distance_traveled_m": float(dist.std()),
            "mean_frame_count": float(frames.mean()), "std_frame_count": float(frames.std()),
            "n_success": outcomes.count("success"), "n_collision": outcomes.count("collision"),
            "n_timeout": outcomes.count("timeout"), "policy_type": rows[0]["policy_type"],
            "frame_cost": rows[0]["frame_cost"], "budget": rows[0]["budget"]}


def update_summary_csv(path, row):
    """Reference update_summary(): replace the run's row (by run_name), keep the others, sort by mean_fps."""
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run_name"] != row["run_name"]]
    rows.append(row)
    rows.sort(key=lambda r: float(r["mean_fps"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
