"""Evaluation tables (AdaptiveSensing/evaluation/metrics.py) against the reference definitions
(adaptive_sensor_policy@dc5b53f eval_adaptive_fps.py: EPISODE_FIELDS, SUMMARY_FIELDS, summarize, update_summary)."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from AdaptiveSensing.evaluation import metrics as M  # noqa: E402


def test_schema_columns():
    assert M.EPISODE_FIELDS == ("episode,seed,policy_type,fixed_fps,mean_fps,success,outcome,collision_type,n_steps,"
                                "adaptive_return,nav_return,episode_frame_count,final_goal_distance_m,"
                                "distance_traveled_m,final_x,final_y,final_theta,start_x,start_y,start_theta,goal_x,"
                                "goal_y,frame_cost,budget").split(",")
    assert M.SUMMARY_FIELDS == ["run_name", "mean_fps", "n_episodes", "success_rate", "mean_adaptive_return",
                                "std_adaptive_return", "mean_nav_return", "std_nav_return", "mean_n_steps",
                                "std_n_steps", "mean_final_goal_distance_m", "mean_distance_traveled_m",
                                "std_distance_traveled_m", "mean_frame_count", "std_frame_count", "n_success",
                                "n_collision", "n_timeout", "policy_type", "frame_cost", "budget"]


def test_adaptive_return():
    assert M.adaptive_return(50.0, 121, 0.1, 500, 15.0) == 50.0 - 0.1 * 120          # reset frame free
    assert M.adaptive_return(50.0, 501, 0.0, 500, 15.0) == 35.0                       # budget penalty once
    assert M.adaptive_return(50.0, 500, 0.0, 500, 15.0) == 50.0


def _rows():
    g = np.random.default_rng(0)
    out = []
    for i in range(7):
        oc = ["success", "collision", "timeout"][i % 3]
        out.append({"episode": i, "mean_fps": float(g.uniform(0.2, 4)), "success": oc == "success", "outcome": oc,
                    "adaptive_return": float(g.normal()), "nav_return": float(g.normal()), "n_steps": int(g.integers(5, 350)),
                    "final_goal_distance_m": float(g.uniform()), "distance_traveled_m": float(g.uniform(0, 8)),
                    "episode_frame_count": int(g.integers(1, 281)), "policy_type": "adaptive", "frame_cost": 0.1,
                    "budget": 500})
    return out


def test_summarize_reference_definitions():
    rows = _rows()
    s = M.summarize(rows, "run")
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    assert s["std_nav_return"] == float(col("nav_return").std())                       # population std (np.std)
    assert s["mean_frame_count"] == float(col("episode_frame_count").mean())
    assert (s["n_success"], s["n_collision"], s["n_timeout"]) == (3, 2, 2)
    assert s["success_rate"] == 3 / 7 and list(s) == M.SUMMARY_FIELDS


def test_update_summary_replaces_and_sorts(tmp_path):
    p = str(tmp_path / "summary.csv")
    rows = _rows()
    for name, f in (("b", 2.0), ("a", 4.0), ("c", 0.5), ("b", 1.0)):
        s = M.summarize(rows, name)
        s["mean_fps"] = f
        M.update_summary_csv(p, s)
    got = list(csv.DictReader(open(p)))
    assert [(r["run_name"], float(r["mean_fps"])) for r in got] == [("c", 0.5), ("b", 1.0), ("a", 4.0)]
