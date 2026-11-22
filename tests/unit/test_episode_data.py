"""Evaluation recording format (tools/episode_data.py): write / read round trip, NaN handling, LiDAR geometry, protocol
summary, and the visualization tools running from the files alone (no torch / st_nav / Isaac Lab in the process)."""
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import episode_data as ED  # noqa: E402


def synthetic_episode(n=12):
    t = np.arange(n + 1) * 0.2
    x, y, yaw = 2.0 - 0.04 * np.arange(n + 1), np.full(n + 1, 1.0), np.full(n + 1, math.pi)
    traj = {"step": np.arange(n + 1), "t_sim": t, "x": x, "y": y, "yaw": yaw, "v_mps": np.full(n + 1, 0.2),
            "w_radps": np.zeros(n + 1), "goal_distance_m": np.hypot(x + 2, y + 2), "clearance_m": np.full(n + 1, 0.3),
            "distance_traveled_m": 0.04 * np.arange(n + 1)}
    steps = {k: np.zeros(n) for k in ED.STEP_FIELDS}
    steps.update({"step": np.arange(1, n + 1), "t_sim": t[1:], "reward": np.full(n, 0.1), "cumulative_reward": 0.1 * np.arange(1, n + 1),
                  "done": (np.arange(1, n + 1) == n).astype(int), "outcome": np.array([""] * (n - 1) + ["timeout"])})
    ang = np.arange(4) * math.pi / 2
    ranges = np.full((n + 1, 4), 1.0, dtype=np.float32)
    ranges[-1] = np.nan
    lidar = {"t_sim": t, "ranges": ranges, "ranges_clean": np.full((n + 1, 4), 1.0, dtype=np.float32), "fresh": np.ones(n + 1, bool),
             "beam_acquired": np.ones((n + 1, 4), bool), "beam_t_acq": np.repeat(t[:, None], 4, 1), "origin_x": x, "origin_y": y,
             "origin_yaw": yaw, "beam_angle_rad": ang.astype(np.float32), "range_max": np.float64(3.5),
             "mount_offset_xyz": np.zeros(3)}
    obstacles = {"t_sim": t, "ids": np.array(["OB1"]), "x": np.linspace(1.4, 2.0, n + 1)[:, None], "y": np.full((n + 1, 1), 1.7),
                 "radius": np.array([0.17]), "active": np.array([True])}
    geometry = {"scene": "test", "arena_interior_half_m": 2.835, "moving_obstacles": [
        {"id": "OB1", "active": True, "radius": 0.17, "path": [[1.4, 1.7], [2.4, 1.7]]}],
        "walls": [{"name": "outer_N", "cx": 0.0, "cy": 2.91, "length": 6.0, "thickness": 0.15, "yaw": 0.0}]}
    meta = {"episode_index": 3, "exp_id": "NAV99_test", "protocol": "p", "policy": "pol", "outcome": "timeout",
            "success": False, "n_steps": n, "duration_s": n * 0.2, "episode_return": 1.2, "min_clearance_m": 0.3,
            "nan_value": float("nan"), "goal": {"x": -2.0, "y": -2.0}, "start": {"x": 2.0, "y": 1.0, "yaw": math.pi, "index": 0},
            "robot": {"footprint_radius_m": 0.177, "v_max_mps": 0.22}, "control": {"control_dt_s": 0.2},
            "terminations": {"goal_tolerance_m": 0.2, "safety_margin_m": 0.05}}
    return meta, geometry, traj, steps, lidar, obstacles


def test_round_trip(tmp_path):
    meta, geo, traj, steps, lidar, obst = synthetic_episode()
    d = tmp_path / "eval" / ED.episode_dir_name(3)
    ED.write_episode(d, meta, geo, traj, steps, lidar, obst)
    assert sorted(p.name for p in d.iterdir()) == ["lidar.npz", "metadata.json", "obstacles.npz", "steps.csv",
                                                    "trajectory.csv", "world_geometry.json"]
    json.loads((d / "metadata.json").read_text())                    # valid JSON (NaN written as null)
    ep = ED.load_episode(d)
    assert ep["meta"]["nan_value"] is None and ep["meta"]["schema_version"] == ED.SCHEMA_VERSION
    assert ep["meta"]["rows"] == {"trajectory": 13, "steps": 12, "lidar": 13, "obstacles": 13}
    for k in ED.TRAJECTORY_FIELDS:
        assert np.allclose(ep["traj"][k], traj[k], atol=1e-6), k
    assert np.allclose(ep["steps"]["cumulative_reward"], steps["cumulative_reward"]) and ep["steps"]["outcome"][-1] == "timeout"
    assert np.isnan(ep["lidar"]["ranges"][-1]).all() and np.array_equal(ep["obstacles"]["ids"], obst["ids"])
    o, ends, hit = ED.lidar_points(ep, 0)                            # beam b at origin_yaw + angle, length = range
    assert np.allclose(ends[0], [1.0, 1.0], atol=1e-5) and hit.all()          # (2, 1) facing -x, range 1
    o, ends, hit = ED.lidar_points(ep, 12)                           # terminal row: falls back to ranges_clean
    assert np.isfinite(ends).all()


def test_protocol_summary(tmp_path):
    ED.update_protocol_summary(tmp_path, {"policy": "a", "success_rate": 0.5})
    ED.update_protocol_summary(tmp_path, {"policy": "b", "success_rate": 0.9})
    ED.update_protocol_summary(tmp_path, {"policy": "a", "success_rate": 0.7})
    rows = list(csv.DictReader(open(tmp_path / "summary.csv")))
    assert {r["policy"]: r["success_rate"] for r in rows} == {"a": "0.7", "b": "0.9"}


def test_visualization_tools_standalone(tmp_path):
    meta, geo, traj, steps, lidar, obst = synthetic_episode()
    ev = tmp_path / "eval"
    ED.write_episode(ev / ED.episode_dir_name(3), meta, geo, traj, steps, lidar, obst)
    with open(ev / "episodes.csv", "w") as f:
        f.write("episode,outcome\n3,timeout\n")
    (ev / "summary.json").write_text("{}")
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin"}
    probe = ("import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__'); "
             "bad = sorted({m.split('.')[0] for m in sys.modules} & {'torch', 'st_nav', 'isaaclab', 'omni'}); "
             "print('FORBIDDEN', bad)")
    for args in ([os.path.join(ROOT, "tools", "plot_episode.py"), str(ev / "episode_0003")],
                 [os.path.join(ROOT, "tools", "plot_eval.py"), str(ev)]):
        r = subprocess.run([sys.executable, "-B", "-c", probe] + args, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-1500:]
        assert "FORBIDDEN []" in r.stdout, r.stdout
    assert (ev / "episode_0003" / "episode_0003.png").exists() and (ev / "eval_overview.png").exists()
