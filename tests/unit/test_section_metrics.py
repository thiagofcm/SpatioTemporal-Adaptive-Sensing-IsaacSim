"""STS01 tables (AdaptiveSensing/evaluation/section_metrics.py) against the temporal reference-schema tables
(AdaptiveSensing/evaluation/metrics.py), on the stored TMP01 recordings (skipped when absent):
  * in-memory collision type / clearance = metrics.collision_type on the episode files
  * a synchronized episode row = metrics.episode_row (every reference column except policy_type)
  * FSE accounting: integer FSE when synchronized, ray-weighted otherwise; adaptive return with FSE
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from AdaptiveSensing.evaluation import metrics as M  # noqa: E402
from AdaptiveSensing.evaluation import section_metrics as SM  # noqa: E402

TMP01 = os.path.join(ROOT, "AdaptiveSensing/eval/temporal/TMP01_chase-ob1ob2_fixed-rate-nav05/ref_chase_c10/"
                           "2026-11-23_20-21-14_ppo_cnn_n4096_s42_model_1499_fixed1Hz")
FOOT = 0.175        # footprint radius (only shifts the clearance; the kind does not depend on it)
needs = pytest.mark.skipif(not os.path.isdir(TMP01), reason="TMP01 recordings not present")


def mem_inputs(ep):
    last = list(csv.DictReader(open(os.path.join(ep, "trajectory.csv"))))[-1]
    walls = json.load(open(os.path.join(ep, "world_geometry.json")))["walls"]
    ob = np.load(os.path.join(ep, "obstacles.npz"))
    xy = np.stack([ob["x"][-1], ob["y"][-1]], -1) if ob["x"].size else None
    return last, walls, xy, (ob["active"] if ob["x"].size else None), (float(ob["radius"][0]) if ob["x"].size else 0.0)


@needs
def test_collision_type_mem_equals_file_based():
    for e in range(100):
        ep = os.path.join(TMP01, f"episode_{e:04d}")
        kind_f, clear_f, _ = M.collision_type(ep, FOOT)
        last, walls, xy, act, r = mem_inputs(ep)
        kind_m, d = SM.collision_type_mem(float(last["x"]), float(last["y"]), walls, xy, act, r)
        assert kind_m == kind_f and d - FOOT == clear_f, e


@needs
def test_synchronized_row_equals_reference_row():
    rows = list(csv.DictReader(open(os.path.join(TMP01, "episodes.csv"))))
    w = (1564 / 4726, 1598 / 4726, 1564 / 4726)
    for e, ref in enumerate(rows[:30]):
        ep = os.path.join(TMP01, f"episode_{e:04d}")
        last, walls, xy, act, r = mem_inputs(ep)
        kind, _ = SM.collision_type_mem(float(last["x"]), float(last["y"]), walls, xy, act, r)
        row = SM.episode_row(episode=e, seed=int(ref["seed"]), hz=(1.0, 1.0, 1.0),
                             fse=int(ref["episode_frame_count"]) * 4726 / 4726,
                             env_outcome={"success": "goal", "collision": "safety_violation", "timeout": "timeout"}[ref["outcome"]],
                             n_steps=int(ref["n_steps"]), nav_return=float(ref["nav_return"]),
                             final_goal_distance_m=float(ref["final_goal_distance_m"]),
                             distance_traveled_m=float(ref["distance_traveled_m"]),
                             final_pose=(float(last["x"]), float(last["y"]), float(last["yaw"])), collision_kind=kind,
                             start=(float(ref["start_x"]), float(ref["start_y"]), float(ref["start_theta"])),
                             goal=(float(ref["goal_x"]), float(ref["goal_y"])), frame_cost=0.0, budget=500,
                             budget_penalty=15.0, weights=w, extra={})
        for k in M.EPISODE_FIELDS:
            if k != "policy_type":
                assert str(row[k]) == ref[k], (e, k, row[k], ref[k])


def test_fse_and_adaptive_return():
    w = (1564 / 4726, 1598 / 4726, 1564 / 4726)
    base = dict(episode=0, seed=0, env_outcome="goal", n_steps=10, nav_return=10.0, final_goal_distance_m=0.1,
                distance_traveled_m=1.0, final_pose=(0, 0, 0), collision_kind="wall", start=(0, 0, 0), goal=(0, 0),
                budget=500, budget_penalty=15.0, weights=w, extra={})
    sync = SM.episode_row(hz=(2.0, 2.0, 2.0), fse=(1 + 4) * 4726 / 4726, frame_cost=0.5, **base)
    assert sync["episode_frame_count"] == 5 and sync["fixed_fps"] == 2.0 and sync["adaptive_return"] == 8.0
    fse = (4726 + 2 * 1564 + 1598) / 4726                     # reset + left twice + center once
    mixed = SM.episode_row(hz=(4.0, 1.0, 0.2), fse=fse, frame_cost=0.5, **base)
    assert mixed["fixed_fps"] == "" and mixed["episode_frame_count"] == round(fse, 6)
    assert mixed["mean_fps"] == round(w[0] * 4 + w[1] * 1 + w[2] * 0.2, 6)
    assert mixed["adaptive_return"] == round(10.0 - 0.5 * (fse - 1), 3)
