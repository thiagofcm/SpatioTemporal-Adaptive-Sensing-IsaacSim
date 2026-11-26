"""Rewrite fixed-rate evaluation tables in the reference schema (AdaptiveSensing/evaluation/metrics.py) from the
stored recordings, without simulating. Recordings and metadata are not modified; the previous tables are kept.

For every evaluation folder (evaluate_fixed_rate.py output) under the given protocol folder(s):
    <eval>/legacy_v1/episodes.csv, summary.json     the previous tables (copied once, never overwritten)
    <eval>/episodes.csv                             reference schema, rebuilt from the legacy row + episode recording
    <eval>/summary.json                             previous content + "summary" (reference aggregation)
    <protocol>/legacy_v1/summary.csv, <protocol>/summary.csv
Sources: n_steps, outcome, nav_return (legacy episode_return), episode_frame_count (legacy accepted_scans), distances,
start / goal: legacy episodes.csv; seed: episode_NNNN/metadata.json random_seeds.episode_seed; final pose: last row of
trajectory.csv (terminal state); collision_type: terminal pose vs world_geometry.json walls and obstacles.npz.
mean_fps of a fixed-rate run is its requested rate (the rate in force never changes). Every reconstructed value is
checked against an independent stored value (metadata.json, trajectory.csv clearance / goal distance).

    python AdaptiveSensing/scripts/regenerate_eval_tables.py <protocol dir> [...] [--frame_cost 0] [--budget 500]
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from AdaptiveSensing.evaluation import metrics as M  # noqa: E402

FOOTPRINT_RADIUS = 0.177       # st_nav.envs.corridor_scene.FOOTPRINT_RADIUS (clearance = distance - footprint radius)
p = argparse.ArgumentParser()
p.add_argument("protocol_dirs", nargs="+")
p.add_argument("--frame_cost", type=float, default=0.0)
p.add_argument("--budget", type=int, default=500)
p.add_argument("--budget_penalty", type=float, default=15.0)
p.add_argument("--summary_name", default="summary.csv", help="protocol summary file to rebuild")
args = p.parse_args()


def legacy(src_dir, name):
    os.makedirs(os.path.join(src_dir, "legacy_v1"), exist_ok=True)
    dst = os.path.join(src_dir, "legacy_v1", name)
    if not os.path.exists(dst) and os.path.exists(os.path.join(src_dir, name)):
        shutil.copy2(os.path.join(src_dir, name), dst)
    return dst


report = {}
for proto in args.protocol_dirs:
    proto = os.path.abspath(proto)
    evals = sorted(d for d in os.listdir(proto) if os.path.isfile(os.path.join(proto, d, "episodes.csv")))
    legacy(proto, args.summary_name)
    summary_rows = []
    for name in evals:
        ed = os.path.join(proto, name)
        old_csv = legacy(ed, "episodes.csv")
        old_json = legacy(ed, "summary.json")
        old = list(csv.DictReader(open(old_csv)))
        cfg = json.load(open(os.path.join(ed, "run_config.json")))
        hz = float(cfg["sensing"]["requested_hz"])
        chk = {"episodes": len(old), "max_clearance_diff_m": 0.0, "max_goal_distance_diff_m": 0.0,
               "metadata_mismatches": 0, "trajectory_length_mismatches": 0}
        rows = []
        for r in old:
            ep = int(r["episode"])
            epd = os.path.join(ed, f"episode_{ep:04d}")
            meta = json.load(open(os.path.join(epd, "metadata.json")))
            row = M.episode_row(epd, episode=ep, seed=meta["random_seeds"]["episode_seed"], policy_type="fixed",
                                fixed_fps=hz, mean_fps=hz, env_outcome=r["outcome"], n_steps=int(r["n_steps"]),
                                nav_return=float(r["episode_return"]), frame_count=int(r["accepted_scans"]),
                                final_goal_distance_m=float(r["final_goal_distance_m"]),
                                distance_traveled_m=float(r["distance_traveled_m"]),
                                start=(float(r["start_x"]), float(r["start_y"]), float(r["start_theta"])),
                                goal=(float(r["goal_x"]), float(r["goal_y"])), frame_cost=args.frame_cost,
                                budget=args.budget, budget_penalty=args.budget_penalty, footprint_radius=FOOTPRINT_RADIUS)
            # independent checks against stored values
            _, clear, last = M.collision_type(epd, FOOTPRINT_RADIUS)
            chk["max_clearance_diff_m"] = max(chk["max_clearance_diff_m"], abs(clear - float(last["clearance_m"])))
            gd = math.hypot(row["goal_x"] - row["final_x"], row["goal_y"] - row["final_y"])
            chk["max_goal_distance_diff_m"] = max(chk["max_goal_distance_diff_m"], abs(gd - float(last["goal_distance_m"])))
            chk["trajectory_length_mismatches"] += int(last["step"]) != row["n_steps"]
            sg = meta["scan_gate"]
            chk["metadata_mismatches"] += int(meta["outcome"] != r["outcome"] or int(meta["n_steps"]) != row["n_steps"]
                                              or int(sg["accepted_scans"]) != row["episode_frame_count"]
                                              or abs(float(meta["episode_return"]) - row["nav_return"]) > 1e-3
                                              or float(sg["requested_hz"]) != hz)
            rows.append(row)
        M.write_episodes(os.path.join(ed, "episodes.csv"), rows)
        s = M.summarize(rows, name)
        js = json.load(open(old_json))
        js["summary"] = s
        js["tables_schema"] = ("AdaptiveSensing/evaluation/metrics.py (reference schema); regenerated from the recordings "
                               "by regenerate_eval_tables.py; previous tables in legacy_v1/")
        json.dump(js, open(os.path.join(ed, "summary.json"), "w"), indent=1)
        summary_rows.append(s)
        report[os.path.relpath(ed, ROOT)] = chk
    out = os.path.join(proto, args.summary_name)
    if os.path.exists(out):
        os.remove(out)
    for s in summary_rows:
        M.update_summary_csv(out, s)
json.dump(report, sys.stdout, indent=1)
print()
