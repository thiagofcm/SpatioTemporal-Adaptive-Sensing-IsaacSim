"""Compare the stochastic inputs (moving-obstacle trajectories, LiDAR noise draws) and outcomes of two recorded
evaluations of the same protocol: python tests/sim/compare_eval_inputs.py <eval_dir_a> <eval_dir_b>
Noise draw of a hit beam: u = (observed - clean) / (a(clean) * clean), a = datasheet relative accuracy; compared where
both evaluations have the same episode, row (acquisition time) and beam with a return in both."""
import csv
import sys

import numpy as np

sys.path.insert(0, "tools")
import episode_data as ED  # noqa: E402


def noise_u(ep, rows):
    L = ep["lidar"]
    obs, cl = L["ranges"][rows].astype(np.float64), L["ranges_clean"][rows].astype(np.float64)
    a = np.where(cl <= 3.0, 0.01, 0.02)
    ok = np.isfinite(obs) & (cl < float(L["range_max"]) - 1e-3) & (obs > 1e-3) & (obs < float(L["range_max"]) - 1e-3)
    return np.where(ok, (obs - cl) / (a * cl), np.nan)


A, B = sys.argv[1], sys.argv[2]
ra = list(csv.DictReader(open(f"{A}/episodes.csv"))); rb = list(csv.DictReader(open(f"{B}/episodes.csv")))
ob_max, ob_rows, u_max, u_n, same_oc, same_len = 0.0, 0, 0.0, 0, 0, 0
for i in range(len(ra)):
    a, b = ED.load_episode(f"{A}/{ED.episode_dir_name(i)}"), ED.load_episode(f"{B}/{ED.episode_dir_name(i)}")
    m = min(len(a["traj"]["x"]), len(b["traj"]["x"]))
    if a["obstacles"]["x"].shape[1]:
        d = np.abs(a["obstacles"]["x"][:m] - b["obstacles"]["x"][:m]) + np.abs(a["obstacles"]["y"][:m] - b["obstacles"]["y"][:m])
        ob_max = max(ob_max, float(np.nanmax(d))); ob_rows += m
    ua, ub = noise_u(a, slice(0, m - 1)), noise_u(b, slice(0, m - 1))
    both = np.isfinite(ua) & np.isfinite(ub)
    if both.any():
        u_max = max(u_max, float(np.abs(ua[both] - ub[both]).max())); u_n += int(both.sum())
    same_oc += ra[i]["outcome"] == rb[i]["outcome"]; same_len += ra[i]["n_steps"] == rb[i]["n_steps"]
sa = sum(r["success"] == "True" for r in ra); sb = sum(r["success"] == "True" for r in rb)
print(f"obstacle trajectories: max |diff| {ob_max:.2e} m over {ob_rows} common rows | LiDAR noise draws: max |diff| {u_max:.2e} "
      f"over {u_n} common (row, beam) returns | success {sa} vs {sb}, same outcome {same_oc}/{len(ra)}, same length {same_len}/{len(ra)}")
