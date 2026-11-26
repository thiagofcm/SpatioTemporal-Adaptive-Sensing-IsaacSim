"""Frame-cost calibration on the TMP01 fixed-rate evaluations (no simulation): how a per-acquisition cost changes the
ranking of the fixed sensing rates by expected episode return, with the TMP02 reward and counting convention.

Per episode of each TMP01 recording (ref_chase_c10, 100 identical episodes per rate):
    navigation_return    = episodes.csv nav_return (sum of the unchanged NAV05 env reward)
    charged_acquisitions = episode_frame_count - 1: TMP01 counts include the reset scan and never the scan at the episode's
                           end instant, exactly as TMP02's frame count (validated: tmp02_validation --fixed); TMP02
                           charges every accepted scan except the reset scan (frames - 1 = causal ticks), including one
                           accepted on the last control step
    penalized_return     = navigation_return - frame_cost * charged_acquisitions        (TMP02 sensing reward, summed)

Crossover of rates A, B: the cost at which their mean penalized returns are equal,
c* = (mean R_A - mean R_B) / (mean N_A - mean N_B); 95 % interval from a paired bootstrap over episodes.

    python AdaptiveSensing/scripts/frame_cost_calibration.py [--variant_2hz <evaluate_fixed_rate.py output dir>]
Output: AdaptiveSensing/eval/temporal/TMP02_chase-ob1ob2_adaptive-nav05/frame_cost_calibration/
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
TMP01 = "AdaptiveSensing/eval/temporal/TMP01_chase-ob1ob2_fixed-rate-nav05/ref_chase_c10"
RUN = "2026-11-23_20-21-14_ppo_cnn_n4096_s42_model_1499_fixed{}Hz"
RATES = [0.2, 0.5, 1.0, 2.0, 4.0]
CONTROL_DT = 0.2
COSTS = [0.0, 0.025, 0.05, 0.1, 0.2]
OUT = "AdaptiveSensing/eval/temporal/TMP02_chase-ob1ob2_adaptive-nav05/frame_cost_calibration"
p = argparse.ArgumentParser()
p.add_argument("--variant_2hz", default=None, help="use this 2 Hz recording instead of the stored one (sensitivity)")
p.add_argument("--grid_max", type=float, default=4.5)
p.add_argument("--grid_step", type=float, default=0.005)
p.add_argument("--bootstrap", type=int, default=5000)
p.add_argument("--tag", default=None)
args = p.parse_args()


def load(hz):
    d = args.variant_2hz if (hz == 2.0 and args.variant_2hz) else os.path.join(ROOT, TMP01, RUN.format(f"{hz:g}"))
    rows = list(csv.DictReader(open(os.path.join(d, "episodes.csv"))))
    f = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    return {"dir": os.path.relpath(d, ROOT), "R": f("nav_return"), "N": f("episode_frame_count") - 1,
            "steps": f("n_steps"), "dur": f("n_steps") * CONTROL_DT, "outcome": np.array([r["outcome"] for r in rows]),
            "episode": np.array([int(r["episode"]) for r in rows])}


D = {hz: load(hz) for hz in RATES}
ep0 = D[RATES[0]]["episode"]
assert all(np.array_equal(D[h]["episode"], ep0) for h in RATES), "episodes must be paired"
n_ep = len(ep0)


def stats(hz, c):
    d = D[hz]
    pen = d["R"] - c * d["N"]
    return {"rate_hz": hz, "frame_cost": c, "penalized_return_mean": pen.mean(), "penalized_return_std": pen.std(ddof=1),
            "navigation_return_mean": d["R"].mean(),
            "success_rate": float((d["outcome"] == "success").mean()),
            "collision_rate": float((d["outcome"] == "collision").mean()),
            "timeout_rate": float((d["outcome"] == "timeout").mean()),
            "charged_acquisitions_mean": d["N"].mean(), "effective_rate_hz_mean": (d["N"] / d["dur"]).mean(),
            "episode_duration_s_mean": d["dur"].mean(), "penalized_reward_per_step_mean": (pen / d["steps"]).mean()}


os.makedirs(os.path.join(ROOT, OUT), exist_ok=True)
sfx = f"_{args.tag}" if args.tag else ""
main = [stats(hz, c) for c in COSTS for hz in RATES]
with open(os.path.join(ROOT, OUT, f"fixed_rate_penalized_returns{sfx}.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(main[0]))
    w.writeheader()
    w.writerows([{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in main])

grid = np.round(np.arange(0.0, args.grid_max + 1e-12, args.grid_step), 6)
M = {hz: np.array([(D[hz]["R"] - c * D[hz]["N"]).mean() for c in grid]) for hz in RATES}
best = [RATES[int(np.argmax([M[h][i] for h in RATES]))] for i in range(len(grid))]
with open(os.path.join(ROOT, OUT, f"dense_grid{sfx}.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame_cost"] + [f"mean_penalized_return_{h:g}Hz" for h in RATES] + ["best_rate_hz", "ranking_best_first"])
    for i, c in enumerate(grid):
        order = sorted(RATES, key=lambda h: -M[h][i])
        w.writerow([c] + [round(M[h][i], 4) for h in RATES] + [best[i], " > ".join(f"{h:g}" for h in order)])

rng = np.random.default_rng(0)
idx = rng.integers(0, n_ep, (args.bootstrap, n_ep))
cross = []
for a, b in itertools.combinations(RATES, 2):
    dR, dN = D[a]["R"] - D[b]["R"], D[a]["N"] - D[b]["N"]
    c = dR.mean() / dN.mean()
    bs = dR[idx].mean(1) / dN[idx].mean(1)
    cross.append({"pair": f"{a:g} vs {b:g} Hz", "crossover_cost": round(float(c), 4),
                  "ci95": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])],
                  "better_below_crossover": f"{(a if dR.mean() > 0 else b):g} Hz",
                  "mean_nav_return_diff": round(float(dR.mean()), 3), "mean_charged_diff": round(float(dN.mean()), 2),
                  "paired_nav_return_diff_se": round(float(dR.std(ddof=1) / np.sqrt(n_ep)), 3)})
changes, prev = [], None
for i, c in enumerate(grid):
    order = tuple(sorted(RATES, key=lambda h: -M[h][i]))
    if order != prev:
        changes.append({"from_cost": float(c), "ranking_best_first": [f"{h:g} Hz" for h in order]})
        prev = order
json.dump({"source": {f"{h:g}Hz": D[h]["dir"] for h in RATES}, "episodes": n_ep,
           "convention": "charged = episode_frame_count - 1 (reset scan not charged; scan at the episode-end instant never "
                         "gated; a scan accepted on the last control step is charged) = TMP02",
           "costs": COSTS, "grid": [float(grid[0]), float(grid[-1]), args.grid_step],
           "crossovers": cross, "ranking_changes_on_grid": changes},
          open(os.path.join(ROOT, OUT, f"crossovers{sfx}.json"), "w"), indent=1)

print(f"{'cost':>6} {'Hz':>4} {'pen.ret':>9} {'sd':>7} {'nav.ret':>8} {'succ':>5} {'coll':>5} {'tout':>5} "
      f"{'charged':>8} {'eff.Hz':>7} {'dur s':>6} {'pen/step':>9}")
for r in main:
    print(f"{r['frame_cost']:6.3f} {r['rate_hz']:4g} {r['penalized_return_mean']:9.2f} {r['penalized_return_std']:7.2f} "
          f"{r['navigation_return_mean']:8.2f} {r['success_rate']:5.2f} {r['collision_rate']:5.2f} {r['timeout_rate']:5.2f} "
          f"{r['charged_acquisitions_mean']:8.1f} {r['effective_rate_hz_mean']:7.3f} {r['episode_duration_s_mean']:6.1f} "
          f"{r['penalized_reward_per_step_mean']:9.4f}")
print("\ncrossovers:")
for x in cross:
    print("  ", x)
print("\nranking changes on the grid:")
for x in changes:
    print("  ", x)
