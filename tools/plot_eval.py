#!/usr/bin/env python3
"""Aggregate figure of a recorded evaluation (or a selection of its episodes), reconstructed from the episode folders
only (numpy + matplotlib): all trajectories coloured by outcome over the scene (per-episode baffles / obstacle paths
drawn faintly), outcome counts, duration and minimum-clearance distributions, success by start pose, and the failed
episodes with their own geometry.

    python tools/plot_eval.py <eval_dir>                          # -> <eval_dir>/eval_overview.png
    python tools/plot_eval.py <eval_dir> --episodes 0 1 25 73     # -> <eval_dir>/eval_selection_0_1_25_73.png
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import episode_data as ED  # noqa: E402
import viz_common as V  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("eval_dir")
    p.add_argument("--episodes", type=int, nargs="+", default=None)
    p.add_argument("--max-failures", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    ev = ED.load_eval(a.eval_dir)
    eps = [ED.load_episode(d) for d in ED.episode_dirs(a.eval_dir, a.episodes)]
    metas = [e["meta"] for e in eps]
    m0 = metas[0]
    fails = [e for e in eps if e["meta"]["outcome"] != "goal"][:a.max_failures]
    nf = len(fails)
    fig = plt.figure(figsize=(20, 9.5 + (4.8 if nf else 0)))
    gs = fig.add_gridspec(2 + (1 if nf else 0), 4, height_ratios=[1.3, 1] + ([1] if nf else []))
    ax = fig.add_subplot(gs[0:2, 0:2])
    V.draw_walls(ax, {"walls": [w for w in eps[0]["geometry"]["walls"] if not w.get("name", "").startswith("Z")]})
    for e in eps:                                                    # per-episode randomized parts, faint
        V.draw_walls(ax, {"walls": [w for w in e["geometry"]["walls"] if w.get("name", "").startswith("Z")]}, alpha=0.04)
        V.draw_obstacle_paths(ax, e["geometry"], alpha=0.03)
    for e in eps:
        c = V.OUTCOME_COLORS.get(e["meta"]["outcome"], "gray")
        ok = e["meta"]["outcome"] == "goal"
        ax.plot(e["traj"]["x"], e["traj"]["y"], color=c, lw=0.7 if ok else 1.8, alpha=0.5 if ok else 1, zorder=3 if ok else 4)
    V.draw_goal(ax, m0)
    cnt = Counter(m["outcome"] for m in metas)
    n = len(metas)
    V.setup_map(ax, eps[0]["geometry"], f"{m0.get('exp_id', '')} {m0.get('protocol', '')}: {cnt.get('goal', 0)}/{n} success  "
                f"({', '.join(f'{k} {v}' for k, v in sorted(cnt.items()))})", fontsize=10)
    for k, col in V.OUTCOME_COLORS.items():
        ax.plot([], [], color=col, label=k)
    ax.legend(fontsize=8, loc="center right")
    a1 = fig.add_subplot(gs[0, 2])
    ks = list(V.OUTCOME_COLORS)
    a1.bar(ks, [cnt.get(k, 0) for k in ks], color=[V.OUTCOME_COLORS[k] for k in ks])
    a1.set_title("outcomes", fontsize=9); a1.tick_params(labelsize=7)
    a2 = fig.add_subplot(gs[0, 3])
    dur = [m["duration_s"] for m in metas if m["outcome"] == "goal"]
    a2.hist(dur, bins=20, color="tab:green"); a2.set_title(f"duration of successes [s] (mean {np.mean(dur) if dur else float('nan'):.1f})", fontsize=9)
    a3 = fig.add_subplot(gs[1, 2])
    mc = [m["min_clearance_m"] for m in metas]
    a3.hist(mc, bins=25, color="tab:blue"); a3.axvline(m0["terminations"]["safety_margin_m"], color="tab:red", ls="--", lw=1)
    a3.set_title("minimum clearance per episode [m] (red: safety margin)", fontsize=9)
    a4 = fig.add_subplot(gs[1, 3])
    groups = sorted({m["start"]["index"] for m in metas})
    a4.bar([str(g) for g in groups], [np.mean([m["success"] for m in metas if m["start"]["index"] == g]) for g in groups],
           color="tab:green")
    a4.set_ylim(0, 1.05); a4.set_title("success rate by start pose index", fontsize=9)
    for i, e in enumerate(fails):
        af = fig.add_subplot(gs[2, i])
        V.draw_walls(af, e["geometry"]); V.draw_obstacle_paths(af, e["geometry"]); V.draw_goal(af, e["meta"])
        af.plot(e["traj"]["x"], e["traj"]["y"], color="tab:red", lw=1.5)
        O = e["obstacles"]
        for j in range(len(O["ids"])):
            if O["active"][j]:
                af.add_patch(plt.Circle(V.last_obstacle_xy(e, j), O["radius"][j], fc="tab:purple", alpha=0.6, ec="k"))
        af.add_patch(plt.Circle((e["traj"]["x"][-1], e["traj"]["y"][-1]), e["meta"]["robot"]["footprint_radius_m"],
                                fc="tab:red", alpha=0.4, ec="k"))
        V.setup_map(af, e["geometry"], V.episode_title(e["meta"]), fontsize=8)
    fig.tight_layout()
    out = Path(a.out) if a.out else Path(a.eval_dir) / (
        "eval_overview.png" if a.episodes is None else "eval_selection_" + "_".join(map(str, a.episodes)) + ".png")
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
