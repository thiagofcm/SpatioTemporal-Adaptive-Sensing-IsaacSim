#!/usr/bin/env python3
"""Static figure of one recorded episode (reconstructed from its folder only; numpy + matplotlib):
map with the trajectory coloured by speed, obstacle tracks, LiDAR scans at selected steps, and time series of
action / command / measured velocity, clearance, goal distance and cumulative reward.

    python tools/plot_episode.py <eval_dir>/episode_0025                  # -> episode_0025/episode_0025.png
    python tools/plot_episode.py --eval <eval_dir> --episodes 0 25 73     # one figure per episode
    python tools/plot_episode.py <episode_dir> --scan-steps 0 40 80       # LiDAR scans drawn at these steps
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import episode_data as ED  # noqa: E402
import viz_common as V  # noqa: E402


def plot_episode(ep_dir, out=None, scan_steps=None):
    ep = ED.load_episode(ep_dir)
    meta, geo, T, St, O = ep["meta"], ep["geometry"], ep["traj"], ep["steps"], ep["obstacles"]
    n = int(meta["n_steps"])
    fig = plt.figure(figsize=(15, 7.5))
    is3d = ED.is_per_scan(ep)
    img3d = is3d and len(ep["lidar"]["grid_shape"]) == 2
    gs = fig.add_gridspec(5 + 2 * img3d, 2, width_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[:, 0])
    V.draw_walls(ax, geo)
    V.draw_obstacle_paths(ax, geo)
    V.draw_goal(ax, meta)
    for j in range(len(O["ids"])):                                  # obstacle tracks, fading with time
        if O["active"][j]:
            ax.scatter(O["x"][:, j], O["y"][:, j], c=np.linspace(0.15, 1, len(O["x"])), cmap="Purples", s=4, zorder=2)
            ax.add_patch(plt.Circle(V.last_obstacle_xy(ep, j), O["radius"][j], fc="tab:purple", alpha=0.5, ec="k"))
    pts = np.stack([T["x"], T["y"]], 1)
    lc = LineCollection(np.stack([pts[:-1], pts[1:]], 1), cmap="viridis", linewidths=2.5)
    lc.set_array(T["v_mps"][1:]); lc.set_clim(0, meta["robot"]["v_max_mps"])
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.01).set_label("v [m/s]", fontsize=8)
    scan_steps = [min(int(k), n) for k in (scan_steps if scan_steps is not None else [0, n // 2, n])]
    for k in scan_steps:
        if is3d:                                                    # 3D returns above the floor, top-down
            o, xy, z, floor = V.scan3d_topdown(ep, k)
            ax.scatter(xy[~floor, 0], xy[~floor, 1], s=1, color="tab:red", zorder=3, linewidths=0)
        else:
            o, ends, hit = ED.lidar_points(ep, k)
            ax.scatter(ends[hit, 0], ends[hit, 1], s=5, color="tab:red", zorder=3)
        ax.plot([o[0]], [o[1]], "o", color="tab:red", ms=3)
    ax.add_patch(plt.Circle((T["x"][-1], T["y"][-1]), meta["robot"]["footprint_radius_m"],
                            fc=V.OUTCOME_COLORS.get(meta["outcome"], "tab:blue"), alpha=0.5, ec="k"))
    V.setup_map(ax, geo, V.episode_title(meta) + "\nred dots: LiDAR hits at the marked steps" +
                (" (3D, floor returns omitted)" if is3d else "") + "; purple: obstacle tracks")
    if img3d:                                                       # range image of the last marked step's scan
        k = scan_steps[-1]
        im = V.range_image_axes(fig.add_subplot(gs[0:2, 1]), ep,
                                f"range image, scan in the observation at step {k} (row 0 = top, col 0 = left)")
        im.set_data(ED.range_image(ep, max(ED.scan_index(ep, k), 0)))
        fig.colorbar(im, ax=im.axes, fraction=0.03, pad=0.01).set_label("range [m]", fontsize=7)
    rows = [("speed [m/s]", [(T["t_sim"], T["v_mps"], "measured v"), (St["t_sim"], St["cmd_v_mps"], "command v")]),
            ("ang. rate [rad/s]", [(T["t_sim"], T["w_radps"], "measured w"), (St["t_sim"], St["cmd_w_radps"], "command w")]),
            ("clearance [m]", [(T["t_sim"], T["clearance_m"], "footprint clearance")]),
            ("goal dist. [m]", [(T["t_sim"], T["goal_distance_m"], None)]),
            ("cum. reward", [(St["t_sim"], St["cumulative_reward"], None)])]
    for i, (lab, curves) in enumerate(rows):
        a = fig.add_subplot(gs[i + 2 * img3d, 1])
        for t, y, l in curves:
            a.plot(t, y, lw=1, label=l)
        if lab.startswith("clearance"):
            a.axhline(meta["terminations"]["safety_margin_m"], color="tab:red", ls="--", lw=0.8, label="safety margin")
        if any(l for _, _, l in curves):
            a.legend(fontsize=7, loc="upper right")
        a.set_ylabel(lab, fontsize=8); a.grid(lw=0.3); a.tick_params(labelsize=7)
    a.set_xlabel("t [s]", fontsize=8)
    fig.suptitle(f"{meta.get('exp_id', '')}  {meta.get('protocol', '')}  {meta.get('policy', '')}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out) if out else Path(ep_dir) / f"{Path(ep_dir).name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("episode_dirs", nargs="*")
    p.add_argument("--eval")
    p.add_argument("--episodes", type=int, nargs="+")
    p.add_argument("--scan-steps", type=int, nargs="+", default=None)
    p.add_argument("--out", default=None, help="output path (single episode only)")
    a = p.parse_args()
    dirs = [Path(d) for d in a.episode_dirs] + (ED.episode_dirs(a.eval, a.episodes) if a.eval else [])
    if not dirs:
        p.error("give episode folders or --eval <dir> --episodes ...")
    for d in dirs:
        print(plot_episode(d, a.out if len(dirs) == 1 else None, a.scan_steps))


if __name__ == "__main__":
    main()
