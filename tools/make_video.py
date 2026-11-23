#!/usr/bin/env python3
"""Top-down MP4 of recorded evaluation episodes, reconstructed only from the episode folders (tools/episode_data.py):
world geometry, robot footprint and heading, trail, LiDAR beams (drawn from the sensor pose at acquisition), moving
obstacles, goal. 3D LiDAR episodes (per-scan layout): the scan's 3D returns projected top-down (coloured by height,
floor returns grey) and, for a single episode, the range image of the scan in the observation. Needs only numpy + matplotlib (+ ffmpeg); no st_nav, Isaac Lab or PyTorch.

    # one episode: map + timelines (speed, angular rate, clearance, goal distance) -> <episode_dir>/<episode>.mp4
    python tools/make_video.py <eval_dir>/episode_0025
    # several episodes side by side (grid)          -> <eval_dir>/episodes_0_1_25.mp4
    python tools/make_video.py --eval <eval_dir> --episodes 0 1 25
    python tools/make_video.py <dir_a>/episode_0003 <dir_b>/episode_0003 --out compare.mp4   # any folders

Options: --speed (playback speed vs simulated time, default 2), --no-lidar, --out.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import episode_data as ED  # noqa: E402
import viz_common as V  # noqa: E402


def ffmpeg_path():
    exe = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")
    if not os.path.exists(exe):
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise SystemExit("ffmpeg not found (PATH, ~/.local/bin or the imageio-ffmpeg package)")
    return exe


def make_panel(ax, ep, lidar=True):
    meta, geo = ep["meta"], ep["geometry"]
    V.draw_walls(ax, geo)
    V.draw_obstacle_paths(ax, geo, alpha=0.12)
    V.draw_goal(ax, meta)
    V.setup_map(ax, geo, V.episode_title(meta) + f"\nstart {meta['start']['index']}  " +
                (f"OB active {meta['moving_obstacles']['active']}" if "moving_obstacles" in meta else ""), fontsize=8)
    is3d = ED.is_per_scan(ep)
    rays = LineCollection([], colors="tab:red", linewidths=0.6, alpha=0.55)
    ax.add_collection(rays)
    pts = ax.scatter(np.zeros(0), np.zeros(0), s=1.5, zorder=3, linewidths=0) if is3d else None
    zmax = float(meta.get("geometry_heights_m", {}).get("wall", 0.6))
    trail, = ax.plot([], [], color="tab:blue", lw=1.5)
    body, head = V.robot_artists(ax, meta)
    obs = V.obstacle_artists(ax, ep)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=8, family="monospace", va="top",
                  bbox=dict(fc="white", alpha=0.8, ec="none"))
    T, n = ep["traj"], int(meta["n_steps"])

    def update(k):
        s = min(k, n)
        set_rays = []
        if lidar and is3d:
            o, xy, z, floor = V.scan3d_topdown(ep, s)
            pts.set_offsets(xy)
            pts.set_facecolors(V.scan3d_colors(z, floor, zmax))
        elif lidar:
            o, ends, hit = ED.lidar_points(ep, s)
            set_rays = [[tuple(o), tuple(e)] for e in ends]
        rays.set_segments(set_rays)
        trail.set_data(T["x"][:s + 1], T["y"][:s + 1])
        V.set_robot(body, head, T["x"][s], T["y"][s], T["yaw"][s])
        V.set_obstacles(obs, ep, s)
        done = k >= n
        body.set_facecolor(V.OUTCOME_COLORS.get(meta["outcome"], "tab:blue") if done else "tab:blue")
        txt.set_text(f"t {T['t_sim'][s]:5.1f} s  v {T['v_mps'][s]:4.2f} m/s\nclr {T['clearance_m'][s]:5.3f} m"
                     + (f"\n{meta['outcome'].upper().replace('_', ' ')}" if done else ""))
    return update


def render(eps, out, speed=2.0, lidar=True, dpi=100):
    single = len(eps) == 1
    dt = float(eps[0]["meta"]["control"]["control_dt_s"])
    n_max = max(int(e["meta"]["n_steps"]) for e in eps)
    frames = n_max + 1 + int(1.5 / dt)
    if single:
        ep = eps[0]
        img3d = lidar and ED.is_per_scan(ep) and len(ep["lidar"]["grid_shape"]) == 2
        fig = plt.figure(figsize=(14, 8.5 if img3d else 7))
        gs = fig.add_gridspec(4 + 2 * img3d, 2, width_ratios=[1.15, 1])
        ax_map = fig.add_subplot(gs[:, 0])
        upd = [make_panel(ax_map, ep, lidar)]
        if img3d:
            im = V.range_image_axes(fig.add_subplot(gs[0:2, 1]), ep)
            fig.colorbar(im, ax=im.axes, fraction=0.03, pad=0.01).set_label("range [m]", fontsize=7)
            n_ep = int(ep["meta"]["n_steps"])
            upd.append(lambda k: im.set_data(ED.range_image(ep, max(ED.scan_index(ep, min(k, n_ep)), 0))))
        T, St = ep["traj"], ep["steps"]
        series = [("v_mps", "v [m/s]", St.get("cmd_v_mps")), ("w_radps", "w [rad/s]", St.get("cmd_w_radps")),
                  ("clearance_m", "clearance [m]", None), ("goal_distance_m", "goal distance [m]", None)]
        cursors = []
        for i, (k, lab, cmd) in enumerate(series):
            a = fig.add_subplot(gs[i + 2 * img3d, 1])
            a.plot(T["t_sim"], T[k], color="tab:blue", lw=1, label="measured" if cmd is not None else None)
            if cmd is not None:
                a.step(St["t_sim"], cmd, where="pre", color="k", lw=0.8, alpha=0.6, label="command")
                a.legend(fontsize=7, loc="upper right")
            if k == "clearance_m":
                a.axhline(ep["meta"]["terminations"]["safety_margin_m"], color="tab:red", lw=0.8, ls="--")
            a.set_ylabel(lab, fontsize=8); a.grid(lw=0.3); a.tick_params(labelsize=7)
            cursors.append(a.axvline(0, color="tab:red", lw=1))
        a.set_xlabel("t [s]", fontsize=8)
        upd.append(lambda k: [c.set_xdata([T["t_sim"][min(k, len(T["t_sim"]) - 1)]] * 2) for c in cursors])
    else:
        nc = min(3, len(eps)); nr = math.ceil(len(eps) / nc)
        fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 5.4 * nr))
        axs = np.atleast_1d(axs).ravel()
        upd = [make_panel(ax, ep, lidar) for ax, ep in zip(axs, eps)]
        for ax in axs[len(eps):]:
            ax.axis("off")
    m = eps[0]["meta"]
    sup = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path()
    w = FFMpegWriter(fps=speed / dt, bitrate=3000, extra_args=["-pix_fmt", "yuv420p"])
    with w.saving(fig, str(out), dpi=dpi):
        for f in range(frames):
            for u in upd:
                u(f)
            sup.set_text(f"{m.get('exp_id', '')}  {m.get('protocol', '')}  {m.get('policy', '')}  |  "
                         f"t = {min(f, n_max) * dt:5.1f} s  ({speed:g}x)")
            w.grab_frame()
    plt.close(fig)
    return out, frames


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("episode_dirs", nargs="*", help="episode folder(s)")
    p.add_argument("--eval", help="evaluation folder (with --episodes)")
    p.add_argument("--episodes", type=int, nargs="+", help="episode indices in --eval")
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--no-lidar", dest="lidar", action="store_false")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    dirs = [Path(d) for d in a.episode_dirs]
    if a.eval:
        dirs += ED.episode_dirs(a.eval, a.episodes)
    if not dirs:
        p.error("give episode folders or --eval <dir> --episodes ...")
    eps = [ED.load_episode(d) for d in dirs]
    if a.out:
        out = Path(a.out)
    elif len(eps) == 1:
        out = dirs[0] / f"{dirs[0].name}.mp4"
    else:
        base = Path(a.eval) if a.eval else dirs[0].parent
        out = base / ("episodes_" + "_".join(str(e["meta"]["episode_index"]) for e in eps) + ".mp4")
    out, frames = render(eps, out, a.speed, a.lidar)
    print(out, frames, "frames")


if __name__ == "__main__":
    main()
