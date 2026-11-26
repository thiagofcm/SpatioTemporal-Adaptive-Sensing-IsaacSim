#!/usr/bin/env python3
"""3D point-cloud MP4 of recorded 3D-LiDAR evaluation episodes (per-scan layout, e.g. NAV04): the points actually
captured (observed, noisy ranges of lidar.npz, rebuilt as origin + range * direction, tools/episode_data.py),
coloured by range. No st_nav, Isaac Lab, PyTorch or GPU; numpy + matplotlib + ffmpeg.

Default output: the point cloud alone (no scene geometry) in the robot frame at the scan's acquisition (x forward,
y left, z up, origin = sensor position projected on the floor), perspective camera behind and above the sensor, dark
background, 1 m floor grid; plus a miniature top-down map (walls, goal, trail, robot, field of view).
  --depth   add the depth (range) image the policy received
  --scene   instead of the robot-frame cloud: world-frame 3D view of the scene (walls / baffles, robot, trail, goal)
            with the captured points; --camera follow|fixed

One frame per control step (5 Hz); the scan shown is the one in the observation at that step (acquired at 4 Hz, held
between acquisitions). With --accumulate N the N previous scans are also drawn, faded (what the robot observed
recently; the policy itself only receives the current scan).

    # one episode -> <episode_dir>/<episode>_cloud.mp4 (with --scene: <episode>_cloud_scene.mp4)
    python tools/make_cloud_video.py <eval_dir>/episode_0005
    python tools/make_cloud_video.py --eval <eval_dir> --episodes 5 7 42          # one video per episode
    python tools/make_cloud_video.py <eval_dir>/episode_0005 --depth --accumulate 8
    python tools/make_cloud_video.py <eval_dir>/episode_0005 --scene --camera fixed

Options: --depth, --scene, --camera follow|fixed (--scene only), --accumulate N (default 0), --speed (default 2),
--depth-max (colour scale, default 8 m), --clean (noise-free ranges), --workers, --dpi, --out.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import episode_data as ED  # noqa: E402
import viz_common as V  # noqa: E402
from make_pov_video import DEPTH_CMAP, box_faces, check_mount, cyl_faces, obstacles_at, quat_to_R, scene_boxes  # noqa: E402
from make_video import ffmpeg_path  # noqa: E402

ROBOT_H = 0.35          # m: drawn robot height in --scene (TurtleBot4 top plate ~0.35 m)
CLOUD_FWD = (-0.3, 6.5)  # m: robot-frame view box (forward)
CLOUD_LAT = 2.0          # m: half width (left / right); the FoV is +-14 deg (+-1.6 m at 6.5 m)
CLOUD_Z = 0.8            # m: height
BG = "#0b0d12"
_G = {}


def smoothed_heading(T, tau_s=1.5):
    """Camera heading for --scene --camera follow: robot yaw low-pass filtered (time constant tau_s), unwrapped."""
    yaw = np.unwrap(np.asarray(T["yaw"], float))
    t = np.asarray(T["t_sim"], float)
    out = yaw.copy()
    for i in range(1, len(yaw)):
        a = 1.0 - math.exp(-(t[i] - t[i - 1]) / tau_s)
        out[i] = out[i - 1] + a * (yaw[i] - out[i - 1])
    return out


def scans_to_draw(s):
    """Scan indices drawn at current scan s (oldest first) and their opacity."""
    acc = _G["accumulate"]
    js = list(range(max(0, s - acc), s + 1))
    return [(j, 1.0 if j == s else 0.12 + 0.45 * (1 - (s - j) / (acc + 1))) for j in js]


# --------------------------------------------------------------------------------------------- main panels
def draw_cloud(ax, s):
    """Captured points in the robot frame of scan s (forward, left, up), coloured by range; no scene geometry."""
    ep, dmax = _G["ep"], _G["depth_max"]
    L = ep["lidar"]
    R = quat_to_R(L["base_quat_wxyz"][s]) @ _G["Rm"]
    o = L["origin_xyz"][s].astype(float)
    fwd = math.atan2(R[1, 0], R[0, 0])                             # sensor heading (yaw of the forward axis)
    c, si = math.cos(fwd), math.sin(fwd)

    def to_robot(p):
        d = p - o
        return np.c_[d[:, 0] * c + d[:, 1] * si, -d[:, 0] * si + d[:, 1] * c, p[:, 2]]

    ax.set_facecolor(BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(False)
    ax.set_axis_off()
    f0, f1 = CLOUD_FWD
    for g in np.arange(math.ceil(f0), f1 + 1e-6, 1.0):            # 1 m floor grid
        ax.plot([g, g], [-CLOUD_LAT, CLOUD_LAT], [0, 0], color="0.28", lw=0.5)
    for g in np.arange(-CLOUD_LAT, CLOUD_LAT + 1e-6, 1.0):
        ax.plot([f0, f1], [g, g], [0, 0], color="0.28", lw=0.5)
    for g in range(1, int(f1) + 1):
        ax.text(g, -CLOUD_LAT - 0.15, 0, f"{g} m", color="0.55", fontsize=7, ha="center")
    # field-of-view edges on the floor plane and the sensor
    az = np.radians(L["azimuth_deg"])
    for a_ in (az.min(), az.max()):
        d = R @ np.array([math.cos(a_), math.sin(a_), 0.0])
        h = math.atan2(d[1], d[0]) - fwd
        ax.plot([0, f1 * math.cos(h)], [0, f1 * math.sin(h)], [0, 0], color="0.45", lw=0.6, ls="--")
    ax.plot([0, 0], [0, 0], [0, o[2]], color="0.7", lw=1.0)
    ax.scatter([0], [0], [o[2]], c="white", s=25, marker="^", depthshade=False)
    n_hit = 0
    for j, alpha in scans_to_draw(s):
        sc = ED.lidar_scan(ep, j, clean=_G["clean"])
        h = sc["hit"]
        q = to_robot(sc["points"][h])
        keep = (q[:, 0] >= f0) & (q[:, 0] <= f1) & (np.abs(q[:, 1]) <= CLOUD_LAT) & (q[:, 2] <= CLOUD_Z)
        q, r = q[keep], sc["ranges"][h][keep]
        order = np.argsort(-r)                                       # far first, near on top
        ax.scatter(q[order, 0], q[order, 1], q[order, 2], c=r[order], cmap=DEPTH_CMAP, vmin=0, vmax=dmax,
                   s=7.0 if j == s else 3.0, alpha=alpha, depthshade=False, linewidths=0)
        if j == s:
            n_hit = int(h.sum())
    ax.set_xlim(*CLOUD_FWD); ax.set_ylim(-CLOUD_LAT, CLOUD_LAT); ax.set_zlim(0, CLOUD_Z)
    ax.set_box_aspect((f1 - f0, 2 * CLOUD_LAT, CLOUD_Z), zoom=2.1)
    ax.set_proj_type("persp", focal_length=0.6)
    ax.view_init(elev=16, azim=180)                                  # camera behind the sensor, looking forward
    return n_hit


def draw_scene(ax, s, row):
    """World-frame 3D view: walls / baffles, robot, trail, goal and the captured points (the --scene view)."""
    ep, dmax = _G["ep"], _G["depth_max"]
    L, T, meta = ep["lidar"], ep["traj"], ep["meta"]
    half = _G["half"]
    boxes = _G["boxes"]
    faces = [f for b in boxes for f in box_faces(b)]
    colors = [(0.95, 0.80, 0.62, 0.10) if b[6] else (0.6, 0.6, 0.62, 0.10) for b in boxes for _ in range(5)]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolors=(0.35, 0.35, 0.35, 0.35), linewidths=0.3))
    obst = obstacles_at(ep, row)
    if obst:
        ax.add_collection3d(Poly3DCollection([f for c in obst for f in cyl_faces(*c)], facecolors=(0.58, 0.4, 0.74, 0.15),
                                             edgecolors="none"))
    x, y, yaw = float(T["x"][row]), float(T["y"][row]), float(T["yaw"][row])
    rr = float(meta["robot"]["footprint_radius_m"])
    rc = V.OUTCOME_COLORS.get(meta["outcome"], "tab:blue") if row >= int(meta["n_steps"]) else "tab:blue"
    ax.add_collection3d(Poly3DCollection(cyl_faces(x, y, rr, ROBOT_H, 20), facecolors=matplotlib.colors.to_rgba(rc, 0.25),
                                         edgecolors="none"))
    ax.plot([x, x + 0.3 * math.cos(yaw)], [y, y + 0.3 * math.sin(yaw)], [ROBOT_H] * 2, color="navy", lw=2)
    ax.plot(T["x"][:row + 1], T["y"][:row + 1], np.zeros(row + 1), color="tab:blue", lw=1.2)
    g = meta["goal"]
    ax.plot([g["x"]], [g["y"]], [0.0], "*", color="darkgreen", ms=12)
    n_hit = 0
    for j, alpha in scans_to_draw(s):
        sc = ED.lidar_scan(ep, j, clean=_G["clean"])
        h = sc["hit"]
        p, r = sc["points"][h], sc["ranges"][h]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=r, cmap=DEPTH_CMAP, vmin=0, vmax=dmax, s=3.0 if j == s else 1.2,
                   alpha=alpha, depthshade=False, linewidths=0)
        if j == s:
            n_hit = int(h.sum())
    o = L["origin_xyz"][s].astype(float)
    R = quat_to_R(L["base_quat_wxyz"][s]) @ _G["Rm"]
    az, el = np.radians(L["azimuth_deg"]), np.radians(L["elevation_deg"])
    for a_ in (az.min(), az.max()):
        for e_ in (el.min(), el.max()):
            d = R @ np.array([math.cos(e_) * math.cos(a_), math.cos(e_) * math.sin(a_), math.sin(e_)])
            ax.plot(*np.c_[o, o + 1.0 * d], color="k", lw=0.6, alpha=0.6)
    ax.scatter([o[0]], [o[1]], [o[2]], c="k", s=15, marker="^", depthshade=False)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(0, 1.2)
    ax.set_box_aspect((2 * half, 2 * half, 1.2), zoom=1.25)
    if _G["camera"] == "follow":
        ax.view_init(elev=32, azim=math.degrees(_G["heading"][row]) + 180)
    else:
        ax.view_init(elev=40, azim=-60)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x [m]", fontsize=7); ax.set_ylabel("y [m]", fontsize=7); ax.set_zlabel("z [m]", fontsize=7)
    return n_hit


# --------------------------------------------------------------------------------------------- small panels
def draw_minimap(ax, s, row):
    ep = _G["ep"]
    L, T, meta = ep["lidar"], ep["traj"], ep["meta"]
    V.draw_walls(ax, ep["geometry"])
    V.draw_goal(ax, meta)
    for ox, oy, orad, _ in obstacles_at(ep, row):
        ax.add_patch(Circle((ox, oy), orad, fc="tab:purple", alpha=0.6, ec="k"))
    ax.plot(T["x"][:row + 1], T["y"][:row + 1], color="tab:blue", lw=1.0)
    o = L["origin_xyz"][s]
    R = quat_to_R(L["base_quat_wxyz"][s]) @ _G["Rm"]
    fwd = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    az = L["azimuth_deg"]
    ax.add_patch(Wedge((o[0], o[1]), 8.0, fwd + float(az.min()), fwd + float(az.max()), fc="gold", alpha=0.25, ec="none"))
    x, y, yaw = float(T["x"][row]), float(T["y"][row]), float(T["yaw"][row])
    rc = V.OUTCOME_COLORS.get(meta["outcome"], "tab:blue") if row >= int(meta["n_steps"]) else "tab:blue"
    ax.add_patch(Circle((x, y), meta["robot"]["footprint_radius_m"], fc=rc, alpha=0.6, ec="navy"))
    ax.plot([x, x + 0.3 * math.cos(yaw)], [y, y + 0.3 * math.sin(yaw)], color="navy", lw=1.5)
    V.setup_map(ax, ep["geometry"])
    ax.set_facecolor("white")


def draw_depth(ax, s, fig):
    ep, dmax = _G["ep"], _G["depth_max"]
    L = ep["lidar"]
    ext = (float(L["azimuth_deg"].max()), float(L["azimuth_deg"].min()), float(L["elevation_deg"].min()),
           float(L["elevation_deg"].max()))
    im = ax.imshow(ED.range_image(ep, s, clean=_G["clean"]), cmap=DEPTH_CMAP, vmin=0, vmax=dmax, aspect="auto",
                   interpolation="nearest", extent=ext)
    ax.set_title(f"depth image = policy input ({'noise-free' if _G['clean'] else 'observed'}), "
                 f"{int(L['grid_shape'][0])} x {int(L['grid_shape'][1])}", fontsize=8, color=_G["fg"])
    ax.set_xlabel("azimuth [deg] (left = +)", fontsize=7, color=_G["fg"])
    ax.set_ylabel("elev. [deg]", fontsize=7, color=_G["fg"])
    ax.tick_params(labelsize=6, colors=_G["fg"])


# --------------------------------------------------------------------------------------------- frame
def render_frame(k, path):
    ep, dmax = _G["ep"], _G["depth_max"]
    L, T, meta = ep["lidar"], ep["traj"], ep["meta"]
    n = int(meta["n_steps"])
    row = min(k, n)
    s = max(ED.scan_index(ep, row), 0)
    scene, depth = _G["scene"], _G["depth"]
    fg = "black" if scene else "0.9"
    _G["fg"] = fg
    fig = plt.figure(figsize=(16, 9), facecolor="white" if scene else BG)
    main_rect = (0.0, 0.25 if depth else 0.0, 0.93, 0.70 if depth else 0.92)
    ax = fig.add_axes(main_rect, projection="3d")
    n_hit = draw_scene(ax, s, row) if scene else draw_cloud(ax, s)
    if not scene:
        ax.set_facecolor(BG)
    # colour bar
    cax = fig.add_axes((0.94, 0.32 if depth else 0.15, 0.012, 0.5 if depth else 0.62))
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DEPTH_CMAP, norm=plt.Normalize(0, dmax)), cax=cax)
    cb.set_label(f"range [m] (scale clipped at {dmax:g} m)", fontsize=8, color=fg)
    cb.ax.tick_params(labelsize=7, colors=fg)
    # miniature map (top right)
    mm = fig.add_axes((0.775, 0.66, 0.155, 0.275))
    draw_minimap(mm, s, row)
    # depth image (bottom strip)
    if depth:
        draw_depth(fig.add_axes((0.18, 0.05, 0.6, 0.19)), s, fig)
    done = k >= n
    acc = len(scans_to_draw(s)) - 1
    status = f"   {meta['outcome'].upper().replace('_', ' ')}" if done else ""
    fig.text(0.01, 0.985, f"{meta.get('exp_id', '')}  {meta.get('protocol', '')}  {meta.get('policy', '')}  |  "
             f"{V.episode_title(meta)}", fontsize=9, color=fg, va="top")
    fig.text(0.01, 0.955, f"t = {row * float(meta['control']['control_dt_s']):5.1f} s   step {row}   scan {s} "
             f"(acquired at {float(L['scan_t_sim'][s]):.2f} s; 4 Hz, held between scans)   "
             f"v {T['v_mps'][row]:4.2f} m/s   clearance {T['clearance_m'][row]:5.3f} m{status}",
             fontsize=10, color=fg, va="top", family="monospace")
    fig.text(0.01, 0.925, f"captured points ({'noise-free' if _G['clean'] else 'observed, noisy'}) coloured by range: "
             f"{n_hit} / {len(L['ranges'][s])} returns" + (f", + previous {acc} scan(s) faded" if acc else "")
             + ("" if scene else "  |  robot frame at acquisition (forward, left, up), 1 m floor grid, dashed = FoV")
             + f"  |  mount z {float(L['mount_offset_xyz'][2]):.2f} m, pitch {float(L['mount_rpy_deg'][1]):g} deg",
             fontsize=8, color=fg, va="top")
    fig.savefig(path, dpi=_G["dpi"], facecolor=fig.get_facecolor())
    plt.close(fig)


def _work(args):
    render_frame(*args)


def render(ep, out, speed=2.0, depth_max=8.0, clean=False, accumulate=0, camera="follow", scene=False, depth=False,
           workers=8, dpi=90):
    if not ED.is_per_scan(ep):
        raise SystemExit(f"{ep['dir']}: not a 3D-LiDAR (per-scan) episode; use tools/make_video.py for 2D episodes")
    if len(ep["lidar"]["grid_shape"]) != 2:
        raise SystemExit("this ray set has no range-image grid")
    _G.update(ep=ep, Rm=check_mount(ep), boxes=scene_boxes(ep), depth_max=depth_max, clean=clean, accumulate=accumulate,
              camera=camera, scene=scene, depth=depth, heading=smoothed_heading(ep["traj"]), dpi=dpi,
              half=float(ep["geometry"].get("arena_interior_half_m", 2.835)) + 0.15)
    dt = float(ep["meta"]["control"]["control_dt_s"])
    frames = int(ep["meta"]["n_steps"]) + 1 + int(1.5 / dt)
    tmp = Path(tempfile.mkdtemp(prefix="cloud_"))
    try:
        jobs = [(k, tmp / f"f{k:05d}.png") for k in range(frames)]
        if workers > 1:
            with get_context("fork").Pool(workers) as pool:
                pool.map(_work, jobs, chunksize=4)
        else:
            for j in jobs:
                _work(j)
        subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-framerate", f"{speed / dt:g}", "-i",
                        str(tmp / "f%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", "-crf", "20", str(out)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out, frames


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("episode_dirs", nargs="*", help="episode folder(s) of a 3D-LiDAR evaluation")
    p.add_argument("--eval", help="evaluation folder (with --episodes)")
    p.add_argument("--episodes", type=int, nargs="+", help="episode indices in --eval")
    p.add_argument("--depth", action="store_true", help="add the depth image the policy received")
    p.add_argument("--scene", action="store_true", help="world-frame 3D view with the scene geometry")
    p.add_argument("--camera", choices=("follow", "fixed"), default="follow", help="--scene camera")
    p.add_argument("--accumulate", type=int, default=0, help="also draw the N previous scans (faded)")
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--depth-max", dest="depth_max", type=float, default=8.0)
    p.add_argument("--clean", action="store_true", help="noise-free ranges instead of the observed ones")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--out", default=None, help="output file (one episode only)")
    a = p.parse_args()
    dirs = [Path(d) for d in a.episode_dirs]
    if a.eval:
        if not a.episodes:
            p.error("--eval needs --episodes (no video for every episode)")
        dirs += ED.episode_dirs(a.eval, a.episodes)
    if not dirs:
        p.error("give episode folders or --eval <dir> --episodes ...")
    if a.out and len(dirs) > 1:
        p.error("--out only with one episode")
    for d in dirs:
        out = Path(a.out) if a.out else d / f"{d.name}_cloud{'_scene' if a.scene else ''}.mp4"
        out, frames = render(ED.load_episode(d), out, a.speed, a.depth_max, a.clean, a.accumulate, a.camera, a.scene,
                             a.depth, a.workers, a.dpi)
        print(out, frames, "frames", flush=True)


if __name__ == "__main__":
    main()
