#!/usr/bin/env python3
"""LiDAR point-of-view MP4 of recorded 3D-LiDAR evaluation episodes (per-scan layout, e.g. NAV04), reconstructed only
from the episode folder (tools/episode_data.py): no st_nav, Isaac Lab, PyTorch or GPU; numpy + matplotlib + ffmpeg.

Each frame (one per control step, 5 Hz; the LiDAR scan in the observation is held between acquisitions at 4 Hz):
  * first-person view from the LiDAR optical centre (sensor pose at acquisition), ray classes coloured as in the
    mount study (infrastructure/M7_lidar3d/mount_study): floor, wall, static obstacle (baffles Z1-Z4), moving
    obstacle, escaping ray (drawn where it passes over a wall / baffle top); scene geometry drawn as shaded faces
  * the same view with the returns coloured by range (depth)
  * top-down map (walls, goal, trail, robot, the scan's returns by class, horizontal FoV)
  * the range image the policy received (observed, noisy ranges) and the ray-class image

Ray classes are computed from the recorded geometry and the noise-free ranges (ranges_clean): end point below 1 mm =
floor; on a wall / baffle box (names Z* = static obstacle; walls win at junctions) or a moving-obstacle cylinder
(position at the scan's control step); no return = range_max. Escape = the ray crosses a wall / baffle footprint (above
its top) before its end point, whatever it hits afterwards; returns from wall tops count as wall (mount-study
definitions, mount_common.classify; agreement checked against its analytic ray caster).

    # one episode -> <episode_dir>/<episode>_pov.mp4
    python tools/make_pov_video.py <eval_dir>/episode_0005
    # several episodes (one video each)
    python tools/make_pov_video.py --eval <eval_dir> --episodes 5 7 42
    python tools/make_pov_video.py <eval_dir>/episode_0005 --speed 1 --depth-max 6 --out pov_ep5.mp4

Options: --speed (playback speed vs simulated time, default 2), --depth-max (colour scale of the depth view and range
image, default 8 m), --clean (noise-free ranges in the depth view and range image), --workers, --dpi, --out.
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
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

import episode_data as ED  # noqa: E402
import viz_common as V  # noqa: E402
from make_video import ffmpeg_path  # noqa: E402

# ray classes and colours of the mount study (mount_common.classify, visual_comparison.CLS_COL)
FLOOR, WALL, STATIC, MOVING, NONE, ESCAPE = 0, 1, 2, 3, 4, 5
CLS_COL = {FLOOR: "#6b6b6b", WALL: "#1f77b4", STATIC: "#ff7f0e", MOVING: "#9467bd", ESCAPE: "#d62728"}
CLS_NAME = {FLOOR: "floor", WALL: "wall", STATIC: "static obstacle (baffle)", MOVING: "moving obstacle",
            NONE: "no return", ESCAPE: "escaping ray (where it passes the wall)"}
FLOOR_TOL, SURF_TOL = 1e-3, 1e-3          # m: classification tolerances on the noise-free end points (exact to ~1e-5 m)
POV_MARGIN_AZ, POV_MARGIN_EL = 6.0, 4.0   # deg: virtual camera field beyond the LiDAR FoV
DEPTH_CMAP = "turbo_r"


# --------------------------------------------------------------------------------------------- geometry
def rot_rpy(rpy_deg):
    """R = Rz(yaw) Ry(pitch) Rx(roll) (sensor -> base_link, REP-103; pitch > 0 tilts the forward axis down)."""
    r, p, y = np.radians(rpy_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_to_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def scene_boxes(ep):
    """Walls / baffles as (cx, cy, half_length, half_thickness, yaw, height, is_static_obstacle)."""
    H = ep["meta"].get("geometry_heights_m", {})
    out = []
    for w in ep["geometry"]["walls"]:
        z = w.get("name", "").startswith("Z")
        out.append((w["cx"], w["cy"], w["length"] / 2, w["thickness"] / 2, w["yaw"],
                    float(H.get("baffle" if z else "wall", 0.6)), z))
    return out


def obstacles_at(ep, row):
    """Active moving obstacles at control-step row `row`: list of (x, y, radius, height)."""
    O, h = ep["obstacles"], float(ep["meta"].get("geometry_heights_m", {}).get("moving_obstacle", 0.5))
    out = []
    for j in range(len(O["ids"])):
        r = min(row, O["x"].shape[0] - 1)
        while r > 0 and not np.isfinite(O["x"][r, j]):
            r -= 1
        if bool(O["active"][j]) and np.isfinite(O["x"][r, j]):
            out.append((float(O["x"][r, j]), float(O["y"][r, j]), float(O["radius"][j]), h))
    return out


def classify(ep, boxes, scan):
    """Per ray of scan `scan`: class (R,) and the range at which escaping rays cross a wall / baffle footprint."""
    L = ep["lidar"]
    sc = ED.lidar_scan(ep, scan, clean=True)
    o, p, r, hit = sc["origin"], sc["points"], sc["ranges"], sc["hit"]
    d = (p - o) / np.maximum(r, 1e-9)[:, None]
    rmax = float(L["range_max"])
    cls = np.full(len(r), NONE, int)
    cls[hit] = FLOOR                                                 # refined below: box / cylinder surfaces win
    t_cross = np.full(len(r), np.inf)
    t_end = np.where(hit, r, rmax)
    top_face = np.zeros(len(r), bool)
    for cx, cy, hl, ht, yaw, h, z in sorted(boxes, key=lambda b: not b[6]):   # baffles first: walls win at junctions
        c, s = math.cos(yaw), math.sin(yaw)
        # end point on this box -> wall / static obstacle
        u = (p[:, 0] - cx) * c + (p[:, 1] - cy) * s
        v = -(p[:, 0] - cx) * s + (p[:, 1] - cy) * c
        on = hit & (np.abs(u) <= hl + SURF_TOL) & (np.abs(v) <= ht + SURF_TOL) & (p[:, 2] <= h + SURF_TOL)
        cls[on] = STATIC if z else WALL
        top_face |= on & (np.abs(p[:, 2] - h) < SURF_TOL)
        # 2D slab test of the ray against the box footprint (infinitely tall box): entry range
        ou, ov = (o[0] - cx) * c + (o[1] - cy) * s, -(o[0] - cx) * s + (o[1] - cy) * c
        du, dv = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
        with np.errstate(divide="ignore", invalid="ignore"):
            t1u, t2u = (-hl - ou) / du, (hl - ou) / du
            t1v, t2v = (-ht - ov) / dv, (ht - ov) / dv
        tu_lo, tu_hi = np.where(np.abs(du) < 1e-12, np.where(abs(ou) <= hl, -np.inf, np.inf), np.minimum(t1u, t2u)), \
            np.where(np.abs(du) < 1e-12, np.where(abs(ou) <= hl, np.inf, -np.inf), np.maximum(t1u, t2u))
        tv_lo, tv_hi = np.where(np.abs(dv) < 1e-12, np.where(abs(ov) <= ht, -np.inf, np.inf), np.minimum(t1v, t2v)), \
            np.where(np.abs(dv) < 1e-12, np.where(abs(ov) <= ht, np.inf, -np.inf), np.maximum(t1v, t2v))
        t_in, t_out = np.maximum(tu_lo, tv_lo), np.minimum(tu_hi, tv_hi)
        ok = (t_in <= t_out) & (t_in >= 0)
        t_cross = np.where(ok & (t_in < t_cross), t_in, t_cross)
    for x, y, rad, h in obstacles_at(ep, sc["step"]):
        on = hit & (np.hypot(p[:, 0] - x, p[:, 1] - y) <= rad + SURF_TOL) & (p[:, 2] <= h + SURF_TOL) & (p[:, 2] >= FLOOR_TOL)
        cls[on] = MOVING
    # mount study (mount_common.classify): escape = passes over a wall / baffle top before its end point (whatever it
    # hits afterwards), except returns from a wall / baffle top face
    esc = (t_cross < t_end - 1e-3) & ~top_face
    cls[esc] = ESCAPE
    return cls, np.where(esc, t_cross, np.nan)


def check_mount(ep):
    L = ep["lidar"]
    Rm = rot_rpy(L["mount_rpy_deg"])
    err = float(np.abs(L["ray_dirs_s"] @ Rm.T - L["ray_dirs_b"]).max())
    if err > 1e-4:
        raise SystemExit(f"mount rotation does not reproduce ray_dirs_b (max error {err:.2e})")
    return Rm


# --------------------------------------------------------------------------------------------- POV drawing
def box_faces(b):
    cx, cy, hl, ht, yaw, h, _ = b
    c, s = math.cos(yaw), math.sin(yaw)
    xy = np.array([(cx + c * u - s * v, cy + s * u + c * v) for u, v in ((-hl, -ht), (hl, -ht), (hl, ht), (-hl, ht))])
    lo, hi = np.c_[xy, np.zeros(4)], np.c_[xy, np.full(4, h)]
    return [hi] + [np.array([lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4], hi[i]]) for i in range(4)]


def cyl_faces(x, y, r, h, n=24):
    a = np.linspace(0, 2 * math.pi, n + 1)
    ring = np.c_[x + r * np.cos(a), y + r * np.sin(a)]
    side = [np.array([[*ring[i], 0], [*ring[i + 1], 0], [*ring[i + 1], h], [*ring[i], h]]) for i in range(n)]
    return side + [np.c_[ring, np.full(n + 1, h)]]


def ang(q):
    """Sensor-frame points (N, 3) -> (azimuth, elevation) in degrees."""
    return np.degrees(np.arctan2(q[:, 1], q[:, 0])), np.degrees(np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1])))


def clip_poly(q, near=0.03):
    """Clip a sensor-frame polygon against the plane x = near (Sutherland-Hodgman)."""
    out = []
    for i in range(len(q)):
        a, b = q[i], q[(i + 1) % len(q)]
        ia, ib = a[0] >= near, b[0] >= near
        if ia:
            out.append(a)
        if ia != ib:
            out.append(a + (near - a[0]) / (b[0] - a[0]) * (b - a))
    return np.array(out)


def densify(poly, step=0.05):
    pts = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        n = max(1, int(np.linalg.norm(b - a) / step))
        pts += [a + (b - a) * k / n for k in range(n)]
    return np.array(pts)


def draw_scene(ax, polys, o, R, lim, half, faint=False):
    """Scene faces (painter's algorithm, far to near) and the 0.5 m floor grid, projected to (azimuth, elevation)."""
    AZ, EL = lim
    drawn = []
    for f, colr in polys:
        q = clip_poly((densify(f) - o) @ R)
        if len(q) < 3:
            continue
        az, el = ang(q)
        if az.min() > AZ + 5 or az.max() < -AZ - 5 or el.min() > EL + 5 or el.max() < -EL - 5:
            continue
        drawn.append((float(np.linalg.norm(q, axis=1).mean()), np.c_[az, el], colr))
    for depth, xy, colr in sorted(drawn, key=lambda d: -d[0]):
        shade = np.clip(np.array(colr) * (1.0 - 0.04 * min(depth, 8)), 0, 1)
        if faint:
            shade = 0.55 + 0.45 * shade
        ax.add_patch(Polygon(xy, closed=True, fc=shade, ec=(0.45, 0.45, 0.45), lw=0.3, alpha=0.5 if faint else 0.9, zorder=1))
    for g in np.arange(-half, half + 1e-6, 0.5):
        for line in (np.array([[g, -half, 0], [g, half, 0]]), np.array([[-half, g, 0], [half, g, 0]])):
            q = (np.linspace(line[0], line[1], 121) - o) @ R
            q = q[q[:, 0] > 0.05]
            if len(q) > 1:
                az, el = ang(q)
                ax.plot(az, el, color="0.82", lw=0.4, zorder=0.5)


def setup_pov(ax, lim, fov, title):
    (AZ, EL), (az0, az1, el0, el1) = lim, fov
    ax.add_patch(Rectangle((az0, el0), az1 - az0, el1 - el0, fc="none", ec="k", lw=0.8, ls="--", zorder=4))
    ax.set_xlim(AZ, -AZ); ax.set_ylim(-EL, EL); ax.set_aspect("equal")
    ax.set_facecolor("#f4f4f4")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("azimuth [deg] (left = +)", fontsize=7); ax.set_ylabel("elevation [deg]", fontsize=7)
    ax.set_title(title, fontsize=9)


# --------------------------------------------------------------------------------------------- frames
_G = {}


def prepare(ep, depth_max, clean):
    if not ED.is_per_scan(ep):
        raise SystemExit(f"{ep['dir']}: not a 3D-LiDAR (per-scan) episode; use tools/make_video.py for 2D episodes")
    L = ep["lidar"]
    if len(L["grid_shape"]) != 2:
        raise SystemExit("this ray set has no range-image grid")
    Rm = check_mount(ep)
    boxes = scene_boxes(ep)
    S = len(L["ranges"])
    cls, cross = zip(*(classify(ep, boxes, s) for s in range(S)))
    az, el = L["azimuth_deg"], L["elevation_deg"]
    fov = (float(az.min()), float(az.max()), float(el.min()), float(el.max()))
    lim = (max(abs(fov[0]), abs(fov[1])) + POV_MARGIN_AZ, max(abs(fov[2]), abs(fov[3])) + POV_MARGIN_EL)
    _G.update(ep=ep, Rm=Rm, boxes=boxes, cls=np.array(cls), cross=np.array(cross), fov=fov, lim=lim,
              depth_max=depth_max, clean=clean, half=float(ep["geometry"].get("arena_interior_half_m", 2.835)) + 0.15)
    return _G["cls"]


def class_image(ep, cls):
    L = ep["lidar"]
    img = np.ones((*[int(v) for v in L["grid_shape"]], 3))
    for c, col in CLS_COL.items():
        m = cls == c
        img[L["grid_row"][m], L["grid_col"][m]] = to_rgb(col)
    return img


def render_frame(k, path):
    ep, Rm, boxes, CLS, CROSS = _G["ep"], _G["Rm"], _G["boxes"], _G["cls"], _G["cross"]
    lim, fov, dmax = _G["lim"], _G["fov"], _G["depth_max"]
    L, T, meta = ep["lidar"], ep["traj"], ep["meta"]
    n = int(meta["n_steps"])
    row = min(k, n)
    s = max(ED.scan_index(ep, row), 0)
    cls, cross = CLS[s], CROSS[s]
    sc_clean = ED.lidar_scan(ep, s, clean=True)
    sc = ED.lidar_scan(ep, s, clean=_G["clean"])
    o = sc_clean["origin"]
    R = quat_to_R(L["base_quat_wxyz"][s]) @ Rm                      # columns = sensor axes in the env frame
    d = (sc_clean["points"] - o) / np.maximum(sc_clean["ranges"], 1e-9)[:, None]
    scan_row = sc["step"]

    polys = [(f, (0.80, 0.80, 0.82) if not b[6] else (0.95, 0.80, 0.62)) for b in boxes for f in box_faces(b)]
    polys += [(f, (0.80, 0.72, 0.88)) for c in obstacles_at(ep, scan_row) for f in cyl_faces(*c)]

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1])
    # --- POV, ray classes (mount-study colours)
    ax = fig.add_subplot(gs[0, 0:3])
    draw_scene(ax, polys, o, R, lim, _G["half"])
    for c in (FLOOR, WALL, STATIC, MOVING, ESCAPE):
        m = cls == c
        if not m.any():
            continue
        rng = cross[m] if c == ESCAPE else sc_clean["ranges"][m]
        a_, e_ = ang((o + rng[:, None] * d[m] - o) @ R)
        ax.scatter(a_, e_, s=6.0 / np.maximum(rng, 0.5), c=CLS_COL[c], marker="x" if c == ESCAPE else "o",
                   linewidths=0.6 if c == ESCAPE else 0, zorder=3)
    frac = {c: 100 * float((cls == c).mean()) for c in (FLOOR, WALL, STATIC, MOVING, NONE, ESCAPE)}
    setup_pov(ax, lim, fov, "LiDAR point of view: ray classes (dashed = LiDAR FoV)\n" + ", ".join(
        f"{CLS_NAME[c].split(' (')[0]} {v:.1f} %" for c, v in frac.items() if v > 0 or c in (FLOOR, ESCAPE)))
    # --- POV, depth
    ax2 = fig.add_subplot(gs[0, 3:6])
    draw_scene(ax2, polys, o, R, lim, _G["half"], faint=True)
    hit = sc["hit"]
    a_, e_ = ang((sc["points"][hit] - o) @ R)
    order = np.argsort(-sc["ranges"][hit])                           # near points on top
    pc = ax2.scatter(a_[order], e_[order], s=9.0, c=sc["ranges"][hit][order], cmap=DEPTH_CMAP, vmin=0, vmax=dmax,
                     linewidths=0, zorder=3)
    setup_pov(ax2, lim, fov, f"LiDAR point of view: range ({'noise-free' if _G['clean'] else 'observed, noisy'}); "
              f"{int(hit.sum())} / {len(hit)} rays with a return")
    fig.colorbar(pc, ax=ax2, fraction=0.025, pad=0.01).set_label(f"range [m] (scale clipped at {dmax:g} m)", fontsize=7)
    # --- top-down
    a3 = fig.add_subplot(gs[1, 0:2])
    V.draw_walls(a3, ep["geometry"])
    V.draw_goal(a3, meta)
    for x, y, r, _ in obstacles_at(ep, row):
        a3.add_patch(Circle((x, y), r, fc="tab:purple", alpha=0.6, ec="k"))
    a3.plot(T["x"][:row + 1], T["y"][:row + 1], color="tab:blue", lw=1.2)
    for c in (FLOOR, WALL, STATIC, MOVING):
        m = cls == c
        p = sc_clean["points"][m]
        a3.scatter(p[:, 0], p[:, 1], s=1.0, c=CLS_COL[c], linewidths=0, alpha=0.35 if c == FLOOR else 0.9, zorder=3)
    m = cls == ESCAPE
    if m.any():
        p = o + cross[m][:, None] * d[m]
        a3.scatter(p[:, 0], p[:, 1], s=6, c=CLS_COL[ESCAPE], marker="x", linewidths=0.6, zorder=4)
    fwd = math.atan2(R[1, 0], R[0, 0])
    for a in (fov[0], fov[1]):
        aa = fwd + math.radians(a)
        a3.plot([o[0], o[0] + 8 * math.cos(aa)], [o[1], o[1] + 8 * math.sin(aa)], color="k", lw=0.6, ls="--")
    x, y, yaw = float(T["x"][row]), float(T["y"][row]), float(T["yaw"][row])
    done = k >= n
    a3.add_patch(Circle((x, y), meta["robot"]["footprint_radius_m"], ec="navy", alpha=0.4,
                        fc=V.OUTCOME_COLORS.get(meta["outcome"], "tab:blue") if done else "tab:blue"))
    a3.plot([x, x + 0.25 * math.cos(yaw)], [y, y + 0.25 * math.sin(yaw)], color="navy", lw=2)
    V.setup_map(a3, ep["geometry"], f"top-down (scan acquired at t = {sc['t_sim']:.2f} s)", fontsize=9)
    a3.text(0.02, 0.98, f"t {T['t_sim'][row]:5.1f} s  v {T['v_mps'][row]:4.2f} m/s\nclr {T['clearance_m'][row]:5.3f} m"
            + (f"\n{meta['outcome'].upper().replace('_', ' ')}" if done else ""), transform=a3.transAxes, fontsize=8,
            family="monospace", va="top", bbox=dict(fc="white", alpha=0.8, ec="none"))
    # --- range image (policy input) and class image
    ext = (fov[1], fov[0], fov[2], fov[3])
    a4 = fig.add_subplot(gs[1, 2:4])
    im = a4.imshow(ED.range_image(ep, s, clean=_G["clean"]), cmap=DEPTH_CMAP, vmin=0, vmax=dmax, aspect="auto",
                   interpolation="nearest", extent=ext)
    a4.set_title(f"range image = policy input ({'noise-free' if _G['clean'] else 'observed'}; "
                 f"no return = {float(L['range_max']):g} m)", fontsize=8)
    fig.colorbar(im, ax=a4, fraction=0.04, pad=0.01).set_label("range [m]", fontsize=7)
    a5 = fig.add_subplot(gs[1, 4:6])
    a5.imshow(class_image(ep, cls), aspect="auto", interpolation="nearest", extent=ext)
    a5.set_title("ray classes (white = no return)", fontsize=8)
    for a in (a4, a5):
        a.tick_params(labelsize=6)
        a.set_xlabel("azimuth [deg]", fontsize=7); a.set_ylabel("elevation [deg]", fontsize=7)
    for c in (FLOOR, WALL, STATIC, MOVING, ESCAPE):
        ax.scatter([], [], c=CLS_COL[c], s=25, marker="x" if c == ESCAPE else "o", label=CLS_NAME[c])
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=9, frameon=False)
    mnt = L["mount_offset_xyz"], L["mount_rpy_deg"]
    fig.suptitle(f"{meta.get('exp_id', '')}  {meta.get('protocol', '')}  {meta.get('policy', '')}  |  "
                 f"{V.episode_title(meta)}\nt = {row * float(meta['control']['control_dt_s']):5.1f} s, step {row}, "
                 f"scan {s} (4 Hz, held between scans)  |  mount z {float(mnt[0][2]):.2f} m, pitch {float(mnt[1][1]):g} deg",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=_G["dpi"])
    plt.close(fig)


def _work(args):
    render_frame(*args)


def render(ep, out, speed=2.0, depth_max=8.0, clean=False, workers=8, dpi=90):
    cls = prepare(ep, depth_max, clean)
    _G["dpi"] = dpi
    dt = float(ep["meta"]["control"]["control_dt_s"])
    n = int(ep["meta"]["n_steps"])
    frames = n + 1 + int(1.5 / dt)
    tmp = Path(tempfile.mkdtemp(prefix="pov_"))
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
    tot = {CLS_NAME[c].split(" (")[0]: round(100 * float((cls == c).mean()), 1) for c in (FLOOR, WALL, STATIC, MOVING, NONE, ESCAPE)}
    return out, frames, tot


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("episode_dirs", nargs="*", help="episode folder(s) of a 3D-LiDAR evaluation")
    p.add_argument("--eval", help="evaluation folder (with --episodes)")
    p.add_argument("--episodes", type=int, nargs="+", help="episode indices in --eval")
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--depth-max", dest="depth_max", type=float, default=8.0)
    p.add_argument("--clean", action="store_true", help="noise-free ranges in the depth view and range image")
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
        out = Path(a.out) if a.out else d / f"{d.name}_pov.mp4"
        out, frames, tot = render(ED.load_episode(d), out, a.speed, a.depth_max, a.clean, a.workers, a.dpi)
        print(out, frames, "frames; ray classes over the episode's scans [%]:", tot, flush=True)


if __name__ == "__main__":
    main()
