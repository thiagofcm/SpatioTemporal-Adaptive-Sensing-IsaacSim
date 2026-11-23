"""Mount study, visual comparison of two mounts (default R1: z 0.50 m, pitch 7.5 deg; R4: z 0.40 m, pitch 6.75 deg).

Offline rendering (matplotlib + ffmpeg): Isaac Sim's RTX renderer cannot create a graphics device on this server, so
the scene is drawn from the same geometry Isaac Sim uses:
  * robot: the TurtleBot4 Standard meshes exported from the Isaac Sim USD (infrastructure/robot_viz/results/tb4_geometry.npz)
  * scene: walls and moving obstacles of a recorded NAV03 evaluation episode (world_geometry.json, obstacles.npz)
  * LiDAR returns: the analytic 3D ray caster (st_nav.geometry.analytic3d), validated against Isaac Sim's GPU mesh ray
    caster in M7 (walls / floor < 0.01 mm, obstacle cylinders 0.75 mm median)
The Mirrorcle housing dimensions are not available: the sensor is drawn as a labeled 3 cm marker at the optical centre.

    isaacrun -g <gpu> -- python infrastructure/M7_lidar3d/mount_study/scripts/visual_comparison.py [--no-video]
Outputs: results/visual_comparison/{1_mounting,2_scene,3_pov}.png, 4_animation.mp4, selection.json
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import get_context

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from mount_common import (BASE_Z, BODY_PITCH_DEG, EVAL_DYNAMIC, OBST_H, OBST_R, RANGE_MAX, RESULTS, ROOT, WALL_H,  # noqa: E402
                          classify, rays_world, sensor_dirs, wall_boxes)
from st_nav.geometry.analytic3d import raycast_3d  # noqa: E402
from st_nav.sensors.ray_tables import rpy_to_quat_wxyz  # noqa: E402
import episode_data as ED  # noqa: E402

sys.path.insert(0, str(ROOT / "infrastructure" / "robot_viz" / "scripts"))
import render_robot as RR  # noqa: E402  (mesh helpers: decimate, shaded, tf)

OUT = RESULTS / "visual_comparison"
CONFIGS = {"R1": (0.50, 7.5), "R4": (0.40, 6.75)}
CFG_COL = {"R1": "#2ca02c", "R4": "#d6278f"}
CLS_COL = {0: "#6b6b6b", 1: "#1f77b4", 2: "#ff7f0e", 3: "#9467bd", 5: "#d62728"}    # 5 = escape
CLS_NAME = {0: "floor", 1: "wall", 2: "static obstacle", 3: "moving obstacle", 5: "escaping ray (where it passes the wall)"}
TOP_PLATE = 0.3498
EL_HALF, AZ_HALF = 7.0, 14.0
dev = "cuda"
DIRS_S, GROW, GCOL = sensor_dirs(dev)


def Ry(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def Rz(rad):
    return np.array([[math.cos(rad), -math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0], [0, 0, 1]])


def robot_T(x, y, yaw):
    T = np.eye(4)
    T[:3, :3] = Rz(yaw) @ Ry(BODY_PITCH_DEG)
    T[:3, 3] = (x, y, BASE_Z)
    return T


def robot_mesh(cell):
    parts = [(RR.part("base_link", "visual", cell), (0.78, 0.78, 0.80))]
    for b in ("left_wheel", "right_wheel"):
        v, f = RR.part(b, "visual", 0.0)
        parts.append(((RR.tf(RR.G[f"rest|{b}"], v), f), (0.15, 0.15, 0.15)))
    return parts


def draw_robot(ax, parts, T=np.eye(4), alpha=1.0):
    for (v, f), rgb in parts:
        RR.add_shaded(ax, RR.tf(T, v), f, rgb, alpha=alpha)


def sensor_frame_world(T_robot, z, pitch):
    """(origin (3,), R (3, 3)) of the sensor frame in the world."""
    R = T_robot[:3, :3] @ Ry(pitch)
    return T_robot[:3, :3] @ np.array([0, 0, z]) + T_robot[:3, 3], R


def corner_dirs():
    d = []
    for az, el in ((AZ_HALF, EL_HALF), (-AZ_HALF, EL_HALF), (-AZ_HALF, -EL_HALF), (AZ_HALF, -EL_HALF)):
        a, e = math.radians(az), math.radians(el)
        d.append([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return np.array(d)


def draw_sensor(ax, o, R, col, label=None, ray_len=0.0, cube=0.03):
    """3 cm marker cube at the optical centre, boresight arrow, FoV frustum (corner rays to ray_len)."""
    h = cube / 2
    c = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)])
    c = o + c @ R.T
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([c[f] for f in faces], facecolor=col, edgecolor="k", lw=0.4, alpha=0.95))
    b = o + R @ np.array([0.18, 0, 0])
    ax.plot(*zip(o, b), color=col, lw=2.0)
    if ray_len > 0:
        cd = corner_dirs() @ R.T
        ends = o + ray_len * cd
        for e in ends:
            ax.plot(*zip(o, e), color=col, lw=0.8, alpha=0.9)
        ax.add_collection3d(Poly3DCollection([ends], facecolor=col, alpha=0.08, edgecolor=col, lw=0.6))
    if label:
        ax.text(*(o + np.array([0, 0, 0.035])), label, fontsize=7, color="k")


# =================================================================================================== 1. mounting
def figure_mounting():
    parts = robot_mesh(0.006)
    views = [("side (from the robot's right)", 0, -90), ("front", 0, 0), ("top", 90, -90), ("isometric", 22, -55)]
    fig = plt.figure(figsize=(22, 11))
    for i, (name, (z, pitch)) in enumerate(CONFIGS.items()):
        o, R = sensor_frame_world(np.eye(4), z, pitch)
        for j, (vname, elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 4, 4 * i + j + 1, projection="3d")
            ax.set_proj_type("ortho")
            draw_robot(ax, parts)
            draw_sensor(ax, o, R, CFG_COL[name], ray_len=0.55)
            if j in (0, 1):                                              # height references
                L = np.array([-0.2, 0.2])
                for zz, ls, lab in ((TOP_PLATE, "--", f"top platform {TOP_PLATE:.3f} m"), (o[2], ":", f"optical centre {o[2]:.3f} m")):
                    if j == 0:
                        ax.plot(L, [-0.25, -0.25], [zz, zz], color="k", ls=ls, lw=0.8)
                        ax.text(-0.24, -0.25, zz + 0.008, lab, fontsize=7)
                    else:
                        ax.plot([0.3, 0.3], L, [zz, zz], color="k", ls=ls, lw=0.8)
            ax.set_xlim(-0.25, 0.6); ax.set_ylim(-0.30, 0.30); ax.set_zlim(0.0, 0.62)
            ax.set_box_aspect((0.85, 0.60, 0.62))
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("x [m]", fontsize=7); ax.set_ylabel("y [m]", fontsize=7); ax.set_zlabel("z [m]", fontsize=7)
            if j == 0:
                ax.set_title(f"{name}: optical centre (0, 0, {z:.2f}) m in base_link = {z - TOP_PLATE:.3f} m above the "
                             f"top platform,\npitch {pitch:g} deg down (boresight arrow; FoV 28 x 14 deg frustum to 0.55 m)",
                             fontsize=9, loc="left")
            else:
                ax.set_title(vname, fontsize=9)
            if j == 0:
                ax.text2D(0.02, 0.02, vname, transform=ax.transAxes, fontsize=9)
    fig.suptitle("LiDAR mount on the TurtleBot4 Standard (meshes exported from the Isaac Sim USD). Sensor = labeled 3 cm "
                 "marker at the optical centre (Mirrorcle housing dimensions not available)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "1_mounting.png", dpi=110)
    plt.close(fig)


# =================================================================================================== scans
def load_episode_state(ep):
    b, tags = wall_boxes(ep["geometry"])
    O = ep["obstacles"]
    T = ep["traj"]
    n = len(T["x"])
    cyl = np.zeros((n, 2, 4), np.float32)
    for j in range(len(O["ids"])):
        if O["active"][j]:
            cyl[:, j] = np.stack([O["x"][:, j], O["y"][:, j], np.full(n, OBST_R), np.full(n, OBST_H)], -1)
    cyl = np.nan_to_num(cyl)
    return b, tags, cyl, T


def scans(x, y, yaw, b, tags, cyl, z, pitch):
    """Batch of poses -> dict of numpy arrays: origin (P, 3), dirs (P, R, 3), t, cls (5 = escape), cross (P, R),
    per-obstacle hit masks (P, 2, R)."""
    tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)  # noqa: E731
    X, Y, YAW = tt(x), tt(y), tt(yaw)
    P = X.shape[0]
    B = tt(b)[None].expand(P, -1, -1).contiguous()
    C = tt(cyl)
    G = torch.tensor(tags, device=dev)
    org, d_w = rays_world(X, Y, YAW, z, pitch, DIRS_S)
    t, cls, esc, _, _ = classify(org, d_w, B, G, C)
    tall = B.clone(); tall[..., 5] = 1e3
    cross = raycast_3d(org, d_w, tall, None, floor=False, max_range=RANGE_MAX)
    hits = []
    for j in range(C.shape[1]):
        tj = raycast_3d(org, d_w, None, C[:, j:j + 1], floor=False, max_range=RANGE_MAX)
        hits.append((cls == 3) & ((tj - t).abs() < 1e-5))
    cls = torch.where(esc, torch.full_like(cls, 5), cls)
    f = lambda a: a.cpu().numpy()  # noqa: E731
    return {"origin": f(org), "dirs": f(d_w), "t": f(t), "cls": f(cls), "cross": f(cross), "ob_hit": f(torch.stack(hits, 1))}


def select_frame():
    """NAV03 trajectory row where BOTH moving obstacles are seen by both mounts (>= 300 rays each) with floor / walls
    still on >= 50 % of the rays: max of the smaller per-obstacle ray count over both mounts and obstacles. (Both
    obstacles are in the 28 deg FoV together only when the nearer one is 0.5-0.65 m away.)"""
    best = (-2, None, None)
    for d in ED.episode_dirs(EVAL_DYNAMIC):
        ep = ED.load_episode(d)
        if not ep["obstacles"]["active"].all():
            continue
        b, tags, cyl, T = load_episode_state(ep)
        score = None
        for name, (z, pitch) in CONFIGS.items():
            S = scans(T["x"], T["y"], T["yaw"], b, tags, cyl, z, pitch)
            per = S["ob_hit"].sum(-1)                                          # (rows, 2) rays on OB1, OB2
            ctx = ((S["cls"] == 0) | (S["cls"] == 1)).mean(1)                   # floor + wall still in view
            n = np.where((per.min(1) >= 300) & (ctx >= 0.5), per.min(1), -1)
            score = n if score is None else np.minimum(score, n)
        r = int(np.argmax(score))
        if score[r] > best[0]:
            best = (int(score[r]), int(ep["meta"]["episode_index"]), r)
    if best[1] is None or best[0] < 0:
        raise SystemExit("no NAV03 frame with both obstacles in view")
    return best


# =================================================================================================== 2. scene
def box_faces(b):
    cx, cy, hx, hy, yaw, h = b
    c = np.array([[sx * hx, sy * hy] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) @ Rz(yaw)[:2, :2].T + (cx, cy)
    lo, hi = np.c_[c, np.zeros(4)], np.c_[c, np.full(4, h)]
    return [lo, hi] + [np.array([lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4], hi[i]]) for i in range(4)]


def cyl_faces(c, n=32):
    a = np.linspace(0, 2 * math.pi, n + 1)
    ring = np.c_[c[0] + c[2] * np.cos(a), c[1] + c[2] * np.sin(a)]
    side = [np.array([[*ring[i], 0], [*ring[i + 1], 0], [*ring[i + 1], c[3]], [*ring[i], c[3]]]) for i in range(n)]
    return side + [np.c_[ring, np.full(n + 1, c[3])]]


def scatter_classes(ax, S, k, three_d=True, s=1.5, label=False):
    o, d, t, cls, cross = S["origin"][k], S["dirs"][k], S["t"][k], S["cls"][k], S["cross"][k]
    pts = o + t[:, None] * d
    for c in (0, 1, 2, 3, 5):
        m = cls == c
        if not m.any():
            if label:
                ax.scatter([], [], c=CLS_COL[c], s=12, label=CLS_NAME[c])
            continue
        p = (o + np.minimum(cross[m], RANGE_MAX)[:, None] * d[m]) if c == 5 else pts[m]
        kw = dict(c=CLS_COL[c], s=s * (3 if c == 5 else 1), linewidths=0, label=CLS_NAME[c] if label else None)
        if three_d:
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], depthshade=False, marker="x" if c == 5 else "o", **kw)
        else:
            ax.scatter(p[:, 0], p[:, 1], marker="x" if c == 5 else "o", **kw)


def figure_scene(ep, row, S_by):
    b, tags, cyl, T = load_episode_state(ep)
    x, y, yaw = float(T["x"][row]), float(T["y"][row]), float(T["yaw"][row])
    parts = robot_mesh(0.012)
    Tr = robot_T(x, y, yaw)
    fig = plt.figure(figsize=(20, 17))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 0.75])
    for i, name in enumerate(CONFIGS):
        z, pitch = CONFIGS[name]
        S = S_by[name]
        o, R = sensor_frame_world(Tr, z, pitch)
        nob = S["ob_hit"][0].sum(-1)
        fr = {CLS_NAME[c].split(" (")[0]: 100 * float((S["cls"][0] == c).mean()) for c in (0, 1, 3, 5)}
        # 3D
        ax = fig.add_subplot(gs[0, i], projection="3d")
        faces = [f for bb in b for f in box_faces(bb)]
        ax.add_collection3d(Poly3DCollection(faces, facecolor=(0.75, 0.75, 0.75), edgecolor=(0.4, 0.4, 0.4), lw=0.2, alpha=0.18))
        for c in cyl[row]:
            if c[2] > 0:
                ax.add_collection3d(Poly3DCollection(cyl_faces(c), facecolor="#c5b0d5", edgecolor="none", alpha=0.35))
        draw_robot(ax, parts, Tr)
        draw_sensor(ax, o, R, CFG_COL[name], ray_len=3.0, cube=0.06)
        scatter_classes(ax, S, 0, s=1.2, label=(i == 0))
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        cxy = np.array([x, y]) + 1.6 * fwd
        ax.set_xlim(cxy[0] - 2.2, cxy[0] + 2.2); ax.set_ylim(cxy[1] - 2.2, cxy[1] + 2.2); ax.set_zlim(0, 1.0)
        ax.set_box_aspect((4.4, 4.4, 1.0), zoom=1.25)
        ax.view_init(elev=24, azim=math.degrees(yaw) + 180 - 35)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{name}: z {z:.2f} m, pitch {pitch:g} deg  |  rays on OB1 / OB2: {nob[0]} / {nob[1]}\n"
                     + ", ".join(f"{k} {v:.1f} %" for k, v in fr.items()), fontsize=10)
        # top-down
        a2 = fig.add_subplot(gs[1, i])
        for w in ep["geometry"]["walls"]:
            a2.add_patch(Polygon(ED.wall_polygon(w), closed=True, fc="0.6", ec="k", lw=0.3))
        for c in cyl[row]:
            if c[2] > 0:
                a2.add_patch(Circle(c[:2], c[2], fc="#c5b0d5", ec="k", lw=0.5))
        a2.add_patch(Circle((x, y), 0.177, fc="none", ec="k"))
        for s_ in (-1, 1):
            a_ = yaw + s_ * math.radians(AZ_HALF)
            a2.plot([x, x + 6 * math.cos(a_)], [y, y + 6 * math.sin(a_)], color=CFG_COL[name], lw=0.8)
        scatter_classes(a2, S, 0, three_d=False, s=1.5)
        a2.set_xlim(-3, 3); a2.set_ylim(-3, 3); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
        a2.set_title("top-down (horizontal FoV +-14 deg)", fontsize=9)
        # side view
        a3 = fig.add_subplot(gs[2, i])
        pts = S["origin"][0] + S["t"][0][:, None] * S["dirs"][0]
        fw = (pts[:, 0] - o[0]) * math.cos(yaw) + (pts[:, 1] - o[1]) * math.sin(yaw)
        for c in (0, 1, 2, 3):
            m = S["cls"][0] == c
            a3.scatter(fw[m], pts[m, 2], s=1.2, c=CLS_COL[c], linewidths=0)
        el = np.degrees(np.arcsin(S["dirs"][0][:, 2]))
        for e in (el.max(), el.min()):
            a3.plot([0, 8 * math.cos(math.radians(e))], [o[2], o[2] + 8 * math.sin(math.radians(e))], color=CFG_COL[name], lw=0.8)
        a3.axhline(0, color="0.3", lw=0.8); a3.axhline(WALL_H, color="0.5", ls="--", lw=0.7); a3.axhline(OBST_H, color="#9467bd", ls=":", lw=0.7)
        a3.plot([0], [o[2]], "^", color=CFG_COL[name], mec="k")
        a3.set_xlim(-0.2, 7); a3.set_ylim(-0.05, 0.8); a3.grid(lw=0.3); a3.tick_params(labelsize=7)
        a3.set_xlabel("distance along the heading [m]", fontsize=8); a3.set_ylabel("z [m]", fontsize=8)
        a3.set_title("side view (walls 0.6 m dashed, obstacles 0.5 m dotted, fan top / bottom rows)", fontsize=9)
    fig.legend(loc="lower center", ncol=5, fontsize=11, markerscale=5, frameon=False)
    fig.suptitle(f"NAV03 (ref_chase_c10, model_500) {ED.episode_dir_name(ep['meta']['episode_index'])}, step {row} "
                 f"(t = {0.2 * row:.1f} s): same robot pose and obstacle positions for both mounts", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT / "2_scene.png", dpi=100)
    plt.close(fig)


# =================================================================================================== 3. POV
POV_AZ, POV_EL = 20.0, 11.0                                             # virtual camera field (deg), > LiDAR FoV


def to_sensor(p, o, R):
    return (p - o) @ R                                                  # world -> sensor frame (R columns = axes)


def ang(q):
    """sensor-frame points (N, 3) -> (azimuth, elevation) in degrees."""
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
            s = (near - a[0]) / (b[0] - a[0])
            out.append(a + s * (b - a))
    return np.array(out)


def densify(poly, step=0.05):
    pts = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        n = max(1, int(np.linalg.norm(b - a) / step))
        pts += [a + (b - a) * k / n for k in range(n)]
    return np.array(pts)


def draw_pov(ax, b, cyl_row, o, R, S, k, title=None, point_size=6.0):
    polys = []
    for bb in b:
        for f in box_faces(bb):
            polys.append((f, (0.80, 0.80, 0.82), 0.9))
    for c in cyl_row:
        if c[2] > 0:
            for f in cyl_faces(c, 24):
                polys.append((f, (0.80, 0.72, 0.88), 0.95))
    drawn = []
    for f, colr, al in polys:
        q = clip_poly(to_sensor(densify(f), o, R))
        if len(q) < 3:
            continue
        az, el = ang(q)
        if az.min() > POV_AZ + 5 or az.max() < -POV_AZ - 5 or el.min() > POV_EL + 5 or el.max() < -POV_EL - 5:
            continue
        depth = float(np.linalg.norm(q, axis=1).mean())
        drawn.append((depth, np.c_[az, el], colr, al))
    for depth, xy, colr, al in sorted(drawn, key=lambda d: -d[0]):     # painter's algorithm, far to near
        shade = np.clip(np.array(colr) * (1.0 - 0.04 * min(depth, 8)), 0, 1)
        ax.add_patch(Polygon(xy, closed=True, fc=shade, ec=(0.45, 0.45, 0.45), lw=0.3, alpha=al, zorder=1))
    for gx in np.arange(-3, 3.01, 0.5):                                 # floor grid (0.5 m) around the scene
        for line in (np.array([[gx, -3, 0], [gx, 3, 0]]), np.array([[-3, gx, 0], [3, gx, 0]])):
            q = to_sensor(np.linspace(line[0], line[1], 121), o, R)
            q = q[q[:, 0] > 0.05]
            if len(q) > 1:
                az, el = ang(q)
                ax.plot(az, el, color="0.85", lw=0.4, zorder=0.5)
    oo, d, t, cls, cross = S["origin"][k], S["dirs"][k], S["t"][k], S["cls"][k], S["cross"][k]
    for c in (0, 1, 2, 3, 5):
        m = cls == c
        if not m.any():
            continue
        rng = np.minimum(cross[m], RANGE_MAX) if c == 5 else t[m]
        q = to_sensor(oo + rng[:, None] * d[m], o, R)
        az, el = ang(q)
        ax.scatter(az, el, s=point_size / np.maximum(rng, 0.5), c=CLS_COL[c], marker="x" if c == 5 else "o", linewidths=0 if c != 5 else 0.6, zorder=3)
    ax.add_patch(Rectangle((-AZ_HALF, -EL_HALF), 2 * AZ_HALF, 2 * EL_HALF, fc="none", ec="k", lw=0.8, ls="--", zorder=4))
    ax.set_xlim(POV_AZ, -POV_AZ); ax.set_ylim(-POV_EL, POV_EL); ax.set_aspect("equal")
    ax.set_facecolor("#f4f4f4")
    ax.tick_params(labelsize=6)
    if title:
        ax.set_title(title, fontsize=9)


def class_image(S, k):
    img = np.ones((34, 139, 3))
    for c in (0, 1, 2, 3, 5):
        m = S["cls"][k] == c
        img[GROW[m], GCOL[m]] = matplotlib.colors.to_rgb(CLS_COL[c])
    return img


def range_img(S, k):
    img = np.full((34, 139), np.nan)
    hit = S["cls"][k] <= 3
    img[GROW[hit], GCOL[hit]] = S["t"][k][hit]
    return img


def figure_pov(ep, row, S_by):
    b, tags, cyl, T = load_episode_state(ep)
    Tr = robot_T(float(T["x"][row]), float(T["y"][row]), float(T["yaw"][row]))
    fig = plt.figure(figsize=(22, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.25, 1.25])
    for i, name in enumerate(CONFIGS):
        z, pitch = CONFIGS[name]
        S = S_by[name]
        o, R = sensor_frame_world(Tr, z, pitch)
        ax = fig.add_subplot(gs[i, 0])
        draw_pov(ax, b, cyl[row], o, R, S, 0, f"{name} (z {z:.2f} m, pitch {pitch:g} deg): first-person view from the optical "
                 "centre\n(scene geometry + LiDAR returns; dashed = LiDAR FoV 28 x 14 deg; axes = azimuth / elevation [deg])")
        a2 = fig.add_subplot(gs[i, 1])
        im = a2.imshow(range_img(S, 0), cmap="viridis_r", vmin=0, vmax=RANGE_MAX, aspect="auto", interpolation="nearest",
                       extent=(14, -14, -7, 7))
        a2.set_title("range image (white = no return)", fontsize=9); a2.tick_params(labelsize=6)
        fig.colorbar(im, ax=a2, fraction=0.04, pad=0.01).set_label("range [m]", fontsize=8)
        a3 = fig.add_subplot(gs[i, 2])
        a3.imshow(class_image(S, 0), aspect="auto", interpolation="nearest", extent=(14, -14, -7, 7))
        fr = ", ".join(f"{CLS_NAME[c].split(' (')[0]} {100 * float((S['cls'][0] == c).mean()):.1f} %" for c in (0, 1, 3, 5))
        a3.set_title(f"ray classes: {fr}", fontsize=9); a3.tick_params(labelsize=6)
    for c in (0, 1, 2, 3, 5):
        fig.axes[0].scatter([], [], c=CLS_COL[c], s=25, marker="x" if c == 5 else "o", label=CLS_NAME[c])
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.suptitle(f"LiDAR point of view, NAV03 {ED.episode_dir_name(ep['meta']['episode_index'])} step {row}", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT / "3_pov.png", dpi=100)
    plt.close(fig)


# =================================================================================================== 4. animation
_A = {}


def anim_frame(k, path):
    ep, b, cyl, T, S_by, scan_row = _A["ep"], _A["b"], _A["cyl"], _A["T"], _A["S_by"], _A["scan_row"]
    x, y, yaw = float(T["x"][k]), float(T["y"][k]), float(T["yaw"][k])
    s = scan_row[k]                                                     # scan in the observation at this step (held, 4 Hz)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    for i, name in enumerate(CONFIGS):
        z, pitch = CONFIGS[name]
        S = S_by[name]
        Ts = robot_T(float(T["x"][s]), float(T["y"][s]), float(T["yaw"][s]))
        o, R = sensor_frame_world(Ts, z, pitch)
        ax = fig.add_subplot(gs[0, i])
        draw_pov(ax, b, cyl[s], o, R, S, s, f"{name} (z {z:.2f} m, pitch {pitch:g} deg): LiDAR point of view", point_size=5)
        a2 = fig.add_subplot(gs[1, i])
        for w in ep["geometry"]["walls"]:
            a2.add_patch(Polygon(ED.wall_polygon(w), closed=True, fc="0.6", ec="k", lw=0.3))
        for c in cyl[k]:
            if c[2] > 0:
                a2.add_patch(Circle(c[:2], c[2], fc="#c5b0d5", ec="k", lw=0.5))
        a2.plot(T["x"][:k + 1], T["y"][:k + 1], color="tab:blue", lw=1)
        a2.add_patch(Circle((x, y), 0.177, fc="tab:blue", alpha=0.3, ec="navy"))
        a2.plot([x, x + 0.25 * math.cos(yaw)], [y, y + 0.25 * math.sin(yaw)], color="navy", lw=2)
        scatter_classes(a2, S, s, three_d=False, s=1.2)
        a2.set_xlim(-3, 3); a2.set_ylim(-3, 3); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
        nob = S["ob_hit"][s].sum(-1)
        a2.set_title(f"top-down; rays on OB1 / OB2: {nob[0]} / {nob[1]}; floor {100 * float((S['cls'][s] == 0).mean()):.1f} %",
                     fontsize=9)
    fig.suptitle(f"NAV03 {ED.episode_dir_name(ep['meta']['episode_index'])}: t = {0.2 * k:5.1f} s (scan acquired at "
                 f"{0.2 * s:4.1f} s, 4 Hz, held between scans)  |  same recorded trajectory and obstacles for R1 and R4",
                 fontsize=11)
    for c in (0, 1, 3, 5):
        fig.axes[0].scatter([], [], c=CLS_COL[c], s=25, marker="x" if c == 5 else "o", label=CLS_NAME[c])
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def acquisition_rows(n, period=0.25, dt=0.2):
    """Rows at which the env's scheduler acquires a scan (due every period, at the first control instant at or after)."""
    rows, nxt = [], 0.0
    for k in range(n):
        t = k * dt
        if t + 0.005 >= nxt:
            rows.append(k)
            nxt += period * max(1, math.ceil((t + 0.005 - nxt) / period))
    held = np.zeros(n, int)
    j = 0
    for k in range(n):
        while j + 1 < len(rows) and rows[j + 1] <= k:
            j += 1
        held[k] = rows[j]
    return held


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--workers", type=int, default=48)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    figure_mounting()
    print("1_mounting.png", flush=True)
    score, ep_i, row = select_frame()
    ep = ED.load_episode(ED.episode_dirs(EVAL_DYNAMIC)[ep_i])
    b, tags, cyl, T = load_episode_state(ep)
    S_all = {name: scans(T["x"], T["y"], T["yaw"], b, tags, cyl, z, pitch) for name, (z, pitch) in CONFIGS.items()}
    S_row = {name: {k: v[row:row + 1] for k, v in S.items()} for name, S in S_all.items()}
    sel = {"eval": str(EVAL_DYNAMIC), "episode": ep_i, "row": row, "t_s": 0.2 * row,
           "min_rays_on_an_obstacle_over_both_mounts": score,
           "robot": {"x": float(T["x"][row]), "y": float(T["y"][row]), "yaw_deg": math.degrees(float(T["yaw"][row]))},
           "obstacles": cyl[row].tolist(), "configs": {k: {"z_mount_m": v[0], "pitch_deg": v[1]} for k, v in CONFIGS.items()},
           "per_config": {name: {"rays_on_OB1_OB2": S["ob_hit"][0].sum(-1).tolist(),
                                 **{CLS_NAME[c].split(" (")[0] + "_pct": 100 * float((S["cls"][0] == c).mean()) for c in (0, 1, 3, 5)}}
                          for name, S in S_row.items()},
           "episode_means": {name: {CLS_NAME[c].split(" (")[0] + "_pct": 100 * float((S["cls"] == c).mean()) for c in (0, 1, 3, 5)}
                             for name, S in S_all.items()}}
    with open(OUT / "selection.json", "w") as f:
        json.dump(sel, f, indent=1)
    print("frame:", json.dumps(sel["per_config"]), flush=True)
    figure_scene(ep, row, S_row)
    print("2_scene.png", flush=True)
    figure_pov(ep, row, S_row)
    print("3_pov.png", flush=True)
    if args.no_video:
        return
    n = len(T["x"])
    _A.update(ep=ep, b=b, cyl=cyl, T=T, S_by=S_all, scan_row=acquisition_rows(n))
    tmp = tempfile.mkdtemp(prefix="lidar_cmp_")
    with get_context("fork").Pool(args.workers) as pool:
        pool.starmap(anim_frame, [(k, os.path.join(tmp, f"f{k:05d}.png")) for k in range(n)])
    ff = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", os.path.join(tmp, "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(OUT / "4_animation.mp4")], check=True)
    shutil.copy(os.path.join(tmp, f"f{row:05d}.png"), OUT / "4_animation_frame_selected.png")
    shutil.rmtree(tmp)
    print(f"4_animation.mp4 ({n} frames at 10 fps = 2x real time)", flush=True)


if __name__ == "__main__":
    main()
