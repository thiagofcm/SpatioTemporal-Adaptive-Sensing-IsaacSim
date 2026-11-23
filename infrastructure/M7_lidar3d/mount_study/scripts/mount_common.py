"""Mount study: shared geometry (no Isaac Sim; torch on the GPU).

The LiDAR returns of a candidate mount (height z, pitch) are computed with the analytic 3D oracle
(st_nav.geometry.analytic3d, validated against the GPU mesh ray caster in M7: walls / floor < 0.01 mm, obstacle
cylinders p50 0.75 mm) for robot poses and scene states taken from recorded evaluations.

Ray classes (mutually exclusive): floor, wall (fixed arena / corridor walls), static obstacle (Z1-Z4 baffles), moving
obstacle (OB1 / OB2), no return (nothing within range_max).
Escape flag (overlaps the classes): the ray passes ABOVE a wall or baffle within range_max, i.e. an infinitely tall wall
would have stopped it before its actual end point (a hit on the wall's top face is a wall return, not an escape).
`escape_outer`: it passes above an outer arena wall (leaves the environment).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "source" / "st_nav"))
import episode_data as ED  # noqa: E402
from st_nav.geometry.analytic3d import raycast_3d  # noqa: E402
from st_nav.sensors.ray_tables import grid_pattern, rpy_to_quat_wxyz  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RANGE_MAX = 12.0
WALL_H, BAFFLE_H, OBST_H, OBST_R = 0.6, 0.6, 0.5, 0.17
BASE_Z = -0.00445                 # base_link height at rest (measured, results/base_attitude_*.json)
BODY_PITCH_DEG = -0.252           # nominal body pitch while driving (nose up), measured
BODY_PITCH_WORST_DEG = -1.31      # worst observed (static scene, 0.1 % of samples)
CURRENT = (0.3998, 5.0)           # current default mount (z, pitch)
EVAL_STATIC = ROOT / "Navigation/eval/NAV01_static_initial/ref_static_c10/2026-11-22_00-44-09_m5_initial_4096_model_200"
EVAL_DYNAMIC = ROOT / "Navigation/eval/NAV03_chase-ob1ob2_dr/ref_chase_c10/2026-11-22_15-52-39_ppo_n4096_s42_model_500"
CLASSES = ("floor", "wall", "static_obstacle", "moving_obstacle", "no_return")


def sensor_dirs(device):
    az, el, row, col = grid_pattern()
    a, e = torch.deg2rad(torch.tensor(az, device=device)), torch.deg2rad(torch.tensor(el, device=device))
    d = torch.stack([torch.cos(e) * torch.cos(a), torch.cos(e) * torch.sin(a), torch.sin(e)], -1)
    return d, np.array(row), np.array(col)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_apply(q, v):
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def wall_boxes(geometry: dict):
    """(W, 6) [cx, cy, hx, hy, yaw, h] and a per-box tag 0 = outer wall, 1 = corridor wall, 2 = baffle."""
    rows, tags = [], []
    for w in geometry["walls"]:
        baffle = w["name"].startswith("Z")
        rows.append([w["cx"], w["cy"], w["length"] / 2, w["thickness"] / 2, w["yaw"], BAFFLE_H if baffle else WALL_H])
        tags.append(2 if baffle else (0 if w["name"].startswith("outer") else 1))
    return np.array(rows, np.float32), np.array(tags)


def load_poses(eval_dir: Path, every: int, n_random: int, seed: int):
    """Robot poses with their scene state: recorded trajectory rows (every `every`-th) + random free-space poses
    (uniform position with footprint clearance >= 0.227 m, uniform yaw) on random rows of random episodes.
    Returns dict of arrays: x, y, yaw, boxes (P, W, 6), cyl (P, K, 4), source (0 = trajectory, 1 = random), ep, row."""
    rng = np.random.default_rng(seed)
    eps = [ED.load_episode(d) for d in ED.episode_dirs(eval_dir)]
    out = {k: [] for k in ("x", "y", "yaw", "boxes", "cyl", "source", "ep", "row")}

    def cyl_at(ep, r):
        O = ep["obstacles"]
        K = len(O["ids"])
        c = np.zeros((2, 4), np.float32)                         # always 2 slots (r = 0: absent)
        for j in range(K):
            if O["active"][j] and np.isfinite(O["x"][r, j]):
                c[j] = [O["x"][r, j], O["y"][r, j], OBST_R, OBST_H]
        return c

    for i, ep in enumerate(eps):
        b, _ = wall_boxes(ep["geometry"])
        T = ep["traj"]
        for r in range(0, len(T["x"]), every):
            for k, v in (("x", T["x"][r]), ("y", T["y"][r]), ("yaw", T["yaw"][r]), ("boxes", b), ("cyl", cyl_at(ep, r)),
                         ("source", 0), ("ep", i), ("row", r)):
                out[k].append(v)
    half = 2.835
    n = 0
    while n < n_random:
        i = int(rng.integers(len(eps)))
        ep = eps[i]
        r = int(rng.integers(len(ep["traj"]["x"])))
        b, _ = wall_boxes(ep["geometry"])
        c = cyl_at(ep, r)
        x, y = rng.uniform(-half, half, 2)
        if clearance(np.array([x, y]), b, c) < 0.177 + 0.05:
            continue
        for k, v in (("x", x), ("y", y), ("yaw", rng.uniform(-math.pi, math.pi)), ("boxes", b), ("cyl", c),
                     ("source", 1), ("ep", i), ("row", r)):
            out[k].append(v)
        n += 1
    return {k: np.stack(v) if k in ("boxes", "cyl") else np.array(v) for k, v in out.items()}


def clearance(p, boxes, cyl):
    d = np.inf
    for cx, cy, hx, hy, yaw, _ in boxes:
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p[0] - cx, p[1] - cy
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        d = min(d, math.hypot(max(abs(lx) - hx, 0), max(abs(ly) - hy, 0)))
    for cx, cy, r, _ in cyl:
        if r > 0:
            d = min(d, math.hypot(p[0] - cx, p[1] - cy) - r)
    return d


def rays_world(poses_x, poses_y, poses_yaw, z_mount, pitch_deg, dirs_s, body_pitch_deg=BODY_PITCH_DEG):
    """Sensor origins (P, 3) and ray directions (P, R, 3) in the env-local frame for mount (0, 0, z_mount), pitch."""
    dev = dirs_s.device
    P = poses_x.shape[0]
    yaw = poses_yaw
    q_yaw = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1)
    q_body = torch.tensor(rpy_to_quat_wxyz((0.0, body_pitch_deg, 0.0)), device=dev, dtype=torch.float32).expand(P, 4)
    q_base = quat_mul(q_yaw, q_body)
    q_m = torch.tensor(rpy_to_quat_wxyz((0.0, pitch_deg, 0.0)), device=dev, dtype=torch.float32)
    d_b = quat_apply(q_m.expand(dirs_s.shape[0], 4), dirs_s)
    mount = torch.tensor([0.0, 0.0, z_mount], device=dev).expand(P, 3)
    org = torch.stack([poses_x, poses_y, torch.full_like(poses_x, BASE_Z)], -1) + quat_apply(q_base, mount)
    d_w = quat_apply(q_base[:, None].expand(-1, d_b.shape[0], -1), d_b[None].expand(P, -1, -1))
    return org, d_w


def classify(org, d_w, boxes, tags, cyl):
    """Per ray: range (P, R), class index (CLASSES), escape (any wall / baffle), escape_outer, top_face (return from the
    top face of a wall / baffle (z = 0.6 m) or of a moving obstacle (z = 0.5 m): a grazing return, one step from an
    escape / from passing over the obstacle)."""
    t, kind = raycast_3d(org, d_w, boxes, cyl, floor=True, max_range=RANGE_MAX, return_kind=True)
    is_baffle = (tags == 2)
    if is_baffle.any():
        t_wall = raycast_3d(org, d_w, boxes[:, ~is_baffle], None, floor=False, max_range=RANGE_MAX)
        box_cls = torch.where((t_wall - t).abs() < 1e-6, 1, 2)
    else:
        box_cls = torch.ones_like(kind)
    cls = torch.where(kind == 1, 0, torch.where(kind == 2, box_cls, torch.where(kind == 3, 3, 4)))
    tall = boxes.clone(); tall[..., 5] = 1e3
    z_end = org[:, None, 2] + t * d_w[..., 2]
    top_face = (kind == 2) & ((z_end - WALL_H).abs() < 1e-4)          # return from a wall top: not an escape
    top_obst = (kind == 3) & ((z_end - OBST_H).abs() < 1e-4)
    t_tall = raycast_3d(org, d_w, tall, None, floor=False, max_range=RANGE_MAX)
    esc = (t_tall < t - 1e-5) & ~top_face
    outer = tall[:, tags == 0]
    t_out = raycast_3d(org, d_w, outer, None, floor=False, max_range=RANGE_MAX)
    esc_out = (t_out < t - 1e-5) & ~top_face
    return t, cls, esc, esc_out, top_face | top_obst


def tb4_mesh_triangles(device, zmin=0.25):
    g = np.load(ROOT / "infrastructure/robot_viz/results/tb4_geometry.npz")
    tris = []
    for k in ("visual", "collision"):
        tri = g[f"base_link|{k}|v"][g[f"base_link|{k}|f"]]
        tris.append(tri[tri[:, :, 2].max(1) > zmin])
    return torch.tensor(np.concatenate(tris), dtype=torch.float32, device=device)


def self_occluded(z_mount, pitch_deg, dirs_s, tris, t_max=0.4):
    """(R,) bool: ray hits the robot's own mesh (base_link frame, Moller-Trumbore)."""
    dev = dirs_s.device
    q_m = torch.tensor(rpy_to_quat_wxyz((0.0, pitch_deg, 0.0)), device=dev, dtype=torch.float32)
    d = quat_apply(q_m.expand(dirs_s.shape[0], 4), dirs_s)
    o = torch.tensor([0.0, 0.0, z_mount], device=dev)
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    out = torch.zeros(d.shape[0], dtype=torch.bool, device=dev)
    for i in range(0, d.shape[0], 512):
        dd = d[i:i + 512, None]
        p = torch.cross(dd.expand(-1, e2.shape[0], -1), e2[None].expand(dd.shape[0], -1, -1), dim=-1)
        det = (e1[None] * p).sum(-1)
        ok = det.abs() > 1e-12
        inv = torch.where(ok, 1.0 / torch.where(ok, det, torch.ones_like(det)), torch.zeros_like(det))
        s = o - v0
        u = (s[None] * p).sum(-1) * inv
        q = torch.cross(s, e1, dim=-1)
        v = (dd * q[None]).sum(-1) * inv
        t = (e2 * q).sum(-1)[None] * inv
        out[i:i + 512] = (ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < t_max)).any(1)
    return out


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
