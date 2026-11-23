"""Analytic 3D ray casting on the GPU (batched torch): the oracle that validates the 3D LiDAR (mesh ray caster).

Scene primitives (env-local frame, metres), as built by the scene runtime (st_nav/envs/corridor_scene.py):
  boxes    : (N, W, 6) [cx, cy, half_length, half_thickness, yaw, height]   walls / baffles, z in [0, height]
  cylinders: (N, C, 4) [cx, cy, r, height]  vertical cylinders, z in [0, height]; r <= 0 = absent (inactive obstacle)
  floor    : the plane z = 0 (optional)
Rays: origins (N, 3), unit directions (N, R, 3). Returns the distance to the first hit (N, R); max_range where nothing
is hit within max_range.
"""
from __future__ import annotations

import torch

_EPS = 1e-6


def _boxes_t(o, d, boxes):
    """(N, R) first-hit distance against oriented boxes (slab test in each box frame), inf if none."""
    if boxes is None or boxes.shape[1] == 0:
        return torch.full(d.shape[:2], float("inf"), device=d.device)
    c, s = torch.cos(boxes[..., 4]), torch.sin(boxes[..., 4])                  # (N, W)
    px, py = o[:, None, 0] - boxes[..., 0], o[:, None, 1] - boxes[..., 1]      # (N, W)
    lo_ = torch.stack([c * px + s * py, -s * px + c * py, o[:, None, 2].expand_as(px)], -1)   # origin in box frame
    dx, dy, dz = d[..., 0:1], d[..., 1:2], d[..., 2:3]                         # (N, R, 1)
    ld = torch.stack([c[:, None] * dx + s[:, None] * dy, -s[:, None] * dx + c[:, None] * dy, dz.expand(-1, -1, c.shape[1])], -1)
    lo_b = torch.stack([-boxes[..., 2], -boxes[..., 3], torch.zeros_like(boxes[..., 2])], -1)[:, None]   # (N, 1, W, 3)
    hi_b = torch.stack([boxes[..., 2], boxes[..., 3], boxes[..., 5]], -1)[:, None]
    p = lo_[:, None]                                                            # (N, 1, W, 3)
    inv = 1.0 / torch.where(ld.abs() < 1e-12, torch.full_like(ld, 1e-12), ld)
    t1, t2 = (lo_b - p) * inv, (hi_b - p) * inv
    tn = torch.minimum(t1, t2).amax(-1)
    tf = torch.maximum(t1, t2).amin(-1)
    hit = (tf >= tn) & (tn > _EPS)
    return torch.where(hit, tn, torch.full_like(tn, float("inf"))).amin(-1)


def _cylinders_t(o, d, cyl):
    """(N, R) first-hit distance against vertical finite cylinders (side + top cap), inf if none."""
    if cyl is None or cyl.shape[1] == 0:
        return torch.full(d.shape[:2], float("inf"), device=d.device)
    inf = float("inf")
    ox, oy, oz = (o[:, None, None, i] for i in range(3))                       # (N, 1, 1)
    dx, dy, dz = (d[..., None, i] for i in range(3))                           # (N, R, 1)
    cx, cy, r, h = (cyl[:, None, :, i] for i in range(4))                      # (N, 1, C)
    fx, fy = ox - cx, oy - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    cc = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * cc
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    a_safe = torch.where(a < 1e-12, torch.full_like(a, 1e-12), a)
    t_side = (-b - sq) / (2 * a_safe)                                          # entering root (origin outside)
    z_side = oz + t_side * dz
    side_ok = (disc >= 0) & (a > 1e-12) & (t_side > _EPS) & (z_side >= 0) & (z_side <= h) & (r > 0)
    t_cap = (h - oz) / torch.where(dz.abs() < 1e-12, torch.full_like(dz, -1e-12), dz)   # top cap (from above)
    xc, yc = fx + t_cap * dx, fy + t_cap * dy
    cap_ok = (oz > h) & (dz < 0) & (t_cap > _EPS) & (xc * xc + yc * yc <= r * r) & (r > 0)
    t = torch.minimum(torch.where(side_ok, t_side, torch.full_like(t_side, inf)),
                      torch.where(cap_ok, t_cap, torch.full_like(t_cap, inf)))
    return t.amin(-1)


def raycast_3d(origins: torch.Tensor, dirs: torch.Tensor, boxes: torch.Tensor | None = None,
               cylinders: torch.Tensor | None = None, floor: bool = True, max_range: float = float("inf"),
               return_kind: bool = False):
    """Distances (N, R); with return_kind also the hit class (N, R) int: 0 none, 1 floor, 2 box, 3 cylinder."""
    inf = float("inf")
    tb = _boxes_t(origins, dirs, boxes)
    tc = _cylinders_t(origins, dirs, cylinders)
    if floor:
        dz = dirs[..., 2]
        tfl = -origins[:, None, 2] / torch.where(dz < -1e-12, dz, torch.full_like(dz, -1e-12))
        tfl = torch.where((dz < -1e-12) & (tfl > _EPS), tfl, torch.full_like(tfl, inf))
    else:
        tfl = torch.full_like(tb, inf)
    t = torch.minimum(torch.minimum(tb, tc), tfl)
    kind = torch.zeros_like(t, dtype=torch.long)
    kind = torch.where(t == tfl, torch.ones_like(kind), kind)
    kind = torch.where(t == tb, torch.full_like(kind, 2), kind)
    kind = torch.where(t == tc, torch.full_like(kind, 3), kind)
    miss = ~(t <= max_range)
    kind = torch.where(miss, torch.zeros_like(kind), kind)
    t = torch.where(miss, torch.full_like(t, max_range), t)
    return (t, kind) if return_kind else t
