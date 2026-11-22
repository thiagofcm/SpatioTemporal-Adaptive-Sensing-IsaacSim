"""Analytic 2D geometry on the GPU (batched torch): ray casting and ground-truth clearance against oriented boxes
(walls) and circles (cylindrical obstacles).

Port of the reference simplified environment (adaptive_sensor_policy/envs/turtlebot_var_scan_rate.py: `_raycast` slab
test + circle intersection, `_wall_clearance` / `_ground_truth_clearance`). Used as
  * the oracle that validates the GPU mesh ray caster (M2),
  * the ground-truth clearance source for rewards and terminations (M3/M4; independent of any sensor).

Conventions: env-local frame, x/y in metres, angles in radians CCW from +x.
  walls   : (N, W, 5) or (W, 5)  [cx, cy, half_length, half_thickness, yaw]   (reference spec: length, thickness are full)
  circles : (N, C, 3)            [cx, cy, r]; inactive circles may use r <= 0 (ignored)
"""
from __future__ import annotations

import math

import torch


def walls_from_specs(specs, default_thickness: float = 0.15, device="cpu") -> torch.Tensor:
    """Reference wall specs (cx, cy, length, yaw[, thickness]) -> (W, 5) [cx, cy, hx, hy, yaw]."""
    rows = []
    for s in specs:
        th = s[4] if len(s) > 4 else default_thickness
        rows.append([s[0], s[1], s[2] / 2.0, th / 2.0, s[3]])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _bcast_walls(walls: torch.Tensor, n: int) -> torch.Tensor:
    return walls.unsqueeze(0).expand(n, -1, -1) if walls.dim() == 2 else walls


def raycast_2d(origin: torch.Tensor, angles: torch.Tensor, walls: torch.Tensor | None, circles: torch.Tensor | None,
               max_range: float = float("inf")) -> torch.Tensor:
    """Distances along rays (N, R) from origin (N, 2) in directions angles (N, R) [world/env yaw already added].

    Returns max_range where nothing is hit (inf by default). Boxes: slab test in each box's local frame (hit if
    t_far >= t_near > 1e-6, i.e. the ray origin outside the box). Circles: first intersection with t > 1e-6.
    """
    n, r = angles.shape
    dx, dy = torch.cos(angles), torch.sin(angles)                         # (N, R)
    t = torch.full((n, r), float("inf"), device=angles.device, dtype=angles.dtype)
    if walls is not None and walls.numel():
        w = _bcast_walls(walls, n)                                        # (N, W, 5)
        c, s = torch.cos(w[..., 4]), torch.sin(w[..., 4])                 # (N, W)
        ox = origin[:, 0:1] - w[..., 0]
        oy = origin[:, 1:2] - w[..., 1]
        lx = (c * ox + s * oy).unsqueeze(-1)                              # (N, W, 1)
        ly = (-s * ox + c * oy).unsqueeze(-1)
        ldx = c.unsqueeze(-1) * dx.unsqueeze(1) + s.unsqueeze(-1) * dy.unsqueeze(1)      # (N, W, R)
        ldy = -s.unsqueeze(-1) * dx.unsqueeze(1) + c.unsqueeze(-1) * dy.unsqueeze(1)
        eps = 1e-12
        ldx = torch.where(ldx.abs() < eps, torch.full_like(ldx, eps), ldx)
        ldy = torch.where(ldy.abs() < eps, torch.full_like(ldy, eps), ldy)
        hx, hy = w[..., 2].unsqueeze(-1), w[..., 3].unsqueeze(-1)
        tx1, tx2 = (-hx - lx) / ldx, (hx - lx) / ldx
        ty1, ty2 = (-hy - ly) / ldy, (hy - ly) / ldy
        t_near = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
        t_far = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
        hit = (t_far >= t_near) & (t_near > 1e-6)
        t = torch.minimum(t, torch.where(hit, t_near, torch.full_like(t_near, float("inf"))).min(dim=1).values)
    if circles is not None and circles.numel():
        cx, cy, cr = circles[..., 0:1], circles[..., 1:2], circles[..., 2:3]          # (N, C, 1)
        fx = origin[:, 0:1, None] - cx                                                  # (N, C, 1)
        fy = origin[:, 1:2, None] - cy
        b = 2.0 * (fx * dx.unsqueeze(1) + fy * dy.unsqueeze(1))                        # (N, C, R)
        cc = fx * fx + fy * fy - cr * cr
        disc = b * b - 4.0 * cc
        ok = (disc >= 0.0) & (cr > 0)
        th = (-b - torch.sqrt(torch.clamp(disc, min=0.0))) / 2.0
        th = torch.where(ok & (th > 1e-6), th, torch.full_like(th, float("inf")))
        t = torch.minimum(t, th.min(dim=1).values)
    return torch.clamp(t, max=max_range)


def wall_clearance(points: torch.Tensor, walls: torch.Tensor) -> torch.Tensor:
    """Distance (N,) from points (N, 2) to the nearest box surface (0 inside a box)."""
    n = points.shape[0]
    w = _bcast_walls(walls, n)
    c, s = torch.cos(w[..., 4]), torch.sin(w[..., 4])
    ox = points[:, 0:1] - w[..., 0]
    oy = points[:, 1:2] - w[..., 1]
    lx = c * ox + s * oy
    ly = -s * ox + c * oy
    ex = torch.clamp(lx.abs() - w[..., 2], min=0.0)
    ey = torch.clamp(ly.abs() - w[..., 3], min=0.0)
    return torch.sqrt(ex * ex + ey * ey).min(dim=1).values


def circle_clearance(points: torch.Tensor, circles: torch.Tensor) -> torch.Tensor:
    """Distance (N,) from points to the nearest active circle surface (inf if none; negative inside)."""
    d = torch.linalg.norm(points[:, None, :] - circles[..., :2], dim=-1) - circles[..., 2]
    d = torch.where(circles[..., 2] > 0, d, torch.full_like(d, float("inf")))
    return d.min(dim=1).values


def deg_beams(n: int = 40, start_deg: float = 0.0, step_deg: float = 9.0) -> torch.Tensor:
    return torch.deg2rad(torch.tensor([start_deg + i * step_deg for i in range(n)]))


def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi
