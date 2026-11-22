"""Scenario `corridor_static_object_random` (reference ST01-ST03): S-corridor arena with four zig-zag baffles Z1-Z4.

Geometry, randomization ranges, constraints and the Z1-Z4 sampler are ported from the reference simplified environment
(adaptive_sensor_policy/envs/scenes/corridor_static_object_random.py and _common.py, commit d58be3fadb). The reference
is not used at run time. tests/unit/test_static_obstacle_sampler.py checks the sampler against golden reference output;
tests/unit/test_corridor_static_random.py cross-checks the constants against the reference when it is available.

Randomization (approved design, see M3 report):
  * Z1-Z4 lengths and thicknesses: drawn once per environment at start-up from a bank of K configurations sampled with
    the ported reference sampler (exact joint distribution incl. the hard/normal mixture and constraints A, B, C).
    (The mesh ray caster bakes mesh sizes at initialization, so sizes cannot change per reset.)
  * Z offsets (Z1_x, Z2_x, Z3_y, Z4_y): resampled at every reset on the GPU, uniformly over their reference ranges,
    rejected until constraint C (d(Z1, Z3) >= 0.70 m) holds for the environment's sizes. This is exactly the reference
    conditional distribution of the offsets given the sizes. Constraints A and B depend on lengths only (always hold).
  * Start pose: fixed (reference default) or start-pose DR (5 points, yaw = pi + U(-10, 10) deg).
  * Goal: uniform from a goal list (reference ST01-ST03 9-goal list by default).
"""
from __future__ import annotations

import math

import numpy as np
import torch

# ---------------------------------------------------------------- reference constants (envs/scenes/_common.py) -------
WALL_THICKNESS = 0.15
ARENA_INTERIOR_HALF = 2.910 - WALL_THICKNESS / 2.0
OUTER_WALLS = [(-2.910, 0.000, 6.00, math.pi / 2), (0.000, 2.910, 6.00, 0.0), (2.910, 0.000, 6.00, -math.pi / 2),
               (0.000, -2.910, 6.00, math.pi)]
CORRIDOR_WALLS = [(0.975, 2.080, 1.660, math.pi / 2), (-0.175, 1.325, 2.450, 0.0), (-1.325, 0.225, 2.350, math.pi / 2),
                  (-2.080, -0.875, 1.660, 0.0), (1.580, -0.325, 2.660, 0.0), (0.325, -1.580, 2.660, math.pi / 2)]
C1, C4, C5, C6 = CORRIDOR_WALLS[0], CORRIDOR_WALLS[3], CORRIDOR_WALLS[4], CORRIDOR_WALLS[5]
FIXED_WALLS = OUTER_WALLS + CORRIDOR_WALLS            # never change

# ------------------------------------------------- reference constants (corridor_static_object_random.py) ------------
NOMINAL = ((0.55, 0.15, 0.1, 0.6), (0.60, 0.15, 0.60, 0.15), (-0.5, -0.2, -0.0, -0.3))   # lengths, thicknesses, offsets
LENGTH_RANGE = (0.10, 0.60)
THICKNESS_RANGE = (0.15, 0.60)
OFFSET_RANGES = ((-0.50, 0.00), (-0.50, 0.00), (-0.55, 0.00), (-0.55, 0.00))   # Z1_x, Z2_x, Z3_y, Z4_y
CORRIDOR_WIDTH = 1.50
MIN_NOMINAL_CLEARANCE = 0.75
MIN_Z1_Z3_DISTANCE = 0.70
GEOMETRY_TOL = 1e-9

# ------------------------------------------------- reference constants (envs/turtlebot_var_scan_rate.py) -------------
STARTING_POS = (2.5, 2.5)
STARTING_YAW = -2.9441970937399127
START_POSE_LIST = ((2.50, 2.50), (2.30, 2.50), (2.50, 2.30), (2.30, 2.30), (2.40, 2.40))
START_YAW_CENTER = math.pi
START_YAW_NOISE = math.radians(10)
GOAL_LIST_DEFAULT = ((-1.8, -2.5),)
# AdaptiveFPS/config/spatiotemporal/ST01-ST03 goal_pose_list (static randomized scene)
GOAL_LIST_ST01 = ((-1.80, -2.50), (-1.50, -2.50), (-2.10, -2.50), (-1.80, -2.20), (-1.60, -2.30), (-2.00, -2.30),
                  (-1.50, -2.20), (-2.10, -2.20), (-1.80, -2.00))


def z_walls(lengths: torch.Tensor, thick: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Batched port of `static_obstacle_walls`: (N, 4) x3 -> (N, 4, 5) [cx, cy, length, yaw, thickness]."""
    L1, L2, L3, L4 = lengths.unbind(-1)
    T1, T2, T3, T4 = thick.unbind(-1)
    o1, o2, o3, o4 = offsets.unbind(-1)
    half_pi = torch.full_like(L1, math.pi / 2)
    zero = torch.zeros_like(L1)
    z1 = torch.stack([C1[0] + o1, C1[1] - (C1[2] + L1) / 2, L1, half_pi, T1], -1)
    z2 = torch.stack([C1[0] + o2, C5[1] + WALL_THICKNESS / 2 + L2 / 2, L2, half_pi, T2], -1)
    z3 = torch.stack([C6[0] - WALL_THICKNESS / 2 - L3 / 2, C5[1] + o3, L3, zero, T3], -1)
    z4 = torch.stack([C4[0] + C4[2] / 2 + L4 / 2, C5[1] + o4, L4, zero, T4], -1)
    return torch.stack([z1, z2, z3, z4], 1)


def _bounds(w: torch.Tensor):
    """Axis-aligned bounds of wall specs (..., 5); yaw is a multiple of pi/2 for these walls (reference _rect_distance)."""
    vertical = torch.sin(w[..., 3]).abs() >= 0.5
    hx = torch.where(vertical, w[..., 4] / 2, w[..., 2] / 2)
    hy = torch.where(vertical, w[..., 2] / 2, w[..., 4] / 2)
    return w[..., 0] - hx, w[..., 0] + hx, w[..., 1] - hy, w[..., 1] + hy


def rect_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched `_rect_distance` between axis-aligned wall rectangles."""
    ax0, ax1, ay0, ay1 = _bounds(a)
    bx0, bx1, by0, by1 = _bounds(b)
    dx = torch.clamp(torch.maximum(ax0 - bx1, bx0 - ax1), min=0.0)
    dy = torch.clamp(torch.maximum(ay0 - by1, by0 - ay1), min=0.0)
    return torch.hypot(dx, dy)


def sample_offsets(lengths: torch.Tensor, thick: torch.Tensor, fallback: torch.Tensor, n_cand: int = 32,
                   generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env offsets ~ U(OFFSET_RANGES) conditioned on d(Z1, Z3) >= MIN_Z1_Z3_DISTANCE (vectorized rejection).

    Returns (offsets (N, 4), used_fallback (N,) bool). Envs without a valid candidate keep `fallback` (their bank
    offsets, valid by construction); with n_cand = 32 this is rare and is counted in validation."""
    n, dev = lengths.shape[0], lengths.device
    lo = torch.tensor([r[0] for r in OFFSET_RANGES], device=dev)
    hi = torch.tensor([r[1] for r in OFFSET_RANGES], device=dev)
    u = torch.rand(n, n_cand, 4, device=dev, generator=generator)
    cand = lo + u * (hi - lo)
    w = z_walls(lengths[:, None].expand(-1, n_cand, -1).reshape(-1, 4), thick[:, None].expand(-1, n_cand, -1).reshape(-1, 4),
                cand.reshape(-1, 4)).reshape(n, n_cand, 4, 5)
    ok = rect_distance(w[:, :, 0], w[:, :, 2]) >= MIN_Z1_Z3_DISTANCE - GEOMETRY_TOL
    first = torch.argmax(ok.int(), 1)
    has = ok.any(1)
    out = torch.where(has[:, None], cand[torch.arange(n, device=dev), first], fallback)
    return out, ~has


# ------------------------------------------- Z1-Z4 configuration sampler (port of the reference, CPU, numpy) -----------
# Port of adaptive_sensor_policy envs/scenes/corridor_static_object_random.py (commit d58be3fadb):
# sample_static_obstacles_params and its helpers, with identical random streams and floating-point operations, so a
# seed gives bit-identical configurations (tests/unit/test_static_obstacle_sampler.py against tests/golden/reference_bank.npz).
HARD_PAIR_PROB = 0.5               # probability that a Z pair uses the "hard" mode (nominal clearance in [0.75, 0.80])
HARD_BAND = 0.05                   # m, width of the hard nominal-clearance band above MIN_NOMINAL_CLEARANCE
MAX_SAMPLING_ATTEMPTS = 1000       # per rejection loop (reference envs/scenes/_common.py)


def static_obstacle_walls(lengths, thicknesses, offsets):
    """Z1-Z4 wall specs (cx, cy, length, yaw, thickness) of one configuration (scalar version of z_walls)."""
    (L1, L2, L3, L4), (T1, T2, T3, T4), (o1, o2, o3, o4) = lengths, thicknesses, offsets
    return [
        (C1[0] + o1, C1[1] - (C1[2] + L1) / 2, L1, math.pi / 2, T1),                  # Z1
        (C1[0] + o2, C5[1] + WALL_THICKNESS / 2 + L2 / 2, L2, math.pi / 2, T2),       # Z2
        (C6[0] - WALL_THICKNESS / 2 - L3 / 2, C5[1] + o3, L3, 0.0, T3),               # Z3
        (C4[0] + C4[2] / 2 + L4 / 2, C5[1] + o4, L4, 0.0, T4),                        # Z4
    ]


def _rect_distance_scalar(a, b):
    """Shortest distance between two axis-aligned wall rectangles (cx, cy, length, yaw, thickness)."""
    def bounds(w):
        hx, hy = (w[2] / 2, w[4] / 2) if abs(math.sin(w[3])) < 0.5 else (w[4] / 2, w[2] / 2)
        return w[0] - hx, w[0] + hx, w[1] - hy, w[1] + hy
    ax0, ax1, ay0, ay1 = bounds(a)
    bx0, bx1, by0, by1 = bounds(b)
    return math.hypot(max(ax0 - bx1, bx0 - ax1, 0.0), max(ay0 - by1, by0 - ay1, 0.0))


def static_obstacles_params(seed, lengths, thicknesses, offsets, pair_modes=None) -> dict:
    """Parameter record of one Z1-Z4 configuration."""
    z1, _, z3, _ = static_obstacle_walls(lengths, thicknesses, offsets)
    return {"static_obstacles_seed": seed,
            "pair_modes": None if pair_modes is None else list(pair_modes),
            "lengths": [float(v) for v in lengths],
            "thicknesses": [float(v) for v in thicknesses],
            "offsets": [float(v) for v in offsets],
            "c12": CORRIDOR_WIDTH - (lengths[0] + lengths[1]),
            "c34": CORRIDOR_WIDTH - (lengths[2] + lengths[3]),
            "d_z1_z3": _rect_distance_scalar(z1, z3)}


def _sample_length_pair(rng):
    """(La, Lb, mode). "hard" (prob. HARD_PAIR_PROB): La + Lb ~ U over the band giving a nominal clearance in
    [MIN_NOMINAL_CLEARANCE, MIN_NOMINAL_CLEARANCE + HARD_BAND], split at random. "normal": (La, Lb) ~ U(LENGTH_RANGE)^2,
    rejected until CORRIDOR_WIDTH - (La + Lb) >= MIN_NOMINAL_CLEARANCE."""
    if rng.random() < HARD_PAIR_PROB:
        max_total = CORRIDOR_WIDTH - MIN_NOMINAL_CLEARANCE
        total = rng.uniform(max_total - HARD_BAND, max_total)
        la = rng.uniform(max(LENGTH_RANGE[0], total - LENGTH_RANGE[1]), min(LENGTH_RANGE[1], total - LENGTH_RANGE[0]))
        return float(la), float(total - la), "hard"
    for _ in range(MAX_SAMPLING_ATTEMPTS):
        la, lb = rng.uniform(*LENGTH_RANGE, size=2)
        if CORRIDOR_WIDTH - (la + lb) >= MIN_NOMINAL_CLEARANCE - GEOMETRY_TOL:
            return float(la), float(lb), "normal"
    raise RuntimeError(f"could not sample a Z length pair with nominal clearance >= {MIN_NOMINAL_CLEARANCE} m")


def sample_static_obstacles_params(seed: int) -> dict:
    """One Z1-Z4 configuration from a seed: three independent streams (lengths, thicknesses, offsets); each length
    pair picks the hard or normal mode; constraint C (d(Z1, Z3) >= MIN_Z1_Z3_DISTANCE) rejects the configuration."""
    rng_length = np.random.default_rng([seed, 0])
    rng_thickness = np.random.default_rng([seed, 1])
    rng_offset = np.random.default_rng([seed, 2])
    for _ in range(MAX_SAMPLING_ATTEMPTS):
        L1, L2, mode12 = _sample_length_pair(rng_length)
        L3, L4, mode34 = _sample_length_pair(rng_length)
        lengths = (L1, L2, L3, L4)
        thicknesses = tuple(float(v) for v in rng_thickness.uniform(*THICKNESS_RANGE, size=4))
        offsets = tuple(float(rng_offset.uniform(lo, hi)) for lo, hi in OFFSET_RANGES)
        params = static_obstacles_params(seed, lengths, thicknesses, offsets, (mode12, mode34))
        if params["d_z1_z3"] >= MIN_Z1_Z3_DISTANCE - GEOMETRY_TOL:
            return params
    raise RuntimeError(f"could not sample a Z1-Z4 configuration with d(Z1, Z3) >= {MIN_Z1_Z3_DISTANCE} m")


def sample_bank(k: int, seed: int = 0) -> dict:
    """K Z1-Z4 configurations (exact reference distribution), seeds seed..seed+k-1."""
    rows = [sample_static_obstacles_params(seed + i) for i in range(k)]
    return {"lengths": np.array([r["lengths"] for r in rows]), "thicknesses": np.array([r["thicknesses"] for r in rows]),
            "offsets": np.array([r["offsets"] for r in rows]), "pair_modes": [r["pair_modes"] for r in rows],
            "seeds": np.arange(seed, seed + k)}


def sample_start(n: int, start_pose_dr: bool, device, generator=None) -> torch.Tensor:
    """(n, 3) start poses x, y, yaw (reference semantics)."""
    if not start_pose_dr:
        return torch.tensor([[*STARTING_POS, STARTING_YAW]], device=device).repeat(n, 1)
    pts = torch.tensor(START_POSE_LIST, device=device)
    idx = torch.randint(len(START_POSE_LIST), (n,), device=device, generator=generator)
    dyaw = (torch.rand(n, device=device, generator=generator) * 2 - 1) * START_YAW_NOISE
    yaw = torch.remainder(START_YAW_CENTER + dyaw + math.pi, 2 * math.pi) - math.pi
    return torch.cat([pts[idx], yaw[:, None]], 1)


def sample_goal(n: int, goals, device, generator=None) -> torch.Tensor:
    g = torch.tensor(goals, device=device, dtype=torch.float32)
    return g[torch.randint(len(goals), (n,), device=device, generator=generator)]
