"""Geometry and settings shared by the corridor scenes (port of the reference envs/scenes/_common.py and the start /
goal constants of envs/turtlebot_var_scan_rate.py, commit d58be3fadb): outer arena, S-corridor walls C1-C6, start
poses, goal lists and their samplers. Not a scene itself."""
from __future__ import annotations

import math

import torch

MAX_SAMPLING_ATTEMPTS = 1000       # per rejection loop (reference envs/scenes/_common.py)

# ---------------------------------------------------------------- reference constants (envs/scenes/_common.py) -------
WALL_THICKNESS = 0.15
ARENA_INTERIOR_HALF = 2.910 - WALL_THICKNESS / 2.0
OUTER_WALLS = [(-2.910, 0.000, 6.00, math.pi / 2), (0.000, 2.910, 6.00, 0.0), (2.910, 0.000, 6.00, -math.pi / 2),
               (0.000, -2.910, 6.00, math.pi)]
CORRIDOR_WALLS = [(0.975, 2.080, 1.660, math.pi / 2), (-0.175, 1.325, 2.450, 0.0), (-1.325, 0.225, 2.350, math.pi / 2),
                  (-2.080, -0.875, 1.660, 0.0), (1.580, -0.325, 2.660, 0.0), (0.325, -1.580, 2.660, math.pi / 2)]
C1, C4, C5, C6 = CORRIDOR_WALLS[0], CORRIDOR_WALLS[3], CORRIDOR_WALLS[4], CORRIDOR_WALLS[5]
FIXED_WALLS = OUTER_WALLS + CORRIDOR_WALLS            # never change

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


def _const(cache, name, make):
    """fast_path "reset_constants": a constant tensor built once per scene (cache: dict) instead of at every reset."""
    if cache is None:
        return make()
    if name not in cache:
        cache[name] = make()
    return cache[name]


def sample_start(n: int, start_pose_dr: bool, device, generator=None, cache=None) -> torch.Tensor:
    """(n, 3) start poses x, y, yaw (reference semantics)."""
    if not start_pose_dr:
        return _const(cache, "start_fixed", lambda: torch.tensor([[*STARTING_POS, STARTING_YAW]], device=device)).repeat(n, 1)
    pts = _const(cache, "start_points", lambda: torch.tensor(START_POSE_LIST, device=device))
    idx = torch.randint(len(START_POSE_LIST), (n,), device=device, generator=generator)
    dyaw = (torch.rand(n, device=device, generator=generator) * 2 - 1) * START_YAW_NOISE
    yaw = torch.remainder(START_YAW_CENTER + dyaw + math.pi, 2 * math.pi) - math.pi
    return torch.cat([pts[idx], yaw[:, None]], 1)


def sample_goal(n: int, goals, device, generator=None, cache=None) -> torch.Tensor:
    g = _const(cache, ("goals", tuple(map(tuple, goals))), lambda: torch.tensor(goals, device=device, dtype=torch.float32))
    return g[torch.randint(len(goals), (n,), device=device, generator=generator)]
