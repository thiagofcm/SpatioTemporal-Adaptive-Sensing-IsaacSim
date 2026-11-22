"""Scene `corridor_dynamic_chase_random` (reference ST04 / ST05): S-corridor arena without the Z1-Z4 baffles, with the
moving obstacles OB1 and OB2 as `random_ping_pong` cylinders (random phase, direction, reversal points and turnaround
pauses; see _obstacle_motion.py) and the obstacle randomization of the reference scene.

Port of adaptive_sensor_policy envs/scenes/corridor_dynamic_chase_random.py + _common.py (commit d58be3fadb). The
reference scene also defines OB3 (`triggered_chase`); it is not implemented here (the ST04 / ST05 experiments use
OB1 and OB1 + OB2 only).

Randomization flags (env cfg, same names as the reference):
  * active_obstacles      fixed set present in every episode, e.g. (1,) or (1, 2)
  * obstacle_presence_dr  each obstacle present with OBSTACLE_PRESENCE_PROB, redrawn if none (needs both listed)
  * obstacle_speed_dr     speed ~ U(OBSTACLE_SPEED_RANGES) per obstacle and episode (else the configured 0.06 m/s)
  * obstacle_y_dr         OB1 / OB2 path y ~ sample_obstacle_path_y (else the configured y)
  * obstacle_radius_dr    NOT supported: the ray caster bakes mesh sizes at start-up (radius fixed at 0.17 m)
"""
from __future__ import annotations

import torch

from ._common import CORRIDOR_WALLS, FIXED_WALLS, OUTER_WALLS  # noqa: F401

NAME = "corridor_dynamic_chase_random"
HAS_BAFFLES = False
WALLS = FIXED_WALLS

RANDOM_REVERSAL_FRAC = 0.25              # reversal regions: outer 25 % of the path at each end
# reference _common.CORRIDOR_DYNAMIC_OBSTACLES[:2] as random_ping_pong (same segments and speed)
MOVING_OBSTACLES = (
    dict(type="random_ping_pong", start=(1.4, 1.7), end=(2.4, 1.7), speed=0.06,                    # OB1
         reversal_start=(0.0, RANDOM_REVERSAL_FRAC), reversal_end=(1.0 - RANDOM_REVERSAL_FRAC, 1.0)),
    dict(type="random_ping_pong", start=(2.4, 1.1), end=(1.4, 1.1), speed=0.06,                    # OB2
         reversal_start=(0.0, RANDOM_REVERSAL_FRAC), reversal_end=(1.0 - RANDOM_REVERSAL_FRAC, 1.0)),
)
OBSTACLE_PRESENCE_PROB = (0.5, 0.5)
OBSTACLE_SPEED_RANGES = ((0.04, 0.08), (0.04, 0.08))       # m/s (configured 0.06)
OBSTACLE_Y_RANGES = ((1.20, 1.70), (0.60, 1.10))           # m, OB1 / OB2 path y (configured 1.70 / 1.10 = upper bounds)
OBSTACLE_Y_GAP_RANGE = (0.55, 0.65)                        # m, y1 - y2 (configured 0.60)
OBSTACLE_RADIUS_RANGE = (0.05, 0.20)                       # reference value; radius randomization not supported here
SUPPORTED_DR = frozenset({"obstacle_presence_dr", "obstacle_speed_dr", "obstacle_y_dr"})

# reference ST04 / ST05 goal_pose_list (AdaptiveFPS/config/spatiotemporal/ST04_chase-ob1_dr.yaml)
GOAL_LIST_ST04 = ((-1.80, -1.70), (-2.00, -1.80), (-2.00, -2.00), (-2.20, -1.70), (-1.80, -2.10), (-2.20, -2.10))


def sample_obstacle_path_y(u: torch.Tensor) -> torch.Tensor:
    """(n, 2) path y of OB1, OB2 from candidate uniforms u (n, n_cand, 2) in [0, 1): (y1, y2) ~ U(OBSTACLE_Y_RANGES[0]) x
    U(OBSTACLE_Y_RANGES[1]) conditioned on y1 - y2 in OBSTACLE_Y_GAP_RANGE (uniform over the feasible region, as the
    reference rejection sampler: the first accepted candidate; ~19 % acceptance, so 64 candidates fail with probability
    ~1e-6 and then fall back to the configured (1.70, 1.10))."""
    n, device = u.shape[0], u.device
    lo = torch.tensor([r[0] for r in OBSTACLE_Y_RANGES], device=device)
    hi = torch.tensor([r[1] for r in OBSTACLE_Y_RANGES], device=device)
    cand = lo + u * (hi - lo)
    gap = cand[..., 0] - cand[..., 1]
    ok = (gap >= OBSTACLE_Y_GAP_RANGE[0]) & (gap <= OBSTACLE_Y_GAP_RANGE[1])
    first = torch.argmax(ok.int(), 1)
    pick = cand[torch.arange(n, device=device), first]
    fallback = torch.tensor([MOVING_OBSTACLES[0]["start"][1], MOVING_OBSTACLES[1]["start"][1]], device=device)
    return torch.where(ok.any(1)[:, None], pick, fallback)
