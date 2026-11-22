"""Parity of the corridor_static_object_random scenario with the reference (read-only import) and constraint checks."""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.dont_write_bytecode = True
from st_nav.scenes import corridor_static_random as S  # noqa: E402

REF = os.path.expanduser("~/adaptive_sensor_policy")


@pytest.fixture(scope="module")
def ref():
    if not os.path.isdir(REF):
        pytest.skip("reference repository not available")
    sys.path.insert(0, REF)
    import envs.scenes._common as common  # noqa: E402
    import envs.scenes.corridor_static_object_random as sor  # noqa: E402
    import envs.turtlebot_var_scan_rate as tb  # noqa: E402
    return common, sor, tb


def test_constants_match_reference(ref):
    common, sor, tb = ref
    assert S.OUTER_WALLS == common.OUTER_WALLS and S.CORRIDOR_WALLS == common.CORRIDOR_WALLS
    assert S.WALL_THICKNESS == common.WALL_THICKNESS
    assert S.LENGTH_RANGE == sor.LENGTH_RANGE and S.THICKNESS_RANGE == sor.THICKNESS_RANGE
    assert S.OFFSET_RANGES == (sor.Z1_X_OFFSET_RANGE, sor.Z2_X_OFFSET_RANGE, sor.Z3_Y_OFFSET_RANGE, sor.Z4_Y_OFFSET_RANGE)
    assert S.MIN_Z1_Z3_DISTANCE == sor.MIN_Z1_Z3_DISTANCE and S.MIN_NOMINAL_CLEARANCE == sor.MIN_NOMINAL_CLEARANCE
    assert S.NOMINAL == sor.NOMINAL_STATIC_OBSTACLES
    assert S.STARTING_POS == tb.STARTING_POS and S.STARTING_YAW == tb.STARTING_YAW
    assert S.START_POSE_LIST == tb.START_POSE_LIST and S.START_YAW_CENTER == tb.START_YAW_CENTER
    assert S.START_YAW_NOISE == tb.START_YAW_NOISE and tuple(S.GOAL_LIST_DEFAULT) == tuple(map(tuple, tb.GOAL_LIST))


def test_z_walls_and_rect_distance_match_reference(ref):
    _, sor, _ = ref
    rows = [sor.sample_static_obstacles_params(s) for s in range(500)]
    L = torch.tensor([r["lengths"] for r in rows], dtype=torch.float64)
    T = torch.tensor([r["thicknesses"] for r in rows], dtype=torch.float64)
    O = torch.tensor([r["offsets"] for r in rows], dtype=torch.float64)
    ours = S.z_walls(L, T, O).numpy()
    for i, r in enumerate(rows):
        refw = np.array(sor.static_obstacle_walls(r["lengths"], r["thicknesses"], r["offsets"]))
        assert np.allclose(ours[i], refw, atol=1e-12)
    d = S.rect_distance(torch.tensor(ours[:, 0]), torch.tensor(ours[:, 2])).numpy()
    assert np.allclose(d, [r["d_z1_z3"] for r in rows], atol=1e-12)


def test_offset_resampling_respects_constraints_and_is_uniform(ref):
    _, sor, _ = ref
    rows = [sor.sample_static_obstacles_params(s) for s in range(256)]
    L = torch.tensor([r["lengths"] for r in rows]).repeat(40, 1).float()
    T = torch.tensor([r["thicknesses"] for r in rows]).repeat(40, 1).float()
    F = torch.tensor([r["offsets"] for r in rows]).repeat(40, 1).float()
    g = torch.Generator().manual_seed(0)
    O, fb = S.sample_offsets(L, T, F, generator=g)
    w = S.z_walls(L, T, O)
    assert torch.all(S.rect_distance(w[:, 0], w[:, 2]) >= S.MIN_Z1_Z3_DISTANCE - 1e-6)
    for j, (lo, hi) in enumerate(S.OFFSET_RANGES):
        assert O[:, j].min() >= lo - 1e-6 and O[:, j].max() <= hi + 1e-6
    assert fb.float().mean() < 0.01
    # constraints A, B (lengths only) hold for every bank entry
    assert torch.all(S.CORRIDOR_WIDTH - (L[:, 0] + L[:, 1]) >= S.MIN_NOMINAL_CLEARANCE - 1e-6)
    assert torch.all(S.CORRIDOR_WIDTH - (L[:, 2] + L[:, 3]) >= S.MIN_NOMINAL_CLEARANCE - 1e-6)


def test_start_and_goal_samplers():
    g = torch.Generator().manual_seed(1)
    s = S.sample_start(20000, True, "cpu", g)
    d = torch.cdist(s[:, :2].double(), torch.tensor(S.START_POSE_LIST, dtype=torch.float64))
    assert d.min(1).values.max() < 1e-6                        # every start is one of the 5 reference points
    dy = torch.remainder(s[:, 2] - math.pi + math.pi, 2 * math.pi) - math.pi
    assert dy.abs().max() <= S.START_YAW_NOISE + 1e-6
    f = S.sample_start(3, False, "cpu")
    assert torch.allclose(f[0], torch.tensor([2.5, 2.5, S.STARTING_YAW]))
    goals = S.sample_goal(9000, S.GOAL_LIST_ST01, "cpu", g)
    counts = np.unique(goals.numpy().round(2), axis=0, return_counts=True)[1]
    assert len(counts) == 9 and counts.min() > 850
