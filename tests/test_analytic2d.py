"""Parity of the torch analytic geometry with the reference simplified environment (read-only import).

The reference `_raycast` / `_wall_clearance` / `_ground_truth_clearance` are called as unbound methods on a minimal
stand-in object, with LiDAR noise off, for random poses in the reference corridor scene with obstacles."""
import math
import os
import sys
import types

import numpy as np
import pytest
import torch

REF = os.path.expanduser("~/adaptive_sensor_policy")
sys.dont_write_bytecode = True

from st_nav.geometry.analytic2d import (circle_clearance, raycast_2d, wall_clearance,  # noqa: E402
                                        walls_from_specs)


@pytest.fixture(scope="module")
def ref():
    if not os.path.isdir(REF):
        pytest.skip("reference repository not available")
    sys.path.insert(0, REF)
    import envs.turtlebot_var_scan_rate as tb  # noqa: E402
    return tb


def _stub(tb, walls_spec, obstacles, radii):
    s = types.SimpleNamespace()
    s.walls = tb._wall_arrays([tuple(w) if len(w) > 4 else (*w, tb.WALL_THICKNESS) for w in walls_spec])
    s.obstacles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 2)
    s.obstacle_radii = np.asarray(radii, dtype=np.float64)
    s.lidar_noise = False
    s._wall_clearance = lambda x, y: tb.TurtleBot_VarScanRate._wall_clearance(s, x, y)
    return s


def test_raycast_and_clearance_match_reference(ref):
    tb = ref
    from envs.scenes import corridor_static_object as sc
    walls_spec = sc.WALLS
    obstacles = [(1.9, 1.4), (-0.6, -0.9), (0.9, -1.9)]
    radii = [0.17, 0.10, 0.05]
    stub = _stub(tb, walls_spec, obstacles, radii)
    walls = walls_from_specs(walls_spec)
    circles = torch.tensor([[o[0], o[1], r] for o, r in zip(obstacles, radii)], dtype=torch.float32)
    rng = np.random.default_rng(0)
    poses = []
    while len(poses) < 300:
        x, y = rng.uniform(-2.8, 2.8, 2)
        if tb.TurtleBot_VarScanRate._wall_clearance(stub, x, y) > 0.05 and \
                all(math.hypot(x - o[0], y - o[1]) > r + 0.05 for o, r in zip(obstacles, radii)):
            poses.append((x, y, rng.uniform(-math.pi, math.pi)))
    P = torch.tensor(poses, dtype=torch.float32)
    # reference beams (its own convention) and offset, to test the geometry only
    beams = torch.tensor(tb.BEAM_OFFSETS, dtype=torch.float32)
    origin = P[:, :2] + tb.LIDAR_OFFSET_X * torch.stack([torch.cos(P[:, 2]), torch.sin(P[:, 2])], 1)
    ours = raycast_2d(origin, P[:, 2:3] + beams[None], walls, circles[None].expand(len(P), -1, -1),
                      max_range=tb.LIDAR_DISTANCE_CAP)
    errs, cerr = [], []
    for i, (x, y, th) in enumerate(poses):
        scan_norm, _ = tb.TurtleBot_VarScanRate._raycast(stub, x, y, th)
        errs.append(np.abs(scan_norm * tb.LIDAR_DISTANCE_CAP - ours[i].numpy()).max())
        d_ref, _ = tb.TurtleBot_VarScanRate._ground_truth_clearance(stub, x, y)
        p = torch.tensor([[x, y]], dtype=torch.float32)
        d_our = min(wall_clearance(p, walls).item(), circle_clearance(p, circles[None]).item())
        cerr.append(abs(d_ref - d_our))
    assert max(errs) < 1e-4, max(errs)          # float32 vs float64
    assert max(cerr) < 1e-5, max(cerr)


def test_simple_box_and_circle():
    walls = walls_from_specs([(2.0, 0.0, 4.0, math.pi / 2, 0.2)])    # 4 m long along y, x in [1.9, 2.1]
    circles = torch.tensor([[[0.0, 1.0, 0.25]]])
    o = torch.zeros(1, 2)
    ang = torch.tensor([[0.0, math.pi / 2, math.pi]])
    d = raycast_2d(o, ang, walls, circles, max_range=3.5)
    assert torch.allclose(d, torch.tensor([[1.9, 0.75, 3.5]]), atol=1e-6)
