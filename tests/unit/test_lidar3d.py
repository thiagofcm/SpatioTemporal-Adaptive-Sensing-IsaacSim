"""CPU tests of the 3D LiDAR building blocks (no simulator): ray grid and ordering, mount rotation convention, keyed
Gaussian range noise, self-occlusion of the default mount by the TurtleBot4 mesh, and the per-scan recording layout
read back by the visualization tools."""
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from st_nav.sensors.noise import GaussianRangeNoise, GaussianRangeNoiseCfg
from st_nav.sensors.ray_tables import RAY_ORDERS, grid_pattern, rpy_to_quat_wxyz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import episode_data as ED  # noqa: E402

MOUNT = (0.0, 0.0, 0.3998)          # Lidar3DCfg default (TB4 top plate 0.3498 m + 5 cm)
PITCH = 5.0


def quat_rotate(q, v):
    return ED.quat_apply(np.asarray(q)[None], np.asarray(v, dtype=np.float64))


def rot_matrix(rpy_deg):
    r, p, y = (math.radians(a) for a in rpy_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def dirs_from(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


# ------------------------------------------------------------------------------------------------ grid
def test_default_grid_cell_centres_row_major_top_left():
    az, el, row, col = grid_pattern()
    assert len(az) == len(el) == len(row) == len(col) == 34 * 139 == 4726
    az, el = np.array(az), np.array(el)
    assert np.isclose(az.max(), 14 - 14 / 139) and np.isclose(az.min(), -(14 - 14 / 139))       # +-13.8993
    assert np.isclose(el.max(), 7 - 7 / 34) and np.isclose(el.min(), -(7 - 7 / 34))             # +-6.7941
    assert np.allclose(np.diff(np.unique(az)), 28 / 139) and np.allclose(np.diff(np.unique(el)), 14 / 34)
    assert az[0] == az.max() and el[0] == el.max()                    # ray 0 = top-left (left = +azimuth)
    assert az[138] == az.min() and el[138] == el.max() and el[139] < el[0]   # first row ends top-right; row 1 below
    i = np.arange(4726)
    assert np.array_equal(np.array(row), i // 139) and np.array_equal(np.array(col), i % 139)   # index = row*139 + col


@pytest.mark.parametrize("order", RAY_ORDERS)
def test_orders_permute_the_same_cells(order):
    ref = {(r, c): (a, e) for a, e, r, c in zip(*grid_pattern())}
    az, el, row, col = grid_pattern(order=order)
    assert len(set(zip(row, col))) == 4726
    for a, e, r, c in zip(az, el, row, col):                          # grid_row / grid_col independent of the order
        assert ref[(r, c)] == (a, e)
    if order == "serpentine_top_left":
        assert col[139] == 138 and row[139] == 1
    if order == "column_major_top_left":
        assert row[:34] == list(range(34)) and set(col[:34]) == {0}
    if order == "row_major_bottom_left":
        assert row[0] == 33 and col[0] == 0


def test_endpoints_placement_and_errors():
    az, el, _, _ = grid_pattern(placement="endpoints")
    assert np.isclose(max(az), 14) and np.isclose(min(el), -7)
    with pytest.raises(ValueError):
        grid_pattern(order="spiral")
    with pytest.raises(ValueError):
        grid_pattern(placement="corners")


# ------------------------------------------------------------------------------------------------ rotation convention
def test_rpy_convention_rep103():
    fwd = np.array([1.0, 0, 0])
    d = quat_rotate(rpy_to_quat_wxyz((0, 5, 0)), fwd)[0]
    assert np.allclose(d, [math.cos(math.radians(5)), 0, -math.sin(math.radians(5))])   # pitch > 0 = looking DOWN
    assert np.allclose(quat_rotate(rpy_to_quat_wxyz((0, 0, 90)), fwd)[0], [0, 1, 0])    # yaw > 0 = turn left
    assert np.allclose(quat_rotate(rpy_to_quat_wxyz((90, 0, 0)), [0, 1, 0])[0], [0, 0, 1])   # roll > 0: y -> z
    rng = np.random.default_rng(0)
    for rpy in rng.uniform(-170, 170, (20, 3)):                       # = Rz(yaw) Ry(pitch) Rx(roll)
        v = rng.normal(size=(5, 3))
        assert np.allclose(quat_rotate(rpy_to_quat_wxyz(tuple(rpy)), v), v @ rot_matrix(rpy).T, atol=1e-9)
    assert rpy_to_quat_wxyz((0, 0, 0)) == (1.0, 0.0, 0.0, 0.0)        # 2D preset: identity, bit-exact


# ------------------------------------------------------------------------------------------------ noise
def test_gaussian_noise_keyed():
    n = GaussianRangeNoise(GaussianRangeNoiseCfg(sigma=0.01), "cpu")
    N, R = 64, 4726
    d = torch.full((N, R), 5.0)
    hit = torch.ones(N, R, dtype=torch.bool)
    hit[:, -100:] = False
    d[:, -100:] = 12.0
    seed, tick = torch.arange(N, dtype=torch.int64) * 7919 + 3, torch.full((N,), 120, dtype=torch.int64)
    y = n(d, hit, 12.0, keys=(seed, tick))
    assert torch.equal(y, n(d, hit, 12.0, keys=(seed, tick)))                           # reproducible
    sub = n(d[5:9], hit[5:9], 12.0, keys=(seed[5:9], tick[5:9]))
    assert torch.equal(sub, y[5:9])                                                     # batch-independent
    assert torch.equal(y[:, -100:], d[:, -100:])                                        # no return: unchanged
    e = (y[:, :-100] - 5.0).double()
    assert abs(e.mean().item()) < 2e-4 and abs(e.std().item() - 0.01) < 2e-4
    t2 = n(d, hit, 12.0, keys=(seed, tick + 1))
    assert not torch.equal(t2, y)                                                       # new tick, new draw
    near = n(torch.full((1, 1000), 11.999), torch.ones(1, 1000, dtype=torch.bool), 12.0, keys=(seed[:1], tick[:1]))
    assert near.max() <= 12.0 and near.min() >= 0.0                                     # clamped to [0, range_max]
    off = GaussianRangeNoise(GaussianRangeNoiseCfg(sigma=0.01, enabled=False), "cpu")
    assert torch.equal(off(d, hit, 12.0, keys=(seed, tick)), d)
    zero = GaussianRangeNoise(GaussianRangeNoiseCfg(sigma=0.0), "cpu")
    assert torch.equal(zero(d, hit, 12.0, keys=(seed, tick)), d)


# ------------------------------------------------------------------------------------------------ self-occlusion
def _ray_triangle_hits(o, dirs, tri, t_max=0.5):
    """Moller-Trumbore: (R,) bool, any triangle hit at 1e-6 < t < t_max."""
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = np.zeros(len(dirs), bool)
    for i in range(0, len(dirs), 256):
        d = dirs[i:i + 256, None, :]
        p = np.cross(d, e2[None])
        det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - v0
        u = (s[None] * p).sum(-1) * inv
        q = np.cross(s, e1)
        v = (d * q[None]).sum(-1) * inv
        t = (e2[None] * q[None]).sum(-1) * inv
        out[i:i + 256] = (ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < t_max)).any(1)
    return out


def test_default_mount_not_occluded_by_robot():
    g = np.load(os.path.join(ROOT, "infrastructure", "robot_viz", "results", "tb4_geometry.npz"))
    assert np.allclose(g["frame|base_link"], np.eye(4))
    az, el, _, _ = grid_pattern()
    dirs = dirs_from(az, el) @ rot_matrix((0, PITCH, 0)).T             # base_link directions
    o = np.array(MOUNT)
    top = max(float(g[f"base_link|{k}|v"][:, 2].max()) for k in ("visual", "collision"))
    assert top < MOUNT[2] - 0.045                                     # mount 5 cm above the highest robot point
    for k in ("visual", "collision"):
        v, f = g[f"base_link|{k}|v"].astype(np.float64), g[f"base_link|{k}|f"]
        tri = v[f]
        tri = tri[tri[:, :, 2].max(1) > 0.30]                         # rays stay above z = 0.36 inside r = 0.18 m
        assert not _ray_triangle_hits(o, dirs, tri).any(), k
    # vertical margin of the lowest ray above the plate at the robot's outer radius (0.178 m)
    lowest = math.radians(-(7 - 7 / 34) - PITCH)
    assert MOUNT[2] + 0.178 * math.tan(lowest) - top > 0.01


# ------------------------------------------------------------------------------------------------ per-scan layout
def per_scan_episode(n=10):
    meta, geo, traj, steps, _, obst = _synthetic()
    az, el, row, col = grid_pattern(cols=8, rows=4)
    R = len(az)
    dirs_s = dirs_from(az, el)
    q_m = rpy_to_quat_wxyz((0, PITCH, 0))
    dirs_b = quat_rotate(q_m, dirs_s)
    scan_rows = np.array([0, 2, 3, 4, 5, 7, 8, 9])                     # 4 Hz scans at 5 Hz control
    S = len(scan_rows)
    rng = np.random.default_rng(1)
    clean = rng.uniform(0.5, 12.0, (S, R)).astype(np.float32)
    clean[:, -1] = 12.0                                               # one no-return ray
    yaw = np.linspace(0, 0.5, S)
    quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1)
    fresh = np.isin(np.arange(n + 1), scan_rows)
    lidar = {"layout": np.array("per_scan"), "t_sim": np.arange(n + 1) * 0.2, "fresh": fresh,
             "row_scan_index": np.searchsorted(scan_rows, np.arange(n + 1), side="right") - 1,
             "scan_step": scan_rows, "scan_t_sim": scan_rows * 0.2, "ranges": clean + 0.01, "ranges_clean": clean,
             "origin_xyz": np.stack([np.linspace(1, 2, S), np.ones(S), np.full(S, 0.4)], 1), "base_quat_wxyz": quat,
             "azimuth_deg": np.array(az), "elevation_deg": np.array(el), "ray_dirs_s": dirs_s, "ray_dirs_b": dirs_b,
             "grid_row": np.array(row), "grid_col": np.array(col), "grid_shape": np.array([4, 8]),
             "mount_offset_xyz": np.array(MOUNT), "mount_rpy_deg": np.array([0, PITCH, 0]), "range_max": np.float64(12.0)}
    meta["geometry_heights_m"] = {"wall": 0.6}
    return meta, geo, traj, steps, lidar, obst


def _synthetic():
    from test_episode_data import synthetic_episode
    return synthetic_episode(10)


def test_per_scan_round_trip_and_geometry(tmp_path):
    meta, geo, traj, steps, lidar, obst = per_scan_episode()
    d = tmp_path / ED.episode_dir_name(3)
    ED.write_episode(d, meta, geo, traj, steps, lidar, obst)
    ep = ED.load_episode(d)
    assert ED.is_per_scan(ep) and ep["meta"]["rows"]["lidar_scans"] == 8
    assert ep["lidar"]["ranges"].dtype == np.float32 and ep["lidar"]["ranges"].shape == (8, 32)
    assert ED.scan_index(ep, 1) == 0 and ED.scan_index(ep, 6) == 4 and ED.scan_index(ep, 10) == 7
    i = 5
    sc = ED.lidar_scan(ep, i, clean=True)
    yaw = 0.5 * i / 7
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    exp = lidar["origin_xyz"][i] + lidar["ranges_clean"][i][:, None] * (lidar["ray_dirs_b"] @ Rz.T)
    assert np.allclose(sc["points"], exp, atol=1e-5) and sc["hit"][:-1].all() and not sc["hit"][-1]
    img = ED.range_image(ep, i)
    assert img.shape == (4, 8) and img[0, 0] == ep["lidar"]["ranges"][i][0] and img[3, 7] == ep["lidar"]["ranges"][i][-1]
    o, ends, hit = ED.lidar_points(ep, 6)                             # 2D view = top-down projection of scan 4
    assert np.allclose(ends, ED.lidar_scan(ep, 4)["points"][:, :2])


def test_visualization_tools_per_scan(tmp_path):
    meta, geo, traj, steps, lidar, obst = per_scan_episode()
    ev = tmp_path / "eval"
    ED.write_episode(ev / ED.episode_dir_name(3), meta, geo, traj, steps, lidar, obst)
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin"}
    probe = ("import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__'); "
             "bad = sorted({m.split('.')[0] for m in sys.modules} & {'torch', 'st_nav', 'isaaclab', 'omni'}); "
             "print('FORBIDDEN', bad)")
    r = subprocess.run([sys.executable, "-B", "-c", probe, os.path.join(ROOT, "tools", "plot_episode.py"),
                        str(ev / "episode_0003")], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "FORBIDDEN []" in r.stdout
    assert (ev / "episode_0003" / "episode_0003.png").exists()


# ------------------------------------------------------------------------------------------------ 3D oracle
def test_analytic3d_primitives():
    from st_nav.geometry.analytic3d import raycast_3d
    o = torch.tensor([[0.0, 0.0, 0.4]])
    boxes = torch.tensor([[[2.0, 0.0, 0.5, 0.075, 0.0, 0.6],          # wall across +x at x = 1.925 (rotated: thin in x)
                           [0.0, 3.0, 1.0, 0.075, 0.0, 0.3]]])        # low wall along x at y = 3, 0.3 m high
    boxes[0, 0, 4] = math.pi / 2                                       # yaw 90: half_length along y, thickness along x
    cyl = torch.tensor([[[-2.0, 0.0, 0.17, 0.5], [0.0, -2.0, 0.0, 0.5]]])   # second one absent (r = 0)

    def ray(az_deg, el_deg):
        a, e = math.radians(az_deg), math.radians(el_deg)
        return [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]

    dirs = torch.tensor([[ray(0, 0), ray(0, 10), ray(90, 0), ray(90, 2), ray(180, 0), ray(-90, 0), ray(-90, -10),
                          ray(0, 30)]])
    t, kind = raycast_3d(o, dirs, boxes, cyl, floor=True, max_range=12.0, return_kind=True)
    t, kind = t[0].tolist(), kind[0].tolist()
    assert math.isclose(t[0], 1.925, abs_tol=1e-5) and kind[0] == 2                     # wall face
    assert kind[1] == 0 and t[1] == 12.0                                                # 10 deg up: over the 0.6 m wall
    assert t[2] == 12.0 and kind[2] == 0                                                # horizontal at z 0.4: over the 0.3 m wall
    assert math.isclose(t[4], 2.0 - 0.17, abs_tol=1e-5) and kind[4] == 3                # cylinder side
    assert t[5] == 12.0 and kind[5] == 0                                                # absent cylinder: nothing
    assert math.isclose(t[6], 0.4 / math.sin(math.radians(10)), abs_tol=1e-5) and kind[6] == 1   # floor
    assert t[7] == 12.0                                                                 # steep up: nothing
    # 2 deg up toward the 0.3 m low wall at y = 2.925: z = 0.4 + 2.925 tan 2deg > 0.3 -> over it
    assert kind[3] == 0
    # top cap: origin above a 0.5 m cylinder, looking down onto it
    t2 = raycast_3d(torch.tensor([[-2.0, 0.0, 1.0]]), torch.tensor([[[0.0, 0.0, -1.0]]]), None, cyl, floor=True)
    assert math.isclose(float(t2[0, 0]), 0.5, abs_tol=1e-6)
