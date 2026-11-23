"""Mount study, part 2: sensor between the decks, at about the original 2D LiDAR height, moved forward.
Sweep over x offset, z height, pitch and vertical FoV (y = 0, roll = yaw = 0; horizontal FoV 28 deg; 139 x 34 grid kept,
so a narrower vertical FoV samples more densely). Same poses, body attitude, oracle and metrics as mount_sweep.py, plus:
  * housing feasibility: the 85 (x) x 84 (y) x 73 (z) mm Mirrorcle enclosure, centred on the optical centre (assumed),
    must not contain robot mesh vertices, must stay below the top plate's underside (collision mesh 0.3443 m) and inside
    the footprint (base shell radius 0.177 m, checked at the enclosure's front corners);
  * near-field: distance ahead of the robot's front edge (x = 0.177 m) from which the horizontal fan covers the full robot
    width (+-0.177 m);
  * vertical coverage: fraction of a 0.6 m wall / a 0.5 m obstacle face covered by the fan at 0.5, 1, 2, 4 m (flat robot,
    nominal attitude), and the row spacing.

    isaacrun -g <gpu> -- python infrastructure/M7_lidar3d/mount_study/scripts/mount_sweep_xzfov.py --tag xzfov_coarse
    isaacrun -g <gpu> -- python .../mount_sweep_xzfov.py --configs "x z pitch fov; ..." --every 2 --n_random 5000 --tag xzfov_dense
"""
import argparse
import csv
import itertools
import math
import time

import numpy as np
import torch

from mount_common import (BASE_Z, BODY_PITCH_DEG, BODY_PITCH_WORST_DEG, CLASSES, EVAL_DYNAMIC, EVAL_STATIC, OBST_H,
                          RESULTS, ROOT, WALL_H, classify, load_poses, quat_apply, quat_mul, save_json, wall_boxes)
from st_nav.sensors.ray_tables import grid_pattern, rpy_to_quat_wxyz
import episode_data as ED

HOUSING = (0.085, 0.084, 0.073)          # Mirrorcle enclosure x, y, z [m]; optical centre assumed at its centre
PLATE_UNDERSIDE = 0.3443                 # top plate underside (collision mesh)
TOP_PLATE = 0.3498                       # top plate upper surface
FOOTPRINT_R = 0.177                      # base shell radius
ROBOT_HALF_W = 0.177
H_HALF = 14.0
ORIG_2D = (-0.040, 0.0, 0.192915)        # original TurtleBot4 RPLIDAR optical centre in base_link
BASELINES = {"original M7 (0, 0.40, 5, 14)": (0.0, 0.3998, 5.0, 14.0), "R4 (0, 0.40, 6.75, 14)": (0.0, 0.40, 6.75, 14.0),
             "R1 (0, 0.50, 7.5, 14)": (0.0, 0.50, 7.5, 14.0)}


def dirs_for(vfov, device):
    az, el, row, col = grid_pattern(h_fov_deg=(-H_HALF, H_HALF), v_fov_deg=(-vfov / 2, vfov / 2))
    a, e = torch.deg2rad(torch.tensor(az, device=device)), torch.deg2rad(torch.tensor(el, device=device))
    return torch.stack([torch.cos(e) * torch.cos(a), torch.cos(e) * torch.sin(a), torch.sin(e)], -1), max(el), min(el)


def rays_world_x(px, py, pyaw, mount, pitch_deg, dirs_s, body_pitch_deg):
    dev = dirs_s.device
    P = px.shape[0]
    q_yaw = torch.stack([torch.cos(pyaw / 2), 0 * pyaw, 0 * pyaw, torch.sin(pyaw / 2)], -1)
    q_body = torch.tensor(rpy_to_quat_wxyz((0.0, body_pitch_deg, 0.0)), device=dev, dtype=torch.float32).expand(P, 4)
    q_base = quat_mul(q_yaw, q_body)
    q_m = torch.tensor(rpy_to_quat_wxyz((0.0, pitch_deg, 0.0)), device=dev, dtype=torch.float32)
    d_b = quat_apply(q_m.expand(dirs_s.shape[0], 4), dirs_s)
    m = torch.tensor(mount, device=dev, dtype=torch.float32).expand(P, 3)
    org = torch.stack([px, py, torch.full_like(px, BASE_Z)], -1) + quat_apply(q_base, m)
    d_w = quat_apply(q_base[:, None].expand(-1, d_b.shape[0], -1), d_b[None].expand(P, -1, -1))
    return org, d_w


def load_mesh(device):
    g = np.load(ROOT / "infrastructure/robot_viz/results/tb4_geometry.npz")
    V = np.concatenate([g["base_link|visual|v"], g["base_link|collision|v"]]).astype(np.float64)
    tris = np.concatenate([g[f"base_link|{k}|v"][g[f"base_link|{k}|f"]] for k in ("visual", "collision")])
    for b in ("left_wheel", "right_wheel"):
        T = g[f"rest|{b}"]
        vw = g[f"{b}|visual|v"] @ T[:3, :3].T + T[:3, 3]
        V = np.concatenate([V, vw])
        tris = np.concatenate([tris, vw[g[f"{b}|visual|f"]]])
    return V, torch.tensor(tris, dtype=torch.float32, device=device)


def housing_check(V, x, z):
    hx, hy, hz = (h / 2 for h in HOUSING)
    inside = ((np.abs(V[:, 0] - x) <= hx) & (np.abs(V[:, 1]) <= hy) & (np.abs(V[:, 2] - z) <= hz)).sum()
    corner_r = math.hypot(x + hx, hy)
    return {"housing_mesh_vertices_inside": int(inside), "housing_top_m": z + hz, "housing_front_x_m": x + hx,
            "housing_front_corner_r_m": corner_r,
            "placement": "on top of the plate" if z - hz >= TOP_PLATE else "between the decks",
            "fits": bool(inside == 0 and (z + hz <= PLATE_UNDERSIDE or z - hz >= TOP_PLATE) and corner_r <= FOOTPRINT_R)}


def self_occluded(mount, pitch, dirs_s, tris, t_max=0.6):
    dev = dirs_s.device
    q_m = torch.tensor(rpy_to_quat_wxyz((0.0, pitch, 0.0)), device=dev, dtype=torch.float32)
    d = quat_apply(q_m.expand(dirs_s.shape[0], 4), dirs_s)
    o = torch.tensor(mount, device=dev, dtype=torch.float32)
    # only triangles that can be reached: in front of the sensor's back plane, within t_max
    c = tris.mean(1)
    sel = ((c - o).norm(dim=1) < t_max) & (c[:, 0] > o[0] - 0.05)
    tr = tris[sel]
    hx, hy, hz = (h / 2 for h in HOUSING)
    inside_h = ((tr[:, :, 0] - o[0]).abs() <= hx) & (tr[:, :, 1].abs() <= hy) & ((tr[:, :, 2] - o[2]).abs() <= hz)
    tr = tr[~inside_h.all(1)]                                      # parts replaced by the housing itself are ignored
    v0, e1, e2 = tr[:, 0], tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0]
    out = torch.zeros(d.shape[0], dtype=torch.bool, device=dev)
    for i in range(0, d.shape[0], 64):
        dd = d[i:i + 64, None]
        p = torch.cross(dd.expand(-1, e2.shape[0], -1), e2[None].expand(dd.shape[0], -1, -1), dim=-1)
        det = (e1[None] * p).sum(-1)
        ok = det.abs() > 1e-12
        inv = torch.where(ok, 1.0 / torch.where(ok, det, torch.ones_like(det)), torch.zeros_like(det))
        s = o - v0
        u = (s[None] * p).sum(-1) * inv
        q = torch.cross(s, e1, dim=-1)
        v = (dd * q[None]).sum(-1) * inv
        t = (e2 * q).sum(-1)[None] * inv
        out[i:i + 64] = (ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < t_max)).any(1)
    return out


BANDS = (0.0, 1.0, 2.0, 4.0, 12.0)
OBIN = (0.0, 0.5, 1.0, 2.0, 3.0, 12.0)


def evaluate(S, mount, pitch, dirs_s, body_pitch, chunk):
    R = dirs_s.shape[0]
    P = S["x"].shape[0]
    acc = {"cls": torch.zeros(5, device=dirs_s.device, dtype=torch.float64), "n": 0, "esc": 0.0, "top": 0.0,
           "pose_esc": 0.0, "pose_floor": 0.0, "band": np.zeros(len(BANDS) - 1), "ob_rays": np.zeros(len(OBIN) - 1),
           "ob_n": np.zeros(len(OBIN) - 1)}
    for i in range(0, P, chunk):
        sl = slice(i, i + chunk)
        org, d_w = rays_world_x(S["x"][sl], S["y"][sl], S["yaw"][sl], mount, pitch, dirs_s, body_pitch)
        t, cls, esc, _, top = classify(org, d_w, S["boxes"][sl], S["tags"], S["cyl"][sl])
        acc["cls"] += torch.nn.functional.one_hot(cls, 5).double().sum((0, 1)); acc["n"] += cls.numel()
        acc["esc"] += float(esc.sum()); acc["top"] += float(top.sum())
        acc["pose_esc"] += float(esc.any(1).sum()); acc["pose_floor"] += float((cls == 0).any(1).sum())
        struct = (cls >= 1) & (cls <= 3)
        for b in range(len(BANDS) - 1):
            acc["band"][b] += float((struct & (t >= BANDS[b]) & (t < BANDS[b + 1])).sum())
        cyl = S["cyl"][sl]
        dx, dy = cyl[..., 0] - S["x"][sl, None], cyl[..., 1] - S["y"][sl, None]
        dist = torch.hypot(dx, dy)
        bear = torch.remainder(torch.atan2(dy, dx) - S["yaw"][sl, None] + math.pi, 2 * math.pi) - math.pi
        vis = (cyl[..., 2] > 0) & (bear.abs() < math.radians(14))
        dmin = torch.where(vis, dist, torch.full_like(dist, 1e9)).amin(1)
        n_ob = (cls == 3).sum(1).double()
        for b in range(len(OBIN) - 1):
            m = (dmin >= OBIN[b]) & (dmin < OBIN[b + 1])
            acc["ob_rays"][b] += float(n_ob[m].sum()); acc["ob_n"][b] += float(m.sum())
    n = acc["n"]
    out = {f"{c}_pct": 100 * float(acc["cls"][k]) / n for k, c in enumerate(CLASSES)}
    out.update(escape_pct=100 * acc["esc"] / n, top_face_pct=100 * acc["top"] / n,
               poses_with_escape_pct=100 * acc["pose_esc"] / P, poses_with_floor_pct=100 * acc["pose_floor"] / P)
    for b in range(len(BANDS) - 1):
        out[f"struct_{BANDS[b]:g}-{BANDS[b + 1]:g}m_pct"] = 100 * acc["band"][b] / n
    for b in range(len(OBIN) - 1):
        out[f"obst_rays_{OBIN[b]:g}-{OBIN[b + 1]:g}m"] = acc["ob_rays"][b] / acc["ob_n"][b] if acc["ob_n"][b] else ""
    return out


def analytic(x, z, pitch, el_max, el_min, body=BODY_PITCH_DEG):
    zo = z + BASE_Z
    top, bot = el_max - pitch - body, el_min - pitch - body
    tb, tt = math.tan(math.radians(-bot)), math.tan(math.radians(top))
    out = {"z_origin_m": zo, "top_row_elev_deg": top, "bottom_row_elev_deg": bot, "row_spacing_deg": (el_max - el_min) / 33,
           "floor_from_m": zo / tb if bot < 0 else float("inf"),
           "top_above_wall_from_m": (WALL_H - zo) / tt if top > 0 else float("inf"),
           "obstacle_blind_below_m": (zo - OBST_H) / tb if zo > OBST_H and bot < 0 else 0.0,
           "full_width_ahead_of_front_m": x + ROBOT_HALF_W / math.tan(math.radians(H_HALF)) - FOOTPRINT_R,
           "zero_escape_bound_pitch_nominal_deg": el_max - body - math.degrees(math.atan((WALL_H - zo) / 8.02)),
           "zero_escape_bound_pitch_worst_deg": el_max - BODY_PITCH_WORST_DEG - math.degrees(math.atan((WALL_H - zo) / 8.02))}
    for d in (0.5, 1.0, 2.0, 4.0):                                 # distance from the sensor
        lo, hi = zo + d * math.tan(math.radians(bot)), zo + d * math.tan(math.radians(top))
        out[f"wall_cover_{d:g}m_pct"] = 100 * max(0.0, min(hi, WALL_H) - max(lo, 0.0)) / WALL_H
        out[f"obst_cover_{d:g}m_pct"] = 100 * max(0.0, min(hi, OBST_H) - max(lo, 0.0)) / OBST_H
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=float, nargs="+", default=[0.0, 0.043, 0.086, 0.129])
    p.add_argument("--z", type=float, nargs="+", default=[0.193, 0.230, 0.265, 0.298])
    p.add_argument("--pitch", type=float, nargs="+", default=[1.5, 2.5, 3.5, 5.0])
    p.add_argument("--vfov", type=float, nargs="+", default=[14.0, 12.0, 10.0, 8.0])
    p.add_argument("--configs", default=None, help='explicit "x z pitch vfov; ..." (replaces the grid)')
    p.add_argument("--baselines", action="store_true", help="also evaluate the original M7 mount, R4 and R1")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--n_random", type=int, default=500)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--tag", default="xzfov_coarse")
    a = p.parse_args()
    dev = "cuda"
    V, tris = load_mesh(dev)
    scenes = {}
    for name, ev, seed in (("static", EVAL_STATIC, 1), ("dynamic", EVAL_DYNAMIC, 2)):
        P = load_poses(ev, a.every, a.n_random, seed)
        _, tags = wall_boxes(ED.load_episode(ED.episode_dirs(ev)[0])["geometry"])
        scenes[name] = {k: torch.tensor(v, device=dev, dtype=torch.float32) if k in ("x", "y", "yaw", "boxes", "cyl") else v
                        for k, v in P.items()}
        scenes[name]["tags"] = torch.tensor(tags, device=dev)
        print(f"{name}: {len(P['x'])} poses", flush=True)
    if a.configs:
        cfgs = [("", *map(float, c.split())) for c in a.configs.split(";") if c.strip()]
    else:
        cfgs = [("", *c) for c in itertools.product(a.x, a.z, a.pitch, a.vfov)]
    if a.baselines:
        cfgs = [(k, *v) for k, v in BASELINES.items()] + cfgs
    dir_cache, rows, t0 = {}, [], time.time()
    for i, (label, x, z, pitch, vfov) in enumerate(cfgs):
        if vfov not in dir_cache:
            dir_cache[vfov] = dirs_for(vfov, dev)
        dirs_s, el_max, el_min = dir_cache[vfov]
        mount = (x, 0.0, z)
        r = {"label": label, "x_m": x, "z_m": z, "pitch_deg": pitch, "vfov_deg": vfov, **housing_check(V, x, z),
             **analytic(x, z, pitch, el_max, el_min)}
        r["self_occluded_rays"] = int(self_occluded(mount, pitch, dirs_s, tris).sum())
        for sc, S in scenes.items():
            for k, v in evaluate(S, mount, pitch, dirs_s, BODY_PITCH_DEG, a.chunk).items():
                r[f"{sc}_{k}"] = v
            w = evaluate(S, mount, pitch, dirs_s, BODY_PITCH_WORST_DEG, a.chunk)
            r[f"{sc}_escape_pct_worst"] = w["escape_pct"]
        for k in ("floor_pct", "wall_pct", "static_obstacle_pct", "moving_obstacle_pct", "no_return_pct", "escape_pct",
                  "escape_pct_worst", "top_face_pct"):
            r[f"both_{k}"] = 0.5 * (r[f"static_{k}"] + r[f"dynamic_{k}"])
        rows.append(r)
        if i % 16 == 0 or i == len(cfgs) - 1:
            print(f"[{i + 1}/{len(cfgs)}] x {x:.3f} z {z:.3f} p {pitch:4.1f} fov {vfov:4.1f}: fits {r['fits']} floor "
                  f"{r['both_floor_pct']:5.2f} esc {r['both_escape_pct']:.3f}/{r['both_escape_pct_worst']:.3f} occl "
                  f"{r['self_occluded_rays']} ({time.time() - t0:.0f} s)", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(RESULTS / f"sweep_{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    save_json(RESULTS / f"sweep_{a.tag}.json", {"args": vars(a), "housing_m": HOUSING, "plate_underside_m": PLATE_UNDERSIDE,
                                                 "footprint_r_m": FOOTPRINT_R, "original_2d_lidar_m": ORIG_2D,
                                                 "poses": {n: int(S["x"].shape[0]) for n, S in scenes.items()}})
    print("wrote", RESULTS / f"sweep_{a.tag}.csv", flush=True)


if __name__ == "__main__":
    main()
