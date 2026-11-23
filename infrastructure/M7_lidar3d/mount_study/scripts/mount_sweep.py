"""Mount study, sweep: every candidate (mount height z, pitch) evaluated on the same robot poses and scene states from
the NAV01 (static) and NAV03 (dynamic) evaluation recordings + random free-space poses. x = y = 0, roll = yaw = 0, FoV,
resolution, range and geometry unchanged.

    isaacrun -g <gpu> -- python infrastructure/M7_lidar3d/mount_study/scripts/mount_sweep.py \
        --z 0.36 0.60 0.02 --pitch -2 16 0.5 --every 10 --n_random 500 --tag coarse
Writes results/sweep_<tag>.csv (+ .json with the pose sets).
"""
import argparse
import csv
import math
import time

import numpy as np
import torch

from mount_common import (BODY_PITCH_DEG, BODY_PITCH_WORST_DEG, BASE_Z, CLASSES, EVAL_DYNAMIC, EVAL_STATIC, OBST_H,
                          RESULTS, WALL_H, classify, load_poses, rays_world, save_json, self_occluded, sensor_dirs,
                          tb4_mesh_triangles, wall_boxes)
import episode_data as ED

p = argparse.ArgumentParser()
p.add_argument("--z", type=float, nargs=3, default=(0.36, 0.60, 0.02), help="start stop step (inclusive)")
p.add_argument("--pitch", type=float, nargs=3, default=(-2.0, 16.0, 0.5))
p.add_argument("--candidates", type=float, nargs="*", default=None, help="explicit z1 p1 z2 p2 ... instead of the grid")
p.add_argument("--every", type=int, default=10)
p.add_argument("--n_random", type=int, default=500)
p.add_argument("--chunk", type=int, default=128)
p.add_argument("--tag", default="coarse")
p.add_argument("--worst_body_pitch", action="store_true", help="also evaluate escapes at the worst observed body pitch")
args = p.parse_args()
dev = "cuda"
RESULTS.mkdir(parents=True, exist_ok=True)
dirs_s, _, _ = sensor_dirs(dev)
R = dirs_s.shape[0]
EL_MAX, EL_MIN = float(torch.rad2deg(torch.asin(dirs_s[:, 2])).max()), float(torch.rad2deg(torch.asin(dirs_s[:, 2])).min())
tris = tb4_mesh_triangles(dev)

scenes = {}
for name, ev, seed in (("static", EVAL_STATIC, 1), ("dynamic", EVAL_DYNAMIC, 2)):
    P = load_poses(ev, args.every, args.n_random, seed)
    _, tags = wall_boxes(ED.load_episode(ED.episode_dirs(ev)[0])["geometry"])
    scenes[name] = {k: torch.tensor(v, device=dev, dtype=torch.float32) if k in ("x", "y", "yaw", "boxes", "cyl") else v
                    for k, v in P.items()}
    scenes[name]["tags"] = torch.tensor(tags, device=dev)
    print(f"{name}: {len(P['x'])} poses ({int((P['source'] == 0).sum())} trajectory, {int((P['source'] == 1).sum())} random)",
          flush=True)

if args.candidates:
    cands = [(args.candidates[i], args.candidates[i + 1]) for i in range(0, len(args.candidates), 2)]
else:
    zs = np.round(np.arange(args.z[0], args.z[1] + 1e-9, args.z[2]), 4)
    ps = np.round(np.arange(args.pitch[0], args.pitch[1] + 1e-9, args.pitch[2]), 4)
    cands = [(float(z), float(pt)) for z in zs for pt in ps]
BANDS = (0.0, 1.0, 2.0, 4.0, 12.0)
OBIN = (0.0, 0.5, 1.0, 2.0, 3.0, 12.0)


def evaluate(S, z, pitch, body_pitch):
    """Aggregates for one scene: class counts, escapes, range bands of structure returns, obstacle rays vs distance."""
    P = S["x"].shape[0]
    acc = {"n_rays": 0, "cls": torch.zeros(5, device=dev, dtype=torch.float64), "esc": 0.0, "esc_out": 0.0,
           "pose_esc": 0.0, "pose_floor": 0.0, "src_cls": torch.zeros(2, 5, device=dev, dtype=torch.float64),
           "src_esc": torch.zeros(2, device=dev, dtype=torch.float64), "src_n": torch.zeros(2, device=dev, dtype=torch.float64),
           "band": torch.zeros(len(BANDS) - 1, device=dev, dtype=torch.float64),
           "ob_rays": torch.zeros(len(OBIN) - 1, device=dev, dtype=torch.float64),
           "ob_n": torch.zeros(len(OBIN) - 1, device=dev, dtype=torch.float64)}
    src_all = torch.tensor(S["source"], device=dev)
    for i in range(0, P, args.chunk):
        sl = slice(i, i + args.chunk)
        org, d_w = rays_world(S["x"][sl], S["y"][sl], S["yaw"][sl], z, pitch, dirs_s, body_pitch)
        t, cls, esc, esc_out, top = classify(org, d_w, S["boxes"][sl], S["tags"], S["cyl"][sl])
        acc["top"] = acc.get("top", 0.0) + float(top.sum())
        oh = torch.nn.functional.one_hot(cls, 5).double()          # (p, R, 5)
        acc["cls"] += oh.sum((0, 1)); acc["n_rays"] += cls.numel()
        acc["esc"] += float(esc.sum()); acc["esc_out"] += float(esc_out.sum())
        acc["pose_esc"] += float(esc.any(1).sum()); acc["pose_floor"] += float((cls == 0).any(1).sum())
        src = src_all[sl]
        for s in (0, 1):
            m = src == s
            acc["src_cls"][s] += oh[m].sum((0, 1)); acc["src_esc"][s] += esc[m].sum(); acc["src_n"][s] += m.sum() * R
        struct = (cls >= 1) & (cls <= 3)
        for b in range(len(BANDS) - 1):
            acc["band"][b] += (struct & (t >= BANDS[b]) & (t < BANDS[b + 1])).sum()
        # moving-obstacle rays per pose vs distance of the nearest active obstacle inside the horizontal FoV (+-14 deg)
        cyl = S["cyl"][sl]
        dx, dy = cyl[..., 0] - S["x"][sl, None], cyl[..., 1] - S["y"][sl, None]
        dist = torch.hypot(dx, dy)
        bear = torch.remainder(torch.atan2(dy, dx) - S["yaw"][sl, None] + math.pi, 2 * math.pi) - math.pi
        vis = (cyl[..., 2] > 0) & (bear.abs() < math.radians(14))
        dmin = torch.where(vis, dist, torch.full_like(dist, 1e9)).amin(1)
        n_ob = (cls == 3).sum(1).double()
        for b in range(len(OBIN) - 1):
            m = (dmin >= OBIN[b]) & (dmin < OBIN[b + 1])
            acc["ob_rays"][b] += n_ob[m].sum(); acc["ob_n"][b] += m.sum()
    n = acc["n_rays"]
    out = {f"{c}_pct": 100 * float(acc["cls"][k]) / n for k, c in enumerate(CLASSES)}
    out["escape_pct"] = 100 * acc["esc"] / n
    out["escape_outer_pct"] = 100 * acc["esc_out"] / n
    out["top_face_pct"] = 100 * acc["top"] / n
    out["poses_with_escape_pct"] = 100 * acc["pose_esc"] / P
    out["poses_with_floor_pct"] = 100 * acc["pose_floor"] / P
    for s, lab in ((0, "traj"), (1, "rand")):
        ns = float(acc["src_n"][s])
        if ns:
            out[f"{lab}_floor_pct"] = 100 * float(acc["src_cls"][s][0]) / ns
            out[f"{lab}_escape_pct"] = 100 * float(acc["src_esc"][s]) / ns
    for b in range(len(BANDS) - 1):
        out[f"struct_{BANDS[b]:g}-{BANDS[b + 1]:g}m_pct"] = 100 * float(acc["band"][b]) / n
    for b in range(len(OBIN) - 1):
        out[f"obst_rays_{OBIN[b]:g}-{OBIN[b + 1]:g}m"] = float(acc["ob_rays"][b] / acc["ob_n"][b]) if acc["ob_n"][b] > 0 else ""
    return out


def analytic(z, pitch, body_pitch=BODY_PITCH_DEG):
    """Flat-floor fan geometry (robot nominal attitude): top / bottom row elevation in the world, distances where the
    bottom row reaches the floor, where the top row exceeds the wall height, and where a 0.5 m obstacle top drops
    below the bottom row (near-field blind distance; 0 if the sensor is below the obstacle top)."""
    zo = z + BASE_Z
    top, bot = EL_MAX - pitch - body_pitch, EL_MIN - pitch - body_pitch
    tb, tt = math.tan(math.radians(-bot)), math.tan(math.radians(top))
    return {"z_origin_m": zo, "top_row_elev_deg": top, "bottom_row_elev_deg": bot,
            "floor_from_m": zo / tb if bot < 0 else float("inf"),
            "top_above_wall_from_m": (WALL_H - zo) / tt if top > 0 and zo < WALL_H else (0.0 if zo >= WALL_H else float("inf")),
            "fan_inside_0-0.6m_up_to_m": min(zo / tb if bot < 0 else float("inf"),
                                             (WALL_H - zo) / tt if top > 0 else float("inf")),
            "obstacle_blind_below_m": (zo - OBST_H) / tb if zo > OBST_H and bot < 0 else 0.0}


rows = []
t0 = time.time()
for ci, (z, pitch) in enumerate(cands):
    r = {"z_mount_m": z, "pitch_deg": pitch, **analytic(z, pitch)}
    occ = self_occluded(z, pitch, dirs_s, tris)
    r["self_occluded_rays"] = int(occ.sum())
    for name, S in scenes.items():
        for k, v in evaluate(S, z, pitch, BODY_PITCH_DEG).items():
            r[f"{name}_{k}"] = v
        if args.worst_body_pitch:
            w = evaluate(S, z, pitch, BODY_PITCH_WORST_DEG)
            r[f"{name}_escape_pct_worst_body_pitch"] = w["escape_pct"]
            r[f"{name}_floor_pct_worst_body_pitch"] = w["floor_pct"]
    for k in ("floor_pct", "wall_pct", "static_obstacle_pct", "moving_obstacle_pct", "no_return_pct", "escape_pct",
              "escape_outer_pct", "top_face_pct"):
        r[f"both_{k}"] = 0.5 * (r[f"static_{k}"] + r[f"dynamic_{k}"])
    rows.append(r)
    if ci % 20 == 0 or ci == len(cands) - 1:
        print(f"[{ci + 1}/{len(cands)}] z {z:.3f} pitch {pitch:5.2f}: floor {r['both_floor_pct']:5.2f} % escape "
              f"{r['both_escape_pct']:6.3f} % no-return {r['both_no_return_pct']:5.2f} % occluded {r['self_occluded_rays']} "
              f"({time.time() - t0:.0f} s)", flush=True)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(RESULTS / f"sweep_{args.tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
save_json(RESULTS / f"sweep_{args.tag}.json", {
    "args": vars(args), "poses": {n: {"n": int(S["x"].shape[0]), "trajectory": int((S["source"] == 0).sum()),
                                      "random": int((S["source"] == 1).sum())} for n, S in scenes.items()},
    "body_pitch_deg": BODY_PITCH_DEG, "worst_body_pitch_deg": BODY_PITCH_WORST_DEG, "base_z_m": BASE_Z,
    "row_elevations_deg": [EL_MIN, EL_MAX]})
print("wrote", RESULTS / f"sweep_{args.tag}.csv", flush=True)
