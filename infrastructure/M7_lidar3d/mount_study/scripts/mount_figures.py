"""Mount study, figures: LiDAR returns of candidate mounts reconstructed for the SAME recorded frames (robot pose,
baffles and obstacle positions from the NAV01 / NAV03 evaluation recordings), one figure per frame:
rows = candidates (baseline first), columns = top-down | side view | 3D point cloud | range image.
Colours / scales / viewpoints are identical across candidates and frames.

    isaacrun -g <gpu> -- python infrastructure/M7_lidar3d/mount_study/scripts/mount_figures.py \
        --candidates 0.3998 5.0 0.45 7.0 ... --labels "current" "A" ...
Frames are chosen automatically (printed and written to results/frames.json):
  F1 static, the trajectory pose with the most escaping rays for the current mount (long view along a corridor)
  F2 static, a trajectory pose with a baffle 0.6-2 m ahead covering 15-50 % of the rays (most floor / wall rays)
  F3 dynamic, a trajectory pose with an active moving obstacle 0.8-1.5 m ahead (inside the horizontal FoV)
"""
import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from mount_common import (CURRENT, EVAL_DYNAMIC, EVAL_STATIC, OBST_H, RANGE_MAX, RESULTS, WALL_H, classify, load_poses,  # noqa: E402
                          rays_world, save_json, sensor_dirs, wall_boxes)
from st_nav.geometry.analytic3d import raycast_3d  # noqa: E402
import episode_data as ED  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--candidates", type=float, nargs="+", required=True, help="z1 pitch1 z2 pitch2 ... (baseline first)")
p.add_argument("--labels", nargs="+", required=True)
p.add_argument("--tag", default="shortlist")
args = p.parse_args()
dev = "cuda"
cands = [(args.candidates[i], args.candidates[i + 1]) for i in range(0, len(args.candidates), 2)]
assert len(cands) == len(args.labels)
dirs_s, grow, gcol = sensor_dirs(dev)

COL = {0: "#6b6b6b", 1: "#1f77b4", 2: "#ff7f0e", 3: "#9467bd", 4: "#d62728"}   # floor, wall, static obst., moving, (escape)
NAMES = {0: "floor", 1: "wall", 2: "static obstacle", 3: "moving obstacle"}
ESC_C = "#d62728"


def pick_frames():
    frames = []
    for name, ev, seed in (("static", EVAL_STATIC, 1), ("dynamic", EVAL_DYNAMIC, 2)):
        P = load_poses(ev, 2, 0, seed)
        _, tags = wall_boxes(ED.load_episode(ED.episode_dirs(ev)[0])["geometry"])
        T = {k: torch.tensor(P[k], device=dev, dtype=torch.float32) for k in ("x", "y", "yaw", "boxes", "cyl")}
        tg = torch.tensor(tags, device=dev)
        esc_n, base_n, ob_d = [], [], []
        for i in range(0, len(P["x"]), 256):
            sl = slice(i, i + 256)
            org, d_w = rays_world(T["x"][sl], T["y"][sl], T["yaw"][sl], *CURRENT, dirs_s)
            t, cls, esc, _, _ = classify(org, d_w, T["boxes"][sl], tg, T["cyl"][sl])
            esc_n.append(esc.sum(1).cpu())
            bf = ((cls == 2) & (t > 0.6) & (t < 2.0)).float().mean(1)        # baffle 0.6-2 m covering 15-50 % of the rays
            base_n.append(torch.where((bf > 0.15) & (bf < 0.5), (cls == 0).sum(1).float() + (cls == 1).sum(1).float() * 0.1,
                                      torch.full_like(bf, -1.0)).cpu())
            cyl = T["cyl"][sl]
            dx, dy = cyl[..., 0] - T["x"][sl, None], cyl[..., 1] - T["y"][sl, None]
            bear = torch.remainder(torch.atan2(dy, dx) - T["yaw"][sl, None] + math.pi, 2 * math.pi) - math.pi
            d = torch.where((cyl[..., 2] > 0) & (bear.abs() < math.radians(8)), torch.hypot(dx, dy), torch.full_like(dx, 1e9))
            ob_d.append(d.amin(1).cpu())
        esc_n, base_n, ob_d = torch.cat(esc_n).numpy(), torch.cat(base_n).numpy(), torch.cat(ob_d).numpy()
        if name == "static":
            frames.append(("F1", name, ev, int(P["ep"][esc_n.argmax()]), int(P["row"][esc_n.argmax()])))
            frames.append(("F2", name, ev, int(P["ep"][base_n.argmax()]), int(P["row"][base_n.argmax()])))
        else:
            ok = np.where((ob_d > 0.8) & (ob_d < 1.5))[0]
            j = ok[np.argmax(esc_n[ok])] if len(ok) else int(np.argmin(np.abs(ob_d - 1.2)))
            frames.append(("F3", name, ev, int(P["ep"][j]), int(P["row"][j])))
    return frames


def frame_state(ev, ep_i, row):
    ep = ED.load_episode(ED.episode_dirs(ev)[ep_i])
    b, tags = wall_boxes(ep["geometry"])
    O = ep["obstacles"]
    cyl = np.zeros((2, 4), np.float32)
    for j in range(len(O["ids"])):
        if O["active"][j]:
            cyl[j] = [O["x"][row, j], O["y"][row, j], 0.17, OBST_H]
    T = ep["traj"]
    return ep, float(T["x"][row]), float(T["y"][row]), float(T["yaw"][row]), b, tags, cyl


def scan(x, y, yaw, b, tags, cyl, z, pitch):
    tt = lambda v: torch.tensor([v], device=dev, dtype=torch.float32)  # noqa: E731
    org, d_w = rays_world(tt(x), tt(y), tt(yaw), z, pitch, dirs_s)
    B, C, G = (torch.tensor(a, device=dev)[None] for a in (b, cyl, tags))
    t, cls, esc, _, _ = classify(org, d_w, B, G[0], C)
    tall = B.clone(); tall[..., 5] = 1e3
    t_cross = raycast_3d(org, d_w, tall, None, floor=False, max_range=RANGE_MAX)     # where the ray passes the wall
    o, d = org[0].cpu().numpy(), d_w[0].cpu().numpy()
    t, cls, esc, tc = t[0].cpu().numpy(), cls[0].cpu().numpy(), esc[0].cpu().numpy(), t_cross[0].cpu().numpy()
    return o, d, t, cls, esc, tc


def draw(fig, gs_row, lab, z, pitch, ep, x, y, yaw, b, cyl, o, d, t, cls, esc, tc, first):
    pts = o + t[:, None] * d
    hit = cls < 4
    ax = fig.add_subplot(gs_row[0])
    for w in ep["geometry"]["walls"]:
        ax.add_patch(Polygon(ED.wall_polygon(w), closed=True, fc="#ffbf80" if w["name"].startswith("Z") else "0.6",
                             ec="k", lw=0.3))
    for c in cyl:
        if c[2] > 0:
            ax.add_patch(Circle(c[:2], c[2], fc="#c5b0d5", ec="k", lw=0.5))
    ax.add_patch(Circle((x, y), 0.177, fc="none", ec="k", lw=0.8))
    ax.plot([x, x + 0.3 * math.cos(yaw)], [y, y + 0.3 * math.sin(yaw)], "k-", lw=1.2)
    for k in (0, 1, 2, 3):
        m = (cls == k) & ~esc
        ax.scatter(pts[m, 0], pts[m, 1], s=0.8, c=COL[k], linewidths=0, label=NAMES[k] if first else None, zorder=3)
    if esc.any():
        cp = o + np.minimum(tc[esc], RANGE_MAX)[:, None] * d[esc]
        ax.scatter(cp[:, 0], cp[:, 1], s=3, c=ESC_C, marker="x", linewidths=0.6, zorder=4,
                   label="escape (where it passes over a wall)" if first else None)
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 3.0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fl, es = 100 * (cls == 0).mean(), 100 * esc.mean()
    ax.set_ylabel(f"{lab}\nz {z:.3f} m, pitch {pitch:g} deg\nfloor {fl:.1f} %  escape {es:.2f} %", fontsize=8)
    if first:
        ax.set_title("top-down", fontsize=9)
    # side view: distance along the heading vs height
    a2 = fig.add_subplot(gs_row[1])
    fwd = (pts[:, 0] - o[0]) * math.cos(yaw) + (pts[:, 1] - o[1]) * math.sin(yaw)
    for k in (0, 1, 2, 3):
        m = (cls == k) & ~esc
        a2.scatter(fwd[m], pts[m, 2], s=1.2, c=COL[k], linewidths=0, zorder=3)
    if esc.any():
        cp = o + np.minimum(tc[esc], RANGE_MAX)[:, None] * d[esc]
        cf = (cp[:, 0] - o[0]) * math.cos(yaw) + (cp[:, 1] - o[1]) * math.sin(yaw)
        a2.scatter(cf, cp[:, 2], s=3, c=ESC_C, marker="x", linewidths=0.6)
    el = np.degrees(np.arcsin(d[:, 2]))
    for e in (el.max(), el.min()):
        L = 9.0
        a2.plot([0, L * math.cos(math.radians(e))], [o[2], o[2] + L * math.sin(math.radians(e))], color="k", lw=0.5, ls=":")
    a2.axhline(0, color="0.3", lw=0.8); a2.axhline(WALL_H, color="0.5", lw=0.6, ls="--")
    a2.axhline(OBST_H, color="#9467bd", lw=0.5, ls=":")
    a2.plot([0], [o[2]], "k^", ms=5)
    a2.set_xlim(-0.2, 8.0); a2.set_ylim(-0.05, 0.85); a2.tick_params(labelsize=6); a2.grid(lw=0.3)
    if first:
        a2.set_title("side view (distance along heading; walls 0.6 m dashed, obstacles 0.5 m dotted, fan edges)", fontsize=8)
    # 3D point cloud
    a3 = fig.add_subplot(gs_row[2], projection="3d")          # robot-centred frame: forward, left, up
    cy_, sy_ = math.cos(yaw), math.sin(yaw)

    def rc(q):
        dx, dy = q[:, 0] - x, q[:, 1] - y
        return cy_ * dx + sy_ * dy, -sy_ * dx + cy_ * dy, q[:, 2]
    for k in (0, 1, 2, 3):
        m = (cls == k) & ~esc
        f_, l_, z_ = rc(pts[m])
        a3.scatter(f_, l_, z_, s=0.4, c=COL[k], linewidths=0, depthshade=False)
    if esc.any():
        f_, l_, z_ = rc(o + np.minimum(tc[esc], RANGE_MAX)[:, None] * d[esc])
        a3.scatter(f_, l_, z_, s=3, c=ESC_C, marker="x", depthshade=False)
    a3.scatter([0], [0], [o[2]], c="k", marker="^", s=14)
    a3.set_xlim(0, 8); a3.set_ylim(-3, 3); a3.set_zlim(0, 1.0)
    a3.view_init(elev=22, azim=-150)                                   # from behind-left of the robot
    a3.set_box_aspect((8, 6, 2.2), zoom=1.45); a3.tick_params(labelsize=5)
    a3.set_xlabel("forward [m]", fontsize=6, labelpad=-8); a3.set_ylabel("left [m]", fontsize=6, labelpad=-8)
    if first:
        a3.set_title("3D point cloud (robot frame, view from behind-left)", fontsize=9)
    # range image
    a4 = fig.add_subplot(gs_row[3])
    img = np.full((34, 139), np.nan)
    img[grow, gcol] = np.where(hit, t, np.nan)
    im = a4.imshow(img, cmap="viridis_r", vmin=0, vmax=RANGE_MAX, aspect="auto", interpolation="nearest",
                   extent=(14, -14, -7, 7))
    cm = np.zeros((34, 139, 4))
    cm[grow[esc], gcol[esc]] = (0.84, 0.15, 0.16, 0.9)
    a4.imshow(cm, aspect="auto", interpolation="nearest", extent=(14, -14, -7, 7))
    a4.tick_params(labelsize=6)
    if first:
        a4.set_title("range image (sensor frame az / el; white = no return, red = escape)", fontsize=8)
    return im


def main():
    frames = pick_frames()
    meta = []
    for fid, name, ev, ep_i, row in frames:
        ep, x, y, yaw, b, tags, cyl = frame_state(ev, ep_i, row)
        meta.append({"frame": fid, "scene": name, "eval": str(ev), "episode": ep_i, "row": row, "x": x, "y": y,
                     "yaw_deg": math.degrees(yaw)})
        n = len(cands)
        fig = plt.figure(figsize=(22, 3.4 * n + 1.0))
        gs = fig.add_gridspec(n, 4, width_ratios=[1, 1.5, 1.4, 1.8], hspace=0.22, wspace=0.12,
                              left=0.06, right=0.94, top=1 - 0.55 / (3.4 * n + 1.0), bottom=0.75 / (3.4 * n + 1.0))
        im = None
        for i, ((z, pitch), lab) in enumerate(zip(cands, args.labels)):
            o, d, t, cls, esc, tc = scan(x, y, yaw, b, tags, cyl, z, pitch)
            im = draw(fig, [gs[i, j] for j in range(4)], lab, z, pitch, ep, x, y, yaw, b, cyl, o, d, t, cls, esc, tc, i == 0)
        fig.legend(loc="lower center", ncol=5, fontsize=10, markerscale=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
        cb = fig.colorbar(im, cax=fig.add_axes([0.948, 0.3, 0.007, 0.4]))
        cb.set_label("range [m]")
        fig.suptitle(f"{fid}: {name} scene, {ED.episode_dir_name(ep_i)} of {ev.name}, step {row} "
                     f"(x {x:.2f}, y {y:.2f}, yaw {math.degrees(yaw):.0f} deg); same pose and obstacles for every mount",
                     fontsize=11, y=1 - 0.15 / (3.4 * n + 1.0))
        out = RESULTS / f"frame_{fid}_{name}_{args.tag}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print(out, flush=True)
    save_json(RESULTS / f"frames_{args.tag}.json", {"frames": meta, "candidates": [
        {"label": l, "z_mount_m": z, "pitch_deg": pt} for l, (z, pt) in zip(args.labels, cands)]})


main()
