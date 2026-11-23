"""Mount study, part 2 renders: mounts with an x offset and a narrower vertical FoV (between the decks) compared with
the original M7 mount, R4 and R1, on the same recorded NAV03 frame (offline rendering, same method as visual_comparison.py).
The Mirrorcle enclosure (85 x 84 x 73 mm) is drawn to scale, centred on the optical centre (assumed).

    isaacrun -g <gpu> -- python infrastructure/M7_lidar3d/mount_study/scripts/visual_comparison_xzfov.py
Outputs: results/visual_comparison_xzfov/{1_mounting,2_scene,3_pov}.png, 4_animation.mp4, selection.json
"""
import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import visual_comparison as VC  # noqa: E402
from mount_common import BODY_PITCH_DEG, EVAL_DYNAMIC, OBST_H, RANGE_MAX, RESULTS, WALL_H, classify  # noqa: E402
from mount_sweep_xzfov import HOUSING, ORIG_2D, dirs_for, rays_world_x  # noqa: E402
from st_nav.geometry.analytic3d import raycast_3d  # noqa: E402
import episode_data as ED  # noqa: E402

OUT = RESULTS / "visual_comparison_xzfov"
DEFAULT = ("original M7: 0, 0.40, 5, 14 | 0 0.3998 5 14; R4: 0, 0.40, 6.75, 14 | 0 0.40 6.75 14; "
           "R1: 0, 0.50, 7.5, 14 | 0 0.50 7.5 14; B10: 0.086, 0.298, 4, 10 | 0.086 0.298 4.0 10; "
           "B8: 0.086, 0.298, 3.25, 8 | 0.086 0.298 3.25 8")
PALETTE = ["#7f7f7f", "#d6278f", "#2ca02c", "#ff7f0e", "#1f77b4"]
dev = "cuda"


def parse(s):
    out = {}
    for item in s.split(";"):
        lab, vals = item.split("|")
        out[lab.strip()] = tuple(float(v) for v in vals.split())
    return out


def sensor_world(T, x, z, pitch):
    R = T[:3, :3] @ VC.Ry(pitch)
    return T[:3, :3] @ np.array([x, 0, z]) + T[:3, 3], R


def corner_dirs(vhalf):
    d = []
    for az, el in ((14, vhalf), (-14, vhalf), (-14, -vhalf), (14, -vhalf)):
        a, e = math.radians(az), math.radians(el)
        d.append([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return np.array(d)


def draw_housing(ax, o, R, col, vfov, ray_len):
    hx, hy, hz = (h / 2 for h in HOUSING)
    c = np.array([[sx, sy, sz] for sx in (-hx, hx) for sy in (-hy, hy) for sz in (-hz, hz)])
    c = o + c @ R.T
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([c[f] for f in faces], facecolor=col, edgecolor="k", lw=0.4, alpha=0.45))
    ax.scatter(*o, color="k", s=10, depthshade=False)
    ax.plot(*zip(o, o + R @ np.array([0.2, 0, 0])), color="k", lw=1.6)
    if ray_len > 0:
        ends = o + ray_len * (corner_dirs(vfov / 2) @ R.T)
        for e in ends:
            ax.plot(*zip(o, e), color=col, lw=0.8)
        ax.add_collection3d(Poly3DCollection([ends], facecolor=col, alpha=0.10, edgecolor=col, lw=0.6))


def scans(T, b, tags, cyl, cfg, rows):
    x, z, pitch, vfov = cfg
    dirs, _, _ = dirs_for(vfov, dev)
    tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)  # noqa: E731
    X, Y, YAW = tt(T["x"][rows]), tt(T["y"][rows]), tt(T["yaw"][rows])
    P = X.shape[0]
    B = tt(b)[None].expand(P, -1, -1).contiguous()
    C = tt(cyl[rows])
    org, d_w = rays_world_x(X, Y, YAW, (x, 0.0, z), pitch, dirs, BODY_PITCH_DEG)
    t, cls, esc, _, _ = classify(org, d_w, B, torch.tensor(tags, device=dev), C)
    tall = B.clone(); tall[..., 5] = 1e3
    cross = raycast_3d(org, d_w, tall, None, floor=False, max_range=RANGE_MAX)
    hits = [(cls == 3) & ((raycast_3d(org, d_w, None, C[:, j:j + 1], floor=False, max_range=RANGE_MAX) - t).abs() < 1e-5)
            for j in range(C.shape[1])]
    cls = torch.where(esc, torch.full_like(cls, 5), cls)
    f = lambda a: a.cpu().numpy()  # noqa: E731
    return {"origin": f(org), "dirs": f(d_w), "t": f(t), "cls": f(cls), "cross": f(cross), "ob_hit": f(torch.stack(hits, 1))}


def figure_mounting(cfgs, cols):
    parts = VC.robot_mesh(0.006)
    views = [("side (from the robot's right)", 0, -90), ("front", 0, 0), ("top", 90, -90), ("isometric", 22, -55)]
    n = len(cfgs)
    fig = plt.figure(figsize=(22, 5.0 * n + 0.6))
    for i, (lab, cfg) in enumerate(cfgs.items()):
        x, z, pitch, vfov = cfg
        o, R = sensor_world(np.eye(4), x, z, pitch)
        for j, (vname, elev, azim) in enumerate(views):
            ax = fig.add_subplot(n, 4, 4 * i + j + 1, projection="3d")
            ax.set_proj_type("ortho")
            VC.draw_robot(ax, parts)
            draw_housing(ax, o, R, cols[lab], vfov, 0.6)
            if j == 0:
                ax.scatter(*ORIG_2D, color="r", marker="x", s=30, depthshade=False)
                for zz, ls, txt in ((0.3498, "--", "top platform 0.350"), (0.3443, ":", "plate underside 0.344"),
                                    (ORIG_2D[2], "-.", "original 2D LiDAR 0.193 (red x)")):
                    ax.plot([-0.2, 0.6], [-0.25, -0.25], [zz, zz], color="k", ls=ls, lw=0.6)
                    ax.text(0.33, -0.25, zz + 0.006, txt, fontsize=6)
            ax.set_xlim(-0.25, 0.6); ax.set_ylim(-0.30, 0.30); ax.set_zlim(0.0, 0.62)
            ax.set_box_aspect((0.85, 0.60, 0.62))
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=5)
            if j == 0:
                ax.set_title(f"{lab}\noptical centre ({x:.3f}, 0, {z:.3f}) m, pitch {pitch:g} deg down, vertical FoV {vfov:g} deg; "
                             f"{'between the decks' if z + HOUSING[2] / 2 <= 0.3443 else 'on the top platform'}", fontsize=8.5, loc="left")
            else:
                ax.set_title(vname, fontsize=8.5)
    fig.suptitle("Mounts on the TurtleBot4 Standard (Isaac Sim meshes). Mirrorcle enclosure 85 x 84 x 73 mm to scale, "
                 "centred on the optical centre (assumed); black dot = optical centre, black line = boresight, "
                 "coloured frustum = FoV to 0.6 m", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (5.0 * n + 0.6)))
    fig.savefig(OUT / "1_mounting.png", dpi=90)
    plt.close(fig)


def figure_scene(ep, row, cfgs, cols, S):
    b, tags, cyl, T = VC.load_episode_state(ep)
    x0, y0, yaw = float(T["x"][row]), float(T["y"][row]), float(T["yaw"][row])
    parts = VC.robot_mesh(0.012)
    Tr = VC.robot_T(x0, y0, yaw)
    n = len(cfgs)
    fig = plt.figure(figsize=(7.2 * n, 16))
    gs = fig.add_gridspec(3, n, height_ratios=[1.3, 1, 0.8])
    for i, (lab, cfg) in enumerate(cfgs.items()):
        x, z, pitch, vfov = cfg
        s = S[lab]
        o, R = sensor_world(Tr, x, z, pitch)
        nob = s["ob_hit"][0].sum(-1)
        fr = ", ".join(f"{VC.CLS_NAME[c].split(' (')[0]} {100 * float((s['cls'][0] == c).mean()):.1f} %" for c in (0, 1, 3, 5))
        ax = fig.add_subplot(gs[0, i], projection="3d")
        ax.add_collection3d(Poly3DCollection([f for bb in b for f in VC.box_faces(bb)], facecolor=(0.75, 0.75, 0.75),
                                             edgecolor=(0.4, 0.4, 0.4), lw=0.2, alpha=0.15))
        for c in cyl[row]:
            if c[2] > 0:
                ax.add_collection3d(Poly3DCollection(VC.cyl_faces(c), facecolor="#c5b0d5", edgecolor="none", alpha=0.3))
        VC.draw_robot(ax, parts, Tr)
        draw_housing(ax, o, R, cols[lab], vfov, 3.0)
        VC.scatter_classes(ax, s, 0, s=1.0, label=(i == 0))
        c2 = np.array([x0, y0]) + 1.6 * np.array([math.cos(yaw), math.sin(yaw)])
        ax.set_xlim(c2[0] - 2.2, c2[0] + 2.2); ax.set_ylim(c2[1] - 2.2, c2[1] + 2.2); ax.set_zlim(0, 1.0)
        ax.set_box_aspect((4.4, 4.4, 1.0), zoom=1.25)
        ax.view_init(elev=24, azim=math.degrees(yaw) + 180 - 35)
        ax.tick_params(labelsize=5)
        ax.set_title(f"{lab}\nrays on OB1 / OB2: {nob[0]} / {nob[1]}\n{fr}", fontsize=9)
        a2 = fig.add_subplot(gs[1, i])
        for w in ep["geometry"]["walls"]:
            a2.add_patch(Polygon(ED.wall_polygon(w), closed=True, fc="0.6", ec="k", lw=0.3))
        for c in cyl[row]:
            if c[2] > 0:
                a2.add_patch(Circle(c[:2], c[2], fc="#c5b0d5", ec="k", lw=0.5))
        a2.add_patch(Circle((x0, y0), 0.177, fc="none", ec="k"))
        for sg in (-1, 1):
            a_ = yaw + sg * math.radians(14)
            a2.plot([o[0], o[0] + 6 * math.cos(a_)], [o[1], o[1] + 6 * math.sin(a_)], color=cols[lab], lw=0.8)
        VC.scatter_classes(a2, s, 0, three_d=False, s=1.2)
        a2.set_xlim(-3, 3); a2.set_ylim(-3, 3); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
        a2.set_title("top-down", fontsize=9)
        a3 = fig.add_subplot(gs[2, i])
        pts = s["origin"][0] + s["t"][0][:, None] * s["dirs"][0]
        fw = (pts[:, 0] - x0) * math.cos(yaw) + (pts[:, 1] - y0) * math.sin(yaw)       # from the robot centre
        for c in (0, 1, 2, 3):
            m = s["cls"][0] == c
            a3.scatter(fw[m], pts[m, 2], s=1.0, c=VC.CLS_COL[c], linewidths=0)
        el = np.degrees(np.arcsin(s["dirs"][0][:, 2]))
        of = (o[0] - x0) * math.cos(yaw) + (o[1] - y0) * math.sin(yaw)
        for e in (el.max(), el.min()):
            a3.plot([of, of + 8 * math.cos(math.radians(e))], [o[2], o[2] + 8 * math.sin(math.radians(e))], color=cols[lab], lw=0.8)
        a3.axhline(0, color="0.3", lw=0.8); a3.axhline(WALL_H, color="0.5", ls="--", lw=0.7); a3.axhline(OBST_H, color="#9467bd", ls=":", lw=0.7)
        a3.axvline(0.177, color="k", lw=0.5, ls=":")
        a3.plot([of], [o[2]], "^", color=cols[lab], mec="k")
        a3.set_xlim(-0.2, 7); a3.set_ylim(-0.05, 0.8); a3.grid(lw=0.3); a3.tick_params(labelsize=6)
        a3.set_xlabel("distance from the robot centre along the heading [m] (dotted: robot front)", fontsize=7)
        a3.set_title("side view (walls 0.6 m dashed, obstacles 0.5 m dotted, fan edges)", fontsize=8)
    fig.legend(loc="lower center", ncol=5, fontsize=11, markerscale=5, frameon=False)
    fig.suptitle(f"NAV03 (ref_chase_c10, model_500) {ED.episode_dir_name(ep['meta']['episode_index'])}, step {row} "
                 f"(t = {0.2 * row:.1f} s): same robot pose and obstacle positions for every mount", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT / "2_scene.png", dpi=80)
    plt.close(fig)


def figure_pov(ep, row, cfgs, cols, S):
    b, tags, cyl, T = VC.load_episode_state(ep)
    Tr = VC.robot_T(float(T["x"][row]), float(T["y"][row]), float(T["yaw"][row]))
    n = len(cfgs)
    fig = plt.figure(figsize=(22, 4.3 * n + 0.8))
    gs = fig.add_gridspec(n, 3, width_ratios=[1.45, 1.25, 1.25])
    for i, (lab, cfg) in enumerate(cfgs.items()):
        x, z, pitch, vfov = cfg
        s = S[lab]
        o, R = sensor_world(Tr, x, z, pitch)
        ax = fig.add_subplot(gs[i, 0])
        VC.draw_pov(ax, b, cyl[row], o, R, s, 0, f"{lab}: first-person view from the optical centre (dashed: LiDAR FoV)")
        for p_ in ax.patches[-1:]:
            p_.set_xy((-14, -vfov / 2)); p_.set_height(vfov)              # FoV rectangle for this vertical FoV
        a2 = fig.add_subplot(gs[i, 1])
        img = VC.range_img(s, 0)
        im = a2.imshow(img, cmap="viridis_r", vmin=0, vmax=RANGE_MAX, aspect="auto", interpolation="nearest",
                       extent=(14, -14, -vfov / 2, vfov / 2))
        a2.set_ylim(-7, 7); a2.set_title(f"range image, vertical FoV {vfov:g} deg (34 rows; white = no return)", fontsize=9)
        a2.tick_params(labelsize=6)
        fig.colorbar(im, ax=a2, fraction=0.04, pad=0.01).set_label("range [m]", fontsize=8)
        a3 = fig.add_subplot(gs[i, 2])
        a3.imshow(VC.class_image(s, 0), aspect="auto", interpolation="nearest", extent=(14, -14, -vfov / 2, vfov / 2))
        a3.set_ylim(-7, 7); a3.set_facecolor("#eeeeee")
        fr = ", ".join(f"{VC.CLS_NAME[c].split(' (')[0]} {100 * float((s['cls'][0] == c).mean()):.1f} %" for c in (0, 1, 3, 5))
        a3.set_title(f"ray classes: {fr}", fontsize=8.5); a3.tick_params(labelsize=6)
    for c in (0, 1, 3, 5):
        fig.axes[0].scatter([], [], c=VC.CLS_COL[c], s=25, marker="x" if c == 5 else "o", label=VC.CLS_NAME[c])
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(f"LiDAR point of view, NAV03 {ED.episode_dir_name(ep['meta']['episode_index'])} step {row} "
                 "(range / class images drawn on a common +-7 deg elevation axis; grey = outside the vertical FoV)", fontsize=12)
    fig.tight_layout(rect=(0, 0.6 / (4.3 * n + 0.8), 1, 1 - 0.4 / (4.3 * n + 0.8)))
    fig.savefig(OUT / "3_pov.png", dpi=90)
    plt.close(fig)


# =================================================================================================== 4. animation
_A = {}


def anim_frame(k, path):
    ep, b, cyl, T, S, cfgs, cols, held = (_A[n] for n in ("ep", "b", "cyl", "T", "S", "cfgs", "cols", "held"))
    x0, y0, yaw = float(T["x"][k]), float(T["y"][k]), float(T["yaw"][k])
    s_row = held[k]                                                    # scan in the observation (4 Hz, held)
    n = len(cfgs)
    fig = plt.figure(figsize=(5.6 * n, 9.6))
    gs = fig.add_gridspec(2, n, height_ratios=[0.75, 1])
    Ts = VC.robot_T(float(T["x"][s_row]), float(T["y"][s_row]), float(T["yaw"][s_row]))
    for i, (lab, cfg) in enumerate(cfgs.items()):
        x, z, pitch, vfov = cfg
        sc = S[lab]
        o, R = sensor_world(Ts, x, z, pitch)
        ax = fig.add_subplot(gs[0, i])
        VC.draw_pov(ax, b, cyl[s_row], o, R, sc, s_row, f"{lab}: LiDAR point of view", point_size=4)
        ax.patches[-1].set_xy((-14, -vfov / 2)); ax.patches[-1].set_height(vfov)
        a2 = fig.add_subplot(gs[1, i])
        for w in ep["geometry"]["walls"]:
            a2.add_patch(Polygon(ED.wall_polygon(w), closed=True, fc="0.6", ec="k", lw=0.3))
        for c in cyl[k]:
            if c[2] > 0:
                a2.add_patch(Circle(c[:2], c[2], fc="#c5b0d5", ec="k", lw=0.5))
        a2.plot(T["x"][:k + 1], T["y"][:k + 1], color="tab:blue", lw=1)
        a2.add_patch(Circle((x0, y0), 0.177, fc=cols[lab], alpha=0.35, ec="k"))
        a2.plot([x0, x0 + 0.25 * math.cos(yaw)], [y0, y0 + 0.25 * math.sin(yaw)], color="k", lw=2)
        VC.scatter_classes(a2, sc, s_row, three_d=False, s=1.0)
        a2.set_xlim(-3, 3); a2.set_ylim(-3, 3); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
        nob = sc["ob_hit"][s_row].sum(-1)
        fl = 100 * float((sc["cls"][s_row] == 0).mean())
        es = 100 * float((sc["cls"][s_row] == 5).mean())
        a2.set_title(f"rays on OB1 / OB2: {nob[0]} / {nob[1]}\nfloor {fl:.1f} %, escaping {es:.2f} %", fontsize=9)
    for c in (0, 1, 3, 5):
        fig.axes[0].scatter([], [], c=VC.CLS_COL[c], s=25, marker="x" if c == 5 else "o", label=VC.CLS_NAME[c])
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(f"NAV03 {ED.episode_dir_name(ep['meta']['episode_index'])}: t = {0.2 * k:5.1f} s (scan acquired at "
                 f"{0.2 * s_row:4.1f} s, 4 Hz, held between scans)  |  same recorded trajectory and obstacles for every mount",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=70)
    plt.close(fig)


def render_animation(ep, b, cyl, T, Sall, cfgs, cols, workers, sel_row):
    import os
    import shutil
    import subprocess
    import tempfile
    from multiprocessing import get_context
    n = len(T["x"])
    _A.update(ep=ep, b=b, cyl=cyl, T=T, S=Sall, cfgs=cfgs, cols=cols, held=VC.acquisition_rows(n))
    tmp = tempfile.mkdtemp(prefix="lidar_cmp2_")
    with get_context("fork").Pool(workers) as pool:
        pool.starmap(anim_frame, [(k, os.path.join(tmp, f"f{k:05d}.png")) for k in range(n)])
    ff = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", os.path.join(tmp, "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(OUT / "4_animation.mp4")], check=True)
    shutil.copy(os.path.join(tmp, f"f{sel_row:05d}.png"), OUT / "4_animation_frame_selected.png")
    shutil.rmtree(tmp)
    print(f"4_animation.mp4 ({n} frames at 10 fps = 2x real time)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default=DEFAULT, help='"label | x z pitch vfov; ..."')
    p.add_argument("--frame", type=int, nargs=2, default=(56, 15), help="NAV03 episode and step (both obstacles in view)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--workers", type=int, default=48)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = parse(a.configs)
    cols = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(cfgs)}
    ep = ED.load_episode(ED.episode_dirs(EVAL_DYNAMIC)[a.frame[0]])
    row = a.frame[1]
    b, tags, cyl, T = VC.load_episode_state(ep)
    S = {lab: scans(T, b, tags, cyl, cfg, np.array([row])) for lab, cfg in cfgs.items()}
    Sall = {lab: scans(T, b, tags, cyl, cfg, np.arange(len(T["x"]))) for lab, cfg in cfgs.items()}
    sel = {"eval": str(EVAL_DYNAMIC), "episode": a.frame[0], "row": row,
           "configs": {lab: dict(zip(("x_m", "z_m", "pitch_deg", "vfov_deg"), cfg)) for lab, cfg in cfgs.items()},
           "frame": {lab: {"rays_on_OB1_OB2": s["ob_hit"][0].sum(-1).tolist(),
                           **{VC.CLS_NAME[c].split(" (")[0] + "_pct": 100 * float((s["cls"][0] == c).mean()) for c in (0, 1, 3, 5)}}
                     for lab, s in S.items()},
           "episode_means": {lab: {VC.CLS_NAME[c].split(" (")[0] + "_pct": 100 * float((s["cls"] == c).mean()) for c in (0, 1, 3, 5)}
                             for lab, s in Sall.items()}}
    with open(OUT / "selection.json", "w") as f:
        json.dump(sel, f, indent=1)
    print(json.dumps(sel["frame"], indent=0), flush=True)
    figure_mounting(cfgs, cols)
    print("1_mounting.png", flush=True)
    figure_scene(ep, row, cfgs, cols, S)
    print("2_scene.png", flush=True)
    figure_pov(ep, row, cfgs, cols, S)
    print("3_pov.png", flush=True)
    if not a.no_video:
        render_animation(ep, b, cyl, T, Sall, cfgs, cols, a.workers, row)


if __name__ == "__main__":
    main()
