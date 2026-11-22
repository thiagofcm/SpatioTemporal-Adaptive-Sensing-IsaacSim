"""GPU-free rendering of the converted TurtleBot4 (matplotlib + ffmpeg).

Inputs : results/viz/tb4_geometry.npz (export_robot_geometry.py), results/viz/tb4_motion.npz (record_robot_motion.py)
Outputs: results/viz/tb4_model_sheet.png        geometry, collision shapes, frames, LiDAR mount and beam pattern
         results/viz/tb4_motion.mp4             recording of the command program (physics poses from PhysX)

    python infrastructure/robot_viz/scripts/render_robot.py [--workers 32]
"""
import argparse
import math
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

ROOT = os.environ.get("ST_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
VIZ = os.path.join(ROOT, "infrastructure", "robot_viz", "results")
G = dict(np.load(os.path.join(VIZ, "tb4_geometry.npz")))     # fully loaded: worker processes must not share a zip handle
LIDAR_N, LIDAR_RES_DEG, LIDAR_RANGE = 40, 9.0, 3.5


def decimate(v, f, cell):
    """Vertex-clustering decimation (grid cell [m]); returns (v, f) with degenerate faces removed."""
    if cell <= 0:
        return v, f
    q = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    nv = np.zeros((inv.max() + 1, 3))
    np.add.at(nv, inv, v)
    nv /= np.bincount(inv)[:, None]
    nf = inv[f]
    ok = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = np.unique(np.sort(nf[ok], axis=1), axis=0)
    return nv, nf


def part(body, kind, cell):
    k = f"{body}|{kind}"
    if f"{k}|v" not in G:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return decimate(G[f"{k}|v"].astype(np.float64), G[f"{k}|f"], cell)


def tf(T, v):
    return v @ T[:3, :3].T + T[:3, 3]


def quat_to_R(q):  # wxyz
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pose(p, q):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q)
    T[:3, 3] = p
    return T


def add_mesh(ax, v, f, color, alpha, lw=0.0, ec=None):
    if len(f) == 0:
        return
    pc = Poly3DCollection(v[f], facecolor=color, alpha=alpha, linewidths=lw, edgecolor=ec or color)
    ax.add_collection3d(pc)


def shaded(v, f, base_rgb, light=(0.4, -0.5, 0.75)):
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    L = np.asarray(light) / np.linalg.norm(light)
    s = 0.45 + 0.55 * np.abs(n @ L)
    return np.clip(np.asarray(base_rgb)[None] * s[:, None], 0, 1)


def add_shaded(ax, v, f, rgb, alpha=1.0):
    if len(f) == 0:
        return
    pc = Poly3DCollection(v[f], facecolors=shaded(v, f, rgb), linewidths=0.0, alpha=alpha)
    ax.add_collection3d(pc)


def triad(ax, T, L=0.06, lw=2.0, label=None):
    o = T[:3, 3]
    for i, c in enumerate(("r", "g", "b")):
        e = o + L * T[:3, i]
        ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], color=c, lw=lw)
    if label:
        ax.text(*(o + np.array([0, 0, 0.012])), label, fontsize=6.5)


def set_equal(ax, c, r):
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(max(c[2] - r, -0.01), c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def outline_2d(ax, pts2, **kw):
    if len(pts2) < 3:
        return
    h = ConvexHull(pts2)
    loop = np.append(h.vertices, h.vertices[0])
    ax.plot(pts2[loop, 0], pts2[loop, 1], **kw)


# ------------------------------------------------------------------ model sheet --------------------------------------
def model_sheet():
    vis_b = part("base_link", "visual", 0.004)
    col_b = part("base_link", "collision", 0.0)
    wheels = {b: (G[f"rest|{b}"], part(b, "visual", 0.0), part(b, "collision", 0.0)) for b in ("left_wheel", "right_wheel")}
    F = {k.split("|")[1]: G[k] for k in G if k.startswith("frame|")}
    lid = F["rplidar_link"]
    fig = plt.figure(figsize=(20, 11.5))

    # (a) visual model
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    add_shaded(ax, *vis_b, (0.75, 0.75, 0.78))
    for b, (T, vw, _) in wheels.items():
        add_shaded(ax, tf(T, vw[0]), vw[1], (0.15, 0.15, 0.15))
    triad(ax, np.eye(4), 0.08, label="base_link")
    triad(ax, lid, 0.06, label="rplidar_link")
    set_equal(ax, (0, 0, 0.17), 0.2)
    ax.view_init(22, -135)
    ax.set_title(f"(a) visual model (official meshes; {len(G['base_link|visual|f'])} tris, drawn decimated)", fontsize=9)

    # (b) collision geometry
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    add_shaded(ax, *col_b, (0.85, 0.3, 0.3), alpha=0.55)
    for b, (T, _, cw) in wheels.items():
        add_shaded(ax, tf(T, cw[0]), cw[1], (0.2, 0.4, 0.9), alpha=0.9)
    triad(ax, np.eye(4), 0.08)
    set_equal(ax, (0, 0, 0.17), 0.2)
    ax.view_init(22, -135)
    ax.set_title("(b) collision geometry as PhysX sees it (convex hulls, primitives;\n"
                 "blue = sphere wheel colliders r=0.03575 m; small sphere = frictionless caster)", fontsize=9)

    # (c) side view x-z
    ax = fig.add_subplot(2, 3, 3)
    ax.axhline(0, color="saddlebrown", lw=1)
    pts = vis_b[0]
    ax.scatter(pts[:, 0], pts[:, 2], s=0.2, color="0.6")
    cv = col_b[0]
    ax.scatter(cv[:, 0], cv[:, 2], s=0.4, color="tab:red", alpha=0.4)
    for b, (T, vw, cw) in wheels.items():
        c = tf(T, cw[0])
        ax.scatter(c[:, 0], c[:, 2], s=0.4, color="tab:blue")
    ax.axhline(lid[2, 3], color="tab:orange", ls="--", lw=1)
    ax.text(0.2, lid[2, 3] + 0.004, f"LiDAR scan plane z = {lid[2, 3]:.4f} m", color="tab:orange", fontsize=8)
    ax.plot(lid[0, 3], lid[2, 3], "o", color="tab:orange")
    ax.annotate("rplidar_link\n(-0.040, 0, 0.1929)", (lid[0, 3], lid[2, 3]), (-0.27, 0.27), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    for name, mk in (("front_caster_link", "s"), ("wheel_drop_left", "^"), ("shell_link", "d")):
        ax.plot(F[name][0, 3], F[name][2, 3], mk, ms=6, label=name)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.02, 0.4)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("x (forward) [m]")
    ax.set_ylabel("z [m]")
    ax.grid(lw=0.3)
    ax.set_title("(c) side view: visual (grey), collision (red), wheels (blue), frames", fontsize=9)

    # (d) top view: footprint, wheels, caster, LiDAR, beams
    ax = fig.add_subplot(2, 3, 4)
    low = cv[cv[:, 2] < 0.15]
    outline_2d(ax, low[:, :2], color="tab:red", lw=1.5, label="collision footprint (z < 0.15 m)")
    outline_2d(ax, pts[:, :2], color="0.5", lw=1, ls=":", label="visual outline")
    for r, ls in ((0.177, "--"), (0.164, ":")):
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(th), r * np.sin(th), color="k", lw=0.6, ls=ls, label=f"circle r = {r} m")
    for b, (T, _, cw) in wheels.items():
        c = tf(T, cw[0])
        outline_2d(ax, c[:, :2], color="tab:blue", lw=1.5)
        ax.plot(T[0, 3], T[1, 3], "b+", ms=8)
    ax.plot(F["front_caster_link"][0, 3], F["front_caster_link"][1, 3], "ks", ms=5, label="caster")
    lx, ly = lid[0, 3], lid[1, 3]
    yaw_lid = math.atan2(lid[1, 0], lid[0, 0])
    # beams: Isaac Lab LidarPatternCfg h-FOV (-180, 180), 9 deg, in the frame the ray caster is attached to
    angs = np.deg2rad(-180 + LIDAR_RES_DEG * np.arange(LIDAR_N))
    for i, a in enumerate(angs):
        L = 0.42
        ax.plot([lx, lx + L * math.cos(a)], [ly, ly + L * math.sin(a)], color="tab:orange", lw=0.6, alpha=0.8)
        if i % 10 == 0:
            ax.text(lx + 0.45 * math.cos(a), ly + 0.45 * math.sin(a), f"b{i}\n{math.degrees(a):.0f}°", fontsize=6,
                    ha="center", va="center", color="tab:orange")
    ax.arrow(lx, ly, 0.08 * math.cos(yaw_lid), 0.08 * math.sin(yaw_lid), width=0.004, color="tab:green")
    ax.text(lx + 0.02, ly + 0.09, "rplidar_link +x\n(URDF yaw +90°)", fontsize=7, color="tab:green")
    ax.arrow(0, 0, 0.12, 0, width=0.004, color="tab:red")
    ax.text(0.12, -0.03, "base +x (forward)", fontsize=7, color="tab:red")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(lw=0.3)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(d) top view: footprint, wheels, caster, LiDAR origin and the 40-beam 9° pattern\n"
                 "(beams drawn in the base frame, i.e. ray caster aligned with base_link; index 0 at -180°)", fontsize=9)

    # (e) front view y-z
    ax = fig.add_subplot(2, 3, 5)
    ax.axhline(0, color="saddlebrown", lw=1)
    ax.scatter(pts[:, 1], pts[:, 2], s=0.2, color="0.6")
    ax.scatter(cv[:, 1], cv[:, 2], s=0.4, color="tab:red", alpha=0.4)
    for b, (T, vw, cw) in wheels.items():
        c = tf(T, cw[0])
        ax.scatter(c[:, 1], c[:, 2], s=0.4, color="tab:blue")
        vv = tf(T, vw[0])
        ax.scatter(vv[:, 1], vv[:, 2], s=0.2, color="k", alpha=0.3)
    ax.axhline(lid[2, 3], color="tab:orange", ls="--", lw=1)
    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.02, 0.4)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.grid(lw=0.3)
    ax.set_title("(e) front view: wheel visual cylinders (black) vs sphere colliders (blue)", fontsize=9)

    # (f) numbers
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    lines = [
        "Converted asset: assets/turtlebot4_standard/usd/turtlebot4_standard.usd",
        "Rigid bodies: base_link 4.99 kg, left_wheel 0.20 kg, right_wheel 0.20 kg (total 5.39 kg, URDF)",
        "Wheels: r = 0.03575 m, joints at y = +-0.1165 m (track 0.233 m), z = 0.0402 m",
        f"Caster: x = {F['front_caster_link'][0, 3]:.3f} m, sphere r = 0.010 m, frictionless",
        f"LiDAR (rplidar_link): ({lid[0, 3]:+.3f}, {lid[1, 3]:+.3f}, {lid[2, 3]:.4f}) m, yaw {math.degrees(yaw_lid):+.1f}°",
        f"Collision footprint: max radius {np.linalg.norm(low[:, :2], axis=1).max():.3f} m "
        f"(front, bumper); rear base r = 0.164 m",
        f"Collision height: {cv[:, 2].max():.3f} m (tower plate)",
        "",
        "Colours: x axis red, y green, z blue (triads in (a)).",
        "base_link origin is at ground level under the wheel axle (Create3 convention).",
    ]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    fig.suptitle("TurtleBot4 Standard as imported into Isaac Sim (rendered offline from the USD; no RTX available)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "tb4_model_sheet.png"), dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ motion video -------------------------------------
M = dict(np.load(os.path.join(VIZ, "tb4_motion.npz")))
BN = [str(b) for b in M["body_names"]]
VIS = {"base_link": part("base_link", "visual", 0.012), "left_wheel": part("left_wheel", "visual", 0.0),
       "right_wheel": part("right_wheel", "visual", 0.0)}
COL = {"base_link": (0.78, 0.78, 0.82), "left_wheel": (0.12, 0.12, 0.12), "right_wheel": (0.12, 0.12, 0.12)}
SPOKE = np.array([[0, 0, 0], [0.035, 0, 0]])                  # marker on each wheel (wheel body frame) to show rotation
LID = G["frame|rplidar_link"]


def frame(i, path):
    t = M["t"]
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.0, 0.05, 0.58, 0.9], projection="3d")
    Tb = pose(M["pos"][i, 0], M["quat"][i, 0])
    c = M["pos"][i, 0]
    # ground grid
    g = np.arange(np.floor(c[0]) - 1, np.floor(c[0]) + 2.01, 0.25)
    h = np.arange(np.floor(c[1]) - 1, np.floor(c[1]) + 2.01, 0.25)
    for x in g:
        ax.plot([x, x], [h[0], h[-1]], [0, 0], color="0.85", lw=0.5)
    for y in h:
        ax.plot([g[0], g[-1]], [y, y], [0, 0], color="0.85", lw=0.5)
    tr = M["pos"][: i + 1, 0]
    ax.plot(tr[:, 0], tr[:, 1], 0 * tr[:, 0], color="tab:purple", lw=1.2)
    for b in BN:
        j = BN.index(b)
        T = pose(M["pos"][i, j], M["quat"][i, j])
        v, f = VIS[b]
        add_shaded(ax, tf(T, v), f, COL[b])
        if b != "base_link":
            s = tf(T, SPOKE)
            ax.plot(s[:, 0], s[:, 1], s[:, 2], color="yellow", lw=2.5)
    Tl = Tb @ LID
    o = Tl[:3, 3]
    yaw = math.atan2(Tb[1, 0], Tb[0, 0])
    for a in np.deg2rad(-180 + 9.0 * np.arange(40)):
        e = o + 0.35 * np.array([math.cos(yaw + a), math.sin(yaw + a), 0])
        ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], color="tab:orange", lw=0.4, alpha=0.6)
    triad(ax, Tb, 0.12, lw=2)
    set_equal(ax, (c[0], c[1], 0.15), 0.42)
    ax.view_init(24, -60)
    ax.set_axis_off()
    ax.set_title(f"t = {t[i]:5.2f} s   command: {M['label'][i]}", fontsize=13, loc="left")
    # top view
    a2 = fig.add_axes([0.6, 0.55, 0.37, 0.4])
    a2.plot(M["pos"][:, 0, 0], M["pos"][:, 0, 1], color="0.85", lw=1)
    a2.plot(tr[:, 0], tr[:, 1], color="tab:purple", lw=1.5)
    a2.plot(c[0], c[1], "o", color="tab:purple")
    a2.arrow(c[0], c[1], 0.15 * math.cos(yaw), 0.15 * math.sin(yaw), width=0.01, color="tab:red")
    a2.set_aspect("equal")
    a2.grid(lw=0.3)
    a2.set_title("top view (world), base_link trajectory", fontsize=9)
    a2.set_xlabel("x [m]")
    a2.set_ylabel("y [m]")
    # time series
    for k, (lab, unit) in enumerate((("v", "m/s"), ("w", "rad/s"))):
        a3 = fig.add_axes([0.62, 0.08 + 0.21 * (1 - k), 0.35, 0.17])
        a3.step(t, M["cmd"][:, k], where="post", color="k", ls="--", lw=0.8, label="command")
        a3.plot(t[: i + 1], M["twist"][: i + 1, k], color="tab:red", lw=1.2, label="PhysX base twist")
        a3.axvline(t[i], color="0.5", lw=0.6)
        a3.set_xlim(t[0], t[-1])
        a3.set_ylabel(f"{lab} [{unit}]", fontsize=8)
        a3.tick_params(labelsize=7)
        if k == 0:
            a3.legend(fontsize=7, loc="upper right")
        else:
            a3.set_xlabel("t [s]", fontsize=8)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--skip_video", action="store_true")
    a = ap.parse_args()
    model_sheet()
    print("[RENDER] model sheet written", flush=True)
    if not a.skip_video:
        tmp = tempfile.mkdtemp(prefix="tb4frames_", dir=os.environ.get("ST_FRAME_TMP"))
        n = len(M["t"])
        with Pool(a.workers) as pool:
            pool.starmap(frame, [(i, os.path.join(tmp, f"f{i:05d}.png")) for i in range(n)])
        fps = 1.0 / np.median(np.diff(M["t"]))
        ff = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")
        out = os.path.join(VIZ, "tb4_motion.mp4")
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i", os.path.join(tmp, "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], check=True)
        shutil.copy(os.path.join(tmp, f"f{n // 2:05d}.png"), os.path.join(VIZ, "tb4_motion_frame_mid.png"))
        shutil.rmtree(tmp)
        print(f"[RENDER] video {out} ({n} frames, {fps:.1f} fps)", flush=True)
