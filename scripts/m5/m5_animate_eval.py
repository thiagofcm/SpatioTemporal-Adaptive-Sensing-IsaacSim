"""Top-down MP4 of selected evaluation episodes (from recording.npz + episodes.csv; no Isaac Sim needed).
    python scripts/m5/m5_animate_eval.py <eval_dir> --episodes 0 1 2 25 73 [--speed 2]"""
import argparse, csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon
from st_nav.scenes import corridor_static_random as S

p = argparse.ArgumentParser(); p.add_argument("eval_dir"); p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 25, 73])
p.add_argument("--speed", type=float, default=2.0); p.add_argument("--out", default=None); a = p.parse_args()
rows = list(csv.DictReader(open(os.path.join(a.eval_dir, "episodes.csv"))))
R = np.load(os.path.join(a.eval_dir, "recording.npz")); P, SC, ang, dt = R["pose"], R["scan"], R["beam_angles"], float(R["step_dt"])
eps = a.episodes
nsteps = [int(rows[i]["n_steps"]) for i in eps]
F = max(nsteps) + 1 + int(1.5 / dt)                        # hold the end ~1.5 s

def wall(ax, cx, cy, L, yaw, th, **kw):
    c, s = math.cos(yaw), math.sin(yaw)
    ax.add_patch(Polygon([(cx + c * x - s * y, cy + s * x + c * y) for x, y in
                          [(-L / 2, -th / 2), (L / 2, -th / 2), (L / 2, th / 2), (-L / 2, th / 2)]], **kw))

def zwalls(r):
    L = torch.tensor([[float(r[f"Z{j}_length"]) for j in range(1, 5)]]); Th = torch.tensor([[float(r[f"Z{j}_thickness"]) for j in range(1, 5)]])
    O = torch.tensor([[float(r[k]) for k in ("Z1_x_offset", "Z2_x_offset", "Z3_y_offset", "Z4_y_offset")]])
    return S.z_walls(L, Th, O)[0].numpy()

nc = 3; nr = math.ceil(len(eps) / nc)
fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 5.4 * nr)); axs = np.atleast_1d(axs).ravel()
art = []
for k, i in enumerate(eps):
    ax, r = axs[k], rows[i]
    for w in S.FIXED_WALLS: wall(ax, w[0], w[1], w[2], w[3], S.WALL_THICKNESS, fc="0.45", ec="k", lw=0.3)
    for z in zwalls(r): wall(ax, z[0], z[1], z[2], z[3], z[4], fc="tab:orange", ec="k", lw=0.4)
    g = (float(r["goal_x"]), float(r["goal_y"]))
    ax.add_patch(Circle(g, 0.2, fc="tab:green", alpha=0.2, ec="darkgreen")); ax.plot(*g, "*", color="darkgreen", ms=10)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ok = r["success"] == "True"
    ax.set_title(f"episode {i}  (obstacle seed {r['static_obstacles_seed']}, start {r['start_pose_index']})", fontsize=9)
    rays = LineCollection([], colors="tab:red", linewidths=0.6, alpha=0.55); ax.add_collection(rays)
    trail, = ax.plot([], [], color="tab:blue", lw=1.5)
    body = Circle((0, 0), 0.177, fc="tab:blue", alpha=0.35, ec="navy"); ax.add_patch(body)
    head, = ax.plot([], [], color="navy", lw=2)
    txt = ax.text(-2.9, 2.62, "", fontsize=8.5, family="monospace", va="top",
                  bbox=dict(fc="white", alpha=0.8, ec="none"))
    art.append((i, rays, trail, body, head, txt, ok, r))
for ax in axs[len(eps):]:
    ax.axis("off")
    ax.text(0.02, 0.6, "M5 initial policy (model_200), deterministic\nreference evaluation protocol: goal (-2, -2),\n"
            "obstacles from seed 3000+i, start from seed 4000+i\n\nblue: TurtleBot4 footprint (r 0.177 m) + heading\n"
            "red: 40 LiDAR beams as observed (noisy, 3.5 m)\norange: Z1-Z4 of this episode\n"
            f"playback {a.speed:g}x real time", fontsize=9, va="top", transform=ax.transAxes)
sup = fig.suptitle("", fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.96))

def draw(f):
    for i, rays, trail, body, head, txt, ok, r in art:
        n = int(r["n_steps"]); s = min(f, n)
        x, y, yaw = P[s, i]
        trail.set_data(P[: s + 1, i, 0], P[: s + 1, i, 1]); body.center = (x, y)
        head.set_data([x, x + 0.25 * math.cos(yaw)], [y, y + 0.25 * math.sin(yaw)])
        sc = SC[s, i] if s < n else SC[s - 1, i]
        segs = [[(x, y), (x + d * math.cos(yaw + t), y + d * math.sin(yaw + t))] for d, t in zip(sc, ang) if np.isfinite(d)]
        rays.set_segments(segs)
        v = 0.0 if s == 0 else math.hypot(*(P[s, i, :2] - P[s - 1, i, :2])) / dt
        status = "running" if f < n else ("GOAL" if ok else r["outcome"].upper().replace("_", " "))
        body.set_facecolor("tab:blue" if f < n else ("tab:green" if ok else "tab:red"))
        txt.set_text(f"t {s * dt:5.1f} s   v {v:4.2f} m/s\n{status}")
    sup.set_text(f"STNav-TB4-Corridor-Static-v0  |  t = {min(f, max(nsteps)) * dt:5.1f} s")

out = a.out or os.path.join(a.eval_dir, "eval_episodes_" + "_".join(map(str, eps)) + ".mp4")
w = FFMpegWriter(fps=a.speed / dt, bitrate=3000)
with w.saving(fig, out, dpi=100):
    for f in range(F):
        draw(f); w.grab_frame()
print(out, F, "frames")
