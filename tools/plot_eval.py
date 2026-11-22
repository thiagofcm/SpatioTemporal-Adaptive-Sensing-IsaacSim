"""Plot an M5 evaluation (episodes.csv + trajectories.npy). python tools/plot_eval.py <eval_dir>"""
import csv, math, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Polygon
from st_nav.scenes import corridor_static_random as S
E = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(E, "episodes.csv"))))
T = np.load(os.path.join(E, "trajectories.npy"))            # (steps, n, 2)

def wall(ax, cx, cy, L, yaw, th, **kw):
    c, s = math.cos(yaw), math.sin(yaw)
    ax.add_patch(Polygon([(cx + c * x - s * y, cy + s * x + c * y) for x, y in
                          [(-L / 2, -th / 2), (L / 2, -th / 2), (L / 2, th / 2), (-L / 2, th / 2)]], **kw))

def zwalls(r):
    L = torch.tensor([[float(r[f"Z{j}_length"]) for j in range(1, 5)]]); Th = torch.tensor([[float(r[f"Z{j}_thickness"]) for j in range(1, 5)]])
    O = torch.tensor([[float(r[k]) for k in ("Z1_x_offset", "Z2_x_offset", "Z3_y_offset", "Z4_y_offset")]])
    return S.z_walls(L, Th, O)[0].numpy()

def base(ax):
    for w in S.FIXED_WALLS: wall(ax, w[0], w[1], w[2], w[3], S.WALL_THICKNESS, fc="0.45", ec="k", lw=0.3)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.grid(lw=0.3)

fail = [int(r["episode"]) for r in rows if r["success"] != "True"]
fig, ax = plt.subplots(1, 1 + max(1, len(fail)), figsize=(7 * (1 + max(1, len(fail))), 7))
base(ax[0])
for r in rows:
    i = int(r["episode"])
    for z in zwalls(r): wall(ax[0], z[0], z[1], z[2], z[3], z[4], fc="tab:orange", alpha=0.03, ec="none")
for r in rows:
    i = int(r["episode"]); ok = r["success"] == "True"
    ax[0].plot(T[:, i, 0], T[:, i, 1], color="tab:green" if ok else "tab:red", lw=0.6 if ok else 2.0, alpha=0.5 if ok else 1)
g = (float(rows[0]["goal_x"]), float(rows[0]["goal_y"]))
for a in ax: a.add_patch(Circle(g, 0.2, fc="none", ec="darkgreen", lw=1.5))
n, s = len(rows), sum(r["success"] == "True" for r in rows)
ax[0].set_title(f"{n} evaluation episodes (seeds 3000+i / 4000+i), goal {g}: {s}/{n} success\n"
                f"green = success, red = failure; orange shading = Z1-Z4 of all episodes", fontsize=9)
for k, i in enumerate(fail):
    a = ax[1 + k]; r = rows[i]; base(a)
    for z in zwalls(r): wall(a, z[0], z[1], z[2], z[3], z[4], fc="tab:orange", ec="k", lw=0.4)
    a.plot(T[:, i, 0], T[:, i, 1], color="tab:red", lw=2)
    last = T[~np.isnan(T[:, i, 0]), i][-1]
    a.add_patch(Circle(last, 0.177, fc="tab:red", alpha=0.3))
    a.set_title(f"episode {i} ({r['outcome']}, step {r['n_steps']}): start {r['start_pose_index']}, "
                f"modes {r['z12_mode']}/{r['z34_mode']}\nmin clearance {r['min_clearance_m']} m", fontsize=9)
    a.set_xlim(-1.0, 3.0); a.set_ylim(-1.0, 3.0)
fig.tight_layout(); out = os.path.join(E, "eval_trajectories.png"); fig.savefig(out, dpi=110); print(out)
