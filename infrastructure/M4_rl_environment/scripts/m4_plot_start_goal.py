"""Start poses (fixed + start-pose DR) and goal positions used by STNav-TB4-Corridor-Static-v0, on the corridor map."""
import math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Polygon, Wedge
from st_nav.scenes import corridor_static_random as S


class cfg:  # CorridorNavEnvCfg.goal_tolerance (cfg module needs Kit/pxr)
    goal_tolerance = 0.20
OUT = os.path.join(os.environ.get("ST_ROOT", "."), "infrastructure", "M4_rl_environment", "results", "figures", "m4_start_goal_poses.png")
L, T, O = (torch.tensor([v]) for v in S.NOMINAL)
zw = S.z_walls(L, T, O)[0].numpy()

def wall(ax, cx, cy, length, yaw, th, **kw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(-length / 2, -th / 2), (length / 2, -th / 2), (length / 2, th / 2), (-length / 2, th / 2)]
    ax.add_patch(Polygon([(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts], **kw))

def robot(ax, x, y, yaw, col, alpha=0.25, label=None):
    ax.add_patch(Circle((x, y), 0.177, fc=col, alpha=alpha, ec=col, lw=1))
    ax.annotate("", (x + 0.3 * math.cos(yaw), y + 0.3 * math.sin(yaw)), (x, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
    if label: ax.plot([], [], "o", color=col, alpha=0.6, label=label)

fig, axes = plt.subplots(1, 3, figsize=(22, 7.8), gridspec_kw={"width_ratios": [1.15, 1, 1]})
ax = axes[0]
for w in S.FIXED_WALLS: wall(ax, w[0], w[1], w[2], w[3], S.WALL_THICKNESS, fc="0.45", ec="k", lw=0.4)
for i, w in enumerate(zw):
    wall(ax, w[0], w[1], w[2], w[3], w[4], fc="tab:orange", ec="k", lw=0.4, alpha=0.8)
    ax.text(w[0], w[1], f"Z{i+1}", ha="center", va="center", fontsize=7)
sx, sy = S.STARTING_POS
robot(ax, sx, sy, S.STARTING_YAW, "tab:blue", 0.35, f"default start ({sx}, {sy}), yaw {math.degrees(S.STARTING_YAW):.1f} deg")
for x, y in S.START_POSE_LIST: ax.plot(x, y, "s", color="tab:purple", ms=4)
for (gx, gy) in S.GOAL_LIST_ST01:
    ax.add_patch(Circle((gx, gy), cfg.goal_tolerance, fc="tab:green", alpha=0.12, ec="tab:green", lw=0.6))
    ax.plot(gx, gy, "*", color="darkgreen", ms=9)
ax.plot([], [], "s", color="tab:purple", label="start-pose DR points (5)")
ax.plot([], [], "*", color="darkgreen", label=f"goal list (9), tolerance {cfg.goal_tolerance} m")
ax.set_xlim(-3.05, 3.05); ax.set_ylim(-3.05, 3.05); ax.set_aspect("equal"); ax.grid(lw=0.3)
ax.legend(fontsize=8, loc="center right")
ax.set_title("Corridor map (nominal Z1-Z4 shown; Z sizes/offsets are randomized)", fontsize=10)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

ax = axes[1]
# start region zoom
ax.add_patch(Wedge((2.4, 2.4), 0.55, math.degrees(S.START_YAW_CENTER - S.START_YAW_NOISE), math.degrees(S.START_YAW_CENTER + S.START_YAW_NOISE),
                   fc="tab:purple", alpha=0.12, ec="tab:purple"))
for i, (x, y) in enumerate(S.START_POSE_LIST):
    for dy in (-S.START_YAW_NOISE, 0.0, S.START_YAW_NOISE):
        a = S.START_YAW_CENTER + dy
        ax.annotate("", (x + 0.12 * math.cos(a), y + 0.12 * math.sin(a)), (x, y),
                    arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=0.8, alpha=0.7 if dy else 1))
    ax.plot(x, y, "s", color="tab:purple", ms=6)
    ax.text(x + 0.02, y + 0.03, f"({x:.2f}, {y:.2f})", fontsize=7.5)
robot(ax, sx, sy, S.STARTING_YAW, "tab:blue", 0.15)
ax.annotate("", (sx + 0.25 * math.cos(S.STARTING_YAW), sy + 0.25 * math.sin(S.STARTING_YAW)), (sx, sy),
            arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
for w in S.FIXED_WALLS: wall(ax, w[0], w[1], w[2], w[3], S.WALL_THICKNESS, fc="0.45", ec="k", lw=0.4)
ax.set_xlim(1.85, 3.0); ax.set_ylim(1.85, 3.0); ax.set_aspect("equal"); ax.grid(lw=0.3)
ax.set_title("Start region (zoom)\nblue: default start (2.50, 2.50), yaw -168.69 deg [used by default]\n"
             "purple: start-pose DR (off by default): 5 points, yaw 180 +- 10 deg (arrows: -10, 0, +10)", fontsize=9)
ins = axes[2]
for w in S.FIXED_WALLS: wall(ins, w[0], w[1], w[2], w[3], S.WALL_THICKNESS, fc="0.45", ec="k", lw=0.3)
for (gx, gy) in S.GOAL_LIST_ST01:
    ins.add_patch(Circle((gx, gy), cfg.goal_tolerance, fc="tab:green", alpha=0.15, ec="tab:green", lw=0.5))
    ins.plot(gx, gy, "*", color="darkgreen", ms=10); ins.text(gx + 0.015, gy + 0.02, f"({gx:.2f},{gy:.2f})", fontsize=8)
ins.set_xlim(-2.45, -1.35); ins.set_ylim(-2.85, -1.85); ins.set_aspect("equal"); ins.set_title("Goal region (zoom): 9 goals, one drawn uniformly per reset\ncircles: 0.20 m success tolerance", fontsize=9); ins.grid(lw=0.3)

fig.suptitle("STNav-TB4-Corridor-Static-v0: start poses and goals (reference ST01-03 settings)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT, dpi=120); print(OUT)
