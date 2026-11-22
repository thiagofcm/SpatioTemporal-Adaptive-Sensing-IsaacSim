"""Top-down plots of the reference scenes, drawn directly from their geometric definitions.

Reads (read-only) ~/adaptive_sensor_policy/envs/scenes and the env constants. Run with bytecode writing disabled so
nothing is written into the reference repository:

    python -I -B plot_reference_scenes.py <adaptive_sensor_policy root> <out_dir>
"""
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

ROOT, OUT = sys.argv[1], sys.argv[2]
sys.dont_write_bytecode = True
sys.path.insert(0, ROOT)
from envs.scenes import SCENE_MODULES  # noqa: E402
from envs.scenes import _common as C  # noqa: E402
from envs.scenes import _obstacle_motion as M  # noqa: E402
from envs.scenes import corridor_dynamic_chase_random as CDR  # noqa: E402
from envs.scenes import corridor_static_object_random as SOR  # noqa: E402
from envs.scenes import stage9 as S9  # noqa: E402

# Env constants (copied from envs/turtlebot_var_scan_rate.py to avoid importing torch/gym; verified 2026-11-21)
STARTING_POS, STARTING_YAW = (2.5, 2.5), -2.9441970937399127
START_POSE_LIST = ((2.50, 2.50), (2.30, 2.50), (2.50, 2.30), (2.30, 2.30), (2.40, 2.40))
GOAL_LIST = [(-1.8, -2.5)]
GOALS_STATIC = [(-1.80, -2.50), (-1.50, -2.50), (-2.10, -2.50), (-1.80, -2.20), (-1.60, -2.30), (-2.00, -2.30),
                (-1.50, -2.20), (-2.10, -2.20), (-1.80, -2.00)]          # ST01-ST03 configs
GOALS_CHASE = [(-1.80, -1.70), (-2.00, -1.80), (-2.00, -2.00), (-2.20, -1.70), (-1.80, -2.10), (-2.20, -2.10)]  # ST04-07
ROBOT_R, COLLISION_THR, GOAL_THR, LIDAR_CAP = 0.17, 0.22, 0.20, 3.5


def wall_corners(w):
    cx, cy, length, yaw = w[:4]
    th = w[4] if len(w) > 4 else C.WALL_THICKNESS
    c, s = math.cos(yaw), math.sin(yaw)
    loc = [(-length / 2, -th / 2), (length / 2, -th / 2), (length / 2, th / 2), (-length / 2, th / 2)]
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in loc]


def draw_walls(ax, walls, **kw):
    style = dict(fc="0.35", ec="k", lw=0.5, zorder=2)
    style.update(kw)
    for w in walls:
        ax.add_patch(Polygon(wall_corners(w), closed=True, **style))


def draw_start_goal(ax, goals, start_dr=False):
    x, y = STARTING_POS
    ax.add_patch(Circle((x, y), ROBOT_R, fc="tab:blue", alpha=0.35, ec="tab:blue", zorder=5))
    ax.add_patch(FancyArrowPatch((x, y), (x + 0.45 * math.cos(STARTING_YAW), y + 0.45 * math.sin(STARTING_YAW)),
                                 arrowstyle="-|>", mutation_scale=10, color="tab:blue", zorder=6))
    if start_dr:
        for p in START_POSE_LIST:
            ax.plot(*p, ".", color="tab:blue", ms=4, zorder=6)
    for g in goals:
        ax.add_patch(Circle(g, GOAL_THR, fc="tab:green", alpha=0.25, ec="tab:green", lw=0.6, zorder=4))
        ax.plot(*g, "*", color="darkgreen", ms=6, zorder=6)


def finish(ax, title):
    ax.set_title(title, fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlim(-3.15, 3.15)
    ax.set_ylim(-3.15, 3.15)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.tick_params(labelsize=6.5)


def seg(ax, a, b, col, lw=1.4, ls="-", label=None):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw, ls=ls, zorder=7, label=label)


fig, axes = plt.subplots(2, 4, figsize=(20, 10.6))
axes = axes.ravel()

# (a) stage9
ax = axes[0]
draw_walls(ax, S9.WALLS)
for i, o in enumerate(S9.STAGE9_OBSTACLES):   # legacy Gazebo keyframe loops (not active in the simplified env)
    k = np.asarray(o["keys"])
    ax.plot(o["spawn"][0] + k[:, 1], o["spawn"][1] + k[:, 2], ls=":", lw=0.8, color=f"C{i}", zorder=3)
draw_start_goal(ax, GOAL_LIST)
finish(ax, "(a) stage9: arena + 8 inner walls\n(dotted: legacy Gazebo keyframe loops, unused here)")

# (b) corridor
ax = axes[1]
draw_walls(ax, SCENE_MODULES["corridor"].WALLS)
for i, w in enumerate(C.CORRIDOR_WALLS):
    ax.text(w[0], w[1], f"C{i + 1}", fontsize=7, color="w", ha="center", va="center", zorder=8,
            bbox=dict(fc="0.2", ec="none", pad=0.5))
draw_start_goal(ax, GOAL_LIST)
finish(ax, "(b) corridor: arena + S-corridor C1-C6\n(start (2.5,2.5) yaw -168.7°, goal (-1.8,-2.5))")

# (c) corridor_static_object (fixed baffles)
ax = axes[2]
mod = SCENE_MODULES["corridor_static_object"]
draw_walls(ax, C.OUTER_WALLS + C.CORRIDOR_WALLS)
draw_walls(ax, mod.STATIC_ZIGZAG_WALLS, fc="tab:orange", alpha=0.9)
draw_start_goal(ax, GOAL_LIST)
finish(ax, "(c) corridor_static_object: 4 fixed zig-zag baffles (orange)")

# (d) corridor_static_object_random: nominal + 40 samples
ax = axes[3]
draw_walls(ax, C.OUTER_WALLS + C.CORRIDOR_WALLS)
for seed in range(40):
    p = SOR.sample_static_obstacles_params(seed)
    draw_walls(ax, SOR.static_obstacle_walls(p["lengths"], p["thicknesses"], p["offsets"]),
               fc="tab:orange", alpha=0.06, ec="none")
draw_walls(ax, SOR.STATIC_ZIGZAG_WALLS_RANDOMIZATION, fc="none", ec="tab:red", lw=1.0)
for i, w in enumerate(SOR.STATIC_ZIGZAG_WALLS_RANDOMIZATION):
    ax.text(w[0] + 0.15, w[1] + 0.12, f"Z{i + 1}", fontsize=7, color="tab:red", zorder=8)
draw_start_goal(ax, GOALS_STATIC, start_dr=True)
finish(ax, "(d) corridor_static_object_random (ST01-03): Z1-Z4 resampled per reset\n"
           "red = nominal; orange density = 40 samples; goals = 9-goal training list")

# (e) corridor_dynamic
ax = axes[4]
draw_walls(ax, SCENE_MODULES["corridor_dynamic"].WALLS)
for i, o in enumerate(C.CORRIDOR_DYNAMIC_OBSTACLES):
    col = ["tab:red", "tab:purple", "tab:brown"][i]
    seg(ax, o["start"], o["end"], col, label=f"OB{i + 1} {o['type']} {o['speed']} m/s")
    ax.add_patch(Circle(o["start"], M.OBSTACLE_RADIUS, fc=col, alpha=0.5, ec=col, zorder=7))
ax.axvline(C.CORRIDOR_DYNAMIC_OBSTACLES[2]["trigger_x"], color="tab:brown", ls="--", lw=0.8)
draw_start_goal(ax, GOAL_LIST)
ax.legend(fontsize=6, loc="lower right")
finish(ax, "(e) corridor_dynamic: OB1/OB2 ping-pong (phase 0 each reset),\n"
           "OB3 moves once when the robot crosses x = 0.2 (dashed)")

# (f) corridor_dynamic_chase
ax = axes[5]
draw_walls(ax, SCENE_MODULES["corridor_dynamic_chase"].WALLS)
for i, o in enumerate(SCENE_MODULES["corridor_dynamic_chase"].MOVING_OBSTACLES):
    col = ["tab:red", "tab:purple", "tab:brown"][i]
    if o["type"] == "triggered_chase":
        ax.add_patch(Circle(o["start"], M.OBSTACLE_RADIUS, fc=col, alpha=0.5, ec=col, zorder=7))
        ax.add_patch(Circle(o["start"], o["speed"] * o["chase_duration"], fc="none", ec=col, ls=":", zorder=7))
        ax.text(o["start"][0] - 0.95, o["start"][1] - 1.25, f"OB3 chases robot\n{o['speed']} m/s for "
                f"{o['chase_duration']:.0f} s\n(dotted: reach 1 m)", fontsize=6, color=col)
    else:
        seg(ax, o["start"], o["end"], col)
        ax.add_patch(Circle(o["start"], M.OBSTACLE_RADIUS, fc=col, alpha=0.5, ec=col, zorder=7))
ax.axvline(0.2, color="tab:brown", ls="--", lw=0.8)
draw_start_goal(ax, GOAL_LIST)
finish(ax, "(f) corridor_dynamic_chase: OB1/OB2 ping-pong +\nOB3 triggered chase (x < 0.2)")

# (g) corridor_dynamic_chase_random (ST04-07)
ax = axes[6]
draw_walls(ax, CDR.WALLS)
for i, o in enumerate(CDR.MOVING_OBSTACLES[:2]):
    col = ["tab:red", "tab:purple"][i]
    ylo, yhi = CDR.OBSTACLE_Y_RANGES[i]
    xs = sorted([o["start"][0], o["end"][0]])
    ax.add_patch(Rectangle((xs[0], ylo), xs[1] - xs[0], yhi - ylo, fc=col, alpha=0.15, ec=col, ls="--", zorder=3))
    L = xs[1] - xs[0]
    for f0, f1 in (o["reversal_start"], o["reversal_end"]):
        a = o["start"][0] + (o["end"][0] - o["start"][0]) * f0
        b = o["start"][0] + (o["end"][0] - o["start"][0]) * f1
        ax.plot([a, b], [yhi + 0.04] * 2, color=col, lw=3, solid_capstyle="butt", zorder=7)
    seg(ax, o["start"], o["end"], col, lw=0.8, ls=":")
    sp = CDR.OBSTACLE_SPEED_RANGES[i]
    ax.text(xs[0] - 0.05, (ylo + yhi) / 2, f"OB{i + 1}\n{sp[0]}-{sp[1]} m/s", fontsize=6, color=col, ha="right",
            va="center", zorder=8)
ob3 = CDR.MOVING_OBSTACLES[2]
ax.add_patch(Circle(ob3["start"], M.OBSTACLE_RADIUS, fc="tab:brown", alpha=0.5, ec="tab:brown", zorder=7))
ax.text(ob3["start"][0] - 0.2, ob3["start"][1] - 0.45, "OB3 chase\n0.08-0.12 m/s", fontsize=6, color="tab:brown")
ax.axvline(0.2, color="tab:brown", ls="--", lw=0.8)
draw_start_goal(ax, GOALS_CHASE, start_dr=True)
finish(ax, "(g) corridor_dynamic_chase_random (ST04-07): random_ping_pong\n"
           "y-bands = path-y DR; thick bars = reversal regions; r ∈ [0.05,0.20]")

# (h) robot / sensor schematic
ax = axes[7]
ax.add_patch(Circle((0, 0), ROBOT_R, fc="tab:blue", alpha=0.3, ec="tab:blue"))
ax.add_patch(Circle((0, 0), COLLISION_THR, fc="none", ec="red", ls="--"))
beams = np.linspace(0.0, 6.28, 40)
for b in beams:
    ax.plot([-0.04, -0.04 + 0.9 * math.cos(b)], [0, 0.9 * math.sin(b)], color="tab:orange", lw=0.6)
ax.plot(-0.04, 0, "k+", ms=6)
ax.text(-0.95, -1.05, "40 beams, linspace(0, 6.28, 40) incl. endpoint\n(beam 0 and 39 ~0.18° apart), cap 3.5 m;\n"
        "origin 0.04 m behind centre; robot r=0.17,\ncollision: GT clearance < 0.22 (dashed); goal < 0.20",
        fontsize=6.5)
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.15, 1.05); ax.set_aspect("equal"); ax.grid(True, lw=0.3)
ax.set_title("(h) robot/LiDAR model of the reference env (not to arena scale)", fontsize=8.5)
ax.tick_params(labelsize=6.5)

fig.suptitle("Reference scenes (adaptive_sensor_policy/envs/scenes), drawn from their definitions. Arena 6 x 6 m "
             "outer (walls 0.15 m); start (2.5, 2.5) facing -x-ish; goal pocket bottom-left", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/reference_scenes_topdown.png", dpi=150)

# numeric facts used in the report
print("arena interior half:", C.ARENA_INTERIOR_HALF)
nom = SOR.static_obstacles_params(None, *SOR.NOMINAL_STATIC_OBSTACLES)
print("nominal static params:", {k: v for k, v in nom.items() if k in ("c12", "c34", "d_z1_z3")})
c = [SOR.sample_static_obstacles_params(s) for s in range(2000)]
print("random static: c12 min %.3f c34 min %.3f dz13 min %.3f" % (min(p["c12"] for p in c), min(p["c34"] for p in c),
                                                                  min(p["d_z1_z3"] for p in c)))
print("n walls per scene:", {k: len(m.WALLS) for k, m in SCENE_MODULES.items()})
