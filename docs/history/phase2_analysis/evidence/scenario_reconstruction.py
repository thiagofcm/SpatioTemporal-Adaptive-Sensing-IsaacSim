"""Top-down reconstruction of the RL-Navigation training/evaluation scenarios (analysis only).

Inputs are hard-coded values whose provenance is given per item:
  [V-code]  read from RL-Navigation source (HEAD 01fc8f8 or the named commit)
  [V-usd]   read from a committed USD file with Kit/pxr (usd_bounds.py)
  [R-fig]   reconstructed from paper figures (segment_render.py / segment_fig6.py), approximate
  [U-fig]   read from a paper figure whose image registration has no anchor in the code (scale +-10 %)

Calibration of the Fig. 6b background render (img-019, 912x700 px) into world metres:
  x = -4.77 + 0.01103 px ,  y = 3.646 - 0.01099 py
  derived from (i) Fig. 6c, where three code-known x positions (moving-obstacle lanes x=1.7, x=0.0 and the
  target x=3.5) fix x_true = 0.90 x_plot + 0.13 (residual < 0.03 m), (ii) the target y and grid squareness for y,
  (iii) the 6b<->6c registration of eight common obstacles. Independent check: the derived pixel scale is
  isotropic (0.01103 vs 0.01099 m/px), and the moving obstacle 1 start (1.7, 1.6) lies exactly where its lane
  meets the bottom face of the reconstructed box O3 (y = 1.95 - 0.35 = 1.60).
  Estimated uncertainty of reconstructed static geometry: about +-0.3 m in position, +-0.1 m in size.

Usage:  python -I scenario_reconstruction.py <out_dir>
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


def px19(x0, x1, y0, y1):
    """Fig 6b render pixel box -> world box (xmin, xmax, ymin, ymax)."""
    fx = lambda p: -4.77 + 0.01103 * p
    fy = lambda p: 3.646 - 0.01099 * p
    return (round(fx(x0), 2), round(fx(x1), 2), round(fy(y1), 2), round(fy(y0), 2))


def px17(x0, x1, y0, y1):
    """Fig 6a render pixel box -> plot-axis box (unanchored registration)."""
    fx = lambda p: -4.27 + 0.01005 * p
    fy = lambda p: 5.06 - 0.01097 * p
    return (round(fx(x0), 2), round(fx(x1), 2), round(fy(y1), 2), round(fy(y0), 2))


def box_to_circle(b):
    return ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, round(((b[1] - b[0]) + (b[3] - b[2])) / 4, 2))


# ---------------- G3: Mar-2024..HEAD training arena (obstacles_dynamic.usd, missing) ----------------
G3 = {
    "name": "G3 training arena (jetbot.py / jetbot_dynamic.py, obstacles_dynamic.usd - missing)",
    "walls_inner": {"x": [-4.33, 4.85], "y": [-3.65, 3.21], "thickness": 0.4, "src": "R-fig"},
    "static_boxes": {  # [R-fig] from img-019 segmentation
        "O1_L_bar": px19(191, 231, 75, 261),
        "O1_L_foot": px19(144, 231, 205, 258),
        "O2_box_top_wall": px19(398, 489, 41, 77),
        "O3_box_top_right": px19(558, 651, 100, 154),
        "O4_box_top": px19(336, 404, 113, 161),
        "O6_bar_right_wall": px19(732, 872, 223, 265),
        "O7_bar_left": px19(108, 262, 457, 513),
        "O8_L_bar": px19(612, 636, 476, 595),
        "O8_L_foot": px19(536, 639, 563, 592),
    },
    "static_cylinders": {
        "O5_cyl_right": box_to_circle(px19(707, 747, 295, 346)),
        "O9_cyl_bottom": box_to_circle(px19(357, 409, 559, 615)),
    },
    "start": {"pos": [-2.0, 0.0], "yaw_deg": 0.0, "src": "V-code"},
    "target": {"pos": [3.5, -2.5], "goal_radius": 0.30, "src": "V-code"},
    "moving": [  # [V-code] jetbot_dynamic.py: DynamicCuboid scale (0.25,0.7,0.2), USD velocity -/+0.8 m/s along y
        {"name": "obstacle1", "x": 1.7, "y0": 1.6, "size": [0.25, 0.7, 0.2], "speed": 0.8, "first_dir": -1},
        {"name": "obstacle2", "x": 0.0, "y0": 2.2, "size": [0.25, 0.7, 0.2], "speed": 0.8, "first_dir": -1},
    ],
    "lidar_range": 2.0,
}

# ---------------- Fig 6a "novel environment" (test only; possibly obstacles4.usd) ----------------
F6A = {
    "name": "Fig. 6a test arena (unanchored scale; consistent with Oct-2023 obstacles4.usd targets)",
    "walls_inner": {"x": [-4.02, 3.82], "y": [-4.43, 4.68], "src": "U-fig"},
    "static_boxes": {
        "bar_left_vertical": px17(131, 179, 249, 389),
        "bar_right": px17(617, 753, 288, 324),
        "box_centre": px17(358, 476, 321, 360),
        "bar_lower_left": px17(162, 306, 551, 595),
        "L_bar": px17(505, 547, 671, 869),
        "L_foot": px17(402, 547, 679, 717),
        "box_bottom_right": px17(584, 703, 672, 714),
    },
    "static_cylinders": {
        "cyl_top_left": box_to_circle(px17(328, 371, 121, 164)),
        "cyl_top_right": box_to_circle(px17(497, 565, 138, 207)),
        "cyl_centre_right": box_to_circle(px17(508, 557, 439, 497)),
        "cyl_bottom": box_to_circle(px17(336, 391, 701, 759)),
    },
    "start": {"pos": [0.0, 0.0], "src": "trajectory data in Fig. 6a"},
    "test_targets_fig6a": [[-3.3, 3.7], [0.0, 3.7], [1.0, 1.0], [-3.3, -3.7], [0.0, -3.7], [3.3, -3.3]],
    "targets_oct2023_code": [[-3.1, 0.0], [3.4, 3.6], [2.0, -3.6], [0.0, -3.4], [-2.3, 3.4]],
}

# ---------------- G1: Oct-2023 arena = committed obstacles.usd (exact) ----------------
G1 = {
    "name": "G1 obstacles.usd (committed; used by Oct-2023 jet_backup.py / first jetbot.py)",
    "walls": [(1.9, 2.1, -2.0, 2.0), (-2.1, -1.9, -2.0, 2.0), (-2.0, 2.0, -2.1, -1.9), (-2.0, 2.0, 1.9, 2.1)],
    "cylinders": [(0.75, 0.75, 0.3), (0.75, -0.75, 0.3), (-0.75, 0.75, 0.3), (-0.75, -0.75, 0.3)],
    "start": [0.0, 0.0],
    "targets": [[1.5, -1.5], [0.0, -1.5]],
}
HOME = {
    "name": "jetbot_home.usd (committed, unused by any task)",
    "boxes": [(-0.541, 1.959, -0.875, -0.525), (-0.55, 1.95, 2.407, 2.757), (1.94, 2.313, -0.851, 2.732),
              (-0.875, -0.502, -0.851, 2.732)],
    "ellipses": [(0.728, 0.183, 0.242, 0.276), (0.728, 1.364, 0.242, 0.276)],
}


def lane_limits(arena, x, half_w=0.125, half_l=0.35):
    """Where a lane at x (box half-width half_w) is blocked above/below its start: bounce limits for the centre."""
    blocks = [b for b in arena["static_boxes"].values() if b[0] < x + half_w and b[1] > x - half_w]
    for cx, cy, r in arena["static_cylinders"].values():
        if abs(cx - x) < r + half_w:
            blocks.append((cx - r, cx + r, cy - r, cy + r))
    return blocks


def bounce_range(arena, mv):
    x, y0 = mv["x"], mv["y0"]
    hl = mv["size"][1] / 2
    blocks = lane_limits(arena, x)
    lo, hi = arena["walls_inner"]["y"]
    for b in blocks:
        if b[3] <= y0:
            lo = max(lo, b[3])
        elif b[2] >= y0:
            hi = min(hi, b[2])
    return round(lo + hl, 2), round(hi - hl, 2)


def tri_wave(y0, lo, hi, v, first_dir, t):
    """Idealised position of a box bouncing between lo and hi at speed v (ignores pushes by the robot)."""
    L = hi - lo
    s0 = (y0 - lo) if first_dir > 0 else (2 * L - (y0 - lo))
    s = (s0 + v * t) % (2 * L)
    return lo + np.where(s <= L, s, 2 * L - s)


def draw_arena(ax, A, approx=True, title=""):
    xi, yi = A["walls_inner"]["x"], A["walls_inner"]["y"]
    th = A["walls_inner"].get("thickness", 0.3)
    ax.add_patch(Rectangle((xi[0] - th, yi[0] - th), xi[1] - xi[0] + 2 * th, yi[1] - yi[0] + 2 * th,
                           fc="0.55", ec="none", alpha=0.6, zorder=0))
    ax.add_patch(Rectangle((xi[0], yi[0]), xi[1] - xi[0], yi[1] - yi[0], fc="white", ec="0.3", lw=1, zorder=0))
    kw = dict(fc="0.45", ec="0.15", lw=0.8, alpha=0.75, zorder=2)
    if approx:
        kw.update(ls="--", hatch="////", fc="0.7")
    for b in A["static_boxes"].values():
        ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], **kw))
    for cx, cy, r in A["static_cylinders"].values():
        ax.add_patch(Circle((cx, cy), r, **kw))
    ax.set_title(title, fontsize=9)


def finish(ax, lim):
    ax.set_aspect("equal")
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.tick_params(labelsize=7)


fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1])

# ---- (A) G3 arena: static + dynamic training, Fig 6c evaluation ----
ax = fig.add_subplot(gs[0, 0])
draw_arena(ax, G3, title="(A) G3 training arena, static & dynamic (Mar 2024 - HEAD; paper Figs. 4, 6b, 6c)\n"
                         "grey hatched = reconstructed from figures (~±0.3 m); coloured = verified in code")
ranges = {}
for mv, col in zip(G3["moving"], ["tab:red", "darkred"]):
    lo, hi = bounce_range(G3, mv)
    ranges[mv["name"]] = (lo, hi)
    w, l = mv["size"][0], mv["size"][1]
    ax.add_patch(Rectangle((mv["x"] - w / 2, lo - l / 2), w, hi - lo + l, fc=col, alpha=0.12, ec=col, ls=":", zorder=1))
    ax.add_patch(Rectangle((mv["x"] - w / 2, mv["y0"] - l / 2), w, l, fc=col, ec="k", lw=0.8, zorder=4))
    ax.add_patch(FancyArrowPatch((mv["x"] + 0.25, mv["y0"]), (mv["x"] + 0.25, mv["y0"] - 0.9), arrowstyle="-|>",
                                 mutation_scale=10, color=col, zorder=5))
    ax.text(mv["x"] + 0.32, mv["y0"] - 0.5, f"{mv['name']}\n±{mv['speed']} m/s\ny∈[{lo:.1f},{hi:.1f}]", fontsize=6.5,
            color=col, zorder=6)
s, g = G3["start"]["pos"], G3["target"]["pos"]
ax.add_patch(Circle(s, G3["lidar_range"], fc="none", ec="tab:blue", ls=":", lw=1, zorder=3))
ax.plot(*s, "o", color="tab:blue", ms=7, zorder=6)
ax.add_patch(FancyArrowPatch(s, (s[0] + 0.5, s[1]), arrowstyle="-|>", mutation_scale=10, color="tab:blue", zorder=6))
ax.text(s[0] - 0.9, s[1] - 0.45, "start (-2,0), yaw 0\n(fixed, every episode/env)", fontsize=6.5, color="tab:blue")
ax.add_patch(Circle(g, G3["target"]["goal_radius"], fc="tab:green", alpha=0.35, ec="tab:green", zorder=6))
ax.plot(*g, "*", color="darkgreen", ms=11, zorder=7)
ax.text(g[0] - 1.6, g[1] - 0.25, "target (3.5,-2.5)\ngoal r=0.30 m", fontsize=6.5, color="darkgreen")
ax.plot([s[0], g[0]], [s[1], g[1]], color="tab:green", ls="--", lw=0.8, zorder=3)
for t in [[3.65, 2.25], [3.65, -2.25], [-1.75, -2.5]]:
    ax.plot(*t, "x", color="purple", ms=7, mew=1.5, zorder=6)
ax.plot([0], [0], "s", color="purple", ms=5, zorder=6)
ax.text(0.1, 0.12, "Fig. 6b test start (0,0)", fontsize=6, color="purple")
ax.text(-4.9, -4.25, "dotted circle = LiDAR 2 m range at start · purple × / ■ = Fig. 6b Turtlebot test targets / start", fontsize=6.5)
for k, b in G3["static_boxes"].items():
    if not k.endswith("foot"):
        ax.text((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, k.split("_")[0], fontsize=6, ha="center", va="center", zorder=8)
for k, (cx, cy, r) in G3["static_cylinders"].items():
    ax.text(cx, cy, k.split("_")[0], fontsize=6, ha="center", va="center", zorder=8)
finish(ax, ((-5, 5.5), (-4.4, 3.9)))

# ---- (B) moving obstacle y(t) over one 30 s episode ----
ax = fig.add_subplot(gs[0, 1])
t = np.linspace(0, 30, 1201)
for mv, col in zip(G3["moving"], ["tab:red", "darkred"]):
    lo, hi = ranges[mv["name"]]
    ax.plot(t, tri_wave(mv["y0"], lo, hi, mv["speed"], mv["first_dir"], t), color=col,
            label=f"{mv['name']} (x={mv['x']}): period ≈ {2 * (hi - lo) / mv['speed']:.1f} s")
ax.axhline(0, color="tab:blue", lw=0.6, ls=":")
ax.axvspan(12.1, 30, color="0.9", zorder=0)
ax.text(12.4, 2.6, "straight-line time to goal at v_max 0.5 m/s ≥ 12.1 s", fontsize=6.5)
ax.set_title("(B) Idealised moving-obstacle motion within an episode (1200 steps = 30 s)\n"
             "identical in all envs; reset to the same pose each episode; direction state persists", fontsize=9)
ax.set_xlabel("time in episode (s)"); ax.set_ylabel("obstacle centre y (m)")
ax.legend(fontsize=7, loc="lower right"); ax.grid(True, lw=0.3); ax.tick_params(labelsize=7)

# ---- (C) Fig. 6a novel/test arena ----
ax = fig.add_subplot(gs[0, 2])
F6A["walls_inner"]["thickness"] = 0.3
draw_arena(ax, F6A, title="(C) Fig. 6a 'novel environment' (static test only)\n"
                          "scale unanchored (±10 %); Oct-2023 code targets overlaid")
ax.plot(0, 0, "o", color="tab:blue", ms=6)
for tg in F6A["test_targets_fig6a"]:
    ax.plot(*tg, "x", color="k", ms=7, mew=1.5)
for tg in F6A["targets_oct2023_code"]:
    ax.plot(*tg, "+", color="tab:orange", ms=9, mew=1.5)
ax.plot([], [], "x", color="k", label="Fig. 6a test targets"); ax.plot([], [], "+", color="tab:orange", label="targets in Oct-2023 jetbot.py (obstacles4.usd)")
ax.legend(fontsize=6.5, loc="upper right")
finish(ax, ((-4.6, 4.4), (-5.0, 5.3)))

# ---- (D) G1 committed obstacles.usd ----
ax = fig.add_subplot(gs[1, 0])
for b in G1["walls"]:
    ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fc="0.4", ec="k"))
for cx, cy, r in G1["cylinders"]:
    ax.add_patch(Circle((cx, cy), r, fc="0.4", ec="k"))
ax.plot(*G1["start"], "o", color="tab:blue", ms=7)
for tg in G1["targets"]:
    ax.plot(*tg, "*", color="darkgreen", ms=11)
ax.plot(-2, 0, "o", mfc="none", mec="red", ms=9); ax.plot(3.5, -2.5, "*", mfc="none", mec="red", ms=11)
ax.text(-2.0, -0.45, "G3 start (-2,0)\nwould be on the wall", fontsize=6.5, color="red", ha="center")
ax.text(3.4, -2.2, "G3 target\noutside", fontsize=6.5, color="red", ha="center")
ax.text(0.1, 0.15, "G1 start (0,0)", fontsize=6.5, color="tab:blue")
ax.text(1.55, -1.35, "(1.5,-1.5)", fontsize=6.5, color="darkgreen"); ax.text(0.05, -1.75, "(0,-1.5)", fontsize=6.5, color="darkgreen")
ax.set_title("(D) G1 = committed obstacles.usd (exact, verified with pxr)\n"
             "Oct-2023 tasks: start (0,0); targets (1.5,-1.5) / (0,-1.5); paper Fig. 2c", fontsize=9)
finish(ax, ((-3, 4.2), (-3, 2.6)))

# ---- (E) jetbot_home.usd ----
ax = fig.add_subplot(gs[1, 1])
for b in HOME["boxes"]:
    ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fc="0.4", ec="k"))
from matplotlib.patches import Ellipse
for cx, cy, rx, ry in HOME["ellipses"]:
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fc="0.4", ec="k"))
ax.set_title("(E) jetbot_home.usd (exact, verified; not referenced by any task)\n~2.4 x 3.3 m room, 2 elliptic posts", fontsize=9)
finish(ax, ((-1.5, 3.0), (-1.3, 3.2)))

# ---- (F) envSpacing 5 m vs G3 arena size ----
ax = fig.add_subplot(gs[1, 2])
xi, yi = G3["walls_inner"]["x"], G3["walls_inner"]["y"]
for i in range(-1, 2):
    for j in range(-1, 2):
        ox, oy = 5.0 * i, 5.0 * j
        ax.add_patch(Rectangle((xi[0] - 0.4 + ox, yi[0] - 0.4 + oy), xi[1] - xi[0] + 0.8, yi[1] - yi[0] + 0.8,
                               fc="none", ec="tab:red" if (i, j) == (0, 0) else "0.5", lw=1.6 if (i, j) == (0, 0) else 0.8,
                               ls="-" if (i, j) == (0, 0) else "--"))
        ax.plot(-2 + ox, oy, "o", color="tab:blue", ms=3)
ax.set_title("(F) envSpacing 5.0 m (committed) vs G3 arena\n"
             "(~9.9 x 7.6 m outer): neighbouring copies overlap", fontsize=9)
finish(ax, ((-11, 11), (-10, 10)))

fig.suptitle("RL-Navigation: original scenarios reconstructed from code, committed USDs and paper figures (arXiv 2501.02902)",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/original_scenarios_topdown.png", dpi=150)

geom = {"G3": G3, "G3_moving_bounce_ranges_y": ranges, "Fig6a": F6A, "G1": G1, "jetbot_home": HOME}
with open(f"{OUT}/scenario_geometry.json", "w") as f:
    json.dump(geom, f, indent=1, default=list)
print("bounce ranges:", ranges)
print("G3 boxes:", json.dumps(G3["static_boxes"]))
print("G3 cyl:", G3["static_cylinders"])
print("F6A boxes:", json.dumps(F6A["static_boxes"]), F6A["static_cylinders"])
